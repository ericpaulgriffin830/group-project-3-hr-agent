"""The scoring harness. Not `verify_expectations.py` (Chris) -- that answers
"did every item's `expected` block hold, right now" as a pre-commit gate, one
bit per item. This answers the brief's actual evaluation questions: broken
down by category and by check type, what is our tool-selection accuracy,
citation accuracy, escalation/clarification accuracy, and action-safety rate
-- the numbers `design-and-evaluation.md` reports, not a pass/fail list.

    uv run python evaluation/run_eval.py [glob] [--concurrency=N] [--out=PATH]

Concurrency, the per-item MCP session, and the throttled-retry pass are
lifted directly from `verify_expectations.py` -- Chris already measured the
right knobs (6 concurrent, retry throttled items serially at concurrency 1)
against Groq's actual rate limits, and a second harness re-discovering that
by trial and error would just waste the shared token budget for no reason.

Two modes, and the brief wants numbers from both:

**In-process (default).** Imports the orchestrator and runs it directly. This is
agent latency -- question in, answer out -- with no network, no Render instance
and no cold start in it. It is the right number for "how long does the agent
take", and the wrong number for "how long does the deployed app take".

**HTTP (`--api-base-url`).** POSTs each item to the deployed `/chat`. This is
what a grader experiences, so it is where the brief's cold-start and warm-start
latencies come from.

`--api-base-url` defaults to **concurrency 1**, deliberately. The free instance
is 0.1 CPU and 512 MB; six concurrent requests queue behind each other and the
p95 then measures our own contention rather than the service. A contended run is
still fine for scoring accuracy -- it is just not a latency measurement, and the
report says so rather than letting the reader assume.

Cold start is taken as a **single timed `GET /health` before any item runs**, and
excluded from the item latencies, so the warm p50/p95 is not polluted by the one
request that paid to wake the instance. That also makes the run's first item warm
instead of quietly absorbing a 60-90 second wake-up.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import pathlib
import statistics
import sys
import time
from datetime import datetime, timezone

# Run as a script from anywhere: put the repo root on the path before importing,
# exactly as verify_expectations.py does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import httpx

from app import llm
from app.agent.mcp_client import MCPClient
from app.agent.orchestrator import answer
from mcp_server.server import mcp

DEFAULT_CONCURRENCY = 6  # see verify_expectations.py's docstring for why 6

#: HTTP mode runs serially unless told otherwise -- see the module docstring.
DEFAULT_HTTP_CONCURRENCY = 1

#: A free Render instance takes ~60-90s to wake. Anything at or above this on the
#: pre-run health probe is a cold start rather than a slow network.
COLD_START_THRESHOLD_S = 15.0

#: One turn can run six tool steps, each its own model call, on 0.1 CPU.
HTTP_TIMEOUT_S = 240.0

#: One bool (pass/fail) per applicable check, per item -- None where the item
#: makes no claim of that kind, so it is correctly excluded from that check's
#: denominator rather than counted as a failure or a free pass.
CHECK_NAMES = (
    "intent", "tool_selection", "citations",
    "confirmation_gate", "escalation", "answer_content",
)


def _throttled(out: dict) -> bool:
    """Did this turn lose a model call to rate limiting? Same signal
    verify_expectations.py uses -- a throttled run degrades gracefully
    (the orchestrator falls through to synthesis with whatever evidence it
    has) rather than erroring, so a scored failure downstream of a throttle
    is the rate limit talking, not the agent."""
    return any(
        s.get("type") == "guardrail" and s.get("status") == "error"
        and "ratelimit" in str(s.get("result_summary", "")).lower()
        for s in out.get("trace", [])
    )


def score(item: dict, out: dict) -> dict[str, bool | None]:
    """Per-check pass/fail, None where the item declares no expectation of
    that kind. Deliberately granular -- verify_expectations.py's `check()`
    collapses everything to one fail list, which is right for "does this
    item still hold" but wrong for "what is our citation accuracy", since a
    wrong `answer_must_mention` would otherwise drag down the citation number
    for an item that cited perfectly."""
    exp = item.get("expected", {})
    trace = out.get("trace", [])
    called = {s["tool"] for s in trace if s.get("type") == "tool_call" and "tool" in s}
    cited = {c["doc_id"] for c in out.get("citations", []) if "doc_id" in c}
    intent = next((s.get("result_summary") for s in trace
                   if s.get("type") == "intent"), None)

    result: dict[str, bool | None] = {name: None for name in CHECK_NAMES}

    if "intent" in exp:
        result["intent"] = intent == exp["intent"]

    if "must_call" in exp or "must_not_call" in exp:
        missing = [t for t in exp.get("must_call", []) if t not in called]
        forbidden = [t for t in exp.get("must_not_call", []) if t in called]
        result["tool_selection"] = not missing and not forbidden

    if "must_cite_doc_ids" in exp:
        result["citations"] = not (set(exp["must_cite_doc_ids"]) - cited)

    if "requires_confirmation" in exp:
        result["confirmation_gate"] = (
            bool(out.get("requires_confirmation")) == exp["requires_confirmation"]
        )

    if "escalation_route" in exp:
        route = (out.get("escalation") or {}).get("route")
        result["escalation"] = route == exp["escalation_route"]

    if "answer_must_mention" in exp:
        answer_text = (out.get("answer") or "").lower()
        result["answer_content"] = all(
            text.lower() in answer_text for text in exp["answer_must_mention"]
        )

    return result


async def _run_one(item: dict, model, semaphore: asyncio.Semaphore) -> dict:
    """One item, on its own MCP session -- see verify_expectations.py's
    docstring on why a session isn't shared across concurrent tasks."""
    async with semaphore:
        started = time.monotonic()
        async with MCPClient(server=mcp) as client:
            await client.discover()
            out = await answer(item["question"], client=client, chat_model=model,
                               employee_id=item.get("employee_id"))
        latency_ms = int((time.monotonic() - started) * 1000)
    return {
        "id": item["id"], "category": item["category"],
        "checks": score(item, out), "throttled": _throttled(out),
        "latency_ms": latency_ms,
    }


async def _run_one_http(item: dict, client: httpx.AsyncClient,
                        semaphore: asyncio.Semaphore, base_url: str) -> dict:
    """One item against the DEPLOYED /chat, scored by the same `score()`.

    Contract B's envelope is the same object the in-process path returns, which is
    the whole reason one scorer serves both: if these two ever needed different
    scoring code, the API would not be returning the contract.
    """
    async with semaphore:
        started = time.monotonic()
        resp = await client.post(
            f"{base_url}/chat",
            json={"question": item["question"],
                  "employee_id": item.get("employee_id"),
                  "confirm_token": item.get("confirm_token")},
        )
        latency_ms = int((time.monotonic() - started) * 1000)
    resp.raise_for_status()
    out = resp.json()
    return {
        "id": item["id"], "category": item["category"],
        "checks": score(item, out), "throttled": _throttled(out),
        "latency_ms": latency_ms,
    }


async def _warm_up(client: httpx.AsyncClient, base_url: str) -> dict:
    """One throwaway /chat after the health probe, timed separately.

    Waking the instance and serving the first question are TWO costs, and measuring
    only the first understates what a grader waits. Measured 2026-09-29 from
    overnight idle: `GET /health` came back in 71.5s, and then the first /chat took
    a further 68s while every later one ran in ~4-15s.

    `/health` reports `index_ready` from the Chroma collection's count, which does
    not force the fastembed ONNX model into memory -- the first retrieval does. So
    the instance can answer `/health` truthfully while still being one big lazy
    load away from answering a question.

    Without this, that 68s landed on whichever item happened to run first and went
    into the warm p95 as agent latency. `docs/DEMO-SCRIPT.md` and `deployed.md`
    already tell a human to send one throwaway question before starting; this is
    the harness doing the same thing, and reporting what it cost.
    """
    started = time.monotonic()
    try:
        resp = await client.post(f"{base_url}/chat",
                                 json={"question": "What is the PTO policy?"})
        elapsed = time.monotonic() - started
        return {"ok": resp.status_code == 200, "elapsed_s": round(elapsed, 1),
                "status": resp.status_code}
    except httpx.RequestError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 1)}


async def _probe_cold_start(client: httpx.AsyncClient, base_url: str) -> dict:
    """Time one GET /health before anything else runs.

    Kept out of the item latencies on purpose. A free instance spun down after 15
    minutes idle takes ~60-90s to wake, and letting that land on whichever item
    happened to go first would put a 90-second outlier in the warm p95 and call it
    agent latency.
    """
    started = time.monotonic()
    try:
        resp = await client.get(f"{base_url}/health")
        elapsed = time.monotonic() - started
        body = resp.json() if resp.status_code == 200 else {"status": resp.status_code}
    except httpx.RequestError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 1)}
    return {
        "ok": resp.status_code == 200,
        "elapsed_s": round(elapsed, 1),
        "was_cold": elapsed >= COLD_START_THRESHOLD_S,
        "health": body,
    }


async def run(pattern: str, concurrency: int,
              api_base_url: str | None = None) -> dict:
    items = []
    for path in sorted(glob.glob(pattern)):
        items.extend(json.load(open(path)))
    if not items:
        raise SystemExit(f"No eval items matched {pattern!r}")

    if api_base_url:
        return await _run_http(items, pattern, concurrency,
                               api_base_url.rstrip("/"))

    semaphore = asyncio.Semaphore(concurrency)
    model = llm.chat_model()

    started = time.monotonic()
    results = await asyncio.gather(
        *(_run_one(item, model, semaphore) for item in items),
        return_exceptions=True,
    )

    # Retry whatever got throttled, serially -- identical rationale to
    # verify_expectations.py: a wide pass absorbs most of the set, and only
    # the casualties pay for a slow, solo second attempt.
    retry_idx = [i for i, r in enumerate(results)
                 if not isinstance(r, BaseException) and r["throttled"]]
    if retry_idx:
        print(f"({len(retry_idx)} item(s) throttled -- retrying serially)\n")
        solo = asyncio.Semaphore(1)
        for i in retry_idx:
            results[i] = await _run_one(items[i], model, solo)

    elapsed_s = time.monotonic() - started
    errors = [(items[i]["id"], r) for i, r in enumerate(results) if isinstance(r, BaseException)]
    clean = [r for r in results if not isinstance(r, BaseException)]
    still_degraded = [r for r in clean if r["throttled"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "in_process",
        "concurrency": concurrency,
        # Above concurrency 1 these latencies include waiting on Groq's rate
        # limiter: the SDK retries a 429 internally, so the wait is inside the
        # measured span and indistinguishable from the agent being slow.
        "latency_is_uncontended": concurrency == 1,
        "pattern": pattern,
        "item_count": len(items),
        "elapsed_s": round(elapsed_s, 1),
        "errors": [{"id": i, "error": f"{type(e).__name__}: {e}"} for i, e in errors],
        "still_throttled_after_retry": [r["id"] for r in still_degraded],
        "results": clean,
    }


async def _run_http(items: list[dict], pattern: str, concurrency: int,
                    base_url: str) -> dict:
    """The deployed-app pass. Same scoring, different transport."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        cold = await _probe_cold_start(client, base_url)
        if not cold["ok"]:
            raise SystemExit(
                f"{base_url}/health did not answer: {cold.get('error') or cold}\n"
                "Nothing was scored -- a run against an unreachable service would "
                "report 0% accuracy and look like an agent failure."
            )

        warmup = await _warm_up(client, base_url)

        semaphore = asyncio.Semaphore(concurrency)
        started = time.monotonic()
        results = await asyncio.gather(
            *(_run_one_http(item, client, semaphore, base_url) for item in items),
            return_exceptions=True,
        )

        retry_idx = [i for i, r in enumerate(results)
                     if not isinstance(r, BaseException) and r["throttled"]]
        if retry_idx:
            print(f"({len(retry_idx)} item(s) throttled -- retrying serially)\n")
            solo = asyncio.Semaphore(1)
            for i in retry_idx:
                try:
                    results[i] = await _run_one_http(items[i], client, solo, base_url)
                except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                    results[i] = exc
        elapsed_s = time.monotonic() - started

    errors = [(items[i]["id"], r) for i, r in enumerate(results)
              if isinstance(r, BaseException)]
    clean = [r for r in results if not isinstance(r, BaseException)]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "http",
        "api_base_url": base_url,
        "concurrency": concurrency,
        # True only at concurrency 1. Above that the latencies include our own
        # queueing on a 0.1-CPU instance and are not a service measurement.
        "latency_is_uncontended": concurrency == 1,
        "cold_start": cold,
        "first_request": warmup,
        "pattern": pattern,
        "item_count": len(items),
        "elapsed_s": round(elapsed_s, 1),
        "errors": [{"id": i, "error": f"{type(e).__name__}: {e}"} for i, e in errors],
        "still_throttled_after_retry": [r["id"] for r in clean if r["throttled"]],
        "results": clean,
    }


def _percentile(values: list[int], pct: float) -> float:
    """Nearest-rank percentile. `statistics.quantiles` interpolates and needs at
    least two points; 27 items and a p95 want the plain definition."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[idx])


def _accuracy(results: list[dict], check: str, category: str | None = None) -> tuple[int, int]:
    """(passes, applicable) for one check, optionally scoped to one category."""
    applicable = [
        r for r in results
        if r["checks"].get(check) is not None
        and (category is None or r["category"] == category)
    ]
    passes = sum(1 for r in applicable if r["checks"][check])
    return passes, len(applicable)


def summarize(report: dict) -> str:
    results = report["results"]
    categories = sorted({r["category"] for r in results})
    where = (f"against {report['api_base_url']} over HTTP"
             if report.get("mode") == "http" else "in-process")
    lines = [
        f"Ran {report['item_count']} item(s) in {report['elapsed_s']}s "
        f"{where}, concurrency {report.get('concurrency', '?')} "
        f"({report['pattern']})",
        "",
        "Per-check accuracy (overall):",
    ]
    for check in CHECK_NAMES:
        p, n = _accuracy(results, check)
        if n:
            lines.append(f"  {check:20} {p}/{n} ({p/n:.0%})")

    lines.append("")
    lines.append("Per-category:")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        # An item "passes" if every check it declares an expectation for held.
        passed = sum(
            1 for r in cat_results
            if all(v is not False for v in r["checks"].values())
        )
        lines.append(f"  {cat:24} {passed}/{len(cat_results)} item(s) fully correct")

    latencies = [r["latency_ms"] for r in results]
    if latencies:
        lines.append("")
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        if report.get("mode") == "http":
            cold = report.get("cold_start", {})
            first = report.get("first_request", {})
            lines.append("Latency, deployed app (end to end over HTTP):")
            lines.append(
                f"  instance wake   {cold.get('elapsed_s', '?')}s   "
                f"(GET /health; "
                f"{'was asleep' if cold.get('was_cold') else 'already awake'})"
            )
            lines.append(
                f"  first question  {first.get('elapsed_s', '?')}s   "
                "(throwaway /chat -- loads the embedding model, which /health "
                "does not)"
            )
            lines.append(
                f"  warm  p50 {p50:.0f}ms   p95 {p95:.0f}ms   "
                f"max {max(latencies):.0f}ms   (n={len(latencies)}, "
                "both costs above excluded)"
            )
            if not report.get("latency_is_uncontended"):
                lines.append(
                    f"  WARNING: concurrency was {report.get('concurrency')}, so these "
                    "include our own queueing on a 0.1-CPU instance. Accuracy is "
                    "unaffected; the latency numbers are not a service measurement. "
                    "Re-run with --concurrency=1 for the reportable figure."
                )
        else:
            lines.append(
                f"Agent latency, in-process: p50 {p50:.0f}ms, p95 {p95:.0f}ms, "
                f"mean {statistics.mean(latencies):.0f}ms, max {max(latencies):.0f}ms"
            )
            lines.append(
                "  No network, no Render instance, no cold start in these numbers. "
                "For the deployed figures run with --api-base-url."
            )
            if not report.get("latency_is_uncontended"):
                lines.append(
                    f"  WARNING: concurrency was {report.get('concurrency')}. The Groq "
                    "SDK retries a 429 inside the request, so rate-limit waiting is "
                    "inside these spans -- the tail measures the shared token budget, "
                    "not the agent. DO NOT REPORT THESE AS LATENCY. Re-run with "
                    "--concurrency=1, or take the figure from a --api-base-url run."
                )

    if report["errors"]:
        lines.append("")
        lines.append(f"{len(report['errors'])} item(s) errored:")
        for e in report["errors"]:
            lines.append(f"  {e['id']}: {e['error']}")

    if report["still_throttled_after_retry"]:
        lines.append("")
        lines.append(
            f"WARNING: {len(report['still_throttled_after_retry'])} item(s) still "
            f"throttled after a serial retry -- their scores are not trustworthy. "
            f"Re-run with --concurrency=1 or wait for the rate limit to clear: "
            f"{report['still_throttled_after_retry']}"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="evaluation/eval_set.*.json")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"Default {DEFAULT_CONCURRENCY} in-process, "
                             f"{DEFAULT_HTTP_CONCURRENCY} with --api-base-url "
                             "(see the module docstring).")
    parser.add_argument("--api-base-url", default=None,
                        help="Run against a deployed API instead of in-process, "
                             "e.g. https://hr-agent-api-s2ux.onrender.com. This is "
                             "where the brief's cold/warm-start latencies come from.")
    parser.add_argument("--out", default=None,
                        help="Where to write the full JSON report. Defaults to "
                             "evaluation/results.json in-process and "
                             "evaluation/results.deployed.json over HTTP, so a "
                             "deployed run cannot quietly overwrite the other. "
                             "Pass '' to skip writing.")
    args = parser.parse_args()

    concurrency = args.concurrency
    if concurrency is None:
        concurrency = (DEFAULT_HTTP_CONCURRENCY if args.api_base_url
                       else DEFAULT_CONCURRENCY)
    out = args.out
    if out is None:
        out = ("evaluation/results.deployed.json" if args.api_base_url
               else "evaluation/results.json")

    report = asyncio.run(run(args.pattern, concurrency, args.api_base_url))
    print(summarize(report))

    if out:
        out_path = pathlib.Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {out_path}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
