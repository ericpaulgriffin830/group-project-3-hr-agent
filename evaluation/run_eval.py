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

Latency here is agent latency (question in, answer out, in-process), not
Render's cold/warm-start latency -- docs/DEPLOYMENT-NOTES.md's cold-start
number is about hitting the deployed HTTP service after 15 minutes idle, a
different measurement taken against a different thing. Don't merge the two
numbers into one report; the brief asks for both, separately, on purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import pathlib
import statistics
import sys
import time
from datetime import datetime, timezone

# Run as a script from anywhere: put the repo root on the path before importing,
# exactly as verify_expectations.py does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import llm
from app.agent.mcp_client import MCPClient
from app.agent.orchestrator import answer
from mcp_server.server import mcp

DEFAULT_CONCURRENCY = 6  # see verify_expectations.py's docstring for why 6

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


async def run(pattern: str, concurrency: int) -> dict:
    items = []
    for path in sorted(glob.glob(pattern)):
        items.extend(json.load(open(path)))
    if not items:
        raise SystemExit(f"No eval items matched {pattern!r}")

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
        "pattern": pattern,
        "item_count": len(items),
        "elapsed_s": round(elapsed_s, 1),
        "errors": [{"id": i, "error": f"{type(e).__name__}: {e}"} for i, e in errors],
        "still_throttled_after_retry": [r["id"] for r in still_degraded],
        "results": clean,
    }


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
    lines = [
        f"Ran {report['item_count']} item(s) in {report['elapsed_s']}s "
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
        lines.append(
            f"Latency: mean {statistics.mean(latencies):.0f}ms, "
            f"median {statistics.median(latencies):.0f}ms, "
            f"max {max(latencies):.0f}ms "
            "(agent latency, in-process -- NOT Render cold/warm-start latency, "
            "see docs/DEPLOYMENT-NOTES.md for that separate measurement)"
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
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--out", default="evaluation/results.json",
                        help="Where to write the full JSON report "
                             "(pass '' to skip writing).")
    args = parser.parse_args()

    report = asyncio.run(run(args.pattern, args.concurrency))
    print(summarize(report))

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {out_path}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
