"""Run every item and report whether its `expected` block actually holds.

Not the scoring harness -- that is run_eval.py (Eric). This exists so an item is
never committed with an expectation nobody checked. An eval set written from
intention measures the author's optimism, and every item in it passes by
construction on the day it is written.

    uv run python evaluation/verify_expectations.py [glob]
"""

from __future__ import annotations

import asyncio
import glob
import json
import pathlib
import sys
import time

# Run as a script from anywhere: put the repo root on the path before importing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import llm
from app.agent.mcp_client import MCPClient
from app.agent.orchestrator import answer
from mcp_server.server import mcp


def _throttled(out: dict) -> bool:
    """Did this turn lose a model call to rate limiting?

    The orchestrator records it as a guardrail step with status error rather than
    raising, so it is visible in the trace and invisible in the answer.
    """
    return any(
        s.get("type") == "guardrail" and s.get("status") == "error"
        and "ratelimit" in str(s.get("result_summary", "")).lower()
        for s in out.get("trace", [])
    )


def check(item: dict, out: dict) -> list[str]:
    exp = item.get("expected", {})
    trace = out.get("trace", [])
    called = [s["tool"] for s in trace if s.get("type") == "tool_call" and "tool" in s]
    cited = {c["doc_id"] for c in out.get("citations", []) if "doc_id" in c}
    intent = next((s.get("result_summary") for s in trace
                   if s.get("type") == "intent"), None)
    fails = []

    if "intent" in exp and intent != exp["intent"]:
        fails.append(f"intent {intent!r} != {exp['intent']!r}")
    for tool in exp.get("must_call", []):
        if tool not in called:
            fails.append(f"missing call {tool}")
    for tool in exp.get("must_not_call", []):
        if tool in called:
            fails.append(f"forbidden call {tool}")
    missing_docs = set(exp.get("must_cite_doc_ids", [])) - cited
    if missing_docs:
        fails.append(f"missing citations {sorted(missing_docs)}")
    if "requires_confirmation" in exp:
        got = bool(out.get("requires_confirmation"))
        if got != exp["requires_confirmation"]:
            fails.append(f"requires_confirmation {got} != {exp['requires_confirmation']}")
    if "escalation_route" in exp:
        route = (out.get("escalation") or {}).get("route")
        if route != exp["escalation_route"]:
            fails.append(f"escalation {route!r} != {exp['escalation_route']!r}")
    for text in exp.get("answer_must_mention", []):
        if text.lower() not in (out.get("answer") or "").lower():
            fails.append(f"answer missing {text!r}")
    return fails


#: How many items run at once. Each item is several LLM round trips and they are
#: independent, so the run is almost entirely waiting -- serial execution spent
#: minutes doing nothing.
#:
#: 4 is measured, not guessed. At 4 the set runs in ~390s with 14/14 holding. At
#: 10 it runs in ~250s and drops to 12/14 -- and the two failures are not agent
#: misbehaviour, they are Groq 429s. Measured directly: 3 of 10 concurrent runs of
#: one item hit OpenAIRateLimitError. Three keys does not help, because the limit
#: that binds here is tokens-per-minute across all of them.
#:
#: The failures are quiet, which is the real hazard. The orchestrator catches a
#: model failure and falls through to synthesis with whatever evidence it has --
#: correct behaviour for a live user, but it means a throttled evaluation run
#: reports plausible wrong answers rather than errors. Hence the warning below.
DEFAULT_CONCURRENCY = 4


async def _run_one(item: dict, model, semaphore: asyncio.Semaphore) -> tuple[dict, list[str]]:
    """One item, on its own MCP session.

    A session per item rather than one shared across tasks: MCP's ClientSession
    runs an anyio task group, and interleaving calls from several tasks through
    one session is not something it promises to survive. In-process sessions are
    cheap, so the safe thing is also the easy thing.
    """
    async with semaphore:
        async with MCPClient(server=mcp) as client:
            await client.discover()
            out = await answer(item["question"], client=client, chat_model=model,
                               employee_id=item.get("employee_id"))
    return item, check(item, out), _throttled(out)


async def main(pattern: str, concurrency: int = DEFAULT_CONCURRENCY) -> int:
    items = []
    for path in sorted(glob.glob(pattern)):
        items.extend(json.load(open(path)))

    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)
    model = llm.chat_model()

    results = await asyncio.gather(
        *(_run_one(item, model, semaphore) for item in items),
        return_exceptions=True,
    )

    # Report in file order, not completion order -- a result set that reshuffles
    # every run is hard to diff against the last one.
    failed = 0
    degraded = 0
    for item, outcome in zip(items, results):
        if isinstance(outcome, BaseException):
            failed += 1
            print(f"ERROR {item['id']:8} {type(outcome).__name__}: {outcome}")
            continue
        _, fails, throttled = outcome
        failed += bool(fails)
        if throttled:
            degraded += 1
        print(f"{'PASS' if not fails else 'FAIL'}  {item['id']:8} "
              f"{item['category']:24} {item['question'][:44]}")
        for f in fails:
            print(f"         - {f}")

    elapsed = time.monotonic() - started
    print(f"\n{len(items) - failed}/{len(items)} expectations hold "
          f"({elapsed:.0f}s, concurrency {concurrency})")

    if degraded:
        print(f"\nWARNING: {degraded} item(s) lost a model call to rate limiting. "
              f"Those results are not trustworthy -- the agent degrades gracefully "
              f"on a throttled call, so a failure here may be the rate limit rather "
              f"than the agent. Re-run at lower concurrency before believing it.")
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pattern = args[0] if args else "evaluation/eval_set.*.json"
    conc = next((int(a.split("=")[1]) for a in sys.argv[1:]
                 if a.startswith("--concurrency=")), DEFAULT_CONCURRENCY)
    raise SystemExit(asyncio.run(main(pattern, conc)))
