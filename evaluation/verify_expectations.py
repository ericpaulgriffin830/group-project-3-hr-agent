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

# Run as a script from anywhere: put the repo root on the path before importing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import llm
from app.agent.mcp_client import MCPClient
from app.agent.orchestrator import answer
from mcp_server.server import mcp


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


async def main(pattern: str) -> int:
    items = []
    for path in sorted(glob.glob(pattern)):
        items.extend(json.load(open(path)))

    failed = 0
    async with MCPClient(server=mcp) as client:
        await client.discover()
        model = llm.chat_model()
        for item in items:
            out = await answer(item["question"], client=client, chat_model=model,
                               employee_id=item.get("employee_id"))
            fails = check(item, out)
            status = "PASS" if not fails else "FAIL"
            failed += bool(fails)
            print(f"{status}  {item['id']:8} {item['category']:24} {item['question'][:44]}")
            for f in fails:
                print(f"         - {f}")
    print(f"\n{len(items) - failed}/{len(items)} expectations hold")
    return 1 if failed else 0


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "evaluation/eval_set.*.json"
    raise SystemExit(asyncio.run(main(pattern)))
