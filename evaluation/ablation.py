"""The retrieval ablation: k in {3,5,8} x {hybrid, vector_only}.

`Group_Assignment.md` asks for at least one ablation. This is that, and it is
deliberately the cheapest honest one available: it calls `app.rag.retrieve`
directly and never touches the LLM.

Two reasons for keeping the model out of it.

**It isolates the variable.** Run the ablation through the full agent and each
cell also absorbs the model's choice of search phrasing, how many times it
searched, and which chunks synthesis chose to cite. A drop at k=3 would then be
unattributable -- retrieval, or the agent reacting to retrieval. Here the only
thing that changes between cells is `k` and `mode`.

**It is deterministic and free.** Every cell is reproducible to the chunk,
re-runnable on a laptop with no API key, and costs nothing against the shared
200k-token daily budget. `run_eval.py` spends that budget on the questions that
genuinely need a model.

Ground truth is `expected.must_cite_doc_ids` from the committed eval sets -- the
documents a correct answer has to rest on. 15 of the 27 items declare it.

    uv run python evaluation/ablation.py [--out=PATH]

**What this measures is retrieval recall, not citation accuracy.** A document
retrieved is a document the agent *could* cite. Whether it did is
`run_eval.py`'s `citations` check, and the two numbers are not interchangeable.

One caveat reported rather than hidden: the two `agentic_multi_document` items
are scored here on the raw question, but in production the agent reformulates
and searches repeatedly, so this understates what the full system retrieves for
them. They are broken out separately for that reason.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.rag import retrieve as retrieve_module
from app.rag.retrieve import retrieve

K_VALUES = (3, 5, 8)
MODES = ("hybrid", "vector_only")

#: Scored apart from the rest: the ablation feeds these the raw question, while
#: the real agent reformulates and searches more than once.
AGENTIC_CATEGORY = "agentic_multi_document"


def _items(pattern: str) -> list[dict]:
    items: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        items.extend(json.load(open(path)))
    scored = [i for i in items
              if i.get("expected", {}).get("must_cite_doc_ids")]
    if not scored:
        raise SystemExit(f"No items with must_cite_doc_ids matched {pattern!r}")
    return scored


def _cell(items: list[dict], k: int, mode: str) -> dict:
    """One (k, mode) cell, over every scored item."""
    per_item = []
    for item in items:
        started = time.monotonic()
        out = retrieve(item["question"], k=k, mode=mode)
        latency_ms = (time.monotonic() - started) * 1000

        got = {c["doc_id"] for c in out["chunks"] if "doc_id" in c}
        required = set(item["expected"]["must_cite_doc_ids"])
        per_item.append({
            "id": item["id"],
            "category": item["category"],
            "required": sorted(required),
            "found": sorted(required & got),
            "missed": sorted(required - got),
            "complete": not (required - got),
            "docs_returned": len(got),
            "chunks_returned": len(out["chunks"]),
            "retrieval_mode": out["retrieval_mode"],
            "latency_ms": round(latency_ms, 1),
        })

    def _summary(rows: list[dict]) -> dict:
        if not rows:
            return {}
        required_total = sum(len(r["required"]) for r in rows)
        found_total = sum(len(r["found"]) for r in rows)
        return {
            # Items where EVERY required document was retrieved. The strict
            # number, and the one the brief's "citation accuracy" rests on:
            # a multi-document answer that finds one of two documents is not
            # 50% right, it is a confidently incomplete answer.
            "items_complete": sum(1 for r in rows if r["complete"]),
            "items": len(rows),
            # Partial credit, kept alongside because it shows whether a cell
            # fails by missing one document or by missing everything.
            "docs_found": found_total,
            "docs_required": required_total,
            "mean_docs_returned": round(
                statistics.mean(r["docs_returned"] for r in rows), 2),
            "median_latency_ms": round(
                statistics.median(r["latency_ms"] for r in rows), 1),
        }

    main_rows = [r for r in per_item if r["category"] != AGENTIC_CATEGORY]
    agentic_rows = [r for r in per_item if r["category"] == AGENTIC_CATEGORY]

    return {
        "k": k,
        "mode": mode,
        "retrieval_only": _summary(main_rows),
        "agentic_raw_question": _summary(agentic_rows),
        "all": _summary(per_item),
        "per_item": per_item,
    }


@contextmanager
def _vector_leg_unavailable():
    """Make the vector leg return nothing, the way an unbuilt index does.

    This is the third cell of the ablation, and it exists because it happened by
    accident and turned out to be the most interesting result in the set.

    `data/chroma` is gitignored -- it is a build artifact of the corpus, rebuilt
    by `scripts.build_index` on every deploy. The first run of this harness was
    made before that script had ever been run locally, so the collection held 0
    chunks. Hybrid scored 12/13 anyway, on BM25 alone. Vector-only scored 0/13.

    That is the actual argument for hybrid retrieval on a corpus this small: not
    ranking quality, where the two are indistinguishable, but what survives when
    half the machinery is missing.

    Reproducible without the patch by deleting `data/chroma` and re-running --
    the patch just means the comparison can live in one report next to the others.
    """
    original = retrieve_module._vector_ranked
    retrieve_module._vector_ranked = lambda query, doc_filter, pool: []
    try:
        yield
    finally:
        retrieve_module._vector_ranked = original


def run(pattern: str, include_degraded: bool = True) -> dict:
    items = _items(pattern)

    # Check the index BEFORE warming. Warming embeds a query, which on a cold
    # machine downloads the fastembed model -- there is no reason to pay that to
    # find out the run cannot produce a valid result anyway.
    from app.rag import store
    indexed = store.count()
    if indexed == 0:
        raise SystemExit(
            "The Chroma collection is empty, so the vector leg cannot contribute "
            "and every 'hybrid' cell would silently be BM25-only.\n"
            "Run `uv run python -m scripts.build_index` first.\n"
            "(retrieve() still reports retrieval_mode='hybrid' in that state, "
            "which is how this was nearly published as a real result.)"
        )

    # Warm the process caches (BM25 index, embeddings, Chroma handle) before
    # timing anything, or the first cell carries the whole cold-start cost and
    # k=3/hybrid looks like the slowest configuration when it is the fastest.
    retrieve("warm the caches", k=3)

    cells = [_cell(items, k, mode) for mode in MODES for k in K_VALUES]

    degraded: list[dict] = []
    if include_degraded:
        with _vector_leg_unavailable():
            for mode in MODES:
                for k in K_VALUES:
                    cell = _cell(items, k, mode)
                    cell["vector_leg"] = "unavailable"
                    degraded.append(cell)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pattern": pattern,
        "scored_items": len(items),
        "chunks_indexed": indexed,
        "k_values": list(K_VALUES),
        "modes": list(MODES),
        "cells": cells,
        "degraded_cells": degraded,
    }


def _table(report: dict, key: str, title: str) -> list[str]:
    rows = [r for r in report["cells"] if r[key]]
    if not rows:
        return []
    lines = [
        "",
        title,
        f"  {'mode':12} {'k':>2}  {'complete':>10}  {'docs found':>11}  "
        f"{'docs ret.':>9}  {'median ms':>9}",
    ]
    for cell in rows:
        s = cell[key]
        pct = s["items_complete"] / s["items"] if s["items"] else 0
        lines.append(
            f"  {cell['mode']:12} {cell['k']:>2}  "
            f"{s['items_complete']:>3}/{s['items']:<3} {pct:>3.0%}  "
            f"{s['docs_found']:>5}/{s['docs_required']:<5}  "
            f"{s['mean_docs_returned']:>9}  {s['median_latency_ms']:>9}"
        )
    return lines


def summarize(report: dict) -> str:
    lines = [
        f"Retrieval ablation over {report['scored_items']} item(s) with "
        f"must_cite_doc_ids ({report['pattern']})",
        f"{report['chunks_indexed']} chunk(s) in the vector index.",
        "No LLM involved -- retrieval only, so every cell is reproducible.",
    ]
    lines += _table(report, "retrieval_only",
                    "Retrieval-only items (policy_qa + multi_document):")
    lines += _table(report, "agentic_raw_question",
                    "Agentic items, scored on the RAW question (understates the "
                    "real system, which reformulates):")

    if report.get("degraded_cells"):
        lines += _table({"cells": report["degraded_cells"]}, "retrieval_only",
                        "SAME items, vector leg UNAVAILABLE (unbuilt index):")

    # What each configuration missed, so a number can be traced to a document.
    lines.append("")
    lines.append("Misses by cell:")
    for cell in report["cells"]:
        missed = [f"{r['id']}({'/'.join(r['missed'])})"
                  for r in cell["per_item"] if r["missed"]]
        label = f"{cell['mode']} k={cell['k']}"
        lines.append(f"  {label:22} {', '.join(missed) if missed else '-- none'}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="evaluation/eval_set.*.json")
    parser.add_argument("--out", default="evaluation/ablation_results.json",
                        help="Where to write the full JSON report "
                             "(pass '' to skip writing).")
    parser.add_argument("--no-degraded", action="store_true",
                        help="Skip the vector-leg-unavailable cells.")
    args = parser.parse_args()

    report = run(args.pattern, include_degraded=not args.no_degraded)
    print(summarize(report))

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
