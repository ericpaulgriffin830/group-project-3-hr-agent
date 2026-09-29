# Evaluation set — item format

**This is Contract E.** The harness (`run_eval.py`, Eric) and the item files (all
three of us) build against it. A format agreed after the items are written is a
format two people implemented differently — that is how the `doc_id` drift
happened, and it cost a rework.

The brief wants **20–30 items** covering straightforward policy questions,
multi-document questions, tool-requiring tasks, ambiguous requests, and
out-of-scope requests, each with a correct or gold answer.

## Split

| Author | Items | Covers |
|---|---|---|
| Chris | `eval_set.chris.json` | agentic, tool-requiring, ambiguous, out-of-scope, action-safety, escalation |
| Eric | `eval_set.eric.json` | policy Q&A gold answers for the corpus he wrote |
| Rob | `eval_set.rob.json` | retrieval-quality items, expected `doc_id` citations |

Separate files on purpose: three people editing one JSON array is a merge conflict
every time. The harness globs `eval_set.*.json`.

## The three harnesses

| Script | Question it answers | LLM? |
|---|---|---|
| `verify_expectations.py` | "Does every item's `expected` block still hold, right now?" One bit per item — a pre-commit gate. | yes |
| `run_eval.py` | "What are our accuracy numbers, by check and by category, and how fast is it?" The numbers section 9 reports. | yes |
| `ablation.py` | "Does retrieval configuration matter — k ∈ {3,5,8}, hybrid vs vector-only?" | **no** |

`run_eval.py` runs two ways, and the brief wants both:

```bash
# Agent latency and accuracy, in-process. No network, no Render, no cold start.
uv run python evaluation/run_eval.py

# What a grader actually experiences. Cold start, first-request cost and warm
# p50/p95, measured separately. Serial by default -- see below.
uv run python evaluation/run_eval.py --api-base-url https://hr-agent-api-s2ux.onrender.com
```

Three things about the HTTP mode that are easy to get wrong:

**It runs at concurrency 1 by default.** The free instance is 0.1 CPU and 512 MB.
Six concurrent requests queue behind each other and the p95 then measures our own
contention, not the service. Raising `--concurrency` is fine for scoring accuracy
faster; the report flags the latency numbers as contended when you do.

**Waking the instance and serving the first question are two different costs.**
`GET /health` reports `index_ready` from the Chroma collection's count, which does
not force the embedding model into memory — the first retrieval does. Measured
2026-09-29 from overnight idle: health came back in 71.5s, and the first `/chat`
took a further 68s while later ones ran in 4–15s. The harness times both, and
excludes both from the warm percentiles.

**It writes to `results.deployed.json`**, not `results.json`, so a deployed run
can't silently overwrite the in-process one.

`ablation.py` deliberately never calls the model: one variable changes per cell, and
every cell is reproducible on a laptop with no API key and no cost against the
shared 200k-token daily budget. It **refuses to run against an empty Chroma
collection** — `retrieve()` reports `retrieval_mode: "hybrid"` even when the vector
leg returned nothing, which is how a BM25-only run was nearly published as a
hybrid-vs-vector result.

> `run_eval.py` is Eric's. The HTTP mode and latency split were added rather than
> forked into a fourth script, for the reason already in its docstring: the
> concurrency and throttled-retry knobs were measured against Groq's real limits,
> and a second harness rediscovering them by trial and error would waste the
> shared budget for nothing.

## One item

```json
{
  "id": "AG-01",
  "category": "agentic_multi_document",
  "question": "Can I work from Colorado for six weeks?",
  "employee_id": "E1007",
  "expected": {
    "intent": "workflow",
    "must_call": ["lookup_employee_profile"],
    "must_not_call": ["create_mock_hr_ticket"],
    "must_cite_doc_ids": ["REMOTE-WORK", "TAX-LOCATION"],
    "requires_confirmation": false,
    "escalation_route": null,
    "answer_must_mention": ["approval"]
  },
  "gold_answer": "Permitted with conditions: ...",
  "notes": "Why this item exists and what it would catch."
}
```

### Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable. Results reference it; never renumber. |
| `category` | yes | One of the categories below. Drives per-category scoring. |
| `question` | yes | Exactly what goes to `/chat`. |
| `employee_id` | no | The persona asking. Omit for questions that are the same for everyone. |
| `expected.intent` | no | `policy_qa` / `workflow` / `clarify` / `refuse`. |
| `expected.must_call` | no | Tools that MUST appear in the trace. Feeds tool-selection accuracy. |
| `expected.must_not_call` | no | Tools that must NOT. Feeds action-safety. |
| `expected.must_cite_doc_ids` | no | `doc_id`s the citations must include. Feeds citation accuracy. |
| `expected.requires_confirmation` | no | Whether the turn must stop for a human. |
| `expected.escalation_route` | no | Expected route, or `null` for none. Feeds escalation accuracy. |
| `expected.answer_must_mention` | no | Substrings the answer must contain — a fact, not a phrasing. |
| `gold_answer` | yes | Short correct answer, for partial-match scoring and for a human reading results. |
| `notes` | yes | Why the item exists. An item nobody can justify gets tuned until it passes. |

### Categories

`policy_qa` · `multi_document` · `agentic_multi_document` · `tool_required` ·
`ambiguous` · `out_of_scope` · `action_safety` · `escalation`

## Two rules

**Expectations are observed, not hoped for.** Every item in
`eval_set.chris.json` was run against the live system before being committed, and
the `expected` block records what it actually does. An eval set written from
intention measures the author's optimism.

**An item that fails is a finding, not a bug in the item.** If an expectation stops
holding, fix the system or record why the expectation was wrong. Editing the
expectation to match current behaviour turns the whole set into a tautology.
