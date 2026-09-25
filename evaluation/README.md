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
