# The two agentic demo tasks

**Rubric item 10, third bullet:** *"Describe the two required agentic demo tasks and
the expected sequence of MCP tool calls for each."* This is that description.

**Owner:** Chris. **Demo segment:** Chris presents Task A, Eric presents Task B.

Both tasks below were run against the live system on 2026-09-22 — real Groq, a real
MCP session, Eric's real corpus. The sequences are observed, not designed.

---

## A note on "expected sequence"

The agent chooses its own tools, so the sequence is not a fixed script. Across runs
of the same question the call count varied between 3 and 6 before the retrieval
fixes, and 5 to 7 after.

So each task below states two things:

- **Required calls** — the calls that MUST happen for the answer to be grounded. If
  one is missing, the answer is either ungrounded or about the wrong person.
- **Observed sequence** — one real trace, to narrate against.

Temperature is 0 and the seed is fixed, but Groq's seed is best-effort rather than
a guarantee, so identical input can still produce a different tool order. **On demo
day, narrate the trace panel on screen rather than a memorised script.** The trace
is rendered live for exactly this reason.

---

## Task A — Out-of-state remote work

> **"Can I work from Colorado for six weeks? I am E1007."**

**Why this one.** It is the multi-document question rubric item 3 requires. Answering
it properly needs the remote work policy (is this allowed, for how long, who
approves), the tax and work location policy (does six weeks in another state create
an obligation), and information security (what are the conditions for working
outside the office). No single document answers it.

**Who is asking.** E1007, Sofia Reyes — Consultant, Consulting Delivery, hybrid,
home state Illinois, manager Alicia Moreno. An ordinary individual contributor, not
an executive, so the approval chain is real.

### Required calls

| # | Tool | Why it must happen |
|---|---|---|
| 1 | `lookup_employee_profile` | The answer depends on her home state. Colorado is a different state *for her*; for a Denver-based colleague the same question is routine. Without this the agent is answering a generic question. |
| 2 | `search_policy_documents` | The governing policy. Must run **without** a `doc_filter` first — filtering to REMOTE-WORK immediately is what made this task single-document for three days. |
| 3 | `get_policy_section` | Full text of the section the snippet points at, so the citation quotes the rule rather than a fragment. |

`check_policy_compliance` is available and the agent may call it; it is not required,
because the same determination is reachable from the retrieved sections.

### Observed sequence — 2026-09-22

```
1. intent      →  workflow
2. tool_call   lookup_employee_profile  {"employee_id": "E1007"}
3. tool_call   search_policy_documents  {"query": "working from another state duration approval…", "k": 10}
4. tool_call   search_policy_documents  {"query": "temporary remote work duration approval manager…", "k": 10}
5. tool_call   search_policy_documents  {"doc_filter": ["REMOTE-WORK"], "query": "temporary remote…"}
6. tool_call   get_policy_section       {"doc_id": "REMOTE-WORK", "section_id": "RW-3"}
7. tool_call   search_policy_documents  {"doc_filter": ["REMOTE-WORK"], "query": "Remote Work L…"}
8. synthesis   →  cites REMOTE-WORK, TAX-LOCATION
```

Note steps 3–4: it searches broadly **before** narrowing at step 5. That ordering is
the fix that made this task multi-document, and it is worth pointing out on camera —
it is the difference between an agent that retrieves and an agent that confirms what
it already assumed.

### What to point at during the demo

- **Step 2** — the agent identifies *who is asking* before answering anything.
- **Steps 3–4** — unfiltered search first. Breadth before depth.
- **Step 6** — a specific section fetched by id, so the citation is exact.
- **The citations** — two documents, from different policy areas. This is the
  multi-document requirement being satisfied visibly.

---

## Task B — PTO request against an insufficient balance and a blackout

> **"Can I take three days of PTO in mid-October? I am E1008."**

**Why this one.** The answer is **no**, for two independent reasons, and both are
checkable against the record:

1. **Balance.** E1008 has **1.5 days** available. Three days is not available.
2. **Blackout.** She is in Consulting Delivery, which has a department blackout
   `BO-2026-CONSULTING-CLIENT-GOLIVE` running **2026-10-05 to 2026-10-16** — exactly
   mid-October — citing `PTO-9`, requiring manager *and* department head approval
   with 10 business days' notice.

A task where the answer is always yes demonstrates nothing. This one exercises the
refusal path against real data, and the two reasons are independent, so the agent
citing only one is a visible partial answer rather than a wrong one.

**Who is asking.** E1008, Grace Lin — Client Support Specialist, Consulting Delivery,
full-time non-exempt, Denver CO, manager Alicia Moreno.

**The scoping matters.** A Data Engineering colleague asking the same question sees
only the company-wide year-end blackout, not this one. `check_pto_balance` filters
blackout periods by the employee's department, so the agent cannot warn the wrong
person about the wrong window.

### Required calls

| # | Tool | Why it must happen |
|---|---|---|
| 1 | `lookup_employee_profile` | Establishes department, which decides which blackouts apply. |
| 2 | `check_pto_balance` | The 1.5-day balance and the scoped blackout windows both come from here. |
| 3 | `search_policy_documents` | The governing PTO rules — notice period, approval chain, what happens when the balance is short. |

### Observed sequence — 2026-09-22

```
1. intent      →  workflow
2. tool_call   lookup_employee_profile  {"employee_id": "E1008"}
3. tool_call   check_pto_balance        {"employee_id": "E1008"}
4. tool_call   search_policy_documents  {"query": "PTO request limit available days exceed balance", "k": 10}
5. tool_call   search_policy_documents  {"query": "insufficient PTO balance request unpaid leave", "k": 10}
6. tool_call   search_policy_documents  {"query": "PTO insufficient balance request unpaid leave", "k": 10}
7. tool_call   search_policy_documents  {"query": "cannot request more PTO than available balance", "k": 10}
8. synthesis   →  cites PTO-HOLIDAYS, LEAVE
```

Steps 5–7 are the agent rephrasing the same question. Honest to show: it is looking
for a rule about requesting more than you have, and reformulating when the first
phrasing does not surface it. It is also the clearest argument for Rob's real
retrieval — semantic search would find that in one call.

### What to point at during the demo

- **Step 3** — the balance and the blackout arrive together, already scoped to her
  department.
- **The answer** — declines, and says *why* on both grounds with citations.
- **Nothing was written.** This task is read-only; the write path is Task B's
  optional extension below.

### Optional extension — the confirmation gate

If time allows, follow up with **"Then file a ticket asking for an exception."**

```
tool_call   create_mock_hr_ticket  {"employee_id": "E1008", …}   → requires_confirmation
guardrail   confirmation gate: create_mock_hr_ticket
```

The agent stops and asks. **No ticket exists at this point.** Confirming re-sends the
same request with the token and the ticket is created.

Three things worth saying while that is on screen:

- The token is an **HMAC over the arguments**, so a token issued for this preview
  cannot be replayed against a different ticket. The human approves the action they
  actually read.
- **The model is never offered `confirm_token`** as a tool argument — it is stripped
  from the schema the model sees, so it cannot invent one.
- The token is **redacted from the trace**, which is rendered in a browser.

This covers the brief's *"Prevent irreversible actions"* requirement and the
action-safety metric in one 30-second beat.

---

## Reproducing both from the UI

The brief requires the grader reproduce both tasks from the UI. Eric's sidebar
carries two **Run demo task** buttons, which POST to `/chat` with `demo_task: "A"`
or `"B"` and the matching `employee_id` — `E1007` and `E1008` respectively.

## Known gaps at time of writing

- **`synthesize()` is not wired.** Both tasks return `basis="unsynthesized"` with
  evidence gathered and citations selected, because Rob's `app/rag/answer.py` does
  not exist yet. The fallback deliberately does not write a plausible answer — a
  convincing stub is indistinguishable from a real one in a demo. The prose appears
  the moment his module lands; nothing else changes.
- **Retrieval is fixture-phase** — IDF keyword scoring with a per-document cap, not
  Rob's hybrid BM25 + vector with RRF. Citations use his real `doc_id`s and
  `section_id`s, so they stay valid when the index replaces the ranking underneath.
