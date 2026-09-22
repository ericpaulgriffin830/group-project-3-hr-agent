# Interface Contracts

**Status: DRAFT — frozen at kickoff 2026-09-20.**
Once frozen, changes require a PR and a note in Teams. Not a side conversation.

These three contracts are what let us build in parallel. Rob builds behind the tool
functions, Chris wraps and orchestrates, Eric builds against the response envelope.
Nobody waits.

---

## Contract A — MCP tool schemas

Eight tools. The brief requires ≥5, with ≥1 using the RAG index and ≥1 using mock
structured data or performing a mock operation.

**Owner:** Chris (`mcp_server/server.py`, `mcp_server/schemas.py`)

> **Why `mcp_server/` and not `mcp/`.** A local package named `mcp/` shadows the
> installed MCP SDK — `import mcp` then finds our directory and every
> `from mcp.server...` fails. The directory name is load-bearing, not cosmetic.
>
> `mcp_server/schemas.py` is the **executable copy of this contract**, and
> `tests/test_schemas.py` validates what the real tools return against it, so the
> prose below and the server cannot drift apart silently. Note the SDK class is
> `MCPServer`; `FastMCP` was its name before SDK 2.x.
**Backed by:** Rob for the two RAG tools; Chris for the rest.

### 1. `search_policy_documents` — RAG
```json
in : {"query": str, "k": int = 5, "doc_filter": [str] | null}
out: {"chunks": [{"doc_id": str, "title": str, "section": str,
                  "snippet": str, "score": float}],
      "retrieval_mode": "hybrid" | "vector_only" | "keyword_only" | "fixture"}
```

### 2. `get_policy_section` — RAG
```json
in : {"doc_id": str, "section_id": str}
out: {"doc_id": str, "title": str, "section": str, "text": str}
```

### 3. `lookup_employee_profile` — mock data
```json
in : {"employee_id": str}
out: {"employee_id": str, "name": str, "role": str, "employment_type": str,
      "location": str, "manager_id": str, "manager_name": str, "hire_date": str,
      "tenure_months": int}
err: {"error": "employee_not_found", "employee_id": str}
```

### 4. `check_pto_balance` — mock data
```json
in : {"employee_id": str}
out: {"employee_id": str, "accrued_days": float, "used_days": float,
      "available_days": float, "blackout_dates": [str], "accrual_rate": float}
```

### 5. `lookup_benefits_status` — mock data
```json
in : {"employee_id": str}
out: {"employee_id": str, "elections": [{"plan": str, "tier": str, "status": str}],
      "eligible": bool, "waiting_period_days": int, "eligibility_date": str}
```

### 6. `check_policy_compliance` — composite
```json
in : {"scenario": str, "employee_id": str, "policy_refs": [str] | null}
out: {"verdict": "compliant" | "non_compliant" | "conditional" | "insufficient_evidence",
      "conditions": [str], "policy_refs": [{"doc_id": str, "section": str}],
      "rationale": str}
```

### 7. `create_mock_hr_ticket` — MOCK WRITE, CONFIRMATION-GATED
```json
in : {"employee_id": str, "category": str, "summary": str,
      "confirm_token": str | null}
out: {"ticket_id": str, "status": "created", "created_at": str}
gate: confirm_token omitted or null ->
      {"requires_confirmation": true, "preview": {...}, "confirm_token": str}
```

### 8. `draft_hr_email` — MOCK WRITE, CONFIRMATION-GATED
```json
in : {"employee_id": str, "recipient_role": str, "intent": str,
      "context": str, "confirm_token": str | null}
out: {"draft": {"to": str, "subject": str, "body": str}, "sent": false}
gate: same as #7. NOTHING IS EVER SENT.
```

**`retrieval_mode: "fixture"`** is what the fixture-phase server returns today. It is
in the enum deliberately rather than dressed up as `keyword_only` — a caller reading a
trace should be able to tell canned data from a real index at a glance. It disappears
when Rob's retrieval lands.

**Error convention — every tool.** Never raise to the agent. Return
`{"error": "<machine_code>", "message": "<human text>"}` so the orchestrator can
degrade gracefully and the trace records what happened.

---

## Contract B — the `/chat` response envelope

**Owner:** Eric (`app/api.py`) — but the shape is shared.
Chris populates `trace`, `basis`, `escalate`. Rob populates `answer`, `citations`.

```json
{
  "answer": "string",
  "citations": [
    {"doc_id": "string", "title": "string", "section": "string", "snippet": "string"}
  ],
  "trace": [
    {"step": 1, "tool": "lookup_employee_profile",
     "args": {"employee_id": "E1001"},
     "result_summary": "Full-time, Pittsburgh PA, manager E-1002",
     "latency_ms": 84, "status": "ok"}
  ],
  "requires_confirmation": false,
  "pending_action": null,
  "basis": "policy_rag | tool_data | both | refused",
  "escalate": false
}
```

`trace` is an OPERATIONAL record — tool, args, result summary, latency, status.
It is NOT chain-of-thought. The brief explicitly forbids exposing hidden reasoning.

`/health` returns:
```json
{"status": "ok", "mcp_connected": true, "tools_discovered": 8, "index_ready": true}
```

---

## Contract C — the RAG / agent seam

**Owner:** Rob (`app/rag/answer.py`). Called by Chris's orchestrator.

```python
def synthesize(
    question: str,
    chunks: list[Chunk],
    tool_results: dict | None = None,
    mode: Literal["policy", "workflow", "refusal"] = "policy",
) -> dict:
    """
    Returns:
      {"answer": str,
       "citations": [{"doc_id","title","section","snippet"}],
       "basis": "policy_rag" | "tool_data" | "both" | "refused",
       "unsupported_flags": [str],
       "confidence": float}
    """
```

The orchestrator decides WHETHER to call this and WHAT evidence to pass.
Rob decides HOW the answer is written and cited.

**Guardrail split:**
- **Rob — evidence guardrails.** Don't assert what the corpus doesn't support.
  Populates `unsupported_flags`. Distinguishes policy fact from recommendation.
- **Chris — action guardrails.** Don't DO what the user didn't confirm.
  Owns `confirm_token`, refusal-to-act, escalation.

---

## Environment variables

So Eric can write `.env.example` without guessing.

| Variable | Example | Owner |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | each member, rotated |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chris |
| `MCP_TRANSPORT` | `stdio` \| `streamable-http` | Chris |
| `MCP_SERVER_URL` | `http://localhost:8765/mcp` | Chris |
| `API_BASE_URL` | `https://hr-agent-api.onrender.com` | Eric |
| `CHROMA_PATH` | `./data/chroma` | Rob |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Rob |
| `RETRIEVAL_K` | `5` | Rob |
| `LOG_LEVEL` | `INFO` | Eric |

Secrets come from the environment only. Never committed.

**`GROQ_MODEL` changed 2026-09-20 — forced, not preference.** `llama-3.3-70b-versatile`
is no longer served by Groq; it 404s with `model_not_found`. Verified against the live
list with `uv run python -m app.llm --models`. `openai/gpt-oss-120b` is the largest model
the free tier reaches that also tool-calls correctly (`gpt-oss-20b` and `qwen/qwen3.8-27b`
both work too, and are faster). Re-check with `--models` before assuming any id still
works — the lineup moves.

**One consequence for the trace.** gpt-oss models return a `reasoning` field alongside
`content`, carrying literal chain-of-thought. The brief forbids exposing it. `app/llm.py`
returns **only** `content`, so nothing downstream can leak it — do not reach around the
client to read the raw response.

---

## Change control

Frozen after kickoff. To change a contract: open a PR touching this file, post in
Teams, get one other member's approval. The cost of a silent change is that two
people build against different assumptions for a day.

**Never hand-merge `uv.lock`.** Take either side, re-run `uv lock`, commit the result.

---

## Contract D — the document ID registry

**Authored by:** Eric (he writes the corpus). **Consumed by:** Rob (chunk metadata)
and Chris (the agent cites it).

**Superseded 2026-09-21.** This section originally specified twelve lowercase
hyphenated ids (`remote-work`, `pto-and-leave`, …) derived from filenames. Eric's
corpus landed in `6e56740` with a different scheme, and the real corpus wins over a
placeholder: his section ids are written into the document bodies
(`## CONDUCT-6 Conflicts of Interest`), so changing them means rewriting twelve
documents, while amending this table is an edit. The ids below are now
authoritative and come from `corpus/manifest.json`.

Every citation the system produces is a `doc_id`. If these and the corpus drift
apart, citations point at documents that don't exist and citation accuracy — a
scored metric — goes to zero. That is why this is a contract and not a convention.

| `doc_id` | Covers | Format | Pages |
|---|---|---|---|
| `HANDBOOK-OVERVIEW` | Overview | markdown | 3.7 |
| `PTO-HOLIDAYS` | PTO and Holidays Policy | markdown | 3.3 |
| `REMOTE-WORK` | Work Arrangement Categories | html | 3.7 |
| `TAX-LOCATION` | Tax and Work Location Policy | markdown | 3.2 |
| `EXPENSE` | Expense and Reimbursement Policy | markdown | 2.5 |
| `EQUIPMENT` | Standard Issue Equipment | html | 2.0 |
| `INFOSEC` | Information Security Policy | markdown | 2.7 |
| `BENEFITS` | Benefits Guide | markdown | 2.3 |
| `LEAVE` | Types of Leave | txt | 2.5 |
| `ONBOARDING` | Onboarding Checklist and Process | markdown | 2.3 |
| `CONDUCT` | Workplace Conduct and Anti-Harassment Policy | markdown | 2.3 |
| `HR-OPS` | HR Service Model | html | 2.0 |

**Total: 32.5 pages, 16,258 words, across markdown / HTML / TXT.**
The brief asks for 30–120 pages and at least two formats; both are met.

**Section IDs** are per-document prefixes with an ordinal — `RW-2`, `CONDUCT-6`,
`HANDBOOK-1` — and they appear in the headings themselves. `get_policy_section`
fetches by that id, and `corpus/manifest.json` lists every one. Matching on the id
rather than the heading text means a reworded heading does not break retrieval.

**All three gaps closed 2026-09-22.** `REMOTE-WORK` now cross-references
`TAX-LOCATION`; `mock_data/blackout_periods.json` carries scoped blackout windows;
and E1008 sits at 1.5 days so Task B exercises its refusal path. `blackout_dates:
list[str]` in Contract A is replaced by structured `blackout_periods` — scope
decides whether a window binds an employee at all, and the notice period and
approval chain are what they actually have to act on.

**Original gaps, for the record — raised with Eric 2026-09-21** — these are corpus edits, not code:

1. `REMOTE-WORK` cross-references `INFOSEC` and `EQUIPMENT` but **not
   `TAX-LOCATION`**. Demo Task A and rubric item 3's multi-document question were
   specced on remote work → security **and** tax. One leg is missing.
2. **No employee record carries blackout dates.** `PTO-HOLIDAYS` describes blackout
   periods and Contract A declares the field, but `mock_data/pto_balances.json`
   has none — so the agent can cite the rule and never apply it to anyone.
3. **No PTO balance is below 3.0 days**, so "can I take three days?" is yes for
   everyone and demo Task B never exercises its refusal path.

Both 2 and 3 are pinned as `xfail` tests in `tests/test_mock_data.py`, so they flip
to passing the moment the data lands.
