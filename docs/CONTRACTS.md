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

**Owner:** Chris (`mcp/server.py`, `mcp/schemas.py`)
**Backed by:** Rob for the two RAG tools; Chris for the rest.

### 1. `search_policy_documents` — RAG
```json
in : {"query": str, "k": int = 5, "doc_filter": [str] | null}
out: {"chunks": [{"doc_id": str, "title": str, "section": str,
                  "snippet": str, "score": float}],
      "retrieval_mode": "hybrid" | "vector_only" | "keyword_only"}
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
     "args": {"employee_id": "E-1043"},
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

**Authored by:** Eric (he writes the corpus).
**Consumed by:** Rob (stores `doc_id` as chunk metadata) and Chris (the agent cites it).

Every citation the system produces is a `doc_id`. If Eric's filenames and these IDs
drift apart, citations point at documents that don't exist and citation accuracy —
a scored metric — goes to zero. So the IDs are a contract, not a filename convention.

**Rule: `doc_id` == the corpus filename without its extension.** Lowercase, hyphenated.

| `doc_id` | Covers | Format |
|---|---|---|
| `pto-and-leave` | accrual, requests, approval, blackout periods | md |
| `holidays` | company holidays, floating days | md |
| `remote-work` | eligibility, out-of-state work, approval chain | md |
| `multi-state-work-and-tax` | nexus thresholds, notification duties | md |
| `travel-and-expense` | reimbursable categories, limits, receipts | md |
| `data-security` | device standards, VPN, remote access | md |
| `benefits-overview` | eligibility, waiting periods, elections | md |
| `onboarding` | first-week checklist, provisioning | html |
| `equipment-and-byod` | issued equipment, personal devices | html |
| `code-of-conduct` | workplace conduct, reporting | pdf |
| `performance-and-promotion` | review cycle, promotion criteria | pdf |
| `hr-case-escalation` | what escalates, to whom, confidentiality | md |

Three formats on purpose — the brief asks for at least two handled.

**Required cross-references.** The brief requires at least one question needing
retrieval from multiple documents. That only works if the corpus is written for it:

- `remote-work` §3.2 must reference BOTH `data-security` and `multi-state-work-and-tax`
- `pto-and-leave` must reference `holidays` for blackout periods
- `benefits-overview` must distinguish full-time / part-time / contractor

**Section IDs.** `get_policy_section(doc_id, section_id)` needs stable section IDs.
Use numbered headings (`## 3.2 Out-of-State Work`); `section_id` is the number only.
