### ROB — Knowledge & Evidence (all of RAG)

**Summary**

* Ingestion, cleaning, heading-aware chunking, embeddings, vector store (Chroma or FAISS), citation metadata
* Top-k retrieval, optional query rewriting or reranking, and the prompting strategy that injects chunks and metadata
* RAG guardrails: out-of-corpus refusals, separating policy facts from recommendations
* The retrieval-backed MCP tools (`search_policy_documents`, `get_policy_section`), exposed as functions Person B wraps
* The retrieval ablation (k values, chunk sizes) and the RAG metrics: groundedness and citation accuracy

**Code**

* app/rag/ingest.py — parse markdown, HTML, PDF *(brief wants 2+ formats; we do 3)*
* app/rag/chunk.py — heading-aware, ~600 tokens / 80 overlap, **fixed seed**
* app/rag/embed.py — **fastembed, BAAI/bge-small-en-v1.5** (the swap)
* app/rag/store.py — Chroma persistent. Metadata: `doc_id`, `title`, `section`, `source_path`, `snippet` — citation accuracy is capped by this
* app/rag/retrieve.py — top-k, optional doc filter, hybrid BM25 + vector, RRF fusion
* app/rag/answer.py — `synthesize()`: prompting strategy injecting chunks + source metadata, cited answer generation, **evidence guardrails** (unsupported-claim limiting, policy-fact vs. recommendation) — **first cut by Thursday against fixtures**
* scripts/build_index.py — one command, deterministic, runs in CI and at deploy
* Backs `search_policy_documents` and `get_policy_section`

**Fixes:** `sentence-transformers` → `fastembed`, re-lock, verify retrieval quality · `quantic-grader` collaborator · README `wget` typo

**Tests:** `test_retrieval.py` — known query returns known doc in top-k, deterministic

**Docs:** RAG section of `design-and-evaluation.md` — justify chunking strategy, retrieval k, embedding model, vector store, and hybrid+RRF

**Demo segment:** corpus, retrieval, grounding & citations, evaluation results

---

### ERIC — UI Surface & Pipeline

**Summary**

* MCP server (`mcp/`) with at least five tools, including the mock-data tools (`lookup_employee_profile`, `check_pto_balance`, `lookup_benefits_status`, `create_mock_hr_ticket`, `draft_hr_email`, `check_policy_compliance`)
* Agent orchestrator and MCP client: tool discovery, intent handling, multi-step workflows, graceful failure handling, confirmation gates for irreversible actions
* The operational trace (tools, arguments, outputs, sources, escalation decision)
* Agent metrics: tool selection accuracy, workflow completion, escalation/clarification accuracy, action-safety pass rate
* Design and demo choreography for the two agentic tasks

**Code**

* app/api.py — FastAPI `/chat` and `/health`. Health returns `{status, mcp_connected, tools_discovered, index_ready}`. **Stub responses day 1** so the UI isn't blocked
* ui/streamlit_app.py — chat, citation cards with snippets, collapsible **trace panel** rendering Contract B's `trace[]`, confirmation dialog on `requires_confirmation`, health badge in the header
* **Two "Run demo task" buttons** in the sidebar — brief requires the grader reproduce both tasks from the UI
* `render.yaml` — two services, `API_BASE_URL` wired
* Render deployment, both services. **Auto-deploy OFF** so the CI gate is real
* .github/workflows/ci.yml — add `ubuntu-latest`, install → lint → import check → pytest → deploy job with `needs: test`
* `.env.example` — full variable set (names come out of tonight's contract freeze)

**Corpus (All): (by Monday 9/21 - blocker for Rob)**

* travel-and-expense · onboarding · code-of-conduct · performance-and-promotion
* pto-and-leave · holidays · benefits-overview · hr-case-escalation *(these carry demo Task B — need real detail on approval chains and blackout periods)*
* remote-work · multi-state-work-and-tax · data-security · equipment-and-byod *(these carry demo Task A)*

**Evaluation**

* evaluation/ablation.py — k ∈ {3,5,8} · chunk 400 vs 800 · hybrid vs vector-only · tools-on vs tools-off
* Latency harness: p50/p95 over 20 queries, **cold-start and warm-start reported separately**
* Gold answers for his 4 docs
* Results tables + charts into `evaluation/results/`
* Gold answers + expected `doc_id` citations for his 4 docs
* Review of all 26 items for consistency
* The **prompt-variant ablation**
* The complex multi-document question (rubric item 3, bullet 5)
* evaluation/run_eval.py — the harness, all six metric families
* Agent-behavior metrics: tool-selection accuracy, workflow completion rate, escalation/clarification accuracy, action-safety pass rate
* The groundedness and citation-accuracy **scorers**

**Tests:** `test_health.py`

**Docs:** `README.md` (extend Rob's) · `deployed.md` — both URLs, health URL, cold-start notes

**Demo segment:** UI, deployment, CI/CD + agentic Task B

---

### CHRIS — Agent & MCP spine

**Summary**

**Code**

* docs/CONTRACTS.md — the 8 tool schemas (A), `/chat` envelope (B), `synthesize()` signature (C) — **tonight**
* app/llm.py — shared Groq client: temp 0, fixed seed, retry/backoff, rotation across three keys — **Mon EOD, Rob is blocked on it**
* mcp/server.py — FastMCP, all 8 tools, dual transport (stdio / streamable-HTTP) — **Mon EOD with hardcoded fixtures; Rob and Eric both blocked on it**
* mcp/schemas.py — pydantic in/out models
* app/agent/mcp_client.py — transport adapter, `list_tools()` discovery on startup, real `ClientSession` throughout
* app/agent/orchestrator.py — LangGraph state graph: intent → RAG-only vs. tools → tool loop (max 6 steps) → hand evidence to Rob's `synthesize()`
* app/agent/trace.py — operational trace per Contract B. Tool, args, result summary, latency. **No chain-of-thought** — brief forbids it
* app/agent/guardrails.py —  **action layer** : confirm-token gate on both write tools, refusal-to-act, escalation
* Failure handling: unknown `employee_id`, MCP tool unavailable, thin evidence, ambiguous → clarify
* `mock_data/` — `employees.json`, `pto.json`, `benefits.json`, `tickets.json`. 8 synthetic employees, 3 locations, 2 managers, full-time/part-time/contractor

**Tests:** `test_mcp_discovery.py` · `test_guardrails.py`

**Docs**

* Architecture diagram (all 7 components: web app, orchestrator, MCP client, MCP server, RAG index, mock data, LLM provider)
* `design-and-evaluation.md` sections: agent orchestration, MCP design, transport choice, tool schemas, safety guardrails
* The two demo-task descriptions + **expected MCP call sequence for each** — explicit rubric item 10 bullet
* Final edit of `design-and-evaluation.md` into one voice

**Lead duties:** run the six calls · contract change control · demo script · run the recording session · track the submission checklist

**Demo segment:** architecture + agentic Task A

---

### Everyone

* Your paragraph in `ai-tooling.md` — what tool, what worked, what didn't
* Your own **mirror repo** + `quantic-grader` added to it
* Free **Groq** and **Render** accounts; Groq key into the shared rotation
* 3-line standup in Teams by 9am daily: done / next / blocked
* **Sat 9/26** async checkpoint — 2-minute screen recording, no call
* Sign the Group Project Agreement (at kickoff)
* **Government ID ready for the recording Thu 10/1** — brief requires all three on camera showing ID
