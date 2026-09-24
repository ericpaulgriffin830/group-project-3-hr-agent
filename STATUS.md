# Project status — 2026-09-24

Living file. Update it rather than starting a new one, so there is one place to look.

**Recording: Thu 10/1 — 7 days. Feature freeze: Mon 9/28 — 4 days. Submit: Fri 10/2.**

---

## Where we are against the rubric

| # | Requirement | State | Owner |
|---|---|---|---|
| 1 | Environment & reproducibility | **Good** — uv, pinned, `.env.example` complete. README still has no deploy or eval instructions | Eric |
| 2 | Corpus ingestion & indexing | **Done** — 12 docs, 32.5pp, ingest/chunk/embed/store | Rob |
| 3 | RAG | **Half** — hybrid BM25+vector+RRF retrieval works; **`synthesize()` does not exist** | Rob |
| 4 | Agentic system | **Done** — LangGraph orchestrator, trace, action guardrails | Chris |
| 5 | MCP server & tools | **Done** — 8 tools, dual transport, real retrieval wired in | Chris |
| 6 | Web application | **Not started** — no `/chat`, no `/health`, no UI | Eric |
| 7 | Deployment | **Not started** — no `render.yaml`, no URL | Eric |
| 8 | CI/CD | **Half** — runs on push/PR, but no Linux runner, no pytest, no deploy gate | Eric |
| 9 | Evaluation | **Half** — 14 of ~26 items written; no harness | all / Eric |
| 10 | Design documentation | **Half** — contracts, demo tasks, deployment notes exist; `design-and-evaluation.md` does not | all |

**273 tests pass.** CI is green on `main`.

---

## Landed this week

**Eric** — the full corpus (12 documents, 4 formats, 16,258 words, a manifest with
per-document ids and a section index) plus six mock datasets. Then closed all three
gaps raised against it within a day: the `REMOTE-WORK` → `TAX-LOCATION`
cross-reference, scoped blackout periods, and a PTO balance low enough that demo
Task B has a real refusal to make.

**Rob** — `app/rag/`: ingest, chunk, embed, store, retrieve. Hybrid BM25 + vector
with RRF fusion, a `vector_only` mode for the ablation, and a per-document cap.
Plus the `fastembed` swap.

**Chris** — the agent and MCP spine: contracts, 8-tool MCP server, shared LLM
client, MCP client, LangGraph orchestrator, trace, action guardrails, 14 eval
items, demo-task descriptions.

### Two things worth knowing

**The deployment blocker is gone.** `sentence-transformers` → `fastembed` took the
Linux dependency tree from **3.46 GB to 115 MB**, with zero torch and zero CUDA.
Render free instances are 512 MB RAM; the old tree could not have run there.

**Rob's retrieval is wired into the MCP tools and the swap changed nothing
downstream.** `search_policy_documents` reports `retrieval_mode: "hybrid"` instead
of `"fixture"`; `get_policy_section` returns 2,057 characters of real section text
where the fixture had a 320-character snippet. Both demo tasks still work and Task A
still cites `REMOTE-WORK` + `TAX-LOCATION`.

That was painless because the fixtures always used Eric's real `doc_id`s and
`section_id`s. Citations stayed valid when the ranking underneath was replaced. It
is the one thing that would have been genuinely expensive to get wrong.

---

## The critical path

**`app/rag/answer.py` — `synthesize()` — does not exist.** It is Contract C, frozen
since 9/21, and the orchestrator has been calling it all week.

Every answer currently returns `basis="unsynthesized"`: correct evidence, correct
citations, correct trace — and a placeholder where the prose should be. The
fallback deliberately does **not** write a plausible answer, because a convincing
stub is indistinguishable from a real one in a demo and in the evaluation.

Both demo tasks are otherwise working end to end. **This one function is what stands
between the demo and real answers.** Nothing else in the project is blocked on
anything.

---

## Open items by owner

### Rob
- [ ] **`app/rag/answer.py` — `synthesize()`.** Contract C, `docs/CONTRACTS.md`.
      Signature and return shape are frozen; the orchestrator already assembles and
      passes the evidence. → **Critical path**
- [ ] Retrieval ablation numbers (k ∈ {3,5,8}, chunk sizes, hybrid vs vector-only —
      `retrieve(mode="vector_only")` already exists for this)
- [ ] Eval items for retrieval quality → `evaluation/eval_set.rob.json`
- [ ] RAG section of `design-and-evaluation.md`

### Eric
- [ ] **`app/api.py`** — `/chat` and `/health`. Contract B. → blocks deployment and
      the UI
- [ ] **`ui/streamlit_app.py`** — chat, citation cards, trace panel, confirmation
      dialog, health badge, two "Run demo task" buttons
- [ ] **`render.yaml` + deploy.** Two free Web Services. Details and the free-tier
      limits are in `docs/DEPLOYMENT-NOTES.md`
- [ ] **CI has no Linux runner, no pytest step, no deploy job.** 273 tests exist and
      CI runs none of them. Render is Linux and CI never tests Linux. The brief
      requires deployment be gated on tests passing — that gate does not exist.
      Deploy trigger is a **deploy hook** in GitHub secrets; no Render API key needed
- [ ] `evaluation/run_eval.py` — the harness
- [ ] Policy-Q&A eval items → `evaluation/eval_set.eric.json`
- [ ] `deployed.md`, and the deployed URL in `README.md`

### Chris
- [x] Agent + MCP spine, trace, guardrails, mock-data adapter
- [x] 14 eval items + `evaluation/README.md` (the item format)
- [x] `docs/DEMO-TASKS.md` — both tasks with their MCP call sequences
- [x] Rob's retrieval wired into the MCP tools
- [ ] Architecture diagram — all 7 components
- [ ] `design-and-evaluation.md`: agent orchestration, MCP design, transport, tool
      schemas, safety guardrails
- [ ] Final edit of `design-and-evaluation.md` into one voice

### Everyone
- [ ] **Groq API keys to Chris.** `app/llm.py` rotates across three; we have one.
      Chris's daily token limit hit zero mid-evaluation on 9/23 — 200k tokens/day,
      and a rehearsal plus a recording on 10/1 is easily 30+ agent turns. The
      rotation is built and unarmed. → **do this today**
- [ ] Your paragraph in `ai-tooling.md` (nobody owns creating the file — Chris will)
- [ ] `quantic-grader` accepted on your own mirror. Chris's is accepted; check yours
- [ ] Government ID in hand for 10/1 — the brief requires all three on camera with it

---

## Evaluation set

`evaluation/README.md` is **Contract E** — the item format, agreed before the items
exist rather than after. Separate `eval_set.<author>.json` files so three people are
not merge-conflicting one array.

| Author | Items | Covers |
|---|---|---|
| Chris | **14** | agentic, tool-requiring, ambiguous, out-of-scope, action-safety, escalation |
| Eric | 0 | policy Q&A gold answers for the corpus he wrote |
| Rob | 0 | retrieval quality, expected `doc_id` citations |

The brief wants 20–30. We need roughly 12 more, split between Eric and Rob, before
Monday's freeze.

**Every item is run before it is committed.** `evaluation/verify_expectations.py`
executes each one against the live system and reports whether its `expected` block
actually holds. An eval set written from intention measures the author's optimism
and passes by construction on the day it is written.

That is not theoretical — running Chris's 14 for the first time gave **8/14** and
found four real agent bugs: the agent answering a benefits question from policy
without checking the record, action requests being researched until the step budget
ran out so "file a ticket" returned advice, a correct refusal being silently
overwritten and lost, and a model failure in `classify` taking the whole turn down.
All fixed. **10/14 now**; two are blocked on `synthesize()` and one is unverified
because the token budget ran out.

---

## Known gaps, recorded rather than discovered later

- **Tool-call sequences are not deterministic.** The same question produced 5–7
  calls across runs at `temperature=0` with a fixed seed — Groq's seed is
  best-effort. On 10/1, narrate the trace panel on screen rather than a memorised
  script. `docs/DEMO-TASKS.md` states the calls that *must* happen separately from
  one observed trace.
- **Free-tier cold start** is ~1 minute after 15 minutes idle, and the 750 monthly
  instance hours are **shared across both services**, not per service. Hit both URLs
  a few minutes before recording. Cold and warm latency must be reported separately
  in the evaluation — the brief asks for it.
- **AG-03, AG-06** in the eval set carry an explicit `blocked_on` for `synthesize()`
  rather than a weakened expectation. They check answer text, which cannot pass
  until the prose exists.

---

## Reference

| Document | What it is |
|---|---|
| `docs/CONTRACTS.md` | Contracts A–D: tool schemas, `/chat` envelope, `synthesize()`, doc-id registry |
| `docs/DEMO-TASKS.md` | The two agentic tasks and their MCP call sequences |
| `docs/DEPLOYMENT-NOTES.md` | Render product choice, free-tier limits, the dependency measurement |
| `evaluation/README.md` | Contract E: the eval item format |
| `Group_Assignment.md` | The lane split |
