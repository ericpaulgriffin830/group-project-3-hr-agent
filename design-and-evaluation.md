# Design and Evaluation

> **Assembly note — delete before submission.** Rob's RAG section and Eric's
> deployment section are still to come; Chris does the final edit into one voice.
> Sections below marked *(Chris)* are complete. Architecture diagram:
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 1. Architecture overview

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the diagram of all seven
components, a sequence diagram of one turn, and the deployment topology.

In one sentence: a Streamlit UI calls a FastAPI `/chat` endpoint, which runs a
LangGraph orchestrator that reaches every tool through an MCP client over a real
session; two of those tools are backed by a Chroma-backed hybrid retriever over a
12-document policy corpus, six by synthetic HR datasets, and every model call goes
through one client against Groq.

---

## 2. Agent orchestration *(Chris)*

### Why a state graph rather than a loop

LangGraph, with a hand-written tool loop rather than a prebuilt ReAct agent.

The decision that drove this was the **confirmation gate**. The brief requires
irreversible actions be confirmed, and a prebuilt agent loop has one shape: model
proposes, tool executes, result returns, repeat. There is no natural place in it to
say *stop, end this turn, and wait for a human*. As an explicit graph, that is one
conditional edge:

```
START → classify → retrieve ──────────→ synthesize → END
                 ↘ agent ⇄ tools ──────↗
```

`after_tools` routes to `synthesize` when a write is pending, instead of back to
`agent`. If it looped back, the *model* would be the thing confirming — exactly
what the gate exists to prevent.

### The nodes

| Node | Does | LLM? |
|---|---|---|
| `classify` | Routes to `policy_qa` / `workflow` / `clarify` / `refuse`. This is rubric item 4's "decide whether RAG alone is sufficient" | yes |
| `retrieve` | RAG-only path: one retrieval, straight to synthesis | no |
| `agent` | Chooses the next tool, or stops | yes |
| `tools` | Executes chosen tools through the MCP client, applies action guardrails | no |
| `synthesize` | Hands evidence to `synthesize()` (Contract C) | yes, in `answer.py` |

**The tool loop is capped at 6 steps.** Without a cap, a model that keeps re-calling
a failing tool loops until the request times out — which is a worse failure than a
partial answer, because it produces nothing at all.

### Decisions worth defending

**`classify` is a separate LLM call, not a branch inside the agent prompt.** It
costs one round trip (measured: ~0.4s of a ~12s turn) and buys a routing decision
that is visible in the trace and scoreable in the evaluation. Asking the agent to
decide its own mode mid-prompt would bury that decision where nothing can measure
it.

**Escalation is deterministic, not model-judged.** Harassment, discrimination,
safety and legal topics route to a human via word-boundary-anchored patterns in
`guardrails.py`. A model deciding case by case will occasionally decide wrong on
the one that mattered, and it isn't reproducible across evaluation runs.
Deterministic patterns over-trigger sometimes, which costs a needless handoff — the
cheap direction to be wrong in. The expensive one is answering a harassment
question with a tidy policy citation and no handoff.

**No evidence produces a refusal, not an answer.** If retrieval returned nothing
and no tool data was gathered, answering would be model priors, which is precisely
what the groundedness metric is meant to catch.

**A model failure never takes the turn down.** Both `classify` and `agent` catch
provider errors, record them as a traced guardrail step, and fall through. This was
not theoretical: Groq returned HTTP 400 `output_parse_failed` (gpt-oss emitting
reasoning prose where a tool call belonged) and later 429s when the daily token
budget ran out. Unhandled, either would have been a dead demo.

### Answering multiple policies at once

Rubric item 3 requires a question needing several documents. Demo Task A — six
weeks of remote work in another state — needs remote work, tax location *and*
information security. Getting it to actually do that took three fixes, and it is
worth recording that **two of the three were not in the retriever**:

1. Retrieval weighted every query term equally, so "work" (in nearly every policy)
   counted the same as "nexus" (in one).
2. Top-k had no per-document diversity, so the document with the most sections
   filled every slot.
3. **The agent was self-filtering** — every search went out with
   `doc_filter=["REMOTE-WORK"]`, so it never looked at another policy regardless of
   how retrieval ranked. The prompt now says to search broadly before narrowing.

Citation selection then needed two attempts. Round-robin across every document
present fixed Task A and broke Task B — a PTO question started citing
`REMOTE-WORK` and `TAX-LOCATION`, because "spread the citations" and "cite the
right things" are different goals and only the second is scored. The shipped
version takes the best by score and admits a second document only if its best chunk
is within 40% of the top score.

---

## 3. MCP design *(Chris)*

### Tools

Eight, against the brief's minimum of five, with at least one RAG-backed and at
least one using mock data. Full schemas in
[`docs/CONTRACTS.md`](docs/CONTRACTS.md) (Contract A) and modelled as pydantic in
`mcp_server/schemas.py`.

Two are RAG-backed (`search_policy_documents`, `get_policy_section`), three read
synthetic HR data, one is composite, and two are confirmation-gated writes.

### The rule that shapes everything: tools never raise

Every tool returns either a success payload or `{"error", "message"}` with a code
from a closed set. A tool that raised across the MCP boundary would take the
agent's whole turn with it, and the brief requires unavailable tools be handled
gracefully.

That distinction is load-bearing in the client too: **a tool declining keeps its own
error code** rather than collapsing into `tool_unavailable`. "That employee does not
exist" and "the server is down" are different answers to the user, and merging them
would make the orchestrator degrade to RAG-only when the tool worked fine.

### Discovery is real

`MCPClient.discover()` calls `list_tools()` at startup and **nothing in the codebase
holds a list of tool names the server did not supply**. A test parses the client for
any collection literal containing two or more known tool names, so a hardcoded
roster fails the suite.

The brief is explicit that hard-coded direct function calls don't count unless
wrapped and invoked through the MCP layer. That's easy to violate by accident and
impossible to see in review — importing `lookup_employee_profile` directly would
work perfectly. So a test walks the AST of every file under `app/agent/` and fails
if `mcp_server` appears in an import. Via AST rather than grep, so it cannot slip in
disguised by formatting.

### The LangChain bridge is hand-written, and why

`langchain-mcp-adapters` exists for exactly this job and **does not work here** — it
imports `RequestContext` from `mcp.shared.context`, which MCP 2.x removed. It
resolves cleanly and then fails at import, which is the worst kind of incompatibility
because `uv sync` stays green.

So `app/agent/tools.py` converts discovered MCP tools into the function-calling shape
`bind_tools()` accepts. It deliberately **executes nothing** — these are
*descriptions*. Execution goes through `MCPClient.call()`. A LangChain tool holding a
local Python callable would be executed by the framework directly, making the brief's
requirement false while everything still appeared to work.

It also **strips `confirm_token` from the schema the model sees**. The token is issued
by the server after a preview and supplied by the human through the UI; a model that
believed it could pass one would try to invent it, which defeats the gate entirely.

### Transport

`MCP_TRANSPORT` selects `stdio` (subprocess — local dev, CI, and the deployed
single-service topology) or `streamable-http` (`MCP_SERVER_URL`, for a split
deployment). **Nothing else in the codebase branches on transport.**

stdio is the default because the brief's recommended free-tier architecture runs the
MCP server in-process with the web app, and because it needs no port, no health
check and no second service. Tests use an in-process session for speed, with one
test deliberately spawning a real subprocess — without it the suite could stay green
while the shipped path was broken.

---

## 4. Tool schemas and contract discipline *(Chris)*

Contracts were frozen on day one, before any of the three lanes had code, so that
all three could build in parallel: **A** the tool schemas, **B** the `/chat`
envelope, **C** `synthesize()`, **D** the document-id registry, **E** the evaluation
item format.

The discipline that made them worth having is that **the contract is executable**.
`mcp_server/schemas.py` models Contract A in pydantic with `extra="forbid"`, and
`tests/test_schemas.py` calls every tool for real and validates the actual payload.
Validating a hand-written dict would only prove the models parse themselves.

Verified by breaking it on purpose: adding an undeclared field to a return produces
1 failure, renaming a tool produces 4.

Three drifts this caught or would have caught:

**`retrieval_mode: "fixture"`** — returned by the server, absent from Contract A's
enum. Added rather than disguised as `keyword_only`: a trace reader should be able
to tell canned data from a real index.

**Contract D, rewritten wholesale.** It originally specified twelve lowercase
hyphenated ids (`remote-work`, `pto-and-leave`) with **zero overlap** with the
corpus that actually shipped (`REMOTE-WORK`, `PTO-HOLIDAYS`). The real corpus won
— its section ids are written into the document bodies, so changing them means
rewriting twelve documents, while amending a table is an edit. Citations produced
against the fixtures stayed valid when Rob's real index replaced the ranking
underneath, because the fixtures had always used his identifiers.

**`check_policy_compliance` was still emitting the old ids** — missed when the
fixtures were converted, found by Rob while tracing evidence into `synthesize()`.
Every citation it contributed pointed at a document that does not exist. Now guarded
by a test that resolves every emitted ref, plus a sweep asserting no tool emits a
`doc_id` outside the manifest.

**Contract C was wrong about its own seam.** It typed `chunks: list[Chunk]` and
`tool_results: dict | None`; the orchestrator passes heterogeneous `list[dict]` and
`list[dict]`. The heterogeneity is the part that matters:
`search_policy_documents` contributes `{doc_id, title, section, snippet, score}`
while `check_policy_compliance` contributes `{doc_id, section}`, and both land in
the same evidence list — so every field but `doc_id` has to be optional. Corrected,
with a test pinning the documented signature to the shipped one.

---

## 5. Safety guardrails *(Chris)*

The split: Rob owns **evidence** guardrails (don't assert what the corpus doesn't
support). This section covers **action** guardrails (don't act without authority).

### The confirmation gate

Both write tools are two-phase. The first call returns a preview and a token and
**writes nothing**; only a second call carrying that token acts.

The token is an **HMAC over the tool name and the canonicalised arguments**. That
argument binding is the whole point: without it, an agent could preview a harmless
ticket, get it confirmed, and then submit the token with a different payload — and
the human would have approved something they never read. Tests pin all four cases:
absent, forged, expired, and **arguments mutated**.

Enforcement is deliberately split. The server mints and verifies, because the server
performs the write. The agent-side policy — which tools are gated, what the user is
shown, that a pending write ends the turn — lives in `guardrails.py`. A gate only in
the agent could be walked around by calling the tool directly; a gate only in the
server would leave the agent with no idea it should stop and ask.

### Refusal to act

Two things a valid token does not authorise:

- **No identified subject.** The agent will not file on behalf of someone it cannot
  name.
- **Someone else's record.** Confirming a ticket against your own record does not
  authorise one against a colleague's, and that is not the agent's call to make.

A refusal **ends the turn**. An early version let the graph loop back to the agent,
which re-entered the tool node and reset the refusal to `None` — so a correct
refusal reached the user as a generic "evidence gathered" answer. Found by the
evaluation set, not by review.

### Escalation

Deterministic patterns route harassment, discrimination, safety, legal and
whistleblowing topics to a named human. Two design points:

**The handoff goes in the answer text, not just the envelope.** An earlier version
set `escalation` in the response and said nothing in the reply — so someone
describing harassment read "I could not find anything in company policy that covers
that" with no hint to talk to anyone. The `escalation` field is for the UI and the
metrics; neither of those is the person who asked.

**Escalation is not refusal.** Someone asking about a disability accommodation gets
the policy *and* a named human. `answer_anyway` stays true.

### Never exposing chain-of-thought

The brief is explicit. `trace.py` **rejects** any step carrying `reasoning`,
`chain_of_thought`, `thought`, `scratchpad`, or similar — loudly, rather than
dropping it silently, so the mistake cannot recur quietly.

That leak is easy and invisible: gpt-oss returns a `reasoning` field right beside
`content`, so a node forwarding a raw provider response would ship literal
chain-of-thought to the UI with nothing appearing wrong. `app/llm.py` returns only
`content` for the same reason.

The trace also **redacts `confirm_token`** from recorded arguments. It renders in a
browser and gets pasted into bug reports; it should show that a token was present,
not what it was.

---

## 6. Determinism and reproducibility *(Chris)*

Every model call goes through `app/llm.py` at `temperature=0` with a fixed seed,
because evaluation reruns only compare if those are set in exactly one place. Two
clients exist — the LangChain one for the graph, the raw one for `synthesize()` —
but the model id and seed are defined once and imported by both.

**Groq's seed is best-effort, not a guarantee.** The same question produced between
5 and 7 tool calls across runs. Anything that must be byte-stable is computed from a
digest instead: confirmation tokens, and mock ticket ids, which originally used
Python's `hash()` and therefore changed every process because string hashing is
salted per run.

Key rotation across three Groq accounts is `with_fallbacks`, and the **order of
composition is load-bearing**: `with_fallbacks([...]).bind_tools(...)` returns
another `RunnableWithFallbacks` that looks correct and does not rotate. Binding
first and composing after is what actually falls through. Because the orchestrator
uses the plain model only for `classify` and the bound model for the entire tool
loop, the wrong order meant the half doing the work had no rotation at all — a
throttled turn classified correctly and then made zero tool calls. A regression test
keeps both orders visible.

---

## 7. RAG design *(Rob — to come)*

Chunking strategy, embedding model, retrieval k, vector store, hybrid + RRF.

## 8. Deployment *(Eric — to come)*

Render topology, free-tier constraints, cold-start behaviour, CI/CD gating.
Measurements available in [`docs/DEPLOYMENT-NOTES.md`](docs/DEPLOYMENT-NOTES.md).

## 9. Evaluation results *(Chris)*

Item format is Contract E, in [`evaluation/README.md`](evaluation/README.md). 27
items across the five kinds the brief names, in three files so three people can
edit them without a merge conflict every time.

**Method note worth keeping:** every item is executed against the live system
before being committed, because an eval set written from intention measures the
author's optimism and passes by construction on the day it is written. Running
Chris's 14 for the first time gave 8/14 and found four real agent bugs.

Three harnesses, because they answer three different questions:

| Script | Question | LLM? |
|---|---|---|
| `verify_expectations.py` | Does every item's `expected` block still hold? One bit per item, a pre-commit gate. | yes |
| `run_eval.py` | What are the accuracy numbers, by check and by category, and how fast is it? | yes |
| `ablation.py` | Does retrieval configuration matter? | **no** |

### 9.1 Answer quality and agent behaviour

Full 27-item run, `evaluation/results.json`, generated 2026-09-29 against the code
in this commit. Every check is scored only on the items that declare an
expectation of that kind — an item silent on citations is excluded from the
citation denominator rather than counted as a free pass, which is why the
denominators differ.

| Check | Result | |
|---|---|---|
| Intent classification | 26/26 | 100% |
| Tool selection | 10/10 | 100% |
| Citation accuracy | 12/15 | **80%** |
| Confirmation gate | 14/14 | 100% |
| Escalation / clarification | 14/14 | 100% |
| Answer content | 2/2 | 100% |

**24 of 27 items fully correct.** By category:

| Category | Items fully correct |
|---|---|
| `policy_qa` | 8/8 |
| `multi_document` | **2/5** |
| `agentic_multi_document` | 2/2 |
| `tool_required` | 4/4 |
| `action_safety` | 3/3 |
| `escalation` | 2/2 |
| `ambiguous` | 1/1 |
| `out_of_scope` | 2/2 |

Action safety is 3/3 and the confirmation gate 14/14, which matters more than the
headline: no run created a ticket or drafted an email without a human token, and
no item scored a forbidden tool call.

### 9.2 The three failures, and which layer each belongs to

All three are `multi_document` citation failures, all three route through the
RAG-only path at `RAG_ONLY_K = 5`, and none were throttled. Because the ablation
measures the same retrieval at the same k without a model in the way, the failures
can be attributed rather than guessed at:

| Item | Required | Retrieved at k=5 | Cited | Layer |
|---|---|---|---|---|
| MD-01 | `REMOTE-WORK`, `TAX-LOCATION`, `INFOSEC` | `REMOTE-WORK`, `TAX-LOCATION` | — | **retrieval** |
| MD-02 | `LEAVE`, `PTO-HOLIDAYS` | both | incomplete | **synthesis** |
| MD-05 | `ONBOARDING`, `TAX-LOCATION` | both | incomplete | **synthesis** |

So **one retrieval failure and two citation failures**, not three of the same
thing. MD-01 asks about working from Portugal and never surfaces the information
security policy — and raising k to 8 does not fix it, so it is a matching problem
rather than a depth problem. MD-02 and MD-05 had every required document in hand
and the answer still did not cite both.

That distinction is the whole reason the ablation exists. Without it, "citation
accuracy 80%" is one number with no address.

### 9.3 Retrieval ablation

`evaluation/ablation.py`, over the 15 items that declare `must_cite_doc_ids`,
scoring whether **every** required document was retrieved. No LLM: one variable
changes per cell, so every number here is reproducible to the chunk on a laptop
with no API key. Two consecutive runs were byte-identical.

13 retrieval-only items (`policy_qa` + `multi_document`):

| Mode | k | Items complete | Documents found | Mean docs returned |
|---|---|---|---|---|
| hybrid | 3 | 11/13 (85%) | 17/19 | 2.00 |
| hybrid | 5 | 12/13 (92%) | 18/19 | 3.23 |
| hybrid | 8 | 12/13 (92%) | 18/19 | 4.92 |
| vector-only | 3 | 12/13 (92%) | 18/19 | 2.08 |
| vector-only | 5 | 12/13 (92%) | 18/19 | 3.15 |
| vector-only | 8 | 12/13 (92%) | 18/19 | 5.15 |

**The honest reading: on this corpus and this eval set, hybrid and vector-only are
indistinguishable.** They are identical at k=5 and k=8, and hybrid is one item
*worse* at k=3. 12 documents and 106 chunks is not enough retrieval surface to
separate two good ranking methods, and reporting a win here would be reporting
noise. k=5 over k=3 is the one comparison that does separate: it recovers MD-05's
`TAX-LOCATION`, and going on to k=8 buys nothing but 1.7 more documents of context
per query.

**Where hybrid does earn its place is degradation.** `data/chroma` is a build
artifact, gitignored and rebuilt by `scripts.build_index` on every deploy. The
first run of this ablation was made before that had ever run locally, so the
collection held 0 chunks:

| Mode | k | Items complete | Vector leg |
|---|---|---|---|
| hybrid | 3 | 12/13 (92%) | **unavailable** |
| hybrid | 5 | 12/13 (92%) | **unavailable** |
| hybrid | 8 | 12/13 (92%) | **unavailable** |
| vector-only | any | **0/13 (0%)** | **unavailable** |

With the embeddings gone, BM25 carried the entire system at full accuracy while
vector-only retrieved nothing at all. That is the argument for fusing two
retrievers on a corpus this small — not ranking quality, but what survives when
half the machinery is missing. It is a cell in the report (`--no-degraded` turns it
off) rather than an anecdote, because it was an accident the first time.

**One defect found by that accident:** `retrieve()` reports
`retrieval_mode: "hybrid"` whether or not the vector leg returned anything, so a
BM25-only run is indistinguishable from a fused one in its own output. That is how
an empty index nearly got published as a real hybrid-vs-vector result. `ablation.py`
now refuses to run against an empty collection; the mislabel itself is still there
and is noted here rather than quietly fixed in someone else's module.

### 9.4 Latency

Two measurements, and the brief asks for both because they are not the same thing.

**Deployed, end to end over HTTP** — what a grader experiences. Measured
2026-09-29 against `https://hr-agent-api-s2ux.onrender.com` from overnight idle:

| | |
|---|---|
| Instance wake (`GET /health`) | **71.5s** |
| First question after wake (`POST /chat`) | **68s** |
| Warm | p50 **15.0s**, p95 68.1s, n=3 |

**Waking the instance and serving the first question are two separate costs**, and
this is the number `deployed.md` was missing. `/health` reports `index_ready` from
the Chroma collection's count, which does not force the fastembed ONNX model into
memory — the first retrieval does. So the service answers `/health` truthfully
while still being one large lazy load away from answering a question. A grader who
opens the app cold and asks immediately waits both, about 2m20s. The advice already
in `docs/DEMO-SCRIPT.md` and `deployed.md` — send one throwaway question first — is
what removes it, and `run_eval.py` now does exactly that and times it.

The warm figures are **n=3 and provisional**. They are labelled that way rather
than rounded up into a table that looks fuller than the evidence:

```bash
uv run python evaluation/run_eval.py \
  --api-base-url https://hr-agent-api-s2ux.onrender.com
```

That run is serial by default. The free instance is 0.1 CPU and 512 MB; at
concurrency 6 the p95 measures our own queueing rather than the service, and the
report says so when you raise it. **It 502'd once under light serial load on
2026-09-29** and recovered on its own in about four minutes — worth knowing before
relying on it live.

**In-process agent latency is deliberately not reported here.** The full 27-item
run at concurrency 6 gives p50 57.8s and p95 346s, and those numbers measure the
shared Groq token budget, not the agent: the SDK retries a 429 *inside* the
request, so rate-limit waiting sits inside the measured span and is
indistinguishable from slowness. Two items were throttled and retried serially.
The harness now prints that warning itself rather than leaving the trap for the
next reader.

### 9.5 What these numbers do not cover

- The warm percentiles are n=3. The command above, run once after deploy, fixes it.
- Two bugs found on 2026-09-29 while verifying the deployed app are fixed in this
  commit but were **not** deployed when the latency above was taken. Accuracy in
  9.1 is from the fixed code; the latency is from the previous build, where
  workflow turns refused early and were therefore *faster* than they now are.
  Re-take it after this merges.
- `eval_set.rob.json` does not exist, so the retrieval-quality items Contract E
  reserves for Rob are not in any of these denominators.

## 10. Demo tasks *(Chris — done)*

Both tasks, with required and observed MCP call sequences, in
[`docs/DEMO-TASKS.md`](docs/DEMO-TASKS.md).
