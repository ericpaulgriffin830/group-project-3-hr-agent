# Demo script — recording Thu 10/1

**7–10 minutes. All three on camera, all three speaking, all three showing
government ID.** Those are brief requirements, not preferences.

Timings below total **8:30**, leaving buffer inside the window. Per-person split is
roughly even.

---

## Before you hit record

- [ ] **Warm both services.** Free instances spin down after 15 minutes idle and
  take ~1 minute to wake. Load the UI and run one throwaway question five
  minutes before starting, or the first thing on camera is a loading screen.
- [ ] **Government ID in hand** — all three.
- [ ] **One browser tab, UI already open**, trace panel visible.
- [ ] **Don't hammer it during rehearsal.** A throttled call degrades *quietly* —
  the agent falls through with less evidence and still answers. The trace panel
  shows the guardrail step; the answer text does not. Space the retakes.
- [ ] Check `/health` reads `mcp_connected: true`, `tools_discovered: 8`,
  `index_ready: true`.

---

## 0:00–0:45 — Open (Chris)

IDs on camera here, all three, then straight into it.

> "Northbridge HR assistant. It answers policy questions from a twelve-document
> corpus and runs multi-step HR workflows against employee records, and every tool
> it calls goes through MCP. We'll show two agentic tasks end to end, then the
> architecture, deployment, CI and evaluation."

---

## 0:45–2:15 — Architecture (Chris, 1:30)

`docs/ARCHITECTURE.md` on screen — the seven-component diagram.

Hit these, not everything:

- **Seven components**, coloured by owner. Web app → orchestrator → MCP client →
  MCP server → RAG index / mock data → LLM.
- **The agent never imports a tool function.** Everything crosses a real MCP
  session. *"The brief says hard-coded direct calls don't count — so a test walks
  the AST of everything under `app/agent/` and fails if the server appears in an
  import."*
- **LangGraph state graph, not a prebuilt ReAct loop** — because a prebuilt loop
  has nowhere to put *stop and wait for a human*, and the confirmation gate is one
  conditional edge.

---

## 2:15–4:15 — Demo Task A (Chris, 2:00)

**"Can I work from Colorado for six weeks? I am E1007."**

Let it run. Narrate the **trace panel as it appears** — do not recite a memorised
sequence. The tool order varies run to run; Groq's seed is best-effort even at
temperature 0.

Point at, in this order:

1. **`lookup_employee_profile`** — identifies *who is asking* before answering.
   Sofia is Illinois-based, so Colorado crosses a state line *for her*.
2. **`search_policy_documents` with no filter** — breadth before depth. *"This is
   the fix that made this task multi-document. It used to filter straight to the
   remote-work policy and never look at anything else."*
3. **`get_policy_section`** — exact section by id, so the citation quotes the rule.
4. **The citations** — `REMOTE-WORK` and `TAX-LOCATION`, two policy areas. *"This
   is the brief's multi-document requirement, visible."*

If it also cites `INFOSEC`, say so — three is better and it happens often.

---

## 4:15–6:15 — Demo Task B + the safety gate (Eric, 2:00)

**"Can I take three days of PTO in mid-October? I am E1008."**

The answer is **no, for two independent reasons**, both checkable:

- **1.5 days available** against a three-day request
- **A Consulting Delivery blackout**, 10/05–10/16, citing `PTO-9`

> "And the blackout is scoped. A Data Engineering colleague asking the same
> question sees only the company-wide year-end window — so the agent can't warn the
> wrong person about the wrong dates."

**Then the gate.** *"File a ticket asking for an exception."*

It stops and asks. **Nothing has been created.** Three things to say while the
confirmation dialog is up:

- The token is an **HMAC over the arguments**, so a token approved for this preview
  can't be replayed against a different ticket.
- The model is **never offered `confirm_token`** — it's stripped from the schema it
  sees, so it can't invent one.
- The token is **redacted from the trace**, because the trace renders in a browser.

Confirm it, show the ticket id.

---

## 6:15–7:15 — Deployment + CI/CD (Eric, 1:00)

- Two free Render Web Services, built from `render.yaml`. MCP server runs
  in-process over stdio — the free-tier architecture the brief recommends.
- **`buildCommand` builds the Chroma index at deploy**, so the index ships with the
  service.
- **CI runs the full suite on Windows, macOS and Linux, and the deploy job is
  gated on `needs: test`.** *"Deployment only happens if tests pass"* — say it,
  it's an explicit rubric line.
- **Cold start:** 15 minutes idle, ~1 minute to wake. Say it out loud; the brief
  asks for it to be explained.

---

## 7:15–8:15 — RAG + evaluation (Rob, 1:00)

Diagram: [`docs/RAG-PIPELINE.md`](RAG-PIPELINE.md) — data flow through every
`app/rag/` file, if there's a question about how a piece connects.

- **Corpus:** 12 HR policy documents, 106 sections. **Chunking:** one section, one
  chunk, never split — so whatever a query hits already contains everything
  `get_policy_section` would return; no second call needed.
- **Retrieval is hybrid:**
  - BM25 keyword search plus vector search over
    `BAAI/bge-small-en-v1.5` embeddings — via **fastembed**, not
    sentence-transformers, goes over Render's free tier storage.
  - The two legs are fused with **weighted RRF**, not a plain
    average. BM25 and cosine similarity are different scales. Testing with equal
    weighting let a keyword collision outrank the actually-relevant document until
    it was reweighted toward vector. Capped at two chunks per document, so one
    heavily-covered policy can't fill every slot.
- **Vector store:** Chroma — small, local, rebuilt from the corpus on every
  deploy. Not a hosted database; 106 chunks doesn't need one.
- **Evaluation:** 27 items across all five required categories, every one run
  against the live system before being committed, not written from intention.
  Three separate harnesses: a pass/fail gate, full accuracy numbers, and a
  no-LLM retrieval ablation that isolates the retriever from model behavior.

> "24 of 27 fully correct. All three misses are multi-document citation cases —
> and the ablation is what tells us which layer actually failed. Two of the three
> had the right documents retrieved already; that's a synthesis problem, not a
> retrieval one. And on this corpus, hybrid and vector-only score about the same
> on ranking quality — hybrid earns its place by degrading gracefully instead of
> failing outright when the vector index isn't available, not by ranking better
> day to day."

---

## 8:15–8:30 — Close (Chris)

> "Repo, deployed URLs and the full design write-up are in the README. Thanks."

---

## If something breaks on camera

**Don't restart the recording for a slow response.** Say *"free tier, cold start"*
and keep talking — the brief explicitly asks you to explain cold-start behaviour,
so it's on-script rather than a failure.

**If an answer comes back thin or misses a citation**, check the trace for a
guardrail step with `status: error`. That's a rate limit, not the agent. Say so,
and re-run the question — that's more credible than pretending it didn't happen,
and graceful degradation is a rubric item you're demonstrating live.

**If a tool errors**, that's also fine to show: tools never raise, they return an
error payload and the agent degrades. One sentence and move on.

---

## Assignments

| Segment                   | Who   | Time |
| ------------------------- | ----- | ---- |
| Open + IDs                | Chris | 0:45 |
| Architecture              | Chris | 1:30 |
| Demo Task A               | Chris | 2:00 |
| Demo Task B + safety gate | Eric  | 2:00 |
| Deployment + CI/CD        | Eric  | 1:00 |
| RAG + evaluation          | Rob   | 1:00 |
| Close                     | Chris | 0:15 |

Eric presents Task B because he built the UI the confirmation dialog renders in.
Rob presents evaluation because the retrieval numbers are his. Swap if anyone
would rather — but each person must speak, and the brief requires the presenter to
explain the MCP tool names, arguments, outputs and citations for their task.
