# Submission checklist

**Chris submits, on behalf of the group. Only one member submits.**

Two links go in: the recorded demo, and the GitHub repo. Work top to bottom on the
day — the ordering is deliberate, because several items can only be verified after
the one above.

---

## Blocking — nothing else matters until these are true

- [x] **Group-repo Render services exist and both respond** (Rob, 2026-09-28).
      Verified 09-29 from cold: UI 200 in 42s, `GET /health` 200 in 81s returning
      `mcp_connected: true, tools_discovered: 8, index_ready: true`. URLs are in
      `README.md` and `deployed.md`.
- [x] **`RENDER_DEPLOY_HOOK_API` and `RENDER_DEPLOY_HOOK_UI` in the group repo's
      Actions secrets** (Rob, 2026-09-28). Confirmed live rather than assumed: the
      deploy job's two `curl`s each returned a real Render deploy id
      (`dep-dassk33bc2fs73a6au80`, `dep-dassk33bc2fs73a6av8g`). A missing secret
      posts to an empty string and fails; these did not.
- [x] **`main` is green.** 345 tests on Windows, macOS and Linux.
- [ ] **`quantic-grader` is an ACCEPTED collaborator on all three repos** — not a
      pending invite. Invites expire after 7 days, and a pending one reads to a
      grader as no submission.
      ```bash
      for r in cmccoy2008 Rob-Ottogalli ericpaulgriffin830; do
        printf "%-22s " "$r"
        gh api "repos/$r/quantic-hr-ai-rag-group-project/collaborators" \
          --jq '[.[].login]|join(", ")' 2>/dev/null || echo "NO REPO"
      done
      ```
      Re-checked 2026-09-29: Chris ✅ accepted · Rob ❌ no grader on the group repo
      (collaborators are the three of us only) · Eric ❌ no mirror repo yet.
      **The group repo is the one that gets submitted, so Rob's is the blocking
      one.**

---

## Required files in the repo

The brief names these explicitly.

- [ ] `README.md` — description, setup, local run, **deployment instructions**,
      and the deployed URL
- [ ] `design-and-evaluation.md` — architecture, RAG, MCP, orchestration, tool
      schemas, guardrails, deployment, evaluation questions + expected answers +
      results
- [ ] `ai-tooling.md` — a paragraph each, including **what did not work**
- [ ] `deployed.md` — URLs, health URL, cold-start notes
- [ ] `evaluation/` — questions, expected answers, scripts, reported results
- [ ] `mock_data/` — synthetic employee, PTO, benefits, ticket data
- [ ] `mcp_server/` — MCP server code and tool definitions *(the brief says
      `mcp/` "or equivalent"; ours is `mcp_server/` because a local `mcp/` package
      shadows the installed SDK)*

**Still open:** `design-and-evaluation.md` sections 7 (Rob) and 8 (Eric) are
stubs, and the assembly note at the top of that file must be **deleted** before
submission. `ai-tooling.md` needs Rob's and Eric's paragraphs.

---

## Evaluation results must be in the file, not just in a script

- [ ] `run_eval.py` executed against the **deployed** app, results committed
- [ ] Answer quality: groundedness, citation accuracy
- [ ] Agent behaviour: tool selection, workflow completion, escalation/clarification
      accuracy, action-safety pass rate
- [ ] Latency p50/p95, **cold-start and warm-start reported separately** — the
      brief asks for this specifically
- [ ] At least one ablation (k ∈ {3,5,8}, chunk size, or hybrid vs vector-only)
- [ ] Results tables land in `design-and-evaluation.md` section 9

**Run the eval against the deployed app, not localhost** — the URLs are in
`deployed.md`. Warm both services first, or the first few items absorb the
cold start and the p50 is meaningless.

**Watch for:** a throttled run reports plausible wrong answers rather than errors.
`verify_expectations.py` warns when this happens. If the warning fires, re-run at
`--concurrency=1` before believing the numbers.

---

## The recording — Thu 10/1

- [ ] 7–10 minutes
- [ ] **All three on camera, all three speaking**
- [ ] **All three show government ID**
- [ ] Both agentic tasks end to end **on the deployed app**
- [ ] For each task, the presenter explains the **MCP tool names, arguments,
      outputs, citations** and the final answer
- [ ] Walkthrough of design, deployment, CI/CD and evaluation results
- [ ] Uploaded, shareable link tested **in a private window** — a link only you can
      open is a link the grader cannot

Script and timings: `docs/DEMO-SCRIPT.md`. Warm both services five minutes before
recording.

---

## Submission day — Fri 10/2

- [ ] Final `main` green
- [ ] Mirror final `main` to Chris's repo
- [ ] Re-verify `quantic-grader` on all three repos (invites can expire between
      sending and submitting)
- [ ] Deployed URL present in `README.md` and `deployed.md`, and **both load**
- [ ] Signed Group Project Agreement final page uploaded
- [ ] Submit **both** links — demo video and repo — via the dashboard
- [ ] Confirm only one of us submitted

---

## Last look before hitting submit

Three things that are cheap to check and expensive to get wrong:

**Open the deployed URL in a private window.** Everything else has been tested
while signed in somewhere.

**Read the top of `design-and-evaluation.md`.** The assembly note is addressed to
us, not to a grader.

**Run one demo task on the deployed app.** If it has been idle it will cold-start —
which is the point of checking rather than discovering it on camera.
