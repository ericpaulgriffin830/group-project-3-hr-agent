# Evaluation Set

This directory is Person C's evaluation-harness deliverable for the project (Section 9: Evaluation of the Agentic RAG Application). It defines a fixed, versioned question/task set the agent is run against, plus how to score the results.

## Files

- **`eval_questions.json`** — the 30-item evaluation set. Each item has: `id`, `category`, `question` (the user-facing prompt), `gold_answer` (the correct answer, grounded in the actual corpus/mock data), `expected_citations` (section IDs the answer should reference), `expected_tools` (MCP tools a correct agent run should call, if any), and `grading_notes` (what a scorer should specifically check for that item).
- **`eval_questions.json` is the source of truth** for both manual review and any automated scoring script (e.g. an LLM-as-judge runner) the team builds on top of it. Every `gold_answer`, `expected_citations`, and `expected_tools` value was checked directly against the current corpus (`corpus/policies/`) and mock data (`mock_data/`) files as of this writing — if either changes, re-validate this file (or re-run it through `validate.py`-style consistency checks) rather than assuming it's still accurate.

## Categories (per the project spec)

| Category | Count | Purpose |
|---|---|---|
| `straightforward` | 8 | Single-document policy lookups (Q01–Q08). Tests basic retrieval + groundedness. |
| `multi_document` | 5 | Questions that require synthesizing 2–3 policy documents (Q09–Q13), e.g. remote work + tax + security together. |
| `tool_requiring` | 9 | Tasks that require calling one or more MCP tools against mock structured data — PTO balance checks, benefits lookups, compliance checks, ticket creation, email drafting (Q14–Q22). Includes the required refusal-path case (Q15) and an action-safety case (Q21). |
| `ambiguous` | 4 | Underspecified requests that should trigger a clarifying question rather than a guessed answer (Q23–Q26). |
| `out_of_scope` | 4 | Requests outside the HR assistant's purpose, including a false-premise question (Q30) (Q27–Q30). |

30 items total, within the required 20–30 range.

## Mapping to evaluation metrics

Each category is designed to exercise specific metrics from the project's evaluation plan:

- **Groundedness** — all `straightforward` and `multi_document` items (Q01–Q13): does the answer's factual content match the cited section(s) verbatim/in substance, with no invented details?
- **Citation accuracy** — every item with a non-empty `expected_citations` list: does the agent cite the correct section ID(s), and only those (no extra, no missing, no mismatched doc-code-vs-section-ID confusion — see the `RW-3`/`RW-4` and `REMOTE-WORK-3` fix noted in `corpus/README.md`)?
- **Tool selection accuracy** — every item with a non-empty `expected_tools` list (Q14–Q22, Q30): did the agent call the correct tool(s) rather than answering from memory or guessing at mock data values?
- **Workflow completion rate** — Q14–Q22: did the agent carry a multi-step task through to a final, usable answer or action (not just an intermediate tool result)?
- **Escalation / clarification accuracy** — Q23–Q26: did the agent ask a clarifying question instead of fabricating an assumption (e.g. picking an arbitrary employee)?
- **Action-safety pass rate** — Q21 (ticket creation) and Q22 (email drafting) are the state-changing/communication actions in this set: did the agent confirm what it was doing rather than silently writing a record, and did it avoid hallucinating IDs or policy numbers it invented on the spot? Q15 (PTO refusal) is also an action-safety case in the sense that the agent must not approve an action it shouldn't.
- **Out-of-scope / hallucination resistance** — Q27–Q30: did the agent decline rather than fabricate an answer, including catching the false premise in Q30 (no Tokyo office exists per `offices.json`)?

## Latency benchmarking

For p50/p95 latency measurement, run the full 30-item set (or a repeated subset) through the deployed agent and record wall-clock time per item. The `tool_requiring` items (Q14–Q22) are the most representative of real end-to-end latency since they include retrieval + tool calls + generation; the `straightforward` items (Q01–Q08) are a useful lower-bound baseline since they typically need only retrieval + generation.

## Ablation / comparison requirement

The project requires at least one ablation or comparison study. This eval set is designed to be re-run under different configurations without changing the questions themselves, e.g.:

- **Retrieval k** — compare answer quality/citation accuracy at k=3 vs. k=6 retrieved chunks.
- **Chunk size** — compare section-level chunking (the corpus's native `PTO-3`-style units) vs. a smaller fixed-size chunk split.
- **Tool availability** — run the `tool_requiring` items (Q14–Q22) with MCP tools enabled vs. disabled, to quantify how much tool access improves correctness on the refusal-path (Q15) and compliance-check (Q17, Q20) cases specifically.
- **Prompt variant** — compare a stricter "cite every claim" system prompt vs. a looser one, measuring the effect on citation accuracy and hallucination rate on the `out_of_scope` items.

Record each run's per-item pass/fail (or score) alongside the configuration used, so the final evaluation report can present a clear before/after comparison table.

## Notes on grounding

Every `gold_answer` in `eval_questions.json` was verified against the actual files in `corpus/policies/` and `mock_data/` (not written from memory of the spec), including:

- Exact dollar figures (EXP-2's $500 initial / $250 annual home-office stipend cap; the $310 monitor example used in both Q11/Q20 and the real denied claim `EXP-3007`).
- Exact business-day figures (RW-4's 20-business-day international cap; PTO-9's 10-business-day blackout notice requirement vs. PTO-3's standard notice).
- The full 12-state Registered States list in TAX-2 (not just the four states with a physical office), used to ground Q12 and Q13 correctly.
- Real mock-data records: Marcus Chen (`E1005`, balance 7.0), Grace Lin (`E1008`, balance 1.5 — the intentional refusal-path case), Priya Nataraj's open ticket `TCK-10021`, Sofia Reyes (`E1007`, Consulting Delivery) against the `BO-2026-CONSULTING-CLIENT-GOLIVE` blackout window, Omar Farouk (`E1010`, part-time) and Jordan Blake (`E1011`, contractor) benefits/PTO ineligibility, and the two denied expense claims `EXP-3005`/`EXP-3007`.

If the corpus or mock data changes in a future update, re-check any affected `gold_answer`/`expected_citations` values rather than assuming this file still matches.
