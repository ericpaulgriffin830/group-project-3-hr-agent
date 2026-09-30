# AI tooling

The brief permits AI tools for building this system and asks us to describe how we
used them, **including what did not work**. One honest paragraph each.

---

## Chris — agent and MCP spine

**Tool:** Claude Code (Opus), driving the terminal directly — reading the repo,
writing files, running the test suite, calling the live Groq API, and exercising
the agent end to end. Essentially all of `app/agent/`, `mcp_server/`,
`app/llm.py`, my 14 evaluation items and the design documentation were produced
this way, with me reviewing and directing rather than typing.

**What worked.** Three things, and they were the same thing in different clothes:
*making the model check its work against something outside itself.*

Freezing the contracts before anyone had code — the eight tool schemas, the
`/chat` envelope, `synthesize()`, the document-id registry — meant three lanes
could be built in parallel against a shared shape. More useful still was making
the contract **executable**: `mcp_server/schemas.py` models it in pydantic and a
test calls every tool for real and validates the actual payload. We verified that
guard bites by breaking things deliberately — adding an undeclared field to a
return produced one failure, renaming a tool produced four.

The evaluation set was the other big win, because of one rule: **every item is run
against the live system before it is committed.** An eval set written from
intention measures the author's optimism and passes by construction on the day it
is written. Running my fourteen for the first time gave **8/14** and surfaced four
real bugs — the agent answering a benefits question from policy without checking
the record, action requests being researched until the step budget ran out so
"file a ticket" returned advice, a correct refusal being silently overwritten by
the graph looping back, and a model failure in the classifier taking the whole
turn down.

And the cross-lane effect was the biggest single find: **Eric's evaluation items
caught a bug in my classifier that mine could not**, because all of mine passed an
`employee_id` and his didn't. First-person questions with no employee attached
("am I eligible for parental leave?") were routing to the tool path with nobody to
look up, so the agent spent its budget on lookups it couldn't make and retrieved
less policy than the simple path would have.

**What did not work.** The model is confidently wrong in ways that cost real time,
and the pattern is consistent: it diagnoses from plausibility rather than
evidence, and the diagnosis sounds as authoritative as a verified one.

- A flaky evaluation item was called "real non-determinism in tool selection, not
  throttling" — the opposite of true. It was throttling. Acting on that would have
  meant tuning a prompt that worked fine. Two days later the same item genuinely
  *was* a tool-selection problem, which is exactly why the first wrong call was
  expensive: it made the second one harder to believe.
- Two commits were spent tuning a citation-selection function before anyone
  noticed it only runs inside an `except ImportError` fallback that no longer
  fires. Optimising dead code, confidently, with tests.
- A test comparing API keys reported "SAME ACCOUNT — rotation buys no extra
  quota." The test only compared keys that had *already failed*, and there was
  one. The conclusion was drawn from a sample of one and stated as a verdict.

None of these survived contact with a measurement. That is the whole lesson:
**the tool is excellent at producing and checking work, and unreliable at
diagnosing it.** The workflow that emerged — make it prove the claim with a
command before believing the claim — is what turned it from a fast way to write
plausible code into a fast way to write correct code. Several times the
verification step is what found the actual bug: the key rotation silently not
protecting the tool loop, `retrieve()` returning every chunk with `score=0.0`,
`check_policy_compliance` emitting document ids that no longer existed.

The honest summary is that it made me faster by a large factor and required
supervision throughout, and those are not in tension.

---

## Rob — knowledge and evidence

**Tool:** Claude Code (Sonnet), used conversationally rather than autonomously — I described the bug or the ask, Claude proposed a diagnosis and a diff, and nothing was written until I'd approved it. I used it for three distinct things: writing the RAG pipeline itself (`chunk.py`, `retrieve.py`, `answer.py`), brainstorming design tradeoffs before committing to one, and getting plain-English explanations of what the retrieval code and CI/CD wiring were actually doing.

**What worked.** The strongest results came from bug reports that arrived pre-narrowed by a teammate. When Chris reported "scores are all 0.0 in hybrid mode" with the exact query attached, Claude reproduced it, found the fused RRF score was never written back onto the chunk dict, fixed it, and wrote a regression test with that exact query baked in — so the same failure can't silently return. The same pattern held for the multi-document citation gap and for a subtler one: a test expected a `refusal` basis on no-evidence queries but got `policy_rag` instead. Claude traced it to two independent causes stacked on top of each other — the vector leg had no relevance floor, so noise chunks were treated as evidence, and separately `answer.py` was forcing a citation fallback even when the model had validly declined to cite anything. I asked for both fixes together rather than picking one, and that turned out to matter: fixing only the floor would have left the second bug masked until the next edge case surfaced it. Claude was also useful as an explainer, not just a coder — walking me through why the CI deploy job needed `RENDER_DEPLOY_HOOK_API`/`_UI` as GitHub secrets rather than environment variables took one short exchange, and I could execute the fix myself with no follow-up questions. Writing the RAG design section of the evaluation doc worked the same way: I fed it the actual failure cases we'd hit (the Colorado/LEAVE-3 fusion-weight problem, the two-chunk-split bug, the fastembed-vs-sentence-transformers footprint decision) and it produced a "justify, don't describe" writeup in the voice the rest of the doc already used, rather than a generic architecture summary.

**What did not work.** Chris and I were both using AI to debug the same symptom independently, and our two sessions disagreed. Chris flagged "scores all 0.0 in hybrid mode" a second time, after we'd already shipped a fix for it. Rather than assuming it was a fresh regression, my session re-ran the exact regression test written for the original fix, confirmed it still passed, and concluded the report was almost certainly coming from a stale checkout on his side rather than a second bug. His AI-assisted debugging, working from his branch and his symptoms, had no way to know a fix already existed on mine — so the same underlying question produced two confident, contradictory-looking answers from two AI sessions that simply didn't share context. It was the right call in the end, but it cost a round of cross-checking that wouldn't have been needed if we'd been debugging from the same source of truth. The lesson generalizes beyond this one bug: AI tooling is only as good as the context it's given, and on a team, that context diverges by default unless someone deliberately synchronizes it.

---

## Eric — UI, API and deployment

*(Eric: same. Worth including the remote-device bridge limitation you hit, where
workflow files couldn't be edited through it — that's exactly the kind of "what
did not work" the brief is asking for.)*
