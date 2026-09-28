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

*(Rob: your paragraph. What tool, how you used it, what worked, and — the part the
brief asks for specifically — what didn't.)*

---

## Eric — UI, API and deployment

*(Eric: same. Worth including the remote-device bridge limitation you hit, where
workflow files couldn't be edited through it — that's exactly the kind of "what
did not work" the brief is asking for.)*
