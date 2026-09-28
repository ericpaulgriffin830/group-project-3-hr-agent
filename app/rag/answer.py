"""Answer synthesis. Contract C's seam between retrieval/tools and the agent.

Contract C types this module's `chunks` argument as `list[Chunk]`. In practice the
orchestrator (app/agent/orchestrator.py, `run_tools`) never constructs a `Chunk` --
it builds a plain-dict `evidence` list by `.extend()`-ing together whatever
citation-shaped payload each tool call returned:

    if call.ok and "chunks" in call.payload:      evidence.extend(call.payload["chunks"])
    if call.ok and call.payload.get("policy_refs"): evidence.extend(call.payload["policy_refs"])

`search_policy_documents` contributes fully-shaped dicts (doc_id, title, section,
snippet, score). `check_policy_compliance` contributes `policy_refs` entries that
are only {doc_id, section} -- no title, no snippet, no score. Both land in the same
list. So `chunks` here is read as a heterogeneous `list[dict]`, every field
`.get()`-accessed defensively, never assumed present. A real `app.rag.chunk.Chunk`
dataclass instance is also accepted (Contract C's literal type, for a caller that
bypasses the MCP layer entirely) -- `_as_dict()` normalizes either shape to the
same dict interface before anything else runs.

Chris is separately unifying the shapes server-side (2026-09-24 sync); this module
does not assume that has landed, because a stale `policy_refs` entry using a
pre-Contract-D doc id (e.g. "pto-and-leave" instead of "PTO-HOLIDAYS") should
degrade to "this reference could not be resolved" rather than crash -- see
`_enrich()`.

Similarly, Contract C types `tool_results` as `dict | None`; the orchestrator
always passes `list[dict]` (each `{tool, ok, payload, error}`), never a bare dict.
This module's signature says what actually arrives.

**How grounding is enforced.** The model is never trusted to say which citations
are real. It is asked to end its answer with a small structured trailer --
`cited_doc_ids` and `unsupported_flags` -- and this module then builds the
`citations` field ITSELF, by looking up each cited doc_id against the evidence
that was actually put in front of the model. A doc_id the model didn't cite is
dropped; a doc_id it invents that never appeared in the prompt cannot match
anything and is silently ignored. The model's prose is free-form; what gets
returned as a citation is not.

`basis` and `confidence` are likewise computed here, not asked of the model.
LLMs self-report confidence poorly, and `basis` only has four valid values -- both
are more reliably derived from what evidence was actually available and used.

**Multi-document citation gap (2026-09-27).** Three eval items were failing here
before this note: FMLA-01 (score bug, see NEUTRAL_SCORE/`_confidence` history) and
MD-01/MD-02, where retrieve() was already surfacing every expected document at
k=5 but the model's answer -- and therefore its META cited_doc_ids -- only ever
drew on one of them. Chris ruled out retrieval as the cause first: widening the
RAG-only path from k=5 to k=10 made MD-02 worse, not better (it went from citing
both LEAVE and PTO-HOLIDAYS to citing PTO-HOLIDAYS alone), which is the opposite
of what a retrieval-side gap would predict and consistent with a wider, noisier
context making the model lean on fewer passages rather than more. So the fix is
here, in SYSTEM_POLICY/SYSTEM_WORKFLOW and TRAILER_INSTRUCTION: explicit
instructions to read every passage shown and address each one that adds a
distinct condition, exception, or step, rather than stopping at the first
passage that looks sufficient. Nothing about k or the retriever changed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Literal

from app.llm import LLMError, LLMNotConfigured, complete
from app.rag.retrieve import get_section

Mode = Literal["policy", "workflow", "refusal"]

#: How many evidence passages go into the prompt. Not retrieval's job to enforce
#: (retrieve.py's own k and orchestrator._diverse_citations already narrow this
#: well below here) -- this is a hard ceiling against any caller passing more.
MAX_PROMPT_CHUNKS = 10

#: Citation snippet length, matching chunk.py's SNIPPET_CHARS convention.
SNIPPET_CHARS = 320

#: Confidence contribution for evidence with no numeric score -- notably
#: check_policy_compliance's policy_refs, which carry no `score` field at all.
#: Not 0 (that would look like retrieval actively rejected it) and not 1 (that
#: would overstate confidence in something no ranking ever touched).
NEUTRAL_SCORE = 0.55


SYSTEM_POLICY = """You are an internal HR assistant. Answer the question using \
ONLY the policy passages given to you below the question.

Rules:
- State only what the passages actually say. Never fill a gap with a plausible-
  sounding assumption about what a policy "probably" says.
- If the passages support a general rule but not every detail asked about, answer
  what they DO support and say plainly what they do not cover.
- Read EVERY passage before answering. Real questions often turn on how two or
  more policies interact (a leave policy and a PTO policy, a remote-work policy
  and a tax policy, an expense policy and an equipment policy). If more than one
  passage bears on the question -- each adds a distinct condition, exception, or
  required step -- your answer must address all of them, not just the first
  passage that looks sufficient. Stopping at one relevant passage when another
  qualifies or extends the answer is an incomplete answer, not a concise one.
- Distinguish a stated policy fact ("PTO accrues at 1.5 days per month") from a
  recommendation you are making ("you may want to confirm timing with your
  manager") -- never present the second as if it were the first.
- Write in plain, direct prose. No headers, no bullet lists, no markdown."""

SYSTEM_WORKFLOW = """You are an internal HR assistant answering a question \
specific to one employee. You are given that employee's own data (profile,
balances, benefits, or a compliance determination) AND relevant policy passages.

Rules:
- Use the employee-specific data for facts about THEM (their balance, their
  location, their eligibility) -- never guess these from policy text alone.
- Use the policy passages for the RULE that applies to them.
- Read EVERY policy passage before answering. If more than one bears on the
  question -- each adds a distinct condition, exception, or required step --
  address all of them, not just the first one that looks sufficient.
- If the employee-specific data needed to answer is missing, say so plainly
  rather than assuming a default case.
- State only what the provided data and passages actually say. Never fill a gap
  with a plausible-sounding assumption.
- Distinguish a stated fact from a recommendation you are making.
- Write in plain, direct prose. No headers, no bullet lists, no markdown."""

#: Appended to whichever system prompt applies. This is the grounding mechanism
#: described in the module docstring -- everything past "META:" is parsed, never
#: shown to the user, and the citations WE return are built from cited_doc_ids
#: cross-referenced against the evidence actually shown, not copied from here.
TRAILER_INSTRUCTION = """

After your answer, on new lines, output exactly this and nothing after it:
META:
{"cited_doc_ids": [...], "unsupported_flags": [...]}

- cited_doc_ids: the doc_id of every policy passage above that your answer
  actually relied on. If your answer drew on more than one passage -- which is
  common when policies interact -- list all of them; do not collapse to a
  single doc_id when several genuinely contributed. Omit any passage you did
  not use. Use doc_ids exactly as given, e.g. "PTO-HOLIDAYS".
- unsupported_flags: short phrases for anything in your answer that goes beyond
  what the provided evidence and data literally state (for example, "assumes a
  standard approval timeline not stated in the retrieved text"). Use [] if
  nothing does."""

_META_RE = re.compile(r"META:\s*(\{.*\})\s*\Z", re.S)


# --------------------------------------------------------------- normalization


def _as_dict(item: object) -> dict:
    """Accept a plain dict, a dataclass instance (Contract C's literal `Chunk`),
    or anything else with a __dict__, and return a plain dict either way."""
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return dict(getattr(item, "__dict__", {}))


def _truncate(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip() + "…"


def _enrich(chunk: dict) -> dict | None:
    """Fill in a sparse evidence dict with a real title/snippet by looking up the
    section directly -- answer.py lives in the same package as retrieve.py, so
    this is a plain function call, not a tool round-trip.

    Returns None if the reference cannot be resolved (unknown doc_id/section --
    including a pre-Contract-D id like check_policy_compliance's fixture still
    uses), since a citation with no real title or snippet is worse than no
    citation at all.
    """
    if chunk.get("title") and chunk.get("snippet"):
        return chunk
    doc_id = chunk.get("doc_id")
    section_id = chunk.get("section_id") or chunk.get("section")
    if not doc_id or not section_id:
        return None
    section = get_section(doc_id, section_id)
    if section.get("error"):
        return None
    return {
        "doc_id": section["doc_id"],
        "title": section["title"],
        "section": section["section"],
        "snippet": _truncate(section["text"]),
        "score": chunk.get("score"),
    }


def _to_citation(entry: dict) -> dict:
    """Strip an internal (enriched) evidence dict to Contract C's citation shape."""
    return {
        "doc_id": entry["doc_id"],
        "title": entry.get("title", ""),
        "section": entry.get("section", ""),
        "snippet": entry.get("snippet", ""),
    }


# --------------------------------------------------------------- prompt build


def _format_evidence(chunks: list[dict]) -> str:
    if not chunks:
        return "(no policy passages retrieved)"
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c['doc_id']} — {c.get('section', '')}\n{c.get('snippet', '')}")
    return "\n\n".join(lines)


def _format_tool_results(tool_results: list[dict]) -> str:
    if not tool_results:
        return "(no employee-specific data retrieved)"
    lines = []
    for r in tool_results:
        payload = {k: v for k, v in (r.get("payload") or {}).items()
                  if k not in ("error", "message")}
        lines.append(f"{r.get('tool')}: {json.dumps(payload, default=str)}")
    return "\n".join(lines)


def _build_prompt(question: str, mode: Mode, evidence_block: str, tool_block: str) -> str:
    if mode == "workflow":
        return (f"Employee-specific data:\n{tool_block}\n\n"
                f"Policy passages:\n{evidence_block}\n\n"
                f"Question: {question}")
    return f"Policy passages:\n{evidence_block}\n\nQuestion: {question}"


# ------------------------------------------------------------- response parse


def _clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^ANSWER:\s*", "", text, flags=re.I)
    return text.strip()


def _parse_completion(raw: str, fallback_ids: list[str]) -> tuple[str, list[str], list[str], bool]:
    """Split the model's raw text into (answer, cited_doc_ids, extra_flags, trailer_ok).

    Falls back to citing every doc_id that was actually shown to the model
    (`fallback_ids`) whenever the trailer is missing or malformed, rather than
    citing nothing -- a caller still gets grounded citations, just not a
    model-narrowed set, and the fallback itself is recorded as an
    unsupported_flag so it is visible, not silent.

    `trailer_ok` is the caller's signal for whether `cited_doc_ids` is a genuine
    answer from the model (True, even when that answer is a validly empty list --
    "I relied on none of these") versus a value THIS function had to invent
    because the trailer could not be read at all (False, in every branch below
    that falls back to `fallback_ids`). The two look identical by the time they
    reach the caller as a plain list -- an empty `cited` and a `cited` we
    replaced with `fallback_ids` for a different reason are not the same claim --
    so the distinction has to travel separately, or a model that correctly found
    nothing relevant looks the same as one whose trailer synthesize() gave up on
    parsing, and the caller cannot tell "nothing applies" from "we don't know
    what applies" without it.
    """
    match = _META_RE.search(raw.strip())
    if not match:
        return (_clean_answer(raw), fallback_ids,
                ["model did not emit the expected META block; citing all provided evidence"],
                False)

    answer_text = _clean_answer(raw[:match.start()])
    try:
        meta = json.loads(match.group(1))
    except json.JSONDecodeError:
        return (answer_text, fallback_ids,
                ["META block was not valid JSON; citing all provided evidence"], False)

    cited = meta.get("cited_doc_ids")
    flags = [f for f in (meta.get("unsupported_flags") or []) if isinstance(f, str)]
    if not isinstance(cited, list) or not all(isinstance(x, str) for x in cited):
        cited = fallback_ids
        flags.append("cited_doc_ids missing or malformed; citing all provided evidence")
        return answer_text, cited, flags, False
    return answer_text, cited, flags, True


# --------------------------------------------------------- basis / confidence


def _basis(resolved: list[dict], ok_tool_results: list[dict], mode: Mode) -> str:
    if mode == "refusal":
        return "refused"
    has_citations = bool(resolved)
    has_tools = bool(ok_tool_results)
    if has_citations and has_tools:
        return "both"
    if has_citations:
        return "policy_rag"
    if has_tools:
        return "tool_data"
    return "refused"


def _confidence(resolved: list[dict], ok_tool_results: list[dict], basis: str,
                unsupported_flags: list[str]) -> float:
    if basis == "refused":
        return 0.0
    if basis == "tool_data":
        # An exact structured lookup (a balance, a profile field), not a ranked
        # retrieval result -- there is no relevance score to average, and a flat
        # baseline is more honest than borrowing NEUTRAL_SCORE from the policy path.
        base = 0.75
    else:
        scores = [c.get("score") for c in resolved if isinstance(c.get("score"), (int, float))]
        base = max(0.0, min(1.0, (sum(scores) / len(scores)) if scores else NEUTRAL_SCORE))
        if basis == "both":
            base = (base + 0.75) / 2
    if unsupported_flags:
        base *= 0.6
    return round(base, 2)


# --------------------------------------------------------------- LLM fallback


def _llm_unavailable_fallback(resolved: list[dict], ok_tool_results: list[dict],
                              unsupported_flags: list[str], mode: Mode) -> dict:
    """app.llm.complete() raised -- no key configured, or every key exhausted.

    Mirrors llm.py's own stated philosophy: never ship a fabricated answer on
    failure. The citations are still real (built the same way as the success
    path, just without a model narrowing them), only the prose is replaced.
    """
    citations = [_to_citation(c) for c in resolved[:5]]
    basis = _basis(resolved[:5], ok_tool_results, mode)
    flags = unsupported_flags + [
        "llm_unavailable: answer synthesis could not run; showing retrieved evidence only"
    ]
    return {
        "answer": ("I found relevant information but could not generate a written "
                  "answer right now (the language model is unavailable). See the "
                  "citations below for the underlying policy text."),
        "citations": citations,
        "basis": basis,
        "unsupported_flags": flags,
        "confidence": 0.0,
    }


# --------------------------------------------------------------------- public


def synthesize(question: str, chunks: list[dict] | None = None,
              tool_results: list[dict] | None = None,
              mode: Mode = "policy") -> dict:
    """Contract C's seam. Writes the answer and selects its citations.

    The orchestrator decides WHETHER to call this and WHAT evidence to pass; this
    function decides HOW the answer is written and cited (docs/CONTRACTS.md's
    guardrail split). It never raises on a normal failure mode (no evidence, no
    LLM key) -- see `_llm_unavailable_fallback` and the no-evidence branch below --
    matching the "never ship a fabricated answer" convention app.llm.py states.
    """
    chunks = [_as_dict(c) for c in (chunks or [])]
    tool_results = list(tool_results or [])

    if mode == "refusal":
        return {"answer": "I don't have enough grounded information to answer that.",
                "citations": [], "basis": "refused",
                "unsupported_flags": [], "confidence": 0.0}

    enriched: list[dict] = []
    seen: set[tuple[str, str]] = set()
    unresolved = 0
    for c in chunks:
        e = _enrich(c)
        if e is None:
            unresolved += 1
            continue
        key = (e["doc_id"], e.get("section", ""))
        if key in seen:
            continue
        seen.add(key)
        enriched.append(e)

    ok_tool_results = [r for r in tool_results if r.get("ok")]

    unsupported_flags: list[str] = []
    if unresolved:
        unsupported_flags.append(
            f"{unresolved} policy reference(s) could not be resolved to real "
            "corpus sections and were dropped.")
    if mode == "workflow" and not ok_tool_results:
        unsupported_flags.append(
            "workflow question answered with no employee-specific data available.")

    if not enriched and not ok_tool_results:
        return {
            "answer": "I could not find anything in company policy or your records that covers that.",
            "citations": [], "basis": "refused",
            "unsupported_flags": unsupported_flags or ["no_evidence"],
            "confidence": 0.0,
        }

    prompt_chunks = enriched[:MAX_PROMPT_CHUNKS]
    by_doc_id = {}
    for e in prompt_chunks:
        by_doc_id.setdefault(e["doc_id"], e)  # first (best-ranked) wins on a repeat

    evidence_block = _format_evidence(prompt_chunks)
    tool_block = _format_tool_results(ok_tool_results)
    system = (SYSTEM_WORKFLOW if mode == "workflow" else SYSTEM_POLICY) + TRAILER_INSTRUCTION
    prompt = _build_prompt(question, mode, evidence_block, tool_block)

    try:
        raw = complete(prompt, system=system).text
    except (LLMNotConfigured, LLMError):
        return _llm_unavailable_fallback(prompt_chunks, ok_tool_results,
                                         unsupported_flags, mode)

    answer_text, cited_ids, model_flags, trailer_ok = _parse_completion(raw, list(by_doc_id))
    unsupported_flags.extend(model_flags)

    resolved = [by_doc_id[d] for d in cited_ids if d in by_doc_id]
    # An explicit, cleanly parsed, EMPTY cited_doc_ids (trailer_ok and no cited_ids
    # at all) is the model saying "none of the evidence applied" -- a real answer,
    # not a parsing gap. That must be allowed to fall through to basis="refused"
    # below, not be converted into a "policy_rag" answer backed by evidence the
    # model never actually relied on. Every other empty-`resolved` case still
    # recovers by showing top evidence: a missing/malformed META block
    # (trailer_ok is False) tells us nothing about what the model used, and a
    # non-empty cited_doc_ids that named only ids that don't resolve (trailer_ok
    # is True but cited_ids is non-empty) is the model claiming a citation that
    # doesn't check out, not declining to cite -- treated the same as "we don't
    # know what applies", not as "nothing applies".
    declined_explicitly = trailer_ok and not cited_ids
    if not resolved and prompt_chunks and not declined_explicitly:
        resolved = prompt_chunks[:3]
        unsupported_flags.append(
            "model did not name a resolvable citation; showing top evidence instead")

    basis = _basis(resolved, ok_tool_results, mode)
    confidence = _confidence(resolved, ok_tool_results, basis, unsupported_flags)
    citations = [_to_citation(c) for c in resolved]

    return {
        "answer": answer_text,
        "citations": citations,
        "basis": basis,
        "unsupported_flags": unsupported_flags,
        "confidence": confidence,
    }
