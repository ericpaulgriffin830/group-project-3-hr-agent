"""Action guardrails. Don't DO what the user didn't confirm.

The split in docs/CONTRACTS.md: Rob owns **evidence** guardrails -- don't assert
what the corpus doesn't support. This file owns the other half -- don't act without
authority, and know when a human should take over.

Three jobs:

**The confirmation gate.** The server mints and verifies the token; it has to, since
it is the thing performing the write. What lives here is the agent-side policy: which
tools are gated, what the user is shown, and the rule that a pending write ends the
turn. Splitting it that way is deliberate -- a gate enforced only in the agent could
be walked around by calling the tool directly, and a gate enforced only in the server
would leave the agent with no idea it should stop and ask.

**Refusal to act.** Some requests must not produce an action even when a tool exists
for it: acting for an employee we cannot identify, or acting on someone else's
record.

**Escalation.** Some topics should reach a person regardless of how good the answer
is. Harassment, discrimination, safety, and anything legal are not questions to
answer well -- they are questions to route. The brief scores "escalation or
clarification accuracy", and the failure that matters is the false negative: quietly
answering a harassment question with a policy citation.

Escalation triggers are deterministic, not model-judged. A model deciding case by
case is a model that will occasionally decide wrong on the one that mattered, and it
is not reproducible across evaluation runs. Deterministic patterns over-trigger
sometimes, which costs a needless handoff -- the cheap direction to be wrong in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: Tools that perform an action. Both are mock -- no ticket system, nothing is sent
#: -- but the brief requires explicit confirmation for exactly these, so the gate is
#: real even though the blast radius is not.
GATED_TOOLS = frozenset(("create_mock_hr_ticket", "draft_hr_email"))

EscalationRoute = Literal["hr_partner", "legal", "safety", "ethics_hotline"]


@dataclass(frozen=True)
class Escalation:
    route: EscalationRoute
    reason: str
    answer_anyway: bool = True

    def as_dict(self) -> dict:
        return {"escalate": True, "route": self.route, "reason": self.reason}


@dataclass(frozen=True)
class ActionDecision:
    """What the agent may do with a tool the model has chosen."""

    allow: bool
    needs_confirmation: bool = False
    refusal: str | None = None


#: (pattern, route, reason). Ordered: the first match wins, so the gravest come
#: first. Word-boundary anchored -- a bare substring makes "class action" match
#: "action" and "assault course" match "assault".
_ESCALATION_PATTERNS: tuple[tuple[str, EscalationRoute, str], ...] = (
    (r"\b(harass(ing|ment|ed)?|hostile work environment)\b", "hr_partner",
     "possible harassment"),
    (r"\b(discriminat(e|ed|ing|ion)|retaliat(e|ed|ion))\b", "hr_partner",
     "possible discrimination or retaliation"),
    (r"\b(assault(ed)?|threaten(ed|ing)?|violence|unsafe|injur(y|ed))\b", "safety",
     "possible safety incident"),
    (r"\b(lawsuit|attorney|lawyer|sue|legal action|subpoena)\b", "legal",
     "legal exposure"),
    (r"\b(fraud|embezzl\w*|bribe\w*|falsif\w*|whistleblow\w*)\b", "ethics_hotline",
     "possible misconduct report"),
    (r"\b(fmla|medical leave|disabilit(y|ies)|accommodation)\b", "hr_partner",
     "protected leave or accommodation"),
)

_COMPILED = tuple((re.compile(p, re.I), route, reason)
                  for p, route, reason in _ESCALATION_PATTERNS)


def classify_escalation(text: str) -> Escalation | None:
    """Should a person see this, regardless of how well we could answer it?

    `answer_anyway` stays True: the useful behaviour is a grounded answer *plus* a
    handoff, not a refusal. Someone asking about accommodation still deserves the
    policy; they also deserve a named human.
    """
    for pattern, route, reason in _COMPILED:
        if pattern.search(text or ""):
            return Escalation(route=route, reason=reason)
    return None


def is_gated(tool: str) -> bool:
    return tool in GATED_TOOLS


def gate_action(tool: str, args: dict, *, employee_id: str | None,
                confirm_token: str | None) -> ActionDecision:
    """Decide whether the agent may perform this tool call.

    Read-only tools pass. Gated tools need an identified subject and a token the
    human supplied; without one, the agent previews and stops.
    """
    if not is_gated(tool):
        return ActionDecision(allow=True)

    subject = args.get("employee_id") or employee_id
    if not subject:
        return ActionDecision(
            allow=False,
            refusal=("I can't file anything without knowing whose record it is. "
                     "Tell me the employee id and I'll prepare it for your review."),
        )

    # Acting on someone else's record is a different act from acting on your own,
    # and the agent is not the right place to decide that it is allowed.
    if employee_id and args.get("employee_id") and args["employee_id"] != employee_id:
        return ActionDecision(
            allow=False,
            refusal=(f"That would act on {args['employee_id']}'s record rather than "
                     f"yours ({employee_id}). An HR partner has to do that."),
        )

    if not confirm_token:
        return ActionDecision(allow=True, needs_confirmation=True)

    return ActionDecision(allow=True)


def confirmation_prompt(tool: str, preview: dict) -> str:
    """What the user is asked before a write happens.

    States the action in plain language and says plainly that nothing has happened
    yet -- a confirmation the reader does not understand is not consent.
    """
    what = {
        "create_mock_hr_ticket": "open an HR ticket",
        "draft_hr_email": "draft a message on your behalf",
    }.get(tool, tool)

    details = ", ".join(f"{k}: {v}" for k, v in preview.items()
                        if k not in ("body",) and v)
    detail_text = f" ({details})" if details else ""
    return (f"This would {what}{detail_text}. Nothing has been created yet — "
            f"confirm and I'll go ahead.")


#: Where each route actually sends someone. Naming a destination is the difference
#: between a handoff and a brush-off.
_ROUTE_CONTACT = {
    "hr_partner": "your HR business partner",
    "legal": "the legal team",
    "safety": "the safety lead, immediately",
    "ethics_hotline": "the confidential ethics hotline",
}


def append_handoff(answer: str, escalation: Escalation) -> str:
    """Put the handoff in the ANSWER, not only in the envelope.

    An escalation the user never reads is metadata. Someone describing harassment
    must see "take this to a person" in the reply itself -- the `escalation` field
    is for the UI and the metrics, and neither of those is the person who asked.
    """
    contact = _ROUTE_CONTACT.get(escalation.route, "an HR partner")
    handoff = (f"\n\nThis should go to {contact} rather than being handled here. "
               f"I've flagged it as {escalation.reason}.")
    return (answer or "").rstrip() + handoff


def refusal_to_act(reason: str) -> dict:
    """A refusal shaped like every other answer, so the UI renders it normally."""
    return {"answer": reason, "citations": [], "answer_basis": "refusal"}
