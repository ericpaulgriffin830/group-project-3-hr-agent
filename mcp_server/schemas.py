"""Contract A, as code.

docs/CONTRACTS.md describes the eight tools in prose. This file is the executable
copy of that description, and `tests/test_schemas.py` asserts the two agree by
validating what the real tools actually return.

That pairing is the whole point. A contract nobody can run drifts silently -- the
prose says one thing, the server returns another, and the first person to find out
is whoever is debugging a citation at 11pm on 9/30. Here, drift fails a test.

**Outputs are `extra="forbid"`.** A tool that starts returning an extra field is
changing the contract, and change control says that goes through a PR and a Teams
note. Silently tolerating the new field defeats that. Inputs are forbid-extra too,
so a caller passing a misspelled argument gets told rather than ignored.

These models are deliberately NOT wired into the tool signatures in server.py. Tools
must never raise across the MCP boundary -- they return an error payload instead --
and annotating a tool `-> EmployeeProfile` would make a validation failure raise
exactly where it is least recoverable. Validation belongs in the tests and at the
orchestrator's edge, not inside the tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed sets. Widening one of these IS a contract change.
# Widened 2026-09-21 to match mock_data/employees.json. Eric's data carries the
# exempt / non-exempt distinction and his policies rely on it -- overtime and leave
# eligibility both turn on it -- so collapsing both to "full_time" would throw away
# a fact the corpus references. The data is the source of truth; the contract follows.
EmploymentType = Literal["full_time_exempt", "full_time_non_exempt",
                         "part_time", "contractor"]
Verdict = Literal["compliant", "non_compliant", "conditional", "insufficient_evidence"]
RecipientRole = Literal["manager", "hr_partner", "benefits"]
RetrievalMode = Literal["hybrid", "vector_only", "keyword_only", "fixture"]

ErrorCode = Literal[
    "employee_not_found",
    "no_pto_record",
    "section_not_found",
    "no_matching_documents",
    "empty_query",
    "empty_scenario",
    "missing_fields",
    "invalid_confirm_token",
]


class Strict(BaseModel):
    """Base for every model here: unknown fields are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------- error shape


class ToolError(Strict):
    """What every tool returns instead of raising.

    The contextual fields are optional because only some errors carry them, but
    they are declared rather than free-form: an error that invents a new field is
    drift like any other.
    """

    error: ErrorCode
    message: str
    employee_id: str | None = None
    doc_id: str | None = None
    section_id: str | None = None


# ------------------------------------------------------------- shared pieces


class PolicyChunk(Strict):
    """One retrieved passage.

    Citation accuracy is capped by what this carries -- an answer can only cite as
    precisely as the chunk that backs it. Rob's real index fills the same shape.
    """

    doc_id: str
    title: str
    section: str
    snippet: str
    score: float


class PolicyRef(Strict):
    doc_id: str
    section: str


class BenefitElection(Strict):
    plan: str
    tier: str
    status: str


class EmailDraft(Strict):
    to: str
    subject: str
    body: str


# ------------------------------------------------------------- 1. RAG search


class SearchPolicyDocumentsIn(Strict):
    query: str
    k: int = 5
    doc_filter: list[str] | None = None


class SearchPolicyDocumentsOut(Strict):
    chunks: list[PolicyChunk]
    retrieval_mode: RetrievalMode


# -------------------------------------------------------------- 2. RAG fetch


class GetPolicySectionIn(Strict):
    doc_id: str
    section_id: str


class GetPolicySectionOut(Strict):
    doc_id: str
    title: str
    section: str
    text: str


# ---------------------------------------------------------- 3. employee data


class LookupEmployeeProfileIn(Strict):
    employee_id: str


class EmployeeProfileOut(Strict):
    employee_id: str
    name: str
    role: str
    employment_type: EmploymentType
    location: str
    # None at the top of the org. A VP with no manager is the honest model; the
    # alternative is a circular reporting line invented to satisfy a type.
    manager_id: str | None = None
    manager_name: str | None = None
    hire_date: str
    tenure_months: int


# --------------------------------------------------------------- 4. PTO data


class CheckPtoBalanceIn(Strict):
    employee_id: str


class PtoBalanceOut(Strict):
    employee_id: str
    accrued_days: float
    used_days: float
    available_days: float
    blackout_dates: list[str]
    accrual_rate: float
    # From Eric's accrual records. `notes` carries the human caveat his data
    # already wrote down -- pro-rated accrual, a leave hold -- which an answer
    # about someone's balance usually needs.
    carryover_days: float = 0.0
    notes: str = ""


# ---------------------------------------------------------- 5. benefits data


class LookupBenefitsStatusIn(Strict):
    employee_id: str


class BenefitsStatusOut(Strict):
    employee_id: str
    elections: list[BenefitElection]
    eligible: bool
    waiting_period_days: int
    # None for a contractor, who never becomes eligible -- distinct from a date
    # that has not arrived yet.
    eligibility_date: str | None = None


# ----------------------------------------------------------- 6. compliance


class CheckPolicyComplianceIn(Strict):
    scenario: str
    employee_id: str
    policy_refs: list[str] | None = None


class PolicyComplianceOut(Strict):
    verdict: Verdict
    conditions: list[str]
    policy_refs: list[PolicyRef]
    rationale: str


# -------------------------------------------- 7 & 8. confirmation-gated writes


class ConfirmationRequired(Strict):
    """Phase one of every write. Nothing has happened yet.

    The token is bound to the arguments, so it cannot be replayed against a
    different action than the one previewed here.
    """

    requires_confirmation: Literal[True]
    preview: dict
    confirm_token: str
    message: str


class CreateMockHrTicketIn(Strict):
    employee_id: str
    category: str
    summary: str
    confirm_token: str | None = None


class TicketCreatedOut(Strict):
    ticket_id: str
    status: Literal["created"]
    created_at: str


class DraftHrEmailIn(Strict):
    employee_id: str
    recipient_role: RecipientRole
    intent: str
    context: str
    confirm_token: str | None = None


class DraftHrEmailOut(Strict):
    draft: EmailDraft
    # Never True. This system has no send capability and must not grow one.
    sent: Literal[False] = False


# --------------------------------------------------------------- the registry

#: tool name -> (input model, success output models). A tool may have more than one
#: success shape: the write tools return a preview first and a result second.
TOOL_SCHEMAS: dict[str, tuple[type[Strict], tuple[type[Strict], ...]]] = {
    "search_policy_documents": (SearchPolicyDocumentsIn, (SearchPolicyDocumentsOut,)),
    "get_policy_section": (GetPolicySectionIn, (GetPolicySectionOut,)),
    "lookup_employee_profile": (LookupEmployeeProfileIn, (EmployeeProfileOut,)),
    "check_pto_balance": (CheckPtoBalanceIn, (PtoBalanceOut,)),
    "lookup_benefits_status": (LookupBenefitsStatusIn, (BenefitsStatusOut,)),
    "check_policy_compliance": (CheckPolicyComplianceIn, (PolicyComplianceOut,)),
    "create_mock_hr_ticket": (
        CreateMockHrTicketIn,
        (ConfirmationRequired, TicketCreatedOut),
    ),
    "draft_hr_email": (DraftHrEmailIn, (ConfirmationRequired, DraftHrEmailOut)),
}


def validate_output(tool: str, payload: dict) -> Strict:
    """Parse a tool's raw payload into whichever contract shape it matches.

    Errors are checked first: `{"error": ...}` is a valid response from every tool,
    and trying the success models on one produces a confusing failure about missing
    fields when the real answer is simply that the tool declined.

    Raises ValueError if the payload matches nothing -- which means the server and
    docs/CONTRACTS.md have drifted apart, and that is worth failing loudly over.
    """
    if tool not in TOOL_SCHEMAS:
        raise ValueError(f"{tool!r} is not one of the eight tools in Contract A.")

    if "error" in payload:
        return ToolError.model_validate(payload)

    _, out_models = TOOL_SCHEMAS[tool]
    failures: list[str] = []
    for model in out_models:
        try:
            return model.model_validate(payload)
        except Exception as exc:  # pydantic.ValidationError, kept broad on purpose
            failures.append(f"{model.__name__}: {exc}")

    raise ValueError(
        f"{tool!r} returned a payload matching no shape in Contract A.\n"
        + "\n".join(failures)
    )
