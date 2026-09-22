"""MCP server exposing HR tools to the agent.

Contract A lives in docs/CONTRACTS.md. The schemas here ARE that contract --
Rob backs the two RAG tools, Eric's CI asserts discovery against this server.

PHASE: fixtures. Every tool returns canned data from fixtures.py so Rob and Eric
can build against real tool calls before the real implementations land.

Two rules that hold in every phase:
  1. Tools NEVER raise to the agent. They return {"error", "message"} so the
     orchestrator can degrade gracefully and the trace records what happened.
  2. Write tools NEVER act without a confirm_token. First call returns a preview
     plus a token; only a second call carrying that token performs the action.

Transport is chosen by MCP_TRANSPORT (stdio | streamable-http).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer

from . import fixtures

mcp = MCPServer("hr-tools")

CONFIRM_SALT = "hr-agent-confirm-v1"


def _token(action: str, **parts: object) -> str:
    """Deterministic confirmation token: same request -> same token."""
    raw = f"{CONFIRM_SALT}|{action}|" + "|".join(f"{k}={parts[k]}" for k in sorted(parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _err(code: str, message: str, **extra: object) -> dict:
    return {"error": code, "message": message, **extra}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- RAG tools

@mcp.tool()
def search_policy_documents(query: str, k: int = 5, doc_filter: list[str] | None = None) -> dict:
    """Search company HR policy documents for passages relevant to a question.

    Use this for any question about what company policy says. Returns passages
    with the document and section they came from, so the answer can cite them.

    doc_filter, when given, must use these exact document ids:
    BENEFITS, CONDUCT, EQUIPMENT, EXPENSE, HANDBOOK-OVERVIEW, HR-OPS, INFOSEC,
    LEAVE, ONBOARDING, PTO-HOLIDAYS, REMOTE-WORK, TAX-LOCATION.
    Omit doc_filter to search everything, which is usually what you want.
    """
    if not query or not query.strip():
        return _err("empty_query", "A non-empty query is required.")

    chunks = fixtures.POLICY_CHUNKS
    if doc_filter:
        chunks = [c for c in chunks if c["doc_id"] in doc_filter]
        if not chunks:
            # Name the valid ids so a wrong guess is corrected in one step rather
            # than re-guessed. Each wasted round trip spends the tool budget and
            # shows up as noise in the demo trace.
            available = sorted({c["doc_id"] for c in fixtures.POLICY_CHUNKS})
            return _err("no_matching_documents",
                        f"No indexed documents match filter {doc_filter}. "
                        f"Valid doc_ids: {', '.join(available)}.")

    terms = {t for t in query.lower().split() if len(t) > 3}
    ranked = sorted(
        chunks,
        key=lambda c: (
            sum(t in (c["snippet"] + c["title"] + c["section"]).lower() for t in terms),
            c["score"],
        ),
        reverse=True,
    )
    return {"chunks": ranked[: max(1, k)], "retrieval_mode": "fixture"}


@mcp.tool()
def get_policy_section(doc_id: str, section_id: str) -> dict:
    """Retrieve the full text of one specific policy section by document and section ID.

    Use after search_policy_documents when a snippet is not enough and the full
    section text is needed to answer precisely.
    """
    text = fixtures.SECTIONS.get((doc_id, section_id))
    if text is None:
        return _err("section_not_found",
                    f"No section '{section_id}' in document '{doc_id}'.",
                    doc_id=doc_id, section_id=section_id)
    title = next((c["title"] for c in fixtures.POLICY_CHUNKS if c["doc_id"] == doc_id), doc_id)
    return {"doc_id": doc_id, "title": title, "section": section_id, "text": text}


# ---------------------------------------------------------- mock-data tools

@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict:
    """Look up an employee's role, employment type, location, manager and tenure.

    Use this first for any question whose answer depends on who is asking --
    eligibility, approval chains, or location-specific rules.
    """
    profile = fixtures.EMPLOYEES.get(employee_id)
    if profile is None:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)
    return dict(profile)


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict:
    """Check an employee's accrued, used and available PTO days, plus blackout dates.

    Use for any question about taking time off.
    """
    if employee_id not in fixtures.EMPLOYEES:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)
    pto = fixtures.PTO.get(employee_id)
    if pto is None:
        return _err("no_pto_record",
                    f"Employee '{employee_id}' has no PTO record (contractors do not accrue).",
                    employee_id=employee_id)
    return {"employee_id": employee_id, **pto}


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict:
    """Look up an employee's benefits elections, eligibility and waiting period."""
    if employee_id not in fixtures.EMPLOYEES:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)
    return {"employee_id": employee_id, **fixtures.BENEFITS[employee_id]}


# ------------------------------------------------------------ composite tool

@mcp.tool()
def check_policy_compliance(scenario: str, employee_id: str,
                            policy_refs: list[str] | None = None) -> dict:
    """Evaluate whether a described scenario complies with policy for this employee.

    Returns a verdict with the conditions that apply and the policy sections relied
    on. Returns 'insufficient_evidence' rather than guessing when policy is unclear.
    """
    profile = fixtures.EMPLOYEES.get(employee_id)
    if profile is None:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)
    if not scenario or not scenario.strip():
        return _err("empty_scenario", "A scenario description is required.")

    s = scenario.lower()
    if "remote" in s or "another state" in s or "work from" in s:
        return {
            "verdict": "conditional",
            "conditions": [
                "Written manager approval required before departure.",
                "HR tax review required for stays over 30 consecutive days.",
                "Managed device with full-disk encryption and active VPN required.",
            ],
            "policy_refs": [
                {"doc_id": "remote-work", "section": "3.2"},
                {"doc_id": "multi-state-work-and-tax", "section": "2.1"},
                {"doc_id": "data-security", "section": "5.4"},
            ],
            "rationale": "Out-of-state work beyond 30 days is permitted with approval and tax review.",
        }
    if "pto" in s or "time off" in s or "vacation" in s:
        pto = fixtures.PTO.get(employee_id, {})
        available = pto.get("available_days", 0.0)
        return {
            "verdict": "conditional" if available > 0 else "non_compliant",
            "conditions": (
                ["Manager approval required for 3+ consecutive days.",
                 "Request at least five business days in advance."]
                if available > 0 else
                [f"Employee has {available} PTO days available."]
            ),
            "policy_refs": [{"doc_id": "pto-and-leave", "section": "1.3"}],
            "rationale": f"Employee has {available} days available; 3+ day requests need approval.",
        }
    return {
        "verdict": "insufficient_evidence",
        "conditions": [],
        "policy_refs": [],
        "rationale": "No policy section clearly governs this scenario. Escalate to HR.",
    }


# ------------------------------------------- mock writes (confirmation-gated)

@mcp.tool()
def create_mock_hr_ticket(employee_id: str, category: str, summary: str,
                          confirm_token: str | None = None) -> dict:
    """Create a mock HR ticket. REQUIRES CONFIRMATION.

    Call once without confirm_token to get a preview and a token. Call again with
    that token to create the ticket. Nothing is created on the first call.
    """
    if employee_id not in fixtures.EMPLOYEES:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)
    if not category or not summary:
        return _err("missing_fields", "Both 'category' and 'summary' are required.")

    expected = _token("create_mock_hr_ticket", employee_id=employee_id,
                      category=category, summary=summary)
    if confirm_token is None:
        return {
            "requires_confirmation": True,
            "preview": {"employee_id": employee_id, "category": category, "summary": summary},
            "confirm_token": expected,
            "message": "No ticket created. Re-call with confirm_token to create it.",
        }
    if confirm_token != expected:
        return _err("invalid_confirm_token",
                    "Confirmation token does not match this request. Nothing was created.")

    # Derived from the confirm token's digest, NOT hash() -- str hashing is salted
    # per process (PYTHONHASHSEED), so hash() would hand the same request a different
    # ticket id on every run and make evaluation reruns incomparable.
    ticket_id = f"HR-{int(expected, 16) % 90000 + 10000}"
    return {"ticket_id": ticket_id, "status": "created", "created_at": _now()}


@mcp.tool()
def draft_hr_email(employee_id: str, recipient_role: str, intent: str, context: str,
                   confirm_token: str | None = None) -> dict:
    """Draft an HR email on the employee's behalf. REQUIRES CONFIRMATION. NEVER SENDS.

    Call once without confirm_token for a preview and token; call again with the
    token to produce the final draft. 'sent' is always false -- this system has no
    send capability.
    """
    if employee_id not in fixtures.EMPLOYEES:
        return _err("employee_not_found",
                    f"No employee record for '{employee_id}'.", employee_id=employee_id)

    profile = fixtures.EMPLOYEES[employee_id]
    expected = _token("draft_hr_email", employee_id=employee_id,
                      recipient_role=recipient_role, intent=intent, context=context)
    if confirm_token is None:
        return {
            "requires_confirmation": True,
            "preview": {"to": recipient_role, "intent": intent},
            "confirm_token": expected,
            "message": "No draft produced. Re-call with confirm_token to draft it.",
        }
    if confirm_token != expected:
        return _err("invalid_confirm_token",
                    "Confirmation token does not match this request. Nothing was drafted.")

    to = profile["manager_name"] if recipient_role == "manager" else "HR Business Partner"
    return {
        "draft": {
            "to": to,
            "subject": f"{intent} - {profile['name']}",
            "body": (f"Hi {to.split()[0]},\n\n{context}\n\n"
                     f"Thanks,\n{profile['name']}"),
        },
        "sent": False,
    }


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport not in {"stdio", "streamable-http"}:
        raise SystemExit(f"MCP_TRANSPORT must be 'stdio' or 'streamable-http', got {transport!r}")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
