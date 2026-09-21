"""The synthetic HR dataset.

Referential integrity matters more here than it looks. The agent answers approval
questions by following manager_id, and a dangling one produces an answer that is
confidently wrong rather than an error -- exactly the failure the brief's
groundedness metric is meant to catch.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from mcp_server import fixtures
from mcp_server.schemas import BenefitsStatusOut, EmployeeProfileOut, PtoBalanceOut

MOCK_DATA = pathlib.Path("mock_data")


def test_the_submission_required_directory_exists():
    """The brief lists mock_data/ as a required directory, by name."""
    assert MOCK_DATA.is_dir()
    for name in ("employees.json", "pto.json", "benefits.json", "tickets.json"):
        assert (MOCK_DATA / name).is_file(), f"{name} missing"


@pytest.mark.parametrize("name", ["employees.json", "pto.json", "benefits.json",
                                  "tickets.json"])
def test_each_file_is_valid_json(name):
    json.loads((MOCK_DATA / name).read_text())


# ------------------------------------------------------- what the spec asks for

def test_eight_employees_three_locations_two_managers():
    """Group_Assignment.md: 8 synthetic employees, 3 locations, 2 managers."""
    employees = fixtures.EMPLOYEES
    assert len(employees) == 8

    locations = {e["location"].split(" - ")[-1] for e in employees.values()}
    assert len(locations) == 3, locations

    managers = {e["manager_id"] for e in employees.values() if e["manager_id"]}
    assert len(managers) == 2, managers


def test_all_three_employment_types_are_present():
    """The escalation and eligibility paths need something real to trigger on."""
    types = {e["employment_type"] for e in fixtures.EMPLOYEES.values()}
    assert types == {"full_time", "part_time", "contractor"}


# ----------------------------------------------------- referential integrity

def test_every_manager_id_resolves_to_a_real_employee():
    for eid, emp in fixtures.EMPLOYEES.items():
        mid = emp["manager_id"]
        if mid is not None:
            assert mid in fixtures.EMPLOYEES, f"{eid} reports to unknown {mid}"


def test_manager_name_matches_the_manager_record():
    """Two copies of one fact drift. The agent quotes the name; it must be right."""
    for eid, emp in fixtures.EMPLOYEES.items():
        mid = emp["manager_id"]
        if mid is not None:
            assert emp["manager_name"] == fixtures.EMPLOYEES[mid]["name"], eid


def test_exactly_one_employee_sits_at_the_top():
    roots = [e for e in fixtures.EMPLOYEES.values() if e["manager_id"] is None]
    assert len(roots) == 1


def test_pto_and_benefits_keys_are_known_employees():
    assert set(fixtures.PTO) <= set(fixtures.EMPLOYEES)
    assert set(fixtures.BENEFITS) <= set(fixtures.EMPLOYEES)


def test_every_employee_has_a_benefits_record():
    """Including the ineligible ones -- 'not eligible' is an answer, silence is a bug."""
    assert set(fixtures.BENEFITS) == set(fixtures.EMPLOYEES)


def test_contractors_have_no_pto_record_at_all():
    """Not a zero-filled one.

    Zero days reads as "you have none left", implying they could earn some. The
    absence is what lets check_pto_balance say contractors do not accrue.
    """
    for eid, emp in fixtures.EMPLOYEES.items():
        if emp["employment_type"] == "contractor":
            assert eid not in fixtures.PTO, f"{eid} is a contractor with a PTO record"


def test_ticket_employee_ids_are_real():
    for ticket in fixtures.TICKETS:
        assert ticket["employee_id"] in fixtures.EMPLOYEES


# --------------------------------------------------------- arithmetic & shape

def test_pto_balances_add_up():
    """accrued - used == available, or the agent quotes a number that is wrong."""
    for eid, pto in fixtures.PTO.items():
        assert pto["accrued_days"] - pto["used_days"] == pytest.approx(
            pto["available_days"]), eid


def test_records_validate_against_contract_a():
    for eid, emp in fixtures.EMPLOYEES.items():
        EmployeeProfileOut.model_validate(emp)
    for eid, pto in fixtures.PTO.items():
        PtoBalanceOut.model_validate({"employee_id": eid, **pto})
    for eid, ben in fixtures.BENEFITS.items():
        BenefitsStatusOut.model_validate({"employee_id": eid, **ben})


# ---------------------------------------------------- the demo needs these

def test_someone_can_take_three_days_and_someone_cannot():
    """Demo Task B is only interesting if the answer is not always yes."""
    balances = [p["available_days"] for p in fixtures.PTO.values()]
    assert any(b >= 3 for b in balances)
    assert any(b < 3 for b in balances)


def test_a_live_waiting_period_exists():
    """Benefits triage needs a part-timer mid-waiting-period, not just eligible/not."""
    assert any(not b["eligible"] and b["waiting_period_days"] > 0
               for b in fixtures.BENEFITS.values())


def test_blackout_dates_exist_for_the_pto_workflow():
    assert any(p["blackout_dates"] for p in fixtures.PTO.values())


# ------------------------------------------- fixture-phase policy consistency

def test_every_advertised_section_is_fetchable():
    """A chunk that names a section get_policy_section cannot return is a trap.

    The agent reads the section id off a search result and asks for it; a miss
    sends it round the tool loop for nothing, burning the step budget and reading
    badly in the demo trace. Rob's real index must hold the same invariant.
    """
    advertised = {(c["doc_id"], c["section"].split()[0])
                  for c in fixtures.POLICY_CHUNKS}
    missing = advertised - set(fixtures.SECTIONS)
    assert not missing, f"advertised but not fetchable: {sorted(missing)}"


def test_the_multi_document_question_is_answerable_from_fixtures():
    """Rubric item 3 needs a question spanning several documents.

    Out-of-state remote work is demo Task A: remote-work plus tax plus security.
    If the fixtures only carried one of those, the demo would lose its point.
    """
    docs = {c["doc_id"] for c in fixtures.POLICY_CHUNKS}
    assert {"remote-work", "multi-state-work-and-tax", "data-security"} <= docs
