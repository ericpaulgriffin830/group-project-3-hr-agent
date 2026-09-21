"""Mock structured data, loaded from mock_data/*.json.

Everything here is synthetic -- the Muppet names are deliberate, since the brief
requires mock data be clearly synthetic and nobody mistakes Fozzie Bear for a real
employee record.

The employee/PTO/benefits/ticket data now lives in mock_data/ as JSON, which the
submission requires as its own directory and which lets Eric's UI and the
evaluation harness read it without importing this package. POLICY_CHUNKS and
SECTIONS stay in Python: they are a stand-in for Rob's retrieval index, not mock
structured data, and they disappear when his index lands.

The SHAPES are Contract A and must not drift -- Rob and Eric build against them.
"""

import json
from pathlib import Path

#: The submission requires a mock_data/ directory, and structured data belongs in
#: data files rather than Python -- Eric's UI and the evaluation harness both read
#: these without importing our package.
MOCK_DATA = Path(__file__).resolve().parent.parent / "mock_data"


def _load(name: str, key: str) -> dict | list:
    """Read one mock_data file. Keys starting with '_' are notes, not data."""
    with open(MOCK_DATA / name) as fh:
        return json.load(fh)[key]


EMPLOYEES: dict = _load("employees.json", "employees")
PTO: dict = _load("pto.json", "pto")
BENEFITS: dict = _load("benefits.json", "benefits")
TICKETS: list = _load("tickets.json", "tickets")

# Stand-in for Rob's index. Keyed loosely so fixture search returns something sane.
POLICY_CHUNKS = [
    {"doc_id": "remote-work", "title": "Remote Work Policy", "section": "3.2 Out-of-State Work",
     "snippet": "Employees working outside their registered work state for more than 30 consecutive "
                "days require written manager approval and an HR tax review.", "score": 0.91},
    {"doc_id": "multi-state-work-and-tax", "title": "Multi-State Work and Tax",
     "section": "2.1 Nexus Thresholds",
     "snippet": "Work performed in a non-registered state beyond 30 days may create employer tax "
                "nexus. HR must be notified before travel begins.", "score": 0.88},
    {"doc_id": "data-security", "title": "Data Security Standards", "section": "5.4 Remote Access",
     "snippet": "Company data may only be accessed over managed devices with full-disk encryption "
                "and active VPN when outside the corporate network.", "score": 0.84},
    {"doc_id": "pto-and-leave", "title": "PTO and Leave", "section": "1.3 Requesting Time Off",
     "snippet": "PTO requests of three or more consecutive days require manager approval submitted "
                "at least five business days in advance.", "score": 0.93},
    {"doc_id": "pto-and-leave", "title": "PTO and Leave", "section": "1.7 Blackout Periods",
     "snippet": "PTO is not granted during posted blackout periods except in cases of documented "
                "emergency approved by a department head.", "score": 0.79},
    {"doc_id": "benefits-overview", "title": "Benefits Overview", "section": "4.1 Eligibility",
     "snippet": "Full-time employees are eligible from date of hire. Part-time employees become "
                "eligible after a 90-day waiting period. Contractors are not eligible.", "score": 0.86},
]

SECTIONS = {
    ("remote-work", "3.2"): "Employees working outside their registered work state for more than 30 "
                            "consecutive days require written manager approval and an HR tax review. "
                            "Requests must be submitted at least 14 days before departure.",
    ("pto-and-leave", "1.3"): "PTO requests of three or more consecutive days require manager approval "
                              "submitted at least five business days in advance. Approval is not "
                              "automatic and depends on team coverage.",
    # Every section a chunk advertises must be fetchable. Search results that name a
    # section get_policy_section cannot return send the agent round the loop for
    # nothing -- it burns the step budget and reads badly in a demo trace.
    ("pto-and-leave", "1.7"): "PTO is not granted during posted blackout periods except in cases of "
                              "documented emergency approved by a department head. Blackout periods "
                              "are published at least 60 days in advance.",
    ("multi-state-work-and-tax", "2.1"): "Work performed in a non-registered state beyond 30 consecutive "
                                         "days may create employer tax nexus. HR must be notified before "
                                         "travel begins so payroll withholding can be adjusted.",
    ("data-security", "5.4"): "Company data may only be accessed over managed devices with full-disk "
                              "encryption and an active VPN when outside the corporate network. "
                              "Personal devices require an approved BYOD attestation on file.",
    ("benefits-overview", "4.1"): "Full-time employees are eligible for benefits from date of hire. "
                                  "Part-time employees become eligible after a 90-day waiting period. "
                                  "Contractors are not eligible for company benefits.",
}
