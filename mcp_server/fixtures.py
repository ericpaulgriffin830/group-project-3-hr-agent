"""Canned data for the fixture phase.

Everything here is synthetic. Replaced by mock_data/*.json (Chris) and the real
RAG index (Rob) later this week. The SHAPES here are Contract A and must not
drift — Rob and Eric build against them.
"""

EMPLOYEES = {
    "E-1043": {
        "employee_id": "E-1043", "name": "Dana Whitfield", "role": "Senior Analyst",
        "employment_type": "full_time", "location": "Pittsburgh, PA",
        "manager_id": "E-1002", "manager_name": "Priya Raman",
        "hire_date": "2023-03-13", "tenure_months": 42,
    },
    "E-1077": {
        "employee_id": "E-1077", "name": "Marcus Bell", "role": "Field Technician",
        "employment_type": "part_time", "location": "Columbus, OH",
        "manager_id": "E-1002", "manager_name": "Priya Raman",
        "hire_date": "2026-06-01", "tenure_months": 3,
    },
    "E-1099": {
        "employee_id": "E-1099", "name": "Alex Reyes", "role": "Contract Designer",
        "employment_type": "contractor", "location": "Remote - Austin, TX",
        "manager_id": "E-1010", "manager_name": "Sam Okafor",
        "hire_date": "2026-01-15", "tenure_months": 8,
    },
}

PTO = {
    "E-1043": {"accrued_days": 18.0, "used_days": 11.5, "available_days": 6.5,
               "blackout_dates": ["2026-12-22", "2026-12-23"], "accrual_rate": 1.5},
    "E-1077": {"accrued_days": 4.0, "used_days": 0.0, "available_days": 4.0,
               "blackout_dates": [], "accrual_rate": 0.5},
    # E-1099 (contractor) deliberately has NO entry. A zero-filled record would have
    # the agent answer "0 days available", which implies he could accrue some; the
    # true answer is that contractors do not accrue at all. The absence is what makes
    # check_pto_balance's no_pto_record branch reachable.
}

BENEFITS = {
    "E-1043": {"elections": [{"plan": "PPO Medical", "tier": "employee+spouse", "status": "active"},
                             {"plan": "Dental", "tier": "employee", "status": "active"}],
               "eligible": True, "waiting_period_days": 0, "eligibility_date": "2023-04-01"},
    "E-1077": {"elections": [], "eligible": False, "waiting_period_days": 90,
               "eligibility_date": "2026-08-30"},
    "E-1099": {"elections": [], "eligible": False, "waiting_period_days": 0,
               "eligibility_date": None},
}

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
}
