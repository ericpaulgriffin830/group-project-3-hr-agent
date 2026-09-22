# Northbridge Analytics Mock Structured Data

Synthetic HR data for the fictional company described in `corpus/`. All
names, emails, IDs, and records are fabricated for this project — no real
person's data is included. Data is deterministic (no randomness) and was
generated as of **2026-09-21**, which the PTO accrual figures in
`pto_balances.json` are calculated against (see `used_ytd_days` /
`accrued_ytd_days` — these are not live-calculated by consuming code,
they are pre-computed snapshots as of that date).

## Files

| File | Format | Purpose |
|---|---|---|
| `employees.json` | JSON | Employee directory: identity, role, manager, employment type, work location, work arrangement |
| `offices.json` | JSON | The 4 Northbridge office locations |
| `pto_balances.json` | JSON | Per-employee PTO accrual/balance, consistent with PTO-2/PTO-4 |
| `benefits_elections.json` | JSON | Per-employee benefits eligibility and elections, consistent with BEN-1/BEN-7 |
| `hr_tickets.json` | JSON | Example HR tickets across several categories from HR-OPS-2 |
| `expense_claims.csv` | CSV | Example expense claims, including one approved-with-cap and one denied example |
| `blackout_periods.json` | JSON | Company-wide and department-scoped PTO blackout windows, consistent with PTO-9 |

## Schemas

### `employees.json`
```
employee_id          string  e.g. "E1005"
full_name            string
work_email           string
job_title            string
department           string
manager_id           string|null  employee_id of manager of record, or null (CEO)
employment_type      enum    full_time_exempt | full_time_non_exempt | part_time | contractor
hire_date            string  YYYY-MM-DD
home_office_id       string  foreign key -> offices.office_id
home_state           string  2-letter US state code
home_country         string  ISO country code (all "US" in this dataset)
work_arrangement     enum    onsite | hybrid | remote   (REMOTE-WORK RW-1)
status               enum    active (only status present in this dataset)
scheduled_hours_per_week  number  present only for part_time employees
```

### `offices.json`
```
office_id    string  e.g. "OFC-CHI"
city         string
state        string
country      string
office_type  enum    HQ | branch
```

### `pto_balances.json`
Mirrors the accrual mechanics in `corpus/policies/02-pto-and-holidays-policy.md`
(PTO-2 accrual rates, PTO-4 carryover cap of 5 days). `balance_days =
carryover_days_from_prior_year + accrued_ytd_days - used_ytd_days`.
Contractors have `eligible: false` and zeroed fields, per PTO-1.
```
employee_id                       string
plan_year                         number
eligible                          boolean
accrual_rate_days_per_month       number   (0 for ineligible)
accrued_ytd_days                  number   accrued since Jan 1 of plan_year through the last completed month
used_ytd_days                     number
carryover_days_from_prior_year    number   capped at 5.0 per PTO-4
balance_days                      number
last_accrual_date                 string|null   YYYY-MM-DD, the most recent month-end accrual
notes                             string|null
```

### `benefits_elections.json`
Mirrors BEN-1 (eligibility by employment type) and BEN-7 (part-time/contractor limits).
```
employee_id                    string
benefits_eligible              boolean
employment_type                string  (denormalized copy from employees.json for convenience)
medical_plan                   string|null   "HDHP+HSA" | "PPO" | null
dental_plan                    string|null
vision_plan                    string|null
retirement_enrolled             boolean
retirement_contribution_pct     number|null
retirement_employer_match_pct   number|null   (4, per BEN-4, when enrolled)
hsa_or_fsa                      string|null   "HSA" | "Healthcare FSA" | null
dependent_care_fsa_enrolled     boolean
waiting_period_met              boolean
effective_date                  string|null   YYYY-MM-DD
notes                           string|null
```

### `hr_tickets.json`
```
ticket_id      string   e.g. "TCK-10021"
employee_id    string
category       string   one of the categories listed in HR-OPS-2
subject        string
status         enum     open | in_review | resolved | closed
created_date   string   YYYY-MM-DD
assigned_to    string   employee_id or role label (e.g. "IT-Helpdesk")
notes          string
```

### `blackout_periods.json`
Mirrors PTO-9 (blackout periods require manager + Department Head approval and
10 business days' notice, rather than the standard PTO-3 process).
```
blackout_id                    string   e.g. "BO-2026-YEAR-END"
scope                          enum     company_wide | department | office
department                     string|null   set when scope == "department"; matches an employees.json department value
start_date, end_date           string   YYYY-MM-DD, inclusive
policy_section_ref             string   always "PTO-9" in this dataset
reason                         string
approval_required              string   "manager_and_department_head" in this dataset
advance_notice_business_days   number
```
Two records are included: a company-wide fiscal year-end close window, and a
Consulting-Delivery-only window tied to a client go-live. A PTO/agent tool
should check a requested date range against both the company-wide record and
any record scoped to the requesting employee's `department` (from
`employees.json`).

### `expense_claims.csv`
Columns: `claim_id, employee_id, date, category, description, amount_usd,
status, policy_section_ref, notes`. `policy_section_ref` points at the
EXPENSE section that governs the claim (e.g. `EXP-2` for a home office
stipend claim), so a `check_policy_compliance`-style tool can join a claim
back to the exact governing policy text.

## Data consistency notes for the RAG/agent build

- **State list is shared.** Every `home_state` in `employees.json` is one
  of Northbridge's Registered States (IL, TX, CO, NC, plus others) as
  defined in `corpus/policies/04-tax-and-work-location-policy.md` (TAX-2).
  Office states (Chicago/IL, Austin/TX, Denver/CO, Raleigh/NC) are all
  Registered States by construction — this was fixed deliberately so the
  corpus and mock data don't contradict each other.
- **Two employees are written to exercise the two required demo tasks:**
  - `E1006` (Priya Nataraj) has an open ticket (`TCK-10021`) requesting
    ~30 business days of remote work from Portugal — a clean "remote work
    eligibility" demo case, since it exceeds the 20-business-day
    international cap in TAX-5 / REMOTE-WORK-4 and should be flagged, not
    silently approved.
  - `E1005` (Marcus Chen) has a PTO balance of 7.0 days as of the data
    snapshot date — enough to approve a straightforward "3 days off next
    week" PTO-request-guidance demo case without hitting an edge case.
- **Intentional edge cases** are included for evaluation purposes:
  `E1011` (contractor) is PTO- and benefits-ineligible; `E1010` (part-time,
  24 hrs/week) has prorated PTO accrual and is benefits-eligible only for
  retirement per BEN-7; `EXP-3005` and `EXP-3007` in the expense CSV are
  denied/partially-denied claims that a compliance-checking tool should
  correctly flag rather than approve; `E1008` (Grace Lin) has a PTO balance
  of **1.5 days**, deliberately below any plausible "a few days off" request,
  so a PTO-request-guidance task actually has to exercise the insufficient-
  balance / refusal path rather than always approving; and `blackout_periods.json`
  gives a "remote work eligibility"-style task a second kind of policy-plus-data
  reasoning to do — e.g. a PTO request from a Consulting Delivery employee
  (`E1006`, `E1007`, `E1008`, or `E1010`) dated Oct 5–16, 2026 should trigger
  the department-scoped blackout in PTO-9, requiring Department Head approval
  rather than a plain manager sign-off.

## Regenerating this data

All files were generated by a single deterministic script
(`build_mock_data.py`, not included in this deliverable folder but
described in `ai-tooling.md`) so that PTO arithmetic can't drift out of
sync with the policy text by hand-editing. If the PTO policy's accrual
rates or carryover cap change, regenerate rather than hand-editing
`pto_balances.json`.
