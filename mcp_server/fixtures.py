"""Mock structured data, adapted from Eric's `mock_data/` onto Contract A.

Eric's datasets are the source of truth. They are shaped for an HR system -- lists
keyed by `employee_id`, `full_name`, `home_office_id` pointing at `offices.json` --
while Contract A is shaped for what the agent needs to answer a question. This
module is the seam between the two.

Adapting here rather than changing Contract A is deliberate. Rob and Eric both
build against the tool output; the data source underneath is ours to change. The
one place Contract A did move is `employment_type`, because Eric's data carries the
exempt / non-exempt distinction and that is a real HR fact his policies rely on --
collapsing it to `full_time` would throw away information the corpus references.

POLICY_CHUNKS and SECTIONS are built from `corpus/manifest.json`, so the
fixture-phase retrieval returns real doc_ids and section_ids from real documents.
When Rob's index lands it replaces the ranking, not the identifiers -- which is
what keeps citations valid across the swap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOCK_DATA = ROOT / "mock_data"
CORPUS = ROOT / "corpus"


def _read(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _by_id(rows: list[dict], key: str = "employee_id") -> dict:
    return {row[key]: row for row in rows}


# --------------------------------------------------------------- raw sources

_EMPLOYEE_ROWS: list[dict] = _read(MOCK_DATA / "employees.json")
_PTO_ROWS: list[dict] = _read(MOCK_DATA / "pto_balances.json")
_BENEFITS_ROWS: list[dict] = _read(MOCK_DATA / "benefits_elections.json")
_OFFICE_ROWS: list[dict] = _read(MOCK_DATA / "offices.json")

OFFICES: dict = _by_id(_OFFICE_ROWS, "office_id")
TICKETS: list = _read(MOCK_DATA / "hr_tickets.json")
BLACKOUTS: list = _read(MOCK_DATA / "blackout_periods.json")

#: employee_id -> department, for scoping blackouts.
_DEPARTMENT: dict = {r["employee_id"]: r.get("department") for r in _EMPLOYEE_ROWS}


def _blackouts_for(employee_id: str) -> list[dict]:
    """Only the blackouts that actually bind this employee.

    A department blackout returned to everyone would have the agent warn a Data
    Engineer about a Consulting Delivery client go-live -- confidently wrong, and
    the kind of wrong a grader notices because it contradicts the record.
    """
    department = _DEPARTMENT.get(employee_id)
    return [b for b in BLACKOUTS
            if b.get("scope") == "company_wide"
            or (b.get("scope") == "department" and b.get("department") == department)]


def _location(row: dict) -> str:
    """A human-readable location string.

    Contract A returns one string because that is what an answer quotes. The
    office lookup is done here so the agent never has to join two datasets to say
    where someone works.
    """
    office = OFFICES.get(row.get("home_office_id"))
    if office:
        return f"{office['city']}, {office['state']}"
    state = row.get("home_state")
    return f"Remote - {state}" if state else "Unknown"


def _tenure_months(hire_date: str, as_of: str = "2026-09-21") -> int:
    from datetime import date

    start = date.fromisoformat(hire_date)
    now = date.fromisoformat(as_of)
    return (now.year - start.year) * 12 + (now.month - start.month)


@lru_cache(maxsize=1)
def _employees() -> dict:
    rows = _by_id(_EMPLOYEE_ROWS)
    out = {}
    for eid, row in rows.items():
        manager = rows.get(row.get("manager_id"))
        out[eid] = {
            "employee_id": eid,
            "name": row["full_name"],
            "role": row["job_title"],
            "employment_type": row["employment_type"],
            # Blackout periods are scoped by department before they are returned
            # (see _blackouts_for), so the department was already deciding the
            # answer -- it just was not on the profile, which left the agent with
            # a department-scoped blackout and no way to tell whether it applied.
            # It hedged, correctly and unhelpfully: "because the employee data
            # does not indicate Grace's department, we cannot determine whether
            # the Consulting Delivery blackout applies to her." It does apply;
            # that is why it was in the list at all.
            "department": row["department"],
            "location": _location(row),
            "manager_id": row.get("manager_id"),
            "manager_name": manager["full_name"] if manager else None,
            "hire_date": row["hire_date"],
            "tenure_months": _tenure_months(row["hire_date"]),
        }
    return out


@lru_cache(maxsize=1)
def _pto() -> dict:
    """Contract A's PTO shape, from Eric's richer accrual records.

    Rows where `eligible` is false are dropped entirely rather than zero-filled.
    A contractor with 0 days reads as "none left", implying they could earn some;
    the absence is what lets check_pto_balance say they do not accrue at all.
    """
    out = {}
    for row in _PTO_ROWS:
        if not row.get("eligible", True):
            continue
        out[row["employee_id"]] = {
            "accrued_days": float(row["accrued_ytd_days"]),
            "used_days": float(row["used_ytd_days"]),
            "available_days": float(row["balance_days"]),
            "carryover_days": float(row.get("carryover_days_from_prior_year", 0)),
            "blackout_periods": _blackouts_for(row["employee_id"]),
            "accrual_rate": float(row["accrual_rate_days_per_month"]),
            "notes": row.get("notes") or "",
        }
    return out


@lru_cache(maxsize=1)
def _benefits() -> dict:
    out = {}
    for row in _BENEFITS_ROWS:
        elections = []
        for plan_key, label in (("medical_plan", "Medical"),
                                ("dental_plan", "Dental"),
                                ("vision_plan", "Vision")):
            value = row.get(plan_key)
            if value and value.lower() not in ("none", "waived", "not enrolled"):
                elections.append({"plan": f"{label} — {value}", "tier": "employee",
                                  "status": "active"})
        if row.get("retirement_enrolled"):
            elections.append({
                "plan": f"Retirement {row.get('retirement_contribution_pct', 0)}%",
                "tier": "employee", "status": "active"})

        out[row["employee_id"]] = {
            "elections": elections,
            "eligible": bool(row.get("benefits_eligible")),
            "waiting_period_days": 0 if row.get("waiting_period_met") else 90,
            "eligibility_date": row.get("effective_date"),
        }
    return out


EMPLOYEES: dict = _employees()
PTO: dict = _pto()
BENEFITS: dict = _benefits()


# ------------------------------------------------- fixture-phase retrieval

@lru_cache(maxsize=1)
def _corpus() -> tuple[list[dict], dict]:
    """Build stand-in chunks and section text from Eric's real corpus.

    The doc_ids and section_ids are HIS -- taken from the manifest and the document
    bodies -- so a citation produced today stays valid when Rob's index replaces
    the ranking underneath. Getting that wrong is how citation accuracy goes to
    zero at the end of the week.
    """
    manifest = _read(CORPUS / "manifest.json")
    chunks: list[dict] = []
    sections: dict = {}

    for doc in manifest["documents"]:
        body = (CORPUS / "policies" / doc["filename"]).read_text()
        for section in doc["sections"]:
            sid, heading = section["section_id"], section["heading"]
            text = _section_text(body, sid)
            if not text:
                continue
            sections[(doc["doc_id"], sid)] = text
            chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": f"{sid} {heading}",
                "snippet": text[:320].strip(),
                "score": 0.5,
            })
    return chunks, sections


def _section_text(body: str, section_id: str) -> str:
    """Pull one section out of a document by its id.

    Eric's section ids appear in the headings themselves (`## CONDUCT-6 ...`) in
    markdown and txt, and inside heading tags in HTML. Matching on the id rather
    than the heading text means a reworded heading does not break retrieval.
    """
    import re

    pattern = re.compile(
        rf"(?:^|\n)[#\s]*(?:<h[1-6][^>]*>\s*)?{re.escape(section_id)}\b(.*?)"
        rf"(?=\n[#\s]*(?:<h[1-6][^>]*>\s*)?[A-Z][A-Z0-9-]+-\d+\b|\Z)",
        re.S,
    )
    match = pattern.search(body)
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return re.sub(r"\s+", " ", text).strip()


POLICY_CHUNKS, SECTIONS = _corpus()


# ------------------------------------------------- fixture-phase ranking

#: Most chunks any single document may contribute to one result set.
#: Without this, the document with the most sections wins every query: REMOTE-WORK
#: has nine sections all containing "work" and "remote", so it filled all five
#: slots and a question spanning remote work AND tax law retrieved only the first.
#: The brief requires "at least one complex question requiring retrieval from
#: multiple policy documents" -- that is impossible if top-k is single-document.
#: Rob's real retriever needs this property too, whatever its scoring.
MAX_CHUNKS_PER_DOC = 2

_STOPWORDS = frozenset(
    ("the", "and", "for", "that", "this", "with", "from", "have", "can", "are",
     "what", "when", "does", "do", "i", "my", "me", "a", "an", "is", "it", "to",
     "of", "in", "on", "if", "be", "as", "at", "or", "any")
)


@lru_cache(maxsize=1)
def _document_frequency() -> dict:
    """How many chunks each term appears in. Rare terms are the informative ones."""
    from collections import Counter

    df: Counter = Counter()
    for chunk in POLICY_CHUNKS:
        text = f"{chunk['snippet']} {chunk['title']} {chunk['section']}".lower()
        df.update(set(_terms(text)))
    return dict(df)


def _terms(text: str) -> list[str]:
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 2 and t not in _STOPWORDS]


def _score(chunk: dict, query_terms: set[str]) -> float:
    """Sum of inverse document frequency over the query terms this chunk matches.

    IDF rather than a raw count because "work" appears in almost every policy and
    "nexus" in one. Counting them equally is what let a tax question rank four
    remote-work sections above the tax policy.
    """
    import math

    df = _document_frequency()
    total = max(len(POLICY_CHUNKS), 1)
    text = set(_terms(f"{chunk['snippet']} {chunk['title']} {chunk['section']}"))
    return sum(math.log(total / (1 + df.get(term, 0)))
               for term in query_terms if term in text)


def rank_chunks(chunks: list[dict], query: str, k: int) -> list[dict]:
    """Rank by IDF, then spread the result across documents.

    Two passes: take the best `MAX_CHUNKS_PER_DOC` from each document in score
    order, then backfill from what is left if k is not yet met. A single document
    can still dominate when it genuinely is the only relevant one -- the cap only
    binds when other documents also matched.
    """
    query_terms = set(_terms(query))
    scored = sorted(
        ({**c, "score": round(_score(c, query_terms), 4)} for c in chunks),
        key=lambda c: c["score"],
        reverse=True,
    )

    picked: list[dict] = []
    per_doc: dict = {}
    for chunk in scored:
        if chunk["score"] <= 0:
            continue
        if per_doc.get(chunk["doc_id"], 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[chunk["doc_id"]] = per_doc.get(chunk["doc_id"], 0) + 1
        picked.append(chunk)
        if len(picked) == k:
            return picked

    for chunk in scored:
        if chunk not in picked:
            picked.append(chunk)
            if len(picked) == k:
                break
    return picked[:k]
