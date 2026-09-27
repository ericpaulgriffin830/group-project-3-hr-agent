"""Corpus ingestion: turn corpus/policies/* into structured sections.

Format is dispatched per file, matching `corpus/manifest.json`'s `format` field --
markdown, html, and txt each use a different heading convention (see
corpus/README.md, "Conventions for ingestion"). This module's only job is turning
those three encodings into one common shape: an ordered list of RawSection per
document. Chunking (chunk.py) and embedding (embed.py) come after.

Metadata (doc_id, title, version, owner, ...) comes from corpus/manifest.json,
never re-derived by scraping frontmatter/meta-tags/header blocks out of the
documents themselves. corpus/README.md is explicit that the manifest is
authoritative for exactly this reason -- so this module only reads the manifest
for metadata and only reads the document bodies for section text.

Inline emphasis is kept as markdown (**bold**, *italic*) for both the markdown
and HTML sources, never stripped to plain text. It goes into the LLM's context in
answer.py, and "approval is **required**" carries information a flattened
"approval is required" does not.

Section coverage is validated against the manifest on every parse: a document
whose parsed section IDs don't match its manifest entry, in order, fails loudly
here (IngestError) rather than silently losing a section that a citation later
points at and finds nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "corpus"
POLICIES = CORPUS / "policies"
MANIFEST = CORPUS / "manifest.json"


class IngestError(RuntimeError):
    """A document's parsed sections don't match what corpus/manifest.json declares.

    Not something to catch and continue past -- a doc_id/section_id the manifest
    promises but ingestion didn't produce is a citation the system will later
    fail to back, and that failure should happen at build time, not in the demo.
    """


@dataclass(frozen=True)
class RawSection:
    """One section, before chunking. `text` keeps inline emphasis as markdown."""

    doc_id: str
    section_id: str
    heading: str
    text: str
    order: int


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    title: str
    filename: str
    format: str
    version: str
    effective_date: str
    owner: str
    applies_to: str | None
    source_path: str
    sections: list[RawSection] = field(default_factory=list)


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# --------------------------------------------------------------- markdown

# `## PTO-3 Requesting and Approving PTO` -- ATX header, section id then heading.
_MD_HEADER = re.compile(r"^##\s+([A-Z][A-Z0-9-]*-\d+)\s+(.+?)\s*$", re.M)


def _parse_markdown_sections(body: str) -> list[tuple[str, str, str]]:
    """Split on `## SECTION-ID Heading` lines.

    Content before the first match (YAML frontmatter, the H1 title) is simply
    never captured by any slice -- there is nothing to strip explicitly, since no
    section owns it and no citation ever needs it.
    """
    matches = list(_MD_HEADER.finditer(body))
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        out.append((m.group(1), m.group(2).strip(), text))
    return out


# --------------------------------------------------------------- html

# The <h2> text itself is "RW-3 Temporary Remote Work From a Non-Home Location...".
_HTML_HEADING = re.compile(r"^([A-Z][A-Z0-9-]*-\d+)\s+(.+?)\s*$")


def _parse_html_sections(html: str) -> list[tuple[str, str, str]]:
    """Split on <h2> tags; convert each section's body to markdown-style text.

    <strong>/<em> become **bold**/*italic* via markdownify -- the same convention
    the markdown sources already use in prose, so answer.py's prompt builder never
    needs to know which source format a chunk came from.

    Sections here are flat: <h2> and <p> siblings directly under <body>, with no
    wrapping <div> or <section> tags (confirmed against every HTML file in the
    corpus), so walking next_sibling from each heading to the next is correct and
    does not need a general-purpose HTML-tree walk.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all("h2")
    out = []
    for h in headings:
        raw = h.get_text(" ", strip=True)
        m = _HTML_HEADING.match(raw)
        if not m:
            raise IngestError(f"HTML <h2> does not match SECTION-ID pattern: {raw!r}")
        section_id, heading = m.group(1), m.group(2)

        parts: list[str] = []
        node = h.next_sibling
        while node is not None and getattr(node, "name", None) != "h2":
            parts.append(str(node))
            node = node.next_sibling

        text = markdownify("".join(parts), heading_style="ATX").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank-line runs from block spacing
        out.append((section_id, heading, text))
    return out


# --------------------------------------------------------------- txt

# corpus/policies/09-leave-of-absence-policy.txt wraps each header in a pair of
# "====...====" rule lines: RULE \n SECTION-ID Heading \n RULE \n body...
_TXT_HEADER_BLOCK = re.compile(
    r"={10,}\n([A-Z][A-Z0-9-]*-\d+)\s+(.+?)\n={10,}\n", re.M
)


def _parse_txt_sections(body: str) -> list[tuple[str, str, str]]:
    """Split on the rule-wrapped header blocks; text runs to the next block."""
    matches = list(_TXT_HEADER_BLOCK.finditer(body))
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        out.append((m.group(1), m.group(2).strip(), text))
    return out


_PARSERS = {
    "markdown": _parse_markdown_sections,
    "html": _parse_html_sections,
    "txt": _parse_txt_sections,
}


# --------------------------------------------------------------- dispatch


def parse_document(doc_meta: dict) -> ParsedDocument:
    """Parse one manifest entry's file, validating section coverage against it."""
    fmt = doc_meta["format"]
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise IngestError(f"{doc_meta['doc_id']}: unsupported corpus format {fmt!r}")

    path = POLICIES / doc_meta["filename"]
    body = path.read_text(encoding="utf-8")
    parsed = parser(body)

    expected_ids = [s["section_id"] for s in doc_meta["sections"]]
    got_ids = [sid for sid, _, _ in parsed]
    if got_ids != expected_ids:
        raise IngestError(
            f"{doc_meta['doc_id']}: parsed section ids {got_ids} do not match "
            f"manifest section ids {expected_ids}"
        )

    sections = [
        RawSection(doc_id=doc_meta["doc_id"], section_id=sid, heading=heading,
                   text=text, order=i)
        for i, (sid, heading, text) in enumerate(parsed)
    ]
    return ParsedDocument(
        doc_id=doc_meta["doc_id"],
        title=doc_meta["title"],
        filename=doc_meta["filename"],
        format=fmt,
        version=doc_meta["version"],
        effective_date=doc_meta["effective_date"],
        owner=doc_meta["owner"],
        applies_to=doc_meta.get("applies_to"),
        # .as_posix(), not str(): relative_to() keeps the host OS's separator,
        # so str() gives "corpus\\policies\\x.md" on Windows. This path is a
        # repo-relative identifier used elsewhere (citations, tests asserting
        # its shape), not a filesystem path this process reopens -- it has to
        # be the same string on every contributor's machine and in CI's three
        # OSes, not just usable on the one that produced it.
        source_path=path.relative_to(ROOT).as_posix(),
        sections=sections,
    )


def load_corpus() -> list[ParsedDocument]:
    """Parse every document the manifest lists, in manifest order."""
    manifest = load_manifest()
    return [parse_document(doc) for doc in manifest["documents"]]
