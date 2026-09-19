"""Page-aware PDF extraction with conservative column detection and section boundaries."""

import hashlib
import re
from collections import Counter
from pathlib import Path

import pdfplumber

from papertrail.errors import ParseError
from papertrail.schema import Chunk, ParsedPaper

PARSER_VERSION = "pdfplumber-sections-v3"
HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[A-Z])\.?\s+[A-Z][A-Za-z][A-Za-z ,:/&()\-]{2,90}|Abstract|References|Bibliography|Acknowledg(?:e)?ments|Appendix(?:\s+.*)?)$",
    re.I,
)


def section_label(section: str) -> str:
    return re.sub(r"^(?:\d+(?:\.\d+)*|[A-Z])\.?\s+", "", section, flags=re.I).lower()


def clean_text(value: str) -> str:
    value = value.replace("\x00", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = re.sub(r"(\w)-\n(?=[a-z])", r"\1", value)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()).strip()


def page_text(page) -> tuple[str, bool]:
    """Split only when most long rows show a large central gutter."""
    words = page.extract_words(x_tolerance=2, y_tolerance=3)
    rows: dict[int, list] = {}
    for word in words:
        if 0.12 * page.height < word["top"] < 0.9 * page.height:
            rows.setdefault(round(word["top"] / 4), []).append(word)
    wide_rows = 0
    gutter_rows = 0
    mid = page.width / 2
    for row in rows.values():
        row.sort(key=lambda word: word["x0"])
        if row[-1]["x1"] - row[0]["x0"] < page.width * 0.55:
            continue
        wide_rows += 1
        if any(
            a["x1"] < mid < b["x0"] and b["x0"] - a["x1"] > page.width * 0.045
            for a, b in zip(row, row[1:], strict=False)
        ):
            gutter_rows += 1
    columns = wide_rows >= 10 and gutter_rows / wide_rows >= 0.6
    if columns:
        left = page.crop((0, 0, mid, page.height)).extract_text(x_tolerance=2) or ""
        right = page.crop((mid, 0, page.width, page.height)).extract_text(x_tolerance=2) or ""
        return clean_text(left + "\n" + right), True
    return clean_text(page.extract_text(x_tolerance=2) or ""), False


def chunk_section(
    text: str, paper_id: str, page: int, section: str, size: int = 180, overlap: int = 30
) -> list[Chunk]:
    words = list(re.finditer(r"\S+", text))
    result = []
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        if end - start < 12 and result:
            break
        if not words:
            break
        a, b = words[start].start(), words[end - 1].end()
        passage = text[a:b]
        identity = f"{paper_id}|{page}|{section}|{a}|{passage}"
        result.append(
            Chunk(
                id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                paper_id=paper_id,
                page=page,
                section=section,
                text=passage,
                start=a,
                end=b,
                reference=section_label(section).startswith(("references", "bibliography")),
            )
        )
        if end == len(words):
            break
    return result


def parse_pdf(path: Path, paper_id: str, max_pages: int = 100) -> ParsedPaper:
    warnings = []
    raw_pages = []
    try:
        with pdfplumber.open(path) as document:
            if not document.pages:
                raise ParseError("The PDF contains no pages.")
            if len(document.pages) > max_pages:
                raise ParseError(
                    f"This PDF has {len(document.pages)} pages; the limit is {max_pages}. No partial briefing was generated."
                )
            for number, page in enumerate(document.pages, 1):
                text, columns = page_text(page)
                raw_pages.append(text)
                if columns:
                    warnings.append(
                        f"Page {number}: two-column reading order inferred. Complex tables or spanning figures may need manual inspection."
                    )
                if len(text.split()) < 25:
                    warnings.append(
                        f"Page {number}: little extractable text; images and scanned content are not OCR-processed."
                    )
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            "PDF parsing failed. The file may be damaged or encrypted; no briefing was generated."
        ) from exc
    if sum(len(p.split()) for p in raw_pages) < 80:
        raise ParseError(
            "The PDF has too little readable text (possibly scanned). OCR is not supported; refusing to summarize unread content."
        )

    # Remove recurring margin lines, not arbitrary recurring body text.
    margins = Counter()
    for text in raw_pages:
        lines = text.splitlines()
        margins.update(set(lines[:2] + lines[-2:]))
    repeated = {x for x, count in margins.items() if count >= max(3, len(raw_pages) * 0.5)}
    chunks = []
    section = "Front matter"
    for number, text in enumerate(raw_pages, 1):
        buffer: list[str] = []
        for line in text.splitlines():
            if line in repeated or re.fullmatch(r"\d{1,3}", line) or re.match(r"arXiv:\d", line):
                continue
            is_heading = (
                HEADING.fullmatch(line)
                and len(line.split()) <= 13
                and not line.endswith((".", ",", ";"))
            )
            if is_heading:
                if buffer:
                    chunks.extend(chunk_section("\n".join(buffer), paper_id, number, section))
                section, buffer = line, []
            else:
                buffer.append(line)
        if buffer:
            chunks.extend(chunk_section("\n".join(buffer), paper_id, number, section))
    if not any(section_label(c.section) == "abstract" for c in chunks):
        warnings.append(
            "No separate Abstract heading detected; the official API abstract remains available in metadata."
        )
    if not any(c.reference for c in chunks):
        warnings.append("No separate References heading detected; section extraction is heuristic.")
    return ParsedPaper(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        pages=len(raw_pages),
        chunks=chunks,
        warnings=warnings,
    )
