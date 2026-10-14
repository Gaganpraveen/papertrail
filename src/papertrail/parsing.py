"""Page-aware PDF extraction with conservative column detection and section boundaries."""

import hashlib
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFEncryptionError, PDFPasswordIncorrect

from papertrail.errors import ParseError
from papertrail.schema import Chunk, ParsedPaper

PARSER_VERSION = "pdfplumber-sections-v5"
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
    """Read repeated central gutters as columns, retaining full-width headers."""
    # Rotated arXiv stamps are margin metadata, not part of the paper's prose.
    page = page.filter(lambda obj: obj.get("upright", True))
    words = page.extract_words(x_tolerance=2, y_tolerance=3)
    rows: list[list] = []
    # Fixed coordinate buckets split near-aligned column baselines when they
    # fall on opposite sides of a bucket boundary. Group by actual distance.
    for word in sorted(words, key=lambda item: item["top"]):
        if not 0.12 * page.height < word["top"] < 0.9 * page.height:
            continue
        if not rows or word["top"] - rows[-1][0]["top"] > 4:
            rows.append([word])
        else:
            rows[-1].append(word)
    wide_rows = 0
    gutter_tops = []
    prose_tops = []
    mid = page.width / 2
    for row in rows:
        row.sort(key=lambda word: word["x0"])
        if row[-1]["x1"] - row[0]["x0"] < page.width * 0.55:
            continue
        wide_rows += 1
        if any(
            a["x1"] < mid < b["x0"] and b["x0"] - a["x1"] > page.width * 0.02
            for a, b in zip(row, row[1:], strict=False)
        ):
            gutter_tops.append(min(word["top"] for word in row))
            sides = (
                [word for word in row if word["x1"] < mid],
                [word for word in row if word["x0"] > mid],
            )
            if all(
                sum(bool(re.fullmatch(r"[A-Za-z]{2,}[.,;:]?", w["text"])) for w in side) >= 4
                for side in sides
            ):
                prose_tops.append(min(word["top"] for word in row))
    columns = wide_rows >= 8 and len(gutter_tops) / wide_rows >= 0.6
    if columns:
        # Preserve a full-width title/figure/caption prefix before the first
        # paired prose line. A figure caption can extend below the top quarter.
        prefix_limit = min(prose_tops) if prose_tops else 0.25 * page.height
        spanning = [
            word["bottom"]
            for word in words
            if word["top"] < prefix_limit and word["x0"] < mid < word["x1"]
        ]
        header_end = 0
        if spanning:
            bottom = max(spanning)
            following = [word["top"] for word in words if word["top"] > bottom + 2]
            header_end = (bottom + min(following)) / 2 if following else bottom + 2
        header = (
            page.crop((0, 0, page.width, header_end)).extract_text(x_tolerance=2) or ""
            if header_end
            else ""
        )
        left = page.crop((0, header_end, mid, page.height)).extract_text(x_tolerance=2) or ""
        right = (
            page.crop((mid, header_end, page.width, page.height)).extract_text(x_tolerance=2) or ""
        )
        return clean_text(header + "\n" + left + "\n" + right), True
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
            if document.doc.encryption is not None:
                raise ParseError(
                    "Encrypted PDFs are not supported, including files that open without a password. Export an unencrypted, text-readable PDF and try again."
                )
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
        if isinstance(exc, (PDFPasswordIncorrect, PDFEncryptionError)) or isinstance(
            exc.__context__, (PDFPasswordIncorrect, PDFEncryptionError)
        ):
            raise ParseError(
                "Encrypted PDFs are not supported. Export an unencrypted, text-readable PDF and try again."
            ) from exc
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
            "No separate Abstract heading detected; opening-page passages are retained for briefing evidence."
        )
    if not any(c.reference for c in chunks):
        warnings.append("No separate References heading detected; section extraction is heuristic.")
    return ParsedPaper(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        pages=len(raw_pages),
        chunks=chunks,
        warnings=warnings,
    )
