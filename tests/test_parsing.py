import pytest
from reportlab.pdfgen import canvas

from papertrail.errors import ParseError
from papertrail.parsing import HEADING, chunk_section, parse_pdf


def test_chunks_keep_page_offsets():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_section(text, "1706.03762v7", 3, "Method")
    assert len(chunks) == 3
    assert len({c.id for c in chunks}) == 3
    assert all(c.page == 3 and text[c.start : c.end] == c.text for c in chunks)
    assert chunks[-1].text.endswith("word399")


def test_pdf_sections_and_references(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = canvas.Canvas(str(path))
    y = 780
    for line in [
        "Abstract",
        *(
            "The research method uses attention to compare evidence and evaluate scientific results."
            for _ in range(8)
        ),
        "1 Introduction",
        "This approach evaluates grounded question answering.",
        "References",
        "A reference paper on scientific retrieval and evaluation.",
    ]:
        doc.drawString(50, y, line)
        y -= 20
    doc.save()
    parsed = parse_pdf(path, "1706.03762v7")
    assert any(c.section == "Abstract" for c in parsed.chunks)
    assert any(c.reference for c in parsed.chunks)
    assert parsed.pages == 1


def test_scanned_or_empty_pdf_fails(tmp_path):
    path = tmp_path / "empty.pdf"
    doc = canvas.Canvas(str(path))
    doc.showPage()
    doc.save()
    with pytest.raises(ParseError, match="little readable"):
        parse_pdf(path, "1706.03762v7")


def test_corrupt_pdf_fails(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(ParseError, match="parsing failed"):
        parse_pdf(path, "1706.03762v7")


def test_page_limit_no_silent_truncation(tmp_path):
    path = tmp_path / "large.pdf"
    doc = canvas.Canvas(str(path))
    for _ in range(2):
        doc.showPage()
    doc.save()
    with pytest.raises(ParseError, match="No partial briefing"):
        parse_pdf(path, "1706.03762v7", max_pages=1)


def test_equations_and_table_rows_are_not_section_headings():
    assert not HEADING.fullmatch("1 n 1 n i i")
    assert not HEADING.fullmatch("8 4.88 25.5 80")
    assert HEADING.fullmatch("3.2.2 Multi-Head Attention")
