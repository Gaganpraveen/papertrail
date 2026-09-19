import pdfplumber
import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from papertrail.errors import ParseError
from papertrail.parsing import HEADING, chunk_section, page_text, parse_pdf


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


def test_narrow_gutter_columns_preserve_reading_order_and_spanning_header(tmp_path):
    path = tmp_path / "two-columns.pdf"
    doc = canvas.Canvas(str(path), pagesize=(612, 792))
    doc.setFont("Helvetica", 12)
    doc.drawCentredString(306, 740, "FULLWIDTHHEADER")
    doc.setFont("Helvetica", 10)
    for row in range(20):
        left = f"Left column sentence {row:02d} explains the method."
        right = f"Right column sentence {row:02d} discusses the results."
        # A realistic 16pt gutter was missed by the previous 4.5%-width rule.
        doc.drawString(298 - stringWidth(left, "Helvetica", 10), 630 - row * 16, left)
        doc.drawString(314, 630 - row * 16, right)
    doc.saveState()
    doc.translate(18, 200)
    doc.rotate(90)
    doc.drawString(0, 0, "ROTATEDMARGINSTAMP")
    doc.restoreState()
    doc.save()

    with pdfplumber.open(path) as document:
        text, columns = page_text(document.pages[0])
    assert columns is True
    assert text.splitlines()[0] == "FULLWIDTHHEADER"
    assert text.index("Left column sentence 19") < text.index("Right column sentence 00")
    assert "ROTATEDMARGINSTAMP" not in text
    assert all(f"Left column sentence {row:02d}" in text for row in range(20))
    assert all(f"Right column sentence {row:02d}" in text for row in range(20))


def test_single_column_prose_is_not_split_at_normal_word_spaces(tmp_path):
    path = tmp_path / "one-column.pdf"
    doc = canvas.Canvas(str(path), pagesize=(612, 792))
    doc.setFont("Helvetica", 10)
    for row in range(20):
        doc.drawString(
            50,
            650 - row * 18,
            f"Sentence {row:02d} explains the research method and the evidence supporting the reported results.",
        )
    doc.save()
    with pdfplumber.open(path) as document:
        text, columns = page_text(document.pages[0])
    assert columns is False
    assert all(f"Sentence {row:02d} explains" in text for row in range(20))
