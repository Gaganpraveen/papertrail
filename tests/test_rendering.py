from papertrail.rendering import export_run, report_html
from papertrail.schema import RunState


def test_export_escapes_untrusted_content(tmp_path, paper, briefing, chunk):
    paper.title = '<script>alert("x")</script>'
    state = RunState(
        id="abcdef123456",
        query="test",
        model="test",
        embedding_model="test",
        paper=paper,
        briefing=briefing,
        status="ready",
    )
    html = report_html(state, [chunk])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "not a live chat" in html
    files = export_run(state, [chunk], tmp_path)
    assert all(file.exists() for file in files.values())
    assert "Limitations" in files["markdown"].read_text()


def test_upload_report_has_honest_metadata_and_local_page_links(tmp_path, paper, briefing, chunk):
    paper.source = "upload"
    paper.document_id = "upload:" + "b" * 64
    paper.source_filename = "research <draft>.pdf"
    paper.title = paper.source_filename
    paper.arxiv_id = paper.url = paper.pdf_url = paper.published = ""
    paper.authors = []
    state = RunState(
        id="abcdef123456",
        query=paper.source_filename,
        model="test",
        embedding_model="test",
        paper=paper,
        briefing=briefing,
        status="ready",
    )
    files = export_run(state, [chunk], tmp_path, pdf_url="/document/abcdef123456/paper.pdf")
    report = files["html"].read_text()
    assert "View on arXiv" not in report
    assert "bibliographic title, authors and publication date have not been extracted" in report
    assert "research &lt;draft&gt;.pdf" in report
    assert "/document/abcdef123456/paper.pdf#page=2" in report
    assert "upload:" + "b" * 64 in report
    offline = report_html(state, [chunk])
    assert 'href="#page=' not in offline
    assert "Page 2" in offline
    assert "Not supplied" in files["markdown"].read_text()
