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
