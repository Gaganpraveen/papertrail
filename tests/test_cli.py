from typer.testing import CliRunner

from papertrail.cli import app


def test_missing_session_is_actionable(tmp_path):
    result = CliRunner().invoke(app, ["--data-dir", str(tmp_path), "ask", "abcdef123456", "Why?"])
    assert result.exit_code == 1
    assert "was not found" in result.output
    assert "Traceback" not in result.output


def test_sessions_needs_no_model(tmp_path):
    result = CliRunner().invoke(app, ["--data-dir", str(tmp_path), "sessions"])
    assert result.exit_code == 0
    assert "Session" in result.output
