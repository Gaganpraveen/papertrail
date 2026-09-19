from unittest.mock import Mock

import pytest

from papertrail.config import Settings
from papertrail.errors import PaperTrailError, SourceError
from papertrail.graph import Agent
from papertrail.schema import Node, ParsedPaper, RunState
from papertrail.storage import Store


def test_failure_resume_does_not_repeat_successful_nodes(
    tmp_path, paper, chunk, briefing, monkeypatch
):
    source = Mock()
    source.search.return_value = ([paper], [])
    source.download.side_effect = [SourceError("network down"), "test-sha"]
    llm = Mock(model="test-model")
    llm.briefing.return_value = briefing
    index = Mock()
    index.build.return_value = "collection"
    index.search.return_value = []
    settings = Settings(data_dir=tmp_path, model="test-model")
    agent = Agent(settings, source=source, index=index, llm=llm)
    parsed = ParsedPaper(sha256="test-sha", pages=1, chunks=[chunk])
    monkeypatch.setattr("papertrail.graph.parse_pdf", lambda *args: parsed)
    monkeypatch.setattr(agent, "briefing_evidence", lambda state: [chunk])
    with pytest.raises(PaperTrailError, match="network down"):
        agent.new("1706.03762v7")
    failed = agent.store.recent()[0]
    assert failed.node == Node.FETCH
    assert failed.status == "failed"
    result = agent.resume(failed.id)
    assert result.status == "ready"
    assert result.node == Node.READY
    assert source.search.call_count == 1
    assert source.download.call_count == 2
    assert llm.briefing.call_count == 1


def test_saved_state_survives_reopening(tmp_path):
    store = Store(tmp_path)
    state = RunState(id="abcdef123456", query="attention", model="local", embedding_model="bge")
    store.save(state)
    assert Store(tmp_path).load(state.id) == state


def test_briefing_retains_opening_context_when_abstract_heading_is_absent(chunk):
    first = chunk.model_copy(update={"id": "a" * 16, "page": 1, "section": "Front matter"})
    second = chunk.model_copy(update={"id": "b" * 16, "page": 1, "section": "Front matter"})
    later = chunk.model_copy(update={"id": "c" * 16, "page": 3, "section": "Method"})
    agent = Agent.__new__(Agent)
    agent.parsed = Mock(
        return_value=ParsedPaper(sha256="test", pages=3, chunks=[first, second, later])
    )
    agent.index = Mock()
    agent.index.search.return_value = []
    state = RunState(id="abcdef123456", query="topic", model="local", embedding_model="bge")
    assert agent.briefing_evidence(state) == [first, second]

    abstract = first.model_copy(update={"section": "Abstract"})
    agent.parsed.return_value.chunks = [abstract, second, later]
    assert agent.briefing_evidence(state) == [abstract]


def test_session_path_traversal_rejected(tmp_path):
    with pytest.raises(PaperTrailError):
        Store(tmp_path).run_dir("../../outside")


def test_checkpoint_model_change_rejected(tmp_path):
    store = Store(tmp_path)
    store.save(
        RunState(
            id="abcdef123456",
            query="attention",
            model="original",
            embedding_model="BAAI/bge-small-en-v1.5",
        )
    )
    agent = Agent(Settings(data_dir=tmp_path, model="different"))
    with pytest.raises(PaperTrailError, match="original model"):
        agent.resume("abcdef123456")
    agent.close()


def test_invalid_input_requests_correction_instead_of_identical_retry(tmp_path):
    source = Mock()
    agent = Agent(Settings(data_dir=tmp_path), source=source)
    try:
        with pytest.raises(PaperTrailError, match="corrected ID or topic") as error:
            agent.new("1706.invalid")
        assert "Retry: papertrail resume" not in str(error.value)
        assert agent.store.recent()[0].node == Node.UNDERSTAND
        source.search.assert_not_called()
    finally:
        agent.close()
