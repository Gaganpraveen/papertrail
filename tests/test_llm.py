import json

import httpx
import pytest

from papertrail.errors import GroundingError, ModelError
from papertrail.grounding import validate_claim
from papertrail.llm import (
    DraftAnswer,
    DraftBriefing,
    DraftClaim,
    Ollama,
    evidence_context,
    resolve_claim,
)
from papertrail.schema import Answer


def test_invalid_output_retries_once_then_fails():
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, json={"message": {"content": '{"bad":true}'}})

    llm = Ollama(
        "http://localhost:11434",
        "test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(GroundingError, match="twice"):
        llm.generate(Answer, "test prompt", lambda a: None)
    assert len(attempts) == 2


def test_abstention_is_preserved():
    answer = DraftAnswer(status="insufficient_evidence")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"message": {"content": answer.model_dump_json()}}
            )
        )
    )
    llm = Ollama("http://localhost:11434", "test", client=client)
    result = llm.answer("What is the author's home address?", [], [])
    assert not result.claims and result.status == "insufficient_evidence"


def test_local_model_missing_actionable():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))
    )
    with pytest.raises(ModelError, match="ollama pull"):
        Ollama("http://localhost", "missing", client=client).available()


def test_prompt_treats_source_as_untrusted(chunk, claim):
    answer = DraftAnswer(
        status="answered", claims=[DraftClaim(text=claim.text, evidence_ids=["E1"])]
    )
    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        if payload["format"]["title"] == "SupportReview":
            review = {"verdicts": [{"claim_index": 0, "supported": True, "reason": "Supported."}]}
            return httpx.Response(200, json={"message": {"content": json.dumps(review)}})
        return httpx.Response(200, json={"message": {"content": answer.model_dump_json()}})

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    llm.answer("What score?", [chunk], [])
    assert "untrusted source data" in captured[0]["messages"][0]["content"]
    assert captured[0]["format"]["additionalProperties"] is False


def test_quotes_are_copied_by_code(chunk):
    context, registry = evidence_context([chunk])
    claim = resolve_claim(DraftClaim(text="A grounded claim.", evidence_ids=["E1"]), registry)
    assert claim.evidence[0].quote == "The model achieves 28.4 BLEU on the translation benchmark."
    assert claim.evidence[0].chunk_id == chunk.id
    with pytest.raises(GroundingError):
        resolve_claim(DraftClaim(text="A guess.", evidence_ids=["E999"]), registry)


@pytest.mark.parametrize(
    "short_statement, claim_text",
    [
        ("We used warmup_steps = 4000.", "Warmup used 4000 steps."),
        ("We used\nwarmup_steps = 4000.", "Warmup used 4000 steps."),
        ("Training lasted 12 hours.", "Training lasted 12 hours."),
    ],
)
def test_short_numeric_source_sentences_remain_citable(chunk, short_statement, claim_text):
    normalized_statement = " ".join(short_statement.split())
    source = chunk.model_copy(
        update={"text": "The learning rate increases during the warmup phase. " + short_statement}
    )
    context, registry = evidence_context([source])
    alias = next(
        identity
        for identity, evidence in registry.items()
        if evidence.quote == normalized_statement
    )
    assert normalized_statement in context
    claim = resolve_claim(DraftClaim(text=claim_text, evidence_ids=[alias]), registry)
    assert claim.evidence[0].quote == normalized_statement
    validate_claim(claim, {source.id: source})


@pytest.mark.parametrize("formula", ["x+y", "x = y", "x^2", "x/y", "x*y", "x-y"])
def test_displaced_math_is_withheld_without_reconstructing_neighboring_prose(chunk, formula):
    corrupt = f"For a fixed offset, the vector is a linear function of\n{formula}\ny."
    rationale = "The authors chose this representation because it may support longer sequences."
    source = chunk.model_copy(update={"text": corrupt + "\n" + rationale})
    original = source.model_dump()
    context, registry = evidence_context([source])
    assert "linear function" not in context
    assert [e.quote for e in registry.values()] == [rationale]
    assert registry["E1"].chunk_id == source.id
    assert source.model_dump() == original


def test_inline_math_and_hyphenated_prose_are_not_isolated_formula_lines(chunk):
    text = "The function x+y represents the offset used by the encoder. The state-of-the-art model uses attention."
    source = chunk.model_copy(update={"text": text})
    context, registry = evidence_context([source])
    assert "function x+y" in context and "state-of-the-art" in context
    assert len(registry) == 2


def test_uncertainty_loss_is_rejected_before_a_permissive_model_reviewer(chunk):
    source = chunk.model_copy(
        update={
            "text": "Many attention heads appear to exhibit behavior related to sentence structure."
        }
    )
    draft = DraftAnswer(
        status="answered",
        claims=[
            DraftClaim(
                text="Many attention heads exhibit behavior related to sentence structure.",
                evidence_ids=["E1"],
            )
        ],
    )
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload["format"]["title"])
        content = (
            draft.model_dump_json()
            if payload["format"]["title"] == "DraftAnswer"
            else json.dumps(
                {"verdicts": [{"claim_index": 0, "supported": True, "reason": "Accepted."}]}
            )
        )
        return httpx.Response(200, json={"message": {"content": content}})

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        with pytest.raises(GroundingError, match="uncertainty qualifier"):
            llm.answer("What do the heads exhibit?", [source], [])
        assert requests == ["DraftAnswer", "DraftAnswer"]
    finally:
        llm.close()


def test_flat_numeric_table_is_withheld_but_neighboring_result_prose_is_citable(chunk):
    flattened_table = (
        "Benchmark Model Task-A Task-B "
        "Baseline-A 23.75 39.2 Baseline-B 24.6 39.92 "
        "Baseline-C 25.16 40.46 Baseline-D 26.03 40.56 "
        "Ensemble-A 26.30 41.16 Ensemble-B 26.36 41.29."
    )
    result_prose = "The model achieves 28.4 BLEU on the translation benchmark."
    source = chunk.model_copy(update={"text": flattened_table + " " + result_prose})
    original = source.model_dump()
    context, registry = evidence_context([source])

    assert flattened_table not in context
    assert "Baseline-A" not in context
    assert result_prose in context
    assert list(registry) == ["E1"]
    assert registry["E1"].quote == result_prose
    assert registry["E1"].chunk_id == source.id
    assert json.loads(context)[0]["evidence"] == [{"id": "E1", "text": result_prose}]
    assert (
        source.model_dump() == original
    )  # Retrieval/source inspection retains the original table.


def test_rejected_briefing_prose_becomes_exact_source_quotes_with_original_ids(chunk, paper):
    second_quote = "The training procedure uses eight GPUs for every experiment."
    second = chunk.model_copy(update={"id": "b" * 16, "page": 3, "text": second_quote})
    _, registry = evidence_context([chunk, second])
    second_alias = next(key for key, evidence in registry.items() if evidence.chunk_id == second.id)
    rejected_prose = "Our model guarantees reliable predictions in every research domain."
    supported = DraftClaim(text="The model achieves 28.4 BLEU.", evidence_ids=["E1"])
    draft = DraftBriefing(
        summary=DraftClaim(text=rejected_prose, evidence_ids=["E1", second_alias]),
        problem=supported,
        method=[supported],
        results=[supported],
        follow_up_questions=["How was training conducted?", "How were results evaluated?"],
    )
    reviews = 0

    def handler(request):
        nonlocal reviews
        payload = json.loads(request.content)
        if payload["format"]["title"] == "DraftBriefing":
            return httpx.Response(200, json={"message": {"content": draft.model_dump_json()}})
        reviews += 1
        count = 3 if reviews == 1 else 1
        verdicts = [
            {
                "claim_index": index,
                "supported": not (reviews == 1 and index == 0),
                "reason": "The claimed universal guarantee is absent from the quoted evidence."
                if reviews == 1 and index == 0
                else "Supported.",
            }
            for index in range(count)
        ]
        return httpx.Response(
            200, json={"message": {"content": json.dumps({"verdicts": verdicts})}}
        )

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        assert llm.extractive_fallbacks == 0
        result = llm.briefing(paper, [chunk, second])
        expected_quotes = [
            "The model achieves 28.4 BLEU on the translation benchmark.",
            second_quote,
        ]
        assert result.summary.text == (
            "Source wording (automatic review requested inspection): "
            + " ".join(f"“{quote}”" for quote in expected_quotes)
        )
        assert rejected_prose not in result.model_dump_json()
        assert [e.chunk_id for e in result.summary.evidence] == [chunk.id, second.id]
        assert [e.quote for e in result.summary.evidence] == expected_quotes
        assert result.results[0].text == supported.text
        assert llm.extractive_fallbacks == 1
        assert llm.calls == 3  # One draft plus two review batches; no paraphrase retry.
    finally:
        llm.close()


def test_rejected_prior_work_limitation_is_omitted_instead_of_relabelled(chunk, paper):
    prior_work = chunk.model_copy(
        update={
            "id": "c" * 16,
            "text": "Earlier recurrent models cannot parallelize computation within individual training examples.",
        }
    )
    _, registry = evidence_context([chunk, prior_work])
    prior_alias = next(
        key for key, evidence in registry.items() if evidence.chunk_id == prior_work.id
    )
    supported = DraftClaim(text="The model achieves 28.4 BLEU.", evidence_ids=["E1"])
    draft = DraftBriefing(
        summary=supported,
        problem=supported,
        method=[supported],
        results=[supported],
        limitations=[
            DraftClaim(
                text="The proposed model cannot parallelize computation within training examples.",
                evidence_ids=[prior_alias],
            )
        ],
        follow_up_questions=["How was training conducted?", "How were results evaluated?"],
    )
    reviews = 0

    def handler(request):
        nonlocal reviews
        payload = json.loads(request.content)
        if payload["format"]["title"] == "DraftBriefing":
            return httpx.Response(200, json={"message": {"content": draft.model_dump_json()}})
        reviews += 1
        count = 3 if reviews == 1 else 2
        verdicts = [
            {
                "claim_index": index,
                "supported": not (reviews == 2 and index == 1),
                "reason": "This limitation concerns prior recurrent models, not the proposed model."
                if reviews == 2 and index == 1
                else "Supported.",
            }
            for index in range(count)
        ]
        return httpx.Response(
            200, json={"message": {"content": json.dumps({"verdicts": verdicts})}}
        )

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        result = llm.briefing(paper, [chunk, prior_work])
        assert result.limitations == []
        assert "No explicit limitations were accepted" in result.limitations_note
        assert "does not establish that the paper has none" in result.limitations_note
        assert "Earlier recurrent models" not in result.model_dump_json()
        assert llm.extractive_fallbacks == 1
    finally:
        llm.close()


def test_qa_support_rejection_retries_once_and_never_uses_extractive_fallback(chunk):
    rejected_prose = "This model guarantees perfect translations in every language."
    draft = DraftAnswer(
        status="answered", claims=[DraftClaim(text=rejected_prose, evidence_ids=["E1"])]
    )
    draft_requests = []
    review_requests = []

    def handler(request):
        payload = json.loads(request.content)
        if payload["format"]["title"] == "DraftAnswer":
            draft_requests.append(payload)
            return httpx.Response(200, json={"message": {"content": draft.model_dump_json()}})
        review_requests.append(payload)
        review = {
            "verdicts": [
                {
                    "claim_index": 0,
                    "supported": False,
                    "reason": "No guarantee appears in evidence.",
                }
            ]
        }
        return httpx.Response(200, json={"message": {"content": json.dumps(review)}})

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        with pytest.raises(GroundingError, match="twice.*Support review rejected"):
            llm.answer("Does this model guarantee perfect translations?", [chunk], [])
        assert len(draft_requests) == len(review_requests) == 2
        assert llm.calls == 4
        assert llm.extractive_fallbacks == 0
        repair_messages = draft_requests[1]["messages"]
        assert repair_messages[-2] == {"role": "assistant", "content": draft.model_dump_json()}
        assert "Support review rejected" in repair_messages[-1]["content"]
        assert "Do not repeat the rejected claim" in repair_messages[-1]["content"]
    finally:
        llm.close()


def test_streamed_response_is_assembled_before_validation_and_timed():
    events, requests = [], []
    draft = '{"status":"insufficient_evidence","claims":[]}'

    def handler(request):
        requests.append(json.loads(request.content))
        lines = [
            {"message": {"content": draft[:20]}, "done": False},
            {"message": {"content": draft[20:]}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "eval_count": 12,
                "load_duration": 1000000,
                "prompt_eval_duration": 2000000,
                "eval_duration": 3000000,
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            text="\n".join(json.dumps(line) for line in lines),
        )

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    llm.observer = events.append
    try:
        result = llm.answer("An unsupported question?", [], [])
        assert result.status == "insufficient_evidence"
        assert requests[0]["stream"] is True
        assert [(e.node, e.status) for e in events] == [
            ("qa.generate", "running"),
            ("qa.generate", "completed"),
        ]
        assert json.loads(events[-1].detail)["server_seconds"]["eval_duration"] == 0.003
        assert llm.total_tokens == 12
    finally:
        llm.close()


def test_interrupted_stream_closes_transport_and_never_accepts_partial_json():
    class InterruptedStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b'{"message":{"content":"{\\"status\\":\\"insufficient_evidence\\",\\"claims\\":[]}"},"done":false}\n'

        def close(self):
            self.closed = True

    stream, events, accepted = InterruptedStream(), [], []
    llm = Ollama(
        "http://localhost",
        "test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "application/x-ndjson"}, stream=stream
                )
            )
        ),
    )
    llm.observer = events.append
    try:
        with pytest.raises(ModelError, match="interrupted"):
            llm.generate(DraftAnswer, "question", accepted.append)
        assert stream.closed and accepted == [] and llm.calls == 0
        assert events[-1].status == "failed"
    finally:
        llm.close()


def test_warmup_has_separate_visible_loading_stage():
    requests, events = [], []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"done": True})

    llm = Ollama(
        "http://localhost", "test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    llm.observer = events.append
    try:
        llm.warmup()
        assert requests[0][0] == "/api/generate"
        assert requests[0][1]["options"]["num_ctx"] == 32768
        assert "prompt" not in requests[0][1]
        assert [(e.node, e.status) for e in events] == [
            ("model.load", "running"),
            ("model.load", "completed"),
        ]
        assert llm.calls == 0
    finally:
        llm.close()


def test_failed_validation_is_reported_for_each_bounded_attempt():
    events = []
    llm = Ollama(
        "http://localhost",
        "test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"message": {"content": '{"bad":true}'}})
            )
        ),
    )
    llm.observer = events.append
    try:
        with pytest.raises(GroundingError, match="twice"):
            llm.generate(DraftAnswer, "question", lambda result: None)
        rejected = [event for event in events if event.status == "rejected"]
        assert len(rejected) == 2
        assert all(event.node == "brief.validate" and event.detail for event in rejected)
        assert llm.calls == 2
    finally:
        llm.close()
