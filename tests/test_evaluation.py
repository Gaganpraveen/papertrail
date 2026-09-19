"""Offline checks for evaluation semantics; these are not retrieval benchmark results."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from papertrail.errors import GroundingError, ModelError
from papertrail.schema import Answer, Hit, ParsedPaper

SPEC = importlib.util.spec_from_file_location(
    "papertrail_evaluation", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_first_support_rank_is_normalized_and_capped_at_five(chunk):
    gold = [{"page": 2, "snippet": "THE MODEL achieves 28.4 BLEU"}]
    irrelevant = chunk.model_copy(update={"text": "Some unrelated passage without the answer."})
    ranked = [Hit(chunk=irrelevant, score=1), Hit(chunk=chunk, score=0.9)]
    score = evaluation.score_hits(ranked, gold)
    assert score["support_hit_at_5"] == 1
    assert score["reciprocal_rank_at_5"] == 0.5
    assert score["first_support_rank"] == 2
    assert score["ranks"][1]["matched_gold_indices"] == [0]
    assert all("text" not in rank for rank in score["ranks"])
    missed = evaluation.score_hits([ranked[0]] * 5 + [ranked[1]], gold)
    assert missed["support_hit_at_5"] == 0
    assert missed["reciprocal_rank_at_5"] == 0
    assert len(missed["ranks"]) == 5


def test_unanswerable_has_null_metrics_and_is_excluded_from_average(chunk):
    gold = [{"page": 2, "snippet": "The model achieves 28.4 BLEU"}]
    hit = evaluation.score_hits([Hit(chunk=chunk, score=1)], gold)
    miss = evaluation.score_hits([], gold)
    unanswerable = evaluation.score_hits([Hit(chunk=chunk, score=1)], [])
    assert unanswerable["support_hit_at_5"] is None
    assert unanswerable["reciprocal_rank_at_5"] is None
    results = [
        {"expected_status": status, "retrieval": dict.fromkeys(evaluation.MODES, score)}
        for status, score in (
            ("answered", hit),
            ("answered", miss),
            ("insufficient_evidence", unanswerable),
        )
    ]
    for summary in evaluation.aggregate_retrieval(results).values():
        assert summary["answerable_questions"] == 2
        assert summary["support_hit_at_5"] == 0.5
        assert summary["mrr_at_5"] == 0.5


def test_gold_must_match_exact_source_and_declared_page(chunk, paper):
    parsed = ParsedPaper(sha256="pdf-sha", pages=2, chunks=[chunk])
    state = SimpleNamespace(paper=paper)
    dataset = {
        "schema_version": 1,
        "source": {"pdf_sha256": "pdf-sha", "pages": 2, "arxiv_id": paper.arxiv_id},
        "questions": [
            {
                "id": "question",
                "question": "What is the BLEU score?",
                "expected_status": "answered",
                "gold": [{"page": 2, "snippet": "The model achieves 28.4 BLEU"}],
            }
        ],
    }
    assert evaluation.validate_dataset(dataset, state, parsed) == {"question": [[chunk.id]]}
    dataset["questions"][0]["gold"][0]["page"] = 1
    with pytest.raises(ValueError, match="declared page"):
        evaluation.validate_dataset(dataset, state, parsed)
    dataset["source"]["pdf_sha256"] = "different-pdf"
    with pytest.raises(ValueError, match="different PDF checksum"):
        evaluation.validate_dataset(dataset, state, parsed)


def test_model_errors_are_not_silently_counted_as_successful_abstentions(chunk):
    class BrokenModel:
        calls = total_tokens = 0

        def answer(self, *args, **kwargs):
            raise ModelError("Model was unavailable")

    case = {"question": "Which random seed?", "expected_status": "insufficient_evidence"}
    result = evaluation.evaluate_answer(BrokenModel(), case, [Hit(chunk=chunk, score=1)])
    assert result["status"] is None
    assert not result["status_agrees"]
    assert result["provenance"]["check"] == "not_completed"


def test_grounding_rejection_is_explicit_and_abstention_has_no_provenance_claim(chunk):
    class RejectedModel:
        calls = total_tokens = 0

        def answer(self, *args, **kwargs):
            raise GroundingError("Unsupported draft")

    case = {"question": "Which random seed?", "expected_status": "insufficient_evidence"}
    result = evaluation.evaluate_answer(RejectedModel(), case, [Hit(chunk=chunk, score=1)])
    assert result["status"] == "insufficient_evidence"
    assert result["status_agrees"]
    assert result["grounding_rejected"]
    assert result["provenance"]["check"] == "no_claims_to_check"


def test_answer_checks_do_not_supply_conversation_history(chunk):
    class AbstainingModel:
        calls = total_tokens = 0

        def answer(self, question, chunks, previous_questions):
            assert previous_questions == []
            assert chunks == [chunk]
            return Answer(status="insufficient_evidence", explanation="No source support")

    case = {"question": "Which random seed?", "expected_status": "insufficient_evidence"}
    result = evaluation.evaluate_answer(AbstainingModel(), case, [Hit(chunk=chunk, score=1)])
    assert result["status_agrees"]
    assert not result["grounding_rejected"]
