"""Evidence aliases must be citable before any model call is made."""

import json

import pytest

from papertrail.grounding import normalized, validate_claim
from papertrail.llm import evidence_context
from papertrail.schema import Claim


@pytest.mark.parametrize(
    "source_text, expected",
    [
        (
            "The model improves general-\nization across diverse language tasks.",
            "The model improves generalization across diverse language tasks.",
        ),
        (
            "The model improves general- \n  ization across diverse language tasks.",
            "The model improves generalization across diverse language tasks.",
        ),
        (
            "The eﬃcient encoder learns ﬁne-grained representations from unlabeled text.",
            "The efficient encoder learns fine-grained representations from unlabeled text.",
        ),
        (
            "The state-of-the-art encoder learns useful representations from unlabeled text.",
            "The state-of-the-art encoder learns useful representations from unlabeled text.",
        ),
        (
            "The ﬁne-\ngrained encoder learns useful representations from unlabeled text.",
            "The finegrained encoder learns useful representations from unlabeled text.",
        ),
    ],
)
def test_alias_normalization_matches_source_provenance(chunk, source_text, expected):
    source = chunk.model_copy(update={"text": source_text})
    original = source.model_dump()
    context, registry = evidence_context([source])
    assert [evidence.quote for evidence in registry.values()] == [expected]
    assert json.loads(context)[0]["evidence"] == [{"id": "E1", "text": expected}]
    for evidence in registry.values():
        validate_claim(
            Claim(text="The source describes the encoder.", evidence=[evidence]),
            {source.id: source},
        )
    assert source.model_dump() == original


def test_long_quote_boundaries_preserve_exact_normalized_substrings(chunk):
    text = "The encoder represents language with contextual features " * 50
    source = chunk.model_copy(update={"text": text + "and general-\nization across tasks."})
    _, registry = evidence_context([source])
    assert len(registry) > 1
    for evidence in registry.values():
        assert len(evidence.quote) <= 1000
        assert len(normalized(evidence.quote).split()) >= 4
        assert normalized(evidence.quote) in normalized(source.text)
        validate_claim(
            Claim(text="The source describes the encoder.", evidence=[evidence]),
            {source.id: source},
        )


def test_caption_interleaved_into_unfinished_prose_is_withheld(chunk):
    malformed = (
        "Our major contribution is further general-\n"
        "Table 6: Ablation over model size. The izing these findings to deep architectures.\n"
        "number of layers is mixed into the other column's sentence."
    )
    source = chunk.model_copy(update={"text": malformed})
    original = source.model_dump()
    context, registry = evidence_context([source])
    assert registry == {}
    assert json.loads(context)[0]["evidence"] == []
    assert source.model_dump() == original


def test_caption_after_completed_prose_does_not_discard_readable_evidence(chunk):
    source = chunk.model_copy(
        update={
            "text": "The model uses bidirectional context to represent tokens.\n"
            "Table 6: The experiment compares several different model sizes."
        }
    )
    _, registry = evidence_context([source])
    assert any("bidirectional context" in evidence.quote for evidence in registry.values())
    for evidence in registry.values():
        assert normalized(evidence.quote) in normalized(source.text)


def test_normalized_word_minimum_is_checked_after_line_hyphen_join(chunk):
    source = chunk.model_copy(update={"text": "The general-\nization improved."})
    _, registry = evidence_context([source])
    assert registry == {}
