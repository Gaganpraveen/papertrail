import pytest

from papertrail.errors import GroundingError
from papertrail.grounding import validate_answer, validate_briefing, validate_claim
from papertrail.schema import Answer, Claim, Evidence


def test_real_quote_passes(claim, chunk):
    validate_claim(claim, {chunk.id: chunk})


def test_forged_citation_rejected(claim, chunk):
    claim.evidence[0].chunk_id = "invented"
    with pytest.raises(GroundingError, match="Unknown"):
        validate_claim(claim, {chunk.id: chunk})


def test_paraphrased_quote_rejected(claim, chunk):
    claim.evidence[0].quote = "The model scored 28.4 BLEU on the translation benchmark."
    with pytest.raises(GroundingError, match="exact passage"):
        validate_claim(claim, {chunk.id: chunk})


def test_invented_number_rejected(claim, chunk):
    claim.text = "The model achieves 98.4 BLEU."
    with pytest.raises(GroundingError, match="98.4"):
        validate_claim(claim, {chunk.id: chunk})


def test_whitespace_normalization(claim, chunk):
    chunk.text = chunk.text.replace("achieves ", "achieves\n")
    validate_claim(claim, {chunk.id: chunk})


def test_answer_requires_evidence():
    with pytest.raises(GroundingError):
        validate_answer(Answer(status="answered", explanation="guess"), {})


def test_abstention_cannot_smuggle_claims(claim, chunk):
    with pytest.raises(GroundingError):
        validate_answer(
            Answer(status="insufficient_evidence", claims=[claim], explanation=""),
            {chunk.id: chunk},
        )


def test_limitations_cannot_be_silently_omitted(briefing, chunk):
    briefing.limitations_note = ""
    with pytest.raises(GroundingError, match="limitations"):
        validate_briefing(briefing, {chunk.id: chunk})


@pytest.mark.parametrize(
    "qualifier",
    [
        "may",
        "might",
        "appear",
        "appears",
        "appeared",
        "seem",
        "seems",
        "seemed",
        "suggest",
        "suggests",
        "suggested",
    ],
)
def test_explicit_uncertainty_cannot_disappear_from_claim(chunk, qualifier):
    quote = (
        f"The findings {qualifier} support a relationship between attention and sentence structure."
    )
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(
        text="The findings support a relationship between attention and sentence structure.",
        evidence=[Evidence(chunk_id=source.id, quote=quote)],
    )
    with pytest.raises(GroundingError, match="uncertainty qualifier"):
        validate_claim(claim, {source.id: source})


@pytest.mark.parametrize(
    "text",
    [
        "Many attention heads appear to exhibit behavior related to sentence structure.",
        "Many attention heads may exhibit behavior related to sentence structure.",
    ],
)
def test_qualified_wording_is_accepted(chunk, text):
    quote = "Many attention heads appear to exhibit behavior related to sentence structure."
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(text=text, evidence=[Evidence(chunk_id=source.id, quote=quote)])
    validate_claim(claim, {source.id: source})


def test_qualifier_guard_does_not_match_ordinary_word_substrings(chunk):
    quote = "The appearance of attention maps is discussed in the appendix."
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(
        text="Attention maps are discussed in the appendix.",
        evidence=[Evidence(chunk_id=source.id, quote=quote)],
    )
    validate_claim(claim, {source.id: source})
