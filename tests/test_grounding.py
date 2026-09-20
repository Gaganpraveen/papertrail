import pytest

from papertrail.errors import GroundingError
from papertrail.grounding import numbers, validate_answer, validate_briefing, validate_claim
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


@pytest.mark.parametrize(
    "source_value, claim_value",
    [
        ("22meV", "22 meV"),
        ("22 meV", "22meV"),
        ("2.5ps", "2.5 ps"),
        ("2.5 ps", "2.5ps"),
        (".5ns", "0.5 ns"),
    ],
)
def test_attached_units_do_not_hide_supported_numeric_values(chunk, source_value, claim_value):
    quote = f"The measured resolution was less than {source_value} in this experiment."
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(
        text=f"The measured resolution was less than {claim_value}.",
        evidence=[Evidence(chunk_id=source.id, quote=quote)],
    )
    validate_claim(claim, {source.id: source})


@pytest.mark.parametrize("fabricated", ["33 meV", "33meV", "2 ps", "25ps"])
def test_attached_units_do_not_allow_fabricated_values(chunk, fabricated):
    quote = "The measured energy resolution was 22meV and the temporal resolution was 2.5ps."
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(
        text=f"The measured resolution was {fabricated}.",
        evidence=[Evidence(chunk_id=source.id, quote=quote)],
    )
    with pytest.raises(GroundingError, match="numeric values missing"):
        validate_claim(claim, {source.id: source})


def test_numeric_unit_extraction_keeps_decimals_and_excludes_letter_led_identifiers():
    assert numbers("22meV 2.5ps .5ns 1,000 samples") == {"22", "2.5", "0.5", "1000"}
    assert numbers("BERT22 model2.5 SQuAD1.1 wien2k identifier_42") == set()


def test_leading_decimal_cannot_bypass_numeric_validation(chunk):
    quote = "The time resolution was .5ns for this experiment."
    source = chunk.model_copy(update={"text": quote})
    claim = Claim(
        text="The time resolution was .8ns.",
        evidence=[Evidence(chunk_id=source.id, quote=quote)],
    )
    with pytest.raises(GroundingError, match="numeric values missing"):
        validate_claim(claim, {source.id: source})


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


def test_briefing_reports_all_invalid_fields_for_one_bounded_repair(briefing, chunk):
    uncertain = chunk.model_copy(
        update={
            "id": "b" * 16,
            "text": "The approach may support more reliable retrieval in this setting.",
        }
    )
    briefing.summary = briefing.summary.model_copy(update={"text": "The model scores 99.5 BLEU."})
    briefing.method[0] = Claim(
        text="The approach supports more reliable retrieval in this setting.",
        evidence=[Evidence(chunk_id=uncertain.id, quote=uncertain.text)],
    )
    briefing.limitations_note = ""
    with pytest.raises(GroundingError) as error:
        validate_briefing(briefing, {chunk.id: chunk, uncertain.id: uncertain})
    message = str(error.value)
    assert "summary: Claim contains numeric values missing" in message
    assert "99.5" in message
    assert "method[0]: Claim removes an uncertainty qualifier" in message
    assert "limitations_note: Missing explicit limitations" in message
    assert len(message) < 700


def test_valid_briefing_validation_preserves_claims_and_evidence(briefing, chunk):
    original = briefing.model_dump()
    validate_briefing(briefing, {chunk.id: chunk})
    assert briefing.model_dump() == original


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
