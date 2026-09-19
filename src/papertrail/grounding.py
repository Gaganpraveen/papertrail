"""Deterministic provenance checks. These do not prove semantic entailment."""

import re
import unicodedata

from papertrail.errors import GroundingError
from papertrail.schema import Answer, Briefing, Chunk, Claim

EPISTEMIC = re.compile(
    r"\b(?:may|might|appear(?:s|ed|ing)?|seem(?:s|ed|ing)?|suggest(?:s|ed|ing)?)\b", re.I
)


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return " ".join(text.split()).casefold()


def numbers(text: str) -> set[str]:
    # Avoid confusing a prose hyphen with a negative sign; compare values, not unit semantics.
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", "")))


def validate_claim(claim: Claim, chunks: dict[str, Chunk]) -> None:
    quotes = []
    for evidence in claim.evidence:
        chunk = chunks.get(evidence.chunk_id)
        if chunk is None:
            raise GroundingError(f"Unknown evidence ID: {evidence.chunk_id}.")
        quote = normalized(evidence.quote)
        if len(quote.split()) < 4 or quote not in normalized(chunk.text):
            raise GroundingError(
                f"Evidence quote is not an exact passage in chunk {evidence.chunk_id}."
            )
        quotes.append(evidence.quote)
    # Conservative: an unused qualified clause can trigger a false rejection.
    # Prefer tighter evidence or qualified wording over silently strengthening it.
    if EPISTEMIC.search(" ".join(quotes)) and not EPISTEMIC.search(claim.text):
        raise GroundingError(
            "Claim removes an uncertainty qualifier from its evidence. Retain qualified wording "
            "such as 'may', 'appear', or 'suggest', or choose evidence supporting an unqualified statement."
        )
    unsupported = numbers(claim.text) - numbers(" ".join(quotes))
    if unsupported:
        raise GroundingError(
            "Claim contains numeric values missing from its evidence: "
            + ", ".join(sorted(unsupported))
        )


def briefing_claims(briefing: Briefing) -> list[Claim]:
    return [
        briefing.summary,
        briefing.problem,
        *briefing.method,
        *briefing.results,
        *briefing.limitations,
    ]


def validate_briefing(briefing: Briefing, chunks: dict[str, Chunk]) -> None:
    for claim in briefing_claims(briefing):
        validate_claim(claim, chunks)
    if not briefing.limitations and not briefing.limitations_note.strip():
        raise GroundingError("Missing explicit limitations or missing-evidence note.")


def validate_answer(answer: Answer, chunks: dict[str, Chunk]) -> None:
    if answer.status == "answered" and not answer.claims:
        raise GroundingError("An answered question needs at least one supported claim.")
    if answer.status == "insufficient_evidence" and answer.claims:
        raise GroundingError(
            "An insufficient-evidence response must not include unsupported claims."
        )
    for claim in answer.claims:
        validate_claim(claim, chunks)
