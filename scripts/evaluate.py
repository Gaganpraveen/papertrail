#!/usr/bin/env python3
"""Reproducible, small retrieval smoke evaluation; optional answer-status checks.

Uses a saved session's existing index and never writes to the session or its QA history.
Install the project first, then run `python scripts/evaluate.py --help`.
"""

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import sys
import time
from contextlib import closing
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from filelock import FileLock, Timeout

from papertrail.config import Settings
from papertrail.errors import GroundingError, PaperTrailError
from papertrail.grounding import normalized, validate_answer
from papertrail.llm import Ollama
from papertrail.retrieval import Embedder, HybridIndex
from papertrail.schema import Answer, Hit, ParsedPaper, RunState, now
from papertrail.storage import atomic_json

MODES = ("dense", "bm25", "hybrid")
K = 5
DEFAULT_QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "attention-questions.json"


def load_session(data_dir: Path, session: str) -> tuple[RunState, ParsedPaper, bytes]:
    """Open SQLite read-only, avoiding Store initialization or checkpoint writes."""
    if not re.fullmatch(r"[a-f0-9]{12}", session):
        raise ValueError("Session must be a 12-character hexadecimal saved session ID.")
    database = (data_dir / "sessions.sqlite3").resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute("SELECT state FROM runs WHERE id=?", (session,)).fetchone()
    if row is None:
        raise ValueError(f"Session {session!r} was not found.")
    state = RunState.model_validate_json(row[0])
    if not state.collection or not state.paper:
        raise ValueError("The saved session must have a selected paper and a completed index.")
    raw = (data_dir / "runs" / session / "parsed.json").read_bytes()
    parsed = ParsedPaper.model_validate_json(raw)
    if parsed.sha256 != state.pdf_sha256:
        raise ValueError("The saved parsed PDF checksum differs from the session PDF checksum.")
    return state, parsed, raw


def validate_dataset(
    dataset: dict, state: RunState, parsed: ParsedPaper
) -> dict[str, list[list[str]]]:
    """Fail before retrieval if labels are malformed or a gold excerpt is absent."""
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation dataset schema_version.")
    if dataset["source"]["pdf_sha256"] != parsed.sha256:
        raise ValueError("Gold labels belong to a different PDF checksum.")
    if state.paper is None or dataset["source"]["arxiv_id"] != state.paper.arxiv_id:
        raise ValueError("Gold labels belong to a different paper/version.")
    if dataset["source"]["pages"] != parsed.pages:
        raise ValueError("Gold labels have a different PDF page count.")
    cases = dataset.get("questions", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Questions must be non-empty with unique IDs.")
    located = {}
    for case in cases:
        status, gold = case["expected_status"], case["gold"]
        if not case["question"].strip() or status not in {"answered", "insufficient_evidence"}:
            raise ValueError(f"Invalid question or expected status: {case['id']}.")
        if bool(gold) != (status == "answered"):
            raise ValueError(f"Only answerable cases must have gold snippets: {case['id']}.")
        locations = []
        for evidence in gold:
            snippet = normalized(evidence["snippet"])
            if len(snippet.split()) < 4:
                raise ValueError(f"Gold excerpt is too short: {case['id']}.")
            matches = [
                chunk.id
                for chunk in parsed.chunks
                if chunk.page == evidence["page"]
                and not chunk.reference
                and snippet in normalized(chunk.text)
            ]
            if not matches:
                raise ValueError(f"Gold excerpt absent from its declared page: {case['id']}.")
            locations.append(matches)
        located[case["id"]] = locations
    if not any(case["expected_status"] == "answered" for case in cases):
        raise ValueError("At least one answerable question is required.")
    return located


def score_hits(hits: list[Hit], gold: list[dict]) -> dict:
    """Any normalized gold snippet in a returned passage counts as a support hit.

    Gold pages validate annotation provenance; scoring is based on passage text.
    No gold means unanswerable, so the retrieval metrics are deliberately null.
    """
    snippets = [normalized(item["snippet"]) for item in gold]
    ranks = []
    for rank, hit in enumerate(hits[:K], 1):
        matches = [i for i, snippet in enumerate(snippets) if snippet in normalized(hit.chunk.text)]
        ranks.append(
            {
                "rank": rank,
                "chunk_id": hit.chunk.id,
                "page": hit.chunk.page,
                "section": hit.chunk.section,
                "score": hit.score,
                "dense_score": hit.dense_score,
                "lexical_score": hit.lexical_score,
                "matched_gold_indices": matches,
            }
        )
    first = next((item["rank"] for item in ranks if item["matched_gold_indices"]), None)
    return {
        "support_hit_at_5": int(first is not None) if gold else None,
        "reciprocal_rank_at_5": (1.0 / first if first else 0.0) if gold else None,
        "first_support_rank": first,
        "ranks": ranks,
    }


def aggregate_retrieval(results: list[dict]) -> dict:
    answerable = [item for item in results if item["expected_status"] == "answered"]
    summary = {}
    for mode in MODES:
        scores = [item["retrieval"][mode] for item in answerable]
        summary[mode] = {
            "answerable_questions": len(scores),
            "support_hits_at_5": sum(score["support_hit_at_5"] for score in scores),
            "support_hit_at_5": sum(score["support_hit_at_5"] for score in scores) / len(scores),
            "mrr_at_5": sum(score["reciprocal_rank_at_5"] for score in scores) / len(scores),
        }
    return summary


def evaluate_answer(llm: Ollama, case: dict, hits: list[Hit]) -> dict:
    """Match the application's safe grounding-rejection behavior without saving history."""
    started, calls, tokens = time.perf_counter(), llm.calls, llm.total_tokens
    chunks = [hit.chunk for hit in hits]
    rejection = None
    try:
        try:
            answer = (
                llm.answer(case["question"], chunks, previous_questions=[])
                if chunks
                else Answer(
                    status="insufficient_evidence",
                    explanation="No readable source passages were retrieved.",
                )
            )
        except GroundingError as exc:
            rejection = str(exc)
            answer = Answer(
                status="insufficient_evidence",
                explanation="A draft answer failed evidence validation; no unverified answer is shown.",
            )
        validate_answer(answer, {chunk.id: chunk for chunk in chunks})
        return {
            "status": answer.status,
            "status_agrees": answer.status == case["expected_status"],
            "answer": answer.model_dump(),
            "grounding_rejected": rejection is not None,
            "grounding_rejection_reason": rejection,
            "provenance": {
                "check": "passed" if answer.claims else "no_claims_to_check",
                "claim_count": len(answer.claims),
                "quote_count": sum(len(claim.evidence) for claim in answer.claims),
                "semantic_correctness": "not_scored",
            },
            "seconds": round(time.perf_counter() - started, 6),
            "model_calls": llm.calls - calls,
            "generated_tokens": llm.total_tokens - tokens,
        }
    except PaperTrailError as exc:
        return {
            "status": None,
            "status_agrees": False,
            "error": str(exc),
            "provenance": {"check": "not_completed", "semantic_correctness": "not_scored"},
            "seconds": round(time.perf_counter() - started, 6),
            "model_calls": llm.calls - calls,
            "generated_tokens": llm.total_tokens - tokens,
        }


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("papertrail-arxiv", "qdrant-client", "fastembed", "onnxruntime"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def run(args: argparse.Namespace) -> dict | None:
    state, parsed, parsed_raw = load_session(args.data_dir, args.session)
    dataset_raw = args.questions.read_bytes()
    dataset = json.loads(dataset_raw)
    locations = validate_dataset(dataset, state, parsed)
    if args.validate_only:
        print(
            f"Validated {len(dataset['questions'])} cases and every gold excerpt; no retrieval/model calls."
        )
        return None

    embedder = Embedder(state.embedding_model, args.data_dir / "models")
    index = HybridIndex(args.data_dir / "vectors", embedder)
    llm = None
    settings = Settings.from_env(data_dir=args.data_dir, model=state.model)
    report = {
        "schema_version": 1,
        "created_at": now(),
        "evaluation_kind": "developer_constructed_smoke_eval_not_held_out",
        "dataset": {
            "path": str(args.questions),
            "sha256": hashlib.sha256(dataset_raw).hexdigest(),
            "methodology": dataset["methodology"],
        },
        "session": {
            "id": state.id,
            "status": state.status,
            "paper": state.paper.model_dump(),
            "collection": state.collection,
            "pdf_sha256": parsed.sha256,
            "parsed_json_sha256": hashlib.sha256(parsed_raw).hexdigest(),
            "pages": parsed.pages,
            "chunks": len(parsed.chunks),
            "eligible_nonreference_chunks": sum(not chunk.reference for chunk in parsed.chunks),
            "embedding_model": state.embedding_model,
            "answer_model": state.model,
        },
        "environment": package_versions(),
        "protocol": {
            "k": K,
            "modes": list(MODES),
            "query": "question verbatim, no conversation history",
            "references": "excluded in every mode",
            "ranking": "application HybridIndex.search; shared overlap suppression in every mode",
            "support_match": "at least one normalized gold snippet occurs inside a retrieved passage",
            "retrieval_denominator": "answerable questions only; unanswerable cases excluded",
            "timing": "single sequential pass; includes lazy initialization; not a latency benchmark",
            "answers": "hybrid top 5 only, direct Ollama.answer, empty previous questions"
            if args.answers
            else "not run",
            "status_agreement": "expected answered/insufficient_evidence only; not answer correctness",
            "provenance": "application quote/ID/numeric checks; does not prove semantic entailment",
            "support_review": "same-model screening inside Ollama.answer; not an independent judge",
            "session_history": "not modified",
        },
        "results": [],
    }
    started = time.perf_counter()
    try:
        if args.answers:
            llm = Ollama(settings.ollama_url, state.model, settings.model_timeout)
            model_info = llm.available()
            report["session"]["ollama_model"] = {
                key: model_info.get(key) for key in ("name", "digest", "size", "details")
            }
        for case in dataset["questions"]:
            print(f"Evaluating {case['id']}: {case['question']}", flush=True)
            result = {**case, "gold_chunk_ids": locations[case["id"]], "retrieval": {}}
            hybrid_hits = []
            for mode in MODES:
                retrieval_start = time.perf_counter()
                hits = index.search(
                    state.collection, parsed.chunks, case["question"], limit=K, mode=mode
                )
                elapsed = time.perf_counter() - retrieval_start
                result["retrieval"][mode] = {
                    **score_hits(hits, case["gold"]),
                    "seconds": round(elapsed, 6),
                }
                if mode == "hybrid":
                    hybrid_hits = hits
            if llm is not None:
                result["answer_check"] = evaluate_answer(llm, case, hybrid_hits)
            report["results"].append(result)
    finally:
        try:
            index.close()
        finally:
            if llm is not None:
                llm.close()

    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    report["retrieval_summary"] = aggregate_retrieval(report["results"])
    report["unanswerable_questions_excluded_from_retrieval_metrics"] = sum(
        result["expected_status"] == "insufficient_evidence" for result in report["results"]
    )
    if args.answers:
        checks = [result["answer_check"] for result in report["results"]]
        report["answer_status_summary"] = {
            "questions": len(checks),
            "completed": sum(check["status"] is not None for check in checks),
            "status_matches": sum(check["status_agrees"] for check in checks),
            "status_agreement": sum(check["status_agrees"] for check in checks) / len(checks),
            "execution_errors": sum(check["status"] is None for check in checks),
            "grounding_rejections": sum(check.get("grounding_rejected", False) for check in checks),
            "note": "Execution errors count as disagreements; status agreement is not answer correctness.",
        }
    atomic_json(args.output, report)
    print(json.dumps(report["retrieval_summary"], indent=2))
    print(f"Saved measured results to {args.output}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Saved session containing the indexed PDF")
    parser.add_argument("--data-dir", type=Path, default=Path(".papertrail"))
    parser.add_argument("--output", type=Path, default=Path("examples/evaluation.json"))
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--answers", action="store_true", help="Also run hybrid top-5 answer-status checks"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check source and labels without model/index access",
    )
    args = parser.parse_args()
    try:
        if not args.data_dir.is_dir():
            raise ValueError(f"Data directory not found: {args.data_dir}")
        # Coordinate with the CLI; do not open local Qdrant during another operation.
        with FileLock(str(args.data_dir / "writer.lock"), timeout=0):
            run(args)
    except Timeout:
        print(
            "Another PaperTrail operation is using this data directory; retry after it finishes.",
            file=sys.stderr,
        )
        return 1
    except (PaperTrailError, OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
