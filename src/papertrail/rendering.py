"""Portable Markdown, JSON and an offline evidence reader. No external assets or scripts."""

import html
from pathlib import Path

from papertrail.schema import Chunk, Claim, RunState
from papertrail.storage import atomic_json


def citation(claim: Claim, chunks: dict[str, Chunk], pdf_url: str) -> str:
    refs = []
    for ev in claim.evidence:
        c = chunks[ev.chunk_id]
        refs.append(f"[p. {c.page}, {c.section}]({pdf_url}#page={c.page})")
    return claim.text + " " + "; ".join(refs)


def markdown(state: RunState, chunks: list[Chunk]) -> str:
    paper, briefing = state.paper, state.briefing
    if paper is None or briefing is None:
        return f"# PaperTrail session {state.id}\n\nStatus: {state.status}\n\n{state.error or ''}\n"
    lookup = {c.id: c for c in chunks}
    lines = [
        f"# {paper.title}",
        "",
        f"**Authors:** {', '.join(paper.authors)}",
        f"**arXiv:** [{paper.arxiv_id}]({paper.url}) · **Published:** {paper.published[:10]}",
        f"**Local model:** {state.model} · **Session:** `{state.id}`",
        "",
        "## Why this paper matters",
        "",
        citation(briefing.summary, lookup, paper.pdf_url),
        "",
        "## Problem",
        "",
        citation(briefing.problem, lookup, paper.pdf_url),
    ]
    for title, claims in (
        ("Method", briefing.method),
        ("Key results and claims", briefing.results),
        ("Limitations", briefing.limitations),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend("- " + citation(c, lookup, paper.pdf_url) for c in claims)
        if title == "Limitations":
            lines.extend(["", briefing.limitations_note])
    lines.extend(["", "## Suggested questions", ""])
    lines.extend("- " + q for q in briefing.follow_up_questions)
    for i, exchange in enumerate(state.exchanges, 1):
        lines.extend(
            [
                "",
                f"## Q{i}: {exchange.question}",
                "",
                f"*{exchange.answer.status.replace('_', ' ')} · {exchange.elapsed_seconds:.1f}s*",
                "",
            ]
        )
        lines.extend(citation(c, lookup, paper.pdf_url) + "\n" for c in exchange.answer.claims)
        if exchange.answer.explanation:
            lines.append(exchange.answer.explanation)
    lines.extend(
        [
            "",
            "## Evidence ledger",
            "",
            "Quotations below were checked against extracted PDF text. This verifies provenance, not semantic entailment.",
            "",
        ]
    )
    all_claims = [
        briefing.summary,
        briefing.problem,
        *briefing.method,
        *briefing.results,
        *briefing.limitations,
        *[claim for ex in state.exchanges for claim in ex.answer.claims],
    ]
    seen = set()
    for claim in all_claims:
        for ev in claim.evidence:
            key = (ev.chunk_id, ev.quote)
            if key in seen:
                continue
            seen.add(key)
            c = lookup[ev.chunk_id]
            lines.extend(
                [
                    f"**`{c.id}` · page {c.page} · {c.section}**",
                    "",
                    "> " + " ".join(ev.quote.split()),
                    "",
                ]
            )
    if state.warnings:
        lines.extend(["## Extraction and retrieval notes", ""])
        lines.extend("- " + w for w in state.warnings)
    return "\n".join(lines) + "\n"


CSS = """
:root{--ink:#183b3a;--muted:#637571;--paper:#faf9f5;--line:#dce3dc;--accent:#176459;--wash:#eff3ed;--gold:#b88e4c}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}a{color:var(--accent);text-underline-offset:3px}.shell{max-width:1360px;margin:auto;display:grid;grid-template-columns:240px 1fr;min-height:100vh}aside{padding:48px 28px;border-right:1px solid var(--line);position:sticky;top:0;height:100vh}.brand{font-size:25px;letter-spacing:-1px;font-weight:750}.brand span{color:var(--gold)}.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:2px;font-weight:650;color:var(--muted)}nav{margin-top:45px}nav a{display:block;padding:10px 0;text-decoration:none;font-size:14px}aside p{font-size:12px;color:var(--muted);margin-top:45px}main{padding:48px 64px 64px;min-width:0}header{padding-bottom:32px;border-bottom:1px solid var(--line)}.top{display:flex;justify-content:space-between;align-items:center}.badge{font-size:11px;padding:5px 10px;border:1px solid var(--line);border-radius:20px;white-space:nowrap}h1{font:500 clamp(32px,4vw,50px)/1.12 Georgia,serif;letter-spacing:-1.4px;max-width:850px;margin:28px 0 18px}h2{font:500 27px/1.25 Georgia,serif;margin:0 0 18px}h3{font-size:15px;margin:12px 0}.meta{font-size:13px;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid var(--line);padding:24px 0;margin-bottom:32px}.stat b{font-size:24px;display:block}.stat span{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted)}section{margin-bottom:36px;scroll-margin-top:24px}.summary{font-size:19px;line-height:1.65;background:var(--wash);padding:28px 30px;border-left:3px solid var(--accent)}.claim{padding:0 0 19px;margin-bottom:17px;border-bottom:1px solid var(--line)}.claim>p{margin:0 0 8px}details{font-size:13px}summary{cursor:pointer;color:var(--accent);font-weight:550}blockquote{margin:12px 0;padding:14px 18px;background:white;border-left:2px solid var(--gold);white-space:pre-line}blockquote footer{margin-top:12px;color:var(--muted);font-size:11px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:35px}.note{font-size:13px;color:var(--muted)}.question{background:white;padding:22px 25px;border:1px solid var(--line);border-radius:8px;margin-bottom:18px}.question h3{margin:0 0 12px;font-size:17px}.status{color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:1px}.trace{display:flex;gap:7px;flex-wrap:wrap}.trace div{background:var(--wash);padding:9px 12px;font-size:12px;border-radius:4px}.trace small{display:block;color:var(--muted)}.notice{border:1px solid var(--line);border-radius:8px;padding:15px 20px;font-size:13px;color:var(--muted)}code{font-size:12px;overflow-wrap:anywhere}li{margin-bottom:9px}.foot{border-top:1px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}@media(max-width:850px){.shell{grid-template-columns:1fr}aside{position:static;height:auto;padding:20px 24px;border-right:0;border-bottom:1px solid var(--line)}aside nav,aside p{display:none}main{padding:28px 22px}.grid{grid-template-columns:1fr}.top{gap:12px;align-items:flex-start}.stats{gap:12px}}@media print{aside{display:none}.shell{display:block}main{padding:0}details{display:block}.question,.claim{break-inside:avoid}}
"""


def report_html(state: RunState, chunks: list[Chunk]) -> str:
    esc = html.escape
    paper, briefing = state.paper, state.briefing
    if not paper or not briefing:
        return f"<!doctype html><title>PaperTrail</title><p>{esc(state.error or state.status)}</p>"
    lookup = {c.id: c for c in chunks}

    def claim_html(claim):
        quotes = []
        for ev in claim.evidence:
            c = lookup[ev.chunk_id]
            quotes.append(
                f'<blockquote>{esc(ev.quote)}<footer><a href="{esc(paper.pdf_url)}#page={c.page}" target="_blank" rel="noopener noreferrer">Page {c.page} · {esc(c.section)} ↗</a><br><code>{c.id}</code></footer></blockquote>'
            )
        return f'<div class="claim"><p>{esc(claim.text)}</p><details><summary>Inspect evidence · {len(quotes)} passage(s)</summary>{"".join(quotes)}</details></div>'

    def section(title, claims, identity):
        return f'<section id="{identity}"><div class="eyebrow">Source-grounded briefing</div><h2>{title}</h2>{"".join(claim_html(c) for c in claims)}</section>'

    questions = "".join(f"<li>{esc(q)}</li>" for q in briefing.follow_up_questions)
    exchanges = "".join(
        f'<article class="question"><div class="status">{esc(ex.answer.status.replace("_", " "))} · {ex.elapsed_seconds:.1f}s</div><h3>{esc(ex.question)}</h3>{"".join(claim_html(c) for c in ex.answer.claims)}<p class="note">{esc(ex.answer.explanation)}</p></article>'
        for ex in state.exchanges
    )
    warnings = "".join(f"<li>{esc(w)}</li>" for w in state.warnings)
    trace = "".join(
        f"<div>{esc(e.node)}<small>{esc(e.status)} · {e.seconds:.1f}s</small></div>"
        for e in state.events
    )
    candidates = "".join(
        f'<li><a href="{esc(p.url)}">{esc(p.title)}</a> <span class="note">{p.relevance:.3f} ranking score</span></li>'
        for p in state.candidates
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><title>{esc(paper.title)} · PaperTrail</title><style>{CSS}</style></head><body><div class="shell"><aside><div class="brand">papertrail<span>.</span></div><div class="eyebrow">Research with receipts</div><nav><a href="#overview">01 / Overview</a><a href="#method">02 / Method & results</a><a href="#limitations">03 / Limitations</a><a href="#questions">04 / Questions & answers</a><a href="#execution">05 / Execution trail</a></nav><p>A saved, inspectable research session.<br><br>Expand any evidence link to read the exact quotation used.</p></aside><main><header id="overview"><div class="top"><div class="eyebrow">Research briefing / {esc(paper.arxiv_id)}</div><span class="badge">SAVED RUN · {esc(state.created_at[:10])}</span></div><h1>{esc(paper.title)}</h1><p class="meta">{esc(", ".join(paper.authors))}</p><p class="meta">Published {esc(paper.published[:10])} &nbsp; / &nbsp; <a href="{esc(paper.url)}">View on arXiv ↗</a></p></header><div class="stats"><div class="stat"><b>{state.pages}</b><span>Pages parsed</span></div><div class="stat"><b>{state.chunk_count}</b><span>Indexed passages</span></div><div class="stat"><b>{len(state.exchanges)}</b><span>Follow-up questions</span></div></div><section class="summary">{claim_html(briefing.summary)}</section>{section("The problem", [briefing.problem], "problem")}<div class="grid">{section("How it works", briefing.method, "method")}{section("What the paper reports", briefing.results, "results")}</div>{section("Limitations", briefing.limitations, "limitations")}<p class="notice">{esc(briefing.limitations_note)}</p><section id="questions"><h2>Questions, with evidence</h2>{exchanges}<details><summary>Suggested follow-up questions</summary><ul>{questions}</ul></details></section><section id="execution"><div class="eyebrow">Inspect the process</div><h2>The execution trail</h2><div class="trace">{trace}</div><p class="note">Model: {esc(state.model)} · Embeddings: {esc(state.embedding_model)}<br>PDF SHA-256: <code>{esc(state.pdf_sha256 or "")}</code></p><details><summary>Candidate papers</summary><ol>{candidates}</ol></details><details><summary>Parsing and retrieval notes ({len(state.warnings)})</summary><ul>{warnings or "<li>No parsing warnings were recorded.</li>"}</ul></details></section><div class="notice">This is an exported result of a real run, not a live chat. Run PaperTrail locally to process another paper or ask new questions. Quote checks establish provenance; they do not prove that every interpretation is correct.</div><p class="foot">PaperTrail · Session {state.id} · No paid API required · Sources remain available on arXiv</p></main></div></body></html>'''


def export_run(state: RunState, chunks: list[Chunk], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "markdown": destination / "briefing.md",
        "json": destination / "session.json",
        "html": destination / "report.html",
    }
    files["markdown"].write_text(markdown(state, chunks), encoding="utf-8")
    atomic_json(files["json"], state.model_dump(mode="json"))
    files["html"].write_text(report_html(state, chunks), encoding="utf-8")
    return files
