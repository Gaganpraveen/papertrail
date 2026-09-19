"""Build the PaperTrail explanatory guide with ReportLab native vectors.

Run from the repository: .venv/bin/python scripts/build_guide.py
The guide deliberately contains no run-specific timing or evaluation claims.
"""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output/pdf/papertrail-guide.pdf"
W, H = 595.276, 841.89
M, WIDTH = 44, W - 88
CREAM = colors.HexColor("#F7F3E8")
TEAL = colors.HexColor("#145D5A")
INK = colors.HexColor("#173533")
MUTED = colors.HexColor("#556C67")
PALE = colors.HexColor("#E5ECE4")
LINE = colors.HexColor("#CDD6CA")
GOLD = colors.HexColor("#B78546")
WHITE = colors.HexColor("#FFFFFF")

# Embed available macOS fonts for consistent preview and portable reading.
# Core PDF fonts remain a functional fallback on other systems.
for font_name, filename in [
    ("Helvetica", "Arial.ttf"),
    ("Helvetica-Bold", "Arial Bold.ttf"),
    ("Times-Roman", "Times New Roman.ttf"),
    ("Courier", "Courier New.ttf"),
]:
    font_path = Path("/System/Library/Fonts/Supplemental") / filename
    if font_path.exists():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
pdfmetrics.registerFontFamily("Helvetica", normal="Helvetica", bold="Helvetica-Bold")


def sty(name="body", size=10.2, leading=14.3, color=INK, bold=False, align=TA_LEFT):
    return ParagraphStyle(
        name,
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        allowWidows=0,
        allowOrphans=0,
    )


BODY = sty()
SMALL = sty("small", 8.7, 12.1, MUTED)
CAP = sty("cap", 8.3, 11.5, TEAL, True)
TABLE = sty("table", 9.2, 12.8)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.9, leading=13, textColor=INK)


def p(c, text, x, top, width, style=BODY):
    """Draw paragraph at a top offset. Return the next top offset."""
    para = Paragraph(text, style)
    _, height = para.wrap(width, H)
    if top + height > H - 58:
        raise ValueError(f"Overflow on page {c.getPageNumber()}: {text[:80]}")
    para.drawOn(c, x, H - top - height)
    return top + height


def label(c, text, top, x=M):
    return p(c, text.upper(), x, top, WIDTH, CAP)


def h2(c, text, top):
    c.setFillColor(TEAL)
    c.setFont("Times-Roman", 20)
    c.drawString(M, H - top - 20, text)
    return top + 32


def page(c, number, section, title, subtitle):
    c.setFillColor(CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(M, H - 47, 23, 3, fill=1, stroke=0)
    p(c, "PAPERTRAIL / FIELD GUIDE", M + 32, 38, 250, CAP)
    p(
        c,
        f"{number:02d} / {section.upper()}",
        W - M - 150,
        38,
        150,
        sty("topright", 8.3, 11.5, TEAL, True, 2),
    )
    c.setFont("Times-Roman", 30)
    c.setFillColor(INK)
    c.drawString(M, H - 99, title)
    y = p(c, subtitle, M, 112, WIDTH, sty("intro", 11.1, 15.6, MUTED)) + 22
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(M, 49, W - M, 49)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(M, 30, "GAGAN P  /  INTERNSHIP PREPARATION")
    c.drawRightString(W - M, 30, f"PAPERTRAIL   {number:02d} / 07")
    return y


def callout(c, title, text, top, fill=PALE):
    title_p = Paragraph(title, sty("calltitle", 10.5, 14, TEAL, True))
    body_p = Paragraph(text, BODY)
    th = title_p.wrap(WIDTH - 32, H)[1]
    bh = body_p.wrap(WIDTH - 32, H)[1]
    height = 16 + th + 6 + bh + 16
    if top + height > H - 58:
        raise ValueError("Callout overflow")
    c.setFillColor(fill)
    c.roundRect(M, H - top - height, WIDTH, height, 7, fill=1, stroke=0)
    title_p.drawOn(c, M + 16, H - top - 16 - th)
    body_p.drawOn(c, M + 16, H - top - 16 - th - 6 - bh)
    return top + height + 18


def table(c, headers, rows, widths, top):
    x0, y = M, top
    for ri, row in enumerate([headers] + rows):
        style = sty("thead", 8.5, 11.4, WHITE, True) if ri == 0 else TABLE
        ps = [Paragraph(t, style) for t in row]
        hs = [q.wrap(w - 20, H)[1] for q, w in zip(ps, widths, strict=True)]
        height = max(hs) + 18
        if y + height > H - 62:
            raise ValueError("Table overflow")
        c.setFillColor(TEAL if ri == 0 else (WHITE if ri % 2 else PALE))
        c.rect(x0, H - y - height, sum(widths), height, fill=1, stroke=0)
        x = x0
        for q, w, ph in zip(ps, widths, hs, strict=True):
            q.drawOn(c, x + 10, H - y - 9 - ph)
            x += w
        y += height
    return y + 18


def bullet(c, title, text, top):
    c.setFillColor(GOLD)
    c.circle(M + 3, H - top - 7, 2.2, fill=1, stroke=0)
    return p(c, f"<b>{title}</b> {text}", M + 15, top, WIDTH - 15) + 11


def arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(TEAL)
    c.setFillColor(TEAL)
    c.setLineWidth(1)
    c.line(x1, H - y1, x2, H - y2)
    s = 3.5
    path = c.beginPath()
    path.moveTo(x2, H - y2)
    if x1 == x2:
        path.lineTo(x2 - s, H - y2 + s * 1.8)
        path.lineTo(x2 + s, H - y2 + s * 1.8)
    else:
        sign = 1 if x2 > x1 else -1
        path.lineTo(x2 - sign * s * 1.8, H - y2 - s)
        path.lineTo(x2 - sign * s * 1.8, H - y2 + s)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def node(c, x, y, w, title, sub, dark=False):
    c.setFillColor(TEAL if dark else PALE)
    c.roundRect(x, H - y - 55, w, 55, 5, fill=1, stroke=0)
    p(
        c,
        title,
        x + 10,
        y + 10,
        w - 20,
        sty("node", 10.3, 13, WHITE if dark else INK, True, TA_CENTER),
    )
    p(
        c,
        sub,
        x + 7,
        y + 29,
        w - 14,
        sty("sub", 7.8, 10, WHITE if dark else MUTED, False, TA_CENTER),
    )


def codebox(c, lines, top):
    text = "<br/>".join(escape(line).replace(" ", "&#160;") for line in lines)
    q = Paragraph(text, CODE)
    height = q.wrap(WIDTH - 24, H)[1] + 22
    if top + height > H - 58:
        raise ValueError("Code overflow")
    c.setFillColor(PALE)
    c.roundRect(M, H - top - height, WIDTH, height, 5, fill=1, stroke=0)
    q.drawOn(c, M + 12, H - top - 11 - (height - 22))
    return top + height + 15


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(DEST), pagesize=(W, H))
    c.setTitle("PaperTrail - Build, explain and demonstrate")
    c.setAuthor("PaperTrail / prepared for Gagan P")
    c.setSubject("Architecture, evidence, interview preparation and reflection draft")

    # 1. Intent and mental model.
    y = page(
        c,
        1,
        "Purpose",
        "Read the paper. Keep the trail.",
        "An explanation and interview guide for Gagan P's AI/ML internship submission.",
    )
    y = callout(
        c,
        "A 30-second description to practise",
        "PaperTrail turns a research topic or arXiv ID into a briefing for one selected paper, then answers follow-up questions from retrieved source passages. Each accepted factual claim carries quotations and page references. The local application saves its state so a failed run can resume, and exports an evidence reader that reviewers can inspect.",
        y,
    )
    y = h2(c, "What a user actually gets", y)
    y = table(
        c,
        ["INPUT", "PROCESS", "OUTPUT"],
        [
            [
                "A topic, arXiv ID or arXiv URL",
                "Find and select one paper; fetch, parse and index it",
                "Summary, problem, method, results and reported limitations",
            ],
            [
                "A question about a saved paper",
                "Retrieve relevant passages; generate and check claims",
                "A cited answer or an explicit insufficient-evidence response",
            ],
            [
                "An existing session ID",
                "Read or export its durable saved state",
                "Markdown, JSON and a portable HTML evidence reader",
            ],
        ],
        [142, 180, WIDTH - 322],
        y,
    )
    y = h2(c, "Three ideas to keep distinct", y)
    y = bullet(
        c,
        "RAG:",
        "retrieve text relevant to a question before asking a model to answer. This changes the evidence supplied at inference time; it does not train the model.",
        y,
    )
    y = bullet(
        c,
        "State graph:",
        "a finite set of named steps and allowed transitions. Python owns the workflow; the model returns structured content within that workflow.",
        y,
    )
    y = bullet(
        c,
        "Provenance:",
        "a route from a claim to the exact source passage. It helps a person audit an answer, but it does not prove that the interpretation is correct.",
        y,
    )
    y = callout(
        c,
        "Scope you can describe precisely",
        "One paper per session. Qwen3.5 4B through Ollama, CPU embeddings, Qdrant local storage and SQLite checkpoints. The CLI and local browser UI share one agent. HTML exports and the public Pages demo display saved results.",
        y,
    )
    c.showPage()

    # 2. Graph, persistence and recovery.
    y = page(
        c,
        2,
        "Architecture",
        "A workflow you can inspect",
        "The graph makes progress and failure explicit. Each successful node advances the saved checkpoint.",
    )
    label(c, "Main pipeline - executable transitions in graph.py", y)
    dy = y + 24
    gap, nw = 15, (WIDTH - 30) / 3
    xs = [M, M + nw + gap, M + 2 * (nw + gap)]
    for i, (title, sub) in enumerate(
        [
            ("Understand", "normalize query or ID"),
            ("Retrieve", "arXiv candidate metadata"),
            ("Select", "rank; pin one revision"),
        ]
    ):
        node(c, xs[i], dy, nw, title, sub)
    arrow(c, xs[0] + nw, dy + 27, xs[1] - 2, dy + 27)
    arrow(c, xs[1] + nw, dy + 27, xs[2] - 2, dy + 27)
    arrow(c, xs[2] + nw / 2, dy + 55, xs[2] + nw / 2, dy + 78)
    row2 = dy + 80
    for i, (title, sub) in enumerate(
        [
            ("Index", "BGE + persistent Qdrant"),
            ("Parse", "pages, sections, chunks"),
            ("Fetch", "bounded PDF + checksum"),
        ]
    ):
        node(c, xs[i], row2, nw, title, sub)
    arrow(c, xs[2], row2 + 27, xs[1] + nw + 2, row2 + 27)
    arrow(c, xs[1], row2 + 27, xs[0] + nw + 2, row2 + 27)
    arrow(c, xs[0] + nw / 2, row2 + 55, xs[0] + nw / 2, row2 + 78)
    row3 = row2 + 80
    for i, (title, sub) in enumerate(
        [
            ("Brief", "draft + support review"),
            ("Validate", "final provenance checks"),
            ("Ready", "ask / inspect / export"),
        ]
    ):
        node(c, xs[i], row3, nw, title, sub, i == 2)
    arrow(c, xs[0] + nw, row3 + 27, xs[1] - 2, row3 + 27)
    arrow(c, xs[1] + nw, row3 + 27, xs[2] - 2, row3 + 27)
    y = row3 + 69
    y = (
        p(
            c,
            "<b>QA loop:</b> Ready -> retrieve passages -> generate and validate answer -> save exchange -> Ready.",
            M,
            y,
            WIDTH,
            SMALL,
        )
        + 19
    )
    y = table(
        c,
        ["STATE OR ARTIFACT", "WHY IT EXISTS"],
        [
            [
                "SQLite RunState",
                "Next node, selected paper, model names, errors, event timings, briefing and conversation history",
            ],
            [
                "Original PDF + parsed.json",
                "Source checksum, page/section text, stable passage IDs and offsets",
            ],
            [
                "Qdrant local collection",
                "Reusable passage vectors and payloads tied to the PDF, embedding model and parser version",
            ],
        ],
        [163, WIDTH - 163],
        y,
    )
    y = callout(
        c,
        "What happens when generation fails?",
        "The saved node remains the failed step. After fixing the cause, resume re-enters that step. Completed earlier nodes stay completed. Stable vector IDs make repeated index upserts idempotent. This is retry-safe progress, not a promise that external work happens exactly once.",
        y,
    )
    c.showPage()

    # 3. Evidence design.
    y = page(
        c,
        3,
        "Evidence",
        "From relevant text to a claim",
        "Retrieval decides what the model can see. Validation decides which drafts the application will accept.",
    )
    y = h2(c, "1 / Retrieve complementary evidence", y)
    y = (
        p(
            c,
            "<b>Dense retrieval</b> uses BGE-small-en-v1.5 through FastEmbed on CPU to find related meaning. <b>BM25</b> rewards exact terms, which helps with method names and technical vocabulary. Reciprocal-rank fusion combines ranks using <b>1 / (60 + rank)</b> from each list; similarity and keyword scores are not comparable probabilities.",
            M,
            y,
            WIDTH,
        )
        + 14
    )
    y = (
        p(
            c,
            "Chunks stay within one page and retain section labels. Briefing retrieval seeks problem, method, results and limitations. QA selects diverse passages and suppresses heavy overlap. References are excluded unless requested. A follow-up can reuse the previous question for retrieval; earlier generated answers are never source evidence.",
            M,
            y,
            WIDTH,
        )
        + 19
    )
    y = h2(c, "2 / Let code own the quotations", y)
    y = table(
        c,
        ["STEP", "WHAT IT CHECKS OR CONTROLS"],
        [
            [
                "Evidence aliases",
                "Python splits retrieved text into sentence-like spans and assigns short IDs such as E1. The model selects IDs and writes a claim.",
            ],
            [
                "Source resolution",
                "Python resolves those aliases and copies quotation text from the source registry. The model does not rewrite the saved quotations.",
            ],
            [
                "Deterministic checks",
                "Citations must resolve to allowed passages; quotations must match normalized source text; numbers in a claim must appear in its evidence.",
            ],
            [
                "Support review",
                "A second LLM pass screens claim support. Rejected briefing prose becomes labeled source excerpts; rejected limitations are removed. Copying text does not prove relevance.",
            ],
            [
                "Bounded recovery",
                "Schema/provenance failures get one repair attempt, then fail closed. QA has no excerpt fallback: rejected support becomes an evidence-validation abstention.",
            ],
        ],
        [119, WIDTH - 119],
        y,
    )
    y = callout(
        c,
        "The honest reliability statement",
        "These layers reduce specific failures; they do not guarantee truth. A real quotation can be misunderstood, and the same local model can miss an error during review. Number matching does not verify units or comparisons. An abstention means the retrieved evidence was insufficient, not that the whole paper contains no answer.",
        y,
    )
    c.showPage()

    # 4. Rehearsal.
    y = page(
        c,
        4,
        "Demonstration",
        "Make the evidence visible",
        "Use the README for installation. Run papertrail web for a live local browser interface, or use the CLI commands below. Generation time depends on the machine.",
    )
    y = label(c, "Preparation commands", y) + 8
    y = codebox(
        c,
        [
            "papertrail doctor --warmup",
            'papertrail digest "1706.03762"',
            "papertrail sessions",
        ],
        y,
    )
    y = (
        p(
            c,
            "Install the configured local model as described in the README, and start Ollama if needed. Initial model downloads and paper fetching need internet access. Copy the real completed session ID from the output. The arXiv ID above is a reproducible input example; no particular result is promised.",
            M,
            y,
            WIDTH,
            SMALL,
        )
        + 16
    )
    y = label(c, "Replace SESSION and CHUNK with real IDs", y) + 8
    y = codebox(
        c,
        [
            "papertrail show SESSION",
            'papertrail ask SESSION "What problem does this paper address?"',
            'papertrail ask SESSION "What is the main method?"',
            'papertrail ask SESSION "What experiments support the method?"',
            "papertrail inspect SESSION CHUNK",
            "papertrail export SESSION --output exports/demo",
        ],
        y,
    )
    y = h2(c, "A compact walkthrough", y)
    y = bullet(
        c,
        "Orient:",
        "show the original input, selected paper and session. Point to the state graph in the README.",
        y,
    )
    y = bullet(
        c,
        "Audit:",
        "open one accepted claim, read its quotation and follow its page reference into the PDF. Explain what the evidence supports.",
        y,
    )
    y = bullet(
        c,
        "Question:",
        "ask one specific follow-up. Read the actual result, including an abstention if that is what the system returns.",
        y,
    )
    y = bullet(
        c,
        "Persist:",
        "exit and reopen the same session. Explain that show/export read saved results, while ask/chat invoke retrieval and generation.",
        y,
    )
    y = callout(
        c,
        "When something fails during a demo",
        "Read the error and saved session ID. For an interrupted pipeline, use papertrail resume SESSION after addressing the cause. If QA generation fails on a ready session, retry the ask command. Keep a clearly labeled export of a real saved run available for the walkthrough.",
        y,
    )
    c.showPage()

    # 5. Decisions and limits.
    y = page(
        c,
        5,
        "Judgment",
        "Defend the choices and their cost",
        "The strongest explanation connects a requirement to a design choice, then acknowledges the tradeoff.",
    )
    y = table(
        c,
        ["CHOICE", "REASON", "COST / BOUNDARY"],
        [
            [
                "Explicit Python graph",
                "Visible state, bounded transitions and simple recovery for a short pipeline",
                "No framework scheduler, distributed execution or complex branching",
            ],
            [
                "Qwen + Ollama",
                "Local structured generation without paid API credentials",
                "Hardware-dependent latency; small-model reasoning and context limits",
            ],
            [
                "FastEmbed + Qdrant local",
                "CPU embeddings and durable vector search without a separate server",
                "One local writer; not a concurrent multi-user service",
            ],
            [
                "SQLite checkpoints",
                "Atomic, inspectable state and persistent question history",
                "Application-level recovery; local artifacts still need to remain intact",
            ],
            [
                "Dense + BM25 fusion",
                "Semantic matches plus exact scientific terminology",
                "Additional retrieval logic; usefulness must be measured, not assumed",
            ],
            [
                "pdfplumber + page chunks",
                "Transparent extraction with page provenance and modest setup",
                "Heuristic columns/sections; OCR, figures and reliable tables are outside scope",
            ],
            [
                "One paper per session",
                "Clear provenance and a bounded, reviewable workflow",
                "No cross-paper synthesis or exhaustive literature search",
            ],
            [
                "Local browser UI",
                "Live forms and background jobs reuse the same Python agent",
                "Loopback only; job status is in memory. Public Pages serves saved results.",
            ],
        ],
        [114, 191, WIDTH - 305],
        y,
    )
    y = h2(c, "Where reliability can still break", y)
    y = (
        p(
            c,
            "Search can select a weak candidate. Parsing can lose an equation or table relationship. Retrieval can miss the decisive passage. Generation can overstate a supported fact. The review model can accept the same mistake. Inspect the failure stage before changing prompts or choosing a larger model.",
            M,
            y,
            WIDTH,
        )
        + 17
    )
    y = callout(
        c,
        "A credible next step",
        "Build a larger, held-out question set across papers and layouts. Label the passages needed to answer each question, compare dense/BM25/hybrid retrieval, then review answer support and abstention quality. Improve the component responsible for measured errors before adding more infrastructure.",
        y,
    )
    c.showPage()

    # 6. Interview questions.
    y = page(
        c,
        6,
        "Interview",
        "Answers you can explain in your words",
        "Practise the reasoning, then point to code or a real artifact. Do not claim results you have not verified.",
    )
    qa = [
        (
            "Why is this a graph instead of one prompt?",
            "Each stage has a clear input, output and failure point. The saved next-node field tells the application where to resume. The LLM does not choose arbitrary tools or control execution.",
        ),
        (
            "Why do you need a vector database and SQLite?",
            "They hold different data. Qdrant retrieves passage vectors and payloads. SQLite stores application state, selected metadata and QA history. The original PDF and parsed text remain separate files.",
        ),
        (
            "How is RAG different from fine-tuning?",
            "RAG supplies relevant paper text when a question is asked. No model weights are changed here. Fine-tuning would change weights using training examples and would not remove the need for source evidence.",
        ),
        (
            "Does a citation mean an answer is correct?",
            "It proves a narrower point: the cited text is traceable to the source. Code checks the provenance, and a second model pass screens the meaning. A person still needs to assess whether the claim faithfully follows from the quotation.",
        ),
        (
            "How do you handle follow-up questions?",
            "Questions and answers persist in the session. A simple referential-query rule can append the previous question to retrieval. The model sees recent questions as context, but past generated answers cannot become factual evidence.",
        ),
        (
            "How would you evaluate this fairly?",
            "Separate retrieval from answer quality. Check whether the required source passages are retrieved, then check claim support and appropriate abstention. Use held-out questions and publish actual results, including cases where hybrid retrieval loses.",
        ),
        (
            "What would you change for production?",
            "First establish an evaluation baseline. Then separate jobs from the interface, use concurrent-safe storage, add access controls and resource limits, and monitor retrieval and validation failures. The current application is deliberately local and single-user.",
        ),
    ]
    for title, text in qa:
        y = p(c, title, M, y, WIDTH, sty("question", 11, 14.7, TEAL, True)) + 5
        y = p(c, text, M, y, WIDTH) + 17
    y = p(
        c,
        "<b>Code reading route:</b> schema.py -> graph.py -> sources.py / parsing.py -> retrieval.py -> llm.py / grounding.py -> storage.py -> cli.py / rendering.py.",
        M,
        y,
        WIDTH,
        SMALL,
    )
    c.showPage()

    # 7. Reflection and references.
    y = page(
        c,
        7,
        "Reflection",
        "A four-minute story with evidence",
        "This is a preparation outline. Adapt the first-person draft in docs/reflection-script.md to match your actual contribution and experience.",
    )
    y = table(
        c,
        ["TARGET", "WHAT TO COVER", "WHAT TO SHOW"],
        [
            [
                "0:00-0:30",
                "Problem, intended reader and one-paper scope",
                "Repository title and a saved briefing",
            ],
            [
                "0:30-1:10",
                "Finite graph, typed state and recovery",
                "README state graph and a session record",
            ],
            [
                "1:10-2:00",
                "Retrieval, evidence aliases, code-owned quotes and support review",
                "One claim beside its source quotation",
            ],
            [
                "2:00-2:45",
                "A follow-up and durable session history",
                "Actual CLI output or a clearly labeled saved run",
            ],
            [
                "2:45-3:30",
                "Tradeoff, limitation and measured improvement plan",
                "One real issue you can explain",
            ],
            [
                "3:30-3:50",
                "Your own contribution and what you learned",
                "Concise closing statement in your own words",
            ],
        ],
        [76, 259, WIDTH - 335],
        y,
    )
    y = (
        p(
            c,
            "The target leaves ten seconds of buffer. Rehearse with a timer; these are presentation allocations, not observed application latency. Show a real result rather than promising a particular answer.",
            M,
            y,
            WIDTH,
            SMALL,
        )
        + 19
    )
    y = h2(c, "Before you submit", y)
    y = (
        p(
            c,
            "Confirm the public repository opens, setup instructions match a fresh environment, example QA comes from a saved run, and any reported checks link to actual evidence. Record the reflection yourself. Keep secrets, downloaded PDFs, local databases and model weights out of the repository.",
            M,
            y,
            WIDTH,
        )
        + 20
    )
    y = h2(c, "Where to read next", y)
    refs = [
        (
            "PaperTrail source, README and technical notes",
            "https://github.com/Gaganpraveen/papertrail",
        ),
        (
            "arXiv API user manual - metadata search interface",
            "https://info.arxiv.org/help/api/user-manual.html",
        ),
        (
            "Ollama structured outputs - JSON schema responses",
            "https://docs.ollama.com/capabilities/structured-outputs",
        ),
        ("Qdrant Python client - local mode", "https://github.com/qdrant/qdrant-client#local-mode"),
    ]
    for title, url in refs:
        y = (
            p(
                c,
                f'<b>{title}</b><br/><link href="{url}" color="#145D5A">{url}</link>',
                M,
                y,
                WIDTH,
                SMALL,
            )
            + 10
        )
    y = p(
        c,
        "Implementation descriptions were checked against the local source. External links are official reference material; they do not certify this application's behavior. This guide intentionally makes no benchmark, test-count or demo-duration claims.",
        M,
        y + 2,
        WIDTH,
        sty("note", 8, 11, MUTED),
    )
    c.showPage()
    c.save()
    print(DEST)


if __name__ == "__main__":
    main()
