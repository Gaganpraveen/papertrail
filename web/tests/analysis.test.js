import test from "node:test";
import assert from "node:assert/strict";
import {
  sentences,
  makeChunks,
  retrieve,
  answerEvidence,
  measurements,
  validateComparison,
  selectBriefing,
  sectionName,
} from "../src/analysis.js";
import { readingLines } from "../src/layout.js";
import { arxivId, safeUrl } from "../src/search.js";
const passage =
  "After a large number of trees is generated, they vote for the most popular class.";
const chunks = [
  { id: "forest-1", page: 2, section: "Method", text: passage },
  {
    id: "forest-2",
    page: 3,
    section: "Results",
    text: "The model reports an error of 12.7 percent on the held-out test dataset.",
  },
  {
    id: "forest-3",
    page: 4,
    section: "References",
    text: "A different paper reports a total training budget of 500 dollars.",
  },
];
test("exact arXiv identity and version are preserved", () => {
  for (const input of [
    "1810.04805v2",
    "https://arxiv.org/abs/1810.04805v2",
    "https://arxiv.org/pdf/1810.04805v2.pdf",
  ])
    assert.equal(arxivId(input), "1810.04805v2");
  assert.equal(arxivId("hep-th/9901001v3"), "hep-th/9901001v3");
  assert.equal(arxivId("hello 1810.04805"), null);
  assert.equal(arxivId("https://evil.test/1810.04805"), null);
});
test("only HTTPS source links are allowed", () => {
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("file:///etc/passwd"), null);
  assert.equal(
    safeUrl("https://arxiv.org/abs/1810.04805"),
    "https://arxiv.org/abs/1810.04805",
  );
});
test("sentence extraction preserves decimal numbers and refuses truncated fragments", () => {
  const text =
    "The model reports an error of 12.7 percent on the held-out test dataset. The next sentence is unfinished and should never appear as a complete quotation";
  assert.deepEqual(sentences(text), [chunks[1].text]);
});
test("chunks preserve sentence boundaries and page provenance", () => {
  const p = [{ number: 2, lines: ["Methods", ...Array(18).fill(passage)] }];
  const result = makeChunks(p, "abcdef1234567890");
  assert.ok(result.length > 1);
  assert.ok(
    result.every(
      (c) => c.page === 2 && c.section === "Method" && c.text.endsWith("."),
    ),
  );
  assert.ok(result.every((c) => c.id.startsWith("abcdef123456-")));
});
test("headings do not consume ordinary prose", () => {
  assert.equal(sectionName("1.1 Introduction"), "Problem");
  assert.equal(sectionName("We use this model to generate predictions."), null);
});
test("retrieval excludes bibliography and answers only from the selected index", () => {
  const result = answerEvidence(
    chunks,
    null,
    "What class do the trees vote for?",
    null,
  );
  assert.equal(result.status, "evidence_found");
  assert.equal(result.evidence[0].text, passage);
  assert.equal(result.evidence[0].page, 2);
  assert.equal(
    answerEvidence([chunks[1]], null, "What class do the trees vote for?", null)
      .status,
    "insufficient_evidence",
  );
  assert.ok(
    retrieve(chunks, "budget dollars", null, null).every(
      (c) => c.section !== "References",
    ),
  );
});
test("unsupported financial question abstains instead of using an unrelated number", () => {
  assert.equal(
    answerEvidence(
      chunks,
      null,
      "What was the total training budget in dollars?",
      null,
    ).status,
    "insufficient_evidence",
  );
});
test("measurements retain decimals, units and quotations", () => {
  const ms = measurements(chunks);
  assert.equal(ms.length, 1);
  assert.equal(ms[0].value, 12.7);
  assert.equal(ms[0].unit, "%");
  assert.equal(ms[0].quote, chunks[1].text);
  assert.equal(ms[0].page, 3);
});
test("comparison refuses mixed units and insufficient values", () => {
  assert.ok(validateComparison([{ unit: "%" }]));
  assert.ok(validateComparison([{ unit: "%" }, { unit: "seconds" }]));
  assert.equal(validateComparison([{ unit: "%" }, { unit: "%" }]), null);
});
test("briefing exposes missing limitations without inventing them", () => {
  assert.deepEqual(selectBriefing(chunks, null, null).Limitations, []);
});
test("full-width prose is not split into columns by equation fragments", () => {
  const item = (str, x, y, width) => ({
    str,
    transform: [1, 0, 0, 12, x, y],
    width,
  });
  const words = [];
  for (let i = 0; i < 14; i++) {
    words.push(
      item(
        "A complete line of prose extending across the page.",
        60,
        700 - i * 20,
        490,
      ),
    );
    words.push(
      item("x", 100, 690 - i * 20, 5),
      item("y", 320, 690 - i * 20, 5),
    );
  }
  const lines = readingLines(words, 612);
  assert.equal(lines[0], "A complete line of prose extending across the page.");
  assert.equal(lines[1], "x y");
});
test("two-column abstracts with different line spacing stay in reading order", () => {
  const item = (str, x, y, width) => ({
    str,
    transform: [1, 0, 0, 12, x, y],
    width,
  });
  const words = [
    item("Paper title", 180, 750, 200),
    item("Abstract", 150, 620, 50),
  ];
  for (let i = 0; i < 20; i++) {
    words.push(
      item(
        "Left column contains the abstract sentence.",
        70,
        600 - i * 12,
        220,
      ),
    );
    words.push(
      item("Right column contains introduction prose.", 310, 620 - i * 13, 220),
    );
  }
  const lines = readingLines(words, 595);
  assert.equal(lines[0], "Paper title");
  assert.equal(lines[1], "Abstract");
  assert.equal(lines[2], "Left column contains the abstract sentence.");
  assert.equal(lines[22], "Right column contains introduction prose.");
});

test("chart-axis ticks are not reported as experimental measurements", () => {
  assert.deepEqual(
    measurements([
      {
        id: "axis",
        page: 8,
        section: "Results",
        text: "The plot displays 80.0 60.0 40.0 20.0 0.0 25.0 50.0 75.0 100.0 % steps in this chart.",
      },
    ]),
    [],
  );
});
