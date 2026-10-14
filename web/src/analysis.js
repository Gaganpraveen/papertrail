// Pure functions shared by the browser worker and the regression tests.
export const STOP = new Set(
  "a an the and or of to in on for is are was were be been by as at it its this that these those with from we our their they you your how what which who when where why does do did can could should would will about paper study research according tell me explain please using use used".split(
    " ",
  ),
);
export function tokens(text) {
  return (text.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g) || [])
    .filter((t) => t.length > 1 && !STOP.has(t))
    .map((t) => (t.length > 4 && t.endsWith("s") ? t.slice(0, -1) : t));
}
export const normalize = (s) => s.normalize("NFKC").replace(/\s+/g, " ").trim();
export const cosine = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export function sentences(text) {
  // Split only after punctuation followed by a new sentence; decimal points stay intact.
  return normalize(text)
    .split(/(?<=[.!?])\s+(?=[A-Z“"])/)
    .filter((s) => s.length >= 45 && s.length <= 1200 && /[.!?]["”]?$/.test(s));
}
export function readableSentence(text) {
  const letters = (text.match(/[a-z]/gi) || []).length;
  const symbols = (text.match(/[=∑Θ≤≥{}]/g) || []).length;
  return (
    letters / text.length > 0.65 &&
    !/@|copyright/i.test(text) &&
    /\b(is|are|was|were|have|has|can|could|will|would|we|our|propose|proposes|use|uses|used|using|reports?|achieves?|show|shows|found|find|gives?|depends?|compare|compares|outperforms?)\b/i.test(
      text,
    ) &&
    symbols < 3 &&
    /^[A-Z“"]/.test(text) &&
    !/^\d/.test(text)
  );
}
export function sectionName(text) {
  const t = text
    .replace(/^\s*[\d.IVX]+\s+/, "")
    .trim()
    .toLowerCase();
  if (t.length > 75 || t.split(/\s+/).length > 9 || /[.!?;]$/.test(t))
    return null;
  if (/^(abstract|summary)$/.test(t)) return "Overview";
  if (/^(introduction|background|motivation|problem.*)$/.test(t))
    return "Problem";
  if (
    /^(.*methods?|.*approach|.*architecture|.*algorithm|.*model|.*training|using random features)$/.test(
      t,
    )
  )
    return "Method";
  if (/^(.*experiments?|.*results?|.*evaluation|.*performance)$/.test(t))
    return "Results";
  if (/^(.*limitations?|.*discussion|.*conclusions?|future work)$/.test(t))
    return "Discussion";
  if (/^(references|bibliography|acknowledg.*)$/.test(t)) return "References";
  return null;
}
export function makeChunks(pages, documentId) {
  let section = "Overview";
  const chunks = [];
  for (const page of pages) {
    let lines = [];
    const flush = () => {
      const text = normalize(
        lines.join("\n").replace(/([a-z])-\n(?=[a-z])/g, "$1"),
      );
      lines = [];
      const parts = text.split(/(?<=[.!?])\s+(?=[A-Z“"])/);
      let buffer = "";
      const emit = () => {
        if (buffer.length >= 60)
          chunks.push({
            id: `${documentId.slice(0, 12)}-${chunks.length}`,
            page: page.number,
            section,
            text: buffer,
          });
        buffer = "";
      };
      for (const part of parts) {
        if (buffer && buffer.length + part.length > 1100) emit();
        buffer += (buffer ? " " : "") + part;
      }
      emit();
    };
    for (const line of page.lines) {
      const heading = sectionName(line);
      if (heading && section !== "References") {
        flush();
        section = heading;
      } else if (!/^\d+$/.test(line.trim())) lines.push(line);
    }
    flush();
  }
  return chunks;
}
export function lexicalScores(chunks, query) {
  const q = [...new Set(tokens(query))],
    docs = chunks.map((c) => tokens(c.text));
  const avg = docs.reduce((n, d) => n + d.length, 0) / Math.max(1, docs.length);
  return docs.map((d) =>
    q.reduce((score, term) => {
      const df = docs.filter((x) => x.includes(term)).length;
      const tf = d.filter((x) => x === term).length;
      const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
      return (
        score +
        (idf * tf * 2.2) / (tf + 1.2 * (0.25 + (0.75 * d.length) / (avg || 1)))
      );
    }, 0),
  );
}
export function retrieve(chunks, query, queryVector, vectors, limit = 8) {
  const lexical = lexicalScores(chunks, query);
  const dense = chunks.map((c, i) =>
    vectors?.[i] && queryVector ? cosine(vectors[i], queryVector) : 0,
  );
  const rankedL = chunks
    .map((_, i) => i)
    .sort((a, b) => lexical[b] - lexical[a]);
  const rankedD = chunks.map((_, i) => i).sort((a, b) => dense[b] - dense[a]);
  return chunks
    .map((c, i) => ({
      ...c,
      lexical: lexical[i],
      semantic: dense[i],
      rank:
        (lexical[i] > 0 ? 1 / (60 + rankedL.indexOf(i)) : 0) +
        (queryVector ? 1 / (60 + rankedD.indexOf(i)) : 0),
    }))
    .filter((c) => c.section !== "References")
    .sort((a, b) => b.rank - a.rank)
    .slice(0, limit);
}
const prompts = {
  Overview: "The main contribution and purpose of this research paper.",
  Problem: "The problem and motivation addressed by this study.",
  Method: "The proposed method model algorithm and experimental procedure.",
  Results:
    "The experimental results findings and evaluation of the proposed method.",
  Limitations:
    "Limitations weaknesses caveats future work and unresolved challenges.",
};
export { prompts };
export function selectBriefing(chunks, vectors, promptVectors) {
  const used = new Set();
  const result = {};
  for (const [section, query] of Object.entries(prompts)) {
    const ranked = retrieve(
      chunks,
      query,
      promptVectors?.[section],
      vectors,
      40,
    );
    if (section === "Overview")
      for (const c of chunks.filter(
        (c) => c.page === 1 && c.section !== "References",
      ))
        if (!ranked.some((r) => r.id === c.id)) ranked.push({ ...c, rank: 0 });
    const candidates = ranked.flatMap((c) =>
      sentences(c.text)
        .filter(readableSentence)
        .map((text, position) => ({
          text,
          page: c.page,
          chunkId: c.id,
          section: c.section,
          score:
            c.rank +
            (c.section === section ? 0.04 : 0) +
            (section === "Limitations" && c.section === "Discussion"
              ? 0.15
              : 0) +
            (section === "Method" &&
            /\b(we propose|we introduce|our method|we implement|consist)\b/i.test(
              text,
            )
              ? 0.15
              : 0) +
            (section === "Overview" && c.page === 1 ? 0.2 : 0) +
            (section === "Problem" && c.page <= 3 ? 0.06 : 0) +
            (section === "Method" &&
            /\b(propose|proposed|consist|using|used|algorithm|method|trained|training)\b/i.test(
              text,
            )
              ? 0.035
              : 0) +
            (section === "Results" &&
            /\b(accuracy|error|improv|outperform|achieve|compare|result|robust)\w*/i.test(
              text,
            )
              ? 0.04
              : 0) -
            position * 0.001,
        })),
    );
    const eligible = candidates.filter(
      (c) =>
        !used.has(c.text) &&
        !/^\[?\d+\]?\s/.test(c.text) &&
        (!["Limitations"].includes(section) ||
          /\b(limitations?|however|future|further work|remains?|drawback|challenges?|cannot|not yet|more work|may not|did not|overheads?|cost|increases)\b/i.test(
            c.text,
          )),
    );
    eligible.sort((a, b) => b.score - a.score);
    const selected = [];
    for (const c of eligible) {
      if (selected.length >= 3) break;
      const terms = new Set(tokens(c.text));
      if (
        selected.some(
          (x) =>
            tokens(x.text).filter((t) => terms.has(t)).length /
              Math.max(1, terms.size) >
            0.7,
        )
      )
        continue;
      selected.push(c);
      used.add(c.text);
    }
    result[section] = selected;
  }
  return result;
}
export function answerEvidence(chunks, vectors, question, queryVector) {
  const q = tokens(question);
  const ranked = retrieve(chunks, question, queryVector, vectors, 8);
  const candidates = ranked
    .flatMap((c) =>
      sentences(c.text)
        .filter(readableSentence)
        .map((text) => {
          const t = new Set(tokens(text)),
            overlap = q.filter((w) => t.has(w)).length / Math.max(1, q.length);
          return {
            text,
            page: c.page,
            chunkId: c.id,
            overlap,
            semantic: c.semantic,
            score: overlap + c.rank,
          };
        }),
    )
    .filter((c) => c.overlap >= 0.45 && (c.semantic >= 0.15 || !queryVector));
  // Numeric questions require a matching value type, not a topical passage alone.
  const numericQuestion =
    /\b(cost|budget|dollars?|price|accuracy|percent|how many|how much)\b/i.test(
      question,
    );
  const financial = /\b(cost|budget|dollars?|price)\b/i.test(question);
  const safe = candidates.filter(
    (c) =>
      (!numericQuestion || /\d/.test(c.text)) &&
      (!financial || /\$|\b(USD|dollars?|budget|costs?|price)\b/i.test(c.text)),
  );
  safe.sort((a, b) => b.score - a.score);
  const seen = new Set();
  const evidence = safe
    .filter((c) => {
      if (seen.has(c.text)) return false;
      seen.add(c.text);
      return true;
    })
    .slice(0, 3);
  return {
    status: evidence.length ? "evidence_found" : "insufficient_evidence",
    evidence,
    explanation: evidence.length
      ? "These source passages are relevant to your question. They are quotations, not a generated interpretation."
      : "The retrieved passages do not provide enough matching evidence. Try a more specific question or inspect the document.",
  };
}
export function measurementProse(text) {
  // Repeated bare ticks usually come from a chart axis or a flattened table.
  return !/(?:\d+(?:\.\d+)?\s+){4,}/.test(text);
}
export function measurements(chunks) {
  const rows = [];
  const seen = new Set();
  const pattern =
    /\b(\d+(?:\.\d+)?)\s*(%|percent\b|BLEU\b|seconds?\b|minutes?\b|hours?\b|days?\b|milliseconds?\b|ms\b|GB\b|MB\b)/gi;
  for (const c of chunks.filter((c) => c.section !== "References"))
    for (const sentence of sentences(c.text).filter(measurementProse))
      for (const m of sentence.matchAll(pattern)) {
        const unit = m[2].toLowerCase().replace(/^percent$/, "%");
        const key = `${m[1]}:${unit}:${c.page}`;
        if (seen.has(key)) continue;
        seen.add(key);
        rows.push({
          id: `m${rows.length}`,
          value: Number(m[1]),
          unit,
          quote: sentence,
          page: c.page,
          chunkId: c.id,
          label: m[0],
        });
      }
  return rows.slice(0, 80);
}
export function validateComparison(items) {
  if (items.length < 2 || items.length > 6)
    return "Select between two and six measurements.";
  if (new Set(items.map((x) => x.unit)).size !== 1)
    return "Choose measurements with the same unit. Different units cannot share this chart.";
  return null;
}
