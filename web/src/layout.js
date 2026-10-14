export function readingLines(items, width) {
  const words = items
    .filter((x) => x.str?.trim())
    .map((x) => ({
      text: x.str,
      x: x.transform[4],
      y: x.transform[5],
      w: x.width,
      h: Math.abs(x.transform[3]),
    }));
  // Column text need not share baselines (abstracts often use a different leading).
  // Use sustained prose on both sides and reject a split through full-width body lines.
  const prose = words.filter(
    (x) => x.text.length >= 25 && x.w >= 80 && x.y > 50,
  );
  const left = prose.filter(
    (x) => x.x < width * 0.48 && x.x + x.w < width * 0.53,
  );
  const right = prose.filter((x) => x.x > width * 0.49 && x.x < width * 0.72);
  const boundary =
    Math.max(...left.map((x) => x.y), ...right.map((x) => x.y)) + 24;
  const crossing = prose.filter(
    (x) => x.y < boundary && x.x < width * 0.3 && x.x + x.w > width * 0.68,
  );
  const bodyStart = crossing.length
    ? Math.min(...crossing.map((x) => x.y)) - 3
    : boundary;
  const split =
    left.filter((x) => x.y < bodyStart).length >= 8 &&
    right.filter((x) => x.y < bodyStart).length >= 8;
  const start = split ? bodyStart : Infinity;
  const groups = split
    ? [
        words.filter((x) => x.y > start + 3),
        words.filter((x) => x.y <= start + 3 && x.x < width * 0.49),
        words.filter((x) => x.y <= start + 3 && x.x >= width * 0.49),
      ]
    : [words];
  return groups.flatMap((group) => {
    const rows = [];
    for (const word of group.sort((a, b) => b.y - a.y || a.x - b.x)) {
      let row = rows.find((r) => Math.abs(r.y - word.y) < 3);
      if (!row) {
        row = { y: word.y, words: [] };
        rows.push(row);
      }
      row.words.push(word);
    }
    return rows
      .sort((a, b) => b.y - a.y)
      .map((r) => {
        const ordered = r.words.sort((a, b) => a.x - b.x);
        return ordered
          .map((w, i) => {
            const previous = ordered[i - 1];
            const join =
              previous &&
              w.x - (previous.x + previous.w) < Math.max(1, w.h * 0.12);
            return (i && !join ? " " : "") + w.text;
          })
          .join("")
          .replace(/\s+/g, " ")
          .trim();
      });
  });
}
