import { describe, expect, it } from "vitest";
import { safeHref } from "@/ui/kit";

describe("safeHref", () => {
  it("keeps plain web links", () => {
    expect(safeHref("https://arxiv.org/abs/2402.02750")).toBe("https://arxiv.org/abs/2402.02750");
    expect(safeHref("http://example.org/paper.pdf")).toBe("http://example.org/paper.pdf");
  });

  it("drops script, data and malformed links from paper metadata", () => {
    expect(safeHref("javascript:alert(1)")).toBeUndefined();
    expect(safeHref(" JavaScript:alert(1)")).toBeUndefined();
    expect(safeHref("data:text/html,<script>alert(1)</script>")).toBeUndefined();
    expect(safeHref("not a url")).toBeUndefined();
    expect(safeHref(undefined)).toBeUndefined();
  });
});
