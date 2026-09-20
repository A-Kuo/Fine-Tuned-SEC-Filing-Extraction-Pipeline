import { describe, expect, it } from "vitest";
import { PROVENANCE } from "../provenance";
import { MARKET_SOURCES, REPO_URL, SITE_NAME, repoFile } from "../site";

describe("site constants", () => {
  it("uses the new brand and none of the old one", () => {
    expect(SITE_NAME).toBe("SEC Edgar Filing Platform");
    expect(SITE_NAME).not.toMatch(/EDGAR-X|Disclosure Matrix/i);
  });

  it("points at the GitHub repo without a .git suffix, and builds file links from it", () => {
    expect(REPO_URL).toBe("https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline");
    expect(REPO_URL.endsWith(".git")).toBe(false);
    expect(repoFile("MODEL_CARD.md")).toBe(`${REPO_URL}/blob/main/MODEL_CARD.md`);
  });

  it("market sources are https links to the sites the status actually comes from", () => {
    const yahoo = new URL(MARKET_SOURCES.yahoo.url);
    const nyse = new URL(MARKET_SOURCES.nyse.url);
    expect(yahoo.protocol).toBe("https:");
    expect(yahoo.hostname).toBe("finance.yahoo.com");
    expect(decodeURIComponent(yahoo.pathname)).toContain("^GSPC");
    expect(nyse.protocol).toBe("https:");
    expect(nyse.hostname).toBe("www.nyse.com");
  });
});

describe("data provenance table", () => {
  it("never claims a part is real while its source is still marked planned", () => {
    for (const row of PROVENANCE) {
      if (row.status === "real") expect(row.source, row.area).not.toMatch(/planned/i);
    }
  });

  it("marks every part whose source is still planned as illustrative", () => {
    for (const row of PROVENANCE.filter((r) => /planned/i.test(r.source))) {
      expect(row.status, row.area).toBe("illustrative");
    }
  });

  it("lists each area once and always has a note", () => {
    const areas = PROVENANCE.map((r) => r.area);
    expect(new Set(areas).size).toBe(areas.length);
    for (const row of PROVENANCE) expect(row.note.length, row.area).toBeGreaterThan(0);
  });
});
