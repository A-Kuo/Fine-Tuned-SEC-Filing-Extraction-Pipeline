import { describe, expect, it } from "vitest";
import { DATA_AS_OF, getFilings, getSectors, getYieldSeries, getYieldSummary } from "../data";
import {
  DEFAULT_FILTERS,
  applyFilingFilters,
  assetOptions,
  describeFilters,
  entityOptions,
  inRange,
  parseFilters,
  slugify,
  windowOptions,
  windowRange,
  windowSubtitle,
  withFilter,
  type FilterState,
} from "../filters";

const entitySlugs = getSectors().map(slugify);

describe("windowRange", () => {
  it("resolves every window against the data as-of date", () => {
    expect(windowRange("cq", "2026-09-18")).toEqual({ from: "2026-07-01", to: "2026-09-18", label: "Q3 FY2026" });
    expect(windowRange("pq1", "2026-09-18")).toEqual({ from: "2026-04-01", to: "2026-06-30", label: "Q2 FY2026" });
    expect(windowRange("pq2", "2026-09-18")).toEqual({ from: "2026-01-01", to: "2026-03-31", label: "Q1 FY2026" });
    expect(windowRange("ytd", "2026-09-18")).toEqual({ from: "2026-01-01", to: "2026-09-18", label: "Year to Date" });
    expect(windowRange("ttm", "2026-09-18")).toEqual({ from: "2025-09-19", to: "2026-09-18", label: "Trailing 12 Months" });
    expect(windowRange("all", "2026-09-18")).toEqual({ from: null, to: "2026-09-18", label: "All Filings" });
  });

  it("rolls previous quarters back across a year boundary", () => {
    expect(windowRange("cq", "2026-02-10")).toMatchObject({ from: "2026-01-01", to: "2026-02-10", label: "Q1 FY2026" });
    expect(windowRange("pq1", "2026-02-10")).toMatchObject({ from: "2025-10-01", to: "2025-12-31", label: "Q4 FY2025" });
    expect(windowRange("pq2", "2026-02-10")).toMatchObject({ from: "2025-07-01", to: "2025-09-30", label: "Q3 FY2025" });
  });

  it("uses a full quarter for the current quarter when the as-of date is its last day", () => {
    expect(windowRange("cq", "2026-12-31")).toMatchObject({ from: "2026-10-01", to: "2026-12-31" });
  });

  it("clamps the trailing-12-month start on a leap day", () => {
    expect(windowRange("ttm", "2024-02-29").from).toBe("2023-03-01");
    expect(windowRange("ttm", "2024-03-01").from).toBe("2023-03-02");
  });

  it("inRange is inclusive on both ends and open on a null start", () => {
    const r = windowRange("pq1", "2026-09-18");
    expect(inRange("2026-04-01", r)).toBe(true);
    expect(inRange("2026-06-30", r)).toBe(true);
    expect(inRange("2026-03-31", r)).toBe(false);
    expect(inRange("2026-07-01", r)).toBe(false);
    expect(inRange("1999-01-01", windowRange("all", "2026-09-18"))).toBe(true);
  });
});

describe("windowSubtitle", () => {
  it("names the quarter for quarter windows and the span otherwise", () => {
    expect(windowSubtitle("cq", DATA_AS_OF)).toBe("Data Normalized: As of Q3 Audit Clearings");
    expect(windowSubtitle("pq2", DATA_AS_OF)).toBe("Data Normalized: As of Q1 Audit Clearings");
    expect(windowSubtitle("ytd", DATA_AS_OF)).toBe("Data Normalized: Year to Date, through 2026-09-18");
  });
});

describe("options", () => {
  it("asset options lead with All Classes and use URL-safe slugs", () => {
    const opts = assetOptions();
    expect(opts[0]).toEqual({ value: "all", label: "All Classes" });
    expect(opts.map((o) => o.value)).toEqual(["all", "treasury-bills", "corporate-bonds", "commodities", "liquidity-cushions"]);
  });

  it("entity options are Consolidated plus each distinct sector, sorted", () => {
    const opts = entityOptions(["Technology", "Energy", "Technology", "Financials"]);
    expect(opts.map((o) => o.label)).toEqual(["Consolidated", "Energy", "Financials", "Technology"]);
    expect(opts[0].value).toBe("all");
  });

  it("window options mirror the six windows", () => {
    expect(windowOptions(DATA_AS_OF).map((o) => o.value)).toEqual(["cq", "pq1", "pq2", "ytd", "ttm", "all"]);
  });
});

describe("parseFilters", () => {
  it("returns defaults for empty input", () => {
    expect(parseFilters({}, entitySlugs)).toEqual(DEFAULT_FILTERS);
  });

  it("accepts valid values from a plain object or URLSearchParams", () => {
    const expected: FilterState = { asset: "commodities", entity: "energy", window: "ttm", q: "meridian" };
    expect(parseFilters({ asset: "commodities", entity: "energy", window: "ttm", q: "meridian" }, entitySlugs)).toEqual(expected);
    expect(parseFilters(new URLSearchParams("asset=commodities&entity=energy&window=ttm&q=meridian"), entitySlugs)).toEqual(expected);
  });

  it("falls back to defaults for anything not on the whitelist", () => {
    const parsed = parseFilters({ asset: "bitcoin", entity: "<script>", window: "forever" }, entitySlugs);
    expect(parsed).toEqual(DEFAULT_FILTERS);
  });

  it("takes the first value of a repeated param", () => {
    expect(parseFilters({ window: ["ytd", "all"] }, entitySlugs).window).toBe("ytd");
  });

  it("trims and length-caps the search text", () => {
    expect(parseFilters({ q: "  hvin  " }, entitySlugs).q).toBe("hvin");
    expect(parseFilters({ q: "x".repeat(500) }, entitySlugs).q).toHaveLength(80);
  });
});

describe("withFilter", () => {
  it("sets a value and preserves the other params", () => {
    expect(withFilter("asset=commodities", "window", "ytd")).toBe("asset=commodities&window=ytd");
  });

  it("removes the param when set back to its default", () => {
    expect(withFilter("asset=commodities&window=ytd", "asset", "all")).toBe("window=ytd");
    expect(withFilter("window=ytd", "window", "cq")).toBe("");
    expect(withFilter("q=abc", "q", "")).toBe("");
  });

  it("accepts a URLSearchParams and encodes values", () => {
    expect(withFilter(new URLSearchParams("entity=energy"), "q", "a b&c")).toBe("entity=energy&q=a+b%26c");
  });
});

describe("applyFilingFilters", () => {
  const rows = getFilings();
  const run = (over: Partial<FilterState>) => applyFilingFilters(rows, { ...DEFAULT_FILTERS, window: "all", ...over }, DATA_AS_OF);

  it("every demo window has data, so no filter combination on the default view is empty", () => {
    for (const w of ["cq", "pq1", "pq2", "ytd", "ttm", "all"] as const) {
      expect(run({ window: w }).length, w).toBeGreaterThan(0);
    }
  });

  it("windows partition the year: the three quarters together equal year-to-date", () => {
    expect(run({ window: "cq" }).length + run({ window: "pq1" }).length + run({ window: "pq2" }).length).toBe(run({ window: "ytd" }).length);
  });

  it("trailing 12 months is a subset of all filings, and all keeps every row", () => {
    expect(run({ window: "all" }).length).toBe(rows.length);
    const ttm = new Set(run({ window: "ttm" }).map((r) => r.cik + r.filingDate + r.formType));
    const all = new Set(run({ window: "all" }).map((r) => r.cik + r.filingDate + r.formType));
    expect([...ttm].every((k) => all.has(k))).toBe(true);
    expect(ttm.size).toBeLessThan(all.size);
  });

  it("returns only rows inside the selected window", () => {
    const range = windowRange("pq1", DATA_AS_OF);
    for (const r of run({ window: "pq1" })) expect(inRange(r.filingDate, range)).toBe(true);
  });

  it("entity filter returns only that sector, and every sector option is non-empty", () => {
    for (const sector of getSectors()) {
      const out = run({ entity: slugify(sector) });
      expect(out.length, sector).toBeGreaterThan(0);
      expect(out.every((r) => r.sector === sector)).toBe(true);
    }
  });

  it("search matches ticker, CIK digits, form type and entity name, case-insensitively", () => {
    expect(run({ q: "hvin" }).every((r) => r.ticker === "HVIN")).toBe(true);
    expect(run({ q: "HVIN" }).length).toBeGreaterThan(0);
    expect(run({ q: "99000101" }).every((r) => r.cik === "0099000101")).toBe(true);
    expect(run({ q: "10-k" }).every((r) => r.formType === "10-K")).toBe(true);
    expect(run({ q: "halvorsen" }).length).toBe(run({ q: "hvin" }).length);
  });

  it("multiple search tokens must all match", () => {
    const out = run({ q: "hvin 10-k" });
    expect(out.length).toBeGreaterThan(0);
    expect(out.every((r) => r.ticker === "HVIN" && r.formType === "10-K")).toBe(true);
    expect(run({ q: "hvin nonexistent-token" })).toEqual([]);
  });

  it("combines entity, window and search", () => {
    const out = run({ entity: "technology", window: "ytd", q: "10-q" });
    const range = windowRange("ytd", DATA_AS_OF);
    expect(out.every((r) => r.sector === "Technology" && r.formType === "10-Q" && inRange(r.filingDate, range))).toBe(true);
  });
});

describe("yield data", () => {
  it("windowed series stay inside the window and are ordered by date", () => {
    const range = windowRange("pq1", DATA_AS_OF);
    const series = getYieldSeries("all", "pq1");
    expect(series.length).toBeGreaterThan(3);
    expect(series.every((p) => inRange(p.date, range))).toBe(true);
    expect([...series].sort((a, b) => a.date.localeCompare(b.date))).toEqual(series);
  });

  it("each asset class has its own distinct series", () => {
    const a = getYieldSeries("treasury-bills", "ttm").map((p) => p.yield);
    const b = getYieldSeries("commodities", "ttm").map((p) => p.yield);
    expect(a).not.toEqual(b);
    expect(getYieldSeries("all", "ttm").map((p) => p.yield)).not.toEqual(a);
  });

  it("summary lists every class plus blended for All, and only the selected class otherwise", () => {
    expect(getYieldSummary("all", "ytd").map((r) => r.slug)).toEqual([
      "treasury-bills", "corporate-bonds", "commodities", "liquidity-cushions", "blended",
    ]);
    expect(getYieldSummary("commodities", "ytd").map((r) => r.slug)).toEqual(["commodities"]);
  });

  it("summary high/low bracket the latest value", () => {
    for (const row of getYieldSummary("all", "ttm")) {
      expect(row.low!).toBeLessThanOrEqual(row.latest!);
      expect(row.high!).toBeGreaterThanOrEqual(row.latest!);
    }
  });
});

describe("describeFilters", () => {
  const options = { asset: assetOptions(), entity: entityOptions(getSectors()), window: windowOptions(DATA_AS_OF) };

  it("lists only the filters that apply on the tab", () => {
    const text = describeFilters(
      { asset: "commodities", entity: "energy", window: "ytd", q: "" },
      { asset: false, entity: true, window: true, search: true },
      options,
    );
    expect(text).toBe("Entity Hierarchy: Energy  |  Filing Window: Year to Date");
  });

  it("includes the search text only when set and applicable", () => {
    const applies = { asset: false, entity: false, window: false, search: true };
    expect(describeFilters({ ...DEFAULT_FILTERS, q: "hvin" }, applies, options)).toBe('Search: "hvin"');
    expect(describeFilters(DEFAULT_FILTERS, applies, options)).toBe("No filters applied");
  });
});
