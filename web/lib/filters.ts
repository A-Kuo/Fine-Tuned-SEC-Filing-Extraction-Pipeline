import { ASSET_CLASS_DEFS } from "./mock-data";
import type { FilingRow } from "./types";

export type WindowKey = "cq" | "pq1" | "pq2" | "ytd" | "ttm" | "all";

export interface FilterState {
  asset: string;
  entity: string;
  window: WindowKey;
  q: string;
}

export interface Option {
  value: string;
  label: string;
}

export interface DateRange {
  from: string | null;
  to: string;
  label: string;
}

export const DEFAULT_FILTERS: FilterState = { asset: "all", entity: "all", window: "cq", q: "" };

const WINDOW_KEYS: WindowKey[] = ["cq", "pq1", "pq2", "ytd", "ttm", "all"];
const MAX_QUERY_LENGTH = 80;

export function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function iso(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function utcDate(year: number, month: number, day: number): Date {
  return new Date(Date.UTC(year, month - 1, day));
}

interface Quarter {
  year: number;
  q: number;
}

function quarterOf(isoDate: string): Quarter {
  const [year, month] = isoDate.split("-").map(Number);
  return { year, q: Math.floor((month - 1) / 3) + 1 };
}

function previousQuarter({ year, q }: Quarter): Quarter {
  return q === 1 ? { year: year - 1, q: 4 } : { year, q: q - 1 };
}

function quarterBounds({ year, q }: Quarter): { from: string; to: string } {
  const startMonth = (q - 1) * 3 + 1;
  return {
    from: iso(utcDate(year, startMonth, 1)),
    to: iso(utcDate(year, startMonth + 3, 0)),
  };
}

const quarterLabel = ({ year, q }: Quarter) => `Q${q} FY${year}`;

export function windowRange(key: WindowKey, asOf: string): DateRange {
  const current = quarterOf(asOf);
  const [year, month, day] = asOf.split("-").map(Number);

  switch (key) {
    case "cq": {
      const b = quarterBounds(current);
      return { from: b.from, to: b.to < asOf ? b.to : asOf, label: quarterLabel(current) };
    }
    case "pq1": {
      const p = previousQuarter(current);
      return { ...quarterBounds(p), label: quarterLabel(p) };
    }
    case "pq2": {
      const p = previousQuarter(previousQuarter(current));
      return { ...quarterBounds(p), label: quarterLabel(p) };
    }
    case "ytd":
      return { from: `${year}-01-01`, to: asOf, label: "Year to Date" };
    case "ttm": {
      // Same calendar day one year earlier (clamped: Feb 29 has no prior-year twin), then +1 day.
      const lastDayPrevYear = new Date(Date.UTC(year - 1, month, 0)).getUTCDate();
      const from = utcDate(year - 1, month, Math.min(day, lastDayPrevYear));
      from.setUTCDate(from.getUTCDate() + 1);
      return { from: iso(from), to: asOf, label: "Trailing 12 Months" };
    }
    case "all":
      return { from: null, to: asOf, label: "All Filings" };
  }
}

export function windowOptions(asOf: string): Option[] {
  return WINDOW_KEYS.map((value) => ({ value, label: windowRange(value, asOf).label }));
}

export function assetOptions(): Option[] {
  return [
    { value: "all", label: "All Classes" },
    ...ASSET_CLASS_DEFS.map((c) => ({ value: c.slug, label: c.label })),
  ];
}

export function entityOptions(sectors: string[]): Option[] {
  const unique = [...new Set(sectors)].sort();
  return [{ value: "all", label: "Consolidated" }, ...unique.map((s) => ({ value: slugify(s), label: s }))];
}

export function windowSubtitle(key: WindowKey, asOf: string): string {
  const range = windowRange(key, asOf);
  if (key === "cq" || key === "pq1" || key === "pq2") {
    return `Data Normalized: As of ${range.label.split(" ")[0]} Audit Clearings`;
  }
  return `Data Normalized: ${range.label}, through ${asOf}`;
}

type Params = URLSearchParams | Record<string, string | string[] | undefined>;

function first(params: Params, key: string): string | undefined {
  if (params instanceof URLSearchParams) return params.get(key) ?? undefined;
  const v = params[key];
  return Array.isArray(v) ? v[0] : v;
}

// Whitelists every value; anything unrecognised falls back to its default so a
// hand-edited or stale URL can never put the page into an impossible state.
export function parseFilters(params: Params, validEntitySlugs: string[]): FilterState {
  const asset = first(params, "asset");
  const entity = first(params, "entity");
  const window = first(params, "window");
  const q = first(params, "q");

  return {
    asset: asset && ASSET_CLASS_DEFS.some((c) => c.slug === asset) ? asset : DEFAULT_FILTERS.asset,
    entity: entity && validEntitySlugs.includes(entity) ? entity : DEFAULT_FILTERS.entity,
    window: window && (WINDOW_KEYS as string[]).includes(window) ? (window as WindowKey) : DEFAULT_FILTERS.window,
    q: (q ?? "").trim().slice(0, MAX_QUERY_LENGTH),
  };
}

// Returns the query string (no leading "?") with `key` set, or removed when it
// equals the default so URLs stay clean.
export function withFilter(current: URLSearchParams | string, key: keyof FilterState, value: string): string {
  const params = new URLSearchParams(typeof current === "string" ? current : current.toString());
  if (value === "" || value === DEFAULT_FILTERS[key]) params.delete(key);
  else params.set(key, value);
  return params.toString();
}

export function inRange(date: string, range: DateRange): boolean {
  return (range.from === null || date >= range.from) && date <= range.to;
}

export function applyFilingFilters(rows: FilingRow[], filters: FilterState, asOf: string): FilingRow[] {
  const range = windowRange(filters.window, asOf);
  const tokens = filters.q.toLowerCase().split(/\s+/).filter(Boolean);

  return rows.filter((row) => {
    if (filters.entity !== "all" && slugify(row.sector) !== filters.entity) return false;
    if (!inRange(row.filingDate, range)) return false;
    if (tokens.length === 0) return true;
    const haystack = `${row.entity} ${row.ticker} ${row.cik} ${row.formType} ${row.sector} ${row.status}`.toLowerCase();
    return tokens.every((t) => haystack.includes(t));
  });
}

export interface FilterApplicability {
  asset: boolean;
  entity: boolean;
  window: boolean;
  search: boolean;
}

export function describeFilters(
  filters: FilterState,
  applies: FilterApplicability,
  options: { asset: Option[]; entity: Option[]; window: Option[] },
): string {
  const label = (opts: Option[], value: string) => opts.find((o) => o.value === value)?.label ?? value;
  const parts: string[] = [];
  if (applies.asset) parts.push(`Asset Class: ${label(options.asset, filters.asset)}`);
  if (applies.entity) parts.push(`Entity Hierarchy: ${label(options.entity, filters.entity)}`);
  if (applies.window) parts.push(`Filing Window: ${label(options.window, filters.window)}`);
  if (applies.search && filters.q) parts.push(`Search: "${filters.q}"`);
  return parts.length ? parts.join("  |  ") : "No filters applied";
}
