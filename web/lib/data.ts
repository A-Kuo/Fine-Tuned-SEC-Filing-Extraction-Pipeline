// The only module pages import data from. Everything reads lib/mock-data.ts
// today; when serving/api.py grows list endpoints these become async fetches
// and the pages (already server components) only need an `await`.

import { inRange, windowRange, type WindowKey } from "./filters";
import { formatUsdCompact, signed } from "./format";
import {
  auditSeed,
  allocationMatrix,
  blendedYieldSeries,
  DATA_AS_OF,
  filingsLedger,
  summaryBase,
  yieldSeriesByClass,
} from "./mock-data";
import type {
  AllocationSegment,
  AuditEvent,
  FilingRow,
  SummaryIndicator,
  YieldPoint,
} from "./types";

export { DATA_AS_OF };

export function getFilings(): FilingRow[] {
  return filingsLedger;
}

export function getSectors(): string[] {
  return [...new Set(filingsLedger.map((f) => f.sector))].sort();
}

export function getAllocation(): AllocationSegment[] {
  return allocationMatrix;
}

export function getAuditSeed(): AuditEvent[] {
  return auditSeed;
}

export function getYieldSeries(assetSlug: string, window: WindowKey): YieldPoint[] {
  const series = assetSlug === "all" ? blendedYieldSeries : (yieldSeriesByClass[assetSlug] ?? blendedYieldSeries);
  const range = windowRange(window, DATA_AS_OF);
  return series.filter((p) => inRange(p.date, range));
}

// When an asset class is selected the first tile becomes that class's market
// value; the other three are portfolio-wide by nature and stay as they are.
export function getSummaryIndicators(assetSlug: string): SummaryIndicator[] {
  const selected = allocationMatrix.find((a) => a.slug === assetSlug);
  const first: SummaryIndicator = selected
    ? {
        label: `${selected.label} Market Value`,
        value: formatUsdCompact(selected.marketValueUsd),
        delta: `${signed(selected.yoyPct)}% YoY, ${selected.pct}% of portfolio`,
        tone: selected.yoyPct >= 0 ? "gain" : "loss",
      }
    : {
        label: summaryBase.aumLabel,
        value: formatUsdCompact(allocationMatrix.reduce((s, a) => s + a.marketValueUsd, 0)),
        delta: summaryBase.aumDelta,
        tone: "gain",
      };

  return [
    first,
    { label: "Net Settlement Position", value: summaryBase.settlementValue, delta: summaryBase.settlementDelta, tone: "neutral" },
    { label: "Regulatory Leverage Ratio", value: summaryBase.leverageValue, delta: summaryBase.leverageDelta, tone: "muted" },
    { label: "Active Entity Disclosures", value: summaryBase.disclosuresValue, delta: summaryBase.disclosuresDelta, tone: "amber" },
  ];
}

export function getFilingStats(rows: FilingRow[]): SummaryIndicator[] {
  const count = (status: FilingRow["status"]) => rows.filter((r) => r.status === status).length;
  const total = rows.length;
  const share = (n: number) => (total === 0 ? "0%" : `${Math.round((n / total) * 100)}% of filings in view`);
  return [
    { label: "Filings in View", value: String(total), delta: "Matching current filters", tone: "neutral" },
    { label: "Processed / Accepted", value: String(count("accepted")), delta: share(count("accepted")), tone: "gain" },
    { label: "In Review", value: String(count("in_review")), delta: share(count("in_review")), tone: "amber" },
    { label: "Filing Deficit / Flagged", value: String(count("flagged")), delta: share(count("flagged")), tone: "loss" },
  ];
}

export interface YieldSummaryRow {
  label: string;
  slug: string;
  latest: number | null;
  change: number | null;
  high: number | null;
  low: number | null;
  avgVolatility: number | null;
  weight: number | null;
}

// Per-class window summary; `assetSlug` narrows to one class, "all" lists
// every class plus the blended portfolio line.
export function getYieldSummary(assetSlug: string, window: WindowKey): YieldSummaryRow[] {
  const range = windowRange(window, DATA_AS_OF);
  const summarise = (label: string, slug: string, points: YieldPoint[], weight: number | null): YieldSummaryRow => {
    const inWindow = points.filter((p) => inRange(p.date, range));
    if (inWindow.length === 0) {
      return { label, slug, latest: null, change: null, high: null, low: null, avgVolatility: null, weight };
    }
    const values = inWindow.map((p) => p.yield);
    return {
      label,
      slug,
      latest: inWindow[inWindow.length - 1].yield,
      change: Math.round((inWindow[inWindow.length - 1].yield - inWindow[0].yield) * 100) / 100,
      high: Math.max(...values),
      low: Math.min(...values),
      avgVolatility: Math.round((inWindow.reduce((s, p) => s + p.volatility, 0) / inWindow.length) * 100) / 100,
      weight,
    };
  };

  const classes = allocationMatrix
    .filter((a) => assetSlug === "all" || a.slug === assetSlug)
    .map((a) => summarise(a.label, a.slug, yieldSeriesByClass[a.slug], a.pct));

  return assetSlug === "all"
    ? [...classes, summarise("Blended Portfolio", "blended", blendedYieldSeries, 100)]
    : classes;
}
