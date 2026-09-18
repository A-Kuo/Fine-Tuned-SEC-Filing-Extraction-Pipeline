// ILLUSTRATIVE DATA. Every issuer, CIK, signatory, status, yield and audit
// event below is invented for demonstration. Issuers are fictional and their
// CIKs (0099000xxx) sit outside the range SEC has assigned, so nothing here
// can be mistaken for a claim about a real registrant. lib/data.ts is the
// only module that reads this file; swap that layer to point at
// serving/api.py when real list endpoints exist.

import type {
  AllocationSegment,
  AuditEvent,
  FilingRow,
  FilingStatus,
  Sector,
  YieldPoint,
} from "./types";

// Windows ("Q3 FY2026", YTD, ...) anchor here, not to the wall clock, so the
// demo never rolls over into an empty current quarter.
export const DATA_AS_OF = "2026-09-18";

export const AUM_USD = 42_860_000_000;

const ISSUERS = {
  HVIN: { entity: "Halvorsen Instruments Corp.", cik: "0099000101", sector: "Technology" },
  TLRB: { entity: "Tallis Regional Bancorp", cik: "0099000102", sector: "Financials" },
  CRVB: { entity: "Corvane Biologics, Inc.", cik: "0099000103", sector: "Healthcare" },
  PLSM: { entity: "Pellham Semiconductor Ltd.", cik: "0099000104", sector: "Technology" },
  OSFH: { entity: "Ostrander Foods Holdings", cik: "0099000105", sector: "Consumer Staples" },
  MBEP: { entity: "Meridian Basin Energy Partners", cik: "0099000106", sector: "Energy" },
  NGRT: { entity: "Northgate Realty Trust", cik: "0099000107", sector: "Real Estate" },
} as const satisfies Record<string, { entity: string; cik: string; sector: Sector }>;

type IssuerKey = keyof typeof ISSUERS;

const filing = (
  ticker: IssuerKey,
  formType: string,
  filingDate: string,
  signatory: string,
  status: FilingStatus,
): FilingRow => ({ ...ISSUERS[ticker], ticker, formType, filingDate, signatory, status });

const CFO = "Chief Financial Officer";
const CEO = "Chief Executive Officer";
const PAO = "Principal Accounting Officer";
const GC = "General Counsel";

export const filingsLedger: FilingRow[] = [
  // Q3 FY2026
  filing("TLRB", "8-K", "2026-09-11", CEO, "in_review"),
  filing("CRVB", "10-K", "2026-09-04", CFO, "in_review"),
  filing("PLSM", "10-K", "2026-08-28", CFO, "accepted"),
  filing("OSFH", "10-Q", "2026-08-25", PAO, "accepted"),
  filing("MBEP", "10-Q", "2026-08-19", CFO, "flagged"),
  filing("NGRT", "10-Q", "2026-08-13", CFO, "accepted"),
  filing("HVIN", "10-Q", "2026-08-07", CFO, "accepted"),
  filing("TLRB", "10-Q", "2026-07-29", CFO, "accepted"),
  filing("CRVB", "8-K", "2026-07-15", GC, "flagged"),
  // Q2 FY2026
  filing("OSFH", "8-K", "2026-06-24", CEO, "accepted"),
  filing("PLSM", "10-Q", "2026-05-28", CFO, "accepted"),
  filing("MBEP", "10-Q", "2026-05-20", CFO, "accepted"),
  filing("NGRT", "10-Q", "2026-05-12", CFO, "accepted"),
  filing("HVIN", "10-Q", "2026-05-06", CFO, "accepted"),
  filing("TLRB", "10-Q", "2026-04-22", CFO, "accepted"),
  // Q1 FY2026
  filing("HVIN", "10-K", "2026-03-26", CEO, "accepted"),
  filing("CRVB", "10-Q", "2026-03-05", CFO, "accepted"),
  filing("NGRT", "10-K", "2026-02-24", CFO, "accepted"),
  filing("TLRB", "10-K", "2026-02-18", PAO, "flagged"),
  filing("MBEP", "10-K", "2026-01-30", CFO, "accepted"),
  // Q4 FY2025
  filing("PLSM", "8-K", "2025-12-11", GC, "accepted"),
  filing("OSFH", "10-Q", "2025-11-13", PAO, "accepted"),
  filing("HVIN", "10-Q", "2025-11-06", CFO, "accepted"),
  filing("CRVB", "8-K", "2025-10-16", CEO, "accepted"),
  // Q3 FY2025
  filing("TLRB", "8-K", "2025-09-09", CEO, "accepted"),
  filing("NGRT", "10-Q", "2025-08-21", CFO, "accepted"),
  filing("MBEP", "10-Q", "2025-08-07", CFO, "accepted"),
].sort((a, b) => (a.filingDate < b.filingDate ? 1 : a.filingDate > b.filingDate ? -1 : 0));

// Semi-monthly points (1st and 16th) from 2025-01 through DATA_AS_OF.
function semiMonthlyDates(): string[] {
  const out: string[] = [];
  for (let year = 2025; year <= 2026; year++) {
    for (let month = 1; month <= 12; month++) {
      for (const day of [1, 16]) {
        const iso = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
        if (iso <= DATA_AS_OF) out.push(iso);
      }
    }
  }
  return out;
}

const round2 = (n: number) => Math.round(n * 100) / 100;

// Deterministic (no randomness) so server and client renders always agree.
function makeSeries(
  base: number,
  slope: number,
  amp: number,
  phase: number,
  volBase: number,
  volAmp: number,
): YieldPoint[] {
  return semiMonthlyDates().map((date, i) => ({
    date,
    yield: round2(base + slope * i + amp * Math.sin(i / 3 + phase)),
    volatility: round2(volBase + volAmp * Math.abs(Math.sin(i / 2 + phase))),
  }));
}

export const ASSET_CLASS_DEFS = [
  { label: "Treasury Bills", slug: "treasury-bills", pct: 38, yoyPct: 5.1, series: makeSeries(4.2, 0.03, 0.06, 0.0, 0.35, 0.12) },
  { label: "Corporate Bonds", slug: "corporate-bonds", pct: 31, yoyPct: 11.8, series: makeSeries(5.1, 0.026, 0.14, 1.1, 0.7, 0.25) },
  { label: "Commodities", slug: "commodities", pct: 17, yoyPct: 31.9, series: makeSeries(3.2, 0.045, 0.55, 2.3, 1.6, 0.6) },
  { label: "Liquidity Cushions", slug: "liquidity-cushions", pct: 14, yoyPct: 2.6, series: makeSeries(4.0, 0.02, 0.03, 0.4, 0.12, 0.05) },
] as const;

export const yieldSeriesByClass: Record<string, YieldPoint[]> = Object.fromEntries(
  ASSET_CLASS_DEFS.map((c) => [c.slug, [...c.series]]),
);

export const blendedYieldSeries: YieldPoint[] = semiMonthlyDates().map((date, i) => ({
  date,
  yield: round2(ASSET_CLASS_DEFS.reduce((sum, c) => sum + (c.pct / 100) * c.series[i].yield, 0)),
  volatility: round2(ASSET_CLASS_DEFS.reduce((sum, c) => sum + (c.pct / 100) * c.series[i].volatility, 0)),
}));

export const allocationMatrix: AllocationSegment[] = ASSET_CLASS_DEFS.map((c) => ({
  label: c.label,
  slug: c.slug,
  pct: c.pct,
  marketValueUsd: Math.round((c.pct / 100) * AUM_USD),
  yoyPct: c.yoyPct,
  yieldPct: c.series[c.series.length - 1].yield,
}));

const SYSTEM_ACTORS = { ingest: "svc-ingest", review: "svc-review", scheduler: "svc-scheduler" };

function filingEvents(): AuditEvent[] {
  const events: AuditEvent[] = [];
  filingsLedger.forEach((row, i) => {
    const object = `${row.ticker} ${row.formType} ${row.filingDate}`;
    events.push({
      id: `sys-${i}-received`,
      ts: `${row.filingDate}T13:02:00Z`,
      actor: SYSTEM_ACTORS.ingest,
      action: "filing.received",
      object,
      detail: `Received from EDGAR feed for ${row.entity}`,
      source: "system",
    });
    const outcome: Record<FilingStatus, { action: string; detail: string; hour: string }> = {
      accepted: { action: "filing.accepted", detail: "Passed schema and completeness checks", hour: "15:10" },
      in_review: { action: "filing.queued_for_review", detail: "Awaiting analyst review", hour: "16:25" },
      flagged: { action: "filing.flagged", detail: "Failed completeness check; returned for amendment", hour: "18:40" },
    };
    const o = outcome[row.status];
    events.push({
      id: `sys-${i}-outcome`,
      ts: `${row.filingDate}T${o.hour}:00Z`,
      actor: SYSTEM_ACTORS.review,
      action: o.action,
      object,
      detail: o.detail,
      source: "system",
    });
  });
  return events;
}

const operatorEvents: AuditEvent[] = [
  { id: "sys-op-1", ts: "2026-09-17T15:12:00Z", actor: "analyst.m", action: "export.csv", object: "Regulatory Filings", detail: "Exported filtered ledger to CSV", source: "system" },
  { id: "sys-op-2", ts: "2026-09-15T09:40:00Z", actor: "compliance.k", action: "policy.updated", object: "Filing retention policy", detail: "Retention period set to 7 years", source: "system" },
  { id: "sys-op-3", ts: "2026-08-31T20:05:00Z", actor: "analyst.m", action: "report.printed", object: "Regulatory Filings", detail: "Printed ledger for Q3 review packet", source: "system" },
  { id: "sys-op-4", ts: "2026-07-01T00:00:00Z", actor: SYSTEM_ACTORS.scheduler, action: "period.rollover", object: "Q3 FY2026", detail: "Filing window opened", source: "system" },
  { id: "sys-op-5", ts: "2026-04-01T00:00:00Z", actor: SYSTEM_ACTORS.scheduler, action: "period.rollover", object: "Q2 FY2026", detail: "Filing window opened", source: "system" },
  { id: "sys-op-6", ts: "2026-01-02T00:00:00Z", actor: SYSTEM_ACTORS.scheduler, action: "period.rollover", object: "Q1 FY2026", detail: "Filing window opened", source: "system" },
  { id: "sys-op-7", ts: "2025-10-01T00:00:00Z", actor: SYSTEM_ACTORS.scheduler, action: "period.rollover", object: "Q4 FY2025", detail: "Filing window opened", source: "system" },
];

export const auditSeed: AuditEvent[] = [...filingEvents(), ...operatorEvents].sort((a, b) =>
  a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0,
);

export const summaryBase = {
  aumLabel: "Total Assets Under Management",
  aumDelta: "+11.4% YoY Alpha",
  settlementValue: "+$614,210.45",
  settlementDelta: "Settled T+1",
  leverageValue: "1.42x",
  leverageDelta: "Compliant / Within SEC Tier 1",
  disclosuresValue: "384 Filings",
  disclosuresDelta: "12 Pending Review",
};
