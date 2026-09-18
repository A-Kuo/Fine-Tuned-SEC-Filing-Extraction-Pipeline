// Illustrative data for the dashboard shell. Shaped like the real records
// this UI is designed to eventually render (see serving/api.py's
// /extractions/{filing_id} and /pipeline/status), not live figures.

export type FilingStatus = "accepted" | "in_review" | "flagged";

export interface FilingRow {
  entity: string;
  ticker: string;
  cik: string;
  formType: string;
  filingDate: string;
  signatory: string;
  status: FilingStatus;
}

export const filingsLedger: FilingRow[] = [
  { entity: "Apple Inc.", ticker: "AAPL", cik: "0000320193", formType: "10-K", filingDate: "2026-08-29", signatory: "L. Maestri, CFO", status: "accepted" },
  { entity: "Microsoft Corporation", ticker: "MSFT", cik: "0000789019", formType: "10-K", filingDate: "2026-08-27", signatory: "A. Hood, CFO", status: "accepted" },
  { entity: "The Coca-Cola Company", ticker: "KO", cik: "0000021344", formType: "10-Q", filingDate: "2026-08-22", signatory: "J. Murphy, CFO", status: "in_review" },
  { entity: "Berkshire Hathaway Inc.", ticker: "BRK.A", cik: "0001067983", formType: "10-Q", filingDate: "2026-08-18", signatory: "M. Hamburg, VP", status: "accepted" },
  { entity: "Moderna, Inc.", ticker: "MRNA", cik: "0001682852", formType: "10-Q", filingDate: "2026-08-14", signatory: "J. Nagle, CFO", status: "flagged" },
  { entity: "Realty Income Corporation", ticker: "O", cik: "0000726728", formType: "10-K", filingDate: "2026-08-11", signatory: "J. Reagan, CFO", status: "accepted" },
  { entity: "JPMorgan Chase & Co.", ticker: "JPM", cik: "0000019617", formType: "10-Q", filingDate: "2026-08-06", signatory: "J. Barnum, CFO", status: "in_review" },
  { entity: "Goldman Sachs Group, Inc.", ticker: "GS", cik: "0000886982", formType: "8-K", filingDate: "2026-08-02", signatory: "D. Solomon, CEO", status: "flagged" },
];

export const yieldSeries = [
  { period: "Q4 '24", cumulativeYield: 4.12, filingVolatility: 0.8 },
  { period: "Q1 '25", cumulativeYield: 4.38, filingVolatility: 1.1 },
  { period: "Q2 '25", cumulativeYield: 4.21, filingVolatility: 1.4 },
  { period: "Q3 '25", cumulativeYield: 4.65, filingVolatility: 0.9 },
  { period: "Q4 '25", cumulativeYield: 4.94, filingVolatility: 1.6 },
  { period: "Q1 '26", cumulativeYield: 5.02, filingVolatility: 1.2 },
  { period: "Q2 '26", cumulativeYield: 5.31, filingVolatility: 2.0 },
  { period: "Q3 '26", cumulativeYield: 5.47, filingVolatility: 1.3 },
];

export interface AllocationSegment {
  label: string;
  pct: number;
}

export const allocationMatrix: AllocationSegment[] = [
  { label: "Treasury Bills", pct: 38 },
  { label: "Corporate Bonds", pct: 31 },
  { label: "Commodities", pct: 17 },
  { label: "Liquidity Cushions", pct: 14 },
];

export const summaryIndicators = [
  {
    label: "Total Assets Under Management",
    value: "$42.86B",
    delta: "+11.4% YoY Alpha",
    tone: "gain" as const,
  },
  {
    label: "Net Settlement Position",
    value: "+$614,210.45",
    delta: "Settled T+1",
    tone: "neutral" as const,
  },
  {
    label: "Regulatory Leverage Ratio",
    value: "1.42x",
    delta: "Compliant / Within SEC Tier 1",
    tone: "muted" as const,
  },
  {
    label: "Active Entity Disclosures",
    value: "384 Filings",
    delta: "12 Pending Review",
    tone: "amber" as const,
  },
];
