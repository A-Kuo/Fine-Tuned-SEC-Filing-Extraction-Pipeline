export type FilingStatus = "accepted" | "in_review" | "flagged";

export type Sector =
  | "Technology"
  | "Financials"
  | "Healthcare"
  | "Consumer Staples"
  | "Energy"
  | "Real Estate";

export interface FilingRow {
  entity: string;
  ticker: string;
  cik: string;
  sector: Sector;
  formType: string;
  filingDate: string;
  signatory: string;
  status: FilingStatus;
}

export interface AllocationSegment {
  label: string;
  slug: string;
  pct: number;
  marketValueUsd: number;
  yoyPct: number;
  yieldPct: number;
}

export interface YieldPoint {
  date: string;
  yield: number;
  volatility: number;
}

export interface AuditEvent {
  id: string;
  ts: string;
  actor: string;
  action: string;
  object: string;
  detail: string;
  source: "system" | "local";
}

export type SummaryTone = "gain" | "loss" | "neutral" | "muted" | "amber";

export interface SummaryIndicator {
  label: string;
  value: string;
  delta: string;
  tone: SummaryTone;
}
