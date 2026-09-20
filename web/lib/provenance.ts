// Single source of truth for "is this data real yet?", shown on the home page.
// Every data increment must update its row in the same change, so the home
// page can never claim more than the dashboard actually does.

export type DataStatus = "real" | "illustrative";

export interface ProvenanceRow {
  area: string;
  source: string;
  status: DataStatus;
  note: string;
}

export const PROVENANCE: ProvenanceRow[] = [
  {
    area: "Regulatory Filings",
    source: "SEC EDGAR submissions API (planned)",
    status: "illustrative",
    note: "Fictional issuers for now. Real filing metadata replaces them first.",
  },
  {
    area: "Entity Hierarchy filter",
    source: "SEC Standard Industrial Classification codes (planned)",
    status: "illustrative",
    note: "Sector groups are invented for now. Real SIC divisions and major groups come with the filings data.",
  },
  {
    area: "Portfolio Matrix and Asset Class filter",
    source: "SEC Form N-PORT fund holdings (planned)",
    status: "illustrative",
    note: "The four asset classes and all portfolio figures are invented for now.",
  },
  {
    area: "Market Yields",
    source: "SEC Form N-PORT coupon rates (planned)",
    status: "illustrative",
    note: "Yield and volatility series are generated, not observed.",
  },
  {
    area: "Audit Trails",
    source: "EDGAR acceptance timestamps (planned) plus this browser's own actions",
    status: "illustrative",
    note: "Your exports, prints and filter changes are recorded for real, in this browser only. System events are invented for now.",
  },
  {
    area: "Market open/closed indicator",
    source: "Yahoo Finance S&P 500 feed, checked against the NYSE calendar",
    status: "real",
    note: "Live. Falls back to the calendar alone, marked (est.), if the feed is unavailable.",
  },
];
