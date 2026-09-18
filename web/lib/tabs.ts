import type { FilterApplicability } from "./filters";

export type TabSlug = "portfolio-matrix" | "regulatory-filings" | "market-yields" | "audit-trails";

export interface TabConfig {
  slug: TabSlug;
  label: string;
  title: string;
  applies: FilterApplicability;
}

export const BASE_PATH = "/financial-dashboard";

export const TABS: TabConfig[] = [
  {
    slug: "portfolio-matrix",
    label: "Portfolio Matrix",
    title: "Portfolio Allocation Analytics",
    applies: { asset: true, entity: false, window: false, search: false },
  },
  {
    slug: "regulatory-filings",
    label: "Regulatory Filings",
    title: "Form 10-K Consolidated Analytics",
    applies: { asset: false, entity: true, window: true, search: true },
  },
  {
    slug: "market-yields",
    label: "Market Yields",
    title: "Market Yield Analytics",
    applies: { asset: true, entity: false, window: true, search: false },
  },
  {
    slug: "audit-trails",
    label: "Audit Trails",
    title: "Audit Trail & Access Log",
    applies: { asset: false, entity: false, window: true, search: false },
  },
];

export function tabBySlug(slug: TabSlug): TabConfig {
  return TABS.find((t) => t.slug === slug)!;
}

export function tabHref(slug: TabSlug, query = ""): string {
  return `${BASE_PATH}/${slug}${query ? `?${query}` : ""}`;
}
