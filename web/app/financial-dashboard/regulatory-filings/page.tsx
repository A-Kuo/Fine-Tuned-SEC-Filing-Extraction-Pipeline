import type { Metadata } from "next";
import { FilingsLedger } from "@/components/dashboard/FilingsLedger";
import { FilterBanner } from "@/components/dashboard/FilterBanner";
import { SummaryRow } from "@/components/dashboard/SummaryRow";
import { DATA_AS_OF, getFilings, getFilingStats } from "@/lib/data";
import { filingsDataset } from "@/lib/export";
import { applyFilingFilters, windowSubtitle } from "@/lib/filters";
import { filterOptions, filtersFrom, type SearchParams } from "@/lib/page-helpers";
import { tabBySlug } from "@/lib/tabs";

export const metadata: Metadata = { title: "Regulatory Filings" };

export default function RegulatoryFilingsPage({ searchParams }: { searchParams: SearchParams }) {
  const filters = filtersFrom(searchParams);
  const rows = applyFilingFilters(getFilings(), filters, DATA_AS_OF);

  return (
    <>
      <FilterBanner
        tab="regulatory-filings"
        title={tabBySlug("regulatory-filings").title}
        subtitle={windowSubtitle(filters.window, DATA_AS_OF)}
        filters={filters}
        options={filterOptions()}
        exportDataset={filingsDataset(rows)}
        xbrlFilings={rows}
      />
      <SummaryRow items={getFilingStats(rows)} />
      <div className="px-6 py-6">
        <FilingsLedger rows={rows} />
      </div>
    </>
  );
}
