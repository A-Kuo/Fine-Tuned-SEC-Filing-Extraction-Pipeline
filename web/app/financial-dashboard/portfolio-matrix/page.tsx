import type { Metadata } from "next";
import { AllocationMatrix } from "@/components/dashboard/AllocationMatrix";
import { AllocationTable } from "@/components/dashboard/AllocationTable";
import { FilterBanner } from "@/components/dashboard/FilterBanner";
import { SummaryRow } from "@/components/dashboard/SummaryRow";
import { DATA_AS_OF, getAllocation, getSummaryIndicators } from "@/lib/data";
import { allocationDataset } from "@/lib/export";
import { filterOptions, filtersFrom, type SearchParams } from "@/lib/page-helpers";
import { tabBySlug } from "@/lib/tabs";

export const metadata: Metadata = { title: "Portfolio Matrix" };

export default function PortfolioMatrixPage({ searchParams }: { searchParams: SearchParams }) {
  const filters = filtersFrom(searchParams);
  const allocation = getAllocation();
  const visible = filters.asset === "all" ? allocation : allocation.filter((a) => a.slug === filters.asset);

  return (
    <>
      <FilterBanner
        tab="portfolio-matrix"
        title={tabBySlug("portfolio-matrix").title}
        subtitle={`Data Normalized: Positions as of ${DATA_AS_OF}`}
        filters={filters}
        options={filterOptions()}
        exportDataset={allocationDataset(visible)}
      />
      <SummaryRow items={getSummaryIndicators(filters.asset)} />
      <div className="px-6 py-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2">
          <AllocationMatrix segments={allocation} selected={filters.asset} />
        </div>
        <div className="lg:col-span-3">
          <AllocationTable rows={visible} showTotal={filters.asset === "all"} />
        </div>
      </div>
    </>
  );
}
