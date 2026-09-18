import type { Metadata } from "next";
import { FilterBanner } from "@/components/dashboard/FilterBanner";
import { YieldChart } from "@/components/dashboard/YieldChart";
import { YieldTable } from "@/components/dashboard/YieldTable";
import { DATA_AS_OF, getAllocation, getYieldSeries, getYieldSummary } from "@/lib/data";
import { yieldDataset } from "@/lib/export";
import { windowSubtitle } from "@/lib/filters";
import { filterOptions, filtersFrom, type SearchParams } from "@/lib/page-helpers";
import { tabBySlug } from "@/lib/tabs";

export const metadata: Metadata = { title: "Market Yields" };

export default function MarketYieldsPage({ searchParams }: { searchParams: SearchParams }) {
  const filters = filtersFrom(searchParams);
  const series = getYieldSeries(filters.asset, filters.window);
  const seriesLabel = getAllocation().find((a) => a.slug === filters.asset)?.label ?? "Blended Portfolio";

  return (
    <>
      <FilterBanner
        tab="market-yields"
        title={tabBySlug("market-yields").title}
        subtitle={windowSubtitle(filters.window, DATA_AS_OF)}
        filters={filters}
        options={filterOptions()}
        exportDataset={yieldDataset(series, seriesLabel)}
      />
      <div className="px-6 py-6 flex flex-col gap-6">
        <YieldChart series={series} seriesLabel={seriesLabel} />
        <YieldTable rows={getYieldSummary(filters.asset, filters.window)} />
      </div>
    </>
  );
}
