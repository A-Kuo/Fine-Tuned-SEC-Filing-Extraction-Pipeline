import type { Metadata } from "next";
import { AuditTrailsView } from "@/components/dashboard/AuditTrailsView";
import { DATA_AS_OF, getAuditSeed } from "@/lib/data";
import { windowRange, windowSubtitle } from "@/lib/filters";
import { filterOptions, filtersFrom, type SearchParams } from "@/lib/page-helpers";
import { tabBySlug } from "@/lib/tabs";

export const metadata: Metadata = { title: "Audit Trails" };

export default function AuditTrailsPage({ searchParams }: { searchParams: SearchParams }) {
  const filters = filtersFrom(searchParams);
  const range = windowRange(filters.window, DATA_AS_OF);

  return (
    <AuditTrailsView
      title={tabBySlug("audit-trails").title}
      subtitle={windowSubtitle(filters.window, DATA_AS_OF)}
      seeded={getAuditSeed()}
      filters={filters}
      options={filterOptions()}
      range={range}
      includesPresent={range.to === DATA_AS_OF}
    />
  );
}
