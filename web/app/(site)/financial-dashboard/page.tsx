import { redirect } from "next/navigation";
import type { SearchParams } from "@/lib/page-helpers";
import { tabHref } from "@/lib/tabs";

export default function FinancialDashboardIndex({ searchParams }: { searchParams: SearchParams }) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(searchParams)) {
    for (const v of Array.isArray(value) ? value : value === undefined ? [] : [value]) params.append(key, v);
  }
  redirect(tabHref("portfolio-matrix", params.toString()));
}
