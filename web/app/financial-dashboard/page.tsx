import { TopNav } from "@/components/dashboard/TopNav";
import { FilterBanner } from "@/components/dashboard/FilterBanner";
import { SummaryRow } from "@/components/dashboard/SummaryRow";
import { YieldChart } from "@/components/dashboard/YieldChart";
import { AllocationMatrix } from "@/components/dashboard/AllocationMatrix";
import { FilingsLedger } from "@/components/dashboard/FilingsLedger";

export const metadata = {
  title: "Form 10-K Consolidated Analytics",
};

export default function FinancialDashboardPage() {
  return (
    <main className="min-h-screen bg-canvas">
      <TopNav />
      <FilterBanner />
      <SummaryRow />

      <div className="px-6 py-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3">
          <YieldChart />
        </div>
        <div className="lg:col-span-2">
          <AllocationMatrix />
        </div>
      </div>

      <div className="px-6 pb-8">
        <FilingsLedger />
      </div>
    </main>
  );
}
