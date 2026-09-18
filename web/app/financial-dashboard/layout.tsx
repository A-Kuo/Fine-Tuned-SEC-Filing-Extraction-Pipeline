import { Suspense } from "react";
import { TopNav } from "@/components/dashboard/TopNav";

// TopNav reads the URL's search params; the fallback is a same-height navy bar
// so the page doesn't jump while it hydrates.
function NavFallback() {
  return <div className="h-14 bg-primary-navy print:hidden" aria-hidden />;
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      <Suspense fallback={<NavFallback />}>
        <TopNav />
      </Suspense>

      <main className="flex-1">{children}</main>

      <footer className="border-t border-border-formal bg-surface px-6 py-3 text-[11px] text-text-muted flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
        <span>
          <span className="font-semibold text-text-secondary">Illustrative data.</span> Issuers, holdings, yields and
          audit events are invented for demonstration and are not real filings.
        </span>
        <span className="print:hidden">Market status: Yahoo Finance, with an NYSE-calendar fallback.</span>
      </footer>
    </div>
  );
}
