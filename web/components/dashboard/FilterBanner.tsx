"use client";

import { Printer, ScrollText } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef } from "react";
import { Dropdown } from "@/components/ui/Dropdown";
import { recordAuditEvent } from "@/lib/audit";
import type { Dataset } from "@/lib/export";
import { describeFilters, withFilter, type FilterState, type Option } from "@/lib/filters";
import { tabBySlug, tabHref, type TabSlug } from "@/lib/tabs";
import type { FilingRow } from "@/lib/types";
import { ExportMenu } from "./ExportMenu";

type DropdownKey = "asset" | "entity" | "window";

const CONTROLS: Array<{ key: DropdownKey; label: string }> = [
  { key: "asset", label: "Asset Class" },
  { key: "entity", label: "Entity Hierarchy" },
  { key: "window", label: "Filing Window" },
];

interface FilterBannerProps {
  tab: TabSlug;
  title: string;
  subtitle: string;
  filters: FilterState;
  options: { asset: Option[]; entity: Option[]; window: Option[] };
  exportDataset: Dataset;
  xbrlFilings?: FilingRow[];
}

const buttonClass =
  "flex items-center gap-1.5 h-8 px-2.5 bg-surface border border-border-formal text-xs text-text-secondary hover:bg-subtle transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-secondary-blue";

export function FilterBanner({ tab, title, subtitle, filters, options, exportDataset, xbrlFilings }: FilterBannerProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const config = tabBySlug(tab);
  const filterSummary = describeFilters(filters, config.applies, options);
  const printedAt = useRef<HTMLTimeElement>(null);

  // beforeprint also fires for Ctrl+P, and the DOM must be current
  // synchronously, so the timestamp is written directly rather than via state.
  useEffect(() => {
    const stamp = () => {
      if (printedAt.current) printedAt.current.textContent = new Date().toLocaleString();
    };
    window.addEventListener("beforeprint", stamp);
    return () => window.removeEventListener("beforeprint", stamp);
  }, []);

  const change = (key: DropdownKey, value: string) => {
    const opts = options[key];
    const from = opts.find((o) => o.value === filters[key])?.label ?? filters[key];
    const to = opts.find((o) => o.value === value)?.label ?? value;
    recordAuditEvent({
      action: "filter.change",
      object: CONTROLS.find((c) => c.key === key)!.label,
      detail: `Changed from "${from}" to "${to}" on ${config.label}`,
    });
    const next = withFilter(searchParams, key, value);
    router.replace(next ? `${pathname}?${next}` : pathname, { scroll: false });
  };

  const printLedger = () => {
    recordAuditEvent({ action: "print.ledger", object: config.label, detail: filterSummary });
    setTimeout(() => window.print(), 0);
  };

  return (
    <div className="border-b border-border-formal bg-surface">
      <div className="hidden print:block px-6 pt-4 pb-3 border-b-2 border-black text-black">
        <p className="text-[11px] font-semibold tracking-wide uppercase">EDGAR-X Disclosure Matrix</p>
        <p className="text-xs mt-1">{filterSummary}</p>
        <p className="text-[11px] mt-1">
          Printed <time ref={printedAt} /> &middot; Illustrative data, not real filings or holdings
        </p>
      </div>

      <div className="px-6 pt-5 pb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-primary-navy tracking-tight">{title}</h1>
          <p className="text-xs text-text-muted mt-1 font-mono">{subtitle}</p>
        </div>
      </div>

      <div className="px-6 pb-4 flex flex-wrap items-center justify-between gap-3 print:hidden">
        <div className="flex flex-wrap gap-2">
          {CONTROLS.map(({ key, label }) => (
            <Dropdown
              key={key}
              label={label}
              options={options[key]}
              value={filters[key]}
              onChange={(value) => change(key, value)}
              disabled={!config.applies[key]}
              disabledReason={`${label} does not apply to ${config.label}`}
            />
          ))}
        </div>

        <div className="flex gap-2">
          <button type="button" onClick={printLedger} title="Print Ledger" className={buttonClass}>
            <Printer className="h-3.5 w-3.5" aria-hidden />
            <span className="hidden xl:inline">Print Ledger</span>
            <span className="xl:hidden sr-only">Print Ledger</span>
          </button>

          <ExportMenu
            tabSlug={tab}
            tabLabel={config.label}
            dataset={exportDataset}
            xbrlFilings={xbrlFilings}
            filterSummary={filterSummary}
          />

          <Link href={tabHref("audit-trails", searchParams.toString())} title="Audit Logs" className={buttonClass}>
            <ScrollText className="h-3.5 w-3.5" aria-hidden />
            <span className="hidden xl:inline">Audit Logs</span>
            <span className="xl:hidden sr-only">Audit Logs</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
