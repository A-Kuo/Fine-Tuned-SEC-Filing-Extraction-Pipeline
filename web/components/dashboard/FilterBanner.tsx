"use client";

import { ChevronDown, Printer, FileDown, ScrollText } from "lucide-react";

const FILTERS = [
  { label: "Asset Class", value: "All Classes" },
  { label: "Entity Hierarchy", value: "Consolidated" },
  { label: "Filing Window", value: "Q3 FY2026" },
];

const ACTIONS = [
  { label: "Print Ledger", Icon: Printer },
  { label: "Export to XBRL/CSV", Icon: FileDown },
  { label: "Audit Logs", Icon: ScrollText },
];

export function FilterBanner() {
  return (
    <div className="border-b border-border-formal bg-surface">
      <div className="px-6 pt-5 pb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-primary-navy tracking-tight">
            Form 10-K Consolidated Analytics
          </h1>
          <p className="text-xs text-text-muted mt-1 font-mono">
            Data Normalized: As of Q3 Audit Clearings
          </p>
        </div>
      </div>

      <div className="px-6 pb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.label}
              className="flex items-center gap-2 h-8 px-3 bg-surface border border-border-formal text-xs text-text-secondary hover:border-secondary-blue/50 transition-colors"
            >
              <span className="text-text-muted">{f.label}:</span>
              <span className="font-medium text-text-primary">{f.value}</span>
              <ChevronDown className="h-3 w-3 text-text-muted" aria-hidden />
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          {ACTIONS.map(({ label, Icon }) => (
            <button
              key={label}
              title={label}
              className="flex items-center gap-1.5 h-8 px-2.5 bg-surface border border-border-formal text-xs text-text-secondary hover:bg-subtle transition-colors"
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              <span className="hidden xl:inline">{label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
