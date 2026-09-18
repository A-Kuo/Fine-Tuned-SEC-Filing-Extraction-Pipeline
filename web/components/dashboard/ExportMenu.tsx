"use client";

import { ChevronDown, FileCode2, FileDown, FileSpreadsheet } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { useDismiss } from "@/components/ui/Dropdown";
import { recordAuditEvent } from "@/lib/audit";
import { downloadText, exportFilename, toCsv, toXbrlInstance, type Dataset } from "@/lib/export";
import type { FilingRow } from "@/lib/types";

interface ExportMenuProps {
  tabSlug: string;
  tabLabel: string;
  dataset: Dataset;
  xbrlFilings?: FilingRow[];
  filterSummary: string;
}

export function ExportMenu({ tabSlug, tabLabel, dataset, xbrlFilings, filterSummary }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const baseId = useId();
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, rootRef);

  useEffect(() => {
    if (open) menuRef.current?.focus();
  }, [open]);

  const xbrlAvailable = xbrlFilings !== undefined;

  const items = [
    {
      key: "csv",
      label: "Export CSV",
      hint: `${dataset.rows.length} rows, current filters`,
      Icon: FileSpreadsheet,
      disabled: false,
      run: () => {
        downloadText(exportFilename(tabSlug, "csv"), "text/csv;charset=utf-8", toCsv(dataset));
        recordAuditEvent({ action: "export.csv", object: tabLabel, detail: `${dataset.rows.length} rows. ${filterSummary}` });
      },
    },
    {
      key: "xbrl",
      label: "Export XBRL",
      hint: xbrlAvailable ? "dei cover-page facts only" : "Available on Regulatory Filings",
      Icon: FileCode2,
      disabled: !xbrlAvailable,
      run: () => {
        downloadText(exportFilename(tabSlug, "xml"), "application/xml;charset=utf-8", toXbrlInstance(xbrlFilings ?? []));
        recordAuditEvent({ action: "export.xbrl", object: tabLabel, detail: `${xbrlFilings?.length ?? 0} filings (dei cover-page facts). ${filterSummary}` });
      },
    },
  ];

  const step = (from: number, dir: 1 | -1) => {
    for (let i = from + dir; i >= 0 && i < items.length; i += dir) if (!items[i].disabled) return i;
    return from;
  };

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActive((i) => step(i, 1));
        break;
      case "ArrowUp":
        event.preventDefault();
        setActive((i) => step(i, -1));
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        if (!items[active].disabled) {
          items[active].run();
          setOpen(false);
          triggerRef.current?.focus();
        }
        break;
      case "Escape":
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div ref={rootRef} className="relative print:hidden">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? `${baseId}-menu` : undefined}
        onClick={() => {
          setActive(0);
          setOpen((o) => !o);
        }}
        className="flex items-center gap-1.5 h-8 px-2.5 bg-surface border border-border-formal text-xs text-text-secondary hover:bg-subtle transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-secondary-blue"
      >
        <FileDown className="h-3.5 w-3.5" aria-hidden />
        <span className="hidden xl:inline">Export to XBRL/CSV</span>
        <span className="xl:hidden sr-only">Export</span>
        <ChevronDown className="h-3 w-3 text-text-muted" aria-hidden />
      </button>

      {open && (
        <div
          ref={menuRef}
          id={`${baseId}-menu`}
          role="menu"
          tabIndex={-1}
          aria-label="Export"
          aria-activedescendant={`${baseId}-item-${active}`}
          onKeyDown={onMenuKeyDown}
          className="absolute right-0 top-full z-30 mt-px w-64 bg-surface border border-primary-navy shadow-flat outline-none"
        >
          {items.map((item, i) => (
            <button
              key={item.key}
              id={`${baseId}-item-${i}`}
              type="button"
              role="menuitem"
              disabled={item.disabled}
              onMouseEnter={() => !item.disabled && setActive(i)}
              onClick={() => {
                item.run();
                setOpen(false);
                triggerRef.current?.focus();
              }}
              className={`w-full flex items-start gap-2.5 px-3 py-2 text-left text-xs border-b border-border-formal last:border-b-0 ${
                item.disabled ? "text-text-muted cursor-not-allowed opacity-60" : i === active ? "bg-subtle text-primary-navy" : "text-text-secondary"
              }`}
            >
              <item.Icon className="h-3.5 w-3.5 mt-0.5 shrink-0" aria-hidden />
              <span>
                <span className="block font-medium">{item.label}</span>
                <span className="block text-[11px] text-text-muted">{item.hint}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
