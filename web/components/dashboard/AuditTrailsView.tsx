"use client";

import { useEffect, useMemo, useRef } from "react";
import { recordAuditEvent, useLocalAuditEvents } from "@/lib/audit";
import { auditDataset } from "@/lib/export";
import { inRange, type DateRange, type FilterState, type Option } from "@/lib/filters";
import { formatUtcTimestamp } from "@/lib/format";
import type { AuditEvent } from "@/lib/types";
import { FilterBanner } from "./FilterBanner";

interface AuditTrailsViewProps {
  title: string;
  subtitle: string;
  seeded: AuditEvent[];
  filters: FilterState;
  options: { asset: Option[]; entity: Option[]; window: Option[] };
  range: DateRange;
  // Events recorded in this browser happen "now", so they only belong in
  // windows that reach the present.
  includesPresent: boolean;
}

const byNewest = (a: AuditEvent, b: AuditEvent) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0);
const HEADERS = ["Timestamp (UTC)", "Actor", "Action", "Object", "Detail", "Source"];

export function AuditTrailsView({ title, subtitle, seeded, filters, options, range, includesPresent }: AuditTrailsViewProps) {
  const local = useLocalAuditEvents();
  const viewLogged = useRef(false);

  useEffect(() => {
    if (viewLogged.current) return;
    viewLogged.current = true;
    recordAuditEvent({ action: "audit.view", object: "Audit Trails", detail: "Opened the audit trail" });
  }, []);

  const events = useMemo(() => {
    const mine = includesPresent ? local.filter((e) => range.from === null || e.ts.slice(0, 10) >= range.from) : [];
    const system = seeded.filter((e) => inRange(e.ts.slice(0, 10), range));
    return [...mine, ...system].sort(byNewest);
  }, [local, seeded, range, includesPresent]);

  return (
    <>
      <FilterBanner
        tab="audit-trails"
        title={title}
        subtitle={subtitle}
        filters={filters}
        options={options}
        exportDataset={auditDataset(events)}
      />

      <div className="px-6 py-6">
        <div className="border border-border-formal bg-surface rounded-md shadow-flat overflow-hidden">
          <div className="px-4 py-3 border-b border-border-formal bg-subtle flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 className="text-sm font-semibold text-primary-navy">Event Log</h2>
            <span className="text-[11px] text-text-muted font-mono tabular-figures" aria-live="polite">
              {events.length} {events.length === 1 ? "event" : "events"}
            </span>
          </div>

          <p className="px-4 py-2 text-[11px] text-text-muted border-b border-border-formal">
            Events marked <span className="font-semibold text-secondary-blue">This browser</span> are recorded in this
            browser&apos;s local storage only. They show what you did in this interface and are not a compliance-grade
            audit log.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-subtle text-primary-navy">
                  {HEADERS.map((h) => (
                    <th key={h} scope="col" className="text-left font-semibold px-4 py-2 border-b border-border-formal whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {events.length === 0 ? (
                  <tr>
                    <td colSpan={HEADERS.length} className="px-4 py-10 text-center text-text-muted">
                      No audit events in this window.
                    </td>
                  </tr>
                ) : (
                  events.map((e, i) => (
                    <tr key={e.id} className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} border-b border-border-formal last:border-b-0`}>
                      <td className="px-4 py-2 font-mono tabular-figures text-text-secondary whitespace-nowrap">{formatUtcTimestamp(e.ts)}</td>
                      <td className="px-4 py-2 text-text-secondary whitespace-nowrap">{e.actor}</td>
                      <td className="px-4 py-2 font-mono text-text-primary whitespace-nowrap">{e.action}</td>
                      <td className="px-4 py-2 text-text-secondary">{e.object}</td>
                      <td className="px-4 py-2 text-text-secondary">{e.detail}</td>
                      <td className="px-4 py-2">
                        <span
                          className={`inline-block rounded-sm border px-2 py-0.5 text-[11px] whitespace-nowrap ${
                            e.source === "local"
                              ? "bg-blue-50 text-secondary-blue border-blue-200 font-semibold"
                              : "bg-slate-100 text-slate-700 border-slate-300"
                          }`}
                        >
                          {e.source === "local" ? "This browser" : "System"}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}
