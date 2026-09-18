import type { AllocationSegment } from "@/lib/types";

const SHADES = ["bg-primary-navy", "bg-secondary-blue", "bg-accent-slate", "bg-slate-300"];

export function AllocationMatrix({ segments, selected }: { segments: AllocationSegment[]; selected: string }) {
  const dim = (slug: string) => (selected !== "all" && slug !== selected ? "opacity-30" : "");

  return (
    <div className="border border-border-formal bg-surface shadow-flat h-full flex flex-col">
      <div className="flex items-baseline justify-between px-4 py-3 border-b border-border-formal">
        <h2 className="text-sm font-semibold text-primary-navy">Asset Allocation Matrix</h2>
        <span className="text-[11px] text-text-muted font-mono">consolidated %</span>
      </div>

      <div className="px-4 pt-4">
        <div className="flex h-6 w-full border border-border-formal overflow-hidden">
          {segments.map((seg, i) => (
            <div
              key={seg.slug}
              className={`${SHADES[i % SHADES.length]} ${dim(seg.slug)}`}
              style={{ width: `${seg.pct}%` }}
              title={`${seg.label}: ${seg.pct}%`}
            />
          ))}
        </div>
      </div>

      <div className="px-4 py-4 flex-1 flex flex-col divide-y divide-border-formal">
        {segments.map((seg, i) => (
          <div
            key={seg.slug}
            className={`flex items-center justify-between py-2.5 first:pt-0 last:pb-0 ${dim(seg.slug)}`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span className={`inline-block w-2.5 h-2.5 shrink-0 ${SHADES[i % SHADES.length]}`} />
              <span className="text-xs text-text-secondary truncate">{seg.label}</span>
            </div>
            <span className="text-xs font-semibold text-text-primary tabular-figures shrink-0">
              {seg.pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
