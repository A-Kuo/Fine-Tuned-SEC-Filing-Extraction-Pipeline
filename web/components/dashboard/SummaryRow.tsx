import type { SummaryIndicator, SummaryTone } from "@/lib/types";

const TONE_TEXT: Record<SummaryTone, string> = {
  gain: "text-market-gain",
  loss: "text-market-loss",
  neutral: "text-secondary-blue",
  muted: "text-accent-slate",
  amber: "text-amber-700",
};

export function SummaryRow({ items }: { items: SummaryIndicator[] }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 border-b border-border-formal bg-surface print:grid-cols-4">
      {items.map((item, i) => (
        <div
          key={item.label}
          className={`px-6 py-4 border-slate-300 ${i > 0 ? "border-t sm:border-t-0 sm:border-l" : ""} ${
            i >= 2 ? "sm:max-lg:border-t" : ""
          } ${i === 2 ? "sm:max-lg:border-l-0" : ""}`}
        >
          <p className="text-[11px] uppercase tracking-wide text-text-muted font-medium">{item.label}</p>
          <p className="text-2xl font-semibold text-text-primary tabular-figures mt-1.5">{item.value}</p>
          <p className={`text-xs font-medium mt-1 ${TONE_TEXT[item.tone]}`}>{item.delta}</p>
        </div>
      ))}
    </div>
  );
}
