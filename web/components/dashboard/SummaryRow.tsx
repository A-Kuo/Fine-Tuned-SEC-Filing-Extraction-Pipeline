import { summaryIndicators } from "@/lib/mock-data";

const TONE_TEXT: Record<string, string> = {
  gain: "text-market-gain",
  neutral: "text-secondary-blue",
  muted: "text-accent-slate",
  amber: "text-amber-700",
};

export function SummaryRow() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 border-b border-border-formal bg-surface">
      {summaryIndicators.map((item, i) => (
        <div
          key={item.label}
          className={`px-6 py-4 ${i > 0 ? "border-t sm:border-t-0 sm:border-l" : ""} border-slate-300`}
        >
          <p className="text-[11px] uppercase tracking-wide text-text-muted font-medium">
            {item.label}
          </p>
          <p className="text-2xl font-semibold text-text-primary tabular-figures mt-1.5">
            {item.value}
          </p>
          <p className={`text-xs font-medium mt-1 ${TONE_TEXT[item.tone]}`}>{item.delta}</p>
        </div>
      ))}
    </div>
  );
}
