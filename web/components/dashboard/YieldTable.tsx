import { signed } from "@/lib/format";
import type { YieldSummaryRow } from "@/lib/data";

const HEADERS = ["Series", "Weight", "Latest Yield", "Change", "High", "Low", "Avg. Volatility"];
const pct = (v: number | null) => (v === null ? "n/a" : `${v.toFixed(2)}%`);

export function YieldTable({ rows }: { rows: YieldSummaryRow[] }) {
  const num = "px-4 py-2.5 text-right tabular-figures";

  return (
    <div className="border border-border-formal bg-surface shadow-flat overflow-hidden">
      <div className="px-4 py-3 border-b border-border-formal bg-subtle">
        <h2 className="text-sm font-semibold text-primary-navy">Yield Summary by Series</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-subtle text-primary-navy">
              {HEADERS.map((h, i) => (
                <th
                  key={h}
                  scope="col"
                  className={`font-semibold px-4 py-2 border-b border-border-formal whitespace-nowrap ${i === 0 ? "text-left" : "text-right"}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={r.slug}
                className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} ${r.slug === "blended" ? "font-semibold bg-subtle" : ""} border-b border-border-formal last:border-b-0`}
              >
                <td className="px-4 py-2.5 font-medium text-text-primary whitespace-nowrap">{r.label}</td>
                <td className={num}>{r.weight === null ? "n/a" : `${r.weight}%`}</td>
                <td className={num}>{pct(r.latest)}</td>
                <td
                  className={`${num} ${r.change === null ? "" : r.change >= 0 ? "text-market-gain" : "text-market-loss"}`}
                >
                  {r.change === null ? "n/a" : `${signed(r.change, 2)} pts`}
                </td>
                <td className={num}>{pct(r.high)}</td>
                <td className={num}>{pct(r.low)}</td>
                <td className={num}>{pct(r.avgVolatility)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
