import { formatUsdCompact, signed } from "@/lib/format";
import type { AllocationSegment } from "@/lib/types";

const HEADERS = ["Asset Class", "Weight", "Market Value", "YoY", "Yield"];

export function AllocationTable({ rows, showTotal }: { rows: AllocationSegment[]; showTotal: boolean }) {
  const totalValue = rows.reduce((sum, r) => sum + r.marketValueUsd, 0);
  const totalWeight = rows.reduce((sum, r) => sum + r.pct, 0);
  const num = "px-4 py-2.5 text-right tabular-figures";

  return (
    <div className="border border-border-formal bg-surface shadow-flat overflow-hidden">
      <div className="px-4 py-3 border-b border-border-formal bg-subtle">
        <h2 className="text-sm font-semibold text-primary-navy">Positions by Asset Class</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-subtle text-primary-navy">
              {HEADERS.map((h, i) => (
                <th
                  key={h}
                  scope="col"
                  className={`font-semibold px-4 py-2 border-b border-border-formal ${i === 0 ? "text-left" : "text-right"}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.slug} className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} border-b border-border-formal`}>
                <td className="px-4 py-2.5 font-medium text-text-primary">{r.label}</td>
                <td className={num}>{r.pct.toFixed(1)}%</td>
                <td className={num}>{formatUsdCompact(r.marketValueUsd)}</td>
                <td className={`${num} ${r.yoyPct >= 0 ? "text-market-gain" : "text-market-loss"}`}>{signed(r.yoyPct)}%</td>
                <td className={num}>{r.yieldPct.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
          {showTotal && (
            <tfoot>
              <tr className="bg-subtle font-semibold text-primary-navy">
                <td className="px-4 py-2.5">Total</td>
                <td className={num}>{totalWeight.toFixed(1)}%</td>
                <td className={num}>{formatUsdCompact(totalValue)}</td>
                <td className={num} />
                <td className={num} />
              </tr>
            </tfoot>
          )}
        </table>
      </div>
    </div>
  );
}
