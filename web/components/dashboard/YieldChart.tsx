"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatChartTick } from "@/lib/format";
import type { YieldPoint } from "@/lib/types";

interface TooltipEntry {
  dataKey: string;
  name: string;
  value: number;
  color: string;
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipEntry[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-none bg-surface border border-primary-navy px-3 py-2 text-xs shadow-flat">
      <p className="font-mono text-text-muted mb-1">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.color }} className="tabular-figures font-medium">
          {entry.name}: {entry.value.toFixed(2)}%
        </p>
      ))}
    </div>
  );
}

// Short windows get day-level ticks ("Sep 16"); long ones get month-level ("Sep '26").
const SHORT_WINDOW_POINTS = 14;

export function YieldChart({ series, seriesLabel }: { series: YieldPoint[]; seriesLabel: string }) {
  const short = series.length <= SHORT_WINDOW_POINTS;

  // Points are semi-monthly, so month-level labels would repeat. Label only the
  // first point of each month, thinned to roughly 8 ticks.
  const monthStarts = series.filter((p) => p.date.endsWith("-01")).map((p) => p.date);
  const step = Math.max(1, Math.ceil(monthStarts.length / 8));
  const ticks = short ? undefined : monthStarts.filter((_, i) => i % step === 0);

  return (
    <div className="border border-border-formal bg-surface shadow-flat">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 px-4 py-3 border-b border-border-formal">
        <h2 className="text-sm font-semibold text-primary-navy">Historical Cumulative Yields &amp; Filing Volatility</h2>
        <span className="text-[11px] text-text-muted font-mono">
          {seriesLabel} &middot; {series.length} {series.length === 1 ? "observation" : "observations"}
        </span>
      </div>

      {series.length === 0 ? (
        <div className="h-72 flex items-center justify-center text-xs text-text-muted">No observations in this window.</div>
      ) : (
        <div className="h-72 px-2 pt-4 pb-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 4, right: 16, left: 4, bottom: 0 }}>
              <CartesianGrid stroke="hsl(217 19% 82%)" strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="date"
                ticks={ticks}
                tick={{ fontSize: 11, fill: "hsl(215 14% 56%)" }}
                axisLine={{ stroke: "hsl(217 19% 82%)" }}
                tickLine={false}
                tickFormatter={(d: string) => formatChartTick(d, short)}
                minTickGap={24}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "hsl(215 14% 56%)" }}
                axisLine={{ stroke: "hsl(217 19% 82%)" }}
                tickLine={false}
                width={38}
                tickFormatter={(v: number) => `${v}%`}
              />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: "hsl(217 19% 82%)" }} />
              <Line
                type="linear"
                dataKey="yield"
                name="Cumulative Yield"
                stroke="hsl(219 46% 16%)"
                strokeWidth={2}
                dot={short ? { r: 2.5, fill: "hsl(219 46% 16%)" } : false}
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="volatility"
                name="Filing Volatility"
                stroke="hsl(211 100% 35%)"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                dot={short ? { r: 2, fill: "hsl(211 100% 35%)" } : false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="flex gap-5 px-4 pb-3 text-[11px] text-text-secondary">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 bg-primary-navy" /> Cumulative Yield
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 bg-secondary-blue" /> Filing Volatility
        </span>
      </div>
    </div>
  );
}
