"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { yieldSeries } from "@/lib/mock-data";

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-none bg-surface border border-primary-navy px-3 py-2 text-xs shadow-flat">
      <p className="font-mono text-text-muted mb-1">{label}</p>
      {payload.map((entry: any) => (
        <p key={entry.dataKey} style={{ color: entry.color }} className="tabular-figures font-medium">
          {entry.name}: {entry.value.toFixed(2)}%
        </p>
      ))}
    </div>
  );
}

export function YieldChart() {
  return (
    <div className="border border-border-formal bg-surface shadow-flat">
      <div className="flex items-baseline justify-between px-4 py-3 border-b border-border-formal">
        <h2 className="text-sm font-semibold text-primary-navy">
          Historical Cumulative Yields &amp; Filing Volatility
        </h2>
        <span className="text-[11px] text-text-muted font-mono">8-quarter trailing</span>
      </div>
      <div className="h-72 px-2 pt-4 pb-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={yieldSeries} margin={{ top: 4, right: 16, left: 4, bottom: 0 }}>
            <CartesianGrid stroke="hsl(217 19% 82%)" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="period"
              tick={{ fontSize: 11, fill: "hsl(215 14% 56%)" }}
              axisLine={{ stroke: "hsl(217 19% 82%)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(215 14% 56%)" }}
              axisLine={{ stroke: "hsl(217 19% 82%)" }}
              tickLine={false}
              width={38}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "hsl(217 19% 82%)" }} />
            <Line
              type="linear"
              dataKey="cumulativeYield"
              name="Cumulative Yield"
              stroke="hsl(219 46% 16%)"
              strokeWidth={2}
              dot={{ r: 2.5, fill: "hsl(219 46% 16%)" }}
              isAnimationActive={false}
            />
            <Line
              type="linear"
              dataKey="filingVolatility"
              name="Filing Volatility"
              stroke="hsl(211 100% 35%)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={{ r: 2, fill: "hsl(211 100% 35%)" }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
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
