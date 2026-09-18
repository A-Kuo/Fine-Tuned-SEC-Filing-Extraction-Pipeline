export function formatUsdCompact(value: number): string {
  if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString("en-US")}`;
}

export function signed(value: number, digits = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

// Deterministic (no locale/timezone) so server and client render identically.
export function formatUtcTimestamp(iso: string): string {
  return `${iso.slice(0, 19).replace("T", " ")}Z`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatChartTick(isoDate: string, short: boolean): string {
  const [year, month, day] = isoDate.split("-").map(Number);
  return short ? `${MONTHS[month - 1]} ${day}` : `${MONTHS[month - 1]} '${String(year).slice(2)}`;
}
