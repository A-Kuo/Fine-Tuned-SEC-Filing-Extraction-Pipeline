import { NextResponse } from "next/server";
import { deriveSession } from "@/lib/market-hours";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The keyless chart endpoint with the smallest payload (~1 KB). It carries
// `regularMarketTime` and `currentTradingPeriod`, but no `marketState`, so the
// state is derived (see deriveSession). Yahoo's API is unofficial: if it is
// unreachable, rate-limits this host, or changes shape, deriveSession() falls
// back to the computed NYSE calendar and the response says so.
const YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=1d&interval=1d";
const TIMEOUT_MS = 4000;

async function fetchYahooMeta(): Promise<unknown> {
  try {
    const res = await fetch(YAHOO_URL, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; EDGAR-X/1.0)", Accept: "application/json" },
      signal: AbortSignal.timeout(TIMEOUT_MS),
      cache: "no-store",
    });
    if (!res.ok) return null;
    const body = await res.json();
    return body?.chart?.result?.[0]?.meta ?? null;
  } catch {
    return null;
  }
}

export async function GET() {
  const session = deriveSession(Date.now(), await fetchYahooMeta());
  return NextResponse.json(session, {
    // Shared edge cache: however many visitors poll, upstream sees ~1 call/min.
    headers: { "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120" },
  });
}
