"use client";

import { CircleDot } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { MarketSession } from "@/lib/market-hours";

const POLL_MS = 60_000;
// After this long without a successful poll, stop showing the last known state:
// a stale "Market Open" is worse than admitting the feed is down.
const STALE_MS = 3 * POLL_MS;

const TONES = {
  open: "bg-market-gain-bg text-market-gain border-market-gain/40",
  extended: "bg-amber-50 text-amber-800 border-amber-300",
  closed: "bg-slate-100 text-slate-700 border-slate-300",
  unknown: "bg-white/10 text-white/70 border-white/20",
} as const;

function describe(session: MarketSession | null) {
  if (!session) return { text: "Status unavailable", tone: TONES.unknown };
  const est = session.source === "computed" ? " (est.)" : "";
  switch (session.state) {
    case "REGULAR":
      return { text: `Market Open${est}`, tone: TONES.open };
    case "PRE":
      return { text: `Pre-Market${est}`, tone: TONES.extended };
    case "POST":
      return { text: `After-Hours${est}`, tone: TONES.extended };
    default:
      return { text: `${session.holiday ? `Closed: ${session.holiday}` : "Market Closed"}${est}`, tone: TONES.closed };
  }
}

const etTime = (ms: number) =>
  new Date(ms).toLocaleTimeString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });

function tooltip(session: MarketSession | null): string {
  if (!session) return "Live market status is unavailable right now.";
  const hours = session.earlyClose ? "9:30 AM to 1:00 PM ET (early close)" : "9:30 AM to 4:00 PM ET";
  const lines = [`NYSE regular session ${hours}`, session.nextChangeLabel];
  if (session.source === "yahoo" && session.lastTradeMs) {
    lines.push(`S&P 500 last traded ${etTime(session.lastTradeMs)} ET`, "Source: Yahoo Finance");
  } else {
    lines.push("Source: NYSE calendar (live feed unavailable, estimate)");
  }
  return lines.join("\n");
}

export function MarketStatusPill() {
  const [session, setSession] = useState<MarketSession | null>(null);
  const [loading, setLoading] = useState(true);
  const lastOk = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;

    async function load() {
      controller?.abort();
      controller = new AbortController();
      try {
        const res = await fetch("/api/market-status", { signal: controller.signal, cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as MarketSession;
        if (cancelled) return;
        lastOk.current = Date.now();
        setSession(data);
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === "AbortError")) return;
        if (Date.now() - lastOk.current > STALE_MS) setSession(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    function schedule() {
      timer = setTimeout(async () => {
        if (document.visibilityState === "visible") await load();
        if (!cancelled) schedule();
      }, POLL_MS);
    }

    const onVisible = () => {
      if (document.visibilityState === "visible") void load();
    };

    void load();
    schedule();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const view = loading ? { text: "Checking market…", tone: TONES.unknown } : describe(session);

  return (
    <span
      role="status"
      aria-live="polite"
      title={loading ? "Checking live market status" : tooltip(session)}
      className={`hidden sm:flex items-center gap-1.5 h-8 px-2.5 text-xs font-semibold border whitespace-nowrap ${view.tone}`}
    >
      <CircleDot className="h-3 w-3" aria-hidden />
      {view.text}
    </span>
  );
}
