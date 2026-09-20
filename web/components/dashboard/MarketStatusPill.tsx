"use client";

import { CircleDot, ExternalLink } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useDismiss } from "@/components/ui/Dropdown";
import type { MarketSession } from "@/lib/market-hours";
import { MARKET_SOURCES } from "@/lib/site";

const POLL_MS = 60_000;
// After this long without a successful poll, stop showing the last known state:
// a stale "Market Open" is worse than admitting the feed is down.
const STALE_MS = 3 * POLL_MS;

const OPEN_DELAY_MS = 120;
const CLOSE_DELAY_MS = 200;

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

function SourceLink({ href, label, note }: { href: string; label: string; note: string }) {
  return (
    <li>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="group block py-1.5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-secondary-blue"
      >
        <span className="inline-flex items-center gap-1.5 font-semibold text-secondary-blue group-hover:underline">
          {label}
          <ExternalLink className="h-3 w-3" aria-hidden />
          <span className="sr-only">(opens in a new tab)</span>
        </span>
        <span className="block text-[11px] text-text-muted">{note}</span>
      </a>
    </li>
  );
}

function Details({ session, loading, statusText }: { session: MarketSession | null; loading: boolean; statusText: string }) {
  const computed = session?.source === "computed";
  const hours = session?.earlyClose ? "9:30 AM to 1:00 PM ET (early close)" : "9:30 AM to 4:00 PM ET";
  const { yahoo, nyse } = MARKET_SOURCES;

  return (
    <div className="px-3.5 py-3 text-xs text-text-secondary">
      <p className="text-[11px] uppercase tracking-wide text-text-muted">NYSE regular session</p>
      <p className="mt-0.5 text-sm font-semibold text-primary-navy">{statusText}</p>

      {loading ? (
        <p className="mt-1">Checking the live market status&hellip;</p>
      ) : !session ? (
        <p className="mt-1">Live market status is unavailable right now. The sources below are where it comes from.</p>
      ) : (
        <dl className="mt-1.5 flex flex-col gap-1">
          <div className="flex justify-between gap-4">
            <dt className="text-text-muted">Hours</dt>
            <dd className="tabular-figures text-right">{hours}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-text-muted">Next</dt>
            <dd className="text-right">{session.nextChangeLabel}</dd>
          </div>
          {!computed && session.lastTradeMs && (
            <div className="flex justify-between gap-4">
              <dt className="text-text-muted">S&amp;P 500 last traded</dt>
              <dd className="tabular-figures text-right">{etTime(session.lastTradeMs)} ET</dd>
            </div>
          )}
        </dl>
      )}

      <div className="mt-3 pt-2 border-t border-border-formal">
        <p className="text-[11px] uppercase tracking-wide text-text-muted">
          {computed ? "Source of this estimate" : "Sources"}
        </p>
        {computed && (
          <p className="mt-1 text-[11px]">The live feed is unavailable, so this is calculated from the NYSE calendar.</p>
        )}
        <ul className="mt-0.5 divide-y divide-border-formal">
          {computed ? (
            <>
              <SourceLink href={nyse.url} label={nyse.label} note={nyse.note} />
              <SourceLink href={yahoo.url} label={yahoo.label} note="Live feed (currently unavailable)" />
            </>
          ) : (
            <>
              <SourceLink href={yahoo.url} label={yahoo.label} note={yahoo.note} />
              <SourceLink href={nyse.url} label={nyse.label} note={nyse.note} />
            </>
          )}
        </ul>
      </div>
    </div>
  );
}

export function MarketStatusPill() {
  const [session, setSession] = useState<MarketSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);
  const lastOk = useRef(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const openTimer = useRef<ReturnType<typeof setTimeout>>();
  const closeTimer = useRef<ReturnType<typeof setTimeout>>();
  const popoverId = useId();

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    // A newer load() aborts the previous one; only the latest request may settle
    // state, otherwise the aborted one's `finally` would clear `loading` while
    // its replacement is still in flight and flash "Status unavailable".
    let latest = 0;

    async function load() {
      controller?.abort();
      controller = new AbortController();
      const mine = ++latest;
      try {
        const res = await fetch("/api/market-status", { signal: controller.signal, cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as MarketSession;
        if (cancelled || mine !== latest) return;
        lastOk.current = Date.now();
        setSession(data);
      } catch (error) {
        if (cancelled || mine !== latest || (error instanceof DOMException && error.name === "AbortError")) return;
        if (Date.now() - lastOk.current > STALE_MS) setSession(null);
      } finally {
        if (!cancelled && mine === latest) setLoading(false);
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

  useEffect(
    () => () => {
      clearTimeout(openTimer.current);
      clearTimeout(closeTimer.current);
    },
    [],
  );

  const scheduleOpen = () => {
    clearTimeout(closeTimer.current);
    openTimer.current = setTimeout(() => setOpen(true), OPEN_DELAY_MS);
  };

  const scheduleClose = () => {
    clearTimeout(openTimer.current);
    closeTimer.current = setTimeout(() => setOpen(false), CLOSE_DELAY_MS);
  };

  const openNow = () => {
    clearTimeout(openTimer.current);
    clearTimeout(closeTimer.current);
    setOpen(true);
  };

  const close = useCallback(() => {
    clearTimeout(openTimer.current);
    clearTimeout(closeTimer.current);
    setOpen(false);
  }, []);

  useDismiss(open, close, rootRef);

  const view = loading ? { text: "Checking market…", tone: TONES.unknown } : describe(session);

  return (
    <div
      ref={rootRef}
      className="relative hidden sm:block"
      onMouseEnter={scheduleOpen}
      onMouseLeave={scheduleClose}
      onFocus={openNow}
      onBlur={(event) => {
        if (!rootRef.current?.contains(event.relatedTarget as Node | null)) close();
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") close();
      }}
    >
      <button
        type="button"
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls={open ? popoverId : undefined}
        onClick={openNow}
        className={`flex items-center gap-1.5 h-8 px-2.5 text-xs font-semibold border whitespace-nowrap cursor-default focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${view.tone}`}
      >
        <CircleDot className="h-3 w-3" aria-hidden />
        <span role="status" aria-live="polite" className="max-w-[11rem] truncate">
          {view.text}
        </span>
      </button>

      {open && (
        // pt-1.5 is a transparent bridge so the cursor can cross from the pill into the popover.
        <div id={popoverId} role="region" aria-label="Market status sources" className="absolute right-0 top-full z-40 w-72 pt-1.5">
          <div className="bg-surface border border-primary-navy shadow-flat text-text-primary">
            <Details session={session} loading={loading} statusText={view.text} />
          </div>
        </div>
      )}
    </div>
  );
}
