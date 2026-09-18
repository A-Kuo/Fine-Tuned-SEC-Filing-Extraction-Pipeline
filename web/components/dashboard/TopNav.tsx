"use client";

import { Landmark, Search, UserRound } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { withFilter } from "@/lib/filters";
import { BASE_PATH, TABS, tabHref } from "@/lib/tabs";
import { MarketStatusPill } from "./MarketStatusPill";

const SEARCH_DEBOUNCE_MS = 300;

export function TopNav() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const query = searchParams.toString();

  const onFilings = pathname.startsWith(`${BASE_PATH}/regulatory-filings`);
  const activeSlug = TABS.find((t) => pathname.startsWith(`${BASE_PATH}/${t.slug}`))?.slug;

  const [text, setText] = useState(searchParams.get("q") ?? "");
  const committed = useRef(text);
  const debounce = useRef<ReturnType<typeof setTimeout>>();

  // Keep the box in step with the URL (back/forward, tab switches) without
  // clobbering text the user is still typing.
  useEffect(() => {
    const urlQ = searchParams.get("q") ?? "";
    if (urlQ !== committed.current) {
      committed.current = urlQ;
      setText(urlQ);
    }
  }, [searchParams]);

  useEffect(() => () => clearTimeout(debounce.current), []);

  const commit = (value: string) => {
    clearTimeout(debounce.current);
    const trimmed = value.trim();
    committed.current = trimmed;
    const next = withFilter(query, "q", trimmed);
    if (onFilings) router.replace(next ? `${pathname}?${next}` : pathname, { scroll: false });
    else router.push(tabHref("regulatory-filings", next));
  };

  const onChange = (value: string) => {
    setText(value);
    clearTimeout(debounce.current);
    // On the filings tab the ledger narrows as you type; elsewhere, searching
    // navigates, so wait for Enter rather than yanking the page mid-word.
    if (onFilings) debounce.current = setTimeout(() => commit(value), SEARCH_DEBOUNCE_MS);
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    commit(text);
  };

  const tabClass = (isActive: boolean) =>
    `flex items-center px-4 h-full text-xs font-medium tracking-wide border border-white/15 -ml-px first:ml-0 transition-colors whitespace-nowrap focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-white ${
      isActive ? "bg-white/10 border-b-2 border-b-white text-white" : "text-white/70 hover:bg-white/5 hover:text-white"
    }`;

  return (
    <div className="print:hidden">
      <header className="h-14 bg-primary-navy text-white flex items-center px-4 gap-6">
        <Link href={tabHref("portfolio-matrix")} className="flex items-center gap-2 shrink-0">
          <Landmark className="h-5 w-5" strokeWidth={1.75} aria-hidden />
          <span className="text-sm font-semibold tracking-tight whitespace-nowrap">EDGAR-X Disclosure Matrix</span>
        </Link>

        <nav aria-label="Dashboard sections" className="hidden md:flex items-stretch h-14 flex-1 min-w-0">
          {TABS.map((tab) => (
            <Link
              key={tab.slug}
              href={tabHref(tab.slug, query)}
              aria-current={tab.slug === activeSlug ? "page" : undefined}
              className={tabClass(tab.slug === activeSlug)}
            >
              {tab.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-3 shrink-0 ml-auto">
          <form role="search" onSubmit={onSubmit} className="hidden lg:flex items-center gap-2 bg-white/10 border border-white/20 h-8 px-2 w-64 focus-within:border-white/60">
            <Search className="h-3.5 w-3.5 text-white/60" aria-hidden />
            <input
              type="search"
              aria-label="Search filings by ticker, CIK, or form type"
              value={text}
              onChange={(e) => onChange(e.target.value)}
              placeholder="Search by ticker, CIK, or Form type..."
              className="bg-transparent text-xs text-white placeholder:text-white/50 outline-none w-full [&::-webkit-search-cancel-button]:appearance-none"
            />
          </form>

          <MarketStatusPill />

          <div className="flex items-center gap-1.5 h-8 px-2 border border-white/20">
            <UserRound className="h-3.5 w-3.5" aria-hidden />
            <span className="text-xs font-medium hidden sm:inline">A. Kuo</span>
          </div>
        </div>
      </header>

      <nav aria-label="Dashboard sections" className="md:hidden h-11 bg-primary-navy text-white flex items-stretch overflow-x-auto border-t border-white/15">
        {TABS.map((tab) => (
          <Link
            key={tab.slug}
            href={tabHref(tab.slug, query)}
            aria-current={tab.slug === activeSlug ? "page" : undefined}
            className={tabClass(tab.slug === activeSlug)}
          >
            {tab.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
