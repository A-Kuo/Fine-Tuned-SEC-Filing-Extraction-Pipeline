"use client";

import { Landmark, Search, CircleDot, UserRound } from "lucide-react";
import { useState } from "react";

const TABS = ["Portfolio Matrix", "Regulatory Filings", "Market Yields", "Audit Trails"];

export function TopNav() {
  const [active, setActive] = useState(TABS[1]);

  return (
    <header className="h-14 bg-primary-navy text-white flex items-center px-4 gap-6">
      <div className="flex items-center gap-2 shrink-0">
        <Landmark className="h-5 w-5" strokeWidth={1.75} aria-hidden />
        <span className="text-sm font-semibold tracking-tight whitespace-nowrap">
          EDGAR-X Disclosure Matrix
        </span>
      </div>

      <nav className="hidden md:flex items-stretch h-14 flex-1 min-w-0">
        {TABS.map((tab) => {
          const isActive = tab === active;
          return (
            <button
              key={tab}
              onClick={() => setActive(tab)}
              className={`px-4 h-full text-xs font-medium tracking-wide border border-white/15 -ml-px first:ml-0 transition-colors ${
                isActive
                  ? "bg-white/10 border-b-2 border-b-white text-white"
                  : "text-white/70 hover:bg-white/5 hover:text-white"
              }`}
            >
              {tab}
            </button>
          );
        })}
      </nav>

      <div className="flex items-center gap-3 shrink-0 ml-auto">
        <div className="hidden lg:flex items-center gap-2 bg-white/10 border border-white/20 h-8 px-2 w-64">
          <Search className="h-3.5 w-3.5 text-white/60" aria-hidden />
          <input
            type="text"
            placeholder="Search by ticker, CIK, or Form type..."
            className="bg-transparent text-xs text-white placeholder:text-white/50 outline-none w-full"
          />
        </div>

        <span className="hidden sm:flex items-center gap-1.5 h-8 px-2.5 bg-market-gain-bg text-market-gain text-xs font-semibold border border-market-gain/40">
          <CircleDot className="h-3 w-3" aria-hidden />
          Market Open
        </span>

        <div className="flex items-center gap-1.5 h-8 px-2 border border-white/20">
          <UserRound className="h-3.5 w-3.5" aria-hidden />
          <span className="text-xs font-medium hidden sm:inline">A. Kuo</span>
        </div>
      </div>
    </header>
  );
}
