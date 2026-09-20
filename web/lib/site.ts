export const SITE_NAME = "SEC Edgar Filing Platform";

export const REPO_URL = "https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline";

export const repoFile = (path: string) => `${REPO_URL}/blob/main/${path}`;

// Where the live market status comes from. The pill's popover links to these
// so a reader can check the source themselves.
export const MARKET_SOURCES = {
  yahoo: {
    label: "Yahoo Finance: S&P 500 (^GSPC)",
    url: "https://finance.yahoo.com/quote/%5EGSPC",
    note: "Live last-trade feed the status is confirmed against",
  },
  nyse: {
    label: "NYSE holidays and trading hours",
    url: "https://www.nyse.com/trade/hours-calendars",
    note: "Calendar used for holidays, early closes and session times",
  },
} as const;
