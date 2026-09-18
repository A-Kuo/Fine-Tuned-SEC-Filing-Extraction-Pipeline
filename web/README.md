# FinDoc Dashboard (web/)

Institutional-style financial disclosure and analytics dashboard, built with
Next.js 14 (App Router, TypeScript), Tailwind CSS, `lucide-react`, and
`recharts`. Replaces the earlier static `frontend/` debug console.

## Pages

`/` redirects to the first tab. All four live under `/financial-dashboard/`:

| Tab | Route | Filters that apply |
|---|---|---|
| Portfolio Matrix | `portfolio-matrix` | Asset Class |
| Regulatory Filings | `regulatory-filings` | Entity Hierarchy, Filing Window, search |
| Market Yields | `market-yields` | Asset Class, Filing Window |
| Audit Trails | `audit-trails` | Filing Window |

Filter state lives in the URL (`?asset=&entity=&window=&q=`), so views are
shareable and survive a reload, and the nav tabs carry the current filters
across pages. A control that doesn't apply to the current tab is shown
disabled rather than silently doing nothing.

## Actions

- **Export**: "Export CSV" downloads the current tab's filtered data (RFC 4180,
  UTF-8 BOM, formula-injection guarded). "Export XBRL" (Regulatory Filings
  only) downloads an XBRL instance of **dei cover-page facts** (document type,
  registrant, CIK, ticker). It is not a financial-statement or filed document.
- **Print Ledger**: prints the current view as a landscape report with the
  navigation and controls hidden and the applied filters in a header.
- **Audit Logs**: jumps to the Audit Trails tab. Exports, prints and filter
  changes made in this UI are recorded there as "This browser" events in
  `localStorage` (capped at 200). That is a per-browser activity trail, not a
  compliance-grade audit log.

## Live market status

The header pill polls `GET /api/market-status` every 60s while the tab is
visible. The route handler calls Yahoo Finance's keyless chart endpoint for
`^GSPC`, whose payload has **no `marketState` field** (that only exists on the
`v7/quote` endpoint, which now needs a crumb and returns 401). The state is
derived instead: the published pre/regular/post windows say when a session is
scheduled, an NYSE calendar (`lib/market-hours.ts`, holidays and early closes)
vetoes closed days, and a fresh last-trade timestamp confirms the regular
session is really trading.

The Yahoo endpoint is unofficial. If it is unreachable, rate-limits the host,
or changes shape, the route falls back to the calendar alone and the pill
shows "(est.)". Responses are edge-cached for 60s, so upstream sees roughly one
call a minute however many people have the page open. No API key is needed.

## Data

Everything on the pages is **illustrative**: fictional issuers, invented CIKs
(`0099000xxx`), yields and audit events (`lib/mock-data.ts`). Windows such as
"Q3 FY2026" anchor to a fixed data as-of date (`DATA_AS_OF`), not the wall
clock. `lib/data.ts` is the only module pages read data through; when
`serving/api.py` gains list endpoints, swap that layer and the (already
server-rendered) pages only need an `await`.

## Local use

```bash
npm install
npm run dev        # http://localhost:3000
npm test           # vitest: filters, export, audit store, market hours
npm run typecheck
npm run build
```

## Deploying to Vercel

Set this Vercel project's **Root Directory to `web`** and its **Framework
Preset to Next.js** (Project Settings -> General; Root Directory alone does
not change a previously chosen preset). No environment variables are required.

## Connecting back to the FastAPI backend

`serving/api.py`'s `GET /` 307-redirects to `config.yaml`'s
`serving.frontend_url` when set. Point that at this app's deployed URL.

The SageMaker Async Inference proxy and the read-only Supabase table browser
that lived in the old `frontend/api/*.py` Vercel functions have not been
ported to this app yet (see `sagemaker/README.md`'s Status section).
