"""Backfill edgar.entities / edgar.filings with real SEC EDGAR filing
metadata (see db/migrations/0010_edgar_filings.sql for the schema this
writes into and why there is no separately-curated SIC lookup table).

For each company: fetch_edgar.py's fetch_submissions() is reused as-is
(rate-limited, retrying, real SEC User-Agent). Every entity field
(name, tickers, exchange, sic, sic_description, filer_category,
fiscal_year_end) and every filing field (form, dates, items, primary
document, is_xbrl) is read straight out of that real response -- nothing
here is guessed or interpolated. `status` is the one derived field, and it
is derived from the real form string, not invented (see derive_status()).

Universe: the first --limit-companies tickers from SEC's own
company_tickers.json (scripts/fetch_edgar.py::load_company_tickers(),
already used elsewhere in this repo) -- no dependency on an unverified
market-cap or exchange-listing endpoint for this proof run.

Usage:
    python scripts/backfill_edgar_filings.py --limit-companies 50 --dry-run
    python scripts/backfill_edgar_filings.py --limit-companies 50
    python scripts/backfill_edgar_filings.py --tickers AAPL MSFT KO --resume

Requires POSTGRES_HOST/PORT/USER/PASSWORD/DB environment variables (same
convention as scripts/backfill_live_schema.py and db/sync/transfer_metrics.py)
unless --dry-run is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import get_project_root, load_config

sys.path.insert(0, str(Path(__file__).parent))
from fetch_edgar import (  # noqa: E402
    RateLimiter,
    clear_checkpoint,
    fetch_submissions,
    filing_url,
    load_checkpoint,
    load_company_tickers,
    save_checkpoint,
)

DEFAULT_FORMS = ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "NT 10-K", "NT 10-Q"]

ENTITY_UPSERT_SQL = """
    INSERT INTO edgar.entities
        (cik, name, tickers, exchange, sic, sic_description, filer_category, fiscal_year_end, updated_at)
    VALUES %s
    ON CONFLICT (cik) DO UPDATE SET
        name = EXCLUDED.name,
        tickers = EXCLUDED.tickers,
        exchange = EXCLUDED.exchange,
        sic = EXCLUDED.sic,
        sic_description = EXCLUDED.sic_description,
        filer_category = EXCLUDED.filer_category,
        fiscal_year_end = EXCLUDED.fiscal_year_end,
        updated_at = EXCLUDED.updated_at
"""

FILING_UPSERT_SQL = """
    INSERT INTO edgar.filings
        (accession_no, cik, form, filing_date, acceptance_ts, report_date,
         items, primary_document, is_xbrl, status, source_url)
    VALUES %s
    ON CONFLICT (accession_no) DO UPDATE SET
        form = EXCLUDED.form,
        filing_date = EXCLUDED.filing_date,
        acceptance_ts = EXCLUDED.acceptance_ts,
        report_date = EXCLUDED.report_date,
        items = EXCLUDED.items,
        primary_document = EXCLUDED.primary_document,
        is_xbrl = EXCLUDED.is_xbrl,
        status = EXCLUDED.status,
        source_url = EXCLUDED.source_url
"""

INGEST_RUN_INSERT_SQL = """
    INSERT INTO edgar.ingest_runs (scope, companies, filings_upserted, started_at, finished_at)
    VALUES (%s, %s, %s, %s, %s)
"""


# ─── Pure field mapping (no network, no DB -- fully unit-testable) ──────────

def derive_status(form: str) -> str:
    """Status is a fact read off the real form string, never a guess: a form
    ending "/A" is an amendment; one starting "NT " is a late-filing
    notification; anything else is a normally accepted filing."""
    if form.endswith("/A"):
        return "amended"
    if form.startswith("NT "):
        return "late_notice"
    return "accepted"


def build_entity_row(cik: int, submissions: dict) -> dict:
    """Every field here is read verbatim from EDGAR's own submissions
    response for this exact company -- see the module docstring."""
    exchanges = submissions.get("exchanges") or []
    return {
        "cik": str(cik).zfill(10),
        "name": submissions.get("name"),
        "tickers": submissions.get("tickers") or [],
        "exchange": exchanges[0] if exchanges else None,
        "sic": submissions.get("sic") or None,
        "sic_description": submissions.get("sicDescription") or None,
        "filer_category": submissions.get("category") or None,
        "fiscal_year_end": submissions.get("fiscalYearEnd") or None,
    }


def pick_entity_filings(submissions: dict, forms: list[str]) -> list[dict]:
    """Every filing matching `forms` (no count cap -- this script backfills
    all of it, unlike fetch_edgar.py::pick_filings()'s per-ticker sample),
    keeping the extra real columns edgar.filings needs (acceptanceDateTime,
    reportDate, items, isXBRL) that pick_filings() doesn't carry."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    def column(name: str, i: int, default=""):
        values = recent.get(name, [])
        return values[i] if i < len(values) else default

    picked = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        picked.append({
            "form": form,
            "accessionNumber": column("accessionNumber", i),
            "filingDate": column("filingDate", i),
            "acceptanceDateTime": column("acceptanceDateTime", i, None) or None,
            "reportDate": column("reportDate", i, None) or None,
            "items": column("items", i, None) or None,
            "primaryDocument": column("primaryDocument", i, ""),
            "isXBRL": bool(column("isXBRL", i, 0)),
        })
    return picked


def build_filing_row(cik: int, entry: dict) -> dict:
    accession = entry["accessionNumber"]
    return {
        "accession_no": accession,
        "cik": str(cik).zfill(10),
        "form": entry["form"],
        "filing_date": entry["filingDate"] or None,
        "acceptance_ts": entry.get("acceptanceDateTime"),
        "report_date": entry.get("reportDate"),
        "items": entry.get("items"),
        "primary_document": entry.get("primaryDocument") or None,
        "is_xbrl": entry.get("isXBRL"),
        "status": derive_status(entry["form"]),
        "source_url": filing_url(cik, accession, entry.get("primaryDocument") or ""),
    }


# ─── DB writes (execute_values, matching db/sync/transfer_metrics.py) ───────

def upsert_entity(cur, row: dict) -> None:
    import psycopg2.extras

    psycopg2.extras.execute_values(
        cur, ENTITY_UPSERT_SQL,
        [(
            row["cik"], row["name"], row["tickers"], row["exchange"], row["sic"],
            row["sic_description"], row["filer_category"], row["fiscal_year_end"],
            datetime.now(timezone.utc),
        )],
    )


def upsert_filings(cur, rows: list[dict]) -> None:
    if not rows:
        return
    import psycopg2.extras

    psycopg2.extras.execute_values(
        cur, FILING_UPSERT_SQL,
        [(
            r["accession_no"], r["cik"], r["form"], r["filing_date"], r["acceptance_ts"],
            r["report_date"], r["items"], r["primary_document"], r["is_xbrl"],
            r["status"], r["source_url"],
        ) for r in rows],
    )


def _connect():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
    )


# ─── Orchestration ───────────────────────────────────────────────────────────

def run(
    tickers: list[str],
    ticker_to_cik: dict[str, int],
    forms: list[str],
    limiter: RateLimiter,
    config: dict,
    conn=None,
    dry_run: bool = False,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Backfill every ticker in `tickers`. Returns a summary dict (also the
    shape written to edgar.ingest_runs). Injectable `conn`/`client` make this
    fully testable with fakes -- no live network or DB required to exercise
    the orchestration logic itself."""
    checkpoint = load_checkpoint(checkpoint_dir) if (resume and checkpoint_dir) else {"fetched": [], "total": 0}
    done_ciks: set[str] = set(checkpoint.get("fetched", []))

    started_at = datetime.now(timezone.utc)
    companies = 0
    filings_upserted = 0
    skipped: list[str] = []

    owns_client = client is None
    client = client or httpx.Client()
    try:
        for ticker in tickers:
            cik = ticker_to_cik.get(ticker)
            if cik is None:
                logger.warning(f"No CIK found for ticker {ticker!r}; skipping")
                skipped.append(ticker)
                continue
            cik_padded = str(cik).zfill(10)
            if cik_padded in done_ciks:
                logger.info(f"Skipping {ticker} (CIK {cik_padded}) -- already in checkpoint")
                continue

            try:
                submissions = fetch_submissions(client, cik, limiter, config)
            except Exception as e:
                logger.error(f"Failed to fetch submissions for {ticker} (CIK {cik}): {e}")
                skipped.append(ticker)
                continue

            entity_row = build_entity_row(cik, submissions)
            filing_entries = pick_entity_filings(submissions, forms)
            filing_rows = [build_filing_row(cik, e) for e in filing_entries]

            if dry_run:
                logger.info(f"[dry-run] {ticker} CIK {cik_padded}: {entity_row['name']!r}, {len(filing_rows)} filings")
            else:
                with conn.cursor() as cur:
                    upsert_entity(cur, entity_row)
                    upsert_filings(cur, filing_rows)
                conn.commit()

            companies += 1
            filings_upserted += len(filing_rows)
            done_ciks.add(cik_padded)
            if checkpoint_dir:
                save_checkpoint(checkpoint_dir, {"fetched": sorted(done_ciks), "total": companies})
    finally:
        if owns_client:
            client.close()

    finished_at = datetime.now(timezone.utc)
    summary = {
        "scope": f"proof:{len(tickers)}-tickers",
        "companies": companies,
        "filings_upserted": filings_upserted,
        "skipped": skipped,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }

    if not dry_run and conn is not None:
        with conn.cursor() as cur:
            cur.execute(INGEST_RUN_INSERT_SQL, (summary["scope"], companies, filings_upserted, started_at, finished_at))
        conn.commit()

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit-companies", type=int, default=50, help="How many tickers to backfill (ignored if --tickers given)")
    p.add_argument("--tickers", nargs="*", default=None, help="Explicit ticker list, overrides --limit-companies")
    p.add_argument("--forms", nargs="*", default=DEFAULT_FORMS)
    p.add_argument("--rps", type=float, default=5.0, help="Requests per second (SEC's own ceiling is 10)")
    p.add_argument("--dry-run", action="store_true", help="Fetch and print, write nothing to Postgres")
    p.add_argument("--resume", action="store_true", help="Skip tickers already recorded in the checkpoint")
    p.add_argument("--clear-checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    checkpoint_dir = get_project_root() / "data" / "edgar_backfill"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_checkpoint:
        clear_checkpoint(checkpoint_dir)

    limiter = RateLimiter(args.rps)

    with httpx.Client() as client:
        ticker_to_cik = load_company_tickers(client, limiter, config)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = list(ticker_to_cik.keys())[: args.limit_companies]

    conn = None if args.dry_run else _connect()
    try:
        summary = run(
            tickers, ticker_to_cik, args.forms, limiter, config,
            conn=conn, dry_run=args.dry_run, checkpoint_dir=checkpoint_dir, resume=args.resume,
        )
    finally:
        if conn is not None:
            conn.close()

    print(
        f"Companies: {summary['companies']}  Filings upserted: {summary['filings_upserted']}  "
        f"Skipped: {len(summary['skipped'])}"
    )
    if summary["skipped"]:
        print(f"Skipped tickers: {', '.join(summary['skipped'])}")
    if summary["companies"] == 0 and tickers:
        sys.exit(1)


if __name__ == "__main__":
    main()
