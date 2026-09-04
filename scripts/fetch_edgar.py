"""Fetch real SEC EDGAR filings for training and evaluation.

Respects SEC fair-access policy:
- User-Agent must identify your application and provide contact info
- Rate limit: max 10 requests per second (we use configurable RPS with sleep)

Data sources:
- Company tickers: https://www.sec.gov/files/company_tickers.json
- Submissions: https://data.sec.gov/submissions/CIK{cik}.json

Usage:
    python scripts/fetch_edgar.py --limit 10 --tickers AAPL MSFT
    python scripts/fetch_edgar.py --limit 100
    python scripts/fetch_edgar.py --resume   # continue from checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import get_project_root, load_config


SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"
TICKERS_URL = f"{SEC_WWW}/files/company_tickers.json"

# Transient HTTP status codes that should be retried
_RETRYABLE = {429, 500, 502, 503, 504}


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


def _retry_get(
    client: httpx.Client,
    url: str,
    headers: dict,
    timeout: float,
    limiter: RateLimiter,
    max_attempts: int = 4,
    backoff_base: float = 2.0,
) -> httpx.Response:
    """GET with exponential backoff on transient errors and 429 rate-limit waits."""
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            r = client.get(url, headers=headers, timeout=timeout)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            if attempt == max_attempts:
                raise
            logger.warning(f"Network error (attempt {attempt}/{max_attempts}): {exc}. Retrying in {delay}s…")
            time.sleep(delay)
            delay *= backoff_base
            continue

        if r.status_code not in _RETRYABLE:
            r.raise_for_status()
            return r

        retry_after = int(r.headers.get("Retry-After", delay))
        wait = max(retry_after, delay)
        if attempt == max_attempts:
            r.raise_for_status()
        logger.warning(f"HTTP {r.status_code} for {url} (attempt {attempt}/{max_attempts}). Waiting {wait}s…")
        time.sleep(wait)
        delay *= backoff_base

    raise RuntimeError(f"Exhausted retries for {url}")


def _client_headers(config: dict) -> dict:
    ua = config.get("edgar", {}).get("user_agent", "FinDocAnalyzer contact@example.com")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}


def _data_headers(config: dict) -> dict:
    ua = config.get("edgar", {}).get("user_agent", "FinDocAnalyzer contact@example.com")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}


def load_company_tickers(client: httpx.Client, limiter: RateLimiter, config: dict) -> dict[str, int]:
    """Return mapping ticker -> CIK (integer)."""
    r = _retry_get(client, TICKERS_URL, _client_headers(config), timeout=60, limiter=limiter)
    data = r.json()
    out: dict[str, int] = {}
    for _k, v in data.items():
        t = v.get("ticker", "").upper()
        cik = int(v["cik_str"])
        out[t] = cik
    return out


def fetch_submissions(client: httpx.Client, cik: int, limiter: RateLimiter, config: dict) -> dict:
    """Fetch company submissions JSON."""
    cik_padded = str(cik).zfill(10)
    url = f"{SEC_DATA}/submissions/CIK{cik_padded}.json"
    r = _retry_get(client, url, _data_headers(config), timeout=60, limiter=limiter)
    return r.json()


def pick_filings(submissions: dict, forms: list[str], max_count: int) -> list[dict]:
    """Select recent filings matching form types."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    form_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    primary_doc = recent.get("primaryDocument", [])
    filing_date = recent.get("filingDate", [])

    picked = []
    for i, form in enumerate(form_list):
        if form not in forms:
            continue
        if i >= len(accession_list):
            break
        picked.append({
            "form": form,
            "accessionNumber": accession_list[i],
            "primaryDocument": primary_doc[i] if i < len(primary_doc) else "",
            "filingDate": filing_date[i] if i < len(filing_date) else "",
        })
        if len(picked) >= max_count:
            break
    return picked


def filing_url(cik: int, accession: str, primary_doc: str) -> str:
    accession_nodash = accession.replace("-", "")
    return f"{SEC_WWW}/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"


def fetch_filing_text(client: httpx.Client, url: str, limiter: RateLimiter, config: dict) -> str:
    r = _retry_get(client, url, _client_headers(config), timeout=120, limiter=limiter)
    return r.text


def html_to_visible_text(html: str) -> str:
    """Strip an EDGAR filing document down to the visible text a human reader
    (or the section-heading regexes in src/extraction/section_parser.py) would
    see -- not the raw HTML/inline-XBRL markup.

    This matters more than it looks: EDGAR renders 10-K prose with inline
    XBRL tagging applied per-word or per-phrase (`<ix:nonFraction>`, nested
    `<span>`s, etc.), so without this step the on-disk .txt this repo trains
    and evaluates against is markup soup, not prose -- the extraction
    pipeline was written and tuned against clean text and was never actually
    exercised against what EDGAR really serves until this function existed.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # The entire machine-readable inline-XBRL block (<ix:header>, tag/context
    # definitions) lives inside <div style="display:none">...</div> right
    # after <body>. It renders as nothing in a browser, but get_text() below
    # doesn't know CSS -- without this it's the first ~15K chars of "text".
    for tag in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
        tag.decompose()
    for tag in soup.find_all("ix:header"):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse the run of blank lines BeautifulSoup's per-tag newlines produce
    # without collapsing genuine paragraph breaks down to nothing.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Chunk long filings for model context limits (character-based)."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + max_chars])
        start = start + max_chars - overlap
    return chunks


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(raw_dir: Path) -> Path:
    return raw_dir / ".fetch_checkpoint.json"


def load_checkpoint(raw_dir: Path) -> dict:
    """Load fetch progress; returns empty dict if no checkpoint."""
    cp = _checkpoint_path(raw_dir)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    return {"fetched": [], "total": 0}


def save_checkpoint(raw_dir: Path, state: dict) -> None:
    _checkpoint_path(raw_dir).write_text(json.dumps(state, indent=2))


def clear_checkpoint(raw_dir: Path) -> None:
    cp = _checkpoint_path(raw_dir)
    if cp.exists():
        cp.unlink()


def rebuild_manifest(raw_dir: Path) -> list[dict]:
    """Rebuild manifest.jsonl by scanning every *.meta.json actually on disk.

    Previously the manifest was written with mode="w" whenever --resume was
    not passed, which truncated it on every fresh run -- even though the
    actual .txt/.meta.json/.xbrl.json files from earlier runs were untouched
    (the checkpoint that gated re-fetching had already been cleared on a
    completed run, so nothing was lost except the manifest's record of it).
    Downstream code (schema_conformance.py, evaluate_real_filings.py,
    sync_normalized_from_pipeline.py) all treat manifest.jsonl as the source
    of truth for "which real filings exist" -- silently losing entries there
    broke that even though the underlying evidence was never actually gone.
    Rebuilding from disk on every run makes the manifest self-healing
    regardless of --resume.
    """
    entries = []
    for meta_path in sorted(raw_dir.glob("*.meta.json")):
        try:
            entries.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping unreadable meta file {meta_path}: {e}")
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR filings")
    parser.add_argument("--limit", type=int, default=None, help="Max filings total")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols to fetch")
    parser.add_argument("--per-ticker", type=int, default=1, help="Max filings per ticker")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    config = load_config()
    edgar_cfg = config.get("edgar", {})
    raw_dir = get_project_root() / edgar_cfg.get("raw_dir", "data/raw_edgar")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_checkpoint:
        clear_checkpoint(raw_dir)
        logger.info("Checkpoint cleared.")

    rps = float(edgar_cfg.get("requests_per_second", 10))
    limiter = RateLimiter(rps)
    forms = edgar_cfg.get("forms", ["10-K"])
    max_run = args.limit or edgar_cfg.get("max_filings_per_run", 100)
    max_chars = (config["model"].get("max_seq_length", 2048) or 2048) * 4

    # Load checkpoint for resume
    checkpoint = load_checkpoint(raw_dir) if args.resume else {"fetched": [], "total": 0}
    already_fetched: set[str] = set(checkpoint.get("fetched", []))
    total = checkpoint.get("total", 0)

    with httpx.Client(follow_redirects=True) as client:
        ticker_map = load_company_tickers(client, limiter, config)
        tickers = args.tickers or edgar_cfg.get("tickers") or []
        if not tickers:
            tickers = sorted(ticker_map.keys())[: min(20, max_run)]

        for ticker in tickers:
            if total >= max_run:
                break
            cik = ticker_map.get(ticker.upper())
            if not cik:
                logger.warning(f"Unknown ticker: {ticker}")
                continue

            try:
                subs = fetch_submissions(client, cik, limiter, config)
            except Exception as e:
                logger.error(f"Submissions failed for {ticker}: {e}")
                continue

            company_name = subs.get("name", "")
            filings = pick_filings(subs, forms, args.per_ticker)

            for f in filings:
                if total >= max_run:
                    break
                acc = f["accessionNumber"]

                # Skip if already fetched in a previous run
                if acc in already_fetched:
                    logger.debug(f"Skipping (checkpoint): {ticker} {acc}")
                    continue

                primary = f["primaryDocument"] or f"{acc.replace('-', '')}.txt"
                url = filing_url(cik, acc, primary)

                try:
                    raw_html = fetch_filing_text(client, url, limiter, config)
                except Exception as e:
                    logger.error(f"Failed download {url}: {e}")
                    continue

                # Keep the raw HTML for XBRL fact scraping below (it needs the
                # inline ix:/xml: tags), but persist only the visible text --
                # see html_to_visible_text()'s docstring for why this matters.
                text = html_to_visible_text(raw_html)

                safe_acc = acc.replace("-", "_")
                base = raw_dir / f"{ticker}_{safe_acc}"
                txt_path = base.with_suffix(".txt")
                txt_path.write_text(text, encoding="utf-8", errors="replace")

                record = {
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "company": company_name,
                    "form": f["form"],
                    "accessionNumber": acc,
                    "filingDate": f["filingDate"],
                    "source_url": url,
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "text_path": str(txt_path.relative_to(get_project_root())),
                }

                # Optional XBRL parse (best-effort)
                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from parse_xbrl import extract_xbrl_facts

                    facts = extract_xbrl_facts(raw_html)
                    if facts:
                        fact_path = base.with_suffix(".xbrl.json")
                        fact_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
                        record["xbrl_path"] = str(fact_path.relative_to(get_project_root()))
                except Exception as e:
                    logger.debug(f"XBRL parse skipped: {e}")

                chunks = chunk_text(text, max_chars=max_chars, overlap=200)
                record["chunks"] = len(chunks)

                # Written only now that xbrl_path/chunks are set -- a meta
                # file that omits them (this used to write before both were
                # known) can't be used by rebuild_manifest() below to
                # reconstruct a complete manifest entry from disk alone.
                meta_path = base.with_suffix(".meta.json")
                meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

                total += 1
                already_fetched.add(acc)

                # Persist checkpoint after every successful filing
                save_checkpoint(raw_dir, {"fetched": list(already_fetched), "total": total})
                logger.info(f"Saved {ticker} {acc} ({len(text)} chars, {len(chunks)} chunks) [{total}/{max_run}]")

        manifest_path = raw_dir / "manifest.jsonl"
        all_entries = rebuild_manifest(raw_dir)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for m in all_entries:
                mf.write(json.dumps(m) + "\n")

        if total >= max_run:
            clear_checkpoint(raw_dir)
            logger.info(f"Run complete ({total} filings). Checkpoint cleared.")
        else:
            logger.info(f"Done. {total} filings saved. Checkpoint retained for --resume.")

        logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
