"""Print a human-checkable summary of what scripts/backfill_edgar_filings.py
actually wrote to edgar.entities / edgar.filings, so a real backfill run can
be spot-checked against EDGAR itself before anything downstream (the web
dashboard) is pointed at this data.

Prints: row counts by form, by status, and by SIC division; the real
ingested date range; and N random (entity, accession_no, source_url) rows
for you to open directly on EDGAR and confirm they're real.

Usage:
    python scripts/verify_edgar_backfill.py
    python scripts/verify_edgar_backfill.py --sample 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _connect():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
    )


def gather_report(cur, sample: int) -> dict:
    """All read-only queries against edgar.*. Pure enough to unit-test with
    a fake cursor that returns canned fetchall()/fetchone() results."""
    report: dict = {}

    cur.execute("SELECT count(*) FROM edgar.entities")
    report["entity_count"] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM edgar.filings")
    report["filing_count"] = cur.fetchone()[0]

    cur.execute("SELECT form, count(*) FROM edgar.filings GROUP BY form ORDER BY count(*) DESC")
    report["by_form"] = cur.fetchall()

    cur.execute("SELECT status, count(*) FROM edgar.filings GROUP BY status ORDER BY count(*) DESC")
    report["by_status"] = cur.fetchall()

    cur.execute(
        "SELECT edgar.sic_division(sic), count(*) FROM edgar.entities GROUP BY edgar.sic_division(sic) ORDER BY 1"
    )
    report["by_sic_division"] = cur.fetchall()

    cur.execute("SELECT min(filing_date), max(filing_date) FROM edgar.filings")
    report["date_range"] = cur.fetchone()

    cur.execute(
        """
        SELECT accession_no, cik, entity_name, form, filing_date, source_url
        FROM edgar.v_filings_ledger
        ORDER BY random()
        LIMIT %s
        """,
        (sample,),
    )
    report["sample_rows"] = cur.fetchall()

    cur.execute(
        """
        SELECT accession_no, count(*) FROM edgar.filings
        GROUP BY accession_no HAVING count(*) > 1
        """
    )
    report["duplicate_accessions"] = cur.fetchall()

    return report


def print_report(report: dict) -> None:
    print(f"Entities: {report['entity_count']}   Filings: {report['filing_count']}")
    print(f"Date range ingested: {report['date_range'][0]} .. {report['date_range'][1]}")

    print("\nBy form:")
    for form, count in report["by_form"]:
        print(f"  {form:<12} {count}")

    print("\nBy status:")
    for status, count in report["by_status"]:
        print(f"  {status:<14} {count}")

    print("\nBy SIC division:")
    for division, count in report["by_sic_division"]:
        print(f"  {division or 'unknown':<10} {count}")

    if report["duplicate_accessions"]:
        print(f"\n!! {len(report['duplicate_accessions'])} duplicate accession numbers found (should be 0):")
        for accession, count in report["duplicate_accessions"]:
            print(f"  {accession}  x{count}")
    else:
        print("\nNo duplicate accession numbers (accession_no is the primary key, as expected).")

    print(f"\n{len(report['sample_rows'])} random rows -- open source_url and confirm each is a real EDGAR filing:")
    for accession, cik, entity_name, form, filing_date, source_url in report["sample_rows"]:
        print(f"  {entity_name} (CIK {cik}) | {form} | {filing_date} | {accession}")
        print(f"    {source_url}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=10, help="Random rows to print for spot-checking")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            report = gather_report(cur, args.sample)
    finally:
        conn.close()
    print_report(report)


if __name__ == "__main__":
    main()
