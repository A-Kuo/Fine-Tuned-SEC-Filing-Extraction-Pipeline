"""PostgreSQL storage for the normalized SEC Filing Intelligence schema.

Mirrors src/database.py's PostgresStorage pattern (raw psycopg2, no ORM)
but writes to the `intel.*` tables defined in
scripts/init_normalized_schema.sql, which coexist alongside the flat
`extractions` table used by architecture A. See docs/BOUNDARY.md for the
xbrl-vs-llm precedence rule enforced here before any financial_metrics
upsert.
"""

from __future__ import annotations

from loguru import logger

from src.extraction.normalizer import filing_record_to_rows
from src.core.schemas import FilingRecord, MetricRecord


class NormalizedStorage:
    """PostgreSQL persistence for FilingRecord data (intel schema)."""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._connection = None
        self._available = False

    def connect(self) -> bool:
        """Establish PostgreSQL connection.

        Uses discrete keyword arguments rather than a postgresql:// URI --
        a URI built by string interpolation breaks for any password
        containing URI-reserved characters (%, ?, /, @, etc.), which a
        raw f-string DSN doesn't escape. Mirrors db/sync/transfer_metrics.py's
        connect(), which never hit this because it always used keyword args.
        """
        try:
            import psycopg2
            self._connection = psycopg2.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                dbname=self._database,
            )
            self._connection.autocommit = True
            self._available = True
            logger.info(f"NormalizedStorage connected: {self._host}:{self._port}/{self._database}")
            return True
        except Exception as e:
            logger.warning(f"NormalizedStorage unavailable: {e}. Records will not be persisted.")
            self._available = False
            return False

    def upsert_filing(self, row: dict) -> bool:
        if not self._available:
            return False
        try:
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.filings (
                    filing_id, cik, accession_no, ticker, company_name,
                    filing_type, filing_date, raw_text_hash, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (filing_id) DO UPDATE SET
                    cik = EXCLUDED.cik,
                    accession_no = EXCLUDED.accession_no,
                    ticker = EXCLUDED.ticker,
                    company_name = EXCLUDED.company_name,
                    filing_type = EXCLUDED.filing_type,
                    filing_date = EXCLUDED.filing_date,
                    raw_text_hash = EXCLUDED.raw_text_hash,
                    updated_at = NOW()
                """,
                (
                    row["filing_id"], row["cik"], row["accession_no"], row["ticker"],
                    row["company_name"], row["filing_type"], row["filing_date"],
                    row["raw_text_hash"],
                ),
            )
            return True
        except Exception as e:
            logger.error(f"intel.filings upsert error: {e}")
            return False

    def insert_section(self, row: dict) -> bool:
        if not self._available:
            return False
        try:
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.filing_sections (
                    filing_id, section_type, title, char_start, char_end, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (filing_id, section_type, char_start) DO NOTHING
                """,
                (
                    row["filing_id"], row["section_type"], row["title"],
                    row["char_start"], row["char_end"], row["confidence"],
                ),
            )
            return True
        except Exception as e:
            logger.error(f"intel.filing_sections insert error: {e}")
            return False

    def upsert_metric(self, filing_id: str, incoming: MetricRecord) -> bool:
        """Upsert a metric, respecting XBRL precedence atomically.

        The precedence rule (an xbrl fact is never overwritten by a
        heuristic/llm fact for the same natural key -- see docs/BOUNDARY.md)
        is enforced directly in the ON CONFLICT ... DO UPDATE's WHERE clause,
        not via a prior SELECT. The previous version did
        SELECT-existing -> resolve_metric_precedence() -> UPSERT-winner,
        which raced: a write from another connection landing between this
        connection's SELECT and its own INSERT/UPDATE could be silently
        clobbered or could silently clobber this one, and two concurrent
        callers for the same natural key could both read "no existing row"
        and both proceed as if creating it fresh. A single statement has no
        such window -- Postgres evaluates the WHERE clause against the
        row as it exists at the moment of the write, atomically.

        Mirrors the exact same clause used in db/sync/transfer_metrics.py's
        standalone bulk-ingestion path, so both writers enforce the rule
        identically rather than maintaining two implementations that could
        drift apart.
        """
        if not self._available:
            return False
        try:
            period = incoming.period or ""
            segment = incoming.segment or ""
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.financial_metrics (
                    filing_id, metric_name, period, segment, value, unit,
                    method, confidence, source_section, evidence_text,
                    model_version, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (filing_id, metric_name, period, segment) DO UPDATE SET
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    method = EXCLUDED.method,
                    confidence = EXCLUDED.confidence,
                    source_section = EXCLUDED.source_section,
                    evidence_text = EXCLUDED.evidence_text,
                    model_version = EXCLUDED.model_version,
                    updated_at = NOW()
                WHERE NOT (
                    intel.financial_metrics.method = 'xbrl'
                    AND EXCLUDED.method <> 'xbrl'
                )
                """,
                (
                    filing_id, incoming.name, period, segment, incoming.value, incoming.unit,
                    incoming.method, incoming.confidence, incoming.source_section,
                    incoming.evidence_text, incoming.model_version,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"intel.financial_metrics upsert error: {e}")
            return False

    def insert_risk_factor(self, row: dict) -> bool:
        if not self._available:
            return False
        try:
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.risk_factors (
                    filing_id, text, source_section, confidence, risk_hash
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (filing_id, risk_hash) DO NOTHING
                """,
                (row["filing_id"], row["text"], row["source_section"],
                 row["confidence"], row["risk_hash"]),
            )
            return True
        except Exception as e:
            logger.error(f"intel.risk_factors insert error: {e}")
            return False

    def upsert_mdna_summary(self, row: dict) -> bool:
        if not self._available:
            return False
        try:
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.mdna_summaries (filing_id, summary, method, model_version)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (filing_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    method = EXCLUDED.method,
                    model_version = EXCLUDED.model_version
                """,
                (row["filing_id"], row["summary"], row["method"], row["model_version"]),
            )
            return True
        except Exception as e:
            logger.error(f"intel.mdna_summaries upsert error: {e}")
            return False

    def log_extraction_run(
        self,
        filing_id: str,
        *,
        pipeline_version: str,
        status: str,
        sections_found: int,
        metrics_found: int,
        risk_factors_found: int,
        duration_ms: int,
        error_message: str | None = None,
    ) -> bool:
        if not self._available:
            return False
        try:
            cur = self._connection.cursor()
            cur.execute(
                """
                INSERT INTO intel.extraction_runs (
                    filing_id, pipeline_version, status, sections_found,
                    metrics_found, risk_factors_found, error_message, duration_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (filing_id, pipeline_version, status, sections_found, metrics_found,
                 risk_factors_found, error_message, duration_ms),
            )
            return True
        except Exception as e:
            logger.error(f"intel.extraction_runs insert error: {e}")
            return False

    def save_filing_record(self, record: FilingRecord) -> bool:
        """Write every table's rows for a FilingRecord in one call."""
        rows = filing_record_to_rows(record)
        filing_id = record.metadata.filing_id

        ok = self.upsert_filing(rows["filings"][0])
        for section_row in rows["filing_sections"]:
            ok = self.insert_section(section_row) and ok
        for metric in record.metrics:
            ok = self.upsert_metric(filing_id, metric) and ok
        for risk_row in rows["risk_factors"]:
            ok = self.insert_risk_factor(risk_row) and ok
        for mdna_row in rows["mdna_summaries"]:
            ok = self.upsert_mdna_summary(mdna_row) and ok
        return ok

    def close(self):
        if self._connection:
            self._connection.close()
            self._available = False
