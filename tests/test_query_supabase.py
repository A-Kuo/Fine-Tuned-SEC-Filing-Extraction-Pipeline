"""Tests for frontend/api/query_supabase.py's safety-critical logic.

The handler class itself (BaseHTTPRequestHandler) needs a real socket to
exercise end-to-end -- these test the two pure pieces that matter for
safety: the table allowlist actually matches this repo's real schema (a
stale allowlist either blocks a real table or, worse, could be edited to
include something that isn't actually a table), and the connection-string
lookup tries the documented env var names in order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.api.query_supabase import ALLOWED_TABLES, MAX_LIMIT, _find_connection_string


class TestAllowedTables:
    def test_matches_real_tables_and_views_in_migrations(self):
        """Every table/view db/migrations/*.sql actually defines. A
        mismatch here means either a real table is unreachable through the
        browser, or (worse) the allowlist claims something queryable that
        isn't a real table."""
        expected = {
            "public.extractions", "public.extraction_logs", "public.model_metrics",
            "public.webhook_failures", "public.ab_test_results", "public.pipeline_stages",
            "public.v_recent_extractions", "public.v_model_metric_latest",
            "intel.filings", "intel.filing_sections", "intel.financial_metrics",
            "intel.risk_factors", "intel.mdna_summaries", "intel.extraction_runs",
            "intel.v_filing_metric_summary",
        }
        assert ALLOWED_TABLES == expected

    def test_no_arbitrary_string_can_bypass_the_allowlist(self):
        """Sanity check the allowlist is a real Python set membership check,
        not something that could accidentally do substring/prefix matching."""
        assert "public.extractions; DROP TABLE extractions;--" not in ALLOWED_TABLES
        assert "intel.filings " not in ALLOWED_TABLES  # trailing space

    def test_max_limit_is_capped(self):
        assert MAX_LIMIT == 200


class TestFindConnectionString:
    def test_returns_none_when_no_env_var_set(self, monkeypatch):
        for name in ["POSTGRES_URL", "POSTGRES_URL_NON_POOLING", "SUPABASE_POSTGRES_URL", "DATABASE_URL"]:
            monkeypatch.delenv(name, raising=False)
        value, which = _find_connection_string()
        assert value is None
        assert which is None

    def test_prefers_postgres_url_first(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_URL", "postgresql://real-one")
        monkeypatch.setenv("DATABASE_URL", "postgresql://fallback")
        value, which = _find_connection_string()
        assert value == "postgresql://real-one"
        assert which == "POSTGRES_URL"

    def test_falls_back_when_preferred_name_absent(self, monkeypatch):
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        monkeypatch.delenv("POSTGRES_URL_NON_POOLING", raising=False)
        monkeypatch.delenv("SUPABASE_POSTGRES_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://fallback")
        value, which = _find_connection_string()
        assert value == "postgresql://fallback"
        assert which == "DATABASE_URL"
