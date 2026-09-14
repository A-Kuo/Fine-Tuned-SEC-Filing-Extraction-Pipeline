"""Read-only Supabase browser: browser -> this function -> Postgres.

Unlike invoke.py/status.py (which proxy to SageMaker over AWS), this talks
directly to Supabase's Postgres using the connection string Vercel's
native Supabase Marketplace integration injects as an environment
variable when a Supabase project is connected via the project's Storage
tab (Settings -> Environment Variables afterward shows exactly which
names landed -- POSTGRES_URL is the standard one for that integration;
this also tries a couple of fallback names in case the exact integration
version differs).

Security: this endpoint is unauthenticated, like the rest of this debug
console (see frontend/README.md -- it's a human-interface debugging tool,
not a production surface). To keep an unauthenticated SQL-adjacent
endpoint from being a real risk even in that context, this does NOT accept
arbitrary SQL. It only allows SELECT * FROM <table> LIMIT <n> against a
fixed allowlist of this repo's own real tables/views (below), with the
table name matched verbatim against the allowlist -- never string-
interpolated from unvalidated input -- and the limit capped at 200.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Every real table/view this repo's migrations define (db/migrations/*.sql).
# Matched verbatim -- adding a table here is the only way to make it
# queryable, there is no way to bypass this from the request itself.
ALLOWED_TABLES = {
    "public.extractions", "public.extraction_logs", "public.model_metrics",
    "public.webhook_failures", "public.ab_test_results", "public.pipeline_stages",
    "public.v_recent_extractions", "public.v_model_metric_latest",
    "intel.filings", "intel.filing_sections", "intel.financial_metrics",
    "intel.risk_factors", "intel.mdna_summaries", "intel.extraction_runs",
    "intel.v_filing_metric_summary",
}
MAX_LIMIT = 200

# Names Vercel's Supabase Marketplace integration may have injected,
# tried in order. POSTGRES_URL is the standard one for that integration.
CONNECTION_ENV_VARS = ["POSTGRES_URL", "POSTGRES_URL_NON_POOLING", "SUPABASE_POSTGRES_URL", "DATABASE_URL"]


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps({"error": message}).encode())


def _find_connection_string() -> tuple[str | None, str | None]:
    """Returns (value, which_env_var_name) or (None, None)."""
    for name in CONNECTION_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)

        if not query:
            # No params at all -- treat as "list what's queryable" rather
            # than an error, useful for the frontend to populate its table
            # picker without hardcoding the allowlist client-side too.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"tables": sorted(ALLOWED_TABLES)}).encode())
            return

        table = (query.get("table") or [None])[0]
        if table not in ALLOWED_TABLES:
            return _error(self, 400, f"'table' must be one of: {sorted(ALLOWED_TABLES)}")

        try:
            limit = min(int((query.get("limit") or ["50"])[0]), MAX_LIMIT)
        except ValueError:
            return _error(self, 400, "'limit' must be an integer")

        conn_string, env_var_used = _find_connection_string()
        if not conn_string:
            return _error(
                self, 503,
                f"No Supabase connection string found in any of {CONNECTION_ENV_VARS}. "
                "Check Vercel project Settings -> Environment Variables for the exact "
                "name the Supabase integration actually set, and add it to CONNECTION_ENV_VARS.",
            )

        import psycopg2

        try:
            conn = psycopg2.connect(conn_string)
        except Exception as e:
            return _error(self, 502, f"Could not connect to Postgres via {env_var_used}: {e}")

        try:
            with conn.cursor() as cur:
                # table is verified against ALLOWED_TABLES (a fixed set
                # this code defines, never derived from the request) right
                # above -- safe to format directly; limit is int-parsed and
                # capped, also safe.
                cur.execute(f"SELECT * FROM {table} LIMIT {limit}")
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        finally:
            conn.close()

        body = {
            "table": table,
            "columns": columns,
            "rows": [[str(v) if v is not None else None for v in row] for row in rows],
            "row_count": len(rows),
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
