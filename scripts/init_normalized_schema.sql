-- Normalized SEC Filing Intelligence schema.
--
-- Lives alongside the flat `extractions` table (scripts/init_db.sql) in the
-- same database, under a dedicated `intel` schema namespace so the two
-- don't collide. This is the foundation-layer schema for the dual-track
-- XBRL + LLM extraction pipeline (src/schemas.py, src/pipeline.py) --
-- see docs/BOUNDARY.md for the xbrl-vs-llm precedence rule this enforces.

CREATE SCHEMA IF NOT EXISTS intel;

CREATE TABLE IF NOT EXISTS intel.filings (
    filing_id VARCHAR(64) PRIMARY KEY,
    cik VARCHAR(10),
    accession_no VARCHAR(20) UNIQUE,
    ticker VARCHAR(10),
    company_name TEXT,
    filing_type VARCHAR(16),
    filing_date DATE,
    raw_text_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intel.filing_sections (
    section_id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL REFERENCES intel.filings(filing_id) ON DELETE CASCADE,
    section_type VARCHAR(50) NOT NULL,
    title TEXT,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (filing_id, section_type, char_start)
);

-- One canonical row per (filing_id, metric_name, period, segment). `method`
-- records which track currently owns the value; period/segment default to
-- '' rather than NULL so the UNIQUE constraint actually deduplicates
-- (Postgres treats NULLs as distinct from each other).
CREATE TABLE IF NOT EXISTS intel.financial_metrics (
    metric_id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL REFERENCES intel.filings(filing_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    period VARCHAR(20) NOT NULL DEFAULT '',
    segment VARCHAR(100) NOT NULL DEFAULT '',
    value NUMERIC(38, 4),
    unit VARCHAR(20) DEFAULT 'usd',
    method VARCHAR(10) NOT NULL CHECK (method IN ('xbrl', 'heuristic', 'llm')),
    confidence REAL NOT NULL,
    source_section VARCHAR(50),
    evidence_text TEXT,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (filing_id, metric_name, period, segment)
);

CREATE TABLE IF NOT EXISTS intel.risk_factors (
    risk_id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL REFERENCES intel.filings(filing_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    source_section VARCHAR(50) DEFAULT 'risk_factors',
    confidence REAL NOT NULL DEFAULT 1.0,
    risk_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (filing_id, risk_hash)
);

CREATE TABLE IF NOT EXISTS intel.mdna_summaries (
    summary_id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL UNIQUE REFERENCES intel.filings(filing_id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    method VARCHAR(10) NOT NULL DEFAULT 'heuristic' CHECK (method IN ('heuristic', 'llm')),
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intel.extraction_runs (
    run_id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL REFERENCES intel.filings(filing_id) ON DELETE CASCADE,
    pipeline_version VARCHAR(50),
    status VARCHAR(20) NOT NULL,  -- 'success', 'partial', 'failed'
    sections_found INTEGER,
    metrics_found INTEGER,
    risk_factors_found INTEGER,
    error_message TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intel_sections_filing ON intel.filing_sections(filing_id);
CREATE INDEX IF NOT EXISTS idx_intel_metrics_filing ON intel.financial_metrics(filing_id);
CREATE INDEX IF NOT EXISTS idx_intel_metrics_name ON intel.financial_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_intel_risk_filing ON intel.risk_factors(filing_id);
CREATE INDEX IF NOT EXISTS idx_intel_runs_filing ON intel.extraction_runs(filing_id);
