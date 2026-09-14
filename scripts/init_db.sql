-- Financial LLM Database Schema
-- Initialized automatically via docker-compose
--
-- Kept in sync with db/migrations/0002_runtime_core.sql + 0006_lineage.sql
-- (the canonical schema definition, applied to Supabase). This file is a
-- separate, independently-maintained copy for local docker-compose --
-- previously drifted from the migrations (missing columns the application
-- code actually reads/writes, e.g. request_text_hash), causing a real
-- UndefinedColumn failure on every store_extraction() call against a
-- docker-compose-provisioned Postgres. Since docker-entrypoint-initdb.d
-- scripts only run once against an empty data volume (never as an
-- incremental migration against existing data), this is a straight
-- rewrite to match reality rather than a series of ALTER statements.

create extension if not exists pgcrypto;

CREATE TABLE IF NOT EXISTS extractions (
    id BIGSERIAL PRIMARY KEY,
    extraction_id UUID NOT NULL DEFAULT gen_random_uuid(),
    filing_id VARCHAR(64) UNIQUE NOT NULL,
    company_name VARCHAR(256),
    ticker VARCHAR(16),
    filing_type VARCHAR(16),
    filing_date DATE,
    fiscal_year_end DATE,
    revenue NUMERIC(20, 2),
    net_income NUMERIC(20, 2),
    total_assets NUMERIC(20, 2),
    total_liabilities NUMERIC(20, 2),
    eps NUMERIC(12, 4),
    sector VARCHAR(128),
    confidence_score REAL,
    extraction_time_ms INTEGER,
    model_version VARCHAR(128),
    method VARCHAR(16) NOT NULL DEFAULT 'llm',
    raw_output TEXT,
    parsed_output JSONB,
    request_text_hash VARCHAR(64),
    source_text_excerpt TEXT,
    prompt_version VARCHAR(64),
    parser_version VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT extractions_method_check CHECK (method IN ('llm', 'xbrl', 'heuristic'))
);

CREATE TABLE IF NOT EXISTS extraction_logs (
    id BIGSERIAL PRIMARY KEY,
    filing_id VARCHAR(64),
    extraction_id UUID,
    status VARCHAR(32) NOT NULL,  -- 'success', 'validation_error', 'timeout', 'error'
    error_message TEXT,
    latency_ms INTEGER,
    model_version VARCHAR(128),
    parser_recovery_stage VARCHAR(32),
    dataset_version VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(128) NOT NULL,
    metric_name VARCHAR(64) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    sample_size INTEGER,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Indexes for common queries (sort direction matches how queries actually
-- filter -- see src/storage/database.py's get_recent_* methods).
CREATE INDEX IF NOT EXISTS idx_extractions_filing_id ON extractions(filing_id);
CREATE INDEX IF NOT EXISTS idx_extractions_company ON extractions(company_name);
CREATE INDEX IF NOT EXISTS idx_extractions_ticker ON extractions(ticker);
CREATE INDEX IF NOT EXISTS idx_extractions_date ON extractions(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_logs_status ON extraction_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_created ON extraction_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_version ON model_metrics(model_version, metric_name, measured_at DESC);

-- Webhook dead-letter queue (failed downstream callbacks)
CREATE TABLE IF NOT EXISTS webhook_failures (
    id BIGSERIAL PRIMARY KEY,
    service VARCHAR(64) NOT NULL,
    target_url TEXT NOT NULL,
    payload JSONB,
    error_message TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_failures_retry ON webhook_failures(next_retry_at) WHERE resolved_at IS NULL;

-- A/B test assignments and outcomes
CREATE TABLE IF NOT EXISTS ab_test_results (
    id BIGSERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    is_challenger BOOLEAN NOT NULL DEFAULT FALSE,
    confidence_score REAL,
    status VARCHAR(32),
    latency_ms INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ab_filing ON ab_test_results(filing_id);
CREATE INDEX IF NOT EXISTS idx_ab_model ON ab_test_results(model_version, created_at DESC);

-- Pipeline stage tracking (optional enrichment queue). extraction_id is a
-- real uuid (not the filing_id) -- src/storage/database.py's
-- upsert_pipeline_stage() generates a fresh one per call and stores the
-- caller's actual filing identifier in the separate filing_id column;
-- passing a filing-id string into a uuid PK used to raise a type error on
-- every real call (fixed this session).
CREATE TABLE IF NOT EXISTS pipeline_stages (
    extraction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id VARCHAR(64),
    stage VARCHAR(32) NOT NULL DEFAULT 'extracted',
    ticker VARCHAR(16),
    stage_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_stage ON pipeline_stages(stage);
