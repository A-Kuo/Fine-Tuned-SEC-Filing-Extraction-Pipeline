-- Financial LLM Database Schema
-- Initialized automatically via docker-compose

CREATE TABLE IF NOT EXISTS extractions (
    id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) UNIQUE NOT NULL,
    company_name VARCHAR(256),
    filing_type VARCHAR(16),
    filing_date DATE,
    revenue NUMERIC(20, 2),
    net_income NUMERIC(20, 2),
    total_assets NUMERIC(20, 2),
    total_liabilities NUMERIC(20, 2),
    eps NUMERIC(10, 4),
    confidence_score REAL,
    extraction_time_ms INTEGER,
    model_version VARCHAR(64),
    raw_output TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS extraction_logs (
    id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64),
    status VARCHAR(16) NOT NULL,  -- 'success', 'validation_error', 'timeout', 'error'
    error_message TEXT,
    latency_ms INTEGER,
    model_version VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL,
    metric_name VARCHAR(64) NOT NULL,
    metric_value REAL NOT NULL,
    sample_size INTEGER,
    measured_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_extractions_filing_id ON extractions(filing_id);
CREATE INDEX IF NOT EXISTS idx_extractions_company ON extractions(company_name);
CREATE INDEX IF NOT EXISTS idx_extractions_date ON extractions(filing_date);
CREATE INDEX IF NOT EXISTS idx_logs_status ON extraction_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_created ON extraction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_version ON model_metrics(model_version, metric_name);

-- Webhook dead-letter queue (failed downstream callbacks)
CREATE TABLE IF NOT EXISTS webhook_failures (
    id SERIAL PRIMARY KEY,
    service VARCHAR(64) NOT NULL,
    target_url TEXT NOT NULL,
    payload JSONB,
    error_message TEXT,
    attempt_count INTEGER DEFAULT 0,
    next_retry_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_webhook_failures_retry ON webhook_failures(next_retry_at) WHERE resolved_at IS NULL;

-- A/B test assignments and outcomes
CREATE TABLE IF NOT EXISTS ab_test_results (
    id SERIAL PRIMARY KEY,
    filing_id VARCHAR(64) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    is_challenger BOOLEAN DEFAULT FALSE,
    confidence_score REAL,
    status VARCHAR(32),
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ab_filing ON ab_test_results(filing_id);
CREATE INDEX IF NOT EXISTS idx_ab_model ON ab_test_results(model_version, created_at);

-- Pipeline stage tracking (optional enrichment queue)
CREATE TABLE IF NOT EXISTS pipeline_stages (
    extraction_id VARCHAR(64) PRIMARY KEY,
    stage VARCHAR(32) NOT NULL DEFAULT 'extracted',
    ticker VARCHAR(16),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_stage ON pipeline_stages(stage);
