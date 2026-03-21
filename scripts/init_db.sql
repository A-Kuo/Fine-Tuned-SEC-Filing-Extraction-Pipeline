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
