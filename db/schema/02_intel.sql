-- Normalized intelligence schema for richer filing-level downstream analysis

create schema if not exists intel;

create table if not exists intel.filings (
    filing_id varchar(64) primary key,
    cik varchar(10),
    accession_no varchar(20) unique,
    ticker varchar(16),
    company_name text,
    filing_type varchar(16),
    filing_date date,
    raw_text_hash varchar(64),
    source_url text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists intel.filing_sections (
    section_id bigserial primary key,
    filing_id varchar(64) not null references intel.filings(filing_id) on delete cascade,
    section_type varchar(50) not null,
    title text,
    char_start integer not null,
    char_end integer not null,
    confidence real not null default 1.0,
    content_hash varchar(64),
    created_at timestamptz not null default now(),
    unique (filing_id, section_type, char_start)
);

create index if not exists idx_intel_filing_sections_filing_id
    on intel.filing_sections (filing_id);

create table if not exists intel.financial_metrics (
    metric_id bigserial primary key,
    filing_id varchar(64) not null references intel.filings(filing_id) on delete cascade,
    metric_name varchar(100) not null,
    period varchar(20) not null default '',
    segment varchar(100) not null default '',
    value numeric(38,4),
    unit varchar(20) not null default 'usd',
    method varchar(16) not null,
    confidence real not null,
    source_section varchar(50),
    evidence_text text,
    model_version varchar(128),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint intel_financial_metrics_method_check check (method in ('xbrl', 'heuristic', 'llm')),
    unique (filing_id, metric_name, period, segment)
);

create index if not exists idx_intel_financial_metrics_filing_id
    on intel.financial_metrics (filing_id);

create index if not exists idx_intel_financial_metrics_name
    on intel.financial_metrics (metric_name);

create table if not exists intel.risk_factors (
    risk_id bigserial primary key,
    filing_id varchar(64) not null references intel.filings(filing_id) on delete cascade,
    text text not null,
    source_section varchar(50) not null default 'risk_factors',
    confidence real not null default 1.0,
    risk_hash varchar(64) not null,
    created_at timestamptz not null default now(),
    unique (filing_id, risk_hash)
);

create index if not exists idx_intel_risk_factors_filing_id
    on intel.risk_factors (filing_id);

create table if not exists intel.mdna_summaries (
    summary_id bigserial primary key,
    filing_id varchar(64) not null unique references intel.filings(filing_id) on delete cascade,
    summary text not null,
    method varchar(16) not null default 'heuristic',
    model_version varchar(128),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint intel_mdna_summaries_method_check check (method in ('heuristic', 'llm'))
);

create table if not exists intel.extraction_runs (
    run_id bigserial primary key,
    filing_id varchar(64) not null references intel.filings(filing_id) on delete cascade,
    pipeline_version varchar(64),
    status varchar(20) not null,
    sections_found integer,
    metrics_found integer,
    risk_factors_found integer,
    error_message text,
    duration_ms integer,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint intel_extraction_runs_status_check check (status in ('success', 'partial', 'failed'))
);

create index if not exists idx_intel_extraction_runs_filing_id
    on intel.extraction_runs (filing_id);