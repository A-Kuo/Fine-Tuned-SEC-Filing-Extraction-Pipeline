-- Core runtime schema used by serving/, monitoring/, and batch workflows

create table if not exists public.extractions (
    id bigserial primary key,
    extraction_id uuid not null default gen_random_uuid(),
    filing_id varchar(64) not null unique,
    company_name varchar(256),
    ticker varchar(16),
    filing_type varchar(16),
    filing_date date,
    fiscal_year_end date,
    revenue numeric(20,2),
    net_income numeric(20,2),
    total_assets numeric(20,2),
    total_liabilities numeric(20,2),
    eps numeric(12,4),
    sector varchar(128),
    confidence_score real,
    extraction_time_ms integer,
    model_version varchar(128),
    method varchar(16) not null default 'llm',
    raw_output text,
    parsed_output jsonb,
    request_text_hash varchar(64),
    source_text_excerpt text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint extractions_method_check check (method in ('llm', 'xbrl', 'heuristic'))
);

create index if not exists idx_extractions_filing_id
    on public.extractions (filing_id);

create index if not exists idx_extractions_ticker
    on public.extractions (ticker);

create index if not exists idx_extractions_filing_date
    on public.extractions (filing_date desc);

create index if not exists idx_extractions_model_version
    on public.extractions (model_version);

create index if not exists idx_extractions_parsed_output_gin
    on public.extractions using gin (parsed_output);

create table if not exists public.extraction_logs (
    id bigserial primary key,
    filing_id varchar(64),
    extraction_id uuid,
    status varchar(32) not null,
    error_message text,
    latency_ms integer,
    model_version varchar(128),
    created_at timestamptz not null default now()
);

create index if not exists idx_extraction_logs_status
    on public.extraction_logs (status);

create index if not exists idx_extraction_logs_created_at
    on public.extraction_logs (created_at desc);

create table if not exists public.model_metrics (
    id bigserial primary key,
    model_version varchar(128) not null,
    metric_name varchar(64) not null,
    metric_value double precision not null,
    sample_size integer,
    measured_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_model_metrics_version_name
    on public.model_metrics (model_version, metric_name, measured_at desc);

create table if not exists public.webhook_failures (
    id bigserial primary key,
    service varchar(64) not null,
    target_url text not null,
    payload jsonb,
    error_message text,
    attempt_count integer not null default 0,
    next_retry_at timestamptz,
    resolved_at timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists idx_webhook_failures_retry
    on public.webhook_failures (next_retry_at)
    where resolved_at is null;

create table if not exists public.ab_test_results (
    id bigserial primary key,
    filing_id varchar(64) not null,
    model_version varchar(128) not null,
    is_challenger boolean not null default false,
    confidence_score real,
    status varchar(32),
    latency_ms integer,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_ab_test_results_filing_id
    on public.ab_test_results (filing_id);

create index if not exists idx_ab_test_results_model_version
    on public.ab_test_results (model_version, created_at desc);

create table if not exists public.pipeline_stages (
    extraction_id uuid primary key,
    filing_id varchar(64),
    stage varchar(32) not null default 'extracted',
    ticker varchar(16),
    stage_payload jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists idx_pipeline_stages_stage
    on public.pipeline_stages (stage);