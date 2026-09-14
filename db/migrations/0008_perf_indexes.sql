-- Missing/mismatched indexes found during a full SQL-optimization audit of
-- every real query in src/storage/database.py. All additive (create index
-- if not exists) -- safe to run against a database with existing data.

-- Cache-miss fallback path (get_extraction_by_text_hash) hits on every
-- /extract request with no prior cache hit; no index existed for it.
create index if not exists idx_extractions_request_text_hash
    on public.extractions (request_text_hash)
    where request_text_hash is not null;

-- Dashboard join (get_recent_extractions_dashboard) joins extraction_logs
-- to extractions on filing_id; extraction_logs.filing_id had no index.
create index if not exists idx_extraction_logs_filing_id
    on public.extraction_logs (filing_id);

-- extraction_logs filtered by dataset_version (0006_lineage.sql), never indexed.
create index if not exists idx_extraction_logs_dataset_version
    on public.extraction_logs (dataset_version)
    where dataset_version is not null;

-- v_recent_extractions (0004_views.sql) orders by created_at desc with no
-- LIMIT and no supporting index (only filing_date desc was indexed).
create index if not exists idx_extractions_created_at
    on public.extractions (created_at desc);

-- Composite-index leading-column mismatches: get_recent_metrics filters
-- WHERE metric_name = %s AND measured_at > ..., but idx_model_metrics_
-- version_name leads with model_version -- a column that query never
-- filters on. Same problem for get_ab_summary (filters only created_at)
-- against idx_ab_test_results_model_version. Add a second, differently-
-- ordered index alongside each rather than reordering (migrations here
-- are additive-only, and the original index still serves other queries).
create index if not exists idx_model_metrics_name_measured
    on public.model_metrics (metric_name, measured_at desc);

create index if not exists idx_ab_test_results_created_at
    on public.ab_test_results (created_at desc);
