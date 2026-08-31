-- Default: deny browser-side public access until product requirements are clear

alter table public.extractions enable row level security;
alter table public.extraction_logs enable row level security;
alter table public.model_metrics enable row level security;
alter table public.webhook_failures enable row level security;
alter table public.ab_test_results enable row level security;
alter table public.pipeline_stages enable row level security;

alter table intel.filings enable row level security;
alter table intel.filing_sections enable row level security;
alter table intel.financial_metrics enable row level security;
alter table intel.risk_factors enable row level security;
alter table intel.mdna_summaries enable row level security;
alter table intel.extraction_runs enable row level security;

-- Service-role-only posture for now: no anon/authenticated policies yet.