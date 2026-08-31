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

create policy "service_role_full_access_extractions"
on public.extractions
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_logs"
on public.extraction_logs
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_metrics"
on public.model_metrics
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_webhook_failures"
on public.webhook_failures
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_ab_results"
on public.ab_test_results
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_pipeline_stages"
on public.pipeline_stages
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_filings"
on intel.filings
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_sections"
on intel.filing_sections
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_metrics"
on intel.financial_metrics
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_risk_factors"
on intel.risk_factors
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_mdna_summaries"
on intel.mdna_summaries
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_intel_extraction_runs"
on intel.extraction_runs
for all
to service_role
using (true)
with check (true);