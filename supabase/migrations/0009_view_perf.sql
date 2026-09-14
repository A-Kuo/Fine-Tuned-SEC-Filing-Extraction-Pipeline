-- intel.v_filing_metric_summary (0004_views.sql) currently cross-joins
-- three independent 1-to-many relationships (financial_metrics,
-- risk_factors, extraction_runs) before count(distinct ...) collapses the
-- resulting fan-out -- correct but wasteful, and gets worse as any one
-- filing accumulates more metrics/risk factors/runs. Zero current Python
-- callers (confirmed via repo-wide grep), so no regression risk from
-- rewriting it. Pre-aggregate each relationship at filing_id grain first,
-- each already backed by an existing idx_intel_*_filing_id index, then
-- join those three small aggregates instead of the raw tables.

create or replace view intel.v_filing_metric_summary as
select
    f.filing_id, f.company_name, f.ticker, f.filing_type, f.filing_date,
    coalesce(m.metric_count, 0)      as metric_count,
    coalesce(r.risk_factor_count, 0) as risk_factor_count,
    er.last_run_at
from intel.filings f
left join (select filing_id, count(*) as metric_count
           from intel.financial_metrics group by filing_id) m
    on m.filing_id = f.filing_id
left join (select filing_id, count(*) as risk_factor_count
           from intel.risk_factors group by filing_id) r
    on r.filing_id = f.filing_id
left join (select filing_id, max(created_at) as last_run_at
           from intel.extraction_runs group by filing_id) er
    on er.filing_id = f.filing_id;
