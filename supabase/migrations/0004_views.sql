create or replace view public.v_recent_extractions as
select
    e.filing_id,
    e.company_name,
    e.ticker,
    e.filing_type,
    e.filing_date,
    e.revenue,
    e.net_income,
    e.confidence_score,
    e.model_version,
    e.created_at
from public.extractions e
order by e.created_at desc;

create or replace view public.v_model_metric_latest as
select distinct on (model_version, metric_name)
    model_version,
    metric_name,
    metric_value,
    sample_size,
    measured_at,
    metadata
from public.model_metrics
order by model_version, metric_name, measured_at desc;

create or replace view intel.v_filing_metric_summary as
select
    f.filing_id,
    f.company_name,
    f.ticker,
    f.filing_type,
    f.filing_date,
    count(distinct m.metric_id) as metric_count,
    count(distinct r.risk_id) as risk_factor_count,
    max(er.created_at) as last_run_at
from intel.filings f
left join intel.financial_metrics m on f.filing_id = m.filing_id
left join intel.risk_factors r on f.filing_id = r.filing_id
left join intel.extraction_runs er on f.filing_id = er.filing_id
group by 1,2,3,4,5;