-- Real SEC EDGAR entity/filing metadata, populated by
-- scripts/backfill_edgar_filings.py. Additive-only: does not touch
-- public.* or intel.*.
--
-- Deliberately no separately-curated "SIC code -> title" seed table: a table
-- like that is a second, independently-maintained source of truth that can
-- drift from what SEC itself reports, and this session had no reliable way
-- to verify one exhaustively. Instead, edgar.entities.sic_description is
-- populated per-entity straight from EDGAR's own submissions API response
-- (config.sicDescription) for that exact company, and the coarser "Entity
-- Hierarchy" grouping (Division A-J) is derived with edgar.sic_division()
-- below from the numeric SIC code using the standard, stable US Census/OSHA
-- SIC-to-division ranges -- a fixed public convention, not a lookup table.
create schema if not exists edgar;

create table if not exists edgar.entities (
    cik varchar(10) primary key,
    name text not null,
    tickers text[] not null default '{}',
    exchange varchar(20),
    sic varchar(4),
    sic_description text,
    filer_category text,
    fiscal_year_end varchar(4),
    updated_at timestamptz not null default now()
);

create table if not exists edgar.filings (
    accession_no varchar(20) primary key,
    cik varchar(10) not null references edgar.entities(cik) on delete cascade,
    form varchar(16) not null,
    filing_date date not null,
    acceptance_ts timestamptz,
    report_date date,
    items text,
    primary_document text,
    is_xbrl boolean,
    -- Derived from the real form string at ingest time (see
    -- scripts/backfill_edgar_filings.py::derive_status): 'amended' for a
    -- form ending "/A", 'late_notice' for one starting "NT ", else
    -- 'accepted'. Never a guess -- always a fact about the form type.
    status varchar(16) not null,
    source_url text not null,
    ingested_at timestamptz not null default now()
);

create index if not exists idx_edgar_filings_date on edgar.filings (filing_date desc);
create index if not exists idx_edgar_filings_cik_date on edgar.filings (cik, filing_date desc);
create index if not exists idx_edgar_filings_form_date on edgar.filings (form, filing_date desc);

-- Gives the dashboard a real "data ingested through" timestamp instead of a
-- hardcoded string.
create table if not exists edgar.ingest_runs (
    id bigserial primary key,
    scope text not null,
    companies int not null default 0,
    filings_upserted int not null default 0,
    started_at timestamptz not null,
    finished_at timestamptz
);

create or replace function edgar.sic_division(sic varchar) returns char(1)
language sql immutable as $$
    select case
        when sic is null or sic !~ '^[0-9]+$' then null
        when sic::int between 100 and 999 then 'A' -- Agriculture, Forestry, Fishing
        when sic::int between 1000 and 1499 then 'B' -- Mining
        when sic::int between 1500 and 1799 then 'C' -- Construction
        when sic::int between 2000 and 3999 then 'D' -- Manufacturing
        when sic::int between 4000 and 4999 then 'E' -- Transportation, Communications, Utilities
        when sic::int between 5000 and 5199 then 'F' -- Wholesale Trade
        when sic::int between 5200 and 5999 then 'G' -- Retail Trade
        when sic::int between 6000 and 6799 then 'H' -- Finance, Insurance, Real Estate
        when sic::int between 7000 and 8999 then 'I' -- Services
        when sic::int between 9100 and 9999 then 'J' -- Public Administration
        else null
    end;
$$;

create or replace view edgar.v_filings_ledger as
select
    f.accession_no,
    f.cik,
    e.name as entity_name,
    e.tickers,
    e.sic,
    e.sic_description,
    edgar.sic_division(e.sic) as sic_division,
    f.form,
    f.filing_date,
    f.acceptance_ts,
    f.report_date,
    f.items,
    f.is_xbrl,
    f.status,
    f.source_url
from edgar.filings f
join edgar.entities e on e.cik = f.cik;

alter table edgar.entities enable row level security;
alter table edgar.filings enable row level security;
alter table edgar.ingest_runs enable row level security;

do $$
begin
    drop policy if exists "service_role_full_access_edgar_entities" on edgar.entities cascade;
    drop policy if exists "service_role_full_access_edgar_filings" on edgar.filings cascade;
    drop policy if exists "service_role_full_access_edgar_ingest_runs" on edgar.ingest_runs cascade;
end $$;

create policy "service_role_full_access_edgar_entities"
on edgar.entities
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_edgar_filings"
on edgar.filings
for all
to service_role
using (true)
with check (true);

create policy "service_role_full_access_edgar_ingest_runs"
on edgar.ingest_runs
for all
to service_role
using (true)
with check (true);
