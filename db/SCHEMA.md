# Database Schema Reference

Two schemas coexist, representing two architectures for the same domain — see [`docs/BOUNDARY.md`](../docs/BOUNDARY.md) for the extraction-scope boundary and the [README's Limitations section](../README.md#limitations) for the current status of each.

- **`public.*`** — flat, one row per filing (architecture A). This is what `src/storage/database.py`'s `DatabaseManager`/`PostgresStorage` actually writes to, and what `serving/api.py` and `monitoring/dashboard.py` read. **Live in production.**
- **`intel.*`** — normalized, one row per metric/period/segment (architecture B). Defined in `db/migrations/0003_intel_schema.sql`. Written only by `src/storage/normalized_storage.py::NormalizedStorage`, whose only caller is `db/sync/sync_normalized_from_pipeline.py` — an offline batch job, not the live request path. See "Why not wire it into serving" below.

Migrations are numbered and applied in order from `db/migrations/`. **All migrations so far are additive** (`create table if not exists`, `alter table ... add column if not exists`) — preserve that convention; never write a migration that drops or renames a column another migration or piece of running code depends on.

## Tables and natural keys

| Table | Natural key (UNIQUE) | Why |
|---|---|---|
| `public.extractions` | `filing_id` | One flat extraction record per filing. |
| `intel.filings` | `filing_id` (PK), `accession_no` | A filing is identified by either its internal id or its real SEC accession number. |
| `intel.filing_sections` | `(filing_id, section_type, char_start)` | The same section type can appear more than once in a filing (rare, but the offset disambiguates); re-running the parser on unchanged text should not create duplicate rows. |
| `intel.financial_metrics` | `(filing_id, metric_name, period, segment)` | **Deliberately excludes `method`.** An `xbrl` fact and an `llm` fact for the same metric/period/segment are the *same fact*, competing for one row — that's what makes precedence resolution (below) meaningful instead of accumulating three rows (xbrl/heuristic/llm) that a reader would then have to reconcile themselves. |
| `intel.risk_factors` | `(filing_id, risk_hash)` | Content-hash dedup — the same risk-factor paragraph re-extracted (e.g. after a heuristic tweak) doesn't create a duplicate row. |
| `intel.mdna_summaries` | `filing_id` | One summary per filing. |
| `intel.extraction_runs` | none (append-only log) | Every run is a new row by design — this is a log, not a current-state table. |

## XBRL precedence: the never-overwrite rule

`intel.financial_metrics.method` is `xbrl`, `heuristic`, or `llm`. The rule (documented in `docs/BOUNDARY.md`, upstream boundary rationale): **an `xbrl` fact is never overwritten by a `heuristic` or `llm` fact for the same natural key.** Among non-`xbrl` methods, and between two `xbrl` writes, last write wins.

**Enforcement is atomic, in SQL, at the point of write** — `src/storage/normalized_storage.py::upsert_metric()` and `db/sync/transfer_metrics.py`'s bulk path both use the identical clause:

```sql
INSERT INTO intel.financial_metrics (...) VALUES (...)
ON CONFLICT (filing_id, metric_name, period, segment) DO UPDATE SET ...
WHERE NOT (
    intel.financial_metrics.method = 'xbrl'
    AND EXCLUDED.method <> 'xbrl'
)
```

This replaced an earlier version that did `SELECT existing row -> resolve in Python -> UPSERT the winner`, which raced: two concurrent writers for the same natural key, or a write landing between another connection's SELECT and its own write, could bypass the rule. A single statement has no such window — Postgres evaluates the `WHERE` clause against the row as it exists at the instant of the write.

**What this does not cover:** a hypothetical *third* writer that INSERTs/UPDATEs `intel.financial_metrics` directly, bypassing both `NormalizedStorage` and the bulk-ingestion script, would not be subject to this guard — it's enforced by the query text these two callers use, not by a database-level trigger or constraint. A `before insert or update` trigger would close that gap, but needs a live Postgres to verify its semantics correctly interact with `ON CONFLICT` before shipping it — not added speculatively here.

## Lineage columns (added `db/migrations/0006_lineage.sql`)

`model_version` existed already (free-text) on most tables. This migration adds its missing siblings so a row can be traced to what produced it:

| Table | New columns |
|---|---|
| `public.extractions` | `prompt_version`, `parser_version` |
| `public.extraction_logs` | `parser_recovery_stage`, `dataset_version` |
| `intel.financial_metrics` | `prompt_version`, `parser_version`, `dataset_version` |
| `intel.mdna_summaries` | `prompt_version`, `parser_version` |
| `intel.extraction_runs` | `prompt_version`, `parser_version`, `dataset_version` |

All nullable — existing rows are unaffected, new writers populate them going forward. None of these are FK references to a versions table (that would be a larger change); they're free-text strings, same convention as the existing `model_version` columns, documented here rather than silently left inconsistent.

## Why not wire `NormalizedStorage` into the live serving path

`serving/api.py`'s hot path only ever builds a flat `ExtractionResult` (architecture A's shape); it never builds a `FilingRecord` (architecture B's shape, which `NormalizedStorage.save_filing_record()` requires). Making every live request also run section-parsing/normalization and a second set of DB writes would add real, currently-unmeasured latency, and there's no Docker/load-testing available in the environment that made this decision to validate the change is safe. Instead, `db/sync/sync_normalized_from_pipeline.py` runs the real pipeline over a known corpus as an offline batch job — `intel.*` gets a real (non-mocked) caller without touching production request latency. Revisit this once a load-testing harness (`evaluation/load_harness.py`) has real numbers to decide against.

## Example queries

```sql
-- All xbrl-sourced revenue figures for a ticker across its filings
select f.ticker, f.filing_date, m.period, m.value
from intel.financial_metrics m
join intel.filings f using (filing_id)
where f.ticker = 'AAPL' and m.metric_name = 'revenue' and m.method = 'xbrl'
order by f.filing_date;

-- Filings where an llm-sourced fact exists but no xbrl fact ever did for
-- the same metric/period/segment (the case the precedence rule is meant to
-- protect: llm is *filling a gap*, not overwriting a tagged fact)
select filing_id, metric_name, period, segment, value
from intel.financial_metrics m1
where method = 'llm'
  and not exists (
    select 1 from intel.financial_metrics m2
    where m2.filing_id = m1.filing_id
      and m2.metric_name = m1.metric_name
      and m2.period = m1.period
      and m2.segment = m1.segment
      and m2.method = 'xbrl'
  );

-- Parser-recovery-stage distribution for a given dataset_version, once
-- extraction_logs.parser_recovery_stage is being populated (see
-- src/extraction/parser_telemetry.py)
select parser_recovery_stage, count(*)
from public.extraction_logs
where dataset_version = 'v1-benchmark_real-real_edgar-...'
group by parser_recovery_stage
order by count(*) desc;
```
