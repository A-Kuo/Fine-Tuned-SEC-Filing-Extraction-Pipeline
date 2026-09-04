-- Adds prompt_version/parser_version/dataset_version columns so a stored
-- row can be traced back to exactly what produced it (model_version already
-- existed as a free-text column on most of these tables; this migration
-- adds its missing siblings, not a replacement for it). See
-- docs/TRUTH_AUDIT.md for why this was missing and db/SCHEMA.md for the
-- full column reference.
--
-- All statements are additive (add column if not exists) -- consistent with
-- every prior migration in this directory. Safe to run against a database
-- that already has data in these tables.

-- public.* is the schema architecture A (src/storage/database.py) actually
-- writes to in production today -- lineage columns here have an immediate,
-- live caller.
alter table public.extractions
    add column if not exists prompt_version varchar(64),
    add column if not exists parser_version varchar(64);

alter table public.extraction_logs
    add column if not exists parser_recovery_stage varchar(32),
    add column if not exists dataset_version varchar(128);

-- intel.* has no live caller as of this migration (see docs/TRUTH_AUDIT.md --
-- src/storage/normalized_storage.py is currently only exercised by
-- db/sync/sync_normalized_from_pipeline.py, an offline batch job, and by
-- mocked tests). Added proactively while the schema is still young and cheap
-- to alter, so a second migration isn't needed the moment it gets a second
-- real caller.
alter table intel.financial_metrics
    add column if not exists prompt_version varchar(64),
    add column if not exists parser_version varchar(64),
    add column if not exists dataset_version varchar(128);

alter table intel.mdna_summaries
    add column if not exists prompt_version varchar(64),
    add column if not exists parser_version varchar(64);

alter table intel.extraction_runs
    add column if not exists prompt_version varchar(64),
    add column if not exists parser_version varchar(64),
    add column if not exists dataset_version varchar(128);
