# Extraction Boundary

Two repositories, one seam. The seam is whether the SEC filer
machine-tagged the number.

## sec-edgar-extraction-pipeline — TAGGED

Owns: iXBRL-tagged facts. Ingestion, rate limiting, amendment
chains, Postgres, audit trail, quality gates, anomaly scoring.
Never: runs a language model. Never: extracts from prose.
Emits: filing documents + deterministic facts, `method='xbrl'`.

## Fine-Tuned-SEC-Filing-Extraction-Pipeline — UNTAGGED

Owns: extraction from narrative — MD&A, footnotes, non-GAAP
reconciliations, untagged tables. Training, eval, GPU serving.
Never: implements EDGAR ingestion, rate limiting, or amendment
logic. Consumes those from the pipeline repo.
Emits: facts marked `method='llm'` with a confidence and model version.

## Precedence

An `llm` fact never overwrites an `xbrl` fact for the same
natural key. XBRL always wins. The reverse is permitted.
