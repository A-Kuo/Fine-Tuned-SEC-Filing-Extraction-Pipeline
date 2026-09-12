# Extraction Boundary

This pipeline treats two kinds of facts differently, depending on whether
the SEC filer machine-tagged the number.

## Tagged (XBRL)

Where a filing carries an iXBRL tag for a fact, this pipeline treats that
value as ground truth: deterministic, machine-tagged, not something a
language model re-derives. Facts sourced this way are marked `method='xbrl'`.

## Untagged (narrative)

A large share of what a filing discloses — MD&A, footnotes, non-GAAP
reconciliations, untagged tables — is never tagged, because XBRL's taxonomy
doesn't cover free-form prose. This is what the fine-tuned LLM and the
heuristic parser extract from. Facts sourced this way are marked
`method='llm'` or `method='heuristic'`, each with a confidence score and
model version.

## Precedence

An `llm` or `heuristic` fact never overwrites an `xbrl` fact for the same
natural key. XBRL always wins. The reverse is permitted.
