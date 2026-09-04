"""Stage-level telemetry for the JSON recovery cascade in postprocessing.py.

Before this module, parse_extraction() had no way to report *which* of its
5 stages recovered a given result -- there was no way to answer "how much of
our schema-conformance rate is the parser saving bad model output, vs. the
model getting it right directly?" (see docs/TRUTH_AUDIT.md). This is purely
additive: parse_extraction() takes an optional `telemetry` parameter that
defaults to None, and every telemetry.record() call is a no-op when
telemetry is None, so none of the three existing call sites need to change.

The 5 stages are already sequential and mutually exclusive (first success
wins) -- this is not a new state machine, it's instrumentation of the one
that already existed. ParseTelemetry.attempts *is* the state trace: reading
it top to bottom shows exactly which stages were tried, in order, and why
each one that failed, failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAGES = ("direct", "fence_strip", "regex_extract", "truncation_repair", "field_fallback")


@dataclass
class ParseAttempt:
    stage: str
    succeeded: bool
    fields_recovered: int | None = None
    reason_code: str | None = None


@dataclass
class ParseTelemetry:
    attempts: list[ParseAttempt] = field(default_factory=list)
    winning_stage: str | None = None
    raw_output_chars: int = 0

    def record(
        self,
        stage: str,
        succeeded: bool,
        *,
        fields_recovered: int | None = None,
        reason_code: str | None = None,
    ) -> None:
        self.attempts.append(ParseAttempt(
            stage=stage, succeeded=succeeded,
            fields_recovered=fields_recovered, reason_code=reason_code,
        ))
        if succeeded and self.winning_stage is None:
            self.winning_stage = stage

    def to_dict(self) -> dict:
        return {
            "winning_stage": self.winning_stage,
            "raw_output_chars": self.raw_output_chars,
            "attempts": [
                {
                    "stage": a.stage,
                    "succeeded": a.succeeded,
                    "fields_recovered": a.fields_recovered,
                    "reason_code": a.reason_code,
                }
                for a in self.attempts
            ],
        }
