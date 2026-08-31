from __future__ import annotations

import hashlib
import re

from src.schemas import (
    FilingRecord,
    MdnaSummaryRecord,
    MetricRecord,
    RiskFactorRecord,
    SectionRecord,
)


MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}


def parse_numeric_value(raw: str | int | float | None) -> float | int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw

    text = raw.lower().replace("$", "").replace(",", "").strip()
    multiplier = 1

    for token, value in MULTIPLIERS.items():
        if token in text:
            multiplier = value
            text = text.replace(token, "").strip()
            break

    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return None

    number = float(match.group(0)) * multiplier
    return int(number) if number.is_integer() else number


def normalize_metric(
    name: str,
    raw_value: str | int | float | None,
    *,
    unit: str | None = "usd",
    period: str | None = None,
    segment: str | None = None,
    method: str = "llm",
    confidence: float = 0.5,
    source_section: str | None = None,
    evidence_text: str | None = None,
    model_version: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        name=name,
        value=parse_numeric_value(raw_value),
        unit=unit,
        period=period,
        segment=segment,
        method=method,
        confidence=confidence,
        source_section=source_section,
        evidence_text=evidence_text,
        model_version=model_version,
    )


def resolve_metric_precedence(
    existing: MetricRecord | None, incoming: MetricRecord
) -> MetricRecord:
    """XBRL never overwritten by LLM/heuristic facts, per docs/BOUNDARY.md.

    Between heuristic and llm, last write wins -- BOUNDARY.md only shields
    xbrl facts, so heuristic and llm freely overwrite each other.
    """
    if existing is None:
        return incoming
    if existing.method == "xbrl" and incoming.method != "xbrl":
        return existing
    return incoming


# ─── FilingRecord -> normalized table rows ───────────────────────────────────

def section_to_row(filing_id: str, section: SectionRecord) -> dict:
    return {
        "filing_id": filing_id,
        "section_type": section.section_type,
        "title": section.title,
        "char_start": section.start,
        "char_end": section.end,
        "confidence": section.confidence,
    }


def metric_to_row(filing_id: str, metric: MetricRecord) -> dict:
    return {
        "filing_id": filing_id,
        "metric_name": metric.name,
        "period": metric.period or "",
        "segment": metric.segment or "",
        "value": metric.value,
        "unit": metric.unit,
        "method": metric.method,
        "confidence": metric.confidence,
        "source_section": metric.source_section,
        "evidence_text": metric.evidence_text,
        "model_version": metric.model_version,
    }


def risk_factor_to_row(filing_id: str, risk: RiskFactorRecord) -> dict:
    risk_hash = hashlib.sha256(risk.text.encode("utf-8")).hexdigest()
    return {
        "filing_id": filing_id,
        "text": risk.text,
        "source_section": risk.source_section,
        "confidence": risk.confidence,
        "risk_hash": risk_hash,
    }


def mdna_to_row(filing_id: str, mdna: MdnaSummaryRecord) -> dict:
    return {
        "filing_id": filing_id,
        "summary": mdna.summary,
        "method": mdna.method,
        "model_version": mdna.model_version,
    }


def filing_record_to_rows(record: FilingRecord) -> dict[str, list[dict]]:
    """Flatten a FilingRecord into row lists keyed by destination table name."""
    filing_id = record.metadata.filing_id

    rows: dict[str, list[dict]] = {
        "filings": [
            {
                "filing_id": filing_id,
                "cik": record.metadata.cik,
                "accession_no": record.metadata.accession_no,
                "ticker": record.metadata.ticker,
                "company_name": record.metadata.company_name,
                "filing_type": record.metadata.filing_type,
                "filing_date": record.metadata.filing_date,
                "raw_text_hash": record.metadata.raw_text_hash,
            }
        ],
        "filing_sections": [section_to_row(filing_id, s) for s in record.sections],
        "financial_metrics": [metric_to_row(filing_id, m) for m in record.metrics],
        "risk_factors": [risk_factor_to_row(filing_id, r) for r in record.risk_factors],
        "mdna_summaries": (
            [mdna_to_row(filing_id, record.mdna)] if record.mdna is not None else []
        ),
    }
    return rows