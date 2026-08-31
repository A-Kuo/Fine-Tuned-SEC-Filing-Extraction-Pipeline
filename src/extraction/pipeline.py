from __future__ import annotations

from typing import TYPE_CHECKING

from src.section_parser import extract_sections
from src.schemas import (
    FilingMetadata,
    FilingRecord,
    MdnaSummaryRecord,
    MetricRecord,
    RiskFactorRecord,
    SectionRecord,
)
from src.normalizer import normalize_metric, resolve_metric_precedence

if TYPE_CHECKING:
    from src.inference import ExtractionEngine
    from src.postprocessing import ExtractionResult


# Flat ExtractionResult financial fields -> normalized metric names.
_LLM_METRIC_FIELDS = {
    "revenue": "revenue",
    "net_income": "net_income",
    "total_assets": "total_assets",
    "total_liabilities": "total_liabilities",
    "eps": "eps",
}


def extraction_result_to_metrics(
    result: "ExtractionResult",
    *,
    model_version: str,
    confidence: float,
    source_section: str | None = None,
) -> list[MetricRecord]:
    """Map architecture A's flat ExtractionResult onto normalized MetricRecords.

    Every metric produced here carries method='llm', since it came from the
    QLoRA extraction engine rather than a deterministic XBRL tag or a keyword
    heuristic.
    """
    metrics: list[MetricRecord] = []
    for field_name, metric_name in _LLM_METRIC_FIELDS.items():
        raw_value = getattr(result, field_name, None)
        if raw_value is None:
            continue
        metrics.append(
            normalize_metric(
                metric_name,
                raw_value=raw_value,
                method="llm",
                confidence=confidence,
                source_section=source_section,
                evidence_text=str(raw_value),
                model_version=model_version,
            )
        )
    return metrics


def extract_llm_metrics(
    section_text: str,
    *,
    engine: "ExtractionEngine",
    filing_id: str | None = None,
    source_section: str | None = None,
) -> list[MetricRecord]:
    """Run architecture A's ExtractionEngine over section text and adapt the
    flat ExtractionResult into normalized MetricRecords (method='llm')."""
    from src.inference import ExtractionRequest

    response = engine.extract(ExtractionRequest(text=section_text, filing_id=filing_id))
    if response.result is None:
        return []
    return extraction_result_to_metrics(
        response.result,
        model_version=response.model_version,
        confidence=response.confidence_score,
        source_section=source_section,
    )


def build_filing_record(
    text: str,
    *,
    filing_id: str,
    filing_type: str,
    company_name: str | None = None,
    ticker: str | None = None,
    filing_date: str | None = None,
    engine: "ExtractionEngine | None" = None,
) -> FilingRecord:
    section_spans = extract_sections(text)

    sections = [
        SectionRecord(
            section_type=s.section_type,
            title=s.title,
            text=s.text,
            start=s.start,
            end=s.end,
            confidence=s.confidence,
        )
        for s in section_spans
    ]

    risk_factors = []
    for s in sections:
        if s.section_type == "risk_factors":
            paragraphs = [p.strip() for p in s.text.split("\n\n") if len(p.strip()) > 120]
            risk_factors.extend(
                RiskFactorRecord(text=p, confidence=s.confidence)
                for p in paragraphs[:10]
            )

    metrics: list[MetricRecord] = []
    for s in sections:
        lower = s.text.lower()
        if "revenue" in lower:
            metrics.append(
                normalize_metric(
                    "revenue",
                    raw_value=s.text,
                    source_section=s.section_type,
                    method="heuristic",
                    confidence=0.4,
                    evidence_text=s.text[:400],
                )
            )

    if engine is not None:
        llm_metrics: list[MetricRecord] = []
        for s in sections:
            if s.section_type in ("mdna", "financial_statements"):
                llm_metrics.extend(
                    extract_llm_metrics(
                        s.text,
                        engine=engine,
                        filing_id=filing_id,
                        source_section=s.section_type,
                    )
                )

        by_key: dict[tuple[str, str | None, str | None], MetricRecord] = {
            (m.name, m.period, m.segment): m for m in metrics
        }
        for incoming in llm_metrics:
            key = (incoming.name, incoming.period, incoming.segment)
            by_key[key] = resolve_metric_precedence(by_key.get(key), incoming)
        metrics = list(by_key.values())

    metadata = FilingMetadata(
        filing_id=filing_id,
        filing_type=filing_type,
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
    )

    mdna = None
    for s in sections:
        if s.section_type == "mdna":
            mdna = MdnaSummaryRecord(summary=s.text[:1000], method="heuristic")
            break

    return FilingRecord(
        metadata=metadata,
        sections=sections,
        metrics=metrics,
        risk_factors=risk_factors,
        mdna=mdna,
    )