from __future__ import annotations

from src.section_parser import extract_sections
from src.schemas import FilingMetadata, FilingRecord, RiskFactorRecord, SectionRecord
from src.normalizer import normalize_metric


def build_filing_record(
    text: str,
    *,
    filing_id: str,
    filing_type: str,
    company_name: str | None = None,
    ticker: str | None = None,
    filing_date: str | None = None,
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

    metrics = []
    for s in sections:
        lower = s.text.lower()
        if "revenue" in lower:
            metrics.append(
                normalize_metric(
                    "revenue",
                    raw_value=s.text,
                    source_section=s.section_type,
                    confidence=0.4,
                    evidence_text=s.text[:400],
                )
            )

    metadata = FilingMetadata(
        filing_id=filing_id,
        filing_type=filing_type,
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
    )

    mdna_summary = None
    for s in sections:
        if s.section_type == "mdna":
            mdna_summary = s.text[:1000]
            break

    return FilingRecord(
        metadata=metadata,
        sections=sections,
        metrics=metrics,
        risk_factors=risk_factors,
        mdna_summary=mdna_summary,
    )