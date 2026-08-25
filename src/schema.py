from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class FilingMetadata(BaseModel):
    filing_id: str
    cik: str | None = None
    ticker: str | None = None
    company_name: str | None = None
    filing_type: Literal["10-K", "10-Q", "S-1"] | str
    filing_date: str | None = None


class SectionRecord(BaseModel):
    section_type: str
    title: str
    text: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class MetricRecord(BaseModel):
    name: str
    value: float | int | str | None = None
    unit: str | None = None
    period: str | None = None
    segment: str | None = None
    method: Literal["xbrl", "heuristic", "llm"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_section: str | None = None
    evidence_text: str | None = None


class RiskFactorRecord(BaseModel):
    text: str
    source_section: str = "risk_factors"
    confidence: float = Field(ge=0.0, le=1.0)


class FilingRecord(BaseModel):
    metadata: FilingMetadata
    sections: list[SectionRecord]
    metrics: list[MetricRecord]
    risk_factors: list[RiskFactorRecord]
    mdna_summary: str | None = None