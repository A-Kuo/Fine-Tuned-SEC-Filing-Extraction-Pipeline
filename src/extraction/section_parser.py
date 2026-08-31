from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class SectionSpan:
    section_type: str
    title: str
    start: int
    end: int
    text: str
    confidence: float


SECTION_PATTERNS = {
    "mdna": [
        re.compile(
            r"(item\s+7[\.\-:\s]+management[’'`s]{0,2}\s+discussion\s+and\s+analysis.*?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(management[’'`s]{0,2}\s+discussion\s+and\s+analysis.*?)",
            re.IGNORECASE,
        ),
    ],
    "risk_factors": [
        re.compile(r"(item\s+1a[\.\-:\s]+risk\s+factors.*?)", re.IGNORECASE),
        re.compile(r"(risk\s+factors.*?)", re.IGNORECASE),
    ],
    "financial_statements": [
        re.compile(
            r"(item\s+8[\.\-:\s]+financial\s+statements.*?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(consolidated\s+statements\s+of\s+operations.*?)",
            re.IGNORECASE,
        ),
    ],
}

STOP_PATTERNS = [
    re.compile(r"\n\s*item\s+\d+[a-z]?[\.\-:\s]", re.IGNORECASE),
    re.compile(r"\n\s*signatures\s*\n", re.IGNORECASE),
]


def _find_first_match(patterns: Iterable[re.Pattern], text: str) -> tuple[int, str] | None:
    matches = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.group(0)))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0]


def _find_end(text: str, start: int) -> int:
    remainder = text[start:]
    stops = []
    for pattern in STOP_PATTERNS:
        match = pattern.search(remainder)
        if match:
            stops.append(start + match.start())
    return min(stops) if stops else len(text)


def extract_sections(text: str) -> list[SectionSpan]:
    spans: list[SectionSpan] = []

    for section_type, patterns in SECTION_PATTERNS.items():
        hit = _find_first_match(patterns, text)
        if not hit:
            continue

        start, title = hit
        end = _find_end(text, start)
        section_text = text[start:end].strip()

        confidence = 0.9 if "item" in title.lower() else 0.7
        spans.append(
            SectionSpan(
                section_type=section_type,
                title=title[:200],
                start=start,
                end=end,
                text=section_text,
                confidence=confidence,
            )
        )

    spans.sort(key=lambda s: s.start)
    return spans