"""Extract ground-truth financial facts from EDGAR HTML / inline XBRL.

Parses common inline XBRL patterns (ix:nonFraction, data-tag attributes)
and falls back to regex for us-gaap context strings.

This is a best-effort parser for labeling and evaluation — not a full XBRL processor.
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree as ET

# Common US-GAAP tags used for ground-truth labeling
GAAP_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def extract_xbrl_facts(html_or_xml: str) -> dict[str, Any]:
    """Extract numeric facts from inline XBRL in HTML or raw XML.

    Returns a dict of tag_local_name -> {value, unit, decimals, raw}
    """
    facts: dict[str, Any] = {}

    # Inline XBRL: ix:nonFraction
    for m in re.finditer(
        r'<(?:ix:)?nonFraction[^>]*name="([^"]+)"[^>]*>([^<]+)</',
        html_or_xml,
        re.IGNORECASE | re.DOTALL,
    ):
        name = m.group(1)
        if ":" in name:
            name = name.split(":")[-1]
        val = m.group(2).strip().replace(",", "")
        try:
            facts[f"ix:{name}"] = {"value": float(val), "source": "ix_nonFraction"}
        except ValueError:
            facts[f"ix:{name}"] = {"value": None, "raw": val, "source": "ix_nonFraction"}

    # Try XML parse for namespace-agnostic elements
    try:
        # Wrap fragment if needed
        if "<html" in html_or_xml.lower()[:2000]:
            snippet = html_or_xml
        else:
            snippet = f"<root>{html_or_xml}</root>"

        root = ET.fromstring(snippet)
        for elem in root.iter():
            tag = _strip_ns(elem.tag).lower()
            if "nonfraction" in tag or "nonnumeric" in tag:
                name = elem.attrib.get("name", "")
                if name:
                    short = name.split(":")[-1] if ":" in name else name
                    text = (elem.text or "").strip()
                    if text:
                        try:
                            num = float(text.replace(",", ""))
                            facts[f"xml:{short}"] = {"value": num, "source": "xml"}
                        except ValueError:
                            facts[f"xml:{short}"] = {"value": None, "raw": text, "source": "xml"}
    except ET.ParseError:
        pass

    # Heuristic: search for revenue / net income in plain text tables (weak)
    for label, pattern in (
        ("Revenues_heuristic", r"Total\s+revenue[,\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"),
        ("NetIncome_heuristic", r"Net\s+income[,\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"),
    ):
        m = re.search(pattern, html_or_xml, re.IGNORECASE)
        if m and label not in facts:
            raw = m.group(1).replace(",", "")
            mult = (m.group(2) or "").lower()
            factor = {"billion": 1e9, "million": 1e6, "thousand": 1e3}.get(mult, 1)
            try:
                facts[label] = {"value": float(raw) * factor, "source": "heuristic"}
            except ValueError:
                pass

    return facts


def map_to_training_fields(facts: dict[str, Any]) -> dict[str, float | None]:
    """Map extracted XBRL facts to FinDocAnalyzer extraction fields (millions USD)."""
    out: dict[str, float | None] = {
        "revenue": None,
        "net_income": None,
        "total_assets": None,
        "total_liabilities": None,
        "eps": None,
    }

    def _get(*keys: str) -> float | None:
        for k in keys:
            for fk, fv in facts.items():
                if k.lower() in fk.lower() and isinstance(fv, dict) and fv.get("value") is not None:
                    v = fv["value"]
                    if isinstance(v, (int, float)):
                        return float(v)
        return None

    out["revenue"] = _get("Revenue", "revenue", "Revenues")
    out["net_income"] = _get("NetIncome", "NetIncomeLoss", "net_income")
    out["total_assets"] = _get("Assets", "Assets")
    out["total_liabilities"] = _get("Liabilities", "Liabilities")
    out["eps"] = _get("EarningsPerShare", "EPS", "eps")

    return out
