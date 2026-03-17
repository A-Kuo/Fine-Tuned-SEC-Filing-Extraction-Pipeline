"""Post-processing for model extraction outputs.

Handles the messy reality of LLM outputs: the model *usually* returns valid
JSON, but sometimes includes markdown fences, trailing text, partial objects,
or hallucinated fields. This module robustly parses whatever the model produces.

Key design decision: we validate but don't reject partial results. A filing
with company_name + filing_type but missing revenue is still useful—we return
it with a lower confidence score and flag the missing fields. This is better
than discarding the entire extraction.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.config import load_config


@dataclass
class ExtractionResult:
    """Structured extraction from a single SEC filing.

    All fields are Optional because the model may not extract every field.
    This mirrors real-world usage where partial extractions are common
    and still valuable.
    """
    filing_id: str | None = None
    company_name: str | None = None
    ticker: str | None = None
    filing_type: str | None = None
    date: str | None = None
    fiscal_year_end: str | None = None
    revenue: str | None = None
    net_income: str | None = None
    total_assets: str | None = None
    total_liabilities: str | None = None
    eps: str | None = None
    sector: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def filled_fields(self) -> int:
        """Count of non-None fields."""
        return sum(1 for v in asdict(self).values() if v is not None)

    @property
    def total_fields(self) -> int:
        return len(asdict(self))

    @property
    def completeness(self) -> float:
        """Fraction of fields that are non-None."""
        return self.filled_fields / self.total_fields


class ValidationError(Exception):
    """Raised when extraction fails validation."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


# ─── JSON Parsing ────────────────────────────────────────────────────────────

def parse_extraction(raw_output: str) -> ExtractionResult:
    """Parse model output into ExtractionResult.

    Handles common LLM output quirks:
    - Markdown code fences (```json ... ```)
    - Trailing text after JSON
    - Missing closing braces
    - Extra whitespace / newlines

    Args:
        raw_output: Raw text from model generation.

    Returns:
        ExtractionResult with whatever fields were successfully parsed.

    Raises:
        json.JSONDecodeError if no valid JSON can be extracted.
    """
    text = raw_output.strip()

    # Strategy 1: Try direct parse
    parsed = _try_parse(text)
    if parsed is not None:
        return _dict_to_result(parsed)

    # Strategy 2: Strip markdown fences
    stripped = _strip_code_fences(text)
    parsed = _try_parse(stripped)
    if parsed is not None:
        return _dict_to_result(parsed)

    # Strategy 3: Extract JSON object with regex
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        parsed = _try_parse(json_match.group())
        if parsed is not None:
            return _dict_to_result(parsed)

    # Strategy 4: Try to fix truncated JSON (missing closing brace)
    fixed = _fix_truncated_json(text)
    if fixed:
        parsed = _try_parse(fixed)
        if parsed is not None:
            logger.warning("Parsed truncated JSON (added missing braces)")
            return _dict_to_result(parsed)

    # Strategy 5: Extract key-value pairs with regex as last resort
    result = _regex_extract(text)
    if result.filled_fields > 0:
        logger.warning(f"Fell back to regex extraction ({result.filled_fields} fields)")
        return result

    raise json.JSONDecodeError("No valid JSON found in model output", text, 0)


def _try_parse(text: str) -> dict | None:
    """Attempt JSON parse, return None on failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
    text = re.sub(r'\n?```\s*$', '', text.strip())
    return text.strip()


def _fix_truncated_json(text: str) -> str | None:
    """Try to fix JSON truncated by max_tokens limit.

    If the model hit the token limit mid-generation, the JSON might be
    missing closing braces. We count open/close braces and add missing ones.
    """
    # Find the start of JSON
    start = text.find('{')
    if start < 0:
        return None

    json_text = text[start:]

    # Count unmatched braces
    open_count = json_text.count('{')
    close_count = json_text.count('}')
    missing = open_count - close_count

    if missing > 0 and missing <= 3:  # Only fix small truncations
        # Remove trailing partial key-value pair
        last_comma = json_text.rfind(',')
        last_complete = json_text.rfind('"', 0, last_comma) if last_comma > 0 else -1

        if last_comma > 0 and last_complete > 0:
            # Find the end of the last complete value
            after_comma = json_text[last_comma + 1:].strip()
            if ':' not in after_comma or '"' not in after_comma.split(':')[-1]:
                # Incomplete pair after last comma; truncate there
                json_text = json_text[:last_comma]

        json_text += '}' * missing
        return json_text

    return None


def _regex_extract(text: str) -> ExtractionResult:
    """Last-resort extraction using regex patterns.

    When JSON parsing fails entirely, try to extract individual fields
    using pattern matching. Better than returning nothing.
    """
    result = ExtractionResult()

    patterns = {
        "company_name": [
            r'"company_name"\s*:\s*"([^"]+)"',
            r'(?:Registrant|Company):\s*(.+?)(?:\n|$)',
        ],
        "ticker": [
            r'"ticker"\s*:\s*"([^"]+)"',
            r'\((?:Ticker:\s*)?([A-Z]{1,5})\)',
        ],
        "filing_type": [
            r'"filing_type"\s*:\s*"([^"]+)"',
            r'FORM\s+(10-[KQ](?:/A)?|8-K)',
        ],
        "date": [
            r'"date"\s*:\s*"([^"]+)"',
            r'Filed?\s*(?:Date)?:\s*(\d{4}-\d{2}-\d{2})',
        ],
        "revenue": [
            r'"revenue"\s*:\s*"([^"]+)"',
            r'Revenue[^:]*:\s*\$?([\d.,]+\s*(?:billion|million|trillion)?)',
        ],
        "net_income": [
            r'"net_income"\s*:\s*"([^"]+)"',
            r'Net\s*Income[^:]*:\s*\$?([\d.,]+\s*(?:billion|million|trillion)?)',
        ],
    }

    for field_name, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                setattr(result, field_name, match.group(1).strip())
                break

    return result


def _dict_to_result(data: dict) -> ExtractionResult:
    """Convert a parsed dict to ExtractionResult, handling key variations."""
    # Normalize keys (handle snake_case, camelCase, spaces)
    normalized = {}
    key_map = {
        "filing_id": ["filing_id", "filingId", "accession_number"],
        "company_name": ["company_name", "companyName", "company", "registrant"],
        "ticker": ["ticker", "symbol", "stock_ticker"],
        "filing_type": ["filing_type", "filingType", "form_type", "formType", "type"],
        "date": ["date", "filing_date", "filingDate", "filed_date"],
        "fiscal_year_end": ["fiscal_year_end", "fiscalYearEnd", "period_end", "periodEnd"],
        "revenue": ["revenue", "total_revenue", "totalRevenue", "net_revenue"],
        "net_income": ["net_income", "netIncome", "net_profit"],
        "total_assets": ["total_assets", "totalAssets", "assets"],
        "total_liabilities": ["total_liabilities", "totalLiabilities", "liabilities"],
        "eps": ["eps", "earnings_per_share", "earningsPerShare", "diluted_eps"],
        "sector": ["sector", "industry", "segment"],
    }

    for result_key, source_keys in key_map.items():
        for src_key in source_keys:
            if src_key in data and data[src_key] is not None:
                normalized[result_key] = str(data[src_key])
                break

    return ExtractionResult(**normalized)


# ─── Validation ──────────────────────────────────────────────────────────────

VALID_FILING_TYPES = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "S-1", "DEF 14A"}


def validate_extraction(result: ExtractionResult) -> tuple[bool, list[str]]:
    """Validate an extraction against business rules.

    Returns:
        (is_valid, errors) tuple. is_valid=True if all required fields pass.
        Partial results may still be usable even if is_valid=False.
    """
    config = load_config()
    required_fields = config["extraction"]["required_fields"]
    errors = []

    # Check required fields are present
    for field_name in required_fields:
        value = getattr(result, field_name, None)
        if not value or value.strip() == "":
            errors.append(f"Missing required field: {field_name}")

    # Validate filing type
    if result.filing_type and result.filing_type not in VALID_FILING_TYPES:
        errors.append(f"Invalid filing type: {result.filing_type}")

    # Validate date format
    if result.date:
        if not _is_valid_date(result.date):
            errors.append(f"Invalid date format: {result.date}")

    if result.fiscal_year_end:
        if not _is_valid_date(result.fiscal_year_end):
            errors.append(f"Invalid fiscal_year_end format: {result.fiscal_year_end}")

    # Validate financial figures (should contain digits)
    financial_fields = ["revenue", "net_income", "total_assets", "total_liabilities", "eps"]
    for field_name in financial_fields:
        value = getattr(result, field_name, None)
        if value and not re.search(r'\d', value):
            errors.append(f"Financial field '{field_name}' has no numeric content: {value}")

    is_valid = len(errors) == 0
    return is_valid, errors


def _is_valid_date(date_str: str) -> bool:
    """Check if string is a valid date in common formats."""
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y"]
    for fmt in formats:
        try:
            datetime.strptime(date_str.strip(), fmt)
            return True
        except ValueError:
            continue
    return False
