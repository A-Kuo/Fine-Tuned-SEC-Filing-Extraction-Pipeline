"""Four benchmark splits for real-world evaluation, each backed by a
DatasetManifest so every split is versioned and checksummed like any other
dataset in this repo.

- synthetic_templated: the repo's one hand-authored synthetic filing --
  the same fixture the README's original 94%/92-99% (unverified) numbers
  were supposedly measured against.
- manually_curated_authentic: real EDGAR 10-Ks from large, well-known
  filers (AAPL, MSFT, KO) -- the kind of filing a synthetic generator
  would most plausibly resemble.
- adversarial_formatting: the same real filings, deliberately corrupted
  with formatting variations real documents actually exhibit (extra
  whitespace, currency symbol variants, HTML-entity leftovers) -- pure
  text transforms, no model needed to build this split.
- out_of_domain_sector: real EDGAR 10-Ks from sectors underrepresented in
  a typical synthetic generator's training distribution (biotech, REIT).

Usage:
    python evaluation/benchmark_splits.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.dataset_manifest import (
    DatasetManifest,
    compute_dataset_checksum,
    compute_record_checksum,
    make_dataset_version,
)

# Sectors deliberately absent from AAPL/MSFT/KO -- see docstring.
OUT_OF_DOMAIN_TICKERS = {"MRNA", "O"}
CURATED_AUTHENTIC_TICKERS = {"AAPL", "MSFT", "KO"}


def _load_manifest_entries() -> list[dict]:
    manifest_path = REPO_ROOT / "data" / "raw_edgar" / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _filing_case(entry: dict) -> dict:
    text_path = REPO_ROOT / entry["text_path"]
    return {
        "corpus": "real_edgar",
        "source": entry["source_url"],
        "filing_id": f"{entry['ticker']}-{entry['accessionNumber']}",
        "filing_type": entry["form"],
        "company_name": entry["company"],
        "ticker": entry["ticker"],
        "filing_date": entry["filingDate"],
        "text": text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else "",
    }


def synthetic_templated_split() -> list[dict]:
    text = (REPO_ROOT / "data" / "sample_10k.txt").read_text(encoding="utf-8")
    return [{
        "corpus": "synthetic",
        "source": "data/sample_10k.txt",
        "filing_id": "synthetic-sample-10k",
        "filing_type": "10-K",
        "text": text,
    }]


def manually_curated_authentic_split() -> list[dict]:
    entries = _load_manifest_entries()
    return [_filing_case(e) for e in entries if e["ticker"] in CURATED_AUTHENTIC_TICKERS]


def out_of_domain_sector_split() -> list[dict]:
    entries = _load_manifest_entries()
    return [_filing_case(e) for e in entries if e["ticker"] in OUT_OF_DOMAIN_TICKERS]


# ─── Adversarial formatting ────────────────────────────────────────────────

def _corrupt_extra_whitespace(text: str) -> str:
    """Real filings sometimes render with irregular whitespace from PDF/HTML
    conversion artifacts."""
    lines = text.splitlines()
    corrupted = []
    for i, line in enumerate(lines):
        if i % 5 == 0 and line.strip():
            corrupted.append("   " + re.sub(r" ", "  ", line))
        else:
            corrupted.append(line)
    return "\n\n".join(corrupted)


def _corrupt_currency_symbols(text: str) -> str:
    """Some filings use USD/US$ instead of $; some use unicode minus."""
    # Must run before the \$ -> USD substitution below, which would
    # otherwise consume the "$" and leave no "-$" substring to replace.
    text = text.replace("-$", "−$")
    return re.sub(r"\$(\d)", r"USD \1", text)


def _corrupt_html_entity_leftovers(text: str) -> str:
    """A common real-world artifact: HTML entities that survive a lossy
    text-extraction pass (this repo's own html_to_visible_text() doesn't
    leave these, but plenty of upstream EDGAR-adjacent tooling does)."""
    replacements = {"'": "&#8217;", "&": "&amp;", '"': "&quot;"}
    out = text
    for char, entity in replacements.items():
        # Corrupt only every other occurrence -- a real leftover artifact is
        # inconsistent, not total.
        parts = out.split(char)
        out = parts[0]
        for i, part in enumerate(parts[1:], start=1):
            out += (entity if i % 2 == 0 else char) + part
    return out


ADVERSARIAL_TRANSFORMS = {
    "extra_whitespace": _corrupt_extra_whitespace,
    "currency_symbol_variants": _corrupt_currency_symbols,
    "html_entity_leftovers": _corrupt_html_entity_leftovers,
}


def adversarial_formatting_split() -> list[dict]:
    """Applies each transform to each curated-authentic filing --
    deliberately corrupted variants of real text, not synthetic garbage."""
    base_cases = manually_curated_authentic_split()
    cases = []
    for base in base_cases:
        for transform_name, transform_fn in ADVERSARIAL_TRANSFORMS.items():
            case = dict(base)
            case["filing_id"] = f"{base['filing_id']}-adv-{transform_name}"
            case["source"] = f"{base['source']} (corrupted: {transform_name})"
            case["text"] = transform_fn(base["text"])
            cases.append(case)
    return cases


SPLITS = {
    "benchmark_synthetic": synthetic_templated_split,
    "benchmark_real": manually_curated_authentic_split,
    "benchmark_adversarial": adversarial_formatting_split,
    "benchmark_ood": out_of_domain_sector_split,
}


def build_manifest_for_split(split_name: str, cases: list[dict]) -> DatasetManifest:
    checksums = [compute_record_checksum({k: v for k, v in c.items() if k != "text"} | {"text_len": len(c.get("text", ""))}) for c in cases]
    checksum = compute_dataset_checksum(checksums)
    generated_at = datetime.now(timezone.utc)
    version = make_dataset_version(
        schema_version=1, source_split=split_name, template_family="benchmark_splits",
        generated_at=generated_at, checksum=checksum,
    )
    return DatasetManifest(
        dataset_version=version,
        source_split=split_name,
        template_family="benchmark_splits",
        checksum=checksum,
        record_count=len(cases),
        created_at=generated_at.isoformat(),
        lineage={"builder": "evaluation/benchmark_splits.py"},
    )


def main():
    out_dir = REPO_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, builder in SPLITS.items():
        cases = builder()
        manifest = build_manifest_for_split(split_name, cases)
        manifest_path = out_dir / f"{manifest.dataset_version}.manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        print(f"{split_name}: {len(cases)} cases -> {manifest_path.relative_to(REPO_ROOT)}")
        if not cases:
            print(f"  WARNING: {split_name} is empty (see module docstring for how to populate it)")


if __name__ == "__main__":
    main()
