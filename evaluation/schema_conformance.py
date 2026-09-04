"""Schema-conformance test: does every extracted FilingRecord validate against
its own declared schema, on both synthetic and real filings?

This does NOT measure extraction *accuracy* (whether the values are correct --
that is evaluate.py / evaluate_real_filings.py). It measures a narrower,
mechanical claim: the extraction pipeline (src/extraction) produces output
that conforms to its own contract (src/core/schemas.py -> FilingRecord),
every time, on real untagged prose, not just on the synthetic examples it was
tuned against.

The schema validated against is derived live from the Pydantic model via
`model_json_schema()`, not hand-maintained -- schemas/api/*.schema.json in
this repo are currently empty stubs (0 bytes), so this is the only schema
that is actually authoritative for FilingRecord's real shape.

Usage:
    python evaluation/schema_conformance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

try:
    import tomllib  # stdlib, Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # backport for 3.10, which pyproject.toml/CI still support

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import FilingRecord
from src.extraction.pipeline import build_filing_record

REPO_ROOT = Path(__file__).parent.parent


def _parser_version() -> str:
    """pyproject.toml semver + the exact commit under test, so a report is
    traceable to the code that produced it even between version bumps."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    semver = pyproject["project"]["version"]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    return f"{semver}+{sha}"


def _synthetic_corpus() -> list[dict]:
    """The repo's one hand-authored synthetic filing (data/sample_10k.txt) --
    the same fixture the README's 94%/92-99% synthetic numbers are measured
    against."""
    text = (REPO_ROOT / "data" / "sample_10k.txt").read_text(encoding="utf-8")
    return [{
        "corpus": "synthetic",
        "source": "data/sample_10k.txt",
        "filing_id": "synthetic-sample-10k",
        "filing_type": "10-K",
        "text": text,
    }]


def _real_edgar_corpus() -> list[dict]:
    """Real, untagged EDGAR 10-K prose, fetched live via scripts/fetch_edgar.py
    (see data/raw_edgar/manifest.jsonl for provenance: CIK, accession number,
    source URL, fetch timestamp)."""
    manifest_path = REPO_ROOT / "data" / "raw_edgar" / "manifest.jsonl"
    if not manifest_path.exists():
        return []

    corpus = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        text_path = REPO_ROOT / entry["text_path"]
        if not text_path.exists():
            continue
        corpus.append({
            "corpus": "real_edgar",
            "source": entry["source_url"],
            "filing_id": f"{entry['ticker']}-{entry['accessionNumber']}",
            "filing_type": entry["form"],
            "company_name": entry["company"],
            "ticker": entry["ticker"],
            "filing_date": entry["filingDate"],
            "text": text_path.read_text(encoding="utf-8", errors="replace"),
        })
    return corpus


def run_conformance(schema: dict) -> dict:
    validator = jsonschema.Draft202012Validator(schema)

    cases = _synthetic_corpus() + _real_edgar_corpus()
    if len(cases) == 1:
        print(
            "WARNING: no real EDGAR filings found (data/raw_edgar/manifest.jsonl "
            "missing or empty). Run scripts/fetch_edgar.py first for a real-filing "
            "conformance measurement, not just the synthetic one.",
            file=sys.stderr,
        )

    per_case = []
    by_corpus: dict[str, dict[str, int]] = {}

    for case in cases:
        corpus = case["corpus"]
        by_corpus.setdefault(corpus, {"pass": 0, "fail": 0})

        record = build_filing_record(
            case["text"],
            filing_id=case["filing_id"],
            filing_type=case["filing_type"],
            company_name=case.get("company_name"),
            ticker=case.get("ticker"),
            filing_date=case.get("filing_date"),
        )
        payload = json.loads(record.model_dump_json())

        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
        passed = len(errors) == 0
        by_corpus[corpus]["pass" if passed else "fail"] += 1

        per_case.append({
            "corpus": corpus,
            "filing_id": case["filing_id"],
            "source": case["source"],
            "passed": passed,
            "n_sections": len(record.sections),
            "n_metrics": len(record.metrics),
            "n_risk_factors": len(record.risk_factors),
            "errors": [
                {"path": list(e.path), "message": e.message} for e in errors[:5]
            ],
        })

    total_pass = sum(c["pass"] for c in by_corpus.values())
    total = sum(c["pass"] + c["fail"] for c in by_corpus.values())

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": _parser_version(),
        "schema_under_test": "src.core.schemas.FilingRecord (live model_json_schema())",
        "note": (
            "schemas/api/extraction-request.schema.json and "
            "extraction_response.schema.json are currently empty (0 bytes) in this "
            "repo -- they are not what was validated against here."
        ),
        "overall": {"numerator": total_pass, "denominator": total},
        "by_corpus": {
            name: {
                "numerator": counts["pass"],
                "denominator": counts["pass"] + counts["fail"],
                "test_corpus": (
                    "data/sample_10k.txt (1 hand-authored synthetic filing)"
                    if name == "synthetic"
                    else "data/raw_edgar/manifest.jsonl (real EDGAR 10-Ks, "
                         "see per_case[].source for exact filing URLs)"
                ),
            }
            for name, counts in by_corpus.items()
        },
        "cases": per_case,
    }


def main():
    schema = FilingRecord.model_json_schema()
    results = run_conformance(schema)

    out_dir = REPO_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "schema_conformance_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Parser version:  {results['parser_version']}")
    for name, c in results["by_corpus"].items():
        print(f"  {name:12s}: {c['numerator']}/{c['denominator']} conformant  ({c['test_corpus']})")
    print(f"  {'overall':12s}: {results['overall']['numerator']}/{results['overall']['denominator']}")
    print(f"\nReport written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
