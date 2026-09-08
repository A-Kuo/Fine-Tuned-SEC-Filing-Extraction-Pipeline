"""Tests for scripts/backfill_live_schema.py.

Covers the field-mapping logic (_build_result) and the per-filing write
sequence (backfill), using a mocked PostgresStorage -- no live Postgres
needed. A dict-key mismatch between manifest entries / .xbrl.json and
ExtractionResult's field names would otherwise silently produce all-null
rows without erroring, so these tests exercise the real field names
end-to-end rather than just checking "no exception raised."
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.backfill_live_schema as backfill_mod
from scripts.backfill_live_schema import _build_result, backfill


def _manifest_entry(tmp_path: Path, with_xbrl: bool = True) -> dict:
    entry = {
        "ticker": "AAPL",
        "cik": "0000320193",
        "company": "Apple Inc.",
        "form": "10-K",
        "accessionNumber": "0000320193-25-000079",
        "filingDate": "2025-10-31",
        "source_url": "https://sec.gov/example",
        "text_path": "data/raw_edgar/AAPL_x.txt",
    }
    if with_xbrl:
        xbrl_path = tmp_path / "AAPL_x.xbrl.json"
        xbrl_path.write_text(json.dumps({
            "ix:Revenues": {"value": 383_285_000_000.0, "source": "ix_nonFraction"},
            "ix:NetIncomeLoss": {"value": 96_995_000_000.0, "source": "ix_nonFraction"},
        }))
        entry["xbrl_path"] = "xbrl_rel.json"  # resolved via monkeypatched get_project_root below
    return entry


class TestBuildResult:
    def test_identity_fields_come_from_manifest_not_guessed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        entry = _manifest_entry(tmp_path, with_xbrl=False)

        result, has_real_facts = _build_result(entry)

        assert result.filing_id == "AAPL-0000320193-25-000079"
        assert result.company_name == "Apple Inc."
        assert result.ticker == "AAPL"
        assert result.filing_type == "10-K"
        assert result.date == "2025-10-31"
        assert has_real_facts is False
        assert result.revenue is None

    def test_numeric_fields_from_real_xbrl_facts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        entry = _manifest_entry(tmp_path, with_xbrl=True)
        entry["xbrl_path"] = "AAPL_x.xbrl.json"

        result, has_real_facts = _build_result(entry)

        assert has_real_facts is True
        # map_to_training_fields() converts magnitude fields to millions USD
        # (matching MODEL_CARD.md's documented field contract)
        assert result.revenue == 383_285.0
        assert result.net_income == 96_995.0

    def test_missing_xbrl_file_on_disk_leaves_fields_none(self, monkeypatch, tmp_path):
        """xbrl_path recorded in the manifest but the file itself is gone --
        must not crash, must not guess."""
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        entry = _manifest_entry(tmp_path, with_xbrl=False)
        entry["xbrl_path"] = "does_not_exist.xbrl.json"

        result, has_real_facts = _build_result(entry)

        assert has_real_facts is False
        assert result.revenue is None


class TestBackfillWriteSequence:
    def _make_storage(self):
        storage = MagicMock()
        storage.store_extraction.return_value = True
        return storage

    def test_writes_all_four_tables_per_filing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        manifest_path = tmp_path / "data" / "raw_edgar"
        manifest_path.mkdir(parents=True)
        entry = _manifest_entry(tmp_path, with_xbrl=False)
        (manifest_path / "manifest.jsonl").write_text(json.dumps(entry) + "\n")

        storage = self._make_storage()
        synced, failed = backfill(storage)

        assert synced == 1
        assert failed == 0
        storage.store_extraction.assert_called_once()
        storage.log_extraction.assert_called_once()
        storage.record_ab_result.assert_called_once()
        storage.upsert_pipeline_stage.assert_called_once()

    def test_ab_result_is_always_single_arm(self, monkeypatch, tmp_path):
        """No second real model variant exists -- every backfilled row must
        be is_challenger=False, never presented as a real A/B comparison."""
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        manifest_path = tmp_path / "data" / "raw_edgar"
        manifest_path.mkdir(parents=True)
        entry = _manifest_entry(tmp_path, with_xbrl=False)
        (manifest_path / "manifest.jsonl").write_text(json.dumps(entry) + "\n")

        storage = self._make_storage()
        backfill(storage)

        _, kwargs = storage.record_ab_result.call_args
        assert kwargs.get("is_challenger") is False

    def test_method_is_always_xbrl_never_llm(self, monkeypatch, tmp_path):
        """Backfilled rows must never claim method='llm' -- the fine-tuned
        model never ran on this data."""
        monkeypatch.setattr(backfill_mod, "get_project_root", lambda: tmp_path)
        manifest_path = tmp_path / "data" / "raw_edgar"
        manifest_path.mkdir(parents=True)
        entry = _manifest_entry(tmp_path, with_xbrl=False)
        (manifest_path / "manifest.jsonl").write_text(json.dumps(entry) + "\n")

        storage = self._make_storage()
        backfill(storage)

        _, kwargs = storage.store_extraction.call_args
        assert kwargs.get("method") == "xbrl"
        assert kwargs.get("model_version") != "llama-sec-v1"
