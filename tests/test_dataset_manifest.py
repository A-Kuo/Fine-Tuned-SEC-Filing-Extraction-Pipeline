"""Tests for src/core/dataset_manifest.py.

The point of these checksums is that they're reproducible evidence -- two
runs over the same content must produce the same dataset_version, regardless
of record order or dict key order.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset_manifest import (
    DatasetManifest,
    compute_dataset_checksum,
    compute_record_checksum,
    make_dataset_version,
)


class TestComputeRecordChecksum:
    def test_deterministic(self):
        record = {"a": 1, "b": 2}
        assert compute_record_checksum(record) == compute_record_checksum(record)

    def test_key_order_does_not_matter(self):
        assert compute_record_checksum({"a": 1, "b": 2}) == compute_record_checksum({"b": 2, "a": 1})

    def test_different_content_different_checksum(self):
        assert compute_record_checksum({"a": 1}) != compute_record_checksum({"a": 2})

    def test_returns_hex_sha256(self):
        checksum = compute_record_checksum({"a": 1})
        assert len(checksum) == 64
        int(checksum, 16)  # raises if not valid hex


class TestComputeDatasetChecksum:
    def test_order_independent(self):
        checksums = ["aaa", "bbb", "ccc"]
        assert compute_dataset_checksum(checksums) == compute_dataset_checksum(list(reversed(checksums)))

    def test_different_record_sets_differ(self):
        assert compute_dataset_checksum(["aaa", "bbb"]) != compute_dataset_checksum(["aaa", "ccc"])

    def test_empty_list_is_stable(self):
        assert compute_dataset_checksum([]) == compute_dataset_checksum([])


class TestMakeDatasetVersion:
    def test_format(self):
        version = make_dataset_version(
            schema_version=1, source_split="test", template_family="synthetic_v1",
            generated_at=datetime(2026, 9, 2, 12, 30, 0, tzinfo=timezone.utc),
            checksum="abcdef1234567890",
        )
        assert version == "v1-test-synthetic_v1-20260902T123000Z-abcdef12"

    def test_uses_only_first_8_checksum_chars(self):
        version = make_dataset_version(1, "test", "x", datetime.now(timezone.utc), "a" * 64)
        assert version.endswith("-" + "a" * 8)
        assert not version.endswith("-" + "a" * 9)


class TestDatasetManifest:
    def test_lineage_defaults_to_empty_dict(self):
        m = DatasetManifest(
            dataset_version="v1-test-x-20260101T000000Z-aaaaaaaa",
            source_split="test", template_family="x", checksum="abc", record_count=10,
        )
        assert m.lineage == {}

    def test_lineage_independent_between_instances(self):
        m1 = DatasetManifest(
            dataset_version="v1", source_split="train", template_family="x",
            checksum="a", record_count=1,
        )
        m2 = DatasetManifest(
            dataset_version="v2", source_split="train", template_family="x",
            checksum="b", record_count=1,
        )
        m1.lineage["model_version"] = "v1"
        assert m2.lineage == {}

    def test_invalid_source_split_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DatasetManifest(
                dataset_version="v1", source_split="not_a_real_split",
                template_family="x", checksum="a", record_count=1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
