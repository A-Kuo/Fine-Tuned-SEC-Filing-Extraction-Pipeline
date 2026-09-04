"""Tests for serving/api.py's idempotency handling: the text-hash fallback
cache key and the in-flight dedup lock.

Before this, a request with no filing_id had NO idempotency key at all --
every such request always ran full extraction, even for identical text
already processed. And even WITH a filing_id, two concurrent requests
arriving before either one's result was persisted would both reach the
model. Real async tests, not mocked-shape-only ones -- this is exactly the
gap tests/test_api.py's misleading "Uses FastAPI's TestClient" docstring
(it doesn't) left uncovered.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import serving.api as api_module
from serving.api import ExtractRequest, _lookup_cached, _text_hash
from src.extraction.postprocessing import ExtractionResult


class TestTextHash:
    def test_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_text_different_hash(self):
        assert _text_hash("hello") != _text_hash("world")

    def test_returns_hex_sha256(self):
        h = _text_hash("hello")
        assert len(h) == 64
        int(h, 16)


class TestLookupCached:
    def test_no_db_returns_none(self, monkeypatch):
        monkeypatch.setattr(api_module.state, "db", None)
        req = ExtractRequest(text="some filing text", filing_id="f-1")
        assert _lookup_cached(req, _text_hash(req.text)) is None

    def test_hits_by_filing_id_first(self, monkeypatch):
        mock_db = MagicMock()
        mock_db.get_extraction.return_value = {"filing_id": "f-1", "company_name": "Acme"}
        monkeypatch.setattr(api_module.state, "db", mock_db)

        req = ExtractRequest(text="text", filing_id="f-1")
        result = _lookup_cached(req, _text_hash(req.text))

        assert result is not None
        assert result.company_name == "Acme"
        assert result.cache_hit is True
        mock_db.get_extraction.assert_called_once_with("f-1")
        mock_db.get_extraction_by_text_hash.assert_not_called()

    def test_falls_back_to_text_hash_when_no_filing_id(self, monkeypatch):
        mock_db = MagicMock()
        mock_db.get_extraction_by_text_hash.return_value = {"filing_id": "recovered-id", "company_name": "Acme"}
        monkeypatch.setattr(api_module.state, "db", mock_db)

        req = ExtractRequest(text="text with no filing_id")
        result = _lookup_cached(req, _text_hash(req.text))

        assert result is not None
        assert result.filing_id == "recovered-id"
        mock_db.get_extraction.assert_not_called()

    def test_falls_back_to_text_hash_when_filing_id_lookup_misses(self, monkeypatch):
        mock_db = MagicMock()
        mock_db.get_extraction.return_value = None
        mock_db.get_extraction_by_text_hash.return_value = {"filing_id": "f-1", "company_name": "Acme"}
        monkeypatch.setattr(api_module.state, "db", mock_db)

        req = ExtractRequest(text="text", filing_id="f-1")
        result = _lookup_cached(req, _text_hash(req.text))

        assert result is not None
        mock_db.get_extraction_by_text_hash.assert_called_once()

    def test_no_cache_entry_returns_none(self, monkeypatch):
        mock_db = MagicMock()
        mock_db.get_extraction.return_value = None
        mock_db.get_extraction_by_text_hash.return_value = None
        monkeypatch.setattr(api_module.state, "db", mock_db)

        req = ExtractRequest(text="text", filing_id="f-1")
        assert _lookup_cached(req, _text_hash(req.text)) is None


class TestInFlightDedup:
    """The core guarantee: two concurrent requests for the same uncached
    text (no filing_id) must result in exactly ONE call to the model, not
    two -- the second should wait on the lock and then reuse whatever the
    first one persisted."""

    @pytest.fixture(autouse=True)
    def _reset_state(self, monkeypatch):
        monkeypatch.setattr(api_module.state, "inflight_locks", {})
        monkeypatch.setattr(api_module.state, "vllm_url", None)
        monkeypatch.setattr(api_module.state, "success_count", 0)
        monkeypatch.setattr(api_module.state, "latencies", __import__("collections").deque(maxlen=10000))

    @pytest.mark.asyncio
    async def test_concurrent_identical_requests_call_model_once(self, monkeypatch):
        # run_extraction calls _extract_local(req) synchronously (no await)
        # when there's no vLLM backend. asyncio.Lock still forces the second
        # gather()'d coroutine to actually suspend at `async with lock`, so
        # this exercises real interleaving, not just sequential calls.
        stored: dict = {}
        call_state = {"count": 0}

        def sync_slow_extract(req):
            call_state["count"] += 1
            from serving.api import ExtractionResponse
            return ExtractionResponse(
                result=ExtractionResult(company_name="Acme", filing_type="10-K"),
                raw_output="{}", latency_ms=50.0, model_version="test-v1", status="success",
                confidence_score=0.9,
            )

        mock_db = MagicMock()
        mock_db.get_extraction.return_value = None
        mock_db.get_extraction_by_text_hash.side_effect = lambda h: stored.get(h)

        def fake_store(filing_id, result, confidence, latency_ms, model_version, raw_output, **kwargs):
            text_hash = kwargs.get("text_hash")
            if text_hash:
                stored[text_hash] = {"filing_id": filing_id, "company_name": result.company_name}
            return True

        mock_db.store_extraction.side_effect = fake_store
        mock_db.upsert_pipeline_stage.return_value = True

        monkeypatch.setattr(api_module.state, "db", mock_db)
        monkeypatch.setattr(api_module, "_extract_local", sync_slow_extract)
        monkeypatch.setattr(api_module.state, "engine", object())  # truthy, so the engine branch is taken

        req = ExtractRequest(text="identical filing text, no filing_id")
        bg1 = MagicMock()
        bg2 = MagicMock()

        # Two "concurrent" calls for the exact same text.
        results = await asyncio.gather(
            api_module.run_extraction(req, bg1),
            api_module.run_extraction(req, bg2),
        )

        assert call_state["count"] == 1, "the model should only be invoked once for two identical concurrent requests"
        assert all(r.company_name == "Acme" for r in results)

    @pytest.mark.asyncio
    async def test_lock_entry_cleaned_up_after_completion(self, monkeypatch):
        mock_db = MagicMock()
        mock_db.get_extraction.return_value = None
        mock_db.get_extraction_by_text_hash.return_value = None
        mock_db.store_extraction.return_value = True
        mock_db.upsert_pipeline_stage.return_value = True
        monkeypatch.setattr(api_module.state, "db", mock_db)

        def sync_extract(req):
            from serving.api import ExtractionResponse
            return ExtractionResponse(
                result=ExtractionResult(company_name="Acme", filing_type="10-K"),
                raw_output="{}", latency_ms=10.0, model_version="test-v1", status="success",
                confidence_score=0.9,
            )

        monkeypatch.setattr(api_module, "_extract_local", sync_extract)
        monkeypatch.setattr(api_module.state, "engine", object())

        req = ExtractRequest(text="some unique text for cleanup test")
        await api_module.run_extraction(req, MagicMock())

        assert len(api_module.state.inflight_locks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
