"""Tests for serving/api.py's /rag/query route (rag_query_route).

Follows tests/test_idempotency.py's real convention for testing this
module: import the module and monkeypatch its `state` singleton directly,
rather than the aspirational (and incorrect) "uses TestClient" docstring
in tests/test_api.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent))

import serving.api as api_module
from serving.api import RagQueryRequest, rag_query_route


class TestRagQueryRoute:
    @pytest.mark.asyncio
    async def test_503_when_rag_storage_unavailable(self, monkeypatch):
        monkeypatch.setattr(api_module.state, "rag_storage", None)
        with pytest.raises(HTTPException) as exc_info:
            await rag_query_route(RagQueryRequest(question="What was revenue?"))
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_503_in_vllm_offload_mode(self, monkeypatch):
        """No local FinancialLLM exists to call .generate() on in vLLM mode
        -- must fail with a clear error, not crash on a None engine."""
        monkeypatch.setattr(api_module.state, "rag_storage", MagicMock())
        monkeypatch.setattr(api_module.state, "vllm_url", "http://vllm:8000")
        with pytest.raises(HTTPException) as exc_info:
            await rag_query_route(RagQueryRequest(question="What was revenue?"))
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_happy_path_returns_answer_and_sources(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.initialize = MagicMock()
        mock_engine.model.generate = MagicMock(return_value=("real answer", 5.0))

        monkeypatch.setattr(api_module.state, "rag_storage", MagicMock(_available=True))
        monkeypatch.setattr(api_module.state, "vllm_url", None)
        monkeypatch.setattr(api_module.state, "engine", mock_engine)
        monkeypatch.setattr(api_module.state, "rag_embedder", None)
        monkeypatch.setattr(api_module.state, "config", {"rag": {"top_k": 3, "max_tokens": 256}})

        fake_embedder = MagicMock()
        fake_embedder.embed = MagicMock(return_value=[[0.1, 0.2]])
        monkeypatch.setattr(api_module, "Embedder", MagicMock(return_value=fake_embedder))

        fake_retrieved = [{"section_id": 1, "filing_id": "AAPL-123", "section_type": "mdna", "content": "real prose"}]
        monkeypatch.setattr(api_module, "retrieve", MagicMock(return_value=fake_retrieved))

        response = await rag_query_route(RagQueryRequest(question="What was revenue?"))

        assert response.answer == "real answer"
        assert response.sources == ["AAPL-123"]
        mock_engine.initialize.assert_called_once()
