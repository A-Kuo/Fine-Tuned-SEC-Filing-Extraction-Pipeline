"""Tests for the FastAPI extraction API.

Tests endpoint schemas, validation, health check, and metrics.
Uses FastAPI's TestClient so no running server is needed.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.api import (
    ExtractRequest,
    ExtractBatchRequest,
    ExtractResponseModel,
    HealthResponse,
    StatsResponse,
)


# ─── Request Schema Tests ───────────────────────────────────────────────────

class TestRequestSchemas:
    def test_extract_request_valid(self):
        req = ExtractRequest(text="SEC FILING...", filing_id="123")
        assert req.text == "SEC FILING..."
        assert req.filing_id == "123"
        assert req.max_tokens == 512  # default

    def test_extract_request_defaults(self):
        req = ExtractRequest(text="test")
        assert req.filing_id is None
        assert req.max_tokens == 512

    def test_extract_request_custom_tokens(self):
        req = ExtractRequest(text="test", max_tokens=1024)
        assert req.max_tokens == 1024

    def test_batch_request_valid(self):
        docs = [ExtractRequest(text=f"doc {i}") for i in range(5)]
        batch = ExtractBatchRequest(documents=docs)
        assert len(batch.documents) == 5


# ─── Response Schema Tests ──────────────────────────────────────────────────

class TestResponseSchemas:
    def test_extract_response_success(self):
        resp = ExtractResponseModel(
            status="success",
            company_name="Apple Inc.",
            filing_type="10-K",
            confidence_score=0.96,
            latency_ms=421,
            model_version="v1",
        )
        assert resp.status == "success"
        assert resp.company_name == "Apple Inc."
        assert resp.error is None

    def test_extract_response_error(self):
        resp = ExtractResponseModel(
            status="error",
            error="Model timeout",
            latency_ms=30000,
        )
        assert resp.status == "error"
        assert resp.error == "Model timeout"
        assert resp.company_name is None

    def test_health_response(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="llama-sec-v1",
            uptime_seconds=3600,
            gpu_memory_mb=7200,
        )
        assert resp.status == "healthy"

    def test_metrics_response(self):
        resp = StatsResponse(
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            avg_latency_ms=420,
            p50_latency_ms=320,
            p95_latency_ms=480,
            p99_latency_ms=920,
        )
        assert resp.total_requests == 1000
        assert resp.p99_latency_ms == 920


# ─── Prompt Building Tests ──────────────────────────────────────────────────

class TestPromptBuilding:
    def test_build_prompt_contains_system_and_user(self):
        from serving.api import _build_prompt_text
        prompt = _build_prompt_text("Apple Inc. 10-K filing...")
        assert "system" in prompt
        assert "Apple Inc." in prompt
        assert "assistant" in prompt

    def test_build_prompt_has_extraction_instruction(self):
        from serving.api import _build_prompt_text
        prompt = _build_prompt_text("test filing")
        assert "Extract structured financial data" in prompt

    def test_build_prompt_has_special_tokens(self):
        from serving.api import _build_prompt_text
        prompt = _build_prompt_text("test")
        # Llama 3.1 chat format tokens
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>" in prompt
        assert "<|eot_id|>" in prompt


# ─── Response Conversion Tests ──────────────────────────────────────────────

class TestResponseConversion:
    def test_to_response_model_success(self):
        from serving.api import _to_response_model
        from src.inference import ExtractionResponse
        from src.postprocessing import ExtractionResult

        result = ExtractionResult(
            filing_id="123",
            company_name="Apple",
            filing_type="10-K",
        )
        response = ExtractionResponse(
            result=result,
            raw_output='{"company_name": "Apple"}',
            latency_ms=421,
            model_version="v1",
            status="success",
            confidence_score=0.96,
        )

        model = _to_response_model(response)
        assert model.status == "success"
        assert model.company_name == "Apple"
        assert model.confidence_score == 0.96

    def test_to_response_model_error(self):
        from serving.api import _to_response_model
        from src.inference import ExtractionResponse

        response = ExtractionResponse(
            result=None,
            raw_output="",
            latency_ms=100,
            model_version="v1",
            status="error",
            error="Parse failed",
        )

        model = _to_response_model(response)
        assert model.status == "error"
        assert model.company_name is None
        assert model.error == "Parse failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
