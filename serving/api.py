"""FastAPI REST API for SEC Filing Extraction.

Provides HTTP endpoints for single and batch extraction. Designed to work
in two modes:

1. **Standalone**: Loads model directly, handles inference internally.
   Simpler setup, good for development and testing.

2. **vLLM Proxy**: Forwards requests to a running vLLM server for
   production-grade throughput. The API layer handles prompt construction,
   post-processing, caching, and monitoring—vLLM handles raw inference.

Endpoints:
    POST /extract         - Single document extraction
    POST /extract/batch   - Batch extraction (up to 32 docs)
    GET  /health          - Health check + model status
    GET  /metrics          - Prometheus-compatible metrics

Usage:
    # Standalone mode
    uvicorn serving.api:app --host 0.0.0.0 --port 8000

    # With vLLM backend
    uvicorn serving.api:app --host 0.0.0.0 --port 8001
    # (vLLM server on port 8000)
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.inference import ExtractionEngine, ExtractionRequest, ExtractionResponse, SYSTEM_PROMPT, EXTRACTION_INSTRUCTION
from src.postprocessing import parse_extraction, validate_extraction


# ─── Request/Response Schemas ────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    """API request for single extraction."""
    text: str = Field(..., description="Raw SEC filing text", max_length=50000)
    filing_id: str | None = Field(None, description="Optional filing ID")
    max_tokens: int = Field(512, ge=64, le=2048)

    model_config = {"json_schema_extra": {
        "example": {
            "text": "SEC FILING - FORM 10-K\nRegistrant: Apple Inc. (AAPL)\n...",
            "filing_id": "000123-23-456",
        }
    }}


class ExtractBatchRequest(BaseModel):
    """API request for batch extraction."""
    documents: list[ExtractRequest] = Field(..., max_length=32)


class ExtractResponseModel(BaseModel):
    """API response for extraction."""
    status: str
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
    confidence_score: float = 0.0
    latency_ms: float = 0.0
    model_version: str = ""
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    gpu_memory_mb: float | None = None


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int


class WebhookRegistration(BaseModel):
    """Register a downstream service for extraction callbacks."""
    service: str = Field(..., description="Service name (e.g., 'ticker-agent')")
    url: str = Field(..., description="Webhook URL to receive callbacks")
    events: list[str] = Field(default=["extraction.complete"], description="Events to subscribe to")
    secret: str | None = Field(None, description="Optional secret for webhook signature verification")


class WebhookRegistrationResponse(BaseModel):
    """Response for webhook registration."""
    status: str
    service: str
    url: str
    events: list[str]
    registered_at: str


class PipelineStatusResponse(BaseModel):
    """Pipeline processing status."""
    stage: str
    extraction_count: int
    pending_enrichment: int
    pending_visualization: int
    last_processed_at: str | None = None
    services: dict[str, str]  # service name → status
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


# ─── App State ───────────────────────────────────────────────────────────────

class AppState:
    """Mutable application state for metrics tracking."""
    def __init__(self):
        self.engine: ExtractionEngine | None = None
        self.vllm_client: httpx.AsyncClient | None = None
        self.vllm_url: str | None = None
        self.start_time: float = time.time()
        self.latencies: list[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.config: dict = {}


state = AppState()


# ─── App Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model/connections on startup, cleanup on shutdown."""
    config = load_config()
    state.config = config

    # Check if vLLM server is available
    vllm_url = f"http://{config['serving']['host']}:{config['serving']['port']}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{vllm_url}/health", timeout=5)
            if resp.status_code == 200:
                state.vllm_url = vllm_url
                state.vllm_client = httpx.AsyncClient(timeout=60)
                logger.info(f"Connected to vLLM backend: {vllm_url}")
    except Exception:
        logger.info("No vLLM backend found, using local model")

    # If no vLLM, load model directly (lazy init on first request)
    if not state.vllm_url:
        state.engine = ExtractionEngine()
        logger.info("Extraction engine initialized (will load model on first request)")

    yield

    # Cleanup
    if state.vllm_client:
        await state.vllm_client.aclose()
    logger.info("Server shutdown")


def create_app(config: dict | None = None) -> FastAPI:
    """Create FastAPI app with configured lifespan."""
    app = FastAPI(
        title="Financial LLM Extraction API",
        description="Extract structured data from SEC filings using fine-tuned Llama 3.1",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routes
    app.add_api_route("/extract", extract_single, methods=["POST"], response_model=ExtractResponseModel)
    app.add_api_route("/extract/batch", extract_batch, methods=["POST"], response_model=list[ExtractResponseModel])
    app.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/metrics", get_metrics, methods=["GET"], response_model=MetricsResponse)
    app.add_api_route("/webhook/register", register_webhook, methods=["POST"], response_model=WebhookRegistrationResponse)
    app.add_api_route("/pipeline/status", pipeline_status, methods=["GET"], response_model=PipelineStatusResponse)

    return app


# ─── Endpoints ───────────────────────────────────────────────────────────────

async def extract_single(request: ExtractRequest) -> ExtractResponseModel:
    """Extract structured data from a single SEC filing.

    Routes to either vLLM backend or local model depending on availability.
    """
    start = time.time()

    try:
        if state.vllm_url and state.vllm_client:
            response = await _extract_via_vllm(request)
        elif state.engine:
            response = _extract_local(request)
        else:
            raise HTTPException(503, "No model backend available")

        latency = (time.time() - start) * 1000
        state.latencies.append(latency)
        state.success_count += 1

        return _to_response_model(response)

    except HTTPException:
        raise
    except Exception as e:
        state.error_count += 1
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")


async def extract_batch(request: ExtractBatchRequest) -> list[ExtractResponseModel]:
    """Extract from multiple filings in one request.

    Batch processing amortizes overhead and is the recommended approach
    for processing large volumes of filings.
    """
    if len(request.documents) > 32:
        raise HTTPException(400, "Maximum 32 documents per batch")

    results = []
    for doc in request.documents:
        try:
            result = await extract_single(doc)
            results.append(result)
        except HTTPException as e:
            results.append(ExtractResponseModel(
                status="error",
                error=str(e.detail),
                latency_ms=0,
            ))

    return results


async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring."""
    model_loaded = False
    model_version = "none"
    gpu_memory = None

    if state.vllm_url:
        model_loaded = True
        model_version = "vllm-backend"
    elif state.engine and state.engine._initialized:
        model_loaded = True
        model_version = state.engine.model.model_version
        mem_stats = state.engine.model.get_memory_stats()
        gpu_memory = mem_stats.get("allocated_mb")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=time.time() - state.start_time,
        gpu_memory_mb=gpu_memory,
    )


async def get_metrics() -> MetricsResponse:
    """Prometheus-compatible metrics endpoint."""
    latencies = sorted(state.latencies) if state.latencies else [0]
    n = len(latencies)

    return MetricsResponse(
        total_requests=state.success_count + state.error_count,
        successful_requests=state.success_count,
        failed_requests=state.error_count,
        avg_latency_ms=sum(latencies) / max(n, 1),
        p50_latency_ms=latencies[n // 2] if n else 0,
        p95_latency_ms=latencies[int(n * 0.95)] if n else 0,
        p99_latency_ms=latencies[int(n * 0.99)] if n else 0,
    )


# ─── Webhook & Pipeline Endpoints ───────────────────────────────────────────

# In-memory webhook registry (use Redis in production)
registered_webhooks: dict[str, WebhookRegistration] = {}


async def register_webhook(registration: WebhookRegistration) -> WebhookRegistrationResponse:
    """Register a downstream service to receive extraction callbacks.

    When extractions complete, registered webhooks will receive POST requests
    with the extraction results. Used for pipeline integration with TickerAgent.

    Example:
        curl -X POST http://localhost:8000/webhook/register \
          -H "Content-Type: application/json" \
          -d '{
            "service": "ticker-agent",
            "url": "http://localhost:8002/ingest",
            "events": ["extraction.complete"]
          }'
    """
    registered_webhooks[registration.service] = registration
    logger.info(f"Registered webhook for {registration.service}: {registration.url}")

    return WebhookRegistrationResponse(
        status="registered",
        service=registration.service,
        url=registration.url,
        events=registration.events,
        registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


async def pipeline_status() -> PipelineStatusResponse:
    """Get current pipeline processing status.

    Returns counts of extractions, pending enrichments, and service health.
    Used by the pipeline orchestrator to monitor flow between stages.
    """
    # Check downstream services
    services = {}

    # Try to connect to ticker-agent
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://localhost:8002/health")
            services["ticker-agent"] = "healthy" if resp.status_code == 200 else "unreachable"
    except Exception:
        services["ticker-agent"] = "offline"

    # Try to connect to viz-framework
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://localhost:8003/health")
            services["viz-framework"] = "healthy" if resp.status_code == 200 else "unreachable"
    except Exception:
        services["viz-framework"] = "offline"

    return PipelineStatusResponse(
        stage="extraction",
        extraction_count=state.success_count,
        pending_enrichment=0,  # Would query database in production
        pending_visualization=0,
        last_processed_at=None,
        services=services,
    )


# ─── Backend Routing ─────────────────────────────────────────────────────────

async def _extract_via_vllm(request: ExtractRequest) -> ExtractionResponse:
    """Send extraction request to vLLM OpenAI-compatible API.

    vLLM exposes /v1/completions which we call with our formatted prompt.
    Post-processing (JSON parsing, validation) happens here, not in vLLM.
    """
    # Build prompt matching training format
    prompt = _build_prompt_text(request.text)

    payload = {
        "model": state.config["model"]["base_model"],
        "prompt": prompt,
        "max_tokens": request.max_tokens,
        "temperature": 0,  # Greedy for deterministic extraction
        "stop": ["</s>", "<|eot_id|>"],
    }

    resp = await state.vllm_client.post(
        f"{state.vllm_url}/v1/completions",
        json=payload,
    )

    if resp.status_code != 200:
        raise HTTPException(502, f"vLLM error: {resp.text}")

    data = resp.json()
    raw_output = data["choices"][0]["text"]

    # Post-process
    try:
        extraction = parse_extraction(raw_output)
        is_valid, errors = validate_extraction(extraction)

        return ExtractionResponse(
            result=extraction,
            raw_output=raw_output,
            latency_ms=data.get("usage", {}).get("total_time_ms", 0),
            model_version="vllm",
            status="success" if is_valid else "validation_error",
            error="; ".join(errors) if errors else None,
            confidence_score=0.8 if is_valid else 0.4,
        )
    except Exception as e:
        return ExtractionResponse(
            result=None,
            raw_output=raw_output,
            latency_ms=0,
            model_version="vllm",
            status="parse_error",
            error=str(e),
        )


def _extract_local(request: ExtractRequest) -> ExtractionResponse:
    """Extract using locally loaded model."""
    req = ExtractionRequest(
        text=request.text,
        filing_id=request.filing_id,
        max_tokens=request.max_tokens,
    )
    return state.engine.extract(req)


def _build_prompt_text(filing_text: str) -> str:
    """Build extraction prompt as plain text (for vLLM /v1/completions)."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{EXTRACTION_INSTRUCTION}\n\n{filing_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _to_response_model(response: ExtractionResponse) -> ExtractResponseModel:
    """Convert internal response to API response model."""
    result = response.result
    return ExtractResponseModel(
        status=response.status,
        filing_id=result.filing_id if result else None,
        company_name=result.company_name if result else None,
        ticker=result.ticker if result else None,
        filing_type=result.filing_type if result else None,
        date=result.date if result else None,
        fiscal_year_end=result.fiscal_year_end if result else None,
        revenue=result.revenue if result else None,
        net_income=result.net_income if result else None,
        total_assets=result.total_assets if result else None,
        total_liabilities=result.total_liabilities if result else None,
        eps=result.eps if result else None,
        sector=result.sector if result else None,
        confidence_score=response.confidence_score,
        latency_ms=response.latency_ms,
        model_version=response.model_version,
        error=response.error,
    )


# ─── Default app instance ───────────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        "serving.api:app",
        host=config["serving"]["host"],
        port=config["serving"]["port"],
        reload=False,
    )
