"""FastAPI REST API for SEC Filing Extraction.

Endpoints:
    POST /extract                 - Single document extraction
    POST /extract/batch           - Batch extraction (up to 32 docs)
    GET  /health                  - Health check + model status
    GET  /metrics                 - Prometheus text exposition format
    GET  /stats                   - JSON request/latency statistics
    GET  /extractions/{filing_id} - Retrieve stored extraction by ID
    POST /webhook/register        - Register downstream webhook
    GET  /webhook/verify          - Webhook integration probe
    GET  /webhook/failures        - Inspect dead-letter queue
    GET  /pipeline/status         - Pipeline + downstream health
    GET  /ab/results              - A/B test summary (when enabled)
    POST /ab/promote              - Promote challenger model to primary
    POST /webhook/alertmanager    - Receive Alertmanager callbacks
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Redis-backed storage for multi-instance rate limiting (falls back to in-memory)
def _make_limiter(redis_url: str | None = None) -> Limiter:
    if redis_url:
        try:
            from limits.storage import RedisStorage
            storage = RedisStorage(redis_url)
            return Limiter(key_func=get_remote_address, storage_uri=redis_url)
        except Exception:
            pass
    return Limiter(key_func=get_remote_address)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_router import assign_for_request
from src.circuit_breaker import CircuitBreaker, check_http_health
from src.config import load_config
from src.database import DatabaseManager
from src.inference import (
    ExtractionEngine,
    ExtractionRequest,
    ExtractionResponse,
    EXTRACTION_INSTRUCTION,
    SYSTEM_PROMPT,
)
from src.logging_config import configure_logging, set_request_id
from src.postprocessing import parse_extraction, validate_extraction
from serving.security import assert_api_key_if_configured

# ─── Prometheus metrics ─────────────────────────────────────────────────────
EXTRACTION_TOTAL = Counter(
    "findoc_extraction_total",
    "Extractions by status and filing type",
    ["status", "filing_type"],
)
EXTRACTION_DURATION = Histogram(
    "findoc_extraction_duration_seconds",
    "Extraction latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
WEBHOOK_DISPATCH_TOTAL = Counter(
    "findoc_webhook_dispatch_total",
    "Webhook delivery attempts",
    ["service", "status"],
)
MODEL_MEMORY = Gauge("findoc_model_memory_bytes", "GPU memory used by model if available")
CIRCUIT_SKIP_TOTAL = Counter(
    "findoc_circuit_skip_total",
    "Enrichment dispatches skipped because circuit is open",
    ["service"],
)

# ─── Request/Response Schemas ────────────────────────────────────────────────


class ExtractRequest(BaseModel):
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
    documents: list[ExtractRequest] = Field(..., max_length=32)


class ExtractResponseModel(BaseModel):
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
    ab_variant: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    gpu_memory_mb: float | None = None


class StatsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class WebhookRegistration(BaseModel):
    service: str = Field(..., description="Service name (e.g., 'ticker-agent')")
    url: str = Field(..., description="Webhook URL to receive callbacks")
    events: list[str] = Field(default=["extraction.complete"], description="Events to subscribe to")
    secret: str | None = Field(None, description="Optional secret for webhook signature verification")


class WebhookRegistrationResponse(BaseModel):
    status: str
    service: str
    url: str
    events: list[str]
    registered_at: str


class PipelineStatusResponse(BaseModel):
    stage: str
    extraction_count: int
    pending_enrichment: int
    pending_visualization: int
    last_processed_at: str | None = None
    services: dict[str, str]
    enrichment_skipped: bool = False
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class ABPromoteRequest(BaseModel):
    promote_challenger: bool = True


# ─── App State ───────────────────────────────────────────────────────────────

registered_webhooks: dict[str, WebhookRegistration] = {}
# Initialised with in-memory storage; replaced with Redis storage in lifespan
limiter = _make_limiter()


class AppState:
    def __init__(self):
        self.engine: ExtractionEngine | None = None
        self.vllm_client: httpx.AsyncClient | None = None
        self.vllm_url: str | None = None
        self.start_time: float = time.time()
        self.latencies: collections.deque[float] = collections.deque(maxlen=10000)
        self.success_count: int = 0
        self.error_count: int = 0
        self.config: dict = {}
        self.db: DatabaseManager | None = None
        self.circuit_ticker: CircuitBreaker = CircuitBreaker("ticker-agent")
        self.circuit_viz: CircuitBreaker = CircuitBreaker("viz-framework")


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global limiter
    config = load_config()
    state.config = config

    log_cfg = config.get("logging", {})
    configure_logging(
        level=log_cfg.get("level", "INFO"),
        fmt=log_cfg.get("format", "text"),
        include_request_id=log_cfg.get("include_request_id", True),
    )

    # Try to upgrade rate limiter to Redis for multi-instance deployments
    db_cfg = config.get("database", {}).get("redis", {})
    redis_host = db_cfg.get("host", "localhost")
    redis_port = db_cfg.get("port", 6379)
    redis_db = db_cfg.get("db", 0)
    redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
    new_limiter = _make_limiter(redis_url)
    limiter = new_limiter
    app.state.limiter = limiter
    logger.info(f"Rate limiter storage: {redis_url if new_limiter is not limiter else 'in-memory'}")

    try:
        state.db = DatabaseManager.from_config(config)
    except Exception as e:
        logger.warning(f"Database not available: {e}")
        state.db = None

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

    if not state.vllm_url:
        state.engine = ExtractionEngine()
        logger.info("Extraction engine initialized (lazy model load)")

    yield

    if state.vllm_client:
        await state.vllm_client.aclose()
    if state.db:
        state.db.close()
    logger.info("Server shutdown")


def create_app(config: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="Financial LLM Extraction API",
        description="Extract structured data from SEC filings using fine-tuned Llama 3.1",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    cfg = config or load_config()
    origins = cfg.get("security", {}).get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.time()
        try:
            assert_api_key_if_configured(request, state.config or None)
        except HTTPException as e:
            return Response(
                content=json.dumps({"detail": e.detail}),
                status_code=e.status_code,
                media_type="application/json",
            )
        response = await call_next(request)
        dur_ms = (time.time() - start) * 1000
        logger.info(f"{request.method} {request.url.path} {response.status_code} {dur_ms:.1f}ms")
        return response

    rate_limit_str = f"{(cfg.get('security') or {}).get('rate_limit_per_minute', 100)}/minute"

    @app.post("/extract", response_model=ExtractResponseModel)
    @limiter.limit(rate_limit_str)
    async def extract_single_route(
        request: Request,
        background_tasks: BackgroundTasks,
        req: ExtractRequest,
    ) -> ExtractResponseModel:
        return await run_extraction(req, background_tasks)

    @app.post("/extract/batch", response_model=list[ExtractResponseModel])
    async def extract_batch_route(
        req: ExtractBatchRequest,
        background_tasks: BackgroundTasks,
    ) -> list[ExtractResponseModel]:
        if len(req.documents) > 32:
            raise HTTPException(400, "Maximum 32 documents per batch")
        out: list[ExtractResponseModel] = []
        for doc in req.documents:
            try:
                m = await run_extraction(doc, background_tasks)
                out.append(m)
            except HTTPException as e:
                out.append(ExtractResponseModel(status="error", error=str(e.detail), latency_ms=0.0))
        return out

    app.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/metrics", prometheus_metrics, methods=["GET"])
    app.add_api_route("/stats", stats_json, methods=["GET"], response_model=StatsResponse)
    app.add_api_route("/webhook/register", register_webhook, methods=["POST"], response_model=WebhookRegistrationResponse)
    app.add_api_route("/webhook/verify", webhook_verify, methods=["GET"])
    app.add_api_route("/webhook/failures", webhook_failures, methods=["GET"])
    app.add_api_route("/pipeline/status", pipeline_status, methods=["GET"], response_model=PipelineStatusResponse)
    app.add_api_route("/ab/results", ab_results, methods=["GET"])
    app.add_api_route("/ab/promote", ab_promote, methods=["POST"])
    app.add_api_route("/webhook/alertmanager", alertmanager_receiver, methods=["POST"])
    app.add_api_route("/extractions/{filing_id}", get_extraction, methods=["GET"])

    return app


# ─── Endpoints ───────────────────────────────────────────────────────────────


async def run_extraction(req: ExtractRequest, background_tasks: BackgroundTasks) -> ExtractResponseModel:
    start = time.time()
    try:
        if state.vllm_url and state.vllm_client:
            response = await _extract_via_vllm(req)
        elif state.engine:
            response = _extract_local(req)
        else:
            raise HTTPException(503, "No model backend available")

        latency = (time.time() - start) * 1000
        state.latencies.append(latency)
        state.success_count += 1
        EXTRACTION_DURATION.observe(latency / 1000.0)
        ft = response.result.filing_type if response.result else "unknown"
        EXTRACTION_TOTAL.labels(status=response.status, filing_type=ft or "unknown").inc()

        model = _to_response_model(response)
        model.ab_variant = _maybe_record_ab(req, response, latency)
        _maybe_persist(req, response, latency)
        background_tasks.add_task(_dispatch_webhooks_background, model)
        return model

    except HTTPException:
        raise
    except Exception as e:
        state.error_count += 1
        EXTRACTION_TOTAL.labels(status="error", filing_type="unknown").inc()
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")


async def health_check() -> HealthResponse:
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
        MODEL_MEMORY.set(mem_stats.get("allocated_mb", 0) * 1e6)

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=time.time() - state.start_time,
        gpu_memory_mb=gpu_memory,
    )


async def prometheus_metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


async def stats_json() -> StatsResponse:
    latencies = sorted(state.latencies) if state.latencies else [0.0]
    n = len(latencies)
    return StatsResponse(
        total_requests=state.success_count + state.error_count,
        successful_requests=state.success_count,
        failed_requests=state.error_count,
        avg_latency_ms=sum(latencies) / max(n, 1),
        p50_latency_ms=latencies[n // 2] if n else 0.0,
        p95_latency_ms=latencies[int(n * 0.95)] if n else 0.0,
        p99_latency_ms=latencies[int(n * 0.99)] if n else 0.0,
    )


async def register_webhook(registration: WebhookRegistration) -> WebhookRegistrationResponse:
    registered_webhooks[registration.service] = registration
    logger.info(f"Registered webhook for {registration.service}: {registration.url}")
    return WebhookRegistrationResponse(
        status="registered",
        service=registration.service,
        url=registration.url,
        events=registration.events,
        registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


async def webhook_verify() -> dict[str, str]:
    return {"status": "ok", "message": "Webhook endpoint ready for FinDocAnalyzer pipeline integration"}


async def webhook_failures(limit: int = 50) -> dict[str, Any]:
    """Return recent failed webhook deliveries from the dead-letter queue."""
    if not state.db:
        return {"failures": [], "message": "Database unavailable"}
    failures = state.db.get_webhook_failures(limit=min(limit, 200))
    return {"count": len(failures), "failures": failures}


async def get_extraction(filing_id: str) -> dict[str, Any]:
    """Look up a previously stored extraction result by filing ID."""
    if not state.db:
        raise HTTPException(503, "Database unavailable")
    result = state.db.get_extraction(filing_id)
    if result is None:
        raise HTTPException(404, f"Extraction not found for filing_id={filing_id!r}")
    return result


async def alertmanager_receiver(request: Request) -> dict[str, Any]:
    """Receive firing/resolved alerts from Prometheus Alertmanager."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    alerts = body.get("alerts", [])
    firing = [a for a in alerts if a.get("status") == "firing"]
    resolved = [a for a in alerts if a.get("status") == "resolved"]
    for a in firing:
        name = a.get("labels", {}).get("alertname", "unknown")
        severity = a.get("labels", {}).get("severity", "warning")
        summary = a.get("annotations", {}).get("summary", "")
        logger.warning(f"[AM firing] {name} severity={severity}: {summary}")
    for a in resolved:
        name = a.get("labels", {}).get("alertname", "unknown")
        logger.info(f"[AM resolved] {name}")
    return {"status": "ok", "firing": len(firing), "resolved": len(resolved)}


async def pipeline_status() -> PipelineStatusResponse:
    cfg = state.config or load_config()
    ticker_url = (cfg.get("serving") or {}).get("ticker_agent_url", "http://localhost:8002")
    viz_url = (cfg.get("serving") or {}).get("viz_framework_url", "http://localhost:8003")

    services: dict[str, str] = {}
    ticker_ok = await asyncio.to_thread(check_http_health, f"{ticker_url}/health")
    if ticker_ok:
        state.circuit_ticker.record_success()
        services["ticker-agent"] = "healthy"
    else:
        state.circuit_ticker.record_failure()
        services["ticker-agent"] = f"unreachable({state.circuit_ticker.state.value})"

    viz_ok = await asyncio.to_thread(check_http_health, f"{viz_url}/health")
    if viz_ok:
        state.circuit_viz.record_success()
        services["viz-framework"] = "healthy"
    else:
        state.circuit_viz.record_failure()
        services["viz-framework"] = f"unreachable({state.circuit_viz.state.value})"

    pending_e = pending_v = 0
    last_ts: str | None = None
    if state.db:
        counts = state.db.get_pipeline_stage_counts()
        pending_e = counts.get("extracted", 0)
        pending_v = counts.get("enriched", 0)
        # optional: query max updated_at — omitted for brevity

    latencies = sorted(state.latencies) if state.latencies else [0.0]
    n = len(latencies)

    enrichment_skipped = (
        state.circuit_ticker.state.value != "closed"
    )

    return PipelineStatusResponse(
        stage="extraction",
        extraction_count=state.success_count,
        pending_enrichment=pending_e,
        pending_visualization=pending_v,
        last_processed_at=last_ts,
        services=services,
        enrichment_skipped=enrichment_skipped,
        failed_requests=state.error_count,
        avg_latency_ms=sum(latencies) / max(n, 1),
        p50_latency_ms=latencies[n // 2] if n else 0.0,
        p95_latency_ms=latencies[int(n * 0.95)] if n else 0.0,
        p99_latency_ms=latencies[int(n * 0.99)] if n else 0.0,
    )


async def ab_results() -> dict[str, Any]:
    if not state.db:
        return {"models": [], "message": "Database unavailable"}
    return {"models": state.db.get_ab_summary()}


async def ab_promote(body: ABPromoteRequest) -> dict[str, str]:
    logger.info(f"A/B promote requested: challenger={body.promote_challenger}")
    return {"status": "accepted", "detail": "Update adapter_path in config and redeploy to promote"}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _maybe_record_ab(req: ExtractRequest, response: ExtractionResponse, latency_ms: float) -> str | None:
    """Record A/B assignment; returns the variant label ('primary' or 'challenger')."""
    cfg = state.config
    ab = assign_for_request(req.filing_id, None, cfg)
    if state.db:
        fid = req.filing_id or "unknown"
        state.db.record_ab_result(
            fid,
            ab.model_version_label,
            ab.use_challenger,
            response.confidence_score,
            response.status,
            int(latency_ms),
        )
    return ab.model_version_label


def _maybe_persist(req: ExtractRequest, response: ExtractionResponse, latency_ms: float) -> None:
    if not state.db or not response.result:
        return
    try:
        state.db.store_extraction(
            req.filing_id or response.result.filing_id or "unknown",
            response.result,
            response.confidence_score,
            latency_ms,
            response.model_version,
            response.raw_output,
            status=response.status,
            error=response.error,
        )
        state.db.upsert_pipeline_stage(
            req.filing_id or response.result.filing_id or "unknown",
            "extracted",
            response.result.ticker,
        )
    except Exception as e:
        logger.warning(f"Persist skipped: {e}")


def _circuit_for_service(service: str) -> CircuitBreaker | None:
    """Return the circuit breaker that guards a registered webhook service."""
    svc_lower = service.lower()
    if "ticker" in svc_lower:
        return state.circuit_ticker
    if "viz" in svc_lower:
        return state.circuit_viz
    return None


async def _dispatch_webhooks_background(resp: ExtractResponseModel) -> None:
    if resp.status not in ("success", "validation_error"):
        return
    payload = {
        "event": "extraction.complete",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": resp.model_dump(),
    }
    body = json.dumps(payload, sort_keys=True)
    cfg = state.config or load_config()
    default_secret = (cfg.get("security") or {}).get("webhook_signing_secret") or ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        for service, reg in registered_webhooks.items():
            if "extraction.complete" not in reg.events:
                continue

            # Circuit breaker guard: skip enrichment when downstream is known-bad
            cb = _circuit_for_service(service)
            if cb and not cb.allow():
                cached = cb.degraded_response()
                logger.warning(
                    f"[circuit-open] Skipping webhook to {service} "
                    f"(state={cb.state.value}). "
                    f"Cached response available: {cached['cached'] is not None}"
                )
                WEBHOOK_DISPATCH_TOTAL.labels(service=service, status="circuit_open").inc()
                CIRCUIT_SKIP_TOTAL.labels(service=service).inc()
                continue

            secret = reg.secret or default_secret
            headers = {"Content-Type": "application/json"}
            if secret:
                sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={sig}"

            delay = 1.0
            succeeded = False
            for attempt in range(3):
                try:
                    r = await client.post(reg.url, content=body, headers=headers)
                    if 200 <= r.status_code < 300:
                        WEBHOOK_DISPATCH_TOTAL.labels(service=service, status="success").inc()
                        if cb:
                            cb.record_success(cached_response=resp.model_dump())
                        succeeded = True
                        break
                    raise RuntimeError(f"HTTP {r.status_code}")
                except Exception as e:
                    WEBHOOK_DISPATCH_TOTAL.labels(
                        service=service,
                        status="retry" if attempt < 2 else "failed",
                    ).inc()
                    if attempt == 2:
                        if state.db:
                            state.db.log_webhook_failure(service, reg.url, payload, str(e), attempt + 1)
                        if cb:
                            cb.record_failure()
                    await asyncio.sleep(delay)
                    delay *= 5

            if not succeeded and cb:
                logger.info(
                    f"Circuit breaker '{service}' state after failures: {cb.state.value}"
                )


async def _extract_via_vllm(request: ExtractRequest) -> ExtractionResponse:
    prompt = _build_prompt_text(request.text)
    payload = {
        "model": state.config["model"]["base_model"],
        "prompt": prompt,
        "max_tokens": request.max_tokens,
        "temperature": 0,
        "stop": ["</s>", "<|eot_id|>"],
    }
    assert state.vllm_client is not None
    resp = await state.vllm_client.post(f"{state.vllm_url}/v1/completions", json=payload)
    if resp.status_code != 200:
        raise HTTPException(502, f"vLLM error: {resp.text}")
    data = resp.json()
    raw_output = data["choices"][0]["text"]
    try:
        extraction = parse_extraction(raw_output)
        is_valid, errors = validate_extraction(extraction)
        return ExtractionResponse(
            result=extraction,
            raw_output=raw_output,
            latency_ms=float(data.get("usage", {}).get("total_time_ms", 0)),
            model_version="vllm",
            status="success" if is_valid else "validation_error",
            error="; ".join(errors) if errors else None,
            confidence_score=0.8 if is_valid else 0.4,
        )
    except Exception as e:
        return ExtractionResponse(
            result=None,
            raw_output=raw_output,
            latency_ms=0.0,
            model_version="vllm",
            status="parse_error",
            error=str(e),
        )


def _extract_local(request: ExtractRequest) -> ExtractionResponse:
    assert state.engine is not None
    req = ExtractionRequest(
        text=request.text,
        filing_id=request.filing_id,
        max_tokens=request.max_tokens,
    )
    return state.engine.extract(req)


def _build_prompt_text(filing_text: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{EXTRACTION_INSTRUCTION}\n\n{filing_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _to_response_model(response: ExtractionResponse) -> ExtractResponseModel:
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
