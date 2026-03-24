# Pipeline API Contracts

This document defines the API contracts between the three pipeline services.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SERVICE INTERCONNECTIONS                       │
└─────────────────────────────────────────────────────────────────────────┘

FinDocAnalyzer (Port 8000)
├── POST /extract           → Immediate extraction
├── POST /extract/batch     → Batch with callback
├── POST /webhook/register  → Register downstream URL
├── GET  /health            → Health check
└── GET  /pipeline/status   → Current processing status
    │
    │ Webhook callback OR Database poll
    ▼
TickerAgent (Port 8002)
├── POST /ingest            ← Receive extracted ticker
├── GET  /enrich/{ticker}   → Get market data
├── POST /batch-enrich      → Batch enrichment
├── GET  /health
└── GET  /market/{ticker}   → Real-time market snapshot
    │
    │ Webhook callback OR Database poll
    ▼
VizFramework (Port 8003)
├── POST /render            ← Receive enriched data
├── GET  /dashboard/{id}    → View dashboard
├── POST /export            → Export visualization
├── GET  /health
└── WebSocket /live         → Real-time updates
```

---

## FinDocAnalyzer → TickerAgent

### Method 1: Webhook (Event-Driven)

**Registration (One-time setup):**
```bash
curl -X POST http://findoc-analyzer:8000/webhook/register \
  -H "Content-Type: application/json" \
  -d '{
    "service": "ticker-agent",
    "url": "http://ticker-agent:8002/ingest",
    "events": ["extraction.complete"],
    "secret": "webhook-secret-key"
  }'
```

**Webhook Payload (Stage 1 → Stage 2):**
```json
{
  "event": "extraction.complete",
  "timestamp": "2025-03-18T14:30:00Z",
  "data": {
    "extraction_id": "550e8400-e29b-41d4-a716-446655440000",
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "financials": {
      "revenue": 394328000000,
      "net_income": 99803000000,
      "eps": 6.13
    },
    "metadata": {
      "confidence_score": 0.94,
      "extracted_at": "2025-03-18T14:30:00Z",
      "model_version": "llama-sec-v1-20250318"
    }
  }
}
```

### Method 2: Polling (Database-Linked)

TickerAgent polls PostgreSQL for new extractions:

```sql
-- Query for unprocessed tickers (run every 30 seconds)
SELECT 
  e.extraction_id,
  e.ticker,
  e.company_name,
  e.revenue,
  e.net_income,
  e.filing_date,
  e.confidence_score
FROM extractions e
LEFT JOIN pipeline_processing p ON e.extraction_id = p.extraction_id
WHERE p.status IS NULL 
   OR p.status = 'pending'
   AND e.confidence_score > 0.8
ORDER BY e.created_at ASC
LIMIT 10;
```

---

## TickerAgent → VizFramework

### Enriched Data Schema (Stage 2 → Stage 3)

```json
{
  "extraction_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "financials": {
    "revenue": 394328000000,
    "net_income": 99803000000,
    "eps": 6.13,
    "currency": "USD",
    "units": "millions"
  },
  "market_context": {
    "current_price": 189.52,
    "price_change_24h": 1.25,
    "volume": 45000000,
    "market_cap": 2900000000000,
    "pe_ratio": 31.2,
    "52_week_high": 199.62,
    "52_week_low": 164.08,
    "sentiment_score": 0.72,
    "analyst_rating": "Buy",
    "enriched_at": "2025-03-18T14:30:05Z"
  },
  "viz_config": {
    "chart_type": "financial_summary",
    "time_range": "1Y",
    "include_peers": true,
    "drill_down": true,
    "export_formats": ["html", "pdf"]
  },
  "metadata": {
    "extracted_at": "2025-03-18T14:30:00Z",
    "enriched_at": "2025-03-18T14:30:05Z",
    "confidence_score": 0.94,
    "pipeline_version": "1.0.0"
  }
}
```

### API Endpoints

**POST /render - Generate Visualization**
```bash
curl -X POST http://viz-framework:8003/render \
  -H "Content-Type: application/json" \
  -d @enriched-data.json
```

Response:
```json
{
  "dashboard_id": "dash-550e8400",
  "url": "http://viz-framework:8003/dashboard/dash-550e8400",
  "status": "rendering",
  "estimated_completion": "2025-03-18T14:30:30Z"
}
```

---

## Error Handling

### Retry Policy

| Stage | Failure Mode | Retry Strategy |
|-------|--------------|----------------|
| Extraction | LLM timeout | 3 retries with exponential backoff |
| Enrichment | API rate limit | 5 retries, 60s delay |
| Visualization | Rendering error | 1 retry, fallback to static image |

### Dead Letter Queue

Failed extractions are stored for manual review:

```sql
-- Query failed pipeline stages
SELECT 
  extraction_id,
  stage,
  error_message,
  failed_at,
  retry_count
FROM pipeline_failures
WHERE resolved = false
ORDER BY failed_at DESC;
```

---

## Authentication

### Inter-Service Auth

Services authenticate via shared secret in headers:

```bash
curl -H "X-Pipeline-Auth: ${PIPELINE_SECRET}" \
     -H "X-Service-Name: findoc-analyzer" \
     http://ticker-agent:8002/ingest
```

### Environment Configuration

```bash
# .env file
PIPELINE_SECRET=your-shared-secret-here
WEBHOOK_VERIFY_SIGNATURES=true
```

---

## Data Retention

| Data Type | Retention Period | Storage |
|-----------|-----------------|---------|
| Raw extractions | 90 days | PostgreSQL |
| Enriched data | 1 year | PostgreSQL |
| Generated dashboards | 30 days | Viz Framework storage |
| Audit logs | 7 years | Separate audit database |
| Failed job logs | 30 days | Pipeline logs volume |

---

## Schema Evolution

Pipeline schemas follow semantic versioning:

| Version | Status | Changes |
|---------|--------|---------|
| 1.0.0 | Current | Initial release |
| 1.1.0 | Planned | Add ESG metrics field |
| 2.0.0 | Planned | Breaking change: nested financials restructuring |

Backward compatibility: Services must accept unknown fields (forward compatibility).
