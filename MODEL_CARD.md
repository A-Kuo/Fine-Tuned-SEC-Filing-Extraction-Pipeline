# Model Card: FinDocAnalyzer (llama-sec-v1)

## Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | llama-sec-v1 |
| **Base Model** | meta-llama/Llama-3.1-8B-Instruct |
| **Architecture** | Transformer (Decoder-only), 8B parameters |
| **Fine-tuning Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Quantization** | NF4 4-bit (NormalFloat) with double quantization |
| **LoRA Rank** | r=16, α=32 |
| **Trainable Parameters** | ~200M (2.5% of total) |
| **Model Size** | ~500 MB (LoRA adapter only) |
| **Total Inference Memory** | ~7.2 GB (base + adapter) |
| **Context Window** | 2048 tokens |
| **License** | MIT (adapter weights), Llama 3.1 Community License (base) |

## Intended Use

### Primary Use Cases
- **Automated SEC filing analysis**: Extract structured financial data from 10-K, 10-Q, and 8-K filings
- **Financial research pipelines**: Populate databases with standardized company metrics
- **Due diligence automation**: Rapid extraction of revenue, net income, assets, and liabilities
- **Compliance auditing**: Structured data extraction for regulatory review workflows

### Supported Document Types
- SEC Form 10-K (Annual reports)
- SEC Form 10-Q (Quarterly reports)
- SEC Form 8-K (Current reports)

### Extracted Fields
| Field | Type | Description |
|-------|------|-------------|
| `filing_id` | string | Unique identifier for the filing |
| `company_name` | string | Legal entity name |
| `ticker` | string | Stock ticker symbol |
| `filing_type` | enum | 10-K, 10-Q, or 8-K |
| `date` | ISO-8601 | Filing date (YYYY-MM-DD) |
| `fiscal_year_end` | string | Fiscal year end date |
| `revenue` | float | Revenue in millions USD |
| `net_income` | float | Net income in millions USD |
| `total_assets` | float | Total assets in millions USD |
| `total_liabilities` | float | Total liabilities in millions USD |
| `eps` | float | Earnings per share |
| `sector` | string | Industry sector classification |

## Performance

### Accuracy Metrics (Synthetic Test Set)
| Metric | Value |
|----------|-------|
| Overall Extraction Accuracy | 94% |
| Field-Level Accuracy | 92–99% per field |

### Latency & Throughput
| Configuration | p50 Latency | Throughput |
|--------------|-------------|------------|
| T4 (16 GB) | ~450 ms/doc | ~40 docs/min |
| A100 (40 GB) | ~320 ms/doc | ~60 docs/min |
| A100 + vLLM | ~250 ms/doc | ~120 docs/min |

### Resource Requirements
| Component | Specification |
|-----------|---------------|
| Minimum GPU | NVIDIA T4 (16 GB VRAM) |
| Recommended GPU | NVIDIA A100 (40 GB VRAM) for batch processing |
| CPU Mode | Not supported (quantization requires CUDA) |
| System RAM | 32 GB recommended for data processing |

## Training Data

### Current (Iteration 1)
- **Source**: Synthetic SEC filings generated from templates
- **Size**: ~500 training examples
- **Distribution**: Balanced across filing types, sectors, and company sizes
- **Format**: Chat-format JSON with system prompt + user instruction + assistant response

### Planned (Iteration 2)
- **Source**: Real SEC EDGAR filings with XBRL ground truth
- **Target Size**: 5,000–10,000 examples
- **Validation**: Financial figures cross-referenced with official XBRL submissions

## Limitations

### Accuracy Limitations
1. **Synthetic training data**: Current model trained on template-generated filings; real-world accuracy on authentic EDGAR documents is unverified (targeted for Iteration 2)
2. **Numeric precision**: ±5% tolerance recommended for financial figures due to unit normalization (thousands vs. millions) and rounding
3. **Truncation effects**: Filings exceeding 2048 tokens are truncated; key information may be lost in long documents

### Scope Limitations
1. **English only**: Model does not support non-English filings
2. **US SEC only**: Not validated on international financial reporting formats (IFRS, UK FCA, etc.)
3. **No temporal reasoning**: Does not track multi-year trends or YoY comparisons across filings
4. **No table extraction**: Cannot parse complex financial tables or XBRL directly; relies on narrative text

### Technical Limitations
1. **GPU dependency**: NF4 quantization requires CUDA; no CPU-only inference path
2. **Context window**: 2048 token limit means very long filings (100+ pages) require chunking
3. **Deterministic output**: Greedy decoding (`temperature=0`) trades creativity for consistency; may miss edge cases

## Ethical Considerations

### Financial Data Sensitivity
- Extracted data should be handled according to your organization's data governance policies
- SEC filings are public documents, but derivative datasets may have licensing restrictions

### Bias and Fairness
- Training data covers S&P 500-style large cap companies predominantly
- Performance may vary on micro-cap, foreign issuer, or non-standard industry filings
- Sector classification relies on standard NAICS/GICS codes; niche industries may be misclassified

### Responsible Use
- **Not for high-frequency trading**: Latency (~300ms) and batch processing nature unsuitable for real-time trading
- **Human-in-the-loop recommended**: Critical financial decisions should validate extraction against primary sources
- **Compliance validation**: Always verify extracted figures against official XBRL submissions for regulatory filings

## Citation

```bibtex
@software{fin_doc_analyzer_2025,
  author = {Kuo, Austin},
  title = {FinDocAnalyzer: SEC Filing Extraction via QLoRA Fine-Tuned Llama 3.1},
  url = {https://github.com/akuo6/financial-llm},
  year = {2025},
  version = {0.1.0}
}
```

## Model Architecture Diagram

```
SEC Filing Text (≤2048 tokens)
           │
           ▼
┌──────────────────────┐
│  Chat Format Prompt  │  Llama 3.1 template
│  + System + Instruction│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Llama 3.1 8B Base   │  NF4 4-bit quantized (frozen)
│  + LoRA Adapters     │  BA matrices r=16 (trainable)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Greedy Generation   │  do_sample=False
│  max_tokens=512      │  Deterministic output
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  5-Strategy Parser   │  JSON extraction + validation
│  + Schema Validation │  Cascade: direct → strip fences
└──────────┬───────────┘           → regex → fix truncation
           ▼
┌──────────────────────┐
│  Structured Output   │  12 financial fields
└──────────────────────┘
```

## Contact & Issues

- **Repository**: https://github.com/akuo6/financial-llm
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Security**: Report security vulnerabilities privately to the repository owner
