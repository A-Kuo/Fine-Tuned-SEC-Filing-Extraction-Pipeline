"""SEC Filing Extraction Engine.

Takes raw filing text and produces structured JSON extractions using
the fine-tuned QLoRA model. Handles prompt construction, inference,
post-processing, and confidence scoring.

The extraction pipeline:
    1. Chunk input text (SEC filings can be 100+ pages)
    2. Build extraction prompt using chat template
    3. Run model inference (greedy decoding for determinism)
    4. Parse + validate JSON output
    5. Score confidence based on generation logprobs

For production: this module is used by both the FastAPI server and
the batch inference script. The model is loaded once and reused.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from loguru import logger

from src.core.config import load_config

if TYPE_CHECKING:
    from src.extraction.model import FinancialLLM

from src.extraction.postprocessing import (
    parse_extraction,
    validate_extraction,
    ExtractionResult,
    ValidationError,
)


# ─── System prompt matching training format ──────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial document analysis expert. Given SEC filing text, "
    "extract structured data and return it as a valid JSON object. "
    "Be precise with company names, filing types, dates, and financial figures. "
    "If a field cannot be determined from the text, set it to null."
)

EXTRACTION_INSTRUCTION = (
    "Extract structured financial data from the following SEC filing. "
    "Return a JSON object with: filing_id, company_name, ticker, filing_type, "
    "date, fiscal_year_end, revenue, net_income, total_assets, total_liabilities, eps, sector."
)


@dataclass
class ExtractionRequest:
    """Single extraction request."""
    text: str
    filing_id: str | None = None  # Optional pre-assigned ID
    max_text_length: int = 6000  # Truncate to fit context window
    max_tokens: int = 512


@dataclass
class ExtractionResponse:
    """Complete extraction response with metadata."""
    result: ExtractionResult | None
    raw_output: str
    latency_ms: float
    model_version: str
    status: str  # 'success', 'validation_error', 'parse_error', 'timeout'
    error: str | None = None
    confidence_score: float = 0.0


class ExtractionEngine:
    """Main extraction engine coordinating model inference + validation.

    Design rationale: separating the engine from the model allows us to
    swap in different backends (local model, vLLM server, API) without
    changing the extraction logic.
    """

    def __init__(self, model: FinancialLLM | None = None, confidence_scoring: str | None = None):
        """Initialize with a loaded model.

        Args:
            model: Pre-loaded FinancialLLM. If None, loads from config.
            confidence_scoring: "heuristic" (default) or "logprob". Reads
                config.yaml's extraction.confidence_scoring if not given
                explicitly. "logprob" is a scaffold, not a working
                implementation -- see _estimate_confidence_from_logprobs().
        """
        self.model = model
        self._initialized = model is not None
        if confidence_scoring is None:
            try:
                confidence_scoring = load_config().get("extraction", {}).get("confidence_scoring", "heuristic")
            except Exception:
                confidence_scoring = "heuristic"
        self._confidence_scoring = confidence_scoring

    def initialize(self) -> None:
        """Lazy-load model on first use."""
        if not self._initialized:
            logger.info("Initializing extraction engine...")
            from src.extraction.model import FinancialLLM
            self.model = FinancialLLM.from_config()
            self._initialized = True
            logger.info("Extraction engine ready")

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Extract structured data from a single SEC filing.

        Pipeline:
            1. Truncate + clean input text
            2. Build chat-format prompt
            3. Run model inference
            4. Parse JSON from output
            5. Validate against schema
            6. Compute confidence score

        Args:
            request: ExtractionRequest with filing text.

        Returns:
            ExtractionResponse with results, metadata, and status.
        """
        self.initialize()

        start_time = time.time()

        try:
            # Step 1: Prepare input
            cleaned_text = self._prepare_text(request.text, request.max_text_length)

            # Step 2: Build prompt
            prompt = self._build_prompt(cleaned_text)

            # Step 3: Model inference
            raw_output, inference_ms = self.model.generate(
                prompt, max_tokens=request.max_tokens
            )

            # Step 4: Parse JSON
            extraction = parse_extraction(raw_output)

            # Step 5: Validate
            is_valid, errors = validate_extraction(extraction)

            if not is_valid:
                return ExtractionResponse(
                    result=extraction,  # Return partial result
                    raw_output=raw_output,
                    latency_ms=inference_ms,
                    model_version=self.model.model_version,
                    status="validation_error",
                    error=f"Validation failed: {'; '.join(errors)}",
                    confidence_score=self._estimate_confidence(extraction, errors),
                )

            # Step 6: Confidence score
            confidence = self._estimate_confidence(extraction, [])

            # Override filing_id if provided
            if request.filing_id and extraction:
                extraction.filing_id = request.filing_id

            total_ms = (time.time() - start_time) * 1000

            return ExtractionResponse(
                result=extraction,
                raw_output=raw_output,
                latency_ms=total_ms,
                model_version=self.model.model_version,
                status="success",
                confidence_score=confidence,
            )

        except json.JSONDecodeError as e:
            total_ms = (time.time() - start_time) * 1000
            return ExtractionResponse(
                result=None,
                raw_output=raw_output if "raw_output" in dir() else "",
                latency_ms=total_ms,
                model_version=self.model.model_version if self.model else "unknown",
                status="parse_error",
                error=f"JSON parse failed: {str(e)}",
            )

        except Exception as e:
            total_ms = (time.time() - start_time) * 1000
            logger.error(f"Extraction failed: {e}")
            return ExtractionResponse(
                result=None,
                raw_output="",
                latency_ms=total_ms,
                model_version=self.model.model_version if self.model else "unknown",
                status="error",
                error=str(e),
            )

    def extract_batch(
        self, requests: list[ExtractionRequest]
    ) -> list[ExtractionResponse]:
        """Extract from multiple filings in a batch.

        Batching gives ~2x throughput by amortizing GPU kernel overhead.
        Documents are grouped by similar length to minimize padding waste.
        """
        self.initialize()

        # Sort by text length for efficient batching (less padding waste)
        indexed_requests = sorted(
            enumerate(requests), key=lambda x: len(x[1].text)
        )

        prompts = []
        for _, req in indexed_requests:
            text = self._prepare_text(req.text, req.max_text_length)
            prompts.append(self._build_prompt(text))

        # Batch inference
        raw_results = self.model.generate_batch(prompts)

        # Post-process each result
        responses = [None] * len(requests)
        for (orig_idx, req), (raw_output, latency_ms) in zip(
            indexed_requests, raw_results
        ):
            try:
                extraction = parse_extraction(raw_output)
                is_valid, errors = validate_extraction(extraction)

                status = "success" if is_valid else "validation_error"
                confidence = self._estimate_confidence(extraction, errors)

                if req.filing_id and extraction:
                    extraction.filing_id = req.filing_id

                responses[orig_idx] = ExtractionResponse(
                    result=extraction,
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    model_version=self.model.model_version,
                    status=status,
                    error=f"Validation: {'; '.join(errors)}" if errors else None,
                    confidence_score=confidence,
                )
            except Exception as e:
                responses[orig_idx] = ExtractionResponse(
                    result=None,
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    model_version=self.model.model_version,
                    status="error",
                    error=str(e),
                )

        return responses

    def _prepare_text(self, text: str, max_length: int) -> str:
        """Clean and truncate filing text.

        SEC filings can be 100+ pages. We truncate to fit the model's
        context window while keeping the most information-dense sections
        (header + financial highlights are usually in the first few pages).
        """
        # Basic cleaning
        text = text.strip()
        # Collapse multiple newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        # Collapse multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")

        # Truncate to max character length
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Filing truncated for processing]"

        return text

    def _build_prompt(self, filing_text: str) -> str:
        """Build the extraction prompt in chat format.

        Must match the training format exactly—any deviation degrades accuracy.
        The tokenizer's chat template handles special tokens (<|begin_of_text|>, etc.).
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{EXTRACTION_INSTRUCTION}\n\n{filing_text}"},
        ]

        if getattr(self.model.tokenizer, "chat_template", None):
            # Apply chat template (adds special tokens for Llama 3.1 Instruct)
            return self.model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Base (non-instruct) checkpoints like meta-llama/Llama-3.1-8B ship no
        # chat template -- fall back to a plain instruction-style prompt so
        # extraction still works instead of raising on every call.
        return f"{SYSTEM_PROMPT}\n\n{EXTRACTION_INSTRUCTION}\n\n{filing_text}\n\nResponse:"

    def _estimate_confidence(
        self,
        extraction: ExtractionResult | None,
        errors: list[str],
        raw_output: str | None = None,
        generation_scores: "list | None" = None,
    ) -> float:
        """Estimate confidence score for an extraction.

        Dispatches on config.yaml's extraction.confidence_scoring
        ("heuristic", the default, or "logprob"). raw_output/generation_scores
        are only used by the logprob path -- existing callers that don't pass
        them keep getting heuristic scoring unchanged.
        """
        if self._confidence_scoring == "logprob":
            return self._estimate_confidence_from_logprobs(
                extraction, errors, raw_output, generation_scores
            )
        return self._estimate_confidence_heuristic(extraction, errors)

    def _estimate_confidence_heuristic(
        self,
        extraction: ExtractionResult | None,
        errors: list[str],
    ) -> float:
        """Heuristic confidence based on field completeness and validation.

        Score breakdown:
            - 0.0-0.3: Parse failure or critical errors
            - 0.3-0.6: Partial extraction, some fields missing
            - 0.6-0.8: Most fields present, some validation issues
            - 0.8-1.0: All required fields valid
        """
        if extraction is None:
            return 0.0

        score = 0.5  # Base score for successful parse

        # Required field bonuses
        required = ["company_name", "filing_type", "date"]
        for field_name in required:
            if getattr(extraction, field_name, None):
                score += 0.1

        # Financial field bonuses
        financial = ["revenue", "net_income", "total_assets"]
        for field_name in financial:
            if getattr(extraction, field_name, None):
                score += 0.05

        # Penalty for validation errors
        score -= len(errors) * 0.1

        return max(0.0, min(1.0, score))

    def _estimate_confidence_from_logprobs(
        self,
        extraction: ExtractionResult | None,
        errors: list[str],
        raw_output: str | None,
        generation_scores: "list | None",
    ) -> float:
        """SCAFFOLD, not a working implementation -- see config.yaml's
        extraction.confidence_scoring docstring for how to enable this path,
        and don't: it raises rather than silently returning a plausible-
        looking but unverified number.

        The intended design: average per-token probability (from
        FinancialLLM.generate()'s output_scores, if the caller requests
        return_dict_in_generate=True/output_scores=True) over the token span
        of each field's generated JSON value, giving a real per-field
        confidence instead of the heuristic's fixed per-field bonus. This
        needs three things this environment cannot provide to build safely:
        (1) a real GPU to run generation and inspect actual score tensors,
        (2) verifying the tokenizer offset-mapping between raw_output's
        string positions and generation_scores' token positions actually
        lines up for this model/tokenizer, and (3) FinancialLLM.generate()
        currently returns only (text, latency_ms) -- it would need extending
        to optionally return scores too, itself untestable without a GPU.
        Writing a plausible-looking implementation without being able to run
        it against a real model would be exactly the kind of unverified
        number this repo's truth audit (docs/TRUTH_AUDIT.md) exists to catch.
        """
        raise NotImplementedError(
            "logprob confidence scoring is a documented scaffold, not "
            "implemented -- see this method's docstring. Set "
            "extraction.confidence_scoring back to 'heuristic' in "
            "config.yaml, or implement and verify this on a GPU."
        )
