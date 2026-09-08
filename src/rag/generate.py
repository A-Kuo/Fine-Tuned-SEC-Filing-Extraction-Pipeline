"""Answer generation over retrieved filing sections.

generate_fn is injected rather than loaded here -- the caller (scripts/
rag_query.py or serving/api.py's /rag/query route) is responsible for
lazily loading FinancialLLM.from_config() and passing its bound .generate
method in. This keeps generate_answer() fully unit-testable with a fake
`lambda p: ("fake answer", 0.0)`, no model download needed, and mirrors
this repo's existing lazy-load convention rather than introducing a new
model-loading path.

Calls generate_fn directly with a plain RAG prompt -- NOT through
ExtractionEngine (src/extraction/inference.py), which builds a
JSON-extraction-specific prompt and is the wrong shape for open-ended QA.
"""

from __future__ import annotations

from typing import Callable

RAG_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the question using only "
    "the excerpts below, which come from real SEC filings. Cite which "
    "filing each fact comes from. If the excerpts don't contain the answer, "
    "say so rather than guessing."
)


def _build_prompt(question: str, retrieved_sections: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{s['filing_id']} / {s['section_type']}]\n{s['content']}"
        for s in retrieved_sections
    )
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"--- Excerpts ---\n{excerpts}\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Answer ---\n"
    )


def generate_answer(
    question: str,
    retrieved_sections: list[dict],
    generate_fn: Callable[[str], tuple[str, float]],
) -> tuple[str, list[str]]:
    """Build a RAG prompt from retrieved sections and call generate_fn.

    Returns (answer_text, cited_filing_ids). cited_filing_ids is simply the
    distinct filing_ids that were actually retrieved and shown to the
    model -- what it was given the opportunity to cite, not a claim about
    what it necessarily used.
    """
    if not retrieved_sections:
        return "No relevant filing sections were found for this question.", []

    prompt = _build_prompt(question, retrieved_sections)
    answer_text, _latency_ms = generate_fn(prompt)

    cited_filing_ids = list(dict.fromkeys(s["filing_id"] for s in retrieved_sections))
    return answer_text, cited_filing_ids
