"""Tests for src/inference.py's ExtractionEngine prompt building.

Regression coverage for a real bug: base (non-instruct) checkpoints like
meta-llama/Llama-3.1-8B ship no tokenizer.chat_template, which made every
extraction call raise via apply_chat_template(). _build_prompt() must fall
back to a plain prompt instead of crashing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.chat_template import LLAMA31_CHAT_TEMPLATE
from src.extraction.inference import EXTRACTION_INSTRUCTION, SYSTEM_PROMPT, ExtractionEngine
from src.extraction.postprocessing import ExtractionResult


def _make_engine(chat_template):
    tokenizer = SimpleNamespace(chat_template=chat_template)

    if chat_template:
        tokenizer.apply_chat_template = (
            lambda messages, tokenize, add_generation_prompt: "CHAT_TEMPLATE_OUTPUT"
        )

    model = SimpleNamespace(tokenizer=tokenizer)
    return ExtractionEngine(model=model)


class TestBuildPrompt:
    def test_uses_chat_template_when_present(self):
        engine = _make_engine(chat_template="{% chat template %}")
        prompt = engine._build_prompt("some filing text")
        assert prompt == "CHAT_TEMPLATE_OUTPUT"

    def test_falls_back_to_plain_prompt_when_no_chat_template(self):
        """Base models (chat_template=None) must not raise."""
        engine = _make_engine(chat_template=None)
        prompt = engine._build_prompt("some filing text")
        assert "some filing text" in prompt
        assert isinstance(prompt, str)
        assert prompt  # non-empty

    def test_falls_back_to_plain_prompt_when_chat_template_is_empty_string(self):
        engine = _make_engine(chat_template="")
        prompt = engine._build_prompt("filing text here")
        assert "filing text here" in prompt


class TestRealTemplateIsUsed:
    """The repo installs LLAMA31_CHAT_TEMPLATE on base checkpoints, so the
    template branch -- not the fallback -- is what actually runs in production.
    """

    def test_build_prompt_matches_the_shared_template(self):
        from jinja2 import Template

        rendered = {}

        def apply(messages, tokenize, add_generation_prompt):
            rendered["out"] = Template(LLAMA31_CHAT_TEMPLATE).render(
                messages=messages, add_generation_prompt=add_generation_prompt
            )
            return rendered["out"]

        tokenizer = SimpleNamespace(
            chat_template=LLAMA31_CHAT_TEMPLATE, apply_chat_template=apply
        )
        engine = ExtractionEngine(model=SimpleNamespace(tokenizer=tokenizer))

        prompt = engine._build_prompt("ACME 10-K body")

        expected = Template(LLAMA31_CHAT_TEMPLATE).render(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{EXTRACTION_INSTRUCTION}\n\nACME 10-K body",
                },
            ],
            add_generation_prompt=True,
        )
        assert prompt == expected
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


class TestConfidenceScoringDispatch:
    """extraction.confidence_scoring defaults to "heuristic" (real, working).
    "logprob" is a documented scaffold (see
    ExtractionEngine._estimate_confidence_from_logprobs()'s docstring) that
    must raise clearly rather than silently returning an unverified number.
    """

    def test_defaults_to_heuristic_scoring(self):
        engine = ExtractionEngine(model=SimpleNamespace(), confidence_scoring="heuristic")
        result = ExtractionResult(company_name="Acme", filing_type="10-K", date="2024-01-01")
        score = engine._estimate_confidence(result, [])
        assert 0.0 <= score <= 1.0

    def test_heuristic_none_extraction_is_zero_confidence(self):
        engine = ExtractionEngine(model=SimpleNamespace(), confidence_scoring="heuristic")
        assert engine._estimate_confidence(None, []) == 0.0

    def test_logprob_scoring_raises_not_implemented(self):
        engine = ExtractionEngine(model=SimpleNamespace(), confidence_scoring="logprob")
        result = ExtractionResult(company_name="Acme")
        with pytest.raises(NotImplementedError):
            engine._estimate_confidence(result, [], raw_output="{}", generation_scores=None)

    def test_explicit_confidence_scoring_overrides_config(self):
        """Passing confidence_scoring explicitly must win over whatever
        config.yaml says, so tests/callers aren't at the mercy of the
        repo-wide default changing under them."""
        engine = ExtractionEngine(model=SimpleNamespace(), confidence_scoring="heuristic")
        assert engine._confidence_scoring == "heuristic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
