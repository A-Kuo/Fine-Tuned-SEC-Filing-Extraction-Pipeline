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


def _make_full_engine(raw_output: str):
    """A fake with a working .generate() too, for exercising extract() end
    to end (not just prompt-building)."""
    tokenizer = SimpleNamespace(chat_template=None)
    model = SimpleNamespace(
        tokenizer=tokenizer,
        model_version="fake-v1",
        generate=lambda prompt, max_tokens=512: (raw_output, 1.0),
    )
    return ExtractionEngine(model=model)


class TestModelLoadFailureDegradesGracefully:
    """Regression coverage for a real bug found via scripts/smoke_test.py
    in CI: initialize() used to sit outside extract()'s own try/except (and
    entirely unguarded in extract_batch()), so a model-load failure (e.g. a
    HuggingFace 401 on a gated repo with no HF_TOKEN, exactly what happened
    in that real CI run) propagated straight out of both methods uncaught.
    serving/api.py's run_extraction() then turned that into a raw
    HTTPException(500) -- precisely the outcome scripts/smoke_test.py's own
    docstring says a missing/unavailable model must NOT produce (a 5xx);
    only a graceful degraded response is acceptable."""

    @staticmethod
    def _install_fake_model_module(monkeypatch, error_message: str):
        """initialize() does `from src.extraction.model import FinancialLLM`
        -- the real module imports peft/torch, not installed in this test
        environment (matches every other model-adjacent test in this repo).
        Inject a fake module into sys.modules instead of importing the real
        one, so this exercises extract()'s/extract_batch()'s own error
        handling without needing those heavy deps."""
        import sys
        from types import ModuleType

        class _FakeFinancialLLM:
            @staticmethod
            def from_config():
                raise OSError(error_message)

        fake_module = ModuleType("src.extraction.model")
        fake_module.FinancialLLM = _FakeFinancialLLM
        monkeypatch.setitem(sys.modules, "src.extraction.model", fake_module)

    def test_extract_returns_error_status_not_an_exception(self, monkeypatch):
        engine = ExtractionEngine(model=None)  # _initialized=False -- initialize() will actually run
        self._install_fake_model_module(
            monkeypatch,
            "You are trying to access a gated repo. "
            "Access to model meta-llama/Llama-3.1-8B is restricted.",
        )

        from src.extraction.inference import ExtractionRequest

        response = engine.extract(ExtractionRequest(text="irrelevant"))

        assert response.status == "error"
        assert "gated repo" in response.error
        assert response.model_version == "unknown"

    def test_extract_batch_returns_error_responses_not_an_exception(self, monkeypatch):
        engine = ExtractionEngine(model=None)
        self._install_fake_model_module(monkeypatch, "gated repo, no HF_TOKEN")

        from src.extraction.inference import ExtractionRequest

        responses = engine.extract_batch([ExtractionRequest(text="a"), ExtractionRequest(text="b")])

        assert len(responses) == 2
        assert all(r.status == "error" for r in responses)
        assert all("gated repo" in r.error for r in responses)


class TestExtractTelemetry:
    """extract() must attach ParseTelemetry to every ExtractionResponse it
    returns -- previously there was no way to tell, from the response
    alone, which of the 5 fallback-parser stages actually recovered a
    result (see src/extraction/parser_telemetry.py)."""

    def test_direct_json_success_reports_winning_stage(self):
        engine = _make_full_engine('{"company_name": "Apple", "filing_type": "10-K", "date": "2024-01-01", "filing_id": "f-1"}')
        from src.extraction.inference import ExtractionRequest

        response = engine.extract(ExtractionRequest(text="irrelevant"))

        assert response.telemetry is not None
        assert response.telemetry.winning_stage == "direct"

    def test_parse_error_still_carries_telemetry(self):
        engine = _make_full_engine("complete garbage, no structure whatsoever")
        from src.extraction.inference import ExtractionRequest

        response = engine.extract(ExtractionRequest(text="irrelevant"))

        assert response.status == "parse_error"
        assert response.telemetry is not None
        assert response.telemetry.winning_stage is None
        assert len(response.telemetry.attempts) > 0


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
