"""Tests for sagemaker/inference.py's request-handling logic.

model_fn (loads a real model) is out of scope -- same convention as the
rest of this repo's model-adjacent tests (fake objects standing in for
FinancialLLM, no real weights loaded). input_fn/predict_fn/output_fn are
pure enough to test directly.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.inference import input_fn, output_fn, predict_fn


def _fake_model(raw_output="{}", latency_ms=1.0, chat_template=None):
    tokenizer = SimpleNamespace(chat_template=chat_template)
    if chat_template:
        tokenizer.apply_chat_template = (
            lambda messages, tokenize, add_generation_prompt: "CHAT_PROMPT"
        )
    return SimpleNamespace(
        tokenizer=tokenizer,
        model_version="fake-v1",
        generate=lambda prompt, max_tokens=512: (raw_output, latency_ms),
    )


class TestInputFn:
    def test_parses_json_body(self):
        assert input_fn('{"task": "extract", "text": "hi"}') == {"task": "extract", "text": "hi"}

    def test_rejects_non_json_content_type(self):
        with pytest.raises(ValueError):
            input_fn("plain text", content_type="text/plain")


class TestPredictFn:
    def test_extract_task_builds_prompt_and_generates(self):
        model = _fake_model(raw_output='{"company_name": "Apple"}')
        result = predict_fn({"task": "extract", "text": "some filing text"}, model)
        assert result["raw_output"] == '{"company_name": "Apple"}'
        assert result["model_version"] == "fake-v1"

    def test_rag_query_task_uses_prompt_directly(self):
        model = _fake_model(raw_output="some answer")
        result = predict_fn({"task": "rag_query", "prompt": "What was revenue?"}, model)
        assert result["raw_output"] == "some answer"

    def test_unknown_task_raises(self):
        model = _fake_model()
        with pytest.raises(ValueError):
            predict_fn({"task": "not_a_real_task"}, model)

    def test_extract_falls_back_to_plain_prompt_without_chat_template(self):
        """Mirrors src/extraction/inference.py's own fallback -- base
        (non-instruct) checkpoints ship no chat_template."""
        model = _fake_model(chat_template=None)
        result = predict_fn({"task": "extract", "text": "filing text"}, model)
        assert result["model_version"] == "fake-v1"


class TestOutputFn:
    def test_serializes_to_json(self):
        body = output_fn({"raw_output": "x"})
        assert json.loads(body) == {"raw_output": "x"}

    def test_rejects_non_json_accept(self):
        with pytest.raises(ValueError):
            output_fn({"raw_output": "x"}, accept="text/plain")
