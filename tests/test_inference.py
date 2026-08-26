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

from src.inference import ExtractionEngine


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
