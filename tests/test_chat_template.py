"""Tests for src/chat_template.py.

The point of this module is that training and inference render the *same*
prompt, so these tests assert the two renderings agree rather than just
checking the template is non-empty.
"""

import sys
from pathlib import Path

import pytest
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat_template import (
    EOT_TOKEN,
    LLAMA31_CHAT_TEMPLATE,
    ensure_chat_template,
    generation_eos_token_ids,
)


class FakeTokenizer:
    """Minimal stand-in: only what the helpers actually touch."""

    def __init__(self, chat_template=None, eos_token_id=128001, vocab=None, unk_token_id=None):
        self.chat_template = chat_template
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self._vocab = vocab if vocab is not None else {EOT_TOKEN: 128009}

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


def render(messages, add_generation_prompt):
    return Template(LLAMA31_CHAT_TEMPLATE).render(
        messages=messages, add_generation_prompt=add_generation_prompt
    )


SYSTEM_USER = [
    {"role": "system", "content": "You are an analyst."},
    {"role": "user", "content": "Extract from: ACME 10-K"},
]
ASSISTANT = {"role": "assistant", "content": '{"revenue": 1}'}


class TestEnsureChatTemplate:
    def test_installs_when_missing(self):
        tok = FakeTokenizer(chat_template=None)
        assert ensure_chat_template(tok) is True
        assert tok.chat_template == LLAMA31_CHAT_TEMPLATE

    def test_installs_when_empty_string(self):
        tok = FakeTokenizer(chat_template="")
        assert ensure_chat_template(tok) is True
        assert tok.chat_template == LLAMA31_CHAT_TEMPLATE

    def test_does_not_clobber_an_existing_template(self):
        """Instruct checkpoints ship a real template -- overwriting it would be
        worse than the missing-template problem this module solves."""
        tok = FakeTokenizer(chat_template="{{ 'INSTRUCT MODEL TEMPLATE' }}")
        assert ensure_chat_template(tok) is False
        assert tok.chat_template == "{{ 'INSTRUCT MODEL TEMPLATE' }}"


class TestTemplateRendering:
    def test_inference_prompt_is_a_prefix_of_the_training_text(self):
        """The invariant that makes the adapter usable: what the model is
        trained on is exactly the inference prompt plus the answer."""
        training_text = render(SYSTEM_USER + [ASSISTANT], add_generation_prompt=False)
        inference_prompt = render(SYSTEM_USER, add_generation_prompt=True)
        assert training_text.startswith(inference_prompt)

    def test_training_text_continues_with_answer_then_terminator(self):
        training_text = render(SYSTEM_USER + [ASSISTANT], add_generation_prompt=False)
        inference_prompt = render(SYSTEM_USER, add_generation_prompt=True)
        assert training_text[len(inference_prompt):] == ASSISTANT["content"] + EOT_TOKEN

    def test_no_bos_token_emitted(self):
        """Both call sites re-tokenize the rendered string, and the tokenizer
        prepends BOS itself -- emitting it here too would double it."""
        assert "<|begin_of_text|>" not in render(SYSTEM_USER, add_generation_prompt=True)

    def test_generation_prompt_opens_an_assistant_turn(self):
        prompt = render(SYSTEM_USER, add_generation_prompt=True)
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_each_turn_is_terminated(self):
        text = render(SYSTEM_USER + [ASSISTANT], add_generation_prompt=False)
        assert text.count(EOT_TOKEN) == 3

    def test_roles_are_rendered_into_headers(self):
        text = render(SYSTEM_USER, add_generation_prompt=False)
        assert "<|start_header_id|>system<|end_header_id|>" in text
        assert "<|start_header_id|>user<|end_header_id|>" in text


class TestGenerationEosTokenIds:
    def test_includes_both_end_of_text_and_eot(self):
        ids = generation_eos_token_ids(FakeTokenizer(eos_token_id=128001))
        assert ids == [128001, 128009]

    def test_omits_eot_when_absent_from_vocab(self):
        tok = FakeTokenizer(eos_token_id=2, vocab={}, unk_token_id=0)
        assert generation_eos_token_ids(tok) == [2]

    def test_no_duplicate_when_eos_is_already_eot(self):
        tok = FakeTokenizer(eos_token_id=128009)
        assert generation_eos_token_ids(tok) == [128009]

    def test_handles_missing_eos_token_id(self):
        tok = FakeTokenizer(eos_token_id=None)
        assert generation_eos_token_ids(tok) == [128009]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
