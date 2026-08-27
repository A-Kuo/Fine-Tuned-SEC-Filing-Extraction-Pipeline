"""Canonical chat template shared by the training and inference paths.

`meta-llama/Llama-3.1-8B` is the **base** checkpoint, not `-Instruct`, so it
ships no `tokenizer.chat_template`. Every call to `apply_chat_template()`
therefore raises, which breaks both:

  * training  -- SFTTrainer applies the template to the `messages` column of
    `data/sec_filings_train.chat.jsonl`, and
  * inference -- `ExtractionEngine._build_prompt()` formats the same
    system/user pair before generating.

Both sides must produce byte-identical formatting: an adapter trained on one
prompt shape and evaluated on another silently loses accuracy. Defining the
template once, here, is what keeps them in sync.

The template reproduces the Llama 3.1 conversation format using special tokens
that *are* already present in the base tokenizer's vocabulary
(`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`) -- only the template
string itself is missing from the base checkpoint.

It deliberately omits the leading `<|begin_of_text|>` BOS token: both call
sites feed the rendered string back through `tokenizer(...)`, which prepends
BOS itself. Emitting it here too would double it.
"""

# Turn-terminator token. Trained on as the stop signal, so generation must
# treat it as EOS -- see generation_eos_token_ids().
EOT_TOKEN = "<|eot_id|>"

LLAMA31_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] | trim }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)


def ensure_chat_template(tokenizer) -> bool:
    """Install the Llama 3.1 chat template if the tokenizer lacks one.

    Instruct checkpoints already carry their own template and are left
    untouched -- overwriting a model's real template would be worse than the
    problem this solves.

    Returns:
        True if a template was installed, False if one was already present.
    """
    if getattr(tokenizer, "chat_template", None):
        return False

    tokenizer.chat_template = LLAMA31_CHAT_TEMPLATE
    return True


def generation_eos_token_ids(tokenizer) -> list[int]:
    """EOS ids that should stop generation, including the turn terminator.

    The base checkpoint's `eos_token` is `<|end_of_text|>`, but a model trained
    with the template above ends each answer with `<|eot_id|>`. Without this,
    generation runs to `max_new_tokens` on every request instead of stopping at
    the end of the JSON object.
    """
    ids: list[int] = []

    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    # convert_tokens_to_ids() returns the UNK id (or None) for absent tokens.
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in ids:
        ids.append(eot_id)

    return ids
