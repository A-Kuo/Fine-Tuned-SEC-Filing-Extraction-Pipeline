"""Data collator for SEC filing extraction fine-tuning.

Handles batching of chat-format training examples with proper label masking.
The key insight: we only compute loss on the assistant's JSON output tokens,
not on the system prompt or user's filing text. This focuses the gradient
signal on learning the extraction mapping.

Label masking math:
    For a sequence [system_tokens, user_tokens, assistant_tokens]:
    labels = [-100, -100, ..., -100, assistant_token_1, ..., assistant_token_n]

    Cross-entropy loss ignores positions where label = -100, so gradients
    only flow through the assistant response. This is critical because
    without masking, the model wastes capacity predicting SEC filing text
    instead of learning the extraction function.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer


IGNORE_INDEX = -100


@dataclass
class FinancialDataCollator:
    """Collate chat-format examples into padded batches with label masking.

    For each example:
    1. Apply chat template to get full token sequence
    2. Find where assistant response starts
    3. Mask all tokens before assistant response in labels
    4. Pad batch to uniform length

    This ensures the training loss (cross-entropy) is computed only over
    the JSON extraction tokens, not the filing text or instruction.
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: str = "longest"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of examples.

        Args:
            features: List of dicts, each with 'messages' (chat format)
                      or 'text' (alpaca format).

        Returns:
            Batch dict with input_ids, attention_mask, labels.
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            input_ids, labels = self._process_single(feature)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones_like(input_ids))

        # Pad to longest in batch
        return self._pad_batch(batch_input_ids, batch_labels, batch_attention_mask)

    def _process_single(self, feature: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single training example into (input_ids, labels)."""
        if "messages" in feature:
            return self._process_chat(feature["messages"])
        elif "text" in feature:
            return self._process_text(feature["text"])
        else:
            raise ValueError(f"Unknown feature format: {list(feature.keys())}")

    def _process_chat(self, messages: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Process chat-format messages with proper label masking.

        Tokenizes the full conversation, then masks everything before
        the assistant's response so loss is only on output tokens.
        """
        # Tokenize full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Tokenize everything except assistant response to find split point
        non_assistant = messages[:-1]  # system + user only
        prefix_text = self.tokenizer.apply_chat_template(
            non_assistant, tokenize=False, add_generation_prompt=True
        )
        prefix_ids = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Create labels: mask prefix with IGNORE_INDEX
        labels = full_ids.clone()
        prefix_len = min(len(prefix_ids), len(labels))
        labels[:prefix_len] = IGNORE_INDEX

        return full_ids, labels

    def _process_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Process alpaca-format text. Masks everything before ### Response:."""
        full_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Find "### Response:" position
        response_marker = "### Response:"
        marker_ids = self.tokenizer(response_marker, add_special_tokens=False).input_ids

        labels = full_ids.clone()

        # Find marker position in token sequence
        marker_pos = self._find_subsequence(full_ids.tolist(), marker_ids)
        if marker_pos >= 0:
            # Mask everything up to and including the marker
            mask_end = marker_pos + len(marker_ids)
            labels[:mask_end] = IGNORE_INDEX
        else:
            # Fallback: mask first 80% (approximate)
            mask_end = int(len(labels) * 0.8)
            labels[:mask_end] = IGNORE_INDEX

        return full_ids, labels

    @staticmethod
    def _find_subsequence(sequence: list, subsequence: list) -> int:
        """Find first occurrence of subsequence in sequence. Returns -1 if not found."""
        sub_len = len(subsequence)
        for i in range(len(sequence) - sub_len + 1):
            if sequence[i : i + sub_len] == subsequence:
                return i
        return -1

    def _pad_batch(
        self,
        input_ids_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pad all sequences in batch to the same length."""
        if self.padding == "longest":
            max_len = max(len(ids) for ids in input_ids_list)
        else:
            max_len = self.max_length

        max_len = min(max_len, self.max_length)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for input_ids, labels, attn_mask in zip(
            input_ids_list, labels_list, attention_mask_list
        ):
            pad_len = max_len - len(input_ids)

            if pad_len > 0:
                # Right-pad
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)]
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), IGNORE_INDEX)]
                )
                attn_mask = torch.cat(
                    [attn_mask, torch.zeros(pad_len, dtype=torch.long)]
                )
            elif pad_len < 0:
                # Truncate
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                attn_mask = attn_mask[:max_len]

            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_mask.append(attn_mask)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
        }
