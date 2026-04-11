"""Format raw JSONL data into chat-template format for QLoRA fine-tuning.

Takes the instruction/input/output JSONL from download_dataset.py and converts
it into the chat template format expected by Llama 3.1's tokenizer + SFTTrainer.

The mathematical connection: QLoRA minimizes the cross-entropy loss between
the model's predicted token distribution and the ground-truth output tokens,
but *only updates the low-rank adapter matrices* (rank r << d_model).
Getting the prompt format right directly affects what the model learns to predict.

Usage:
    python scripts/format_data.py --input data/sec_filings_train.jsonl --format chat
    python scripts/format_data.py --input data/sec_filings_train.jsonl --format alpaca
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, get_project_root

console = Console()

# ─── System prompt for the financial extraction task ─────────────────────────

SYSTEM_PROMPT = (
    "You are a financial document analysis expert. Given SEC filing text, "
    "extract structured data and return it as a valid JSON object. "
    "Be precise with company names, filing types, dates, and financial figures. "
    "If a field cannot be determined from the text, set it to null."
)


def format_as_chat(example: dict) -> dict:
    """Convert to Llama 3.1 chat template format.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {instruction}\n\n{input}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {output}<|eot_id|>

    This is the native format for Llama 3.1 instruction-tuned models.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "id": example.get("id", ""),
        "messages": messages,
    }


def format_as_alpaca(example: dict) -> dict:
    """Convert to Alpaca-style instruction format.

    Format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    Simpler format, compatible with more training frameworks.
    """
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {
        "id": example.get("id", ""),
        "text": text,
    }


def format_as_completion(example: dict) -> dict:
    """Convert to simple prompt-completion format.

    Used for basic causal LM fine-tuning without chat templates.
    The prompt and completion are concatenated with a separator.
    """
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"SEC Filing:\n{example['input']}\n\n"
        f"Extracted Data:\n"
    )
    return {
        "id": example.get("id", ""),
        "prompt": prompt,
        "completion": example["output"],
    }


FORMAT_FN = {
    "chat": format_as_chat,
    "alpaca": format_as_alpaca,
    "completion": format_as_completion,
}


def format_dataset(
    input_path: Path,
    output_path: Path,
    fmt: str = "chat",
    max_samples: int | None = None,
) -> int:
    """Format entire dataset file.

    Args:
        input_path: Raw JSONL from download_dataset.py.
        output_path: Formatted JSONL for SFTTrainer.
        fmt: Format type ('chat', 'alpaca', 'completion').
        max_samples: Optional limit on number of examples.

    Returns:
        Number of examples formatted.
    """
    format_fn = FORMAT_FN[fmt]

    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc=f"Formatting ({fmt})"):
            if max_samples and count >= max_samples:
                break

            example = json.loads(line.strip())
            formatted = format_fn(example)
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    logger.info(f"Formatted {count} examples → {output_path}")
    return count


def validate_formatted_data(path: Path, fmt: str, num_check: int = 5) -> bool:
    """Quick validation that formatted data is well-formed."""
    errors = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= num_check:
                break
            try:
                data = json.loads(line)
                if fmt == "chat":
                    assert "messages" in data, "Missing 'messages' key"
                    assert len(data["messages"]) == 3, "Expected 3 messages (system/user/assistant)"
                    assert data["messages"][0]["role"] == "system"
                    assert data["messages"][1]["role"] == "user"
                    assert data["messages"][2]["role"] == "assistant"
                    # Verify output is valid JSON
                    json.loads(data["messages"][2]["content"])
                elif fmt == "alpaca":
                    assert "text" in data, "Missing 'text' key"
                    assert "### Instruction:" in data["text"]
                    assert "### Response:" in data["text"]
                elif fmt == "completion":
                    assert "prompt" in data and "completion" in data
                    json.loads(data["completion"])
            except Exception as e:
                errors.append(f"Line {i}: {e}")

    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        return False

    console.print(f"[green][OK][/green] Validation passed ({num_check} examples checked)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Format data for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: input with format suffix)",
    )
    parser.add_argument(
        "--format", type=str, choices=["chat", "alpaca", "completion"],
        default="chat", help="Output format (default: chat)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--validate", action="store_true", help="Validate output after formatting")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(f".{args.format}.jsonl")

    count = format_dataset(input_path, output_path, args.format, args.max_samples)
    console.print(f"\n[bold green][OK][/bold green] Formatted {count} examples → {output_path}")

    if args.validate:
        validate_formatted_data(output_path, args.format)


if __name__ == "__main__":
    main()
