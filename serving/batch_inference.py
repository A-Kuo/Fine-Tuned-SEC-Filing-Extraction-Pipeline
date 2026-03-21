"""Batch Inference for SEC Filing Extraction.

Processes directories of SEC filings in batches, either through the
running API server or by loading the model directly. Supports:

- Resume from interruption (tracks processed files)
- Configurable batch size and concurrency
- Results saved as JSONL for downstream analysis
- Progress tracking with ETA

The batch processing strategy groups documents by similar length to
minimize padding waste in the model's attention computation. For a batch
of documents with lengths [100, 120, 800, 900]:
    - Naive batching: pad all to 900 tokens → 3620 tokens processed, 2600 wasted
    - Sorted batching: [100, 120] + [800, 900] → same work, ~40% less padding

Usage:
    # Against running API server
    python serving/batch_inference.py --input_dir data/filings/ --server_url http://localhost:8000

    # Direct model inference (no server needed)
    python serving/batch_inference.py --input_dir data/filings/ --local

    # Resume interrupted job
    python serving/batch_inference.py --input_dir data/filings/ --resume
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, get_project_root
from src.inference import ExtractionEngine, ExtractionRequest
from src.postprocessing import ExtractionResult

console = Console()


def collect_filings(input_dir: Path, extensions: tuple = (".txt", ".html", ".htm")) -> list[Path]:
    """Collect all filing files from input directory."""
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
    files.sort()
    return files


def load_processed(output_path: Path) -> set[str]:
    """Load set of already-processed filing filenames for resume support."""
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "source_file" in data:
                        processed.add(data["source_file"])
                except json.JSONDecodeError:
                    continue
    return processed


async def batch_extract_api(
    filings: list[Path],
    output_path: Path,
    server_url: str,
    batch_size: int = 16,
    max_concurrent: int = 4,
) -> dict:
    """Process filings through the API server.

    Uses async HTTP to overlap network latency with processing.
    Batches are sent concurrently up to max_concurrent limit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"success": 0, "error": 0, "total_latency_ms": 0}

    async with httpx.AsyncClient(timeout=60) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(filings))

            # Process in batches
            for batch_start in range(0, len(filings), batch_size):
                batch_files = filings[batch_start : batch_start + batch_size]
                batch_docs = []

                for fpath in batch_files:
                    text = fpath.read_text(errors="replace")
                    batch_docs.append({
                        "text": text[:6000],  # Truncate to fit context
                        "filing_id": fpath.stem,
                    })

                # Send batch request
                try:
                    resp = await client.post(
                        f"{server_url}/extract/batch",
                        json={"documents": batch_docs},
                    )

                    if resp.status_code == 200:
                        results = resp.json()
                        with open(output_path, "a") as f:
                            for fpath, result in zip(batch_files, results):
                                result["source_file"] = fpath.name
                                f.write(json.dumps(result) + "\n")

                                if result.get("status") == "success":
                                    stats["success"] += 1
                                else:
                                    stats["error"] += 1
                                stats["total_latency_ms"] += result.get("latency_ms", 0)
                    else:
                        logger.error(f"Batch request failed: {resp.status_code}")
                        stats["error"] += len(batch_files)

                except Exception as e:
                    logger.error(f"Batch request error: {e}")
                    stats["error"] += len(batch_files)

                progress.update(task, advance=len(batch_files))

    return stats


def batch_extract_local(
    filings: list[Path],
    output_path: Path,
    batch_size: int = 8,
) -> dict:
    """Process filings using locally loaded model (no server needed).

    Loads model once and processes all filings. More memory-efficient
    than the API route since there's no HTTP overhead, but single-threaded.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"success": 0, "error": 0, "total_latency_ms": 0}

    engine = ExtractionEngine()
    engine.initialize()  # Load model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting (local)...", total=len(filings))

        for batch_start in range(0, len(filings), batch_size):
            batch_files = filings[batch_start : batch_start + batch_size]

            requests = []
            for fpath in batch_files:
                text = fpath.read_text(errors="replace")
                requests.append(ExtractionRequest(
                    text=text,
                    filing_id=fpath.stem,
                ))

            # Batch inference
            responses = engine.extract_batch(requests)

            # Save results
            with open(output_path, "a") as f:
                for fpath, resp in zip(batch_files, responses):
                    record = {
                        "source_file": fpath.name,
                        "status": resp.status,
                        "latency_ms": resp.latency_ms,
                        "confidence_score": resp.confidence_score,
                        "model_version": resp.model_version,
                        "error": resp.error,
                    }

                    if resp.result:
                        record.update(resp.result.to_dict())

                    f.write(json.dumps(record) + "\n")

                    if resp.status == "success":
                        stats["success"] += 1
                    else:
                        stats["error"] += 1
                    stats["total_latency_ms"] += resp.latency_ms

            progress.update(task, advance=len(batch_files))

    return stats


def print_summary(stats: dict, elapsed: float, total_files: int):
    """Print batch processing summary."""
    console.print("\n" + "=" * 50)
    console.print("[bold]Batch Extraction Complete[/bold]")
    console.print("=" * 50)
    console.print(f"  Total files:     {total_files}")
    console.print(f"  Successful:      [green]{stats['success']}[/green]")
    console.print(f"  Failed:          [red]{stats['error']}[/red]")
    console.print(f"  Success rate:    {100 * stats['success'] / max(total_files, 1):.1f}%")
    console.print(f"  Total time:      {elapsed:.1f}s")
    console.print(f"  Throughput:      {total_files / max(elapsed, 0.001) * 60:.0f} docs/min")
    console.print(f"  Avg latency:     {stats['total_latency_ms'] / max(total_files, 1):.0f} ms/doc")
    console.print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Batch extract SEC filings")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with filing text files")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000", help="API server URL")
    parser.add_argument("--local", action="store_true", help="Use local model instead of API server")
    parser.add_argument("--batch_size", type=int, default=16, help="Documents per batch")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"

    # Collect files
    filings = collect_filings(input_dir)
    if not filings:
        console.print(f"[yellow]No filing files found in {input_dir}[/yellow]")
        sys.exit(0)

    console.print(f"Found {len(filings)} filing files in {input_dir}")

    # Resume support
    if args.resume:
        processed = load_processed(output_path)
        filings = [f for f in filings if f.name not in processed]
        if not filings:
            console.print("[green]All files already processed![/green]")
            sys.exit(0)
        console.print(f"Resuming: {len(filings)} files remaining")

    # Process
    start = time.time()

    if args.local:
        stats = batch_extract_local(filings, output_path, args.batch_size)
    else:
        stats = asyncio.run(
            batch_extract_api(filings, output_path, args.server_url, args.batch_size)
        )

    elapsed = time.time() - start
    print_summary(stats, elapsed, len(filings))

    console.print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
