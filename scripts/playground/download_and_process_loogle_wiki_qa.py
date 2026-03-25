#!/usr/bin/env python3
"""Download and process xiezhq/loogle-wiki-qa from Hugging Face.

This script downloads the raw dataset JSON and converts it into flat JSONL files:
1) contexts.jsonl: one row per context.
2) qa_flattened.jsonl: one row per question-answer pair with joined context text.
3) long_qa.jsonl: long-context subset based on a text-length threshold.
4) long_qa_infer.jsonl: optional chat.completions output for long-context subset.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


HF_DATASET_URL = (
    "https://huggingface.co/datasets/xiezhq/loogle-wiki-qa/resolve/main/loogle_wiki_qa.json"
)


def download_json(url: str, output_path: Path) -> dict[str, Any]:
    """Download and parse JSON from url, then store raw content locally."""
    try:
        with urlopen(url, timeout=60) as response:
            raw_bytes = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download dataset from {url}: {exc}") from exc

    output_path.write_bytes(raw_bytes)

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Downloaded file is not valid JSON: {output_path} ({exc})"
        ) from exc


def validate_schema(obj: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Validate expected schema and return contexts + queries."""
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected top-level JSON object, got: {type(obj)!r}")

    contexts = obj.get("contexts")
    queries = obj.get("queries")

    if not isinstance(contexts, dict):
        raise RuntimeError("Expected 'contexts' to be a dict[str, str].")
    if not isinstance(queries, list):
        raise RuntimeError("Expected 'queries' to be a list[dict].")

    for i, item in enumerate(queries):
        if not isinstance(item, dict):
            raise RuntimeError(f"Query at index {i} is not a dict: {type(item)!r}")
        missing = {"context", "question", "reference_answer"} - set(item.keys())
        if missing:
            raise RuntimeError(f"Query at index {i} missing keys: {sorted(missing)}")

    return contexts, queries


def write_contexts_jsonl(contexts: dict[str, str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for context_id, context_text in contexts.items():
            row = {"context_id": context_id, "context": context_text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qa_flattened_jsonl(
    contexts: dict[str, str], queries: list[dict[str, str]], output_path: Path
) -> tuple[int, int]:
    missing_context = 0
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, query in enumerate(queries):
            context_id = str(query.get("context", ""))
            context_text = contexts.get(context_id)
            if context_text is None:
                missing_context += 1
                context_text = ""

            row = {
                "id": idx,
                "context_id": context_id,
                "question": str(query.get("question", "")),
                "reference_answer": str(query.get("reference_answer", "")),
                "context": context_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written, missing_context


def build_flattened_rows(
    contexts: dict[str, str], queries: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    missing_context = 0
    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(queries):
        context_id = str(query.get("context", ""))
        context_text = contexts.get(context_id)
        if context_text is None:
            missing_context += 1
            context_text = ""
        rows.append(
            {
                "id": idx,
                "context_id": context_id,
                "question": str(query.get("question", "")),
                "reference_answer": str(query.get("reference_answer", "")),
                "context": context_text,
            }
        )
    return rows, missing_context


def write_rows_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_long_rows(
    rows: list[dict[str, Any]], min_context_chars: int, max_samples: int | None
) -> list[dict[str, Any]]:
    long_rows = [r for r in rows if len(r.get("context", "")) >= min_context_chars]
    if max_samples is not None:
        long_rows = long_rows[:max_samples]
    return long_rows


def build_prompt(row: dict[str, Any]) -> str:
    return (
        "You are a QA assistant. Answer the question strictly based on the context.\n\n"
        f"Context:\n{row['context']}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Return only the final answer."
    )


def run_chat_completion_on_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    openai_api_base: str,
    openai_api_key: str,
    model_name: str | None,
    max_completion_tokens: int,
    temperature: float,
) -> int:
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install it with: pip install openai"
        )

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    model = model_name
    if not model:
        models = client.models.list()
        if not models.data:
            raise RuntimeError("No model available from the OpenAI-compatible endpoint.")
        model = models.data[0].id
        print(f"[Info] Auto selected model: {model}")

    completed = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = build_prompt(row)
            start_time = time.time()
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_completion_tokens=1500,
                temperature=temperature,
                extra_body={"ignore_eos": True},
            )
            latency_s = time.time() - start_time
            prediction = response.choices[0].message.content if response.choices else ""

            out_row = {
                "id": row["id"],
                "context_id": row["context_id"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "prediction": prediction,
                "latency_s": round(latency_s, 4),
                "model": model,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            completed += 1
            print(
                f"[Infer] id={row['id']} "
                f"context_chars={len(row['context'])} "
                f"latency={latency_s:.3f}s"
            )

    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download xiezhq/loogle-wiki-qa and export normalized JSONL files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/loogle_wiki_qa"),
        help="Directory to store raw and processed files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if raw file already exists.",
    )
    parser.add_argument(
        "--long-context-min-chars",
        type=int,
        default=20000,
        help="Filter threshold: keep rows whose context length >= this value.",
    )
    parser.add_argument(
        "--long-context-max-samples",
        type=int,
        default=20,
        help="Cap number of long-context rows (set <=0 for no cap).",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run OpenAI chat.completions on filtered long-context rows.",
    )
    parser.add_argument(
        "--openai-api-base",
        type=str,
        default="http://localhost:8188/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name for chat.completions. Empty means auto-pick first served model.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=500,
        help="max_completion_tokens for chat.completions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for chat.completions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "loogle_wiki_qa.raw.json"
    contexts_jsonl_path = output_dir / "contexts.jsonl"
    qa_jsonl_path = output_dir / "qa_flattened.jsonl"
    long_qa_jsonl_path = output_dir / "long_qa.jsonl"
    long_qa_infer_jsonl_path = output_dir / "long_qa_infer.jsonl"

    if raw_path.exists() and not args.force:
        print(f"[Info] Reusing existing raw file: {raw_path}")
        obj = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        print(f"[Info] Downloading dataset from: {HF_DATASET_URL}")
        obj = download_json(HF_DATASET_URL, raw_path)
        print(f"[Info] Raw dataset saved to: {raw_path}")

    contexts, queries = validate_schema(obj)
    rows, missing_context = build_flattened_rows(contexts, queries)

    write_contexts_jsonl(contexts, contexts_jsonl_path)
    write_rows_jsonl(rows, qa_jsonl_path)

    max_samples = (
        args.long_context_max_samples if args.long_context_max_samples > 0 else None
    )
    long_rows = filter_long_rows(
        rows=rows,
        min_context_chars=args.long_context_min_chars,
        max_samples=max_samples,
    )
    write_rows_jsonl(long_rows, long_qa_jsonl_path)

    print("[Done] Processing completed.")
    print(f"  contexts: {len(contexts)} -> {contexts_jsonl_path}")
    print(f"  queries: {len(rows)} -> {qa_jsonl_path}")
    print(f"  missing context refs: {missing_context}")
    print(
        "  long rows: "
        f"{len(long_rows)} (min_context_chars={args.long_context_min_chars}) "
        f"-> {long_qa_jsonl_path}"
    )

    if args.run_inference:
        print("[Info] Running chat.completions inference on long rows...")
        completed = run_chat_completion_on_rows(
            rows=long_rows,
            output_path=long_qa_infer_jsonl_path,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
            model_name=args.model or None,
            max_completion_tokens=500,
            temperature=args.temperature,
        )
        print(f"  inference rows: {completed} -> {long_qa_infer_jsonl_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
