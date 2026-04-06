#!/usr/bin/env python3
"""Download THUDM/LongBench-v2 from Hugging Face.

This script downloads the public `data.json` file, validates the basic schema,
and writes a small summary JSON next to the downloaded dataset.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


HF_DATASET_URL = "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download THUDM/LongBench-v2 and save it locally."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/longbench_v2"),
        help="Directory used to store downloaded files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="data.json",
        help="Downloaded dataset filename.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists.",
    )
    return parser.parse_args()


def download_json(url: str, output_path: Path) -> list[dict[str, Any]]:
    try:
        with urlopen(url, timeout=120) as response:
            raw_bytes = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download dataset from {url}: {exc}") from exc

    output_path.write_bytes(raw_bytes)

    try:
        obj = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Downloaded file is not valid JSON: {output_path} ({exc})"
        ) from exc

    return validate_rows(obj)


def validate_rows(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected top-level list, got: {type(obj)!r}")

    required_keys = {
        "_id",
        "context",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "domain",
        "difficulty",
        "length",
    }

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            raise RuntimeError(f"Row {idx} is not a dict: {type(item)!r}")
        missing = required_keys - set(item.keys())
        if missing:
            raise RuntimeError(f"Row {idx} missing keys: {sorted(missing)}")
        rows.append(item)
    return rows


def build_summary(rows: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    domain_counts = Counter(str(row.get("domain", "unknown")) for row in rows)
    difficulty_counts = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    length_counts = Counter(str(row.get("length", "unknown")) for row in rows)
    context_chars = [len(str(row.get("context", ""))) for row in rows]

    sorted_chars = sorted(context_chars)
    p90_idx = min(len(sorted_chars) - 1, int(len(sorted_chars) * 0.9))
    p95_idx = min(len(sorted_chars) - 1, int(len(sorted_chars) * 0.95))

    return {
        "source_url": HF_DATASET_URL,
        "dataset_path": str(dataset_path),
        "rows": len(rows),
        "domain_counts": dict(domain_counts),
        "difficulty_counts": dict(difficulty_counts),
        "length_counts": dict(length_counts),
        "context_chars": {
            "min": min(context_chars),
            "p50": sorted_chars[len(sorted_chars) // 2],
            "p90": sorted_chars[p90_idx],
            "p95": sorted_chars[p95_idx],
            "max": max(context_chars),
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / args.output_name
    summary_path = output_dir / "download_summary.json"

    if dataset_path.exists() and not args.force:
        print(f"[Info] Reusing existing dataset: {dataset_path}")
        rows = validate_rows(json.loads(dataset_path.read_text(encoding="utf-8")))
    else:
        print(f"[Info] Downloading dataset from: {HF_DATASET_URL}")
        rows = download_json(HF_DATASET_URL, dataset_path)
        print(f"[Info] Dataset saved to: {dataset_path}")

    summary = build_summary(rows, dataset_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("[Done] Download completed.")
    print(f"  rows: {summary['rows']}")
    print(f"  dataset: {dataset_path}")
    print(f"  summary: {summary_path}")
    print(f"  domains: {summary['domain_counts']}")
    print(f"  context_chars: {summary['context_chars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
