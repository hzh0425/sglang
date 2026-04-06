#!/usr/bin/env python3
"""Filter local LongBench-v2 data into long-context subsets.

This script creates:
1) an official long subset using the dataset's `length` label, and
2) a percentile-based ultra-long subset using context character length.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create filtered long-context subsets from local LongBench-v2 data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/longbench_v2/data.json"),
        help="Local LongBench-v2 JSON/JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/longbench_v2"),
        help="Directory used to store filtered outputs.",
    )
    parser.add_argument(
        "--official-long-label",
        type=str,
        default="long",
        help="Value in the `length` field treated as the official long split.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=90,
        help="Percentile threshold for the ultra-long subset, based on context chars.",
    )
    parser.add_argument(
        "--min-context-chars",
        type=int,
        default=0,
        help="Optional absolute lower bound on context chars for the ultra-long subset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for filtered subsets. Set <=0 to disable.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to parse JSONL line {line_no} in {path}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(f"Expected dict row at line {line_no}, got {type(row)!r}")
                rows.append(row)
        return rows

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected top-level list in {path}, got {type(obj)!r}")
    rows = []
    for idx, row in enumerate(obj):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected dict row at index {idx}, got {type(row)!r}")
        rows.append(row)
    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile_threshold(rows: list[dict[str, Any]], percentile: int) -> int:
    if not 0 <= percentile <= 100:
        raise ValueError("--percentile must be between 0 and 100")
    context_chars = sorted(len(str(row.get("context", ""))) for row in rows)
    idx = min(len(context_chars) - 1, int(len(context_chars) * (percentile / 100.0)))
    return context_chars[idx]


def maybe_cap(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return rows
    return rows[:max_samples]


def build_summary(
    input_path: Path,
    total_rows: int,
    official_rows: list[dict[str, Any]],
    official_long_label: str,
    ultra_rows: list[dict[str, Any]],
    ultra_threshold: int,
    percentile: int,
    min_context_chars: int,
    max_samples: int,
) -> dict[str, Any]:
    ultra_domain_counts = Counter(str(row.get("domain", "unknown")) for row in ultra_rows)
    official_domain_counts = Counter(str(row.get("domain", "unknown")) for row in official_rows)
    return {
        "input_path": str(input_path),
        "total_rows": total_rows,
        "official_long_label": official_long_label,
        "official_long_rows": len(official_rows),
        "official_long_domain_counts": dict(official_domain_counts),
        "ultra_long_rule": {
            "percentile": percentile,
            "percentile_threshold_context_chars": ultra_threshold,
            "min_context_chars": min_context_chars,
            "effective_threshold": max(ultra_threshold, min_context_chars),
            "max_samples": max_samples if max_samples > 0 else None,
        },
        "ultra_long_rows": len(ultra_rows),
        "ultra_long_domain_counts": dict(ultra_domain_counts),
    }


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda row: len(str(row.get("context", ""))), reverse=True)
    official_rows = [
        row for row in sorted_rows if str(row.get("length", "")) == args.official_long_label
    ]
    official_rows = maybe_cap(official_rows, args.max_samples)

    ultra_threshold = percentile_threshold(sorted_rows, args.percentile)
    effective_threshold = max(ultra_threshold, args.min_context_chars)
    ultra_rows = [
        row
        for row in sorted_rows
        if len(str(row.get("context", ""))) >= effective_threshold
    ]
    ultra_rows = maybe_cap(ultra_rows, args.max_samples)

    official_json = output_dir / "long_only.json"
    official_jsonl = output_dir / "long_only.jsonl"
    ultra_json = output_dir / f"ultra_long_p{args.percentile}.json"
    ultra_jsonl = output_dir / f"ultra_long_p{args.percentile}.jsonl"
    summary_path = output_dir / "subset_summary.json"

    write_json(official_rows, official_json)
    write_jsonl(official_rows, official_jsonl)
    write_json(ultra_rows, ultra_json)
    write_jsonl(ultra_rows, ultra_jsonl)

    summary = build_summary(
        input_path=args.input_path,
        total_rows=len(rows),
        official_rows=official_rows,
        official_long_label=args.official_long_label,
        ultra_rows=ultra_rows,
        ultra_threshold=ultra_threshold,
        percentile=args.percentile,
        min_context_chars=args.min_context_chars,
        max_samples=args.max_samples,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Filtering completed.")
    print(f"  input_rows: {len(rows)}")
    print(f"  official_long_rows: {len(official_rows)} -> {official_json}")
    print(
        "  ultra_long_rows: "
        f"{len(ultra_rows)} "
        f"(threshold={effective_threshold}, percentile=p{args.percentile}) "
        f"-> {ultra_json}"
    )
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
