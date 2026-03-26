#!/usr/bin/env python3
"""Plot HiSparse hit-rate JSONL metrics into SVG line charts.

This script reads multiple JSONL files, extracts ``step``, ``hit_rate``, and
``miss_count``, optionally truncates to the first N steps, averages every K
steps, and writes two SVG charts.
"""

import argparse
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


DEFAULT_INPUTS = [
    Path("data/hit_rate/hisparse_hit_rate_2048.jsonl"),
    Path("data/hit_rate/hisparse_hit_rate_fifo_4096.jsonl"),
    Path("data/hit_rate/hisparse_hit_rate_random_4096.jsonl"),
    Path("data/hit_rate/hisparse_hit_rate_lru_4096.jsonl"),
]
COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot HiSparse hit_rate/miss_count curves from JSONL files into SVG "
            "charts using plain line segments."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input JSONL files. Defaults to the four HiSparse hit-rate files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hit_rate"),
        help="Directory for generated SVG files.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=1500,
        help="Keep rows with step <= max-step. Use <= 0 to keep all rows.",
    )
    parser.add_argument(
        "--avg-window",
        type=int,
        default=10,
        help="Average every K rows after filtering. Use 1 for raw lines.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="SVG width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="SVG height in pixels.",
    )
    return parser.parse_args()


def load_rows(path: Path, max_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            if max_step > 0 and int(row["step"]) > max_step:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows left after filtering: {path}")
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]], window: int
) -> tuple[list[float], list[float], list[float]]:
    if window <= 0:
        raise ValueError(f"avg-window must be positive, got {window}")

    if window == 1:
        return (
            [float(row["step"]) for row in rows],
            [float(row["hit_rate"]) for row in rows],
            [float(row["miss_count"]) for row in rows],
        )

    agg_steps: list[float] = []
    agg_hit_rate: list[float] = []
    agg_miss_count: list[float] = []

    # Preserve the exact starting point at step=1, then average later windows.
    first_row = rows[0]
    agg_steps.append(float(first_row["step"]))
    agg_hit_rate.append(float(first_row["hit_rate"]))
    agg_miss_count.append(float(first_row["miss_count"]))

    for i in range(1, len(rows), window):
        chunk = rows[i : i + window]
        agg_steps.append(sum(float(row["step"]) for row in chunk) / len(chunk))
        agg_hit_rate.append(sum(float(row["hit_rate"]) for row in chunk) / len(chunk))
        agg_miss_count.append(sum(float(row["miss_count"]) for row in chunk) / len(chunk))
    return agg_steps, agg_hit_rate, agg_miss_count


def label_from_path(path: Path) -> str:
    return path.stem.removeprefix("hisparse_hit_rate_")


def build_output_name(metric_key: str, max_step: int, avg_window: int) -> str:
    name = f"hisparse_{metric_key}_comparison"
    if max_step > 0:
        name += f"_first{max_step}"
    if avg_window > 1:
        name += f"_avg{avg_window}"
    return name + ".svg"


def metric_title(metric_key: str, max_step: int, avg_window: int) -> str:
    title = "Hit Rate" if metric_key == "hit_rate" else "Miss Count"
    suffix_parts = []
    if max_step > 0:
        suffix_parts.append(f"first {max_step} steps")
    if avg_window > 1:
        suffix_parts.append(f"avg every {avg_window} steps")
    if suffix_parts:
        title += " (" + ", ".join(suffix_parts) + ")"
    return title


def tick_values(min_x: float, max_x: float, count: int = 10) -> list[float]:
    if max_x <= min_x:
        return [min_x]
    return [min_x + (max_x - min_x) * i / count for i in range(count + 1)]


def svg_line_chart(
    metric_key: str,
    title: str,
    output_path: Path,
    series: list[tuple[str, list[float], list[float], list[float]]],
    width: int,
    height: int,
) -> None:
    left, right, top, bottom = 90, 220, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_steps = [step for _, steps, _, _ in series for step in steps]
    if metric_key == "hit_rate":
        all_vals = [value for _, _, values, _ in series for value in values]
    else:
        all_vals = [value for _, _, _, values in series for value in values]

    min_x, max_x = min(all_steps), max(all_steps)
    min_y, max_y = min(all_vals), max(all_vals)
    y_pad = (max_y - min_y) * 0.08 if max_y > min_y else 1.0
    min_y -= y_pad
    max_y += y_pad
    if metric_key == "hit_rate":
        min_y = max(0.0, min_y)
        max_y = min(1.0, max_y)

    def sx(x: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        if max_y == min_y:
            return top + plot_h / 2
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="22" '
        f'font-family="Arial">{escape(title)}</text>'
    )

    for i in range(6):
        y_val = min_y + (max_y - min_y) * i / 5
        y = sy(y_val)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        label = f"{y_val:.3f}" if metric_key == "hit_rate" else f"{y_val:.1f}"
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="14" font-family="Arial" fill="#374151">{label}</text>'
        )

    for x_val in tick_values(min_x, max_x):
        x = sx(x_val)
        tick_label = f"{int(round(x_val))}"
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" '
            f'font-size="14" font-family="Arial" fill="#374151">{tick_label}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" '
        f'font-size="16" font-family="Arial">step</text>'
    )
    parts.append(
        f'<text x="26" y="{top + plot_h / 2}" text-anchor="middle" font-size="16" '
        f'font-family="Arial" transform="rotate(-90 26 {top + plot_h / 2})">'
        f"{escape(metric_key)}</text>"
    )

    legend_x = left + plot_w + 30
    legend_y = top + 20
    for idx, (label, steps, hit_rates, miss_counts) in enumerate(series):
        values = hit_rates if metric_key == "hit_rate" else miss_counts
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, values))
        color = COLORS[idx % len(COLORS)]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        ly = legend_y + idx * 30
        parts.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 28}" y2="{ly}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 38}" y="{ly + 5}" font-size="15" '
            f'font-family="Arial" fill="#111827">{escape(label)}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, list[float], list[float], list[float]]] = []
    for input_path in args.inputs:
        rows = load_rows(input_path, args.max_step)
        steps, hit_rates, miss_counts = aggregate_rows(rows, args.avg_window)
        series.append((label_from_path(input_path), steps, hit_rates, miss_counts))

    outputs = []
    for metric_key in ("hit_rate", "miss_count"):
        output_path = args.output_dir / build_output_name(
            metric_key=metric_key,
            max_step=args.max_step,
            avg_window=args.avg_window,
        )
        svg_line_chart(
            metric_key=metric_key,
            title=metric_title(metric_key, args.max_step, args.avg_window),
            output_path=output_path,
            series=series,
            width=args.width,
            height=args.height,
        )
        outputs.append(output_path)

    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()
