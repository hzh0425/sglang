#!/usr/bin/env python3
"""Refresh est_time literals from sglang-ci-stats/model.json.

Usage:
    python scripts/ci/update_est_time.py [--dry-run] \\
        [--model-url URL] [--summary-file PATH]
"""

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/sgl-project/sglang-ci-stats/main/model.json"
)

# AMD / NPU live in separate workflows and are not scraped by sglang-ci-stats.
BACKENDS = ("cuda", "cpu")
REGISTER_FUNCTIONS = {f"register_{backend}_ci" for backend in BACKENDS}
REGISTRATION_PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")
_UNSET = object()

# A change is "significant" if |delta| >= this many seconds AND the relative
# change is at least SIGNIFICANT_REL_DELTA. Dual threshold filters out both
# tiny absolute drifts on long tests and small-but-noisy relative swings on
# short tests.
SIGNIFICANT_ABS_DELTA = 30
SIGNIFICANT_REL_DELTA = 0.3


@dataclass(frozen=True)
class RegistrationMatch:
    suite: str
    old_value: int | float
    value_start: int
    value_end: int


def fetch_model(url):
    """Curl model.json. Fail loudly on network or parse errors -- the
    weekly workflow will surface the failure rather than silently making
    a no-op PR."""
    out = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _constant_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    return _UNSET


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    return None


def _line_offsets(content):
    offsets = [0]
    total = 0
    for line in content.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _absolute_offset(line_offsets, node):
    return line_offsets[node.lineno - 1] + node.col_offset


def _absolute_end_offset(line_offsets, node):
    return line_offsets[node.end_lineno - 1] + node.end_col_offset


def _registration_from_call(node, line_offsets):
    func_name = _call_name(node.func)
    if func_name not in REGISTER_FUNCTIONS:
        return None
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    if len(node.args) > len(REGISTRATION_PARAM_ORDER):
        return None

    values = {}
    value_nodes = {}
    seen = set()
    for name, arg in zip(REGISTRATION_PARAM_ORDER, node.args):
        values[name] = _constant_value(arg)
        value_nodes[name] = arg
        seen.add(name)

    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in seen:
            return None
        values[keyword.arg] = _constant_value(keyword.value)
        value_nodes[keyword.arg] = keyword.value
        seen.add(keyword.arg)

    est_time = values.get("est_time", _UNSET)
    est_time_node = value_nodes.get("est_time")
    if not isinstance(est_time, (int, float)) or est_time_node is None:
        return None

    legacy_suite = values.get("suite", _UNSET)
    stage = values.get("stage", _UNSET)
    runner_config = values.get("runner_config", _UNSET)
    if isinstance(stage, str) and isinstance(runner_config, str):
        suite = f"{stage}-test-{runner_config}"
    elif isinstance(legacy_suite, str):
        suite = legacy_suite
    else:
        return None

    old_value = int(est_time) if float(est_time).is_integer() else est_time
    return RegistrationMatch(
        suite=suite,
        old_value=old_value,
        value_start=_absolute_offset(line_offsets, est_time_node),
        value_end=_absolute_end_offset(line_offsets, est_time_node),
    )


def find_registration_matches(content, filename="<string>"):
    """Return CI registration calls with source spans for their est_time value."""
    tree = ast.parse(content, filename=filename)
    line_offsets = _line_offsets(content)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        match = _registration_from_call(node, line_offsets)
        if match is not None:
            matches.append(match)
    return matches


def update_files(model, dry_run=False):
    """Walk `model.est`, apply each p90 to the matching register call.

    Returns list of (relpath, suite, old, new) for every changed entry.
    """
    by_file = defaultdict(list)
    for suite, files in model.get("est", {}).items():
        for relpath, p90 in files.items():
            by_file[relpath].append((suite, p90))

    changes = []
    for relpath, entries in sorted(by_file.items()):
        filepath = REPO_ROOT / relpath
        if not filepath.exists():
            continue
        content = filepath.read_text()
        new_content = content
        matches_by_suite = defaultdict(list)
        for match in find_registration_matches(content, filename=str(filepath)):
            matches_by_suite[match.suite].append(match)

        replacements = []
        used_spans = set()
        for suite, p90 in entries:
            for match in matches_by_suite.get(suite, []):
                span = (match.value_start, match.value_end)
                if span in used_spans:
                    continue
                used_spans.add(span)
                old_val = match.old_value
                if old_val != p90:
                    replacements.append((match.value_start, match.value_end, str(p90)))
                    changes.append((relpath, suite, old_val, p90))
                    print(
                        f"  {relpath}: suite={suite!r} " f"est_time {old_val} -> {p90}",
                        file=sys.stderr,
                    )
                break  # one (file, suite) -> at most one register call

        for start, end, replacement in sorted(replacements, reverse=True):
            new_content = new_content[:start] + replacement + new_content[end:]

        if new_content != content and not dry_run:
            filepath.write_text(new_content)

    return changes


def is_significant(old, new):
    delta = abs(new - old)
    return (
        delta >= SIGNIFICANT_ABS_DELTA and delta / max(old, 1) >= SIGNIFICANT_REL_DELTA
    )


def write_summary(changes, summary_file):
    """Write a markdown summary of significant est_time changes."""
    sig = [c for c in changes if is_significant(c[2], c[3])]
    sig.sort(key=lambda c: abs(c[3] - c[2]), reverse=True)

    lines = []
    if sig:
        lines.append(
            f"### Significant est_time changes "
            f"({len(sig)} of {len(changes)} updates)"
        )
        lines.append("")
        lines.append("| File | Suite | Old (s) | New (s) | Δ |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for relpath, suite, old, new in sig:
            delta = new - old
            sign = "+" if delta > 0 else ""
            pct = round(delta / max(old, 1) * 100)
            lines.append(
                f"| `{Path(relpath).name}` | `{suite}` | "
                f"{old} | {new} | {sign}{delta} ({sign}{pct}%) |"
            )
    else:
        lines.append(
            f"_{len(changes)} est_time update(s); none exceeded both "
            f"±{SIGNIFICANT_ABS_DELTA}s and "
            f"±{int(SIGNIFICANT_REL_DELTA * 100)}% thresholds._"
        )

    Path(summary_file).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="URL of model.json from sglang-ci-stats (file:// is OK for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without modifying files",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write a markdown summary of significant changes to this path",
    )
    args = parser.parse_args()

    print(f"Fetching {args.model_url}", file=sys.stderr)
    model = fetch_model(args.model_url)
    print(
        f"  model data_as_of={model.get('data_as_of')} "
        f"n_runs={model.get('n_runs')} "
        f"n_suites={len(model.get('est', {}))}",
        file=sys.stderr,
    )

    changes = update_files(model, dry_run=args.dry_run)

    n_files = len({c[0] for c in changes})
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"\n{action} {len(changes)} est_time entries across {n_files} files",
        file=sys.stderr,
    )

    if args.summary_file:
        write_summary(changes, args.summary_file)
        print(f"Wrote summary to {args.summary_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
