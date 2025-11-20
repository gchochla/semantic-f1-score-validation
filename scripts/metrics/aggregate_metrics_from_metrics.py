"""Aggregate metrics.yml files into aggregated_metrics.yml with mean ± 95% CI."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping

import yaml

CI_Z = 1.96  # 95% confidence for large samples


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-experiment metrics.yml files into aggregated_metrics.yml "
            "using mean ± 95% confidence intervals."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Directories (or globs) that contain metrics.yml files. If a metrics.yml "
            "file path is given, its parent directory is processed."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip directories where aggregated_metrics.yml already exists.",
    )
    parser.add_argument(
        "--ci-z",
        type=float,
        default=CI_Z,
        help="Z-score multiplier for the confidence interval (default: 1.96).",
    )
    return parser.parse_args(list(argv))


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gather_metric_values(metrics_data: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for exp_key, exp_metrics in metrics_data.items():
        if not isinstance(exp_metrics, Mapping):
            continue
        description = str(exp_metrics.get("description", ""))
        metrics_for_desc = grouped.setdefault(description, {})
        for metric_name, metric_val in exp_metrics.items():
            if metric_name == "description":
                continue
            numeric_val = coerce_float(metric_val)
            if numeric_val is None:
                continue
            metrics_for_desc.setdefault(metric_name, []).append(numeric_val)
    return grouped


def format_mean_ci(values: List[float], z_score: float) -> str:
    if not values:
        return "0.0000+-0.0000"
    mu = mean(values)
    if len(values) == 1:
        hw = 0.0
    else:
        std = pstdev(values)  # population std to match existing convention
        hw = z_score * std / math.sqrt(len(values))
    return f"{mu:.4f}+-{hw:.4f}"


def aggregate_metrics(metrics_data: Mapping[str, Mapping[str, Any]], z_score: float) -> Dict[str, Dict[str, str]]:
    aggregated: Dict[str, Dict[str, str]] = {}
    grouped = gather_metric_values(metrics_data)
    for description, metric_values in grouped.items():
        agg_for_desc: Dict[str, str] = {}
        for metric_name, values in metric_values.items():
            agg_for_desc[metric_name] = format_mean_ci(values, z_score)
        aggregated[description] = agg_for_desc
    return aggregated


def load_metrics_file(path: str) -> Mapping[str, Mapping[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping at root of {path}, got {type(data).__name__}")
    return data


def find_target_dirs(path_patterns: Iterable[str]) -> List[str]:
    dirs: List[str] = []
    for pattern in path_patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            expanded = [pattern]
        for path in expanded:
            if path.endswith("metrics.yml") and os.path.isfile(path):
                candidate = os.path.dirname(path)
            else:
                candidate = path
            if not os.path.isdir(candidate):
                print(f"[WARN] Skipping non-directory path: {path}", file=sys.stderr)
                continue
            metrics_path = os.path.join(candidate, "metrics.yml")
            if not os.path.exists(metrics_path):
                print(f"[WARN] No metrics.yml found in {candidate}; skipping.", file=sys.stderr)
                continue
            dirs.append(os.path.abspath(candidate))
    return sorted(set(dirs))


def write_aggregated_metrics(directory: str, aggregated: Mapping[str, Mapping[str, str]]) -> None:
    output_path = os.path.join(directory, "aggregated_metrics.yml")
    with open(output_path, "w") as f:
        yaml.safe_dump(aggregated, f, sort_keys=True)


def process_directory(directory: str, skip_existing: bool, z_score: float) -> None:
    metrics_path = os.path.join(directory, "metrics.yml")
    agg_path = os.path.join(directory, "aggregated_metrics.yml")

    if skip_existing and os.path.exists(agg_path):
        print(f"[INFO] Skipping {directory}; aggregated_metrics.yml already exists.")
        return

    metrics_data = load_metrics_file(metrics_path)
    aggregated = aggregate_metrics(metrics_data, z_score)
    if not aggregated:
        print(f"[WARN] No valid metric values found in {metrics_path}; skipping write.")
        return

    write_aggregated_metrics(directory, aggregated)
    print(f"[OK] Wrote aggregated metrics: {agg_path}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    target_dirs = find_target_dirs(args.paths)
    if not target_dirs:
        print("[ERROR] No directories with metrics.yml files found.", file=sys.stderr)
        return 1

    for directory in target_dirs:
        try:
            process_directory(directory, args.skip_existing, args.ci_z)
        except Exception as exc:
            print(f"[ERROR] Failed to process {directory}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
