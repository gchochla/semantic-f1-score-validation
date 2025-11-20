#!/usr/bin/env python3
"""
Compare DEMUX experiments with different early stopping criteria.

Reads `metrics.yml` from logs under `logs/DEMUX/<Dataset>/` and `logs/DEMUX_2/<Dataset>/`
for experiments with different early stopping criteria:
  - samples_f1/semantic_samples_f1 pairs
  - micro_f1/semantic_micro_f1 pairs

Combines results across multiple seeds/experiments with the same configuration.

Outputs in `logs/analysis/semantic_f1/study_e/`:
  - tables/<Dataset>_<es_type>_comparison.csv: side-by-side means/95%CIs + p-values per metric
  - tables/all_datasets_comparison.csv: long-format table across datasets and ES types (includes p and significance)
  - figures/<Dataset>_<es_type>_dev_comparison.png|.pdf: grouped bars of key dev metrics
  - figures/<Dataset>_<es_type>_test_comparison.png|.pdf: grouped bars of key test metrics
  - figures/<Dataset>_<es_type>_dev_comparison_faceted.png|.pdf: per-metric panels (each pair has its own basis)
  - figures/<Dataset>_<es_type>_test_comparison_faceted.png|.pdf: per-metric panels (each pair has its own basis)
  - figures/<Dataset>_<es_type>_{dev|test}_comparison_slope.png|.pdf: slope chart connecting ES conditions

Notes:
  - Aggregates metric values from individual experiments in metrics.yml files across multiple
    directories with the same configuration.
  - Computes 95% confidence intervals and uses them for error bars and z-tests for p-values.
  - Significance is annotated with *, **, *** at p<0.05, 0.01, 0.001 respectively.
  - We include all metrics found in the YAML files. Plots focus on
    common summary metrics:
      ['micro_f1','macro_f1','samples_f1','jaccard_score',
       'semantic_micro_f1','semantic_macro_f1','semantic_samples_f1']
  - Different early stopping metric types are processed separately and labeled accordingly.
"""


import os
import sys
import argparse
from pathlib import Path
import math

import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR


def parse_mean_ci(value) -> tuple[float, float]:
    """Parse a value like "0.123+-0.045" or a plain float to (mean, ci_halfwidth)."""
    if isinstance(value, (int, float)):
        return float(value), 0.0
    if isinstance(value, str):
        # Expect format like "0.1459+-0.026"; be robust to spaces
        parts = value.replace(" ", "").split("+-")
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        # Fallback: try to cast as float
        try:
            return float(value), 0.0
        except ValueError:
            return float("nan"), float("nan")
    # Unhandled type
    return float("nan"), float("nan")


def load_aggregated_metrics(path: Path) -> dict[str, tuple[float, float]]:
    """Load aggregated_metrics.yml into a dict: metric -> (mean, ci_halfwidth)."""
    with path.open("r") as f:
        data = yaml.safe_load(f)
    # Files appear as mapping with key '' mapping to the metrics dict.
    if (
        isinstance(data, dict)
        and len(data) == 1
        and ('' in data or None in data)
    ):
        metrics_map = data.get('', data.get(None, {}))
    else:
        metrics_map = data

    parsed = {}
    for k, v in metrics_map.items():
        parsed[k] = parse_mean_ci(v)
    return parsed


def load_and_aggregate_metrics(
    dirs: list[Path],
) -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """Load metrics from multiple metrics.yml files and aggregate them.

    Returns:
        - metrics_dict: metric -> (mean, ci_halfwidth)
        - per_exp_values: metric -> list of values for p-value calculation
    """
    import numpy as np

    all_values: dict[str, list[float]] = {}

    # Collect all experiment values from all directories
    for exp_dir in dirs:
        metrics_path = exp_dir / "metrics.yml"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r") as f:
            data = yaml.safe_load(f) or {}

        # Extract values from each experiment in this file
        for exp_key, exp_data in data.items():
            if not str(exp_key).startswith("experiment_"):
                continue
            if not isinstance(exp_data, dict):
                continue

            for metric_name, metric_value in exp_data.items():
                # Only aggregate "best_dev_" and "test_" metrics (skip training curves)
                if not (
                    metric_name.startswith("best_dev_")
                    or metric_name.startswith("test_")
                ):
                    continue
                if isinstance(metric_value, (int, float)) and not isinstance(
                    metric_value, bool
                ):
                    all_values.setdefault(metric_name, []).append(
                        float(metric_value)
                    )

    # Compute aggregated statistics
    metrics_dict = {}
    for metric, values in all_values.items():
        if not values:
            continue
        values_arr = np.array(values)
        mean_val = float(np.mean(values_arr))
        if len(values) > 1:
            # 95% CI half-width using t-distribution approximation
            std_val = float(np.std(values_arr, ddof=1))
            n = len(values)
            # For small n, use t-critical ~= 2.0; for large n approaches 1.96
            t_critical = 2.0 if n <= 30 else 1.96
            ci_half = t_critical * std_val / np.sqrt(n)
        else:
            ci_half = 0.0
        metrics_dict[metric] = (mean_val, ci_half)

    return metrics_dict, all_values


def find_run_dirs(
    dataset_dirs: list[Path],
) -> dict[str, tuple[list[Path], list[Path]]]:
    """Find run directories across multiple dataset directories.

    Returns dict mapping early stopping type to (baseline_dirs, semantic_dirs).
    Early stopping types: 'samples_f1', 'micro_f1'
    """
    es_type_to_runs = {}

    for ds_dir in dataset_dirs:
        if not ds_dir.exists() or not ds_dir.is_dir():
            continue

        for child in ds_dir.iterdir():
            if not child.is_dir():
                continue

            name = child.name

            # Detect early stopping type and baseline vs semantic
            if "-semantic_micro_f1-" in name:
                es_type = "micro_f1"
                is_semantic = True
            elif "-micro_f1-" in name and "-semantic_micro_f1-" not in name:
                es_type = "micro_f1"
                is_semantic = False
            elif "-semantic_samples_f1-" in name:
                es_type = "samples_f1"
                is_semantic = True
            elif "-samples_f1-" in name and "-semantic_samples_f1-" not in name:
                es_type = "samples_f1"
                is_semantic = False
            else:
                continue

            # Initialize entry if needed
            if es_type not in es_type_to_runs:
                es_type_to_runs[es_type] = ([], [])

            if is_semantic:
                es_type_to_runs[es_type][1].append(child)
            else:
                es_type_to_runs[es_type][0].append(child)

    return es_type_to_runs


def find_run_dirs_legacy(dataset_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (samples_f1_dir, semantic_samples_f1_dir) if found in dataset_dir."""
    samples = None
    semantic = None
    for child in dataset_dir.iterdir():
        name = child.name
        if not child.is_dir():
            continue
        if "-semantic_samples_f1-" in name:
            semantic = child
        elif "-samples_f1-" in name:
            # Avoid double-counting semantic variants which also contain 'samples_f1'
            if "-semantic_samples_f1-" not in name:
                samples = child
    return samples, semantic


def _norm_p_from_z(z: float) -> float:
    """Two-sided p-value from z using erfc (no scipy)."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _welch_like_p_from_samples(a: list[float], b: list[float]) -> float:
    """Approximate two-sided p-value using normal approx from sample means.

    Falls back to z based on standard errors. Suitable when n is moderate.
    """
    if not a or not b:
        return float('nan')
    import numpy as np

    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    ma, mb = float(np.mean(a_arr)), float(np.mean(b_arr))
    sa, sb = float(np.std(a_arr, ddof=1)) if len(a_arr) > 1 else 0.0
    sb = float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0
    na, nb = max(1, len(a_arr)), max(1, len(b_arr))
    # Standard error of difference
    se = math.sqrt((sa * sa) / na + (sb * sb) / nb)
    if se == 0:
        return 1.0 if abs(mb - ma) < 1e-12 else 0.0
    z = (mb - ma) / se
    return _norm_p_from_z(z)


def _p_from_ci(
    mean_a: float,
    ci_a: float,
    mean_b: float,
    ci_b: float,
    samples_a: list[float] | None = None,
    samples_b: list[float] | None = None,
) -> float:
    """Compute two-sided p-value for difference.

    Prefer CI-based z-test using CI half-widths (95%): SE = ci/1.96.
    If CI is missing/invalid, fall back to sample-based SE using per-experiment values.
    """
    try:
        if all(math.isfinite(x) and x >= 0 for x in (ci_a, ci_b)) and not (
            ci_a == 0 and ci_b == 0
        ):
            se_a = ci_a / 1.96
            se_b = ci_b / 1.96
            se = math.sqrt(se_a * se_a + se_b * se_b)
            if se == 0:
                return 1.0 if abs(mean_b - mean_a) < 1e-12 else 0.0
            z = (mean_b - mean_a) / se
            return _norm_p_from_z(z)
    except Exception:
        pass

    # Fallback to per-experiment values
    if samples_a is not None and samples_b is not None:
        return _welch_like_p_from_samples(samples_a, samples_b)
    return float('nan')


def build_comparison_table(
    ds_name: str,
    es_type: str,
    metrics_baseline: dict[str, tuple[float, float]],
    metrics_semantic: dict[str, tuple[float, float]],
    expvals_baseline: dict[str, list[float]] | None = None,
    expvals_semantic: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Create wide table with both ES settings side-by-side (mean and CI) and p-values."""
    all_keys = sorted(
        set(metrics_baseline.keys()) | set(metrics_semantic.keys())
    )
    rows = []
    for key in all_keys:
        m_base = metrics_baseline.get(key, (float("nan"), float("nan")))
        m_sem = metrics_semantic.get(key, (float("nan"), float("nan")))
        sv_a = (expvals_baseline or {}).get(key)
        sv_b = (expvals_semantic or {}).get(key)
        p_val = _p_from_ci(m_base[0], m_base[1], m_sem[0], m_sem[1], sv_a, sv_b)
        # Significance stars
        if math.isfinite(p_val):
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = ""
        else:
            stars = ""

        # Create column names based on early stopping type
        baseline_col = f"ES={es_type}_mean"
        semantic_col = f"ES=semantic_{es_type}_mean"
        baseline_ci_col = f"ES={es_type}_ci"
        semantic_ci_col = f"ES=semantic_{es_type}_ci"

        rows.append(
            {
                "dataset": ds_name,
                "es_type": es_type,
                "metric": key,
                baseline_col: m_base[0],
                baseline_ci_col: m_base[1],
                semantic_col: m_sem[0],
                semantic_ci_col: m_sem[1],
                f"delta_mean(semantic - {es_type})": m_sem[0] - m_base[0],
                "p_value": p_val,
                "significance": stars,
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_bars(
    df: pd.DataFrame,
    ds_name: str,
    es_type: str,
    split_prefix: str,
    out_dir: Path,
    metrics_order: list[str],
):
    """Plot grouped bars for selected metrics within a split (dev/test)."""
    # Filter to desired split and strip the split prefix for clean x-labels
    plot_df = df[df["metric"].str.startswith(split_prefix)].copy()
    if plot_df.empty:
        return

    plot_df["metric_clean"] = plot_df["metric"].str[len(split_prefix) :]
    # Keep only chosen metrics
    plot_df = plot_df[plot_df["metric_clean"].isin(metrics_order)]
    if plot_df.empty:
        return

    # Order metrics as specified
    plot_df["metric_clean"] = pd.Categorical(
        plot_df["metric_clean"], categories=metrics_order, ordered=True
    )
    plot_df = plot_df.sort_values("metric_clean")

    # Pretty labels: Title Case and underscores to spaces
    def nice_label(s: str) -> str:
        return s.replace("_", " ").title()

    x = [nice_label(v) for v in plot_df["metric_clean"].tolist()]

    # Get column names based on es_type
    baseline_col = f"ES={es_type}_mean"
    semantic_col = f"ES=semantic_{es_type}_mean"
    baseline_ci_col = f"ES={es_type}_ci"
    semantic_ci_col = f"ES=semantic_{es_type}_ci"

    baseline_means = plot_df[baseline_col].tolist()
    sem_means = plot_df[semantic_col].tolist()
    baseline_cis = plot_df[baseline_ci_col].tolist()
    sem_cis = plot_df[semantic_ci_col].tolist()

    plt.figure(figsize=(10, 4))
    width = 0.38
    xs = range(len(x))
    plt.bar(
        [i - width / 2 for i in xs],
        baseline_means,
        yerr=baseline_cis,
        capsize=3,
        width=width,
        label="Hard",
        color="#4C78A8",
    )
    plt.bar(
        [i + width / 2 for i in xs],
        sem_means,
        yerr=sem_cis,
        capsize=3,
        width=width,
        label="Semantic",
        color="#F58518",
    )
    plt.xticks(list(xs), [str(m) for m in x], rotation=25, ha="right")
    plt.ylabel("Mean value")
    ds_title = ds_name.replace("_", " ").title()
    title_split = "Dev" if split_prefix.startswith("best_dev") else "Test"
    es_display = es_type.replace("_", " ").title()
    plt.title(
        f"{ds_title}: {title_split} Metrics by Early Stopping ({es_display})"
    )
    plt.legend()
    # Y-limits: consider error bars and leave space for significance stars
    lowers = [m - c for m, c in zip(baseline_means, baseline_cis)] + [
        m - c for m, c in zip(sem_means, sem_cis)
    ]
    uppers = [m + c for m, c in zip(baseline_means, baseline_cis)] + [
        m + c for m, c in zip(sem_means, sem_cis)
    ]
    y_min, y_max, span = _tight_ylim_from_ci(lowers, uppers)

    # Check if we have any significance stars and add extra space for them
    has_stars = any(plot_df.get("significance", [""] * len(x)))
    if has_stars:
        star_space = 0.15 * span  # Add 15% more space for stars
        y_max = min(1.0, y_max + star_space)

    plt.ylim(y_min, y_max)

    # Add significance stars above pairs (before tight_layout)
    ax = plt.gca()
    for i, star in enumerate(plot_df.get("significance", [""] * len(x))):
        if not star:
            continue
        pair_upper = max(
            baseline_means[i] + baseline_cis[i], sem_means[i] + sem_cis[i]
        )
        # Position stars with some padding above the highest bar+CI
        star_y = pair_upper + 0.03 * span
        # Ensure stars don't go above the plot area
        star_y = min(star_y, y_max - 0.02 * span)
        ax.text(i, star_y, star, ha="center", va="bottom", fontsize=12)

    plt.tight_layout()

    base = (
        out_dir / f"{ds_name}_{es_type}_{split_prefix.rstrip('_')}_comparison"
    )
    plt.savefig(base.with_suffix('.png'), dpi=200)
    plt.savefig(base.with_suffix('.pdf'))
    plt.close()


def _nice_label(s: str) -> str:
    return s.replace("_", " ").title()


def _tight_ylim(min_val: float, max_val: float) -> tuple[float, float]:
    # Tighten based on provided bounds and add 5% whitespace top/bottom
    y_min = max(0.0, min_val)
    y_max = min(1.0, max_val)
    span = max(1e-6, y_max - y_min)
    pad = 0.05 * span
    return max(0.0, y_min - pad), min(1.0, y_max + pad)


def _tight_ylim_from_ci(
    lowers: list[float], uppers: list[float]
) -> tuple[float, float, float]:
    """Compute y-limits from lower/upper arrays with 5% whitespace.

    Returns (y_min, y_max, span) after clamping to [0,1].
    """
    finite_l = [x for x in lowers if x is not None and math.isfinite(x)]
    finite_u = [x for x in uppers if x is not None and math.isfinite(x)]
    if not finite_l or not finite_u:
        return 0.0, 1.0, 1.0
    data_min = max(0.0, min(finite_l))
    data_max = min(1.0, max(finite_u))
    span = max(1e-6, data_max - data_min)
    pad = 0.05 * span
    y_min = max(0.0, data_min - pad)
    y_max = min(1.0, data_max + pad)
    return y_min, y_max, (y_max - y_min)


def plot_faceted_pairs(
    df: pd.DataFrame,
    ds_name: str,
    es_type: str,
    split_prefix: str,
    out_dir: Path,
    metrics_order: list[str],
):
    """Per-metric panels so each pair has its own basis (subplot-specific y-limits)."""
    plot_df = df[df["metric"].str.startswith(split_prefix)].copy()
    if plot_df.empty:
        return
    plot_df["metric_clean"] = plot_df["metric"].str[len(split_prefix) :]
    plot_df = plot_df[plot_df["metric_clean"].isin(metrics_order)]
    if plot_df.empty:
        return
    plot_df["metric_clean"] = pd.Categorical(
        plot_df["metric_clean"], categories=metrics_order, ordered=True
    )
    plot_df = plot_df.sort_values("metric_clean").reset_index(drop=True)

    k = len(plot_df)
    fig, axes = plt.subplots(
        1, k, figsize=(max(6, 2.4 * k), 3.4), sharex=False, sharey=False
    )
    if k == 1:
        axes = [axes]

    # Get column names based on es_type
    baseline_col = f"ES={es_type}_mean"
    semantic_col = f"ES=semantic_{es_type}_mean"
    baseline_ci_col = f"ES={es_type}_ci"
    semantic_ci_col = f"ES=semantic_{es_type}_ci"

    for i, (ax, (_, row)) in enumerate(zip(axes, plot_df.iterrows())):
        es_display = es_type.replace("_", " ").title()
        x = [f"ES: {es_display}", f"ES: Semantic {es_display}"]
        y = [row[baseline_col], row[semantic_col]]
        ci = [row[baseline_ci_col], row[semantic_ci_col]]
        xs = [0, 1]
        label = ["Hard", "Semantic"] if i == len(axes) - 1 else [None, None]
        ax.bar(
            xs,
            y,
            yerr=ci,
            capsize=3,
            color=[HARD_COLOR, SEM_COLOR],
            label=label,
            width=0.6,
        )
        ax.set_xticks(xs, [])
        ax.set_title(_nice_label(str(row["metric_clean"])), fontsize=15)
        lowers = [y[0] - ci[0], y[1] - ci[1]]
        uppers = [y[0] + ci[0], y[1] + ci[1]]
        ymin, ymax, span = _tight_ylim_from_ci(lowers, uppers)

        # Check if we have a significance star and adjust y-limits
        star = str(row.get("significance", ""))
        if star:
            star_space = 0.15 * span  # Add space for star
            ymax = min(1.0, ymax + star_space)

        ax.set_ylim(ymin, ymax)

        # Significance star
        if star:
            pair_upper = max(y[0] + ci[0], y[1] + ci[1])
            # Position star with padding above the highest bar+CI
            star_y = pair_upper + 0.05 * span
            # Ensure star doesn't go above the plot area
            star_y = min(star_y, ymax - 0.02 * span)
            ax.text(0.5, star_y, star, ha="center", va="bottom", fontsize=12)

    # shared legend at bottom
    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        fontsize=20,
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
    )

    ds_title = ds_name.replace("_", " ")
    title_split = "Dev" if split_prefix.startswith("best_dev") else "Test"
    es_display = es_type.replace("_", " ").title()
    fig.suptitle(
        f"{ds_title}: {title_split} Metrics by Early Stopping ({es_display})",
        fontweight='bold',
        fontsize=26,
    )
    fig.tight_layout(rect=[0, 0.12, 0.98, 1])

    base = (
        out_dir
        / f"{ds_name}_{es_type}_{split_prefix.rstrip('_')}_comparison_faceted"
    )
    fig.savefig(base.with_suffix('.png'), dpi=200)
    fig.savefig(base.with_suffix('.pdf'))
    print(f"[OK] Saved faceted figure to {base}")

    plt.close(fig)


def plot_slope_chart(
    df: pd.DataFrame,
    ds_name: str,
    es_type: str,
    split_prefix: str,
    out_dir: Path,
    metrics_order: list[str],
):
    """Slope chart connecting ES: baseline vs ES: Semantic for each metric with CIs."""
    plot_df = df[df["metric"].str.startswith(split_prefix)].copy()
    if plot_df.empty:
        return
    plot_df["metric_clean"] = plot_df["metric"].str[len(split_prefix) :]
    plot_df = plot_df[plot_df["metric_clean"].isin(metrics_order)]
    if plot_df.empty:
        return
    plot_df["metric_clean"] = pd.Categorical(
        plot_df["metric_clean"], categories=metrics_order, ordered=True
    )
    plot_df = plot_df.sort_values("metric_clean").reset_index(drop=True)

    # Get column names based on es_type
    baseline_col = f"ES={es_type}_mean"
    semantic_col = f"ES=semantic_{es_type}_mean"
    baseline_ci_col = f"ES={es_type}_ci"
    semantic_ci_col = f"ES=semantic_{es_type}_ci"

    # Build data
    names = [_nice_label(str(m)) for m in plot_df["metric_clean"].tolist()]
    baseline_means = plot_df[baseline_col].tolist()
    sem_means = plot_df[semantic_col].tolist()
    baseline_cis = plot_df[baseline_ci_col].tolist()
    sem_cis = plot_df[semantic_ci_col].tolist()

    x_positions = [0, 1]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 4), 5))
    # Pre-compute tight y-limits from CI so we can place stars safely
    lowers = [a - b for a, b in zip(baseline_means, baseline_cis)] + [
        a - b for a, b in zip(sem_means, sem_cis)
    ]
    uppers = [a + b for a, b in zip(baseline_means, baseline_cis)] + [
        a + b for a, b in zip(sem_means, sem_cis)
    ]
    ymin, ymax, span = _tight_ylim_from_ci(lowers, uppers)

    # Check if we have any significance stars and add extra space for them
    has_stars = any(plot_df.get("significance", [""] * len(plot_df)))
    if has_stars:
        star_space = 0.15 * span  # Add 15% more space for stars
        ymax = min(1.0, ymax + star_space)

    ax.set_ylim(ymin, ymax)
    for i, name in enumerate(names):
        y0, y1 = baseline_means[i], sem_means[i]
        c0, c1 = baseline_cis[i], sem_cis[i]
        color = "#F58518" if y1 >= y0 else "#4C78A8"
        ax.plot(x_positions, [y0, y1], color=color, alpha=0.8)
        ax.errorbar([x_positions[0]], [y0], yerr=[c0], fmt='o', color="#4C78A8")
        ax.errorbar([x_positions[1]], [y1], yerr=[c1], fmt='o', color="#F58518")
        ax.text(
            x_positions[1] + 0.02,
            y1,
            name,
            va='center',
            fontsize=9,
            color='black',
        )
        # Significance star near midpoint
        star = plot_df.iloc[i].get("significance", "")
        if star:
            pair_upper = max(y0 + c0, y1 + c1)
            # Position star with padding above the highest point+CI
            star_y = pair_upper + 0.05 * span
            # Ensure star doesn't go above the plot area
            star_y = min(star_y, ymax - 0.02 * span)
            ax.text(0.5, star_y, star, ha='center', va='bottom', fontsize=12)

    es_display = es_type.replace("_", " ").title()
    ax.set_xticks(
        x_positions, [f"ES: {es_display}", f"ES: Semantic {es_display}"]
    )
    ax.grid(axis='y', alpha=0.2)

    ds_title = ds_name.replace("_", " ").title()
    title_split = "Dev" if split_prefix.startswith("best_dev") else "Test"
    ax.set_title(
        f"{ds_title}: {title_split} Metrics (Slope Chart, {es_display})"
    )
    fig.tight_layout()

    base = (
        out_dir
        / f"{ds_name}_{es_type}_{split_prefix.rstrip('_')}_comparison_slope"
    )
    fig.savefig(base.with_suffix('.png'), dpi=200)
    fig.savefig(base.with_suffix('.pdf'))
    plt.close(fig)


def create_summary_table(
    all_tables: list[pd.DataFrame], tables_dir: Path
) -> None:
    """Create a summary table showing which early stopping method has higher means."""
    if not all_tables:
        return

    long_df = pd.concat(all_tables, ignore_index=True)

    # Split metric names into split and metric_name
    def split_metric(m: str) -> tuple[str, str]:
        if m.startswith("best_dev_"):
            return "dev", m[len("best_dev_") :]
        if m.startswith("test_"):
            return "test", m[len("test_") :]
        return "", m

    splits, names = zip(*[split_metric(m) for m in long_df["metric"].tolist()])
    long_df = long_df.assign(split=list(splits), metric_name=list(names))

    # Filter to only dev and test splits
    long_df = long_df[long_df["split"].isin(["dev", "test"])]

    # Filter to only micro/macro/samples F1 scores (not individual label F1s)
    f1_metrics = [
        "micro_f1",
        "macro_f1",
        "samples_f1",
        "semantic_micro_f1",
        "semantic_macro_f1",
        "semantic_samples_f1",
        "jaccard_score",
    ]
    long_df = long_df[long_df["metric_name"].isin(f1_metrics)]

    # Group by dataset, es_type, and metric_name to summarize across splits
    summary_rows = []

    for (dataset, es_type, metric_name), group in long_df.groupby(
        ["dataset", "es_type", "metric_name"]
    ):
        # Get dev and test results
        dev_data = group[group["split"] == "dev"]
        test_data = group[group["split"] == "test"]

        # Count wins for dev
        dev_semantic_wins = 0
        dev_semantic_sig = 0
        dev_hard_wins = 0
        dev_hard_sig = 0

        if not dev_data.empty:
            row = dev_data.iloc[0]
            # Get the column names dynamically based on es_type
            baseline_col = f"ES={es_type}_mean"
            semantic_col = f"ES=semantic_{es_type}_mean"

            baseline_mean = row[baseline_col]
            semantic_mean = row[semantic_col]
            is_significant = row.get("significance", "") != ""

            if semantic_mean > baseline_mean:
                dev_semantic_wins = 1
                if is_significant:
                    dev_semantic_sig = 1
            else:
                dev_hard_wins = 1
                if is_significant:
                    dev_hard_sig = 1

        # Count wins for test
        test_semantic_wins = 0
        test_semantic_sig = 0
        test_hard_wins = 0
        test_hard_sig = 0

        if not test_data.empty:
            row = test_data.iloc[0]
            baseline_col = f"ES={es_type}_mean"
            semantic_col = f"ES=semantic_{es_type}_mean"

            baseline_mean = row[baseline_col]
            semantic_mean = row[semantic_col]
            is_significant = row.get("significance", "") != ""

            if semantic_mean > baseline_mean:
                test_semantic_wins = 1
                if is_significant:
                    test_semantic_sig = 1
            else:
                test_hard_wins = 1
                if is_significant:
                    test_hard_sig = 1

        summary_rows.append(
            {
                "dataset": dataset,
                "es_type": es_type,
                "metric": metric_name,
                "dev_semantic_wins": (
                    f"{dev_semantic_wins} ({dev_semantic_sig})"
                    if dev_semantic_wins > 0
                    else "0 (0)"
                ),
                "dev_hard_wins": (
                    f"{dev_hard_wins} ({dev_hard_sig})"
                    if dev_hard_wins > 0
                    else "0 (0)"
                ),
                "test_semantic_wins": (
                    f"{test_semantic_wins} ({test_semantic_sig})"
                    if test_semantic_wins > 0
                    else "0 (0)"
                ),
                "test_hard_wins": (
                    f"{test_hard_wins} ({test_hard_sig})"
                    if test_hard_wins > 0
                    else "0 (0)"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Add totals row
    if not summary_df.empty:
        # Parse the win counts for totals
        def parse_wins(col_name: str) -> tuple[int, int]:
            total_wins = 0
            total_sig = 0
            for val in summary_df[col_name]:
                if val != "0 (0)":
                    wins, sig = val.split(" (")
                    total_wins += int(wins)
                    total_sig += int(sig.rstrip(")"))
            return total_wins, total_sig

        dev_sem_wins, dev_sem_sig = parse_wins("dev_semantic_wins")
        dev_hard_wins, dev_hard_sig = parse_wins("dev_hard_wins")
        test_sem_wins, test_sem_sig = parse_wins("test_semantic_wins")
        test_hard_wins, test_hard_sig = parse_wins("test_hard_wins")

        totals_row = pd.DataFrame(
            [
                {
                    "dataset": "TOTAL",
                    "es_type": "",
                    "metric": "",
                    "dev_semantic_wins": f"{dev_sem_wins} ({dev_sem_sig})",
                    "dev_hard_wins": f"{dev_hard_wins} ({dev_hard_sig})",
                    "test_semantic_wins": f"{test_sem_wins} ({test_sem_sig})",
                    "test_hard_wins": f"{test_hard_wins} ({test_hard_sig})",
                }
            ]
        )

        summary_df = pd.concat([summary_df, totals_row], ignore_index=True)

    # Save summary table
    summary_csv = tables_dir / "wins_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote wins summary table: {summary_csv}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY: Wins by Early Stopping Method")
    print("(Numbers show total wins, with significant wins in parentheses)")
    print("=" * 80)
    print(
        f"{'Dataset':<12} {'ES Type':<10} {'Metric':<20} {'Dev Sem':<10} {'Dev Hard':<10} {'Test Sem':<10} {'Test Hard':<10}"
    )
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<12} {row['es_type']:<10} {row['metric']:<20} {row['dev_semantic_wins']:<10} {row['dev_hard_wins']:<10} {row['test_semantic_wins']:<10} {row['test_hard_wins']:<10}"
        )


def create_latex_table(
    all_tables: list[pd.DataFrame], tables_dir: Path
) -> None:
    """Create a LaTeX table using the same aggregation logic as the console output."""
    if not all_tables:
        return

    # Use the same logic as create_summary_table to get the summary_df
    long_df = pd.concat(all_tables, ignore_index=True)

    # Split metric names into split and metric_name
    def split_metric(m: str) -> tuple[str, str]:
        if m.startswith("best_dev_"):
            return "dev", m[len("best_dev_") :]
        if m.startswith("test_"):
            return "test", m[len("test_") :]
        return "", m

    splits, names = zip(*[split_metric(m) for m in long_df["metric"].tolist()])
    long_df = long_df.assign(split=list(splits), metric_name=list(names))

    # Filter to only dev and test splits
    long_df = long_df[long_df["split"].isin(["dev", "test"])]

    # Filter to only micro/macro/samples F1 scores (not individual label F1s)
    f1_metrics = [
        "micro_f1",
        "macro_f1",
        "samples_f1",
        "semantic_micro_f1",
        "semantic_macro_f1",
        "semantic_samples_f1",
    ]
    long_df = long_df[long_df["metric_name"].isin(f1_metrics)]

    # Group by dataset, es_type, and metric_name to summarize across splits
    summary_rows = []

    for (dataset, es_type, metric_name), group in long_df.groupby(
        ["dataset", "es_type", "metric_name"]
    ):
        # Get dev and test results
        dev_data = group[group["split"] == "dev"]
        test_data = group[group["split"] == "test"]

        # Count wins for dev
        dev_semantic_wins = 0
        dev_semantic_sig = 0
        dev_hard_wins = 0
        dev_hard_sig = 0

        if not dev_data.empty:
            row = dev_data.iloc[0]
            # Get the column names dynamically based on es_type
            baseline_col = f"ES={es_type}_mean"
            semantic_col = f"ES=semantic_{es_type}_mean"

            baseline_mean = row[baseline_col]
            semantic_mean = row[semantic_col]
            is_significant = row.get("significance", "") != ""

            if semantic_mean > baseline_mean:
                dev_semantic_wins = 1
                if is_significant:
                    dev_semantic_sig = 1
            else:
                dev_hard_wins = 1
                if is_significant:
                    dev_hard_sig = 1

        # Count wins for test
        test_semantic_wins = 0
        test_semantic_sig = 0
        test_hard_wins = 0
        test_hard_sig = 0

        if not test_data.empty:
            row = test_data.iloc[0]
            baseline_col = f"ES={es_type}_mean"
            semantic_col = f"ES=semantic_{es_type}_mean"

            baseline_mean = row[baseline_col]
            semantic_mean = row[semantic_col]
            is_significant = row.get("significance", "") != ""

            if semantic_mean > baseline_mean:
                test_semantic_wins = 1
                if is_significant:
                    test_semantic_sig = 1
            else:
                test_hard_wins = 1
                if is_significant:
                    test_hard_sig = 1

        summary_rows.append(
            {
                "dataset": dataset,
                "es_type": es_type,
                "metric": metric_name,
                "dev_semantic_wins": (
                    f"{dev_semantic_wins} ({dev_semantic_sig})"
                    if dev_semantic_wins > 0
                    else "0 (0)"
                ),
                "dev_hard_wins": (
                    f"{dev_hard_wins} ({dev_hard_sig})"
                    if dev_hard_wins > 0
                    else "0 (0)"
                ),
                "test_semantic_wins": (
                    f"{test_semantic_wins} ({test_semantic_sig})"
                    if test_semantic_wins > 0
                    else "0 (0)"
                ),
                "test_hard_wins": (
                    f"{test_hard_wins} ({test_hard_sig})"
                    if test_hard_wins > 0
                    else "0 (0)"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Calculate totals using the same logic as create_summary_table
    if not summary_df.empty:
        # Parse the win counts for totals
        def parse_wins(col_name: str) -> tuple[int, int]:
            total_wins = 0
            total_sig = 0
            for val in summary_df[col_name]:
                if val != "0 (0)":
                    wins, sig = val.split(" (")
                    total_wins += int(wins)
                    total_sig += int(sig.rstrip(")"))
            return total_wins, total_sig

        dev_sem_wins, dev_sem_sig = parse_wins("dev_semantic_wins")
        dev_hard_wins, dev_hard_sig = parse_wins("dev_hard_wins")
        test_sem_wins, test_sem_sig = parse_wins("test_semantic_wins")
        test_hard_wins, test_hard_sig = parse_wins("test_hard_wins")

    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(
        "\\caption{Early Stopping Method Comparison: Wins Across All F1 Metrics}"
    )
    latex_content.append("\\label{tab:early_stopping_wins}")
    latex_content.append("\\begin{tabular}{llcccc}")
    latex_content.append("\\toprule")
    latex_content.append(
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{ES Type} & \\multicolumn{2}{c}{Dev} & \\multicolumn{2}{c}{Test} \\\\"
    )
    latex_content.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    latex_content.append(" & & Semantic & Hard & Semantic & Hard \\\\")
    latex_content.append("\\midrule")

    # Group by dataset and es_type for display
    for (dataset, es_type), group in summary_df.groupby(["dataset", "es_type"]):
        # Aggregate wins across all metrics for this dataset-es_type combination
        ds_dev_sem_wins = sum(
            1 for val in group["dev_semantic_wins"] if val != "0 (0)"
        )
        ds_dev_sem_sig = sum(
            int(val.split(" (")[1].rstrip(")"))
            for val in group["dev_semantic_wins"]
            if val != "0 (0)"
        )
        ds_dev_hard_wins = sum(
            1 for val in group["dev_hard_wins"] if val != "0 (0)"
        )
        ds_dev_hard_sig = sum(
            int(val.split(" (")[1].rstrip(")"))
            for val in group["dev_hard_wins"]
            if val != "0 (0)"
        )
        ds_test_sem_wins = sum(
            1 for val in group["test_semantic_wins"] if val != "0 (0)"
        )
        ds_test_sem_sig = sum(
            int(val.split(" (")[1].rstrip(")"))
            for val in group["test_semantic_wins"]
            if val != "0 (0)"
        )
        ds_test_hard_wins = sum(
            1 for val in group["test_hard_wins"] if val != "0 (0)"
        )
        ds_test_hard_sig = sum(
            int(val.split(" (")[1].rstrip(")"))
            for val in group["test_hard_wins"]
            if val != "0 (0)"
        )

        # Format dataset and es_type
        dataset_clean = dataset
        es_type_clean = es_type.replace(
            "_", "\\_"
        )  # Escape underscores for LaTeX

        # Format as "X (Y)" where X is total wins, Y is significant wins
        dev_sem = f"{ds_dev_sem_wins} ({ds_dev_sem_sig})"
        dev_hard = f"{ds_dev_hard_wins} ({ds_dev_hard_sig})"
        test_sem = f"{ds_test_sem_wins} ({ds_test_sem_sig})"
        test_hard = f"{ds_test_hard_wins} ({ds_test_hard_sig})"

        latex_content.append(
            f"{dataset_clean} & {es_type_clean} & {dev_sem} & {dev_hard} & {test_sem} & {test_hard} \\\\"
        )

    # Add totals row using the same calculation as console output
    if not summary_df.empty:
        latex_content.append("\\midrule")
        latex_content.append(
            f"\\textbf{{Total}} & & \\textbf{{{dev_sem_wins} ({dev_sem_sig})}} & \\textbf{{{dev_hard_wins} ({dev_hard_sig})}} & \\textbf{{{test_sem_wins} ({test_sem_sig})}} & \\textbf{{{test_hard_wins} ({test_hard_sig})}} \\\\"
        )

    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\begin{tablenotes}")
    latex_content.append("\\small")
    latex_content.append(
        "\\item Numbers show total wins across all F1 metrics (micro, macro, samples, and their semantic variants)."
    )
    latex_content.append(
        "\\item Significant wins (p < 0.05) are shown in parentheses."
    )
    latex_content.append(
        "\\item ES Type: Early stopping metric used (samples\\_f1 or micro\\_f1)."
    )
    latex_content.append("\\end{tablenotes}")
    latex_content.append("\\end{table}")

    # Write LaTeX file
    latex_file = tables_dir / "wins_summary.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"[OK] Wrote LaTeX table: {latex_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DEMUX early stopping settings."
    )
    parser.add_argument(
        "--input_roots",
        type=str,
        nargs="*",
        default=["logs/DEMUX", "logs/DEMUX_2"],
        help="Root directories containing dataset subfolders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="logs/analysis/semantic_f1/study_e",
        help="Output directory for tables and figures.",
    )
    parser.add_argument(
        "--plot_metrics",
        type=str,
        nargs="*",
        default=[
            "micro_f1",
            "macro_f1",
            "samples_f1",
            "jaccard_score",
            "semantic_micro_f1",
            "semantic_macro_f1",
            "semantic_samples_f1",
        ],
        help="Which metric names (without split prefix) to plot in figures.",
    )
    args = parser.parse_args()

    input_roots = [Path(root) for root in args.input_roots]
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_tables: list[pd.DataFrame] = []

    # Collect all dataset names across all roots
    all_datasets = set()
    for root in input_roots:
        if root.exists() and root.is_dir():
            all_datasets.update(d.name for d in root.iterdir() if d.is_dir())

    for ds_name in sorted(all_datasets):
        print(f"\n[INFO] Processing dataset: {ds_name}")

        # Collect dataset directories from all roots
        dataset_dirs = []
        for root in input_roots:
            ds_dir = root / ds_name
            if ds_dir.exists() and ds_dir.is_dir():
                dataset_dirs.append(ds_dir)

        if not dataset_dirs:
            print(f"[WARN] Skipping {ds_name}: not found in any input root")
            continue

        # Find all run directories and group by early stopping type
        es_type_to_runs = find_run_dirs(dataset_dirs)

        if not es_type_to_runs:
            print(f"[WARN] Skipping {ds_name}: no valid experiment runs found")
            continue

        # Process each early stopping type separately
        for es_type, (baseline_dirs, semantic_dirs) in es_type_to_runs.items():
            if not baseline_dirs or not semantic_dirs:
                print(
                    f"[WARN] Skipping {ds_name} {es_type}: incomplete pair (baseline: {len(baseline_dirs)}, semantic: {len(semantic_dirs)})"
                )
                continue

            print(
                f"[INFO] Processing {ds_name} with early stopping type: {es_type}"
            )
            print(
                f"[INFO] Baseline dirs: {len(baseline_dirs)}, Semantic dirs: {len(semantic_dirs)}"
            )

            # Load and aggregate metrics from all directories
            m_baseline, exp_baseline = load_and_aggregate_metrics(baseline_dirs)
            m_semantic, exp_semantic = load_and_aggregate_metrics(semantic_dirs)

            if not m_baseline or not m_semantic:
                print(f"[WARN] Skipping {ds_name} {es_type}: no metrics found")
                continue

            table = build_comparison_table(
                ds_name,
                es_type,
                m_baseline,
                m_semantic,
                exp_baseline,
                exp_semantic,
            )
            all_tables.append(table)

            # Save per-dataset-and-es-type table
            out_csv = tables_dir / f"{ds_name}_{es_type}_comparison.csv"
            table.to_csv(out_csv, index=False)
            print(f"[OK] Wrote table: {out_csv}")

            # Plots for dev and test splits
            for split_prefix in ("best_dev_", "test_"):
                plot_grouped_bars(
                    table,
                    ds_name,
                    es_type,
                    split_prefix=split_prefix,
                    out_dir=figures_dir,
                    metrics_order=args.plot_metrics,
                )
                # Additional views to emphasize small differences
                plot_faceted_pairs(
                    table,
                    ds_name,
                    es_type,
                    split_prefix=split_prefix,
                    out_dir=figures_dir,
                    metrics_order=args.plot_metrics,
                )
                plot_slope_chart(
                    table,
                    ds_name,
                    es_type,
                    split_prefix=split_prefix,
                    out_dir=figures_dir,
                    metrics_order=args.plot_metrics,
                )

    if all_tables:
        long_df = pd.concat(all_tables, ignore_index=True)

        # Also save a long-form with split and metric name separated
        def split_metric(m: str) -> tuple[str, str]:
            if m.startswith("best_dev_"):
                return "dev", m[len("best_dev_") :]
            if m.startswith("test_"):
                return "test", m[len("test_") :]
            return "", m

        splits, names = zip(
            *[split_metric(m) for m in long_df["metric"].tolist()]
        )
        long_df = long_df.assign(split=list(splits), metric_name=list(names))

        all_csv = tables_dir / "all_datasets_comparison.csv"
        long_df.to_csv(all_csv, index=False)
        print(f"[OK] Wrote combined table: {all_csv}")

        # Create summary table of wins
        create_summary_table(all_tables, tables_dir)

        # Create LaTeX table of wins
        create_latex_table(all_tables, tables_dir)
    else:
        print(
            "[INFO] No datasets processed. Check input root directories structure."
        )


if __name__ == "__main__":
    main()
