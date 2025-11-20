from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

# Import study_a1 for utility functions and colors
try:
    import scripts.semantic_f1.study_a1 as study_a1
    from scripts.semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR
except Exception:
    # Fallback: import sibling when executed directly via path
    sys.path.append(os.path.dirname(__file__))
    import study_a1  # type: ignore

    try:
        from __init__ import HARD_COLOR, SEM_COLOR, THIRD_COLOR
    except Exception:
        # Final fallback to manual colors
        HARD_COLOR = '#d62728'  # red
        SEM_COLOR = '#2ca02c'  # green
        THIRD_COLOR = '#ff7f0e'  # orange


def bootstrap_metric_comparison(
    gold_labels: list[list[str]],
    pred_labels: list[list[str]],
    S_matrix: pd.DataFrame,
    metric_1_name: str,
    metric_2_name: str,
    B: int = 200,
    seed: int = 123,
) -> tuple[float, float, float, float, float, float]:
    """Bootstrap confidence intervals for two semantic metrics and their gap.

    Args:
        gold_labels: Gold standard labels per example
        pred_labels: Predicted labels per example
        S_matrix: Similarity matrix
        metric_1_name: Name of first metric ("semantic_f1", "semantic_precision", etc.)
        metric_2_name: Name of second metric
        B: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Tuple of (metric1_mean, metric1_lo, metric1_hi, metric2_mean, metric2_lo, metric2_hi)
    """
    from semantic_f1_score.semantic_f1 import samples_semantic_f1_score
    from semantic_f1_score import hungarian_score

    rng = np.random.default_rng(seed)
    n = len(gold_labels)
    base_idx = np.arange(n)

    metric1_vals = []
    metric2_vals = []

    for _ in range(B):
        # Bootstrap sample
        idx = rng.choice(base_idx, size=n, replace=True)
        gold_boot = [gold_labels[i] for i in idx]
        pred_boot = [pred_labels[i] for i in idx]

        # Compute metrics
        if metric_1_name == "semantic_f1":
            metric1_val = float(
                samples_semantic_f1_score(pred_boot, gold_boot, S_matrix)
            )
        elif metric_1_name == "semantic_precision":
            result = samples_semantic_f1_score(
                pred_boot, gold_boot, S_matrix, return_components=True
            )
            metric1_val = float(result['precision'])
        elif metric_1_name == "semantic_recall":
            result = samples_semantic_f1_score(
                pred_boot, gold_boot, S_matrix, return_components=True
            )
            metric1_val = float(result['recall'])
        elif metric_1_name == "hungarian_score":
            # Suppress Hungarian debug output
            import io

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                metric1_val = float(
                    hungarian_score(pred_boot, gold_boot, S_matrix)
                )
            finally:
                sys.stdout = old_stdout
        else:
            raise ValueError(f"Unknown metric: {metric_1_name}")

        if metric_2_name == "semantic_f1":
            metric2_val = float(
                samples_semantic_f1_score(pred_boot, gold_boot, S_matrix)
            )
        elif metric_2_name == "semantic_precision":
            result = samples_semantic_f1_score(
                pred_boot, gold_boot, S_matrix, return_components=True
            )
            metric2_val = float(result['precision'])
        elif metric_2_name == "semantic_recall":
            result = samples_semantic_f1_score(
                pred_boot, gold_boot, S_matrix, return_components=True
            )
            metric2_val = float(result['recall'])
        elif metric_2_name == "hungarian_score":
            # Suppress Hungarian debug output
            import io

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                metric2_val = float(
                    hungarian_score(pred_boot, gold_boot, S_matrix)
                )
            finally:
                sys.stdout = old_stdout
        else:
            raise ValueError(f"Unknown metric: {metric_2_name}")

        metric1_vals.append(metric1_val)
        metric2_vals.append(metric2_val)

    # Compute CIs
    metric1_vals = np.array(metric1_vals)
    metric2_vals = np.array(metric2_vals)

    metric1_mean = float(metric1_vals.mean())
    metric1_lo, metric1_hi = np.percentile(metric1_vals, [2.5, 97.5])

    metric2_mean = float(metric2_vals.mean())
    metric2_lo, metric2_hi = np.percentile(metric2_vals, [2.5, 97.5])

    return (
        metric1_mean,
        float(metric1_lo),
        float(metric1_hi),
        metric2_mean,
        float(metric2_lo),
        float(metric2_hi),
    )


def plot_precision_comparison(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot Fig A10: Semantic F1, Precision & Recall across two-mode frequency.

    Following PROJECT.md specification: Shows line plots of semantic F1,
    semantic precision, and semantic recall aggregated across all other
    variables (k, q) for clarity.
    """
    logger.info("Generating Fig A10: Precision comparison plot")

    # Generate data using same approach as main script
    try:
        from scripts.semantic_f1.study_a4 import create_precision_test_data
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        create_precision_test_data = study_a4.create_precision_test_data

    df_results = create_precision_test_data(cfg)

    # Group by two_mode_freq and compute means across k and q
    grouped = (
        df_results.groupby('two_mode_freq')
        .agg(
            {
                'semantic_f1': 'mean',
                'semantic_precision': 'mean',
                'semantic_recall': 'mean',
                'gap_f1_minus_precision': 'mean',
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = grouped['two_mode_freq']
    y1 = grouped['semantic_f1']
    y2 = grouped['semantic_precision']
    y3 = grouped['semantic_recall']

    # Plot lines
    ax.plot(
        x,
        y1,
        'o-',
        color=SEM_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic F1',
    )
    ax.plot(
        x,
        y2,
        's-',
        color=THIRD_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic Precision',
    )
    ax.plot(
        x,
        y3,
        '^-',
        color=HARD_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic Recall',
    )

    ax.set_xlabel('Frequency of Two Modes in Gold Labels', fontsize=16)
    ax.set_ylabel('Metric Score', fontsize=16)
    ax.set_title(
        'Semantic F1 vs Semantic Precision', fontsize=20, fontweight='bold'
    )
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Set dynamic y-axis limits based on data range
    all_values = [y1, y2, y3]
    y_min = min(min(vals) for vals in all_values)
    y_max = max(max(vals) for vals in all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range))

    # Save plot
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"fig_A10_precision_comparison_{ts}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(outdir, f"fig_A10_precision_comparison_{ts}.pdf"),
        bbox_inches='tight',
    )
    plt.close(fig)

    logger.info(f"Saved Fig A10 plots to {outdir}")
    logger.info(
        f"Sample metrics: F1={y1.mean():.3f}, Precision={y2.mean():.3f}, Recall={y3.mean():.3f}"
    )


def plot_recall_comparison(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot Fig A11: Semantic F1, Precision & Recall across two-mode frequency.

    Following PROJECT.md specification: Shows line plots of semantic F1,
    semantic precision, and semantic recall aggregated across all other
    variables (k, q) for clarity.
    """
    logger.info("Generating Fig A11: Recall comparison plot")

    # Generate data using same approach as main script
    try:
        from scripts.semantic_f1.study_a4 import create_recall_test_data
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        create_recall_test_data = study_a4.create_recall_test_data

    df_results = create_recall_test_data(cfg)

    # Group by two_mode_freq and compute means across k and q
    grouped = (
        df_results.groupby('two_mode_freq')
        .agg(
            {
                'semantic_f1': 'mean',
                'semantic_precision': 'mean',
                'semantic_recall': 'mean',
                'gap_f1_minus_recall': 'mean',
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = grouped['two_mode_freq']
    y1 = grouped['semantic_f1']
    y2 = grouped['semantic_precision']
    y3 = grouped['semantic_recall']

    # Plot lines
    ax.plot(
        x,
        y1,
        'o-',
        color=SEM_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic F1',
    )
    ax.plot(
        x,
        y2,
        's-',
        color=THIRD_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic Precision',
    )
    ax.plot(
        x,
        y3,
        '^-',
        color=HARD_COLOR,
        linewidth=2,
        markersize=6,
        label='Semantic Recall',
    )

    ax.set_xlabel('Frequency of Two Modes in Predictor', fontsize=16)
    ax.set_ylabel('Metric Score', fontsize=16)
    ax.set_title(
        'Semantic F1 vs Semantic Recall', fontsize=20, fontweight='bold'
    )
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    # Set dynamic y-axis limits based on data range
    all_values = [y1, y2, y3]
    y_min = min(min(vals) for vals in all_values)
    y_max = max(max(vals) for vals in all_values)
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range))

    # Save plot
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"fig_A11_recall_comparison_{ts}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(outdir, f"fig_A11_recall_comparison_{ts}.pdf"),
        bbox_inches='tight',
    )
    plt.close(fig)

    logger.info(f"Saved Fig A11 plots to {outdir}")
    logger.info(
        f"Sample metrics: F1={y1.mean():.3f}, Precision={y2.mean():.3f}, Recall={y3.mean():.3f}"
    )


def plot_hungarian_comparison(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot Fig A12: Semantic F1 vs Hungarian across prediction counts and two-mode frequency.

    Following PROJECT.md specification: Shows subplots of line plots where
    x-axis is number of predictions, columns in subplot are frequency of two modes.
    """
    logger.info("Generating Fig A12: Hungarian comparison plot")

    # Generate data using same approach as main script
    try:
        from scripts.semantic_f1.study_a4 import create_hungarian_test_data
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        create_hungarian_test_data = study_a4.create_hungarian_test_data

    df_results = create_hungarian_test_data(cfg)

    # Get unique two-mode frequencies for subplots
    two_mode_freqs = sorted(df_results['two_mode_freq'].unique())

    ncols = min(3, len(two_mode_freqs))
    nrows = (len(two_mode_freqs) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True
    )
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Ensure we have enough axes for all subplots
    if len(axes) < len(two_mode_freqs):
        # Add extra subplot(s) if needed
        for _ in range(len(two_mode_freqs) - len(axes)):
            axes = list(axes) + [fig.add_subplot(nrows, ncols, len(axes) + 1)]

    for i, freq in enumerate(two_mode_freqs):
        if i >= len(axes):
            break

        ax = axes[i]

        # Filter data for this frequency
        freq_data = df_results[df_results['two_mode_freq'] == freq]

        # Group by num_predictions and compute means across k
        grouped = (
            freq_data.groupby('num_predictions')
            .agg(
                {
                    'semantic_f1': 'mean',
                    'hungarian_score': 'mean',
                    'gap_f1_minus_hungarian': 'mean',
                }
            )
            .reset_index()
        )

        x = grouped['num_predictions']
        y1 = grouped['semantic_f1']
        y2 = grouped['hungarian_score']

        # Plot lines
        ax.plot(
            x,
            y1,
            'o-',
            color=SEM_COLOR,
            linewidth=2,
            markersize=6,
            label='Semantic F1',
        )
        ax.plot(
            x,
            y2,
            's-',
            color=HARD_COLOR,
            linewidth=2,
            markersize=6,
            label='Hungarian Score',
        )

        if i // ncols == nrows - 1:
            ax.set_xlabel('Number of Predictions', fontsize=18)
        if i % ncols == 0:
            ax.set_ylabel('Metric Score', fontsize=18)
        ax.set_title(f'Two-Mode Freq = {freq*100:.1f}%', fontsize=18)
        if i == len(two_mode_freqs) - 1:
            ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)

        # # Set dynamic y-axis limits based on data range for this subplot
        # all_subplot_values = [y1, y2]
        # y_min = min(min(vals) for vals in all_subplot_values)
        # y_max = max(max(vals) for vals in all_subplot_values)
        # y_range = y_max - y_min
        # ax.set_ylim(
        #     max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range)
        # )

    # Hide any unused subplots
    for j in range(len(two_mode_freqs), len(axes)):
        axes[j].set_visible(False)

    # Overall title
    fig.suptitle(
        'Semantic F1 vs Extended Hungarian', fontsize=22, fontweight='bold'
    )

    # Save plot
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"fig_A12_hungarian_comparison_{ts}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(outdir, f"fig_A12_hungarian_comparison_{ts}.pdf"),
        bbox_inches='tight',
    )
    plt.close(fig)

    logger.info(f"Saved Fig A12 plots to {outdir}")

    # Log sample performance differences
    overall_gap = df_results['gap_f1_minus_hungarian'].mean()
    logger.info(f"Overall F1-Hungarian gap: {overall_gap:.4f}")

    logger.info(f"Saved Fig A12 plots to {outdir}")


def plot_gap_analysis(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot gap analysis across all three scenarios for supplementary figure.

    Shows when Semantic F1 outperforms each baseline by plotting the gaps.
    """
    logger.info("Generating supplementary gap analysis plot")

    # Import the data generation functions
    try:
        from scripts.semantic_f1.study_a4 import (
            create_precision_test_data,
            create_recall_test_data,
            create_hungarian_test_data,
        )
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        create_precision_test_data = study_a4.create_precision_test_data
        create_recall_test_data = study_a4.create_recall_test_data
        create_hungarian_test_data = study_a4.create_hungarian_test_data

    # Generate all three datasets
    precision_data = create_precision_test_data(cfg)
    recall_data = create_recall_test_data(cfg)
    hungarian_data = create_hungarian_test_data(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Precision gap plot
    prec_grouped = precision_data.groupby('two_mode_freq')[
        'gap_f1_minus_precision'
    ].mean()
    axes[0].plot(
        prec_grouped.index,
        prec_grouped.values,
        'o-',
        color=THIRD_COLOR,
        linewidth=2,
    )
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Two-Mode Frequency')
    axes[0].set_ylabel('Gap (F1 - Precision)')
    axes[0].set_title('(a) Precision Test')
    axes[0].grid(True, alpha=0.3)

    # Recall gap plot
    rec_grouped = recall_data.groupby('two_mode_freq')[
        'gap_f1_minus_recall'
    ].mean()
    axes[1].plot(
        rec_grouped.index,
        rec_grouped.values,
        'o-',
        color=HARD_COLOR,
        linewidth=2,
    )
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Two-Mode Frequency')
    axes[1].set_ylabel('Gap (F1 - Recall)')
    axes[1].set_title('(b) Recall Test')
    axes[1].grid(True, alpha=0.3)

    # Hungarian gap plot
    hung_grouped = hungarian_data.groupby('two_mode_freq')[
        'gap_f1_minus_hungarian'
    ].mean()
    axes[2].plot(
        hung_grouped.index,
        hung_grouped.values,
        'o-',
        color='purple',
        linewidth=2,
    )
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Two-Mode Frequency')
    axes[2].set_ylabel('Gap (F1 - Hungarian)')
    axes[2].set_title('(c) Hungarian Test')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        'Study A.4: When Semantic F1 Outperforms Baselines', fontsize=14
    )
    fig.tight_layout()

    # Save plot
    fig.savefig(
        os.path.join(outdir, f"fig_A4_gap_analysis_{ts}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(outdir, f"fig_A4_gap_analysis_{ts}.pdf"),
        bbox_inches='tight',
    )
    plt.close(fig)

    logger.info(f"Saved gap analysis plot to {outdir}")


def plot_confidence_intervals_detailed(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create detailed confidence interval plots for each scenario.

    This generates more detailed CI plots by actually computing bootstrap CIs
    for representative parameter combinations.
    """
    logger.info("Generating detailed confidence interval plots")

    # Import required functions
    try:
        from scripts.semantic_f1.study_a4 import (
            make_bimodal_structure,
            sample_bimodal_labels,
            determine_dominant_mode,
            sample_from_mode,
        )
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        make_bimodal_structure = study_a4.make_bimodal_structure
        sample_bimodal_labels = study_a4.sample_bimodal_labels
        determine_dominant_mode = study_a4.determine_dominant_mode
        sample_from_mode = study_a4.sample_from_mode

    # Use smaller dataset for CI computation
    ci_cfg = cfg.__class__(
        scenario=cfg.scenario,
        num_labels=cfg.num_labels,
        num_examples=min(cfg.num_examples, 1000),  # Smaller for speed
        k_values=[cfg.k_values[0]] if cfg.k_values else [2],  # Single k value
        p_values=[0.5],  # Single p value
        two_mode_frequencies=cfg.two_mode_frequencies,
        bootstrap=100,  # Smaller bootstrap for speed
        seed=cfg.seed,
        outdir=cfg.outdir,
    )

    labels = study_a1.make_labels(ci_cfg.num_labels)
    S_ideal = study_a1.make_similarity_ring(ci_cfg.num_labels)
    pos_center, neg_center = make_bimodal_structure(ci_cfg.num_labels)

    # Compute CIs for precision test
    k = ci_cfg.k_values[0]
    p = ci_cfg.p_values[0]

    precision_results = {}

    for two_mode_freq in ci_cfg.two_mode_frequencies:
        # Generate gold labels
        gold_indices = []
        for _ in range(ci_cfg.num_examples):
            gold_idx = sample_bimodal_labels(
                ci_cfg.num_labels, k, pos_center, neg_center, two_mode_freq
            )
            gold_indices.append(gold_idx)

        gold_labels = study_a1.to_label_names(gold_indices, labels)

        # Generate predictions (unimodal predictor)
        pred_indices = []
        for gold_idx in gold_indices:
            dominant_mode = determine_dominant_mode(
                gold_idx, pos_center, neg_center, ci_cfg.num_labels
            )
            if np.random.random() < p:
                pred_idx = sample_from_mode(
                    ci_cfg.num_labels, dominant_mode, k, pos_center, neg_center
                )
            else:
                pred_idx = gold_idx
            pred_indices.append(pred_idx)

        pred_labels = study_a1.to_label_names(pred_indices, labels)

        # Bootstrap CIs
        f1_mean, f1_lo, f1_hi, prec_mean, prec_lo, prec_hi = (
            bootstrap_metric_comparison(
                gold_labels,
                pred_labels,
                S_ideal,
                "semantic_f1",
                "semantic_precision",
                B=ci_cfg.bootstrap,
                seed=ci_cfg.seed,
            )
        )

        precision_results[two_mode_freq] = {
            'f1_mean': f1_mean,
            'f1_lo': f1_lo,
            'f1_hi': f1_hi,
            'prec_mean': prec_mean,
            'prec_lo': prec_lo,
            'prec_hi': prec_hi,
        }

    # Plot precision CIs
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    freqs = list(precision_results.keys())
    f1_means = [precision_results[f]['f1_mean'] for f in freqs]
    f1_los = [precision_results[f]['f1_lo'] for f in freqs]
    f1_his = [precision_results[f]['f1_hi'] for f in freqs]

    prec_means = [precision_results[f]['prec_mean'] for f in freqs]
    prec_los = [precision_results[f]['prec_lo'] for f in freqs]
    prec_his = [precision_results[f]['prec_hi'] for f in freqs]

    # Plot with error bars
    ax.errorbar(
        freqs,
        f1_means,
        yerr=[
            np.array(f1_means) - np.array(f1_los),
            np.array(f1_his) - np.array(f1_means),
        ],
        fmt='o-',
        color=SEM_COLOR,
        linewidth=2,
        capsize=5,
        capthick=2,
        label='Semantic F1',
        markersize=6,
    )

    ax.errorbar(
        freqs,
        prec_means,
        yerr=[
            np.array(prec_means) - np.array(prec_los),
            np.array(prec_his) - np.array(prec_means),
        ],
        fmt='s-',
        color=THIRD_COLOR,
        linewidth=2,
        capsize=5,
        capthick=2,
        label='Semantic Precision',
        markersize=6,
    )

    ax.set_xlabel('Two-Mode Frequency')
    ax.set_ylabel('Metric Score')
    ax.set_title('Fig A10: Semantic F1 vs Precision with 95% CIs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set dynamic y-axis limits based on confidence interval data
    all_data = f1_means + prec_means + f1_los + f1_his + prec_los + prec_his
    y_min = min(all_data)
    y_max = max(all_data)
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range))

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"fig_A10_precision_with_CIs_{ts}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(outdir, f"fig_A10_precision_with_CIs_{ts}.pdf"),
        bbox_inches='tight',
    )
    plt.close(fig)

    logger.info(f"Saved detailed CI plots to {outdir}")


def create_summary_table(
    cfg: Any,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create LaTeX summary table for Study A.4 results."""
    logger.info("Creating Study A.4 summary table")

    # Import the data generation functions
    try:
        from scripts.semantic_f1.study_a4 import (
            create_precision_test_data,
            create_recall_test_data,
            create_hungarian_test_data,
        )
    except ImportError:
        # Fallback for when running from within the package
        sys.path.append(os.path.dirname(__file__))
        import study_a4

        create_precision_test_data = study_a4.create_precision_test_data
        create_recall_test_data = study_a4.create_recall_test_data
        create_hungarian_test_data = study_a4.create_hungarian_test_data

    # Generate all datasets
    precision_data = create_precision_test_data(cfg)
    recall_data = create_recall_test_data(cfg)
    hungarian_data = create_hungarian_test_data(cfg)

    # Compute summary statistics
    prec_summary = (
        precision_data.groupby('two_mode_freq')
        .agg(
            {
                'semantic_f1': 'mean',
                'semantic_precision': 'mean',
                'gap_f1_minus_precision': 'mean',
            }
        )
        .round(3)
    )

    rec_summary = (
        recall_data.groupby('two_mode_freq')
        .agg(
            {
                'semantic_f1': 'mean',
                'semantic_recall': 'mean',
                'gap_f1_minus_recall': 'mean',
            }
        )
        .round(3)
    )

    hung_summary = (
        hungarian_data.groupby('two_mode_freq')
        .agg(
            {
                'semantic_f1': 'mean',
                'hungarian_score': 'mean',
                'gap_f1_minus_hungarian': 'mean',
            }
        )
        .round(3)
    )

    # Write LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Two-Mode Freq & Semantic F1 & Baseline & Gap \\\\",
        "\\midrule",
        "\\multicolumn{4}{c}{\\textbf{Precision Test}} \\\\",
    ]

    for freq in prec_summary.index:
        f1_val = prec_summary.loc[freq, 'semantic_f1']
        prec_val = prec_summary.loc[freq, 'semantic_precision']
        gap_val = prec_summary.loc[freq, 'gap_f1_minus_precision']
        latex_lines.append(
            f"{freq:.1f} & {f1_val:.3f} & {prec_val:.3f} & {gap_val:+.3f} \\\\"
        )

    latex_lines.extend(
        [
            "\\midrule",
            "\\multicolumn{4}{c}{\\textbf{Recall Test}} \\\\",
        ]
    )

    for freq in rec_summary.index:
        f1_val = rec_summary.loc[freq, 'semantic_f1']
        rec_val = rec_summary.loc[freq, 'semantic_recall']
        gap_val = rec_summary.loc[freq, 'gap_f1_minus_recall']
        latex_lines.append(
            f"{freq:.1f} & {f1_val:.3f} & {rec_val:.3f} & {gap_val:+.3f} \\\\"
        )

    latex_lines.extend(
        [
            "\\midrule",
            "\\multicolumn{4}{c}{\\textbf{Hungarian Test}} \\\\",
        ]
    )

    for freq in hung_summary.index:
        f1_val = hung_summary.loc[freq, 'semantic_f1']
        hung_val = hung_summary.loc[freq, 'hungarian_score']
        gap_val = hung_summary.loc[freq, 'gap_f1_minus_hungarian']
        latex_lines.append(
            f"{freq:.1f} & {f1_val:.3f} & {hung_val:.3f} & {gap_val:+.3f} \\\\"
        )

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Study A.4: Semantic F1 vs Baselines Summary. Gap = Semantic F1 - Baseline.}",
            "\\label{tab:study_a4_summary}",
            "\\end{table}",
        ]
    )

    # Write to file
    table_path = os.path.join(outdir, f"table_A4_summary_{ts}.tex")
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    logger.info(f"Saved summary table to {table_path}")
