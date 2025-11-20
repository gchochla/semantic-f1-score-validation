from __future__ import annotations

import os
import math
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import study_a1 for utility functions
try:
    import scripts.semantic_f1.study_a1 as study_a1
except Exception:
    # Fallback: import sibling when executed directly via path
    sys.path.append(os.path.dirname(__file__))
    import study_a1  # type: ignore


def plot_metric_vs_p_curves(
    cfg: Any,
    labels: list[str],
    S_ideal: pd.DataFrame,
    helpers: dict[str, Any],
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot metric vs probability curves for Fig A6.

    Compare Prototype-Bimodal and Prototype-Within-Mode classifiers across q
    with the near-miss predictor across p. For each sem-hard f1 pair, the three
    classifiers should share the x axis since p and q have the same range.
    """
    if not cfg.q_values:
        return

    sample_gold_labels_bimodal: Callable = helpers["sample_gold_labels_bimodal"]
    to_label_names: Callable = helpers["to_label_names"]
    prototype_bimodal_predictor: Callable = helpers[
        "prototype_bimodal_predictor"
    ]
    prototype_within_mode_predictor: Callable = helpers[
        "prototype_within_mode_predictor"
    ]
    make_bimodal_weights: Callable = helpers["make_bimodal_weights"]
    select_prototypes: Callable = helpers["select_prototypes"]
    semantic_f1_score: Callable = helpers["semantic_f1_score"]
    f1_score: Callable = helpers["f1_score"]
    MultiLabelBinarizer = helpers["MultiLabelBinarizer"]

    # Use representative parameter values
    k = cfg.k_values[len(cfg.k_values) // 2] if cfg.k_values else 2
    m = (
        cfg.prototype_sizes[len(cfg.prototype_sizes) // 2]
        if cfg.prototype_sizes
        else 3
    )
    imbalance = 0.5  # balanced

    n = cfg.num_labels
    w_pos, w_neg = make_bimodal_weights(n, cfg.kappa)
    pos_prototypes, neg_prototypes = select_prototypes(n, w_pos, w_neg, m)

    # Generate consistent gold set
    rng = np.random.default_rng(cfg.seed)
    gold_indices = [
        sample_gold_labels_bimodal(n, k, w_pos, w_neg, imbalance)
        for _ in range(cfg.num_examples)
    ]
    gold_indices = [sorted(set(g)) for g in gold_indices]
    gold_names = study_a1.to_label_names(gold_indices, labels)

    # Compute metrics for each probability value - use shared x-axis
    prob_values = cfg.q_values  # q and p have same range
    metrics_data = {
        "bimodal": {"hard": {}, "semantic": {}},
        "within_mode": {"hard": {}, "semantic": {}},
    }

    for avg_type in ["micro", "macro", "samples"]:  # Include samples for A6
        for pred_type in metrics_data:
            metrics_data[pred_type]["hard"][avg_type] = []
            metrics_data[pred_type]["semantic"][avg_type] = []

    # For CI bands, keep predictions per prob
    pred_series: dict[str, list[list[list[str]]]] = {
        "bimodal": [],
        "within_mode": [],
    }

    for prob in prob_values:
        # Generate predictions for all three predictors

        # Bimodal predictor (uses q)
        bimodal_preds = [
            prototype_bimodal_predictor(
                g, w_pos, w_neg, pos_prototypes, neg_prototypes, prob
            )
            for g in gold_indices
        ]
        bimodal_pred_names = study_a1.to_label_names(
            [sorted(set(x)) for x in bimodal_preds], labels
        )

        # Within-mode predictor (uses q)
        within_preds = [
            prototype_within_mode_predictor(
                g,
                w_pos,
                w_neg,
                pos_prototypes,
                neg_prototypes,
                k,
                prob,
                2.0,  # beta=2.0
            )
            for g in gold_indices
        ]
        within_pred_names = study_a1.to_label_names(
            [sorted(set(x)) for x in within_preds], labels
        )

        predictors = {
            "bimodal": bimodal_pred_names,
            "within_mode": within_pred_names,
        }

        for pred_name, pred_labels in predictors.items():
            # Compute hard metrics
            mlb = MultiLabelBinarizer(classes=labels)
            Y_true = mlb.fit_transform(gold_names)
            Y_pred = mlb.transform(pred_labels)

            hard_micro = f1_score(
                Y_true, Y_pred, average="micro", zero_division=0
            )
            hard_macro = f1_score(
                Y_true, Y_pred, average="macro", zero_division=0
            )
            hard_samples = f1_score(
                Y_true, Y_pred, average="samples", zero_division=0
            )

            metrics_data[pred_name]["hard"]["micro"].append(hard_micro)
            metrics_data[pred_name]["hard"]["macro"].append(hard_macro)
            metrics_data[pred_name]["hard"]["samples"].append(hard_samples)

            # Compute semantic metrics
            sem_micro = float(
                semantic_f1_score(
                    gold_names, pred_labels, S_ideal, average="micro"
                )
            )
            sem_macro = float(
                semantic_f1_score(
                    gold_names, pred_labels, S_ideal, average="macro"
                )
            )
            sem_samples = float(
                semantic_f1_score(
                    gold_names, pred_labels, S_ideal, average="samples"
                )
            )

            metrics_data[pred_name]["semantic"]["micro"].append(sem_micro)
            metrics_data[pred_name]["semantic"]["macro"].append(sem_macro)
            metrics_data[pred_name]["semantic"]["samples"].append(sem_samples)

        # Retain predictor outputs for bootstrap per prob
        pred_series["bimodal"].append(bimodal_pred_names)
        pred_series["within_mode"].append(within_pred_names)

    # Pre-compute Near-Miss performance at a single user-defined p for horizontal reference
    near_vals = {"hard": {}, "semantic": {}}
    near_ci = {"hard": {}, "semantic": {}}
    if hasattr(cfg, "near_p") and cfg.near_p is not None:
        near_radius = cfg.near_radii[0]
        near_preds_once = [
            helpers["near_miss_predictor"](
                g, n, float(cfg.near_p), radius=near_radius
            )
            for g in gold_indices
        ]
        near_pred_names_once = study_a1.to_label_names(
            [sorted(set(x)) for x in near_preds_once], labels
        )

        # Bootstrap helper: single-metric CI
        def _bootstrap_metric_ci(
            trues, preds, metric_name: str, B: int, seed: int
        ):
            rng = np.random.default_rng(seed)
            n = len(trues)
            mlb_local = MultiLabelBinarizer(classes=labels)
            mlb_local.fit([[]])
            base_idx = np.arange(n)
            vals = []
            for _ in range(B):
                idx = rng.choice(base_idx, size=n, replace=True)
                T = [trues[i] for i in idx]
                P = [preds[i] for i in idx]
                if metric_name.startswith("hard_"):
                    avg = metric_name.split("_")[1]
                    Yt = mlb_local.transform(T)
                    Yp = mlb_local.transform(P)
                    vals.append(
                        float(f1_score(Yt, Yp, average=avg, zero_division=0))
                    )
                else:
                    avg = metric_name.split("_")[1]
                    vals.append(
                        float(semantic_f1_score(T, P, S_ideal, average=avg))
                    )
            vals = np.array(vals, dtype=float)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            return float(vals.mean()), float(lo), float(hi)

        for avg_type in ["micro", "macro", "samples"]:
            mean_h, lo_h, hi_h = _bootstrap_metric_ci(
                gold_names,
                near_pred_names_once,
                f"hard_{avg_type}",
                B=getattr(cfg, "bootstrap", 200),
                seed=cfg.seed + 700,
            )
            mean_s, lo_s, hi_s = _bootstrap_metric_ci(
                gold_names,
                near_pred_names_once,
                f"sem_{avg_type}",
                B=getattr(cfg, "bootstrap", 200),
                seed=cfg.seed + 701,
            )
            near_vals["hard"][avg_type] = mean_h
            near_ci["hard"][avg_type] = (lo_h, hi_h)
            near_vals["semantic"][avg_type] = mean_s
            near_ci["semantic"][avg_type] = (lo_s, hi_s)

    # Create figures for each averaging type (micro, macro, samples)
    for avg_type in ["micro", "macro", "samples"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # Plot Hard F1
        ax1.plot(
            prob_values,
            metrics_data["bimodal"]["hard"][avg_type],
            'o-',
            color='#E45756',
            label='Prototype-Bimodal',
            linewidth=2,
            markersize=6,
        )
        # CI band for bimodal
        try:
            bi_lo, bi_hi = [], []
            for i, preds_here in enumerate(pred_series["bimodal"]):
                # compute CI for this prob
                # reuse bootstrap helper defined above
                rng_seed = cfg.seed + 100 + i
                # inline compute to avoid scope issues
                rng = np.random.default_rng(rng_seed)
                n_bs = len(gold_names)
                mlb_loc = MultiLabelBinarizer(classes=labels)
                mlb_loc.fit([[]])
                base_idx = np.arange(n_bs)
                vals = []
                for _ in range(getattr(cfg, "bootstrap", 200)):
                    idx = rng.choice(base_idx, size=n_bs, replace=True)
                    T = [gold_names[j] for j in idx]
                    P = [preds_here[j] for j in idx]
                    avg = avg_type
                    Yt = mlb_loc.transform(T)
                    Yp = mlb_loc.transform(P)
                    vals.append(
                        float(f1_score(Yt, Yp, average=avg, zero_division=0))
                    )
                vals = np.array(vals, dtype=float)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                bi_lo.append(float(lo))
                bi_hi.append(float(hi))
            ax1.fill_between(
                prob_values,
                bi_lo,
                bi_hi,
                color='#E45756',
                alpha=0.15,
                linewidth=0,
            )
        except Exception:
            pass
        ax1.plot(
            prob_values,
            metrics_data["within_mode"]["hard"][avg_type],
            's-',
            color='#54A24B',
            label='Prototype-Within-Mode',
            linewidth=2,
            markersize=6,
        )
        # CI band for within-mode
        try:
            wm_lo, wm_hi = [], []
            for i, preds_here in enumerate(pred_series["within_mode"]):
                rng_seed = cfg.seed + 200 + i
                rng = np.random.default_rng(rng_seed)
                n_bs = len(gold_names)
                mlb_loc = MultiLabelBinarizer(classes=labels)
                mlb_loc.fit([[]])
                base_idx = np.arange(n_bs)
                vals = []
                for _ in range(getattr(cfg, "bootstrap", 200)):
                    idx = rng.choice(base_idx, size=n_bs, replace=True)
                    T = [gold_names[j] for j in idx]
                    P = [preds_here[j] for j in idx]
                    avg = avg_type
                    Yt = mlb_loc.transform(T)
                    Yp = mlb_loc.transform(P)
                    vals.append(
                        float(f1_score(Yt, Yp, average=avg, zero_division=0))
                    )
                vals = np.array(vals, dtype=float)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                wm_lo.append(float(lo))
                wm_hi.append(float(hi))
            ax1.fill_between(
                prob_values,
                wm_lo,
                wm_hi,
                color='#54A24B',
                alpha=0.15,
                linewidth=0,
            )
        except Exception:
            pass
        # Add near-miss as a horizontal line at performance for given p
        if near_vals["hard"].get(avg_type) is not None:
            ax1.axhline(
                y=near_vals["hard"][avg_type],
                color="#4C78A8",
                linestyle="--",
                linewidth=2,
                label=f"Near-Miss (p={cfg.near_p:.2f})",
            )
            try:
                lo, hi = near_ci["hard"][avg_type]
                ax1.axhspan(lo, hi, color="#4C78A8", alpha=0.12)
            except Exception:
                pass

        ax1.set_xlabel("q", fontsize=20)
        # Only show ticks at provided q values
        ax1.set_xticks(prob_values)
        ax1.set_xticklabels([f"{v:.2f}" for v in prob_values])
        # ax1.set_ylabel(f"Hard F1 Score ({avg_type})")
        ax1.set_title(f"Hard F1", fontsize=20)
        ax1.legend(fontsize=20, loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)

        # Plot Semantic F1
        ax2.plot(
            prob_values,
            metrics_data["bimodal"]["semantic"][avg_type],
            'o-',
            color='#E45756',
            label='Prototype-Bimodal',
            linewidth=2,
            markersize=6,
        )
        # CI band for bimodal semantic
        try:
            bi_lo, bi_hi = [], []
            for i, preds_here in enumerate(pred_series["bimodal"]):
                rng_seed = cfg.seed + 300 + i
                rng = np.random.default_rng(rng_seed)
                n_bs = len(gold_names)
                base_idx = np.arange(n_bs)
                vals = []
                for _ in range(getattr(cfg, "bootstrap", 200)):
                    idx = rng.choice(base_idx, size=n_bs, replace=True)
                    T = [gold_names[j] for j in idx]
                    P = [preds_here[j] for j in idx]
                    vals.append(
                        float(
                            semantic_f1_score(T, P, S_ideal, average=avg_type)
                        )
                    )
                vals = np.array(vals, dtype=float)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                bi_lo.append(float(lo))
                bi_hi.append(float(hi))
            ax2.fill_between(
                prob_values,
                bi_lo,
                bi_hi,
                color='#E45756',
                alpha=0.15,
                linewidth=0,
            )
        except Exception:
            pass
        ax2.plot(
            prob_values,
            metrics_data["within_mode"]["semantic"][avg_type],
            's-',
            color='#54A24B',
            label='Prototype-Within-Mode',
            linewidth=2,
            markersize=6,
        )
        # CI band for within-mode semantic
        try:
            wm_lo, wm_hi = [], []
            for i, preds_here in enumerate(pred_series["within_mode"]):
                rng_seed = cfg.seed + 400 + i
                rng = np.random.default_rng(rng_seed)
                n_bs = len(gold_names)
                base_idx = np.arange(n_bs)
                vals = []
                for _ in range(getattr(cfg, "bootstrap", 200)):
                    idx = rng.choice(base_idx, size=n_bs, replace=True)
                    T = [gold_names[j] for j in idx]
                    P = [preds_here[j] for j in idx]
                    vals.append(
                        float(
                            semantic_f1_score(T, P, S_ideal, average=avg_type)
                        )
                    )
                vals = np.array(vals, dtype=float)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                wm_lo.append(float(lo))
                wm_hi.append(float(hi))
            ax2.fill_between(
                prob_values,
                wm_lo,
                wm_hi,
                color='#54A24B',
                alpha=0.15,
                linewidth=0,
            )
        except Exception:
            pass
        # Add near-miss horizontal line to semantic panel as well
        if near_vals["semantic"].get(avg_type) is not None:
            ax2.axhline(
                y=near_vals["semantic"][avg_type],
                color="#4C78A8",
                linestyle="--",
                linewidth=2,
                label=f"Near-Miss (p={cfg.near_p:.2f})",
            )
            try:
                lo, hi = near_ci["semantic"][avg_type]
                ax2.axhspan(lo, hi, color="#4C78A8", alpha=0.12)
            except Exception:
                pass

        ax2.set_xlabel("q", fontsize=20)
        # Only show ticks at provided q values
        ax2.set_xticks(prob_values)
        ax2.set_xticklabels([f"{v:.2f}" for v in prob_values])
        # ax2.set_ylabel(f"Semantic F1 Score ({avg_type})")
        # ax2.set_title(f"Semantic F1 (ideal) — {avg_type.capitalize()}")
        ax2.set_title(f"Semantic F1", fontsize=20)
        # ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)

        # Set reasonable y-limits for both subplots
        all_hard_vals = []
        all_sem_vals = []
        for pred in metrics_data:
            all_hard_vals.extend(metrics_data[pred]["hard"][avg_type])
            all_sem_vals.extend(metrics_data[pred]["semantic"][avg_type])

        # Include near-miss performance in y-limits if available
        if near_vals["hard"].get(avg_type) is not None:
            all_hard_vals.append(near_vals["hard"][avg_type])
        y_min_hard = max(0, min(all_hard_vals) - 0.05)
        y_max_hard = min(1, max(all_hard_vals) + 0.05)

        if near_vals["semantic"].get(avg_type) is not None:
            all_sem_vals.append(near_vals["semantic"][avg_type])
        y_min_sem = max(0, min(all_sem_vals) - 0.05)
        y_max_sem = min(1, max(all_sem_vals) + 0.05)
        ax2.set_ylim(min([y_min_sem, y_min_hard]), max([y_max_sem, y_max_hard]))

        # Remove figure code from title (no "Fig A6")
        plt.suptitle(
            f"Predictor Comparison — {avg_type.capitalize()} F1",
            fontweight="bold",
            fontsize=24,
        )
        plt.tight_layout()

        fig_path_png = os.path.join(
            outdir, f"fig_A6_metric_vs_prob_{avg_type}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A6_metric_vs_prob_{avg_type}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)

        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")


def plot_imbalance_heatmaps(
    df_summary: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot heatmap of metrics vs (imbalance, q) for Fig A7.

    Shows how class imbalance and probability q interact, expecting Semantic F1
    to resist inflation from majority-mode bias.
    """
    if df_summary.empty:
        return

    # Filter to representative parameters and bimodal predictor
    df = df_summary[df_summary["predictor"] == "bimodal"].copy()
    if df.empty:
        logger.warning("No bimodal predictor data found for heatmaps")
        return

    # Check if we have enough variation for heatmaps
    unique_imbalances = df["imbalance"].nunique()
    unique_q_values = df["q"].nunique()

    if unique_imbalances < 2:
        logger.warning(
            f"Not enough imbalance variation for heatmap (only {unique_imbalances} values). Skipping Fig A7."
        )
        return

    if unique_q_values < 2:
        logger.warning(
            f"Not enough q variation for heatmap (only {unique_q_values} values). Skipping Fig A7."
        )
        return

    # Use median k and m values, but handle non-integer medians
    k_values = sorted(df["k"].unique())
    m_values = sorted(df["m"].unique())

    # Pick representative values (prefer median, but use middle value for small sets)
    k_rep = k_values[len(k_values) // 2] if len(k_values) > 1 else k_values[0]
    m_rep = m_values[len(m_values) // 2] if len(m_values) > 1 else m_values[0]

    df_filtered = df[(df["k"] == k_rep) & (df["m"] == m_rep)].copy()

    if df_filtered.empty:
        logger.warning(
            f"No data after filtering for k={k_rep}, m={m_rep}. Using all bimodal data."
        )
        df_filtered = df.copy()

    # Build both heatmaps and place them on the same figure
    metrics = [
        ("hard_micro", "Hard F1 (micro)"),
        ("sem_micro:ideal", "Semantic F1 (micro, ideal)"),
    ]

    heatmaps = []
    for metric_name, _title in metrics:
        df_metric = df_filtered[df_filtered["metric"] == metric_name].copy()
        if df_metric.empty:
            logger.warning(f"No data found for metric {metric_name}")
            heatmaps.append(None)
            continue

        heatmap_data = df_metric.pivot_table(
            index="imbalance", columns="q", values="value", aggfunc="mean"
        )
        if heatmap_data.empty:
            logger.warning(f"No heatmap data for {metric_name}")
            heatmaps.append(None)
            continue
        heatmaps.append(heatmap_data)

    if not any(h is not None for h in heatmaps):
        return

    # Create one figure per averaging type (micro, macro, samples)
    for avg_type in ["micro", "macro", "samples"]:
        metrics = [
            (f"hard_{avg_type}", f"Hard F1 ({avg_type})"),
            (f"sem_{avg_type}:ideal", f"Semantic F1 ({avg_type}, ideal)"),
        ]

        heatmaps2 = []
        for metric_name, _title in metrics:
            df_metric = df_filtered[df_filtered["metric"] == metric_name].copy()
            if df_metric.empty:
                logger.warning(f"No data found for metric {metric_name}")
                heatmaps2.append(None)
                continue

            heatmap_data = df_metric.pivot_table(
                index="imbalance", columns="q", values="value", aggfunc="mean"
            )
            if heatmap_data.empty:
                logger.warning(f"No heatmap data for {metric_name}")
                heatmaps2.append(None)
                continue
            heatmaps2.append(heatmap_data)

        if not any(h is not None for h in heatmaps2):
            continue

        # Shared color scale across both heatmaps
        vals = []
        for h in heatmaps2:
            if h is not None:
                vals.append(np.nanmin(h.values))
                vals.append(np.nanmax(h.values))
        vmin = min(vals) if vals else None
        vmax = max(vals) if vals else None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ims = []
        for idx, (ax, heatmap_data, (_metric_name, title_suffix)) in enumerate(
            zip(axes, heatmaps2, metrics)
        ):
            if heatmap_data is None:
                ax.axis('off')
                ims.append(None)
                continue
            im = ax.imshow(
                heatmap_data.values,
                cmap="viridis",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ims.append(im)
            # Set ticks and labels using only provided values
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([f"{q:.2f}" for q in heatmap_data.columns])
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels([f"{imb:.2f}" for imb in heatmap_data.index])
            ax.set_xlabel("q")
            # Y label only on the left subplot; remove duplicated y ticks on right
            if idx == 0:
                ax.set_ylabel("Imbalance ratio")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_title(
                f"Bimodal Predictor — {title_suffix}\n(k={k_rep}, m={m_rep})"
            )

            # Annotate values
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    val = heatmap_data.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.3f}",
                            ha="center",
                            va="center",
                            color="white" if val < 0.5 else "black",
                            fontsize=8,
                        )

        # Single colorbar on the right spanning both subplots
        mappable = ims[1] if len(ims) > 1 and ims[1] is not None else ims[0]
        if mappable is not None:
            cbar = fig.colorbar(
                mappable,
                ax=axes.ravel().tolist(),
                location='right',
                fraction=0.046,
                pad=0.04,
            )
            cbar.set_label("F1 Score")

        # Remove figure code from title (no "Fig A7"); include averaging type
        plt.suptitle(
            f"Study A.2 — Imbalance vs q Heatmaps — {avg_type}",
            fontweight="bold",
        )

        fig_path_png = os.path.join(
            outdir, f"fig_A7_imbalance_heatmaps_{avg_type}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A7_imbalance_heatmaps_{avg_type}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)

        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")


def plot_stratified_curves(
    cfg: Any,
    labels: list[str],
    S_list: dict[str, Any],
    helpers: dict[str, Any],
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Plot stratified curves for pure-mode vs mixed-mode gold (Fig A8).

    Shows separate metric curves for examples where gold labels belong to a single
    mode vs examples spanning both modes, one row per metric for 2 predictors.
    Share y axis within each row. Expect largest Semantic-Hard gap on pure-mode cases.
    """
    if not cfg.q_values:
        return

    sample_gold_labels_bimodal: Callable = helpers["sample_gold_labels_bimodal"]
    to_label_names: Callable = helpers["to_label_names"]
    prototype_bimodal_predictor: Callable = helpers[
        "prototype_bimodal_predictor"
    ]
    prototype_within_mode_predictor: Callable = helpers[
        "prototype_within_mode_predictor"
    ]
    make_bimodal_weights: Callable = helpers["make_bimodal_weights"]
    select_prototypes: Callable = helpers["select_prototypes"]
    semantic_f1_score: Callable = helpers["semantic_f1_score"]
    f1_score: Callable = helpers["f1_score"]
    MultiLabelBinarizer = helpers["MultiLabelBinarizer"]

    # Use representative parameter values
    k = (
        max(cfg.k_values) if cfg.k_values else 3
    )  # Use larger k to get mixed-mode examples
    m = (
        cfg.prototype_sizes[len(cfg.prototype_sizes) // 2]
        if cfg.prototype_sizes
        else 3
    )
    imbalance = 0.5  # balanced

    n = cfg.num_labels
    w_pos, w_neg = make_bimodal_weights(n, cfg.kappa)
    pos_prototypes, neg_prototypes = select_prototypes(n, w_pos, w_neg, m)

    # Define mode purity threshold
    purity_threshold = 0.8  # If 80%+ weight in one mode, consider "pure"

    # Create figure with 3 rows (micro, macro, samples) and 2 columns (bimodal, within-mode)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    predictor_configs = [
        ("bimodal", "Prototype-Bimodal"),
        ("within_mode", "Prototype-Within-Mode"),
    ]

    for row_idx, avg_type in enumerate(["micro", "macro", "samples"]):
        for col_idx, (pred_type, pred_title) in enumerate(predictor_configs):
            ax = axes[row_idx, col_idx]

            # Collect data for each q value
            q_values = []
            hard_pure, sem_pure = [], []
            hard_mixed, sem_mixed = [], []
            # CI containers
            hard_pure_lo, hard_pure_hi = [], []
            sem_pure_lo, sem_pure_hi = [], []
            hard_mixed_lo, hard_mixed_hi = [], []
            sem_mixed_lo, sem_mixed_hi = [], []

            # Bootstrap helper
            def _boot_ci_single(
                T_names, P_names, metric_name: str, B: int, seed: int
            ):
                rng = np.random.default_rng(seed)
                nloc = len(T_names)
                mlb_loc = MultiLabelBinarizer(classes=labels)
                mlb_loc.fit([[]])
                base_idx = np.arange(nloc)
                vals = []
                for _ in range(B):
                    idx = rng.choice(base_idx, size=nloc, replace=True)
                    T = [T_names[i] for i in idx]
                    P = [P_names[i] for i in idx]
                    if metric_name.startswith("hard_"):
                        avg = metric_name.split("_")[1]
                        Yt = mlb_loc.transform(T)
                        Yp = mlb_loc.transform(P)
                        vals.append(
                            float(
                                f1_score(Yt, Yp, average=avg, zero_division=0)
                            )
                        )
                    else:
                        avg = metric_name.split("_")[1]
                        vals.append(
                            float(
                                semantic_f1_score(
                                    T, P, S_list["ideal"], average=avg
                                )
                            )
                        )
                vals = np.array(vals, dtype=float)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                return float(lo), float(hi)

            for q in cfg.q_values:
                # Generate examples
                rng = np.random.default_rng(cfg.seed + int(q * 1000))
                gold_indices = []
                is_pure_mode = []

                for _ in range(cfg.num_examples):
                    g = sample_gold_labels_bimodal(
                        n, k, w_pos, w_neg, imbalance
                    )
                    gold_indices.append(g)

                    # Check mode purity
                    W_pos = sum(w_pos[i] for i in g)
                    W_neg = sum(w_neg[i] for i in g)
                    total_weight = W_pos + W_neg
                    if total_weight > 0:
                        pure = (W_pos / total_weight >= purity_threshold) or (
                            W_neg / total_weight >= purity_threshold
                        )
                    else:
                        pure = True
                    is_pure_mode.append(pure)

                gold_names = study_a1.to_label_names(gold_indices, labels)

                # Generate predictions based on predictor type
                if pred_type == "bimodal":
                    preds = [
                        prototype_bimodal_predictor(
                            g, w_pos, w_neg, pos_prototypes, neg_prototypes, q
                        )
                        for g in gold_indices
                    ]
                elif pred_type == "within_mode":
                    preds = [
                        prototype_within_mode_predictor(
                            g,
                            w_pos,
                            w_neg,
                            pos_prototypes,
                            neg_prototypes,
                            k,
                            q,
                            2.0,
                        )
                        for g in gold_indices
                    ]

                pred_names = study_a1.to_label_names(
                    [sorted(set(x)) for x in preds], labels
                )

                # Separate pure and mixed examples
                pure_indices = [
                    i for i, is_pure in enumerate(is_pure_mode) if is_pure
                ]
                mixed_indices = [
                    i for i, is_pure in enumerate(is_pure_mode) if not is_pure
                ]

                if len(pure_indices) == 0 or len(mixed_indices) == 0:
                    # Skip if we don't have both types
                    continue

                # Evaluate on pure examples
                gold_pure = [gold_names[i] for i in pure_indices]
                pred_pure = [pred_names[i] for i in pure_indices]

                mlb_pure = MultiLabelBinarizer(classes=labels)
                Y_true_pure = mlb_pure.fit_transform(gold_pure)
                Y_pred_pure = mlb_pure.transform(pred_pure)

                if avg_type == "micro":
                    hard_pure_val = f1_score(
                        Y_true_pure,
                        Y_pred_pure,
                        average="micro",
                        zero_division=0,
                    )
                    sem_pure_val = float(
                        semantic_f1_score(
                            gold_pure,
                            pred_pure,
                            S_list["ideal"],
                            average="micro",
                        )
                    )
                    # CIs
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "hard_micro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1000 + int(q * 100),
                    )
                    hard_pure_lo.append(lo)
                    hard_pure_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "sem_micro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1001 + int(q * 100),
                    )
                    sem_pure_lo.append(lo)
                    sem_pure_hi.append(hi)
                elif avg_type == "macro":
                    hard_pure_val = f1_score(
                        Y_true_pure,
                        Y_pred_pure,
                        average="macro",
                        zero_division=0,
                    )
                    sem_pure_val = float(
                        semantic_f1_score(
                            gold_pure,
                            pred_pure,
                            S_list["ideal"],
                            average="macro",
                        )
                    )
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "hard_macro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1002 + int(q * 100),
                    )
                    hard_pure_lo.append(lo)
                    hard_pure_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "sem_macro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1003 + int(q * 100),
                    )
                    sem_pure_lo.append(lo)
                    sem_pure_hi.append(hi)
                else:  # samples
                    hard_pure_val = f1_score(
                        Y_true_pure,
                        Y_pred_pure,
                        average="samples",
                        zero_division=0,
                    )
                    sem_pure_val = float(
                        semantic_f1_score(
                            gold_pure,
                            pred_pure,
                            S_list["ideal"],
                            average="samples",
                        )
                    )
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "hard_samples",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1004 + int(q * 100),
                    )
                    hard_pure_lo.append(lo)
                    hard_pure_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_pure,
                        pred_pure,
                        "sem_samples",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1005 + int(q * 100),
                    )
                    sem_pure_lo.append(lo)
                    sem_pure_hi.append(hi)

                # Evaluate on mixed examples
                gold_mixed = [gold_names[i] for i in mixed_indices]
                pred_mixed = [pred_names[i] for i in mixed_indices]

                mlb_mixed = MultiLabelBinarizer(classes=labels)
                Y_true_mixed = mlb_mixed.fit_transform(gold_mixed)
                Y_pred_mixed = mlb_mixed.transform(pred_mixed)

                if avg_type == "micro":
                    hard_mixed_val = f1_score(
                        Y_true_mixed,
                        Y_pred_mixed,
                        average="micro",
                        zero_division=0,
                    )
                    sem_mixed_val = float(
                        semantic_f1_score(
                            gold_mixed,
                            pred_mixed,
                            S_list["ideal"],
                            average="micro",
                        )
                    )
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "hard_micro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1006 + int(q * 100),
                    )
                    hard_mixed_lo.append(lo)
                    hard_mixed_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "sem_micro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1007 + int(q * 100),
                    )
                    sem_mixed_lo.append(lo)
                    sem_mixed_hi.append(hi)
                elif avg_type == "macro":
                    hard_mixed_val = f1_score(
                        Y_true_mixed,
                        Y_pred_mixed,
                        average="macro",
                        zero_division=0,
                    )
                    sem_mixed_val = float(
                        semantic_f1_score(
                            gold_mixed,
                            pred_mixed,
                            S_list["ideal"],
                            average="macro",
                        )
                    )
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "hard_macro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1008 + int(q * 100),
                    )
                    hard_mixed_lo.append(lo)
                    hard_mixed_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "sem_macro",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1009 + int(q * 100),
                    )
                    sem_mixed_lo.append(lo)
                    sem_mixed_hi.append(hi)
                else:  # samples
                    hard_mixed_val = f1_score(
                        Y_true_mixed,
                        Y_pred_mixed,
                        average="samples",
                        zero_division=0,
                    )
                    sem_mixed_val = float(
                        semantic_f1_score(
                            gold_mixed,
                            pred_mixed,
                            S_list["ideal"],
                            average="samples",
                        )
                    )
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "hard_samples",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1010 + int(q * 100),
                    )
                    hard_mixed_lo.append(lo)
                    hard_mixed_hi.append(hi)
                    lo, hi = _boot_ci_single(
                        gold_mixed,
                        pred_mixed,
                        "sem_samples",
                        B=getattr(cfg, "bootstrap", 200),
                        seed=cfg.seed + 1011 + int(q * 100),
                    )
                    sem_mixed_lo.append(lo)
                    sem_mixed_hi.append(hi)

                q_values.append(q)
                hard_pure.append(hard_pure_val)
                sem_pure.append(sem_pure_val)
                hard_mixed.append(hard_mixed_val)
                sem_mixed.append(sem_mixed_val)

            if len(q_values) == 0:
                continue

            # Plot curves for this predictor and averaging type
            ax.plot(
                q_values,
                hard_pure,
                'o-',
                color='#4C78A8',
                label='Hard F1 (Pure)',
                linewidth=2,
                markersize=5,
            )
            # CI band for Hard (Pure)
            ax.fill_between(
                q_values,
                hard_pure_lo,
                hard_pure_hi,
                color='#4C78A8',
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                q_values,
                sem_pure,
                's-',
                color='#E45756',
                label='Semantic F1 (Pure)',
                linewidth=2,
                markersize=5,
            )
            # CI band for Semantic (Pure)
            ax.fill_between(
                q_values,
                sem_pure_lo,
                sem_pure_hi,
                color='#E45756',
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                q_values,
                hard_mixed,
                'o--',
                color='#54A24B',
                label='Hard F1 (Mixed)',
                linewidth=2,
                markersize=5,
            )
            # CI band for Hard (Mixed)
            ax.fill_between(
                q_values,
                hard_mixed_lo,
                hard_mixed_hi,
                color='#54A24B',
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                q_values,
                sem_mixed,
                's--',
                color='#FF9D9A',
                label='Semantic F1 (Mixed)',
                linewidth=2,
                markersize=5,
            )
            # CI band for Semantic (Mixed)
            ax.fill_between(
                q_values,
                sem_mixed_lo,
                sem_mixed_hi,
                color='#FF9D9A',
                alpha=0.12,
                linewidth=0,
            )

            ax.set_xlabel("q")
            # Update samples label wording only for samples row
            ax.set_ylabel(f"{avg_type.title()} F1 score")
            ax.set_title(f"{pred_title} — {avg_type.capitalize()}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)

            # Set y-limits to encompass all data
            all_vals = hard_pure + sem_pure + hard_mixed + sem_mixed
            if all_vals:
                y_min = max(0, min(all_vals) - 0.05)
                y_max = min(1, max(all_vals) + 0.05)
                ax.set_ylim(y_min, y_max)

    # Remove figure code from title (no "Fig A8")
    plt.suptitle(
        "Study A.2 — Stratified Analysis: Pure vs Mixed Mode Gold",
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()

    fig_path_png = os.path.join(outdir, f"fig_A8_stratified_curves_{ts}.png")
    fig_path_pdf = os.path.join(outdir, f"fig_A8_stratified_curves_{ts}.pdf")
    fig.savefig(fig_path_png, dpi=200)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    logger.info("Wrote: %s", fig_path_png)
    logger.info("Wrote: %s", fig_path_pdf)
    print(f"Wrote: {fig_path_png}")
    print(f"Wrote: {fig_path_pdf}")


def write_bootstrap_table(
    df_boot: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Write LaTeX table for bootstrap mean ± 95% CI for all predictors (Tab A2).

    Creates a focused table showing key comparisons between predictors.
    """
    if df_boot.empty:
        logger.warning("No bootstrap data available for table")
        return

    def _latex_escape(s: str) -> str:
        return str(s).replace("_", "\\_")

    # Filter to most important comparisons and representative parameters
    key_comparisons = [
        ("bimodal", "near"),
        ("within_mode", "bimodal"),
        ("bimodal", "far"),
    ]

    # Use median parameter values for cleaner table, but be flexible for small datasets
    if not df_boot.empty:
        k_med = df_boot["k"].median()
        m_med = df_boot["m"].median()
        imb_med = df_boot["imbalance"].median()

        # First try with median filtering
        df_filtered = df_boot[
            (df_boot["k"] == k_med)
            & (df_boot["m"] == m_med)
            & (df_boot["imbalance"] == imb_med)
            & (df_boot["pred_a"].isin([comp[0] for comp in key_comparisons]))
            & (df_boot["pred_b"].isin([comp[1] for comp in key_comparisons]))
        ].copy()

        # If no data after filtering, be less restrictive
        if df_filtered.empty:
            logger.warning(
                "No data with median parameters, using all available data"
            )
            df_filtered = df_boot[
                (df_boot["pred_a"].isin([comp[0] for comp in key_comparisons]))
                & (
                    df_boot["pred_b"].isin(
                        [comp[1] for comp in key_comparisons]
                    )
                )
            ].copy()
    else:
        return

    # Focus on key metrics
    key_metrics = [
        "hard_micro",
        "sem_micro:ideal",
        "hard_macro",
        "sem_macro:ideal",
    ]
    df_filtered = df_filtered[df_filtered["metric"].isin(key_metrics)].copy()

    if df_filtered.empty:
        logger.warning("No data available for bootstrap table after filtering")
        return

    rows = []
    for _, row in df_filtered.sort_values(
        ["pred_a", "pred_b", "q", "metric"]
    ).iterrows():
        pred_a = _latex_escape(row["pred_a"])
        pred_b = _latex_escape(row["pred_b"])
        q = float(row["q"])
        metric = _latex_escape(row["metric"])
        mean = float(row["mean_gap"])
        lo = float(row["ci95_lo"])
        hi = float(row["ci95_hi"])

        rows.append(
            f"{pred_a} & {pred_b} & {q:.2f} & {metric} & {mean:.3f} & [{lo:.3f}, {hi:.3f}] \\\\"
        )

    tex = []
    tex.append("% Auto-generated by study_a2.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{llrlll}")
    tex.append("\\toprule")
    tex.append(
        "Predictor A & Predictor B & q & Metric & Mean Gap & 95\\% CI \\\\"
    )
    tex.append("\\midrule")
    tex.extend(rows)
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    # Build caption based on available data
    if 'k_med' in locals() and 'm_med' in locals() and 'imb_med' in locals():
        caption = f"\\caption{{Study A.2: Bootstrap mean gaps (A$-$B) $\\pm$ 95\\% CI for k={int(k_med)}, m={int(m_med)}, imbalance={imb_med:.1f}.}}"
    else:
        caption = "\\caption{Study A.2: Bootstrap mean gaps (A$-$B) $\\pm$ 95\\% CI for key parameter combinations.}"

    tex.append(caption)
    tex.append("\\label{tab:a2_bootstrap}")
    tex.append("\\end{table}")

    tab_path = os.path.join(outdir, f"tab_A2_bootstrap_gaps_{ts}.tex")
    with open(tab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    logger.info("Wrote: %s", tab_path)
    print(f"Wrote: {tab_path}")


def plot_predictor_comparison_heatmaps(
    df_summary: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create heatmaps comparing different predictors across parameter space.

    Shows performance differences between bimodal, within-mode, near, and far predictors.
    """
    if df_summary.empty:
        return

    # Focus on micro metrics for clarity
    metrics_to_plot = ["hard_micro", "sem_micro:ideal"]
    predictors_to_plot = ["near", "far", "bimodal", "within_mode"]

    for metric in metrics_to_plot:
        df_metric = df_summary[
            (df_summary["metric"] == metric)
            & (df_summary["predictor"].isin(predictors_to_plot))
        ].copy()

        if df_metric.empty:
            continue

        # Use median m and imbalance for cleaner visualization
        m_med = df_metric["m"].median()
        imb_med = df_metric["imbalance"].median()
        df_filtered = df_metric[
            (df_metric["m"] == m_med) & (df_metric["imbalance"] == imb_med)
        ].copy()

        if df_filtered.empty:
            continue

        # Create subplot for each predictor
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, predictor in enumerate(predictors_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            pred_data = df_filtered[
                df_filtered["predictor"] == predictor
            ].copy()
            if pred_data.empty:
                ax.set_title(f"{predictor} (No Data)")
                continue

            # Create heatmap data
            heatmap_data = pred_data.pivot_table(
                index="k", columns="q", values="value", aggfunc="mean"
            )

            if heatmap_data.empty:
                ax.set_title(f"{predictor} (No Data)")
                continue

            im = ax.imshow(heatmap_data.values, cmap="viridis", aspect="auto")

            # Set labels
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([f"{q:.2f}" for q in heatmap_data.columns])
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels([f"{k}" for k in heatmap_data.index])

            ax.set_xlabel("q")
            ax.set_ylabel("k (labels per example)")
            ax.set_title(predictor.replace("_", "-").title())

        metric_name = "Hard F1" if metric.startswith("hard") else "Semantic F1"
        plt.suptitle(
            f"Study A.2 — Predictor Comparison — {metric_name} (micro)",
            fontweight="bold",
        )
        plt.tight_layout()

        fig_path_png = os.path.join(
            outdir, f"fig_A2_pred_comparison_{metric}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A2_pred_comparison_{metric}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)

        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")
