from __future__ import annotations

from typing import Any, Callable
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR

DECEPTIVE_COLOR = THIRD_COLOR  # Green color for deceptive Euclidean
PERMUTED_COLOR = "#9467BD"  # Purple color for permuted control


def plot_metric_vs_radius_grids(
    cfg: Any,
    labels: list[str],
    S_ideal: pd.DataFrame,
    S_deceptive: pd.DataFrame | None,
    helpers: dict[str, Any],
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Render three k×p grids (micro/macro/samples) with CIs and shared legend."""

    if not cfg.k_values or not cfg.p_values:
        return

    radii_union = sorted(set(cfg.near_radii + cfg.far_radii))
    if len(radii_union) < 1:
        return

    nrows = len(cfg.k_values)
    p_values = [p for p in deepcopy(cfg.p_values) if p != 0]
    ncols = len(p_values)

    fig_w = max(5.5, 3.2 * ncols)
    fig_h = max(4.5, 2.4 * nrows + 0.8)

    sample_gold_labels: Callable = helpers["sample_gold_labels"]
    to_label_names: Callable = helpers["to_label_names"]
    perturb_labels: Callable = helpers["perturb_labels"]
    semantic_f1_score: Callable = helpers["semantic_f1_score"]
    f1_score: Callable = helpers["f1_score"]
    MultiLabelBinarizer = helpers["MultiLabelBinarizer"]

    S_map: dict[str, pd.DataFrame] = {"ideal": S_ideal}
    if S_deceptive is not None:
        S_map["deceptive"] = S_deceptive
    sem_variant_order = list(S_map.keys())

    def compute_metrics_with_ci(
        T_names: list[list[str]],
        P_names: list[list[str]],
        class_labels: list[str],
        S_lookup: dict[str, pd.DataFrame],
        B: int,
        seed: int,
    ) -> dict[str, tuple[float, float, float]]:
        rng = np.random.default_rng(seed)
        n_ex = len(T_names)
        idx_base = np.arange(n_ex)
        mlb = MultiLabelBinarizer(classes=class_labels)
        mlb.fit([[]])

        def eval_all(T_sel, P_sel):
            Yt = mlb.transform(T_sel)
            Yp = mlb.transform(P_sel)
            out: dict[str, float] = {
                "hard_micro": float(
                    f1_score(Yt, Yp, average="micro", zero_division=0)
                ),
                "hard_macro": float(
                    f1_score(Yt, Yp, average="macro", zero_division=0)
                ),
                "hard_samples": float(
                    f1_score(Yt, Yp, average="samples", zero_division=0)
                ),
            }
            for name, S in S_lookup.items():
                out[f"sem_micro:{name}"] = float(
                    semantic_f1_score(T_sel, P_sel, S, average="micro")
                )
                out[f"sem_macro:{name}"] = float(
                    semantic_f1_score(T_sel, P_sel, S, average="macro")
                )
                out[f"sem_samples:{name}"] = float(
                    semantic_f1_score(T_sel, P_sel, S, average="samples")
                )
            return out

        draw_keys = [
            "hard_micro",
            "hard_macro",
            "hard_samples",
        ]
        for name in S_lookup:
            draw_keys.extend(
                [
                    f"sem_micro:{name}",
                    f"sem_macro:{name}",
                    f"sem_samples:{name}",
                ]
            )
        draws = {k: [] for k in draw_keys}

        for _ in range(B):
            sel = rng.choice(idx_base, size=n_ex, replace=True)
            T_sel = [T_names[i] for i in sel]
            P_sel = [P_names[i] for i in sel]
            vals = eval_all(T_sel, P_sel)
            for k in draws:
                draws[k].append(vals[k])

        stats = {}
        for key, arr in draws.items():
            a = np.array(arr, dtype=float)
            mean = float(a.mean())
            lo, hi = np.percentile(a, [2.5, 97.5])
            stats[key] = (mean, float(lo), float(hi))
        return stats

    kp_stats: dict[
        tuple[int, float], dict[str, dict[int, tuple[float, float, float]]]
    ] = {}

    for k_plot in cfg.k_values:
        gold_idx_cache = [
            sample_gold_labels(cfg.num_labels, k_plot)
            for _ in range(cfg.num_examples)
        ]
        gold_idx_cache = [sorted(set(g)) for g in gold_idx_cache]
        gold_names_cache = to_label_names(gold_idx_cache, labels)

        for p_plot in p_values:
            series_stats: dict[str, dict[int, tuple[float, float, float]]] = {}
            for r in radii_union:
                preds_idx = [
                    perturb_labels(g, cfg.num_labels, p=p_plot, radius=r)
                    for g in gold_idx_cache
                ]
                preds_idx = [sorted(set(x)) for x in preds_idx]
                preds_names = to_label_names(preds_idx, labels)

                stats = compute_metrics_with_ci(
                    gold_names_cache,
                    preds_names,
                    labels,
                    S_map,
                    B=cfg.bootstrap,
                    seed=cfg.seed,
                )
                for mkey, triple in stats.items():
                    series_stats.setdefault(mkey, {})[r] = triple
            kp_stats[(k_plot, float(p_plot))] = series_stats

    def display_label(series_key: str) -> str:
        if series_key.startswith("hard"):
            return "Hard F1"
        variant = series_key.split(":", 1)[1]
        if variant == "ideal":
            return "Sem F1 (ideal)"
        if variant == "deceptive":
            return "Sem F1 (decep)"
        if variant == "permuted":
            return "Sem F1 (perm)"
        return f"Sem F1 ({variant})"

    sem_style_lookup = {
        "ideal": dict(color=SEM_COLOR, linestyle="-", marker="s"),
        "deceptive": dict(color=DECEPTIVE_COLOR, linestyle="--", marker="^"),
        "permuted": dict(color=PERMUTED_COLOR, linestyle="-.", marker="D"),
    }

    for avg_name in ["micro", "macro", "samples"]:
        series_keys = [f"hard_{avg_name}"] + [
            f"sem_{avg_name}:{variant}" for variant in sem_variant_order
        ]

        series_styles: dict[str, dict[str, Any]] = {}
        for key in series_keys:
            if key.startswith("hard"):
                series_styles[key] = dict(
                    color=HARD_COLOR, linestyle="-", marker="o"
                )
            else:
                variant = key.split(":", 1)[1]
                base_style = sem_style_lookup.get(
                    variant,
                    dict(color="#666666", linestyle="-.", marker="D"),
                )
                series_styles[key] = base_style

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        shared_handles = None
        shared_labels = None

        for i, k_plot in enumerate(cfg.k_values):
            for j, p_plot in enumerate(p_values):
                ax = axes[i][j]
                series_stats = kp_stats[(k_plot, float(p_plot))]
                for series_key in series_keys:
                    rmap = series_stats.get(series_key, {})
                    rs = sorted(rmap.keys())
                    if not rs:
                        continue
                    vals = [rmap[x][0] for x in rs]
                    los = [rmap[x][1] for x in rs]
                    his = [rmap[x][2] for x in rs]
                    style = series_styles[series_key]
                    ax.fill_between(
                        rs,
                        los,
                        his,
                        color=style["color"],
                        alpha=0.15,
                        linewidth=0,
                    )
                    ax.plot(
                        rs,
                        vals,
                        label=display_label(series_key),
                        **style,
                    )
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)

                if shared_handles is None or shared_labels is None:
                    handles, labels_ = ax.get_legend_handles_labels()
                    shared_handles, shared_labels = handles, labels_
                ax.legend().remove()

        for j, pval in enumerate(p_values):
            axes[0][j].set_title(f"p={pval:.2f}", fontsize=16)

        for i, kval in enumerate(cfg.k_values):
            bbox = axes[i][0].get_position()
            x = bbox.x0 - 0.03
            y = bbox.y0 + bbox.height / 2
            fig.text(
                x,
                y,
                f"k={kval}",
                rotation=90,
                va="center",
                ha="right",
                fontsize=18,
            )

        fig.suptitle(
            f"{avg_name.capitalize()} F1 score grid",
            fontweight="bold",
            fontsize=20,
        )

        fig.supxlabel("Hop Radius r", fontsize=16, y=0.05)

        if shared_handles and shared_labels:
            fig.legend(
                handles=shared_handles,
                labels=shared_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(series_keys), 3),
                frameon=False,
                fontsize=22,
            )

        fig_path_png = os.path.join(
            outdir, f"fig_A1_metric_vs_radius_grid_{avg_name}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A1_metric_vs_radius_grid_{avg_name}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")


def plot_cross_jump_prob_grids(
    df_summary: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Render three k×cross_jump_prob grids (micro/macro/samples), averaging across p values."""

    if df_summary.empty or 'cross_jump_prob' not in df_summary.columns:
        return

    # Get unique values
    ks = sorted(df_summary["k"].unique())
    # Get unique values
    ks = sorted(df_summary["k"].unique())
    cross_jump_probs = sorted(df_summary["cross_jump_prob"].unique())

    if len(ks) == 0 or len(cross_jump_probs) == 0:
        return

    nrows = len(ks)
    ncols = len(cross_jump_probs)

    fig_w = max(5.5, 3.2 * ncols)
    fig_h = max(4.5, 2.4 * nrows + 0.8)

    def display_label(series_key: str) -> str:
        if series_key.startswith("hard"):
            return "Hard F1"
        if "ideal" in series_key:
            return "Sem F1 (ideal)"
        if "deceptive" in series_key:
            return "Sem F1 (decep)"
        if "permuted" in series_key:
            return "Sem F1 (perm)"
        return f"Sem F1"

    sem_style_lookup = {
        "ideal": dict(color=SEM_COLOR, linestyle="-", marker="s"),
        "deceptive": dict(color=DECEPTIVE_COLOR, linestyle="--", marker="^"),
        "permuted": dict(color=PERMUTED_COLOR, linestyle="-.", marker="D"),
    }

    for avg_name in ["micro", "macro", "samples"]:

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        shared_handles = None
        shared_labels = None

        for i, k_val in enumerate(ks):
            for j, cjp_val in enumerate(cross_jump_probs):
                ax = axes[i][j]

                # Filter data for this k and cross_jump_prob, average across p values
                subset = df_summary[
                    (df_summary["k"] == k_val)
                    & (df_summary["cross_jump_prob"] == cjp_val)
                ].copy()

                if subset.empty:
                    continue

                # Average across p values, group by S and radius pairs
                avg_data = (
                    subset.groupby(['S', 'r_near', 'r_far'])
                    .agg(
                        {
                            f'sem_{avg_name}_A': 'mean',
                            f'sem_{avg_name}_B': 'mean',
                            f'hard_{avg_name}_A': 'mean',
                            f'hard_{avg_name}_B': 'mean',
                        }
                    )
                    .reset_index()
                )

                # Plot hard F1 - combine near and far radii
                hard_radii = []
                hard_values = []

                for _, row in avg_data.iterrows():
                    hard_radii.extend([row['r_near'], row['r_far']])
                    hard_values.extend(
                        [row[f'hard_{avg_name}_A'], row[f'hard_{avg_name}_B']]
                    )

                if hard_radii:
                    # Remove duplicates and sort
                    radius_value_pairs = list(zip(hard_radii, hard_values))
                    unique_pairs = {}
                    for r, v in radius_value_pairs:
                        if r not in unique_pairs:
                            unique_pairs[r] = []
                        unique_pairs[r].append(v)

                    # Average values for same radius
                    radii_sorted = sorted(unique_pairs.keys())
                    values_averaged = [
                        np.mean(unique_pairs[r]) for r in radii_sorted
                    ]

                    ax.plot(
                        radii_sorted,
                        values_averaged,
                        label=display_label(f"hard_{avg_name}"),
                        color=HARD_COLOR,
                        linestyle="-",
                        marker="o",
                    )

                # Plot semantic F1 for each S variant
                for s_val in ['ideal', 'deceptive', 'permuted']:
                    s_data = avg_data[avg_data['S'] == s_val]

                    if not s_data.empty:
                        sem_radii = []
                        sem_values = []

                        for _, row in s_data.iterrows():
                            sem_radii.extend([row['r_near'], row['r_far']])
                            sem_values.extend(
                                [
                                    row[f'sem_{avg_name}_A'],
                                    row[f'sem_{avg_name}_B'],
                                ]
                            )

                        if sem_radii:
                            # Remove duplicates and sort
                            radius_value_pairs = list(
                                zip(sem_radii, sem_values)
                            )
                            unique_pairs = {}
                            for r, v in radius_value_pairs:
                                if r not in unique_pairs:
                                    unique_pairs[r] = []
                                unique_pairs[r].append(v)

                            # Average values for same radius
                            radii_sorted = sorted(unique_pairs.keys())
                            values_averaged = [
                                np.mean(unique_pairs[r]) for r in radii_sorted
                            ]

                            style = sem_style_lookup.get(
                                s_val,
                                dict(
                                    color="#666666", linestyle="-.", marker="D"
                                ),
                            )
                            ax.plot(
                                radii_sorted,
                                values_averaged,
                                label=display_label(f"sem_{avg_name}:{s_val}"),
                                **style,
                            )

                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)

                if shared_handles is None or shared_labels is None:
                    handles, labels_ = ax.get_legend_handles_labels()
                    shared_handles, shared_labels = handles, labels_
                ax.legend().remove()

        # Set titles and labels
        for j, cjp in enumerate(cross_jump_probs):
            axes[0][j].set_title(
                rf"$\mathbf{{p_{{\mathrm{{jump}}}}={cjp:.2f}}}$",
                fontsize=16,
            )

            # Set a main figure title if not already set
            fig.suptitle(
                rf"Cross-jump probability grid: {avg_name.title()} F1 scores by k and $p_{{\mathrm{{jump}}}}$",
                fontweight="bold",
                fontsize=20,
            )

        for i, kval in enumerate(ks):
            bbox = axes[i][0].get_position()
            fig.text(
                bbox.x0 - 0.02,
                bbox.y0 + bbox.height / 2,
                f"k={kval}",
                rotation=90,
                va="center",
                ha="right",
                fontsize=16,
                weight="bold",
            )

        for i in range(nrows):
            axes[i][-1].set_ylabel("F1 Score")
        for j in range(ncols):
            axes[-1][j].set_xlabel("Radius")

        if shared_handles and shared_labels:
            fig.legend(
                shared_handles,
                shared_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.08),  # Move legend up to avoid cutoff
                ncol=len(shared_labels),
                frameon=False,  # Remove border
                fontsize=14,
            )

        fig_path_png = os.path.join(
            outdir, f"fig_A9_cross_jump_prob_grid_{avg_name}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A9_cross_jump_prob_grid_{avg_name}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")


def plot_kendall_tau_bars(
    df_tau_all: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Aggregate bar chart for negative Kendall tau (micro only).

    - Uses -tau so that larger bars indicate stronger monotonic decrease.
    - Excludes alpha=0 and alpha=1 semantic variants (they collapse to other cases).
    """
    if df_tau_all.empty:
        return

    # Focus on micro variants; exclude all alpha mixes entirely
    df = df_tau_all[df_tau_all["metric"].str.contains("micro")].copy()
    df = df[~df["metric"].str.contains(":alpha=")].copy()

    # Keep only core variants for legend simplicity
    allowed = {
        "hard_micro",
        "sem_micro:ideal",
        "sem_micro:permuted",
        "sem_micro:deceptive",
    }
    df = df[df["metric"].isin(allowed)].copy()

    # Aggregate mean negative tau per metric across (k,p)
    tau_agg = df.groupby("metric")["tau"].mean().reset_index()
    tau_agg["neg_tau"] = -tau_agg["tau"].astype(float)

    def _disp(m: str) -> str:
        if m == "hard_micro":
            return "Hard"
        if m == "sem_micro:ideal":
            return "Semantic (ideal)"
        if m == "sem_micro:permuted":
            return "Semantic (permute)"
        if m == "sem_micro:deceptive":
            return "Semantic (deceptive)"
        return m

    tau_agg["label"] = tau_agg["metric"].apply(_disp)
    tau_agg = tau_agg.sort_values("neg_tau", ascending=False)

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.bar(range(len(tau_agg)), tau_agg["neg_tau"], color="#4C78A8")
    ax.set_xticks(range(len(tau_agg)))
    ax.set_xticklabels(tau_agg["label"], rotation=30, ha="right")
    ax.set_ylabel("Negative Kendall tau vs radius")
    ax.set_title("Negative Kendall tau (Higher is Better) — Micro F1 Score")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig_a2_png = os.path.join(outdir, f"fig_A2_kendall_tau_bars_{ts}.png")
    fig_a2_pdf = os.path.join(outdir, f"fig_A2_kendall_tau_bars_{ts}.pdf")
    fig.savefig(fig_a2_png, dpi=200)
    fig.savefig(fig_a2_pdf)
    plt.close(fig)
    logger.info("Wrote: %s", fig_a2_png)
    logger.info("Wrote: %s", fig_a2_pdf)
    print(f"Wrote: {fig_a2_png}")
    print(f"Wrote: {fig_a2_pdf}")


def plot_kendall_tau_grids(
    df_tau_all: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Facet grids of negative Kendall tau bars for each averaging metric.

    Produces three figures (micro/macro/samples), each a k×p grid. Every subplot
    shows bars for Hard vs Semantic variants (S choices). Alpha endpoints are removed.
    """
    if df_tau_all.empty:
        return

    # Helper: filter to core variants only (exclude all alpha mixes)
    def _keep_metric(name: str, avg: str) -> bool:
        core = {
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:permuted",
            f"sem_{avg}:deceptive",
        }
        return name in core

    def _variant_label(m: str, avg: str) -> str:
        if m == f"hard_{avg}":
            return "Hard"
        if m == f"sem_{avg}:ideal":
            return "Semantic (ideal)"
        if m == f"sem_{avg}:permuted":
            return "Semantic (permute)"
        if m == f"sem_{avg}:deceptive":
            return "Semantic (deceptive)"
        return m

    ks = sorted(df_tau_all["k"].unique()) if "k" in df_tau_all.columns else []
    ps = sorted(df_tau_all["p"].unique()) if "p" in df_tau_all.columns else []
    ps = [p for p in ps if p != 0.0]  # exclude p=0
    if not ks or not ps:
        return

    # iterate over averaging schemes
    for avg in ["micro", "macro", "samples"]:
        df = df_tau_all.copy()
        df = df[df["metric"].apply(lambda m: _keep_metric(m, avg))]
        if df.empty:
            continue

        # Build a consistent metric order across subplots
        desired_order = [
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:deceptive",
            f"sem_{avg}:permuted",
        ]
        metrics_order = [
            m for m in desired_order if m in set(df["metric"].unique())
        ]

        # Style mapping (consistent across the grid)
        # Fixed styles for clarity
        style_map: dict[str, dict[str, object]] = {
            f"hard_{avg}": {"color": HARD_COLOR, "marker": "o"},
            f"sem_{avg}:ideal": {"color": SEM_COLOR, "marker": "s"},
            f"sem_{avg}:deceptive": {
                "color": DECEPTIVE_COLOR,
                "marker": "^",
            },
            f"sem_{avg}:permuted": {
                "color": THIRD_COLOR,
                "marker": "D",
            },
        }

        # Precompute mean -tau per (k,p,metric)
        df_vals = (
            df.assign(neg_tau=lambda x: -x["tau"].astype(float))
            .groupby(["k", "p", "metric"], as_index=False)["neg_tau"]
            .mean()
        )

        # Styles and labels for grouped bars
        style_map: dict[str, dict[str, object]] = {
            f"hard_{avg}": {"color": HARD_COLOR, "label": "Hard"},
            f"sem_{avg}:ideal": {
                "color": SEM_COLOR,
                "label": "Semantic (ideal)",
            },
            f"sem_{avg}:deceptive": {
                "color": DECEPTIVE_COLOR,
                "label": "Semantic (deceptive)",
            },
            f"sem_{avg}:permuted": {
                "color": THIRD_COLOR,
                "label": "Semantic (permute)",
            },
        }

        # Prepare single figure with one subplot per k stacked vertically
        ps_sorted = sorted(ps)
        M = len(metrics_order)
        # Compute global y-range with padding across all ks
        if df_vals.empty:
            continue
        gmax = float(df_vals["neg_tau"].max())
        gmin = float(df_vals["neg_tau"].min())
        yr = max(1e-6, gmax - gmin)
        pad = 0.08 * yr + 0.05
        y_max = max(0.0, gmax + pad)
        y_min = min(0.0, gmin - pad)

        fig_w = max(7.0, 1.6 + 1.0 * len(ps_sorted))
        fig_h = max(3.0, 2.4 * len(ks) + 1.1)  # room for legend and titles
        fig, axes = plt.subplots(
            nrows=len(ks),
            ncols=1,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharex=True,
        )

        x = np.arange(len(ps_sorted))
        bar_w = min(0.22, 0.7 / max(1, M))
        offsets = np.linspace(-bar_w * (M - 1) / 2, bar_w * (M - 1) / 2, M)

        for i, k_val in enumerate(ks):
            ax = axes[i][0]
            subk = df_vals[df_vals["k"] == k_val]
            for j, m in enumerate(metrics_order):
                heights = []
                for p_val in ps_sorted:
                    row = subk[(subk["p"] == p_val) & (subk["metric"] == m)]
                    heights.append(
                        float(row.iloc[0]["neg_tau"]) if not row.empty else 0.0
                    )
                color = style_map[m]["color"]
                label = style_map[m]["label"]
                bars = ax.bar(
                    x + offsets[j],
                    heights,
                    width=bar_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    label=label,
                )
                # Annotate each bar with numeric value (clamped inside y-limits)
                for rect, val in zip(bars, heights):
                    txt = f"{val:.2f}"
                    y = rect.get_height()
                    dy = 0.02 * (y_max - y_min)
                    y_anno = y + (dy if y >= 0 else -dy)
                    # clamp within limits
                    y_anno = min(max(y_anno, y_min + 0.01), y_max - 0.01)
                    va = "bottom" if y >= 0 else "top"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        y_anno,
                        txt,
                        ha="center",
                        va=va,
                        fontsize=8,
                    )

            ax.axhline(0.0, color="#000000", alpha=0.2, linewidth=1.0)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylabel(f"k={k_val}")

        axes[-1][0].set_xticks(x)
        axes[-1][0].set_xticklabels([f"{p:.2f}" for p in ps_sorted])
        axes[-1][0].set_xlabel("p", fontsize=16)

        # Shared legend at bottom
        handles = []
        labels_ = []
        for m in metrics_order:
            handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=style_map[m]["color"],
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
            labels_.append(style_map[m]["label"])
        fig.legend(
            handles,
            labels_,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=min(3, len(metrics_order)),
            frameon=False,
            fontsize=15,
        )
        # plt.subplots_adjust(bottom=0.15)

        def _avg_full_name(a: str) -> str:
            return {
                "micro": "Micro F1",
                "macro": "Macro F1",
                "samples": "Sample F1",
            }.get(a, a)

        fig.suptitle(
            f"Negative Kendall τ — {_avg_full_name(avg)}",
            fontweight="bold",
            fontsize=20,
        )

        fig_path_png = os.path.join(
            outdir, f"fig_A2_kendall_tau_group_{avg}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A2_kendall_tau_group_{avg}_{ts}.pdf"
        )
        fig.tight_layout(rect=[0, 0.035, 1, 1])
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
    """Write LaTeX table for bootstrap mean ± 95% CI per configuration.

    Mirrors the previous inline implementation from study_a1.py.
    """
    if df_boot.empty:
        return

    def _latex_escape(s: str) -> str:
        return str(s).replace("_", "\\_")

    rows = []
    for _, row in df_boot.sort_values(
        ["k", "p", "r_near", "r_far", "S", "metric"]
    ).iterrows():
        k_ = int(row["k"])  # noqa: N806
        p_ = float(row["p"])  # noqa: N806
        rn = int(row["r_near"])  # noqa: N806
        rf = int(row["r_far"])  # noqa: N806
        Sname = _latex_escape(row["S"])  # noqa: N806
        metric = _latex_escape(row["metric"])  # noqa: N806
        mean = float(row["mean_gap"])  # noqa: N806
        lo = float(row["ci95_lo"])  # noqa: N806
        hi = float(row["ci95_hi"])  # noqa: N806
        rows.append(
            f"{k_} & {p_:.2f} & {rn} & {rf} & {Sname} & {metric} & {mean:.3f} [$ {lo:.3f}$, $ {hi:.3f}$ ] \\\\"
        )

    tex = []
    tex.append("% Auto-generated by study_a1.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{rrrrlll}")
    tex.append(
        "k & p & r\\_near & r\\_far & S & metric & mean $\\pm$ 95\\% CI \\\\"
    )
    tex.append("\\hline")
    tex.extend(rows)
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{Study A.1: Bootstrap mean gap (A--B) $\\pm$ 95\\% CI across configurations.}"
    )
    tex.append("\\label{tab:a1_bootstrap}")
    tex.append("\\end{table}")

    tab_path = os.path.join(outdir, f"tab_A1_bootstrap_gaps_{ts}.tex")
    with open(tab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    logger.info("Wrote: %s", tab_path)
    print(f"Wrote: {tab_path}")


def plot_kendall_tau_linepanels(
    df_tau_all: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Produce per-metric figures with line plots across p, one row per k.

    Uses negative Kendall tau, core variants only: Hard, Semantic (ideal),
    Semantic (deceptive), Semantic (permute).
    """
    if df_tau_all.empty:
        return

    def _keep_metric(name: str, avg: str) -> bool:
        core = {
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:deceptive",
            f"sem_{avg}:permuted",
        }
        return name in core

    ks = sorted(df_tau_all["k"].unique()) if "k" in df_tau_all.columns else []
    ps = sorted(df_tau_all["p"].unique()) if "p" in df_tau_all.columns else []
    if not ks or not ps:
        return

    for avg in ["micro", "macro", "samples"]:
        df = df_tau_all.copy()
        df = df[df["metric"].apply(lambda m: _keep_metric(m, avg))]
        if df.empty:
            continue

        order = [
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:deceptive",
            f"sem_{avg}:permuted",
        ]
        metrics_order = [m for m in order if m in set(df["metric"].unique())]

        style_map: dict[str, dict[str, object]] = {
            f"hard_{avg}": {
                "color": "#4C78A8",
                "marker": "o",
                "label": "Hard",
                "ls": "--",
            },
            f"sem_{avg}:ideal": {
                "color": "#F58518",
                "marker": "s",
                "label": "Semantic (ideal)",
                "ls": "--",
            },
            f"sem_{avg}:deceptive": {
                "color": DECEPTIVE_COLOR,
                "marker": "d",
                "label": "Semantic (deceptive)",
                "ls": "--",
            },
            f"sem_{avg}:permuted": {
                "color": THIRD_COLOR,
                "marker": "^",
                "label": "Semantic (permute)",
                "ls": "--",
            },
        }

        df_vals = (
            df.assign(neg_tau=lambda x: -x["tau"].astype(float))
            .groupby(["k", "p", "metric"], as_index=False)["neg_tau"]
            .mean()
        )
        if df_vals.empty:
            continue

        # y-range with padding
        gmax = float(df_vals["neg_tau"].max())
        gmin = float(df_vals["neg_tau"].min())
        yr = max(1e-6, gmax - gmin)
        pad = 0.08 * yr + 0.02
        y_max = max(0.0, gmax + pad)
        y_min = min(0.0, gmin - pad)

        ps_sorted = sorted(ps)
        fig_w = max(3.8, 0.8 + 0.3 * len(ps_sorted))
        fig_h = max(3.0, 2.2 * len(ks) + 1.2)
        fig, axes = plt.subplots(
            nrows=len(ks),
            ncols=1,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharex=True,
        )

        for i, k_val in enumerate(ks):
            ax = axes[i][0]
            subk = df_vals[df_vals["k"] == k_val]
            for m in metrics_order:
                style = style_map[m]
                ys = []
                for p_val in ps_sorted:
                    row = subk[(subk["p"] == p_val) & (subk["metric"] == m)]
                    ys.append(
                        float(row.iloc[0]["neg_tau"]) if not row.empty else 0.0
                    )
                ax.plot(
                    ps_sorted,
                    ys,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.6,
                    linestyle=style.get("ls", "--"),
                )
            ax.axhline(0.0, color="#000000", alpha=0.2, linewidth=1.0)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylabel(f"k={k_val}")
            # Set x-ticks to only the evaluated p values
            ax.set_xticks(ps_sorted)
            if i < len(ks) - 1:
                ax.set_xticklabels([])

        axes[-1][0].set_xlabel("p")
        axes[-1][0].set_xticklabels([f"{p:.2f}" for p in ps_sorted])
        # Shared legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(2, len(metrics_order)),
            frameon=False,
        )

        def _avg_full_name(a: str) -> str:
            return {
                "micro": "Micro F1 Score",
                "macro": "Macro F1 Score",
                "samples": "Examples F1 Score",
            }.get(a, a)

        # Short, multi-line title to reduce width
        fig.suptitle(
            "\n".join(
                [
                    "Study A.1 — Negative Kendall tau ↑",
                    f"{_avg_full_name(avg)}",
                ]
            ),
            fontweight="bold",
            fontsize=11,
            y=0.96,
            linespacing=1.2,
        )
        plt.subplots_adjust(
            top=0.85, bottom=0.20, left=0.18, right=0.98, hspace=0.32
        )

        fig_path_png = os.path.join(
            outdir, f"fig_A2_kendall_tau_lines_{avg}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A2_kendall_tau_lines_{avg}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        logger.info("Wrote: %s", fig_path_pdf)
        print(f"Wrote: {fig_path_png}")
        print(f"Wrote: {fig_path_pdf}")


def plot_bootstrap_heatmaps(
    df_boot: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create heatmap visualizations of bootstrap mean gaps instead of long table.

    Produces focused heatmaps for key parameter combinations:
    1. k×p heatmaps for each metric type (hard/semantic) under ideal S
    2. S×metric heatmap for representative (k,p) values with proper labels
    """
    if df_boot.empty:
        return

    import seaborn as sns

    # Filter to most important cases for cleaner visualization
    df = df_boot.copy()

    # 1. Primary heatmaps: k×p for ideal S, separate for each metric type
    df_ideal = df[df["S"] == "ideal"].copy()
    if not df_ideal.empty:
        # Group metrics by type
        metric_groups = {
            "micro": ["hard_micro", "sem_micro"],
            "macro": ["hard_macro", "sem_macro"],
            "samples": ["hard_samples", "sem_samples"],
        }

        for avg_type, metrics in metric_groups.items():
            df_metrics = df_ideal[df_ideal["metric"].isin(metrics)].copy()
            if df_metrics.empty:
                continue

            # Aggregate across radius pairs for each metric
            df_agg = (
                df_metrics.groupby(["k", "p", "metric"])["mean_gap"]
                .mean()
                .reset_index()
            )

            # Create subplot for hard vs semantic comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            import numpy as np

            # Calculate data for both metrics to get shared scale
            pivot_data = {}
            for metric in metrics:
                df_metric = df_agg[df_agg["metric"] == metric]
                if not df_metric.empty:
                    pivot_data[metric] = df_metric.pivot(
                        index="k", columns="p", values="mean_gap"
                    )

            # Get common color scale across both heatmaps
            if len(pivot_data) == 2:
                all_values = []
                for pivot in pivot_data.values():
                    all_values.extend(pivot.values.flatten())
                all_values = [v for v in all_values if not np.isnan(v)]
                vmin, vmax = min(all_values), max(all_values)
            else:
                vmin, vmax = None, None

            for i, metric in enumerate(metrics):
                if metric not in pivot_data:
                    continue

                df_pivot = pivot_data[metric]

                ax = axes[i]
                # Only show colorbar on the rightmost plot
                show_cbar = i == len(metrics) - 1
                sns.heatmap(
                    df_pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlBu_r",
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax,
                    cbar=show_cbar,
                    cbar_kws=(
                        {"label": "Mean Bootstrap Gap (A-B)"}
                        if show_cbar
                        else None
                    ),
                )

                metric_name = (
                    "Hard F1" if metric.startswith("hard") else "Semantic F1"
                )
                ax.set_title(
                    f"{metric_name} ({avg_type.capitalize()})",
                    fontweight="bold",
                    pad=15,
                )
                ax.set_xlabel("Perturbation probability (p)")
                if i == 0:
                    ax.set_ylabel("Number of labels (k)")
                else:
                    ax.set_ylabel("")

            plt.suptitle(
                f"Study A.1: Bootstrap Mean Gap - {avg_type.capitalize()} Averaging (Ideal S)",
                fontweight="bold",
                y=0.98,
                fontsize=14,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            fig_path_png = os.path.join(
                outdir, f"fig_A3_heatmap_main_{avg_type}_{ts}.png"
            )
            fig_path_pdf = os.path.join(
                outdir, f"fig_A3_heatmap_main_{avg_type}_{ts}.pdf"
            )
            fig.savefig(fig_path_png, dpi=200)
            fig.savefig(fig_path_pdf)
            plt.close(fig)
            logger.info("Wrote: %s", fig_path_png)
            print(f"Wrote: {fig_path_png}")

    # 2. Secondary heatmap: S×metric for representative k,p values with proper labels
    # Use median k,p values for cleaner view
    ks = sorted(df["k"].unique())
    ps = sorted(df["p"].unique())
    k_mid = ks[len(ks) // 2] if ks else 2
    p_mid = ps[len(ps) // 2] if ps else 0.5

    df_repr = df[(df["k"] == k_mid) & (df["p"] == p_mid)].copy()
    if not df_repr.empty:
        # Filter out alpha=1.0 and alpha=0.0 cases
        df_repr = df_repr[
            ~df_repr["S"].isin(["alpha=1.00", "alpha=0.00"])
        ].copy()

        # Aggregate across radius pairs
        df_sm = (
            df_repr.groupby(["S", "metric"])["mean_gap"].mean().reset_index()
        )

        # Clean up similarity matrix names
        def clean_s_name(s_name):
            if s_name == "ideal":
                return "Ideal"
            elif s_name == "permuted":
                return "Permuted"
            elif s_name == "deceptive":
                return "Deceptive"
            elif s_name.startswith("alpha="):
                alpha_val = float(s_name.split("=")[1])
                return f"α={alpha_val:.2f}"
            else:
                return s_name

        # Clean up metric names
        def clean_metric_name(metric):
            if metric == "hard_micro":
                return "Hard F1 (micro)"
            elif metric == "hard_macro":
                return "Hard F1 (macro)"
            elif metric == "hard_samples":
                return "Hard F1 (samples)"
            elif metric == "sem_micro":
                return "Semantic F1 (micro)"
            elif metric == "sem_macro":
                return "Semantic F1 (macro)"
            elif metric == "sem_samples":
                return "Semantic F1 (samples)"
            else:
                return metric

        df_sm["S_clean"] = df_sm["S"].apply(clean_s_name)
        df_sm["metric_clean"] = df_sm["metric"].apply(clean_metric_name)

        df_pivot_sm = df_sm.pivot(
            index="S_clean", columns="metric_clean", values="mean_gap"
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            df_pivot_sm,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            center=0,
            ax=ax,
            cbar_kws={"label": "Mean Bootstrap Gap (A-B)"},
        )
        ax.set_title(
            f"Study A.1: Bootstrap Gap by Similarity Matrix & Metric (k={k_mid}, p={p_mid})",
            fontweight="bold",
        )
        ax.set_xlabel("Evaluation Metric")
        ax.set_ylabel("Similarity Matrix")
        plt.xticks(rotation=45, ha="right")

        fig_path_png = os.path.join(outdir, f"fig_A3_heatmap_S_metric_{ts}.png")
        fig_path_pdf = os.path.join(outdir, f"fig_A3_heatmap_S_metric_{ts}.pdf")
        fig.tight_layout()
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        print(f"Wrote: {fig_path_png}")


def plot_bootstrap_summary_charts(
    df_boot: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create summary visualizations focusing on key findings from bootstrap analysis.

    Creates separate figures for each metric type (micro/macro/samples) with:
    - Left panel: Box plots of effect sizes by similarity matrix (including hard F1)
    - Right panel: Bar chart comparing Hard vs Semantic F1 (hard F1 shown once)
    """
    if df_boot.empty:
        return

    # Filter out alpha=1.0 and alpha=0.0 cases
    df_filtered = df_boot[
        ~df_boot["S"].isin(["alpha=1.00", "alpha=0.00"])
    ].copy()

    # Group by metric type
    metric_groups = {
        "micro": ["hard_micro", "sem_micro"],
        "macro": ["hard_macro", "sem_macro"],
        "samples": ["hard_samples", "sem_samples"],
    }

    for avg_type, metrics in metric_groups.items():
        df_metrics = df_filtered[df_filtered["metric"].isin(metrics)].copy()
        if df_metrics.empty:
            continue

        # Create single panel figure (removing right panel)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Box plot of effect sizes by S (including Hard F1)
        s_order = ["ideal", "deceptive", "permuted"]  # Base ordering
        available_s = [s for s in s_order if s in df_metrics["S"].unique()]
        # Add alpha values in sorted order, excluding endpoints
        alpha_s = [
            s
            for s in df_metrics["S"].unique()
            if s.startswith("a=") and s not in ["a=1.00", "a=0.00"]
        ]
        alpha_s_sorted = sorted(
            alpha_s, key=lambda x: float(x.split("=")[1]), reverse=True
        )
        s_order_final = available_s + alpha_s_sorted

        import seaborn as sns
        import pandas as pd

        # Separate Hard F1 and Semantic F1 data
        df_semantic = df_metrics[
            df_metrics["metric"].str.startswith("sem")
        ].copy()
        df_hard = df_metrics[df_metrics["metric"].str.startswith("hard")].copy()

        # Create combined data with Hard F1 as a single category
        df_combined = df_semantic.copy()

        # Add Hard F1 as a single category (aggregate across all similarity matrices)
        if not df_hard.empty:
            df_hard_single = df_hard.copy()
            df_hard_single["S"] = "Hard F1"  # Single category for Hard F1
            df_combined = pd.concat(
                [df_combined, df_hard_single], ignore_index=True
            )

        # Update the order to include Hard F1
        s_order_with_hard = s_order_final + ["Hard F1"]

        # Create comprehensive color mapping for all possible similarity matrix types
        color_mapping = {
            "ideal": SEM_COLOR,
            "deceptive": DECEPTIVE_COLOR,
            "permuted": PERMUTED_COLOR,
            "Hard F1": HARD_COLOR,
        }

        # Add colors for alpha variants and other similarity matrices
        alpha_colors = [
            "#E377C2",
            "#BCBD22",
            "#17BECF",
            "#AEC7E8",
            "#FFBB78",
            "#C49C94",
            "#F7B6D3",
            "#C7C7C7",
            "#DBDB8D",
        ]
        other_similarity_matrices = [
            s for s in s_order_final if s not in color_mapping
        ]

        for i, s in enumerate(other_similarity_matrices):
            if i < len(alpha_colors):
                color_mapping[s] = alpha_colors[i]
            else:
                # Fallback to gray for any additional similarity matrices
                color_mapping[s] = "#808080"

        # Build palette in correct order
        palette = [color_mapping.get(s, "#808080") for s in s_order_with_hard]

        # Ensure palette matches the unique values in the data
        unique_s_in_data = df_combined["S"].unique()
        palette_dict = {
            s: color_mapping.get(s, "#808080") for s in unique_s_in_data
        }

        sns.boxplot(
            data=df_combined,
            y="S",
            x="mean_gap",
            hue="S",
            order=s_order_with_hard,
            ax=ax,
            palette=palette_dict,  # Use dictionary instead of list
            legend=False,
        )

        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Bootstrap Mean Gap (A-B)", fontsize=14)
        ax.set_ylabel("Similarity Matrix", fontsize=18)
        ax.set_title(
            f"Bootstrap Effect Size Distribution - {avg_type.capitalize()} F1",
            fontweight="bold",
            fontsize=20,
        )

        # Fix tick labels properly by setting ticks first
        current_ticks = ax.get_yticks()
        current_labels = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticks(current_ticks)
        ax.set_yticklabels(current_labels, rotation=45, fontsize=16)

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = []
        if "ideal" in s_order_final:
            legend_elements.append(
                Patch(facecolor=SEM_COLOR, label="Sem (ideal)")
            )
        if "deceptive" in s_order_final:
            legend_elements.append(
                Patch(facecolor=DECEPTIVE_COLOR, label="Sem (decep)")
            )
        if "permuted" in s_order_final:
            legend_elements.append(
                Patch(facecolor=PERMUTED_COLOR, label="Sem (perm)")
            )
        legend_elements.append(Patch(facecolor=HARD_COLOR, label="Hard"))
        ax.legend(
            handles=legend_elements,
            title="Metric Type",
            loc="best",
            fontsize=14,
        )

        plt.tight_layout()
        fig_path_png = os.path.join(
            outdir, f"fig_A4_summary_{avg_type}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A4_summary_{avg_type}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        print(f"Wrote: {fig_path_png}")


def plot_bootstrap_confidence_intervals(
    df_boot: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Create confidence interval plots for key parameter combinations."""
    if df_boot.empty:
        return

    # Focus on ideal S for clearest signal
    df_ideal = df_boot[df_boot["S"] == "ideal"].copy()
    if df_ideal.empty:
        return

    # Create separate plots for hard vs semantic metrics
    for metric_prefix in ["hard", "sem"]:
        df_metrics = df_ideal[
            df_ideal["metric"].str.startswith(metric_prefix)
        ].copy()
        if df_metrics.empty:
            continue

        metric_type = "Hard F1" if metric_prefix == "hard" else "Semantic F1"

        # Aggregate across radius combinations for cleaner view
        df_agg = (
            df_metrics.groupby(["k", "p", "metric"])
            .agg({"mean_gap": "mean", "ci95_lo": "mean", "ci95_hi": "mean"})
            .reset_index()
        )

        # Create subplot grid: k rows, metric columns
        metrics = sorted(df_agg["metric"].unique())
        ks = sorted(df_agg["k"].unique())

        fig, axes = plt.subplots(
            len(ks),
            len(metrics),
            figsize=(4 * len(metrics), 3 * len(ks)),
            squeeze=False,
        )

        for i, k_val in enumerate(ks):
            for j, metric in enumerate(metrics):
                ax = axes[i][j]
                subset = df_agg[
                    (df_agg["k"] == k_val) & (df_agg["metric"] == metric)
                ]

                if not subset.empty:
                    ps = sorted(subset["p"].unique())
                    means = [
                        subset[subset["p"] == p]["mean_gap"].iloc[0] for p in ps
                    ]
                    lows = [
                        subset[subset["p"] == p]["ci95_lo"].iloc[0] for p in ps
                    ]
                    highs = [
                        subset[subset["p"] == p]["ci95_hi"].iloc[0] for p in ps
                    ]

                    ax.errorbar(
                        ps,
                        means,
                        yerr=[
                            np.array(means) - np.array(lows),
                            np.array(highs) - np.array(means),
                        ],
                        fmt='o-',
                        capsize=5,
                        capthick=2,
                    )
                    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
                    ax.grid(True, alpha=0.3)

                if i == 0:
                    ax.set_title(metric.replace("_", " ").title())
                if j == 0:
                    ax.set_ylabel(f"k={k_val}\nBootstrap Gap")
                if i == len(ks) - 1:
                    ax.set_xlabel("Perturbation probability (p)")

        plt.suptitle(
            f"Study A.1: {metric_type} Bootstrap Confidence Intervals (Ideal S)",
            fontweight="bold",
        )
        plt.tight_layout()

        fig_path_png = os.path.join(
            outdir, f"fig_A5_confidence_intervals_{metric_prefix}_{ts}.png"
        )
        fig_path_pdf = os.path.join(
            outdir, f"fig_A5_confidence_intervals_{metric_prefix}_{ts}.pdf"
        )
        fig.savefig(fig_path_png, dpi=200)
        fig.savefig(fig_path_pdf)
        plt.close(fig)
        logger.info("Wrote: %s", fig_path_png)
        print(f"Wrote: {fig_path_png}")


def write_kendall_tau_table(
    df_tau_all: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Write LaTeX tables of negative Kendall tau by k×variant with p columns.

    One table per averaging metric (micro/macro/samples).
    """
    if df_tau_all.empty:
        return

    def _keep_metric(name: str, avg: str) -> bool:
        core = {
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:deceptive",
            f"sem_{avg}:permuted",
        }
        return name in core

    ks = sorted(df_tau_all["k"].unique()) if "k" in df_tau_all.columns else []
    ps = sorted(df_tau_all["p"].unique()) if "p" in df_tau_all.columns else []
    if not ks or not ps:
        return

    def _avg_full_name(a: str) -> str:
        return {
            "micro": "Micro F1 Score",
            "macro": "Macro F1 Score",
            "samples": "Examples F1 Score",
        }.get(a, a)

    for avg in ["micro", "macro", "samples"]:
        df = df_tau_all.copy()
        df = df[df["metric"].apply(lambda m: _keep_metric(m, avg))]
        if df.empty:
            continue

        order = [
            f"hard_{avg}",
            f"sem_{avg}:ideal",
            f"sem_{avg}:deceptive",
            f"sem_{avg}:permuted",
        ]
        metrics_order = [m for m in order if m in set(df["metric"].unique())]

        df_vals = (
            df.assign(neg_tau=lambda x: -x["tau"].astype(float))
            .groupby(["k", "p", "metric"], as_index=False)["neg_tau"]
            .mean()
        )

        # Build LaTeX table
        def esc(s: str) -> str:
            return str(s).replace("_", "\\_")

        cols_p = [f"{p:.2f}" for p in ps]
        lines = []
        lines.append(
            "% Auto-generated by study_a1_plots.write_kendall_tau_table"
        )
        lines.append("% Requires \\usepackage{multirow}")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\setlength{\\tabcolsep}{6pt}")
        lines.append("\\begin{tabular}{ll" + "r" * len(ps) + "}")
        lines.append(
            "k & variant & "
            + " ".join([f"\\multicolumn{{1}}{{c}}{{{esc(p)}}}" for p in cols_p])
            + " \\\\"
        )
        lines.append("\\hline")
        for k_val in ks:
            nvar = len(metrics_order)
            for idx_m, m in enumerate(metrics_order):
                label = {
                    f"hard_{avg}": "Hard",
                    f"sem_{avg}:ideal": "Semantic (ideal)",
                    f"sem_{avg}:deceptive": "Semantic (deceptive)",
                    f"sem_{avg}:permuted": "Semantic (permute)",
                }[m]
                row_vals = []
                for p_val in ps:
                    row = df_vals[
                        (df_vals["k"] == k_val)
                        & (df_vals["p"] == p_val)
                        & (df_vals["metric"] == m)
                    ]
                    v = float(row.iloc[0]["neg_tau"]) if not row.empty else 0.0
                    row_vals.append(f"{v:.2f}")
                if idx_m == 0:
                    lines.append(
                        f"\\multirow{{{nvar}}}{{*}}{{{k_val}}} & {esc(label)} & "
                        + " & ".join(row_vals)
                        + " \\\\"
                    )
                else:
                    lines.append(
                        f" & {esc(label)} & " + " & ".join(row_vals) + " \\\\"
                    )
        lines.append("\\end{tabular}")
        lines.append(
            f"\\caption{{Study A.1: Negative Kendall tau (Higher is Better) — {_avg_full_name(avg)} across p (columns), for k and variants.}}"
        )
        lines.append(f"\\label{{tab:a2_kendall_{avg}}}")
        lines.append("\\end{table}")

        tab_path = os.path.join(outdir, f"tab_A2_kendall_tau_{avg}_{ts}.tex")
        with open(tab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Wrote: %s", tab_path)
        print(f"Wrote: {tab_path}")


def write_bootstrap_table_summary(
    df_boot: pd.DataFrame,
    outdir: str,
    ts: int,
    logger: Any,
) -> None:
    """Write a concise LaTeX table summarizing key bootstrap results.

    Instead of the massive table with all combinations, create focused summaries:
    1. Main table: Ideal S only, aggregated across radius pairs
    2. Appendix table: Key comparisons across similarity matrices
    """
    if df_boot.empty:
        return

    def _latex_escape(s: str) -> str:
        return str(s).replace("_", "\\_")

    # === MAIN SUMMARY TABLE ===
    # Focus on ideal S, aggregate across radius pairs
    df_ideal = df_boot[df_boot["S"] == "ideal"].copy()
    if not df_ideal.empty:
        # Aggregate across r_near, r_far combinations
        df_main = (
            df_ideal.groupby(["k", "p", "metric"])
            .agg({"mean_gap": "mean", "ci95_lo": "mean", "ci95_hi": "mean"})
            .reset_index()
        )

        # Organize by metric type for better readability
        metric_order = [
            "hard_micro",
            "hard_macro",
            "hard_samples",
            "sem_micro",
            "sem_macro",
            "sem_samples",
        ]
        df_main = df_main[df_main["metric"].isin(metric_order)]
        df_main["metric"] = pd.Categorical(
            df_main["metric"], categories=metric_order, ordered=True
        )
        df_main = df_main.sort_values(["metric", "k", "p"])

        rows = []
        current_metric = None
        for _, row in df_main.iterrows():
            metric = row["metric"]
            if metric != current_metric:
                if current_metric is not None:
                    rows.append("\\midrule")
                current_metric = metric

            k_ = int(row["k"])
            p_ = float(row["p"])
            mean = float(row["mean_gap"])
            lo = float(row["ci95_lo"])
            hi = float(row["ci95_hi"])
            metric_display = (
                _latex_escape(str(metric))
                .replace("hard\\_", "Hard ")
                .replace("sem\\_", "Sem ")
            )

            rows.append(
                f"{metric_display} & {k_} & {p_:.2f} & {mean:.3f} & [{lo:.3f}, {hi:.3f}] \\\\"
            )

        tex = []
        tex.append(
            "% Auto-generated summary table (replaces overly long Table A1)"
        )
        tex.append("\\begin{table}[t]")
        tex.append("\\centering")
        tex.append("\\small")
        tex.append("\\begin{tabular}{lrrrr}")
        tex.append("\\toprule")
        tex.append("Metric & k & p & Mean Gap & 95\\% CI \\\\")
        tex.append("\\midrule")
        tex.extend(rows)
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")
        tex.append(
            "\\caption{Study A.1: Bootstrap mean gap (Near--Far) with 95\\% CI for ideal similarity matrix. Values averaged across radius combinations for clarity.}"
        )
        tex.append("\\label{tab:a1_bootstrap_summary}")
        tex.append("\\end{table}")

        tab_path = os.path.join(outdir, f"tab_A1_bootstrap_summary_{ts}.tex")
        with open(tab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
        logger.info("Wrote: %s", tab_path)
        print(f"Wrote: {tab_path}")

    # === SIMILARITY MATRIX COMPARISON TABLE ===
    # Show how different S choices affect results
    ks_repr = [df_boot["k"].median()] if not df_boot.empty else [2]
    ps_repr = [df_boot["p"].median()] if not df_boot.empty else [0.5]

    df_s_comp = df_boot[
        (df_boot["k"].isin(ks_repr)) & (df_boot["p"].isin(ps_repr))
    ].copy()

    if not df_s_comp.empty:
        # Aggregate across radius pairs, focus on micro metrics
        df_s_micro = df_s_comp[df_s_comp["metric"].str.contains("micro")].copy()
        df_s_agg = (
            df_s_micro.groupby(["S", "metric"])
            .agg({"mean_gap": "mean", "ci95_lo": "mean", "ci95_hi": "mean"})
            .reset_index()
        )

        rows_s = []
        for _, row in df_s_agg.sort_values(["metric", "S"]).iterrows():
            s_name = _latex_escape(str(row["S"]))
            metric = _latex_escape(str(row["metric"]))
            mean = float(row["mean_gap"])
            lo = float(row["ci95_lo"])
            hi = float(row["ci95_hi"])

            rows_s.append(
                f"{s_name} & {metric} & {mean:.3f} & [{lo:.3f}, {hi:.3f}] \\\\"
            )

        tex_s = []
        tex_s.append("% Similarity matrix comparison")
        tex_s.append("\\begin{table}[t]")
        tex_s.append("\\centering")
        tex_s.append("\\small")
        tex_s.append("\\begin{tabular}{llrr}")
        tex_s.append("\\toprule")
        tex_s.append("Similarity Matrix & Metric & Mean Gap & 95\\% CI \\\\")
        tex_s.append("\\midrule")
        tex_s.extend(rows_s)
        tex_s.append("\\bottomrule")
        tex_s.append("\\end{tabular}")
        tex_s.append(
            f"\\caption{{Study A.1: Effect of similarity matrix choice on bootstrap gaps (k={int(ks_repr[0])}, p={ps_repr[0]:.2f}). Micro-averaging results only.}}"
        )
        tex_s.append("\\label{tab:a1_similarity_comparison}")
        tex_s.append("\\end{table}")

        tab_s_path = os.path.join(outdir, f"tab_A1_similarity_comp_{ts}.tex")
        with open(tab_s_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_s) + "\n")
        logger.info("Wrote: %s", tab_s_path)
        print(f"Wrote: {tab_s_path}")
