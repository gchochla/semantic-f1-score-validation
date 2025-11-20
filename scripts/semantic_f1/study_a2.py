#!/usr/bin/env python
"""
Study A.2 — Macro-Mode vs Fine-Grained Predictors (bimodal prototype test)

This script generates synthetic data with bimodal structure (positive/negative modes)
and tests how Semantic F1 vs Hard F1 handle prototype-based predictors that capture
macro-modes but miss fine-grained labels.

Outputs
- CSVs: per‑configuration summaries, bootstrap CIs.
- Figures: metric vs probability curves (Fig A6), imbalance heatmaps (Fig A7), 
  stratified curves (Fig A8).
- LaTeX table: bootstrap mean ± 95% CI for all predictors (Tab A2).

Example
  python scripts/semantic_f1/study_a2.py \
      --num-labels 24 --num-examples 20000 \
      --k 1 2 3 --q 0.0 0.25 0.5 0.75 1.0 \
      --prototype-sizes 2 3 4 --imbalance-ratios 0.3 0.5 0.7 \
      --outdir logs/semantic_f1/study_a2

Notes
- Builds on Study A.1 but focuses on bimodal prototype predictors.
- Tests whether Semantic F1 properly rewards mode-correct but fine-label-wrong predictions.
- Uses circular label arrangement with soft positive/negative mode definitions.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from semantic_f1_score import semantic_f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# External plotting helpers with robust import when run as a script
try:
    import scripts.semantic_f1.study_a2_plots as study_a2_plots
    import scripts.semantic_f1.study_a1 as study_a1
except Exception:
    # Fallback: import sibling when executed directly via path
    sys.path.append(os.path.dirname(__file__))
    import study_a2_plots  # type: ignore
    import study_a1  # type: ignore


RNG = np.random.default_rng(42)


def make_bimodal_weights(
    n: int, kappa: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create positive and negative mode membership weights for circular labels.

    Positive mode peaks at angle 0, negative mode peaks at angle π.
    Uses von Mises-like concentration with parameter kappa.

    Args:
        n: Number of labels on the circle.
        kappa: Concentration parameter (higher = sharper modes).

    Returns:
        Tuple of (w_pos, w_neg) arrays, each length n, normalized to sum to 1.
    """
    thetas = np.linspace(0.0, 2 * math.pi, num=n, endpoint=False)

    # Positive mode centered at theta=0
    w_pos = np.exp(kappa * np.cos(thetas))
    w_pos = w_pos / w_pos.sum()

    # Negative mode centered at theta=π
    w_neg = np.exp(kappa * np.cos(thetas - math.pi))
    w_neg = w_neg / w_neg.sum()

    return w_pos, w_neg


def compute_mode_weights(
    gold_indices: list[int], w_pos: np.ndarray, w_neg: np.ndarray
) -> tuple[float, float]:
    """Compute positive and negative mode weights for a gold label set.

    Args:
        gold_indices: List of gold label indices.
        w_pos: Positive mode weights per label.
        w_neg: Negative mode weights per label.

    Returns:
        Tuple of (W_pos, W_neg) total weights.
    """
    W_pos = sum(w_pos[i] for i in gold_indices)
    W_neg = sum(w_neg[i] for i in gold_indices)
    return W_pos, W_neg


def select_prototypes(
    n: int, w_pos: np.ndarray, w_neg: np.ndarray, m: int
) -> tuple[list[int], list[int]]:
    """Select prototype labels for positive and negative modes.

    Args:
        n: Total number of labels.
        w_pos: Positive mode weights.
        w_neg: Negative mode weights.
        m: Number of prototypes per mode.

    Returns:
        Tuple of (pos_prototypes, neg_prototypes) as lists of indices.
    """
    pos_prototypes = list(np.argsort(w_pos)[-m:])
    neg_prototypes = list(np.argsort(w_neg)[-m:])
    return pos_prototypes, neg_prototypes


@dataclass
class SynthConfigA2:
    """Configuration for Study A.2 bimodal prototype experiment.

    Fields
    - ``num_labels``: Number of labels on the ring.
    - ``num_examples``: Number of synthetic examples per configuration.
    - ``k_values``: Numbers of labels per example to sample for gold.
    - ``q_values``: Probabilities for various predictor behaviors (renamed from p_values to differentiate from Study A.1).
    - ``prototype_sizes``: Numbers of prototype labels per mode (m).
    - ``imbalance_ratios``: Ratios of positive to negative examples.
    - ``kappa``: Mode concentration parameter.
    - ``beta``: Within-mode tail weight for Prototype-Within-Mode.
    - ``near_radii``: List of radii for near-miss predictor (distance for label replacement).
    - ``far_radii``: List of radii for far-miss predictor (distance for label replacement).
    - ``outdir``: Output directory for CSVs/figures/tables.
    - ``seed``: RNG seed for reproducibility.
    - ``bootstrap``: Bootstrap iterations for CI estimation.
    """

    num_labels: int
    num_examples: int
    k_values: list[int]
    q_values: list[float]
    prototype_sizes: list[int]
    imbalance_ratios: list[float]
    kappa: float = 2.0
    beta: float = 2.0
    near_radii: list[int] = field(default_factory=lambda: [1])
    far_radii: list[int] = field(default_factory=lambda: [3])
    # A6-only: user-defined near-miss probability p
    near_p: float = 0.5
    outdir: str = "logs/semantic_f1/study_a2"
    seed: int = 123
    bootstrap: int = 200


def sample_gold_labels_bimodal(
    n: int,
    k: int,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    imbalance_ratio: float = 0.5,
) -> list[int]:
    """Sample gold labels with bimodal structure and class imbalance.

    First decides on mode (positive vs negative) based on imbalance_ratio,
    then samples k labels from that mode's distribution.

    Args:
        n: Total number of labels.
        k: Number of labels to sample.
        w_pos: Positive mode weights.
        w_neg: Negative mode weights.
        imbalance_ratio: Probability of selecting positive mode.

    Returns:
        List of distinct label indices.
    """
    # Choose mode
    if RNG.random() < imbalance_ratio:
        weights = w_pos
    else:
        weights = w_neg

    # Sample from chosen mode
    chosen = RNG.choice(np.arange(n), size=min(k, n), replace=False, p=weights)
    return list(map(int, chosen))


def prototype_bimodal_predictor(
    gold_indices: list[int],
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    pos_prototypes: list[int],
    neg_prototypes: list[int],
    q: float,
) -> list[int]:
    """Prototype-Bimodal predictor from Study A.2.

    Computes mode weights for gold, predicts prototype of dominant mode
    with probability q, opposite mode with probability (1-q).

    Args:
        gold_indices: Gold label indices.
        w_pos: Positive mode weights.
        w_neg: Negative mode weights.
        pos_prototypes: Positive prototype indices.
        neg_prototypes: Negative prototype indices.
        q: Probability of predicting correct mode.

    Returns:
        List of predicted label indices.
    """
    W_pos, W_neg = compute_mode_weights(gold_indices, w_pos, w_neg)

    # Determine dominant mode
    if W_pos >= W_neg:
        correct_prototypes = pos_prototypes
        wrong_prototypes = neg_prototypes
    else:
        correct_prototypes = neg_prototypes
        wrong_prototypes = pos_prototypes

    # Choose mode to predict
    if RNG.random() < q:
        return correct_prototypes.copy()
    else:
        return wrong_prototypes.copy()


def prototype_within_mode_predictor(
    gold_indices: list[int],
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    pos_prototypes: list[int],
    neg_prototypes: list[int],
    k: int,
    q: float,
    beta: float,
) -> list[int]:
    """Prototype-Within-Mode predictor from Study A.2.

    First chooses correct mode with probability q, then samples k labels
    from a distribution peaked on prototypes with tail controlled by beta.

    Args:
        gold_indices: Gold label indices.
        w_pos: Positive mode weights.
        w_neg: Negative mode weights.
        pos_prototypes: Positive prototype indices.
        neg_prototypes: Negative prototype indices.
        k: Number of labels to predict.
        q: Probability of choosing correct mode.
        beta: Tail weight parameter (higher = more concentrated on prototypes).

    Returns:
        List of predicted label indices.
    """
    W_pos, W_neg = compute_mode_weights(gold_indices, w_pos, w_neg)

    # Determine mode to use
    if RNG.random() < q:
        # Use correct mode
        if W_pos >= W_neg:
            mode_weights = w_pos
            prototypes = pos_prototypes
        else:
            mode_weights = w_neg
            prototypes = neg_prototypes
    else:
        # Use wrong mode
        if W_pos >= W_neg:
            mode_weights = w_neg
            prototypes = neg_prototypes
        else:
            mode_weights = w_pos
            prototypes = pos_prototypes

    # Create distribution peaked on prototypes
    n = len(mode_weights)
    peaked_weights = mode_weights.copy()

    # Amplify prototype weights
    for proto_idx in prototypes:
        peaked_weights[proto_idx] *= 1.0 + beta

    # Normalize
    peaked_weights = peaked_weights / peaked_weights.sum()

    # Sample k labels
    chosen = RNG.choice(
        np.arange(n), size=min(k, n), replace=False, p=peaked_weights
    )
    return list(map(int, chosen))


def near_miss_predictor(
    gold_indices: list[int], n: int, p: float, radius: int = 1
) -> list[int]:
    """Near-miss predictor from Study A.1 (for comparison in A.2)."""
    return study_a1.perturb_labels(gold_indices, n, p, radius)


def far_miss_predictor(
    gold_indices: list[int], n: int, p: float, radius: int = 3
) -> list[int]:
    """Far-miss predictor from Study A.1 (for comparison in A.2)."""
    return near_miss_predictor(gold_indices, n, p, radius)


def evaluate_predictors_on_config(
    cfg: SynthConfigA2,
    k: int,
    q: float,
    m: int,
    imbalance: float,
    labels: list[str],
    S_list: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Evaluate all predictors on a single configuration and return metrics.

    Args:
        cfg: Experiment configuration.
        k: Number of labels per example.
        q: Probability parameter for predictors.
        m: Prototype size.
        imbalance: Imbalance ratio.
        labels: Label names.
        S_list: Dictionary of similarity matrices.

    Returns:
        DataFrame with columns: predictor, metric, value, near_radius, far_radius (when applicable).
    """
    n = cfg.num_labels
    w_pos, w_neg = make_bimodal_weights(n, cfg.kappa)
    pos_prototypes, neg_prototypes = select_prototypes(n, w_pos, w_neg, m)

    # Generate gold labels
    gold_indices = [
        sample_gold_labels_bimodal(n, k, w_pos, w_neg, imbalance)
        for _ in range(cfg.num_examples)
    ]
    gold_indices = [sorted(set(g)) for g in gold_indices]
    gold_names = study_a1.to_label_names(gold_indices, labels)

    results = []

    # Generate predictions from all predictors
    predictors = {}

    # Near-miss predictors (from A.1) - iterate over all configured radii
    for near_radius in cfg.near_radii:
        near_preds = [
            near_miss_predictor(g, n, q, radius=near_radius)
            for g in gold_indices
        ]
        predictor_name = (
            f"near_r{near_radius}" if len(cfg.near_radii) > 1 else "near"
        )
        predictors[predictor_name] = study_a1.to_label_names(
            [sorted(set(x)) for x in near_preds], labels
        )

    # Far-miss predictors (from A.1) - iterate over all configured radii
    for far_radius in cfg.far_radii:
        far_preds = [
            far_miss_predictor(g, n, q, radius=far_radius) for g in gold_indices
        ]
        predictor_name = (
            f"far_r{far_radius}" if len(cfg.far_radii) > 1 else "far"
        )
        predictors[predictor_name] = study_a1.to_label_names(
            [sorted(set(x)) for x in far_preds], labels
        )

    # Prototype-Bimodal
    bimodal_preds = [
        prototype_bimodal_predictor(
            g, w_pos, w_neg, pos_prototypes, neg_prototypes, q
        )
        for g in gold_indices
    ]
    predictors["bimodal"] = study_a1.to_label_names(
        [sorted(set(x)) for x in bimodal_preds], labels
    )

    # Prototype-Within-Mode
    within_preds = [
        prototype_within_mode_predictor(
            g, w_pos, w_neg, pos_prototypes, neg_prototypes, k, q, cfg.beta
        )
        for g in gold_indices
    ]
    predictors["within_mode"] = study_a1.to_label_names(
        [sorted(set(x)) for x in within_preds], labels
    )

    # Evaluate all predictors
    rows = []
    for pred_name, pred_labels in predictors.items():
        # Hard metrics
        hard_vals = study_a1.eval_hard_metrics(gold_names, pred_labels, labels)
        for metric_name, val in hard_vals.items():
            rows.append(
                {"predictor": pred_name, "metric": metric_name, "value": val}
            )

        # Semantic metrics for each similarity matrix
        for s_name, S in S_list.items():
            sem_vals = study_a1.eval_semantic_metrics(
                gold_names, pred_labels, S
            )
            for metric_name, val in sem_vals.items():
                full_metric_name = f"{metric_name}:{s_name}"
                rows.append(
                    {
                        "predictor": pred_name,
                        "metric": full_metric_name,
                        "value": val,
                    }
                )

    return pd.DataFrame(rows)


def run_one_config(
    cfg: SynthConfigA2,
    outdir: str,
):
    """Run Study A.2 for a single configuration and write artifacts.

    Side effects
    - Writes CSVs: summary, bootstrap results.
    - Writes figures: Fig A6/A7/A8 (PNG/PDF).
    - Writes LaTeX table: Tab A2 with bootstrap CIs.
    - Logs to ``<outdir>/log.txt``.

    Args:
        cfg: Experiment configuration.
        outdir: Output directory (created if missing).
    """
    os.makedirs(outdir, exist_ok=True)

    # set up file logger
    logger = logging.getLogger("study_a2")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if re-run in interactive sessions
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(os.path.join(outdir, "log.txt"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Starting Study A.2 with config: %s", cfg)
    labels = study_a1.make_labels(cfg.num_labels)
    S_ideal = study_a1.make_similarity_ring(cfg.num_labels)

    S_list = {
        "ideal": S_ideal,
    }

    logger.info(
        "Built similarity matrices: ideal=%s",
        S_ideal.shape,
    )

    # Placeholders for summaries
    rows_summary = []
    rows_boot = []

    # Iterate through all parameter combinations
    for k in cfg.k_values:
        for q in cfg.q_values:
            for m in cfg.prototype_sizes:
                for imbalance in cfg.imbalance_ratios:
                    logger.info(
                        "Evaluating k=%d, q=%.2f, m=%d, imbalance=%.2f",
                        k,
                        q,
                        m,
                        imbalance,
                    )

                    # Evaluate all predictors on this configuration
                    df_metrics = evaluate_predictors_on_config(
                        cfg, k, q, m, imbalance, labels, S_list
                    )

                    # Add configuration info to summary
                    for _, row in df_metrics.iterrows():
                        rows_summary.append(
                            {
                                "k": k,
                                "q": q,
                                "m": m,
                                "imbalance": imbalance,
                                "predictor": row["predictor"],
                                "metric": row["metric"],
                                "value": row["value"],
                            }
                        )

                    # Compute bootstrap CIs for key comparisons
                    # Compare bimodal vs near-miss, within-mode vs bimodal, etc.
                    n = cfg.num_labels
                    w_pos, w_neg = make_bimodal_weights(n, cfg.kappa)
                    pos_prototypes, neg_prototypes = select_prototypes(
                        n, w_pos, w_neg, m
                    )

                    # Generate consistent gold for bootstrap
                    gold_indices = [
                        sample_gold_labels_bimodal(
                            n, k, w_pos, w_neg, imbalance
                        )
                        for _ in range(cfg.num_examples)
                    ]
                    gold_indices = [sorted(set(g)) for g in gold_indices]
                    gold_names = study_a1.to_label_names(gold_indices, labels)

                    # Generate all predictor outputs for bootstrap
                    # Use first radius for each predictor type for bootstrap comparisons
                    near_radius = cfg.near_radii[0]
                    far_radius = cfg.far_radii[0]

                    near_preds = [
                        near_miss_predictor(g, n, q, radius=near_radius)
                        for g in gold_indices
                    ]
                    far_preds = [
                        far_miss_predictor(g, n, q, radius=far_radius)
                        for g in gold_indices
                    ]
                    bimodal_preds = [
                        prototype_bimodal_predictor(
                            g, w_pos, w_neg, pos_prototypes, neg_prototypes, q
                        )
                        for g in gold_indices
                    ]
                    within_preds = [
                        prototype_within_mode_predictor(
                            g,
                            w_pos,
                            w_neg,
                            pos_prototypes,
                            neg_prototypes,
                            k,
                            q,
                            cfg.beta,
                        )
                        for g in gold_indices
                    ]

                    # import pdb

                    # pdb.set_trace()

                    pred_names = {
                        "near": study_a1.to_label_names(
                            [sorted(set(x)) for x in near_preds], labels
                        ),
                        "far": study_a1.to_label_names(
                            [sorted(set(x)) for x in far_preds], labels
                        ),
                        "bimodal": study_a1.to_label_names(
                            [sorted(set(x)) for x in bimodal_preds], labels
                        ),
                        "within_mode": study_a1.to_label_names(
                            [sorted(set(x)) for x in within_preds], labels
                        ),
                    }

                    # Bootstrap comparisons
                    comparisons = [
                        ("bimodal", "near"),
                        ("within_mode", "bimodal"),
                        ("bimodal", "far"),
                        ("within_mode", "far"),
                    ]

                    for pred_a, pred_b in comparisons:
                        for s_name, S in S_list.items():
                            for metric_name in [
                                "hard_micro",
                                "hard_macro",
                                "hard_samples",
                                "sem_micro",
                                "sem_macro",
                                "sem_samples",
                            ]:
                                try:
                                    mean_gap, ci_lo, ci_hi = (
                                        study_a1.bootstrap_ci(
                                            gold_names,
                                            pred_names[pred_a],
                                            pred_names[pred_b],
                                            labels,
                                            S,
                                            metric_name,
                                            cfg.bootstrap,
                                            cfg.seed
                                            + k
                                            + int(q * 100)
                                            + m
                                            + int(imbalance * 100),
                                        )
                                    )

                                    full_metric = (
                                        f"{metric_name}:{s_name}"
                                        if metric_name.startswith("sem")
                                        else metric_name
                                    )

                                    rows_boot.append(
                                        {
                                            "k": k,
                                            "q": q,
                                            "m": m,
                                            "imbalance": imbalance,
                                            "pred_a": pred_a,
                                            "pred_b": pred_b,
                                            "S": s_name,
                                            "metric": full_metric,
                                            "mean_gap": mean_gap,
                                            "ci95_lo": ci_lo,
                                            "ci95_hi": ci_hi,
                                        }
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Bootstrap failed for %s vs %s, metric %s:%s - %s",
                                        pred_a,
                                        pred_b,
                                        metric_name,
                                        s_name,
                                        e,
                                    )

    # Write outputs
    ts = int(time.time())
    df_summary = pd.DataFrame(rows_summary)
    df_boot = pd.DataFrame(rows_boot)

    path_summary = os.path.join(outdir, f"studyA2_summary_{ts}.csv")
    path_boot = os.path.join(outdir, f"studyA2_bootstrap_{ts}.csv")
    df_summary.to_csv(path_summary, index=False)
    df_boot.to_csv(path_boot, index=False)

    logger.info(
        "Output shapes — summary=%s, bootstrap=%s",
        df_summary.shape,
        df_boot.shape,
    )
    logger.info("Wrote: %s", path_summary)
    logger.info("Wrote: %s", path_boot)
    # keep prints for quick terminal visibility
    print(f"Wrote: {path_summary}")
    print(f"Wrote: {path_boot}")

    # --------------------------
    # Figures and Tables outputs
    # --------------------------
    try:
        helpers = {
            "sample_gold_labels_bimodal": sample_gold_labels_bimodal,
            "to_label_names": study_a1.to_label_names,
            "prototype_bimodal_predictor": prototype_bimodal_predictor,
            "prototype_within_mode_predictor": prototype_within_mode_predictor,
            "near_miss_predictor": near_miss_predictor,
            "far_miss_predictor": far_miss_predictor,
            "make_bimodal_weights": make_bimodal_weights,
            "select_prototypes": select_prototypes,
            "semantic_f1_score": semantic_f1_score,
            "f1_score": f1_score,
            "MultiLabelBinarizer": MultiLabelBinarizer,
        }

        study_a2_plots.plot_metric_vs_p_curves(
            cfg, labels, S_ideal, helpers, outdir, ts, logger
        )
        study_a2_plots.plot_imbalance_heatmaps(df_summary, outdir, ts, logger)
        study_a2_plots.plot_stratified_curves(
            cfg, labels, S_list, helpers, outdir, ts, logger
        )
        study_a2_plots.write_bootstrap_table(df_boot, outdir, ts, logger)
    except Exception as e:
        logger.error("Plotting failed: %s", e)
        print(f"Warning: Plotting failed: {e}")


def parse_args() -> SynthConfigA2:
    """Parse CLI arguments into a ``SynthConfigA2``.

    Returns:
        A populated ``SynthConfigA2`` reflecting command-line options.
    """
    ap = argparse.ArgumentParser(
        description="Study A.2 bimodal prototype evaluation for Semantic F1"
    )
    ap.add_argument(
        "--num-labels",
        type=int,
        default=24,
        help="Number of labels to arrange on the ring (unit circle). Controls the label space size.",
    )
    ap.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of synthetic examples to generate per configuration.",
    )
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of numbers of labels per example to sample for gold sets.",
    )
    ap.add_argument(
        "--q",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="List of probabilities for various predictor behaviors (e.g., correct mode selection). Renamed from --p to differentiate from Study A.1.",
    )
    ap.add_argument(
        "--prototype-sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="List of numbers of prototype labels per mode (macro-mode predictors).",
    )
    ap.add_argument(
        "--imbalance-ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="List of ratios of positive to negative examples (controls class imbalance in gold sets).",
    )
    ap.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Mode concentration parameter for bimodal label distributions (higher = sharper modes).",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Within-mode tail weight for Prototype-Within-Mode predictor (higher = more concentrated on prototypes).",
    )
    ap.add_argument(
        "--near-radii",
        type=int,
        nargs="+",
        default=[1],
        help="List of radii for near-miss predictor A from Study A.1 (distance for label replacement).",
    )
    ap.add_argument(
        "--far-radii",
        type=int,
        nargs="+",
        default=[3],
        help="List of radii for far-miss predictor B from Study A.1 (distance for label replacement).",
    )
    ap.add_argument(
        "--near-p",
        type=float,
        default=0.5,
        help="Near-miss predictor probability p (used in Fig A6; vertical line at x=1-p).",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap iterations for confidence interval estimation per metric.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="logs/semantic_f1/study_a2",
        help="Output directory for CSVs, figures, and tables.",
    )
    args = ap.parse_args()
    return SynthConfigA2(
        num_labels=args.num_labels,
        num_examples=args.num_examples,
        k_values=args.k,
        q_values=args.q,
        prototype_sizes=args.prototype_sizes,
        imbalance_ratios=args.imbalance_ratios,
        kappa=args.kappa,
        beta=args.beta,
        near_radii=args.near_radii,
        far_radii=args.far_radii,
        near_p=args.near_p,
        seed=args.seed,
        outdir=args.outdir,
        bootstrap=args.bootstrap,
    )


def main():
    """Entry point: parse args and run Study A.2."""
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    run_one_config(cfg, outdir=cfg.outdir)


if __name__ == "__main__":
    main()
