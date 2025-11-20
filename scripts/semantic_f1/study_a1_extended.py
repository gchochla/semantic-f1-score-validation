#!/usr/bin/env python
"""
Study A.1 — Synthetic (construct validity)

This script generates a synthetic label graph, constructs similarity matrices,
simulates gold labels and near/far‑miss predictors, and evaluates Hard vs
Semantic F1 metrics under multiple similarity settings. This includes semantic
precision and recall.

Outputs
- CSVs: per‑configuration summaries, Kendall tau vs radius, bootstrap CIs.
- Figures: metric vs radius (Fig A1) and Kendall tau bars (Fig A2).
- LaTeX table: bootstrap mean ± 95% CI for A vs B (Tab A1).

Example
  python scripts/semantic_f1/study_a1_extended.py \
      --num-labels 24 --num-examples 20000 \
      --k 1 2 3 --p 0.0 0.25 0.5 0.75 1.0 \
      --near-radii 1 2 --far-radii 3 4 \
      --alpha 1.0 0.75 0.5 0.25 0.0 \
      --outdir logs/semantic_f1/study_a

Notes
- Hard F1 uses sklearn (micro/macro/samples).
- Semantic metrics use llm_subj.metrics.semantic_f1.*.
- No network access required; relies on numpy/pandas/sklearn/scipy/matplotlib.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
import logging
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from semantic_f1_score import semantic_f1_score
from semantic_f1_score.semantic_f1 import samples_semantic_f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# External plotting helpers with robust import when run as a script
try:
    import scripts.semantic_f1.study_a1_plots_extended as study_a1_plots
except Exception:
    # Fallback: import sibling when executed directly via path
    sys.path.append(os.path.dirname(__file__))
    import study_a1_plots_extended as study_a1_plots  # type: ignore


RNG = np.random.default_rng(42)


def make_labels(n: int) -> list[str]:
    """Create canonical string labels ``[c0, c1, ..., c{n-1}]``.

    Args:
        n: Number of labels to create.

    Returns:
        List of label names of length ``n``.
    """
    return [f"c{i}" for i in range(n)]


def make_similarity_ring(n: int) -> pd.DataFrame:
    """Build ring-based similarity ``S`` using normalized cosine proximity.

    Labels lie on a unit circle at angles ``theta_i = 2π i / n`` and
    ``S[i,j] = (1 + cos(theta_i - theta_j)) / 2`` in ``[0,1]``. This avoids
    exponential kernels and directly matches the design note to use normalized
    cosine similarity as the ground-truth structure.

    Args:
        n: Number of labels arranged on a unit circle.

    Returns:
        A symmetric ``n x n`` DataFrame with values in ``[0,1]`` indexed by
        label names.
    """
    labels = make_labels(n)
    thetas = np.linspace(0.0, 2 * math.pi, num=n, endpoint=False)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dtheta = thetas[i] - thetas[j]
            mat[i, j] = (1.0 + math.cos(dtheta)) / 2.0
    return pd.DataFrame(mat, index=labels, columns=labels)


def identity_similarity(labels: list[str]) -> pd.DataFrame:
    """Return the identity similarity matrix for the given labels.

    Args:
        labels: Ordered label names to use for both axes.

    Returns:
        ``len(labels) x len(labels)`` identity matrix as a DataFrame.
    """
    n = len(labels)
    mat = np.eye(n, dtype=float)
    return pd.DataFrame(mat, index=labels, columns=labels)


def permuted_rows_similarity(
    S: pd.DataFrame, seed: int | None = None
) -> pd.DataFrame:
    """Create a degraded ``S`` by permuting only the rows.

    Intentionally breaks row/column alignment while preserving column order,
    simulating an invalid similarity source (as in Study A.1 controls).

    Args:
        S: Original similarity matrix with matching index/columns.
        seed: RNG seed for reproducibility.

    Returns:
        A copy of ``S`` with rows permuted and index reset to the original
        label names (so labels appear correct but alignment is broken).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(S))
    # Permute only rows (index) to intentionally break row/column alignment
    S_perm = S.copy().iloc[perm, :]
    S_perm.index = S.index  # keep original label names on rows
    return S_perm


def mix_similarity(
    S: pd.DataFrame,
    U: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Mix two similarity matrices as ``S_alpha = alpha*S + (1-alpha)*U``.

    Args:
        S: Primary similarity matrix.
        U: Secondary/noise similarity matrix (e.g., identity or uniform).
        alpha: Mixture weight in ``[0,1]``.

    Returns:
        Mixed similarity matrix (DataFrame) with the same indexing as ``S``.
    """
    assert 0.0 <= alpha <= 1.0
    mat = alpha * S.values + (1.0 - alpha) * U.values
    return pd.DataFrame(mat, index=S.index, columns=S.columns)


def make_uniform_noise_U(
    labels: list[str], offdiag: float = 0.5
) -> pd.DataFrame:
    """Create a uniform-noise matrix with unit diagonal and constant off-diagonal.

    Args:
        labels: Label names for axes.
        offdiag: Off-diagonal similarity value in ``[0,1]``.

    Returns:
        DataFrame where ``U[i,i]=1`` and ``U[i,j]=offdiag`` for ``i!=j``.
    """
    n = len(labels)
    mat = np.full((n, n), offdiag, dtype=float)
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=labels, columns=labels)


def make_uniform_gaussian_noise_U(
    labels: list[str], stddev: float = 0.1
) -> pd.DataFrame:
    """Create a Gaussian-noise matrix with unit diagonal and Gaussian off-diagonal.

    Args:
        labels: Label names for axes.
        stddev: Standard deviation for Gaussian noise added to off-diagonal.

    Returns:
        DataFrame where ``U[i,i]=1`` and ``U[i,j] ~ N(0, stddev^2)`` for ``i!=j``.
        Values are clipped to ``[0,1]``.
    """
    n = len(labels)
    mat = np.random.normal(loc=0.0, scale=stddev, size=(n, n)).astype(float)
    np.fill_diagonal(mat, 1.0)
    mat = np.clip(mat, 0.0, 1.0)
    return pd.DataFrame(mat, index=labels, columns=labels)


@dataclass
class SynthConfig:
    """Configuration for Study A.1 synthetic experiment.

    Fields
    - ``num_labels``: Number of labels on the ring.
    - ``num_examples``: Number of synthetic examples per configuration.
    - ``k_values``: Numbers of labels per example to sample for gold.
    - ``p_values``: Probabilities to perturb each gold label.
    - ``near_radii``: Radii for near-miss predictor A.
    - ``far_radii``: Radii for far-miss predictor B.
    - ``alphas``: Mix weights for ``S_alpha`` sensitivity sweeps.
    - ``outdir``: Output directory for CSVs/figures/tables.
    - ``seed``: RNG seed for reproducibility.
    - ``noise_mode``: Which ``U`` to use in the mix ("identity" or "uniform").
    - ``bootstrap``: Bootstrap iterations for CI estimation.
    """

    num_labels: int
    num_examples: int
    k_values: list[int]
    p_values: list[float]
    near_radii: list[int]
    far_radii: list[int]
    alphas: list[float]
    outdir: str = "logs/semantic_f1/study_a"
    seed: int = 123
    noise_mode: Literal["identity", "uniform", "gaussian"] = "identity"
    bootstrap: int = 200


def sample_gold_labels(n: int, k: int) -> list[int]:
    """Sample a gold set using normalized cosine weights around a latent center.

    Draw a center angle ``theta_c ~ Uniform[0,2π)``. For each label angle
    ``theta_j``, compute ``w_j = (1 + cos(Δθ(theta_j, theta_c))) / 2`` where
    ``Δθ`` is minimal angular difference in ``[0,π]``. Normalize ``w`` and sample
    up to ``k`` distinct labels without replacement from all ``n`` labels.

    Returns:
        List of distinct label indices (size ``<= k``).
    """
    theta_c = RNG.uniform(0.0, 2 * math.pi)
    thetas = np.linspace(0.0, 2 * math.pi, num=n, endpoint=False)

    dtheta = np.abs(thetas - theta_c)
    dtheta = np.minimum(dtheta, 2 * math.pi - dtheta)
    weights = (1.0 + np.cos(dtheta)) / 2.0
    s = float(weights.sum())
    p = (weights / s) if s > 0 else None

    chosen = RNG.choice(np.arange(n), size=min(k, n), replace=False, p=p)
    return list(map(int, chosen))


def perturb_labels(
    gold: Iterable[int],
    n: int,
    p: float,
    radius: int,
) -> list[int]:
    """Apply near/far perturbations to a gold label set on the ring.

    Each gold label is replaced with probability ``p`` by a label exactly
    ``radius`` steps away clockwise/counter‑clockwise; otherwise kept.

    Args:
        gold: Iterable of gold label indices.
        n: Total number of labels on the ring.
        p: Per-label replacement probability in ``[0,1]``.
        radius: Step distance for the replacement.

    Returns:
        List of unique perturbed indices (set semantics remove duplicates).
    """
    out: set[int] = set()
    for g in gold:
        if RNG.random() < p:
            # move exactly `radius` steps left or right on ring
            direction = -1 if RNG.random() < 0.5 else 1
            out.add((g + direction * radius) % n)
        else:
            out.add(g)
    return list(out)


def to_label_names(xs: list[list[int]], labels: list[str]) -> list[list[str]]:
    """Map index-encoded label sets to string names.

    Args:
        xs: List of examples, each a list of integer indices.
        labels: Master list mapping index → name.

    Returns:
        Same structure as ``xs`` but with names.
    """
    return [[labels[i] for i in row] for row in xs]


def eval_hard_metrics(
    trues: list[list[str]], preds: list[list[str]], labels: list[str]
) -> dict[str, float]:
    """Compute sklearn Hard F1 variants for multilabel sets.

    Args:
        trues: Gold labels per example (strings).
        preds: Predicted labels per example (strings).
        labels: Global class order for binarization.

    Returns:
        Dict with keys ``{"hard_micro","hard_macro","hard_samples"}``.
    """
    mlb = MultiLabelBinarizer(classes=labels)
    Y_true = mlb.fit_transform(trues)
    Y_pred = mlb.transform(preds)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    samples = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
    return {"hard_micro": micro, "hard_macro": macro, "hard_samples": samples}


def eval_semantic_metrics(
    trues: list[list[str]],
    preds: list[list[str]],
    S: pd.DataFrame,
    metrics_filter: set[str] | None = None,
) -> dict[str, float]:
    """Compute Semantic F1, Precision, and Recall variants under a given similarity matrix.

    Args:
        trues: Gold labels per example (strings).
        preds: Predicted labels per example (strings).
        S: Similarity matrix ``[0,1]`` indexed by label names.
        metrics_filter: Optional set of metric names to compute. If None, computes all.
                       Supports filtering by averaging mode prefixes: "micro", "macro", "samples"
                       or full metric names like "sem_f1_micro".

    Returns:
        Dict with keys for F1, precision, and recall for each averaging mode:
        ``{"sem_f1_micro", "sem_precision_micro", "sem_recall_micro",
           "sem_f1_macro", "sem_precision_macro", "sem_recall_macro",
           "sem_f1_samples", "sem_precision_samples", "sem_recall_samples"}``.
    """
    # Get components for each averaging mode using samples_semantic_f1_score
    results = {}

    # Helper to check if we should compute a metric
    def should_compute(avg_mode: str) -> bool:
        if metrics_filter is None:
            return True
        # Check if any metric with this averaging mode is requested
        return any(
            avg_mode in m
            or m
            in [
                f"sem_f1_{avg_mode}",
                f"sem_precision_{avg_mode}",
                f"sem_recall_{avg_mode}",
            ]
            for m in metrics_filter
        )

    # Samples averaging with components
    if should_compute("samples"):
        samples_results = samples_semantic_f1_score(
            preds, trues, S, return_components=True
        )
        results["sem_f1_samples"] = float(samples_results["f1"])
        results["sem_precision_samples"] = float(samples_results["precision"])
        results["sem_recall_samples"] = float(samples_results["recall"])

    # Micro averaging with components
    if should_compute("micro"):
        from semantic_f1_score.semantic_f1 import semantic_micro_f1_score

        micro_results = semantic_micro_f1_score(
            preds, trues, S, return_components=True
        )
        results["sem_f1_micro"] = float(micro_results["f1"])
        results["sem_precision_micro"] = float(micro_results["precision"])
        results["sem_recall_micro"] = float(micro_results["recall"])

    # Macro averaging with components
    if should_compute("macro"):
        from semantic_f1_score.semantic_f1 import semantic_macro_f1_score

        macro_results = semantic_macro_f1_score(
            preds, trues, S, return_components=True
        )
        results["sem_f1_macro"] = float(macro_results["f1"])
        results["sem_precision_macro"] = float(macro_results["precision"])
        results["sem_recall_macro"] = float(macro_results["recall"])

    return results


def bootstrap_ci_all_metrics(
    trues: list[list[str]],
    preds_a: list[list[str]],
    preds_b: list[list[str]],
    labels: list[str],
    S: pd.DataFrame,
    B: int = 200,
    seed: int | None = 1234,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap the mean gap ``perf_a - perf_b`` for all metrics with 95% CI.

    Performs bootstrapping ONCE and computes all metrics from each bootstrap sample.
    This is much more efficient than calling bootstrap_ci separately for each metric.

    Args:
        trues: Gold labels per example.
        preds_a: Predictions from system A.
        preds_b: Predictions from system B.
        labels: Class order for hard metrics.
        S: Similarity matrix for semantic metrics.
        B: Number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict mapping metric name to tuple ``(mean_gap, ci95_lo, ci95_hi)``.
    """
    rng = np.random.default_rng(seed)
    n = len(trues)

    # Prepare encoder for hard F1 only once
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score as sk_f1

    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([[]])

    # All metrics we'll compute
    all_metrics = [
        "hard_micro",
        "hard_macro",
        "hard_samples",
        "sem_f1_micro",
        "sem_f1_macro",
        "sem_f1_samples",
        "sem_precision_micro",
        "sem_precision_macro",
        "sem_precision_samples",
        "sem_recall_micro",
        "sem_recall_macro",
        "sem_recall_samples",
    ]

    # Storage for gaps across bootstrap iterations
    gaps_by_metric: dict[str, list[float]] = {m: [] for m in all_metrics}

    base_idx = np.arange(n)
    for _ in tqdm(range(B), desc="Bootstrap iterations", leave=False):
        idx = rng.choice(base_idx, size=n, replace=True)
        T = [trues[i] for i in idx]
        Pa = [preds_a[i] for i in idx]
        Pb = [preds_b[i] for i in idx]

        # Compute hard metrics
        Y_true = mlb.transform(T)
        Y_pred_a = mlb.transform(Pa)
        Y_pred_b = mlb.transform(Pb)

        for avg in ["micro", "macro", "samples"]:
            metric_name = f"hard_{avg}"
            val_a = float(sk_f1(Y_true, Y_pred_a, average=avg, zero_division=0))
            val_b = float(sk_f1(Y_true, Y_pred_b, average=avg, zero_division=0))
            gaps_by_metric[metric_name].append(val_a - val_b)

        # Compute semantic metrics (all at once)
        sem_a = eval_semantic_metrics(T, Pa, S)
        sem_b = eval_semantic_metrics(T, Pb, S)

        for sem_metric in [
            "sem_f1_micro",
            "sem_f1_macro",
            "sem_f1_samples",
            "sem_precision_micro",
            "sem_precision_macro",
            "sem_precision_samples",
            "sem_recall_micro",
            "sem_recall_macro",
            "sem_recall_samples",
        ]:
            gaps_by_metric[sem_metric].append(
                sem_a[sem_metric] - sem_b[sem_metric]
            )

    # Compute statistics for each metric
    results = {}
    for metric_name, gaps in gaps_by_metric.items():
        gaps_arr = np.array(gaps, dtype=float)
        mean = float(gaps_arr.mean())
        lo, hi = np.percentile(gaps_arr, [2.5, 97.5])
        results[metric_name] = (mean, float(lo), float(hi))

    return results


def bootstrap_ci(
    trues: list[list[str]],
    preds_a: list[list[str]],
    preds_b: list[list[str]],
    labels: list[str],
    S: pd.DataFrame,
    metric: Literal[
        "hard_micro",
        "hard_macro",
        "hard_samples",
        "sem_f1_micro",
        "sem_f1_macro",
        "sem_f1_samples",
        "sem_precision_micro",
        "sem_precision_macro",
        "sem_precision_samples",
        "sem_recall_micro",
        "sem_recall_macro",
        "sem_recall_samples",
    ],
    B: int = 200,
    seed: int | None = 1234,
) -> tuple[float, float, float]:
    """Bootstrap the mean gap ``perf_a - perf_b`` for a selected metric with 95% CI.

    DEPRECATED: Use bootstrap_ci_all_metrics for better performance when computing
    multiple metrics.

    Resamples examples with replacement ``B`` times, recomputing the chosen
    metric for each bootstrap sample and returning the mean and percentile CI.

    Args:
        trues: Gold labels per example.
        preds_a: Predictions from system A.
        preds_b: Predictions from system B.
        labels: Class order for hard metrics.
        S: Similarity matrix for semantic metrics.
        metric: Which metric to evaluate (hard/semantic x micro/macro/samples).
        B: Number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        Tuple ``(mean_gap, ci95_lo, ci95_hi)``.
    """
    # Just call the optimized version and extract the requested metric
    all_results = bootstrap_ci_all_metrics(
        trues, preds_a, preds_b, labels, S, B, seed
    )
    return all_results[metric]


def kendall_tau_vs_radius(
    n: int,
    k: int,
    p: float,
    radii: list[int],
    S_list: dict[str, pd.DataFrame],
    num_examples: int,
    labels: list[str],
) -> pd.DataFrame:
    """Compute Kendall's tau between radius and metric curves.

    Builds a fixed gold set and, for each radius, evaluates Hard and Semantic
    metrics. Returns one row per metric with tau and p-value vs the radius list.

    Args:
        n: Number of labels on the ring.
        k: Gold labels per example.
        p: Per-label perturbation probability.
        radii: Radii to evaluate.
        S_list: Mapping name → similarity matrix for semantic metrics.
        num_examples: Number of examples for the gold set.
        labels: Label names.

    Returns:
        DataFrame with columns ``{"metric","tau","pvalue"}``.
    """
    # Build a common set of trues, and then generate preds for each radius
    gold = [sample_gold_labels(n, k) for _ in range(num_examples)]
    gold_names = to_label_names([sorted(set(g)) for g in gold], labels)

    metrics_curve: dict[str, list[float]] = {}
    for r in radii:
        preds = [perturb_labels(g, n, p=p, radius=r) for g in gold]
        preds_names = to_label_names([sorted(set(x)) for x in preds], labels)

        # hard F1s
        hard_vals = eval_hard_metrics(gold_names, preds_names, labels)
        for k_ in hard_vals:
            metrics_curve.setdefault(k_, []).append(hard_vals[k_])

        # semantic for each S
        for name, S in S_list.items():
            sem_vals = eval_semantic_metrics(gold_names, preds_names, S)
            for k_, v in sem_vals.items():
                metrics_curve.setdefault(f"{k_}:{name}", []).append(v)

    # Now compute tau for each curve against radius list
    df_rows = []
    for mname, vals in metrics_curve.items():
        tau, pval = kendalltau(radii, vals)
        df_rows.append(
            {"metric": mname, "tau": float(tau), "pvalue": float(pval)}
        )

    return pd.DataFrame(df_rows)


def metrics_vs_radius(
    n: int,
    k: int,
    p: float,
    radii: list[int],
    S_list: dict[str, pd.DataFrame],
    num_examples: int,
    labels: list[str],
) -> pd.DataFrame:
    """Compute per-radius metric curves for Hard/Semantic F1.

    Produces rows with columns ``{"radius","metric","value"}``. Semantic
    metric names are namespaced as ``"sem_<avg>:<Sname>"``.

    Args:
        n: Number of labels on the ring.
        k: Gold labels per example.
        p: Per-label perturbation probability.
        radii: Radii to evaluate.
        S_list: Mapping name → similarity matrix for semantic metrics.
        num_examples: Number of examples for the gold set.
        labels: Label names.

    Returns:
        Long-form DataFrame of metric values per radius.
    """
    gold = [sample_gold_labels(n, k) for _ in range(num_examples)]
    gold_names = to_label_names([sorted(set(g)) for g in gold], labels)

    rows = []
    for r in radii:
        preds = [perturb_labels(g, n, p=p, radius=r) for g in gold]
        preds_names = to_label_names([sorted(set(x)) for x in preds], labels)

        hard_vals = eval_hard_metrics(gold_names, preds_names, labels)
        for mname, val in hard_vals.items():
            rows.append({"radius": r, "metric": mname, "value": float(val)})

        for name, S in S_list.items():
            sem_vals = eval_semantic_metrics(gold_names, preds_names, S)
            for mname, val in sem_vals.items():
                rows.append(
                    {
                        "radius": r,
                        "metric": f"{mname}:{name}",
                        "value": float(val),
                    }
                )
    return pd.DataFrame(rows)


def run_one_config(
    cfg: SynthConfig,
    outdir: str,
):
    """Run Study A.1 for a single configuration and write artifacts.

    Side effects
    - Writes CSVs: summary, Kendall tau, bootstrap results.
    - Writes figures: Fig A1/A2 (PNG/PDF).
    - Writes LaTeX table: Tab A1 with bootstrap CIs.
    - Logs to ``<outdir>/log.txt``.

    Args:
        cfg: Experiment configuration.
        outdir: Output directory (created if missing).
    """
    os.makedirs(outdir, exist_ok=True)

    # set up file logger
    logger = logging.getLogger("study_a")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if re-run in interactive sessions
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(os.path.join(outdir, "log.txt"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Starting Study A with config: %s", cfg)
    labels = make_labels(cfg.num_labels)
    S_ideal = make_similarity_ring(cfg.num_labels)
    if cfg.noise_mode == "uniform":
        U = make_uniform_noise_U(labels, offdiag=0.5)
    elif cfg.noise_mode == "gaussian":
        U = make_uniform_gaussian_noise_U(labels, stddev=0.5)
    else:
        U = identity_similarity(labels)
    logger.info(
        "Built similarity matrices: ideal=%s, noise_mode=%s",
        S_ideal.shape,
        cfg.noise_mode,
    )

    # Placeholders for summaries
    rows_summary = []
    rows_tau = []
    rows_boot = []

    # Iterate k and p; generate a gold set for each (to keep fair across A/B/radii)
    for k in tqdm(cfg.k_values, desc="Processing k values"):
        for p in tqdm(
            cfg.p_values, desc=f"Processing p values (k={k})", leave=False
        ):
            logger.info(
                "Begin (k=%s, p=%.2f): generating gold set of %s examples",
                k,
                p,
                cfg.num_examples,
            )
            # Pre-generate a gold dataset
            gold_idx = [
                sample_gold_labels(cfg.num_labels, k)
                for _ in range(cfg.num_examples)
            ]
            gold_idx = [sorted(set(g)) for g in gold_idx]
            gold = to_label_names(gold_idx, labels)

            # Define S variants
            S_perm = permuted_rows_similarity(S_ideal, seed=cfg.seed)
            S_alpha_list = {
                f"a={a:.2f}": mix_similarity(S_ideal, U, a) for a in cfg.alphas
            }
            S_variants: dict[str, pd.DataFrame] = {
                "ideal": S_ideal,
                "permuted": S_perm,
                **S_alpha_list,
            }
            logger.info(
                "Prepared %d S variants for (k=%s, p=%.2f)",
                len(S_variants),
                k,
                p,
            )

            # Evaluate kendall tau vs radius curves for this (k,p)
            radii_union = sorted(set(cfg.near_radii + cfg.far_radii))
            df_tau = kendall_tau_vs_radius(
                cfg.num_labels,
                k,
                p,
                radii_union,
                {k_: v for k_, v in S_variants.items() if k_ != "identity"},
                cfg.num_examples,
                labels,
            )
            df_tau["k"] = k
            df_tau["p"] = p
            rows_tau.append(df_tau)
            logger.info(
                "Computed Kendall tau for (k=%s, p=%.2f) across %d radii → %d rows",
                k,
                p,
                len(radii_union),
                len(df_tau),
            )

            # For tab A1: compare predictor A (near) vs B (far)
            for r_near in tqdm(
                cfg.near_radii,
                desc=f"Near radii (k={k}, p={p:.2f})",
                leave=False,
            ):
                for r_far in tqdm(
                    cfg.far_radii,
                    desc=f"Far radii (r_near={r_near})",
                    leave=False,
                ):
                    logger.info(
                        "Evaluating near=%s vs far=%s (k=%s, p=%.2f) across %d S variants",
                        r_near,
                        r_far,
                        k,
                        p,
                        len(S_variants),
                    )
                    # Simulate A and B on the same gold
                    preds_a_idx = [
                        perturb_labels(g, cfg.num_labels, p=p, radius=r_near)
                        for g in gold_idx
                    ]
                    preds_b_idx = [
                        perturb_labels(g, cfg.num_labels, p=p, radius=r_far)
                        for g in gold_idx
                    ]
                    preds_a = to_label_names(
                        [sorted(set(x)) for x in preds_a_idx], labels
                    )
                    preds_b = to_label_names(
                        [sorted(set(x)) for x in preds_b_idx], labels
                    )

                    # Evaluate hard and semantic metrics under all S
                    hard_a = eval_hard_metrics(gold, preds_a, labels)
                    hard_b = eval_hard_metrics(gold, preds_b, labels)

                    for name_s, S in tqdm(
                        S_variants.items(),
                        desc=f"S variants (r_near={r_near}, r_far={r_far})",
                        leave=False,
                    ):
                        sem_a = eval_semantic_metrics(gold, preds_a, S)
                        sem_b = eval_semantic_metrics(gold, preds_b, S)

                        row = dict(
                            k=k,
                            p=p,
                            r_near=r_near,
                            r_far=r_far,
                            S=name_s,
                        )
                        # merge metrics
                        for mname, aval in hard_a.items():
                            row[f"{mname}_A"] = aval
                            row[f"{mname}_B"] = hard_b[mname]
                            row[f"{mname}_gap"] = aval - hard_b[mname]
                        for mname, aval in sem_a.items():
                            row[f"{mname}_A"] = aval
                            row[f"{mname}_B"] = sem_b[mname]
                            row[f"{mname}_gap"] = aval - sem_b[mname]
                        rows_summary.append(row)

                        # Bootstrap CIs for all metrics at once (much more efficient)
                        logger.info(
                            "Bootstrapping CI for all metrics (B=%d) [near=%s, far=%s, S=%s]",
                            cfg.bootstrap,
                            r_near,
                            r_far,
                            name_s,
                        )
                        bootstrap_results = bootstrap_ci_all_metrics(
                            gold,
                            preds_a,
                            preds_b,
                            labels,
                            S,
                            B=cfg.bootstrap,
                            seed=cfg.seed,
                        )

                        # Store results for all metrics
                        if p != 0:
                            for m, (mean, lo, hi) in bootstrap_results.items():
                                rows_boot.append(
                                    dict(
                                        k=k,
                                        p=p,
                                        r_near=r_near,
                                        r_far=r_far,
                                        S=name_s,
                                        metric=m,
                                        mean_gap=mean,
                                        ci95_lo=lo,
                                        ci95_hi=hi,
                                    )
                                )

    # Write outputs
    ts = int(time.time())
    df_summary = pd.DataFrame(rows_summary)
    df_tau_all = (
        pd.concat(rows_tau, ignore_index=True) if rows_tau else pd.DataFrame()
    )
    df_boot = pd.DataFrame(rows_boot)

    path_summary = os.path.join(outdir, f"studyA_summary_{ts}.csv")
    path_tau = os.path.join(outdir, f"studyA_kendall_tau_{ts}.csv")
    path_boot = os.path.join(outdir, f"studyA_bootstrap_{ts}.csv")
    df_summary.to_csv(path_summary, index=False)
    df_tau_all.to_csv(path_tau, index=False)
    df_boot.to_csv(path_boot, index=False)

    logger.info(
        "Output shapes — summary=%s, kendall_tau=%s, bootstrap=%s",
        df_summary.shape,
        df_tau_all.shape,
        df_boot.shape,
    )
    logger.info("Wrote: %s", path_summary)
    logger.info("Wrote: %s", path_tau)
    logger.info("Wrote: %s", path_boot)
    # keep prints for quick terminal visibility
    print(f"Wrote: {path_summary}")
    print(f"Wrote: {path_tau}")
    print(f"Wrote: {path_boot}")

    # --------------------------
    # Figures and Tables outputs
    # --------------------------
    try:
        # Use external plotting functions
        helpers = dict(
            sample_gold_labels=sample_gold_labels,
            to_label_names=to_label_names,
            perturb_labels=perturb_labels,
            semantic_f1_score=semantic_f1_score,
            f1_score=f1_score,
            MultiLabelBinarizer=MultiLabelBinarizer,
        )
        study_a1_plots.plot_metric_vs_radius_grids(
            cfg=cfg,
            labels=labels,
            S_ideal=S_ideal,
            helpers=helpers,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        study_a1_plots.plot_kendall_tau_bars(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # Faceted grids: negative Kendall tau per (k,p) for each averaging metric
        study_a1_plots.plot_kendall_tau_grids(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # A2 (lines): per-k line panels over p
        study_a1_plots.plot_kendall_tau_linepanels(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # A2 Table: negative Kendall tau values by k×variant with p columns
        study_a1_plots.write_kendall_tau_table(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Replace the massive table with useful visualizations and summary tables

        # Heatmap visualizations of bootstrap results
        study_a1_plots.plot_bootstrap_heatmaps(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Summary charts showing key findings
        study_a1_plots.plot_bootstrap_summary_charts(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Confidence interval plots for detailed analysis
        study_a1_plots.plot_bootstrap_confidence_intervals(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Concise summary table (replaces the original massive table)
        study_a1_plots.write_bootstrap_table_summary(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Optional: Still generate the original full table for appendix/reference
        # (Commented out by default due to excessive length)
        # study_a1_plots.write_bootstrap_table(
        #     df_boot=df_boot,
        #     outdir=outdir,
        #     ts=ts,
        #     logger=logger,
        # )
    except Exception as e:  # pragma: no cover - plotting is best-effort
        # Do not crash the run if plotting fails; log and proceed
        logging.getLogger("study_a").exception(
            "Non-fatal error while producing figures/tables: %s", e
        )


def parse_args() -> SynthConfig:
    """Parse CLI arguments into a ``SynthConfig``.

    Returns:
        A populated ``SynthConfig`` reflecting command-line options.
    """
    ap = argparse.ArgumentParser(
        description="Study A synthetic evaluation for Semantic F1"
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
        "--p",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="List of probabilities to perturb each gold label (controls prediction noise).",
    )
    ap.add_argument(
        "--near-radii",
        type=int,
        nargs="+",
        default=[1, 2],
        help="List of radii for near-miss predictor A (distance for label replacement).",
    )
    ap.add_argument(
        "--far-radii",
        type=int,
        nargs="+",
        default=[3, 4],
        help="List of radii for far-miss predictor B (distance for label replacement).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help="List of mixture weights for similarity matrix sensitivity sweeps (alpha in S_alpha = alpha*S + (1-alpha)*U).",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--noise-mode",
        choices=["identity", "uniform", "gaussian"],
        default="identity",
        help="Type of noise matrix U to use in similarity mixing: 'identity' for identity matrix, 'uniform' for constant off-diagonal.",
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
        default="logs/semantic_f1/study_a",
        help="Output directory for CSVs, figures, and tables.",
    )
    args = ap.parse_args()
    return SynthConfig(
        num_labels=args.num_labels,
        num_examples=args.num_examples,
        k_values=args.k,
        p_values=args.p,
        near_radii=args.near_radii,
        far_radii=args.far_radii,
        alphas=args.alpha,
        seed=args.seed,
        noise_mode=args.noise_mode,
        outdir=args.outdir,
        bootstrap=args.bootstrap,
    )


def main():
    """Entry point: parse args and run Study A.1."""
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    run_one_config(cfg, outdir=cfg.outdir)


if __name__ == "__main__":
    main()
