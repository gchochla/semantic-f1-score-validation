#!/usr/bin/env python
"""
Study A.4 — Comparison to baselines

This script validates that Semantic F1's two-step approach (precision and recall)
is necessary by constructing synthetic scenarios where individual components fail
but Semantic F1 succeeds. Tests three scenarios comparing Semantic F1 against:
1. Semantic Precision alone (precision test)
2. Semantic Recall alone (recall test) 
3. Extended Hungarian matching (hungarian test)

Updated scenarios:

1. **Precision Test**: Bimodal gold labels, unimodal predictor that predicts only
   k//2 labels from one mode when gold has two modes. Shows Semantic F1 (harmonic 
   mean) performs better than semantic precision alone when recall matters.

2. **Recall Test**: Unimodal gold labels around center 0, bimodal predictor that
   samples k//2 from gold mode + k//2 from opposite mode when two-mode-probability
   is active. Shows Semantic F1 performs better than semantic recall alone when
   precision matters.

3. **Hungarian Test**: Compares Semantic F1 against Extended Hungarian matching
   when both frequency of two modes in gold AND number of predictions vary.
   Shows two-step F1 approach handles varying prediction counts better than
   pure assignment-based matching.

Outputs
- CSVs: per‑configuration summaries with gap metrics.
- Figures: semantic F1 vs baseline across two-mode frequency (Fig A10-A12).

Example Usage:
  # Test basic functionality
  python scripts/semantic_f1/study_a4.py test
  
  # Run precision scenario (semantic F1 vs semantic precision)
  python scripts/semantic_f1/study_a4.py precision \
      --num-labels 24 --num-examples 10000 \
      --k 1 2 3 --q 0.0 0.25 0.5 0.75 1.0 \
      --two-mode-frequencies 0.0 0.2 0.4 0.6 0.8 1.0 \
      --outdir logs/semantic_f1/study_a4
      
  # Run recall scenario (semantic F1 vs semantic recall)
  python scripts/semantic_f1/study_a4.py recall \
      --num-labels 24 --num-examples 10000 \
      --k 1 2 3 --q 0.0 0.25 0.5 0.75 1.0 \
      --two-mode-frequencies 0.0 0.2 0.4 0.6 0.8 1.0 \
      --outdir logs/semantic_f1/study_a4
      
  # Run Hungarian scenario (semantic F1 vs Hungarian)
  python scripts/semantic_f1/study_a4.py hungarian \
      --num-labels 24 --num-examples 10000 \
      --k 1 2 3 --prediction-counts 1 2 3 4 5 \
      --two-mode-frequencies 0.0 0.2 0.4 0.6 0.8 1.0 \
      --outdir logs/semantic_f1/study_a4

Notes
- Parameter 'q' replaces 'p' from Study A.1 to avoid confusion (per PROJECT.md).
- All comparisons use "samples" averaging (mean across examples).
- Tests specific failure modes where individual components fail but F1 succeeds.
- Uses circular label arrangement with proper bimodal structure.
- Hungarian algorithm may print debug information during execution.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, List, Tuple
import logging

import numpy as np
import pandas as pd
from semantic_f1_score import semantic_f1_score, hungarian_score

# Import shared utilities from study_a1 and plotting
try:
    import scripts.semantic_f1.study_a1 as study_a1
    import scripts.semantic_f1.study_a4_plots as study_a4_plots
except Exception:
    sys.path.append(os.path.dirname(__file__))
    import study_a1  # type: ignore
    import study_a4_plots as study_a4_plots  # type: ignore

RNG = np.random.default_rng(42)


@dataclass
class StudyA4Config:
    """Configuration for Study A.4 baseline comparison experiment.

    Fields:
    - scenario: Which test scenario to run ("precision", "recall", "hungarian")
    - num_labels: Number of labels on the ring
    - num_examples: Number of synthetic examples per configuration
    - k_values: Numbers of labels per example to sample for gold
    - q_values: Probabilities to perturb each gold label (renamed from p_values per PROJECT.md)
    - two_mode_frequencies: Frequencies of two-mode examples (0.0 = always unimodal, 1.0 = always bimodal)
    - prediction_counts: For Hungarian test, number of predictions to make (in addition to k_values)
    - outdir: Output directory for CSVs/figures/tables
    - seed: RNG seed for reproducibility
    - bootstrap: Bootstrap iterations for CI estimation
    - no_bootstrap: Skip expensive bootstrap CI computation
    - radius: Radius around center for label sampling and perturbation
    """

    scenario: Literal["precision", "recall", "hungarian"]
    num_labels: int
    num_examples: int
    k_values: List[int]
    q_values: List[float]  # Changed from p_values to q_values
    two_mode_frequencies: List[float]
    prediction_counts: List[int] = None  # Only used for Hungarian scenario
    outdir: str = "logs/semantic_f1/study_a4"
    seed: int = 123
    bootstrap: int = 25
    no_bootstrap: bool = False
    radius: int = 2
    temperature: float = 2.0
    pred_temperature: float = 2.0


def make_bimodal_structure(n: int, kappa: float = 2.0) -> Tuple[int, int]:
    """Create two mode centers for circular labels.

    Args:
        n: Number of labels on the circle
        kappa: Concentration parameter (unused in simplified version)

    Returns:
        Tuple of (positive_mode_center, negative_mode_center) as label indices
    """
    # Two centers opposite each other on the circle
    pos_center = 0
    neg_center = n // 2
    return pos_center, neg_center


def sample_around_center(
    center: int, k: int, n: int, temperature: float
) -> List[int]:
    """Sample k labels around a center using softmax-weighted cosine similarity.

    Args:
        center: Center label index
        k: Number of labels to sample
        n: Total number of labels
        radius: Radius around center (used for concentration parameter)

    Returns:
        List of sampled label indices
    """
    # Create cosine similarity weights centered around the given center
    thetas = np.linspace(0.0, 2 * math.pi, num=n, endpoint=False)
    center_theta = thetas[center]

    # Calculate raw cosine similarity scores
    cos_similarities = np.cos(thetas - center_theta)

    # Apply softmax with temperature for concentration around mode
    # Ensure temperature is not too small to avoid numerical issues
    safe_temperature = max(temperature, 1e-6)
    logits = cos_similarities / safe_temperature
    weights = np.exp(
        logits - np.max(logits)
    )  # Subtract max for numerical stability
    weights = weights / weights.sum()

    # Sample without replacement
    if k >= n:
        return list(range(n))

    indices = RNG.choice(n, size=k, replace=False, p=weights)
    return sorted(indices.tolist())


def sample_bimodal_labels(
    n: int,
    k: int,
    pos_center: int,
    neg_center: int,
    two_mode_prob: float,
    temperature: float,
) -> List[int]:
    """Sample labels with controlled probability of being bimodal.

    Args:
        n: Total number of labels
        k: Number of labels to sample
        pos_center: Positive mode center index
        neg_center: Negative mode center index
        two_mode_prob: Probability of sampling from both modes
        radius: Radius parameter for concentration

    Returns:
        List of sampled label indices
    """
    if RNG.random() < two_mode_prob and k >= 2:
        # Sample from both modes - k//2 from each
        k_per_mode = k // 2
        k_remaining = k - 2 * k_per_mode

        pos_labels = sample_around_center(
            pos_center, k_per_mode, n, temperature
        )
        neg_labels = sample_around_center(
            neg_center, k_per_mode, n, temperature
        )

        all_labels = pos_labels + neg_labels

        # If k is odd, add one more from random mode
        if k_remaining > 0:
            extra_center = pos_center if RNG.random() < 0.5 else neg_center
            extra_labels = sample_around_center(
                extra_center, k_remaining, n, temperature
            )
            all_labels.extend(extra_labels)

        return sorted(list(set(all_labels)))  # Remove duplicates and sort
    else:
        # Sample from single mode
        center = pos_center if RNG.random() < 0.5 else neg_center
        return sample_around_center(center, k, n, temperature)


def sample_from_mode(
    n: int,
    mode: str,
    k: int,
    pos_center: int,
    neg_center: int,
    temperature: float,
) -> List[int]:
    """Sample k labels from the specified mode.

    Args:
        n: Total number of labels
        mode: "positive" or "negative"
        k: Number of labels to sample
        pos_center: Positive mode center index
        neg_center: Negative mode center index
        temperature: Temperature parameter for concentration

    Returns:
        List of sampled label indices
    """
    center = pos_center if mode == "positive" else neg_center
    return sample_around_center(center, k, n, temperature)


def calculate_mode_weights(
    label_indices: List[int], pos_center: int, neg_center: int, n: int
) -> dict:
    """Calculate mode weights W+(Y), W-(Y) for a set of labels.

    Following PROJECT.md specification for proper mode determination.

    Args:
        label_indices: List of label indices
        pos_center: Positive mode center index
        neg_center: Negative mode center index
        n: Total number of labels

    Returns:
        Dict with 'pos' and 'neg' weights indicating membership in each mode
    """
    if not label_indices:
        return {'pos': 0.0, 'neg': 0.0}

    def circular_distance(idx1: int, idx2: int) -> int:
        """Calculate circular distance between two indices."""
        return min(abs(idx1 - idx2), n - abs(idx1 - idx2))

    # Define fuzzy membership based on distance to mode centers
    # Labels closer to a mode center have higher membership weight
    mode_radius = n // 6  # Define what's considered "close" to a mode

    pos_weight = 0.0
    neg_weight = 0.0

    for idx in label_indices:
        pos_dist = circular_distance(idx, pos_center)
        neg_dist = circular_distance(idx, neg_center)

        # Use exponential decay for membership weights
        pos_membership = (
            np.exp(-pos_dist / mode_radius) if pos_dist <= mode_radius else 0.0
        )
        neg_membership = (
            np.exp(-neg_dist / mode_radius) if neg_dist <= mode_radius else 0.0
        )

        pos_weight += pos_membership
        neg_weight += neg_membership

    # Normalize by number of labels
    pos_weight /= len(label_indices)
    neg_weight /= len(label_indices)

    return {'pos': pos_weight, 'neg': neg_weight}


def determine_dominant_mode(
    label_indices: List[int], pos_center: int, neg_center: int, n: int
) -> str:
    """Determine whether a set of labels is primarily positive or negative mode.

    Args:
        label_indices: List of label indices
        pos_center: Positive mode center index
        neg_center: Negative mode center index
        n: Total number of labels

    Returns:
        "positive" or "negative" based on closest mode
    """

    def distance_to_center(idx: int, center: int) -> int:
        """Circular distance to center."""
        return min(abs(idx - center), n - abs(idx - center))

    pos_distances = [
        distance_to_center(idx, pos_center) for idx in label_indices
    ]
    neg_distances = [
        distance_to_center(idx, neg_center) for idx in label_indices
    ]

    avg_pos_distance = sum(pos_distances) / len(pos_distances)
    avg_neg_distance = sum(neg_distances) / len(neg_distances)

    return "positive" if avg_pos_distance <= avg_neg_distance else "negative"


def create_precision_test_data(cfg: StudyA4Config) -> pd.DataFrame:
    """Create test data for precision scenario: bimodal gold, unimodal predictor.

    Following PROJECT.md specification:
    - Gold labels: Can be one or two modes based on two-mode-probability
      - If one mode: sample k labels around that center
      - If two modes: sample k//2 from each center
    - Predictor: Always unimodal, predicts k//2 labels when gold is bimodal
      - For bimodal gold: pick only one of the gold modes and predict k//2 labels
      - For unimodal gold: predict normally (with optional hopping)

    This tests whether semantic F1 handles cases better than semantic precision
    alone when predictor gets mode right but predicts fewer labels.
    """
    labels = study_a1.make_labels(cfg.num_labels)
    S_ideal = study_a1.make_similarity_ring(cfg.num_labels)
    pos_center, neg_center = make_bimodal_structure(cfg.num_labels)

    rows = []

    for k in cfg.k_values:
        for q in cfg.q_values:  # Changed from p to q per PROJECT.md
            for two_mode_freq in cfg.two_mode_frequencies:
                # Generate gold labels with controlled bimodal frequency
                gold_indices = []
                for i in range(cfg.num_examples):
                    gold_idx = sample_bimodal_labels(
                        cfg.num_labels,
                        k,
                        pos_center,
                        neg_center,
                        two_mode_freq,
                        cfg.temperature,
                    )
                    gold_indices.append(gold_idx)

                gold_labels = study_a1.to_label_names(gold_indices, labels)

                # Create unimodal predictor following PROJECT.md spec
                pred_indices = []
                for gold_idx in gold_indices:
                    # Determine if gold is actually bimodal by checking mode representation
                    gold_mode_weights = calculate_mode_weights(
                        gold_idx, pos_center, neg_center, cfg.num_labels
                    )
                    is_gold_bimodal = (
                        gold_mode_weights['pos'] > 0
                        and gold_mode_weights['neg'] > 0
                    )

                    if is_gold_bimodal:
                        # For bimodal gold: predictor picks ONE mode and predicts k//2 labels
                        dominant_mode = (
                            'pos'
                            if gold_mode_weights['pos']
                            >= gold_mode_weights['neg']
                            else 'neg'
                        )
                        center = (
                            pos_center if dominant_mode == 'pos' else neg_center
                        )
                        num_pred_labels = max(1, k // 2)  # At least 1 label

                        pred_idx = sample_around_center(
                            center,
                            num_pred_labels,
                            cfg.num_labels,
                            temperature=cfg.pred_temperature,
                        )
                    else:
                        # For unimodal gold: predict normally with optional hopping
                        if RNG.random() < q:  # Apply hopping perturbation
                            dominant_mode = determine_dominant_mode(
                                gold_idx, pos_center, neg_center, cfg.num_labels
                            )
                            center = (
                                pos_center
                                if dominant_mode == "positive"
                                else neg_center
                            )
                            pred_idx = sample_around_center(
                                center,
                                k,
                                cfg.num_labels,
                                temperature=cfg.pred_temperature,
                            )
                        else:  # Keep original
                            pred_idx = gold_idx

                    pred_indices.append(pred_idx)

                pred_labels = study_a1.to_label_names(pred_indices, labels)

                # Get semantic F1 components using samples averaging
                from semantic_f1_score.semantic_f1 import (
                    samples_semantic_f1_score,
                )

                semantic_results = samples_semantic_f1_score(
                    pred_labels, gold_labels, S_ideal, return_components=True
                )

                # Store results - comparing semantic F1 vs semantic precision (and include recall)
                row = {
                    'k': k,
                    'q': q,  # Changed from p to q
                    'two_mode_freq': two_mode_freq,
                    'semantic_f1': semantic_results['f1'],
                    'semantic_precision': semantic_results['precision'],
                    'semantic_recall': semantic_results['recall'],
                    'gap_f1_minus_precision': semantic_results['f1']
                    - semantic_results['precision'],
                }
                rows.append(row)

    return pd.DataFrame(rows)


def create_recall_test_data(cfg: StudyA4Config) -> pd.DataFrame:
    """Create test data for recall scenario: unimodal gold, bimodal predictor.

    Following PROJECT.md specification:
    - Gold labels: Always unimodal around center 0
    - Predictor: Can be bimodal based on two-mode-probability
      - If bimodal: predict k//2 from gold mode + k//2 from opposite mode (center num_labels//2)
      - If unimodal: predict by hopping around the single gold mode

    This tests whether semantic F1 handles cases better than semantic recall
    alone when predictor includes correct mode plus additional mode labels.
    """
    labels = study_a1.make_labels(cfg.num_labels)
    S_ideal = study_a1.make_similarity_ring(cfg.num_labels)
    pos_center, neg_center = make_bimodal_structure(cfg.num_labels)

    # Force pos_center to be 0 per PROJECT.md specification
    pos_center = 0
    neg_center = cfg.num_labels // 2

    rows = []

    for k in cfg.k_values:
        for q in cfg.q_values:  # Changed from p to q
            for two_mode_freq in cfg.two_mode_frequencies:
                # Generate UNIMODAL gold labels around center 0
                gold_indices = []
                for _ in range(cfg.num_examples):
                    # Always use center 0 for gold (unimodal)
                    gold_idx = sample_around_center(
                        pos_center,  # Always center 0
                        k,
                        cfg.num_labels,
                        cfg.temperature,
                    )
                    gold_indices.append(gold_idx)

                gold_labels = study_a1.to_label_names(gold_indices, labels)

                # Create bimodal predictor with controlled frequency
                pred_indices = []
                for gold_idx in gold_indices:
                    if RNG.random() < two_mode_freq:
                        # Predict from BOTH modes: k//2 from gold mode + k//2 from opposite mode
                        k_per_mode = max(1, k // 2)  # At least 1 from each mode
                        k_remaining = k - 2 * k_per_mode

                        # Sample from gold mode (around center 0)
                        gold_mode_labels = sample_around_center(
                            pos_center,
                            k_per_mode,
                            cfg.num_labels,
                            temperature=cfg.pred_temperature,
                        )

                        # Sample from opposite mode (around center num_labels//2)
                        opposite_mode_labels = sample_around_center(
                            neg_center,
                            k_per_mode,
                            cfg.num_labels,
                            temperature=cfg.pred_temperature,
                        )

                        pred_idx = sorted(
                            list(set(gold_mode_labels + opposite_mode_labels))
                        )

                        # If k is odd, add one more label from a random mode
                        if k_remaining > 0:
                            extra_center = (
                                pos_center if RNG.random() < 0.5 else neg_center
                            )
                            extra_labels = sample_around_center(
                                extra_center,
                                k_remaining,
                                cfg.num_labels,
                                temperature=cfg.pred_temperature,
                            )
                            pred_idx = sorted(
                                list(set(pred_idx + extra_labels))
                            )
                    else:
                        # Predict unimodally by hopping around the gold mode
                        if RNG.random() < q:
                            # Apply hopping around center 0
                            pred_idx = sample_around_center(
                                pos_center,
                                k,
                                cfg.num_labels,
                                temperature=cfg.pred_temperature,
                            )
                        else:
                            # Keep original gold labels
                            pred_idx = gold_idx

                    pred_indices.append(pred_idx)

                pred_labels = study_a1.to_label_names(pred_indices, labels)

                # Get semantic F1 components using samples averaging
                from semantic_f1_score.semantic_f1 import (
                    samples_semantic_f1_score,
                )

                semantic_results = samples_semantic_f1_score(
                    pred_labels, gold_labels, S_ideal, return_components=True
                )

                # Store results - comparing semantic F1 vs semantic recall (and include precision)
                row = {
                    'k': k,
                    'q': q,  # Changed from p to q
                    'two_mode_freq': two_mode_freq,
                    'semantic_f1': semantic_results['f1'],
                    'semantic_precision': semantic_results['precision'],
                    'semantic_recall': semantic_results['recall'],
                    'gap_f1_minus_recall': semantic_results['f1']
                    - semantic_results['recall'],
                }
                rows.append(row)

    return pd.DataFrame(rows)


def create_hungarian_test_data(cfg: StudyA4Config) -> pd.DataFrame:
    """Create test data for Hungarian scenario: comparing semantic F1 vs Hungarian.

    Following PROJECT.md specification:
    - Gold labels: Can be bimodal based on two-mode-frequency
    - Predictor: Varies number of predictions (prediction_counts parameter)
    - Shows semantic F1's two-step approach vs Hungarian assignment matching

    This tests whether semantic F1's harmonic mean approach handles varying
    prediction counts better than pure assignment-based Hungarian matching.
    """
    labels = study_a1.make_labels(cfg.num_labels)
    S_ideal = study_a1.make_similarity_ring(cfg.num_labels)
    pos_center, neg_center = make_bimodal_structure(cfg.num_labels)

    prediction_counts = cfg.prediction_counts or cfg.k_values

    rows = []

    for k in cfg.k_values:
        for num_preds in prediction_counts:
            for two_mode_freq in cfg.two_mode_frequencies:
                # Generate bimodal gold labels based on two_mode_freq
                gold_indices = []
                for _ in range(cfg.num_examples):
                    gold_idx = sample_bimodal_labels(
                        cfg.num_labels,
                        k,
                        pos_center,
                        neg_center,
                        two_mode_freq,
                        cfg.temperature,
                    )
                    gold_indices.append(gold_idx)

                gold_labels = study_a1.to_label_names(gold_indices, labels)

                # Create predictor with varying prediction counts
                pred_indices = []
                for gold_idx in gold_indices:
                    # Determine gold's mode structure for informed prediction
                    gold_mode_weights = calculate_mode_weights(
                        gold_idx, pos_center, neg_center, cfg.num_labels
                    )

                    # Choose prediction strategy based on gold structure
                    if gold_mode_weights['pos'] > gold_mode_weights['neg']:
                        # Gold is more positive-leaning, predict around positive center
                        center = pos_center
                    elif gold_mode_weights['neg'] > gold_mode_weights['pos']:
                        # Gold is more negative-leaning, predict around negative center
                        center = neg_center
                    else:
                        # Balanced or no clear mode, pick randomly
                        center = (
                            pos_center if RNG.random() < 0.5 else neg_center
                        )

                    # Predict num_preds labels around the chosen center
                    pred_idx = sample_around_center(
                        center,
                        num_preds,
                        cfg.num_labels,
                        temperature=cfg.pred_temperature,
                    )
                    pred_indices.append(pred_idx)

                pred_labels = study_a1.to_label_names(pred_indices, labels)

                # Get semantic F1 using samples averaging
                semantic_f1 = semantic_f1_score(
                    gold_labels, pred_labels, S_ideal, average='samples'
                )

                # Extended Hungarian F1 - using samples-level averaging like semantic F1
                # Note: Hungarian implementation may print debug info to stdout
                import io
                import sys

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                try:
                    hungarian_score_val = hungarian_score(
                        pred_labels, gold_labels, S_ideal
                    )
                finally:
                    sys.stdout = old_stdout

                # Store results - comparing semantic F1 vs Hungarian
                row = {
                    'k': k,
                    'num_predictions': num_preds,
                    'two_mode_freq': two_mode_freq,
                    'semantic_f1': semantic_f1,
                    'hungarian_score': hungarian_score_val,
                    'gap_f1_minus_hungarian': semantic_f1 - hungarian_score_val,
                }
                rows.append(row)

    return pd.DataFrame(rows)


def run_scenario(cfg: StudyA4Config) -> None:
    """Run the specified scenario and save results."""
    os.makedirs(cfg.outdir, exist_ok=True)

    # Set up logging
    logger = logging.getLogger(f"study_a4_{cfg.scenario}")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(
            os.path.join(cfg.outdir, f"log_{cfg.scenario}.txt"), mode="w"
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(
        f"Starting Study A.4 {cfg.scenario} scenario with config: %s", cfg
    )

    # Validate configuration
    assert cfg.num_labels > 0, "num_labels must be positive"
    assert cfg.num_examples > 0, "num_examples must be positive"
    assert all(k > 0 for k in cfg.k_values), "All k values must be positive"
    assert all(
        0.0 <= q <= 1.0 for q in cfg.q_values
    ), "All q values must be in [0,1]"
    assert all(
        0.0 <= f <= 1.0 for f in cfg.two_mode_frequencies
    ), "All frequencies must be in [0,1]"

    # Generate data based on scenario
    if cfg.scenario == "precision":
        df_results = create_precision_test_data(cfg)
        scenario_suffix = "precision"
    elif cfg.scenario == "recall":
        df_results = create_recall_test_data(cfg)
        scenario_suffix = "recall"
    elif cfg.scenario == "hungarian":
        df_results = create_hungarian_test_data(cfg)
        scenario_suffix = "hungarian"
    else:
        raise ValueError(f"Unknown scenario: {cfg.scenario}")

    # Save results
    ts = int(time.time())
    output_path = os.path.join(
        cfg.outdir, f"study_a4_{scenario_suffix}_results_{ts}.csv"
    )
    df_results.to_csv(output_path, index=False)

    logger.info(f"Saved results to: {output_path}")
    logger.info(f"Results shape: {df_results.shape}")
    logger.info(f"Sample metrics from first few rows:")
    for i, row in df_results.head(3).iterrows():
        logger.info(f"  Row {i}: {dict(row)}")

    print(f"Saved results to: {output_path}")
    print(f"Results shape: {df_results.shape}")
    print(f"First few columns: {list(df_results.columns)}")

    # --------------------------
    # Generate plots and tables
    # --------------------------
    try:
        logger.info("Generating plots and tables...")

        # Generate scenario-specific plots
        if cfg.scenario == "precision":
            study_a4_plots.plot_precision_comparison(
                cfg, cfg.outdir, ts, logger
            )
            if not cfg.no_bootstrap:
                study_a4_plots.plot_confidence_intervals_detailed(
                    cfg, cfg.outdir, ts, logger
                )
            else:
                logger.info(
                    "Skipping expensive bootstrap CI computation (--no-bootstrap)"
                )
        elif cfg.scenario == "recall":
            study_a4_plots.plot_recall_comparison(cfg, cfg.outdir, ts, logger)
        elif cfg.scenario == "hungarian":
            study_a4_plots.plot_hungarian_comparison(
                cfg, cfg.outdir, ts, logger
            )

        # Generate gap analysis (works for all scenarios)
        study_a4_plots.plot_gap_analysis(cfg, cfg.outdir, ts, logger)

        # Generate summary table
        study_a4_plots.create_summary_table(cfg, cfg.outdir, ts, logger)

        logger.info("Successfully generated all plots and tables")

    except Exception as e:
        logger.warning(f"Error generating plots: {e}")
        print(f"Warning: Could not generate plots - {e}")


def test_basic_functionality():
    """Test basic functionality with minimal configuration."""
    print("Testing basic functionality...")

    # Test precision scenario with minimal config
    cfg = StudyA4Config(
        scenario="precision",
        num_labels=8,
        num_examples=100,
        k_values=[2],
        q_values=[0.0, 0.5],  # Changed from p_values to q_values
        two_mode_frequencies=[0.0, 0.5, 1.0],
        outdir="/tmp/test_study_a4",
        bootstrap=10,
        radius=2,
    )

    try:
        # Test data generation
        df = create_precision_test_data(cfg)
        assert len(df) > 0, "No data generated"
        expected_cols = [
            'k',
            'q',  # Changed from 'p' to 'q'
            'two_mode_freq',
            'semantic_f1',
            'semantic_precision',
            'semantic_recall',
            'gap_f1_minus_precision',
        ]
        assert all(
            col in df.columns for col in expected_cols
        ), f"Missing columns. Got: {list(df.columns)}"
        print(f"  ✓ Precision test: Generated {len(df)} rows")
        print(
            f"    Sample gap (F1 - Precision): {df['gap_f1_minus_precision'].mean():.4f}"
        )

        # Test recall scenario
        cfg.scenario = "recall"
        df = create_recall_test_data(cfg)
        assert len(df) > 0, "No data generated"
        expected_cols = [
            'k',
            'q',  # Changed from 'p' to 'q'
            'two_mode_freq',
            'semantic_f1',
            'semantic_precision',
            'semantic_recall',
            'gap_f1_minus_recall',
        ]
        assert all(
            col in df.columns for col in expected_cols
        ), f"Missing columns. Got: {list(df.columns)}"
        print(f"  ✓ Recall test: Generated {len(df)} rows")
        print(
            f"    Sample gap (F1 - Recall): {df['gap_f1_minus_recall'].mean():.4f}"
        )

        # Test Hungarian scenario
        cfg.scenario = "hungarian"
        cfg.prediction_counts = [1, 3]
        df = create_hungarian_test_data(cfg)
        assert len(df) > 0, "No data generated"
        expected_cols = [
            'k',
            'num_predictions',
            'two_mode_freq',
            'semantic_f1',
            'hungarian_score',
            'gap_f1_minus_hungarian',
        ]
        assert all(
            col in df.columns for col in expected_cols
        ), f"Missing columns. Got: {list(df.columns)}"
        print(f"  ✓ Hungarian test: Generated {len(df)} rows")
        print(
            f"    Sample gap (F1 - Hungarian): {df['gap_f1_minus_hungarian'].mean():.4f}"
        )

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


def parse_args() -> StudyA4Config:
    """Parse command line arguments using subparsers for different scenarios."""
    parser = argparse.ArgumentParser(
        description="Study A.4 - Comparison to baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers for different scenarios
    subparsers = parser.add_subparsers(dest='scenario', help='Test scenario')
    subparsers.required = True

    # Test command
    test_parser = subparsers.add_parser(
        'test', help='Run basic functionality tests'
    )

    # Common arguments function
    def add_common_args(subparser):
        subparser.add_argument(
            "--num-labels",
            type=int,
            default=24,
            help="Number of labels arranged on the circle",
        )
        subparser.add_argument(
            "--num-examples",
            type=int,
            default=10000,
            help="Number of synthetic examples per configuration",
        )
        subparser.add_argument(
            "--k",
            type=int,
            nargs="+",
            default=[1, 2, 3],
            help="Numbers of labels per example to sample for gold",
        )
        subparser.add_argument(
            "--q",
            type=float,
            nargs="+",
            default=[0.0, 0.25, 0.5, 0.75, 1.0],
            help="Probabilities to perturb each gold label (renamed from p for Study A.4)",
        )
        subparser.add_argument(
            "--two-mode-frequencies",
            type=float,
            nargs="+",
            default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            help="Frequencies of two-mode examples (0.0=always unimodal, 1.0=always bimodal)",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=123,
            help="Random seed for reproducibility",
        )
        subparser.add_argument(
            "--bootstrap",
            type=int,
            default=25,
            help="Number of bootstrap iterations for confidence intervals",
        )
        subparser.add_argument(
            "--no-bootstrap",
            action="store_true",
            help="Skip expensive bootstrap confidence interval computation",
        )
        subparser.add_argument(
            "--radius",
            type=int,
            default=2,
            help="Radius around center for label sampling and perturbation",
        )
        subparser.add_argument(
            "--temperature",
            type=float,
            default=2.0,
            help="Temperature parameter for label sampling",
        )
        subparser.add_argument(
            "--pred-temperature",
            type=float,
            default=2.0,
            help="Temperature parameter for prediction sampling",
        )
        subparser.add_argument(
            "--outdir",
            type=str,
            default="logs/semantic_f1/study_a4",
            help="Output directory for results",
        )

    # Precision scenario: bimodal gold, unimodal predictor
    precision_parser = subparsers.add_parser(
        'precision',
        help='Test precision scenario: bimodal gold labels, unimodal predictor',
    )
    add_common_args(precision_parser)

    # Recall scenario: unimodal gold, bimodal predictor
    recall_parser = subparsers.add_parser(
        'recall',
        help='Test recall scenario: unimodal gold labels, bimodal predictor',
    )
    add_common_args(recall_parser)

    # Hungarian scenario: varying prediction counts
    hungarian_parser = subparsers.add_parser(
        'hungarian',
        help='Test Hungarian scenario: varying prediction counts vs extended Hungarian',
    )
    add_common_args(hungarian_parser)
    hungarian_parser.add_argument(
        "--prediction-counts",
        type=int,
        nargs="+",
        default=None,
        help="Numbers of predictions to make (defaults to k values)",
    )

    args = parser.parse_args()

    if args.scenario == 'test':
        return None  # Special case for testing

    return StudyA4Config(
        scenario=args.scenario,
        num_labels=args.num_labels,
        num_examples=args.num_examples,
        k_values=args.k,
        q_values=args.q,  # Changed from p_values to q_values
        two_mode_frequencies=args.two_mode_frequencies,
        prediction_counts=getattr(args, 'prediction_counts', None),
        seed=args.seed,
        bootstrap=args.bootstrap,
        no_bootstrap=args.no_bootstrap,
        radius=args.radius,
        temperature=args.temperature,
        pred_temperature=args.pred_temperature,
        outdir=args.outdir,
    )


def main():
    """Entry point: parse args and run the specified scenario."""
    cfg = parse_args()

    if cfg is None:
        # Run tests
        test_basic_functionality()
    else:
        run_scenario(cfg)


if __name__ == "__main__":
    main()
