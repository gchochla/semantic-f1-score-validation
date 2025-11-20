#!/usr/bin/env python3
"""
Study C: Output Score Thresholding Smoothness Analysis (Final Version)

This version directly uses predefined semantic similarity matrices and bypasses dataset loading issues.

To run the complete Study C analysis for all three datasets (SemEval, GoEmotions, MFRC):

```bash
python scripts/semantic_f1/study_c_final.py \
    --datasets SemEval GoEmotions MFRC --logs_dir logs_emnlp/DEMUX \
    --output_dir logs/analysis/semantic_f1/study_c_demux \
    --thres 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95

python scripts/semantic_f1/study_c_final.py \
    --datasets SemEval GoEmotions MFRC --logs_dir logs/LoRA-best/ \
    --output_dir logs/analysis/semantic_f1/study_c_llama \
    --thres 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
```
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import f1_score
from semantic_f1_score import semantic_f1_score
from scipy.stats import kendalltau
from legm import splitify_namespace

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm_subj import DATASETS, text_preprocessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR

warnings.filterwarnings('ignore')


def build_dataset_for_task(task: str, exp_params: Dict) -> Any:
    """Build dataset instance for the given task and parameters using splitify_namespace."""

    try:
        DatasetCls = DATASETS[task]
    except KeyError:
        raise ValueError(
            f"Unknown task '{task}': not found in llm_subj.DATASETS"
        )

    # Convert dict to namespace
    args_namespace = argparse.Namespace(**exp_params)

    # Handle text_preprocessor parameter like in the prompting scripts
    if (
        hasattr(args_namespace, 'text_preprocessor')
        and args_namespace.text_preprocessor
    ):
        args_namespace.text_preprocessor = text_preprocessor[
            DatasetCls.source_domain
        ]()
    else:
        args_namespace.text_preprocessor = None

    # Use splitify_namespace to properly handle parameters - check for test split first
    splits = exp_params.get("test_splits", ["test"])
    if isinstance(splits, str):
        splits = [splits]

    # Use the first test split for simplicity
    split_name = splits[0] if splits else "test"

    print(f"  Building dataset for {task} with split '{split_name}'")
    print(f"  Available exp_params keys: {list(exp_params.keys())}")

    # Use splitify_namespace which handles filtering parameters properly
    dataset = DatasetCls(
        init__namespace=splitify_namespace(args_namespace, split_name)
    )
    return dataset


def get_similarity_matrix_and_labels(
    task: str, exp_params: Dict, num_labels: int
) -> Tuple[List[str], Any]:
    """Get similarity matrix and label set from dataset."""

    try:
        dataset = build_dataset_for_task(task, exp_params)

        # Get label set
        if hasattr(dataset, 'label_set'):
            label_set = list(dataset.label_set)
        else:
            # Fallback to generic labels
            label_set = [f"label_{i}" for i in range(num_labels)]

        # Get semantic similarity matrix
        semantic_matrix = None
        if (
            hasattr(dataset, 'label_similarity')
            and dataset.label_similarity is not None
        ):
            semantic_matrix = dataset.label_similarity
        elif (
            hasattr(dataset.__class__, 'label_similarity')
            and dataset.__class__.label_similarity is not None
        ):
            semantic_matrix = dataset.__class__.label_similarity

        print(f"  Task: {task}, Labels: {len(label_set)}")
        print(f"  Label set: {label_set}")
        if semantic_matrix is not None:
            print(
                f"  Semantic matrix shape: {semantic_matrix.shape if hasattr(semantic_matrix, 'shape') else 'no shape'}"
            )
            print(
                f"  Semantic matrix index: {list(semantic_matrix.index) if hasattr(semantic_matrix, 'index') else 'no index'}"
            )
            print(f"  Semantic matrix available: True")
        else:
            print(f"  Semantic matrix available: False")

        return label_set, semantic_matrix

    except Exception as e:
        print(f"  Warning: Could not build dataset for {task}: {e}")
        import traceback

        print(f"  Traceback: {traceback.format_exc()}")

        # Fallback to generic labels and no similarity matrix
        label_set = [f"label_{i}" for i in range(num_labels)]
        semantic_matrix = None
        return label_set, semantic_matrix


def load_experiment_data_efficient(
    exp_dir: str, max_examples: int = None
) -> Tuple[Dict, List[List[float]], List[List[int]]]:
    """Load experiment data efficiently."""

    print(f"  Loading experiment data from {os.path.basename(exp_dir)}...")

    # Load parameters
    params_path = os.path.join(exp_dir, "params.yml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    exp_keys = sorted(
        k for k in params.keys() if str(k).startswith("experiment_")
    )
    exp_params = params[exp_keys[0]]

    # Load indexed metrics efficiently
    indexed_path = os.path.join(exp_dir, "indexed_metrics.yml")
    print(f"  Loading indexed metrics...")

    with open(indexed_path, "r") as f:
        indexed_data = yaml.safe_load(f)

    first_exp_key = sorted(
        k for k in indexed_data.keys() if k.startswith("experiment_")
    )[0]
    exp_data = indexed_data[first_exp_key]

    test_scores = []
    ground_truth = []
    count = 0

    for example_key, example_data in exp_data.items():
        if example_key == "description":
            continue
        if max_examples and count >= max_examples:
            break

        scores = example_data.get("test_scores", [])
        gt = example_data.get("test_gt", [])

        if scores and gt:
            test_scores.append(scores)
            # Convert to integer binary format
            ground_truth.append([int(float(x)) for x in gt])
            count += 1

    print(
        f"  Loaded {len(test_scores)} examples with {len(test_scores[0]) if test_scores else 0} labels each"
    )

    # Print some statistics
    if test_scores:
        scores_array = np.array(test_scores)
        print(
            f"  Score range: [{scores_array.min():.3f}, {scores_array.max():.3f}]"
        )
        print(f"  Score mean: {scores_array.mean():.3f}")

        gt_array = np.array(ground_truth)
        print(
            f"  Ground truth labels per example: {gt_array.sum(axis=1).mean():.2f} ± {gt_array.sum(axis=1).std():.2f}"
        )

    return exp_params, test_scores, ground_truth


def apply_threshold_robust(
    scores: List[List[float]], threshold: float
) -> List[List[int]]:
    """Apply threshold with robust handling."""
    predictions = []
    for score_list in scores:
        pred = [1 if score >= threshold else 0 for score in score_list]
        predictions.append(pred)
    return predictions


def compute_metrics_robust(
    y_true: List[List[int]],
    y_pred: List[List[int]],
    y_true_labels: List[List[str]],
    y_pred_labels: List[List[str]],
    semantic_matrix: pd.DataFrame = None,
) -> Dict[str, float]:
    """Compute both Hard and Semantic F1 metrics with robust error handling."""

    # Convert to numpy arrays for hard metrics
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    # Check for valid data
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_array.shape} vs y_pred {y_pred_array.shape}"
        )

    # Compute Hard F1 metrics
    metrics = {}
    try:
        metrics["hard_micro_f1"] = float(
            f1_score(
                y_true_array, y_pred_array, average="micro", zero_division=0
            )
        )
        metrics["hard_macro_f1"] = float(
            f1_score(
                y_true_array, y_pred_array, average="macro", zero_division=0
            )
        )
        metrics["hard_sample_f1"] = float(
            f1_score(
                y_true_array, y_pred_array, average="samples", zero_division=0
            )
        )
    except Exception as e:
        print(f"    Warning: Error computing Hard F1 metrics: {e}")
        metrics.update(
            {"hard_micro_f1": 0.0, "hard_macro_f1": 0.0, "hard_sample_f1": 0.0}
        )

    # Compute Semantic F1 metrics if similarity matrix is available
    if semantic_matrix is not None:
        try:
            metrics["semantic_micro_f1"] = float(
                semantic_f1_score(
                    y_true_labels,
                    y_pred_labels,
                    semantic_matrix,
                    average="micro",
                )
            )
            metrics["semantic_macro_f1"] = float(
                semantic_f1_score(
                    y_true_labels,
                    y_pred_labels,
                    semantic_matrix,
                    average="macro",
                )
            )
            metrics["semantic_sample_f1"] = float(
                semantic_f1_score(
                    y_true_labels,
                    y_pred_labels,
                    semantic_matrix,
                    average="samples",
                )
            )
        except Exception as e:
            print(f"    Warning: Error computing Semantic F1 metrics: {e}")
            # Fall back to hard metrics if semantic computation fails
            metrics.update(
                {
                    "semantic_micro_f1": metrics.get("hard_micro_f1", 0.0),
                    "semantic_macro_f1": metrics.get("hard_macro_f1", 0.0),
                    "semantic_sample_f1": metrics.get("hard_sample_f1", 0.0),
                }
            )
    else:
        # No semantic matrix available, use hard metrics as fallback
        metrics.update(
            {
                "semantic_micro_f1": metrics.get("hard_micro_f1", 0.0),
                "semantic_macro_f1": metrics.get("hard_macro_f1", 0.0),
                "semantic_sample_f1": metrics.get("hard_sample_f1", 0.0),
            }
        )

    return metrics


def analyze_experiment_enhanced(
    exp_dir: str, thresholds: List[float], max_examples: int = None
) -> Dict[str, Any]:
    """Enhanced experiment analysis with both Hard and Semantic F1 metrics."""

    print(f"\nAnalyzing experiment: {os.path.basename(exp_dir)}")

    # Load data
    try:
        exp_params, test_scores, ground_truth = load_experiment_data_efficient(
            exp_dir, max_examples
        )
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    if not test_scores:
        print(f"  No test scores found")
        return None

    # Get label set and similarity matrix for the task
    task = exp_params.get("task")
    if not task:
        print(f"  Warning: No task found in params")
        return None

    # Load label set and similarity matrix from the dataset
    num_model_labels = len(test_scores[0]) if test_scores else 0
    label_set, semantic_matrix = get_similarity_matrix_and_labels(
        task, exp_params, num_model_labels
    )

    # Ensure label set matches the number of model outputs
    if len(label_set) != num_model_labels:
        print(
            f"  Warning: Label set length ({len(label_set)}) doesn't match model outputs ({num_model_labels})"
        )
        if len(label_set) > num_model_labels:
            label_set = label_set[:num_model_labels]
            print(f"  Truncated label set to: {label_set}")
        else:
            # Pad with generic labels if needed
            while len(label_set) < num_model_labels:
                label_set.append(f"label_{len(label_set)}")
            print(f"  Padded label set to: {label_set}")

    # Filter semantic matrix to match the label set if needed
    if semantic_matrix is not None and hasattr(semantic_matrix, 'loc'):
        try:
            # Filter to only the labels we're using
            available_labels = [
                l for l in label_set if l in semantic_matrix.index
            ]
            if len(available_labels) > 0:
                semantic_matrix = semantic_matrix.loc[
                    available_labels, available_labels
                ]
                # Update label set to match filtered matrix order
                label_set = available_labels
                print(
                    f"  Filtered semantic matrix to {len(available_labels)} labels"
                )
            else:
                print(
                    f"  Warning: No labels from model output found in semantic matrix"
                )
                semantic_matrix = None
        except Exception as e:
            print(f"  Warning: Could not filter semantic matrix: {e}")
            semantic_matrix = None

        print(f"  Task: {task}, Labels: {len(label_set)}")
        print(f"  Label set: {label_set}")
        print(f"  Semantic matrix type: {type(semantic_matrix)}")
        if semantic_matrix is not None:
            print(
                f"  Semantic matrix shape: {semantic_matrix.shape if hasattr(semantic_matrix, 'shape') else 'no shape'}"
            )
            print(f"  Semantic matrix available: True")
        else:
            print(f"  Semantic matrix available: False")
    else:
        print(f"  Warning: Unknown task {task}, using generic labels")
        label_set = [f"label_{i}" for i in range(len(test_scores[0]))]
        semantic_matrix = None

    # Convert ground truth to label format
    ground_truth_labels = []
    for gt_binary in ground_truth:
        gt_labels = [
            label_set[i] for i, val in enumerate(gt_binary) if val == 1
        ]
        ground_truth_labels.append(gt_labels)

    # Analyze across thresholds
    results = {
        "task": task,
        "model": exp_params.get("model_name_or_path", "unknown"),
        "exp_dir": exp_dir,
        "thresholds": thresholds,
        "num_examples": len(test_scores),
        "num_labels": len(test_scores[0]),
        "metrics": {
            "hard_micro_f1": [],
            "hard_macro_f1": [],
            "hard_sample_f1": [],
            "semantic_micro_f1": [],
            "semantic_macro_f1": [],
            "semantic_sample_f1": [],
        },
    }

    print(f"  Computing metrics across {len(thresholds)} thresholds...")

    # Track prediction statistics for debugging
    pred_stats = []

    for i, threshold in enumerate(thresholds):
        print(f"    Threshold {threshold:.1f}...", end=" ")

        # Apply threshold to get binary predictions
        predictions_binary = apply_threshold_robust(test_scores, threshold)

        # Convert to label format for semantic F1
        predictions_labels = []
        for pred_binary in predictions_binary:
            pred_labels = [
                label_set[i] for i, val in enumerate(pred_binary) if val == 1
            ]
            predictions_labels.append(pred_labels)

        # Compute both hard and semantic metrics
        metrics = compute_metrics_robust(
            ground_truth,
            predictions_binary,
            ground_truth_labels,
            predictions_labels,
            semantic_matrix,
        )

        # Store results
        for metric_name, value in metrics.items():
            if metric_name in results["metrics"]:
                results["metrics"][metric_name].append(value)

        # Track prediction statistics
        pred_array = np.array(predictions_binary)
        total_preds = pred_array.sum()
        total_labels = pred_array.size
        pred_rate = total_preds / total_labels if total_labels > 0 else 0
        pred_stats.append(pred_rate)

        print(
            f"Hard: {metrics.get('hard_micro_f1', 0):.3f}, Semantic: {metrics.get('semantic_micro_f1', 0):.3f}, Pred rate: {pred_rate:.3f}"
        )

    # Compute smoothness and monotonicity metrics
    results["smoothness"] = {}
    results["smoothness_relative"] = {}
    results["smoothness_range_normalized"] = {}
    results["smoothness_mean_normalized"] = {}
    results["monotonicity"] = {}
    results["pred_stats"] = pred_stats

    for metric_name, values in results["metrics"].items():
        if len(values) > 1:
            # Smoothness: average absolute change between consecutive points
            step_changes = [
                abs(values[i + 1] - values[i]) for i in range(len(values) - 1)
            ]
            smoothness_index = np.mean(step_changes)

            # Relative smoothness: coefficient of variation approach
            mean_value = np.mean(values)
            relative_smoothness = (
                np.std(step_changes) / mean_value if mean_value > 0 else 0.0
            )

            # Range-normalized smoothness
            value_range = max(values) - min(values)
            range_normalized_smoothness = (
                smoothness_index / value_range if value_range > 0 else 0.0
            )

            # Mean-normalized smoothness
            mean_normalized_smoothness = (
                smoothness_index / mean_value if mean_value > 0 else 0.0
            )

            # Monotonicity: Kendall's tau correlation with threshold order
            # Higher threshold should generally lead to lower F1 (negative correlation expected)
            thresholds_ordered = sorted(results["thresholds"])
            tau, p_value = kendalltau(thresholds_ordered, values)
            monotonicity_index = (
                -tau
            )  # We want negative tau (F1 should decrease with threshold)
        else:
            smoothness_index = 0.0
            relative_smoothness = 0.0
            range_normalized_smoothness = 0.0
            mean_normalized_smoothness = 0.0
            monotonicity_index = 0.0

        results["smoothness"][metric_name] = smoothness_index
        results["smoothness_relative"][metric_name] = relative_smoothness
        results["smoothness_range_normalized"][
            metric_name
        ] = range_normalized_smoothness
        results["smoothness_mean_normalized"][
            metric_name
        ] = mean_normalized_smoothness
        results["monotonicity"][metric_name] = monotonicity_index
        print(f"    {metric_name} smoothness index: {smoothness_index:.6f}")
        print(
            f"    {metric_name} relative smoothness: {relative_smoothness:.6f}"
        )
        print(
            f"    {metric_name} range-normalized smoothness: {range_normalized_smoothness:.6f}"
        )
        print(
            f"    {metric_name} mean-normalized smoothness: {mean_normalized_smoothness:.6f}"
        )
        print(f"    {metric_name} monotonicity index: {monotonicity_index:.6f}")

    return results


def create_enhanced_plots(results_list: List[Dict], output_dir: str) -> None:
    """Create enhanced plots with both Hard and Semantic F1 visualization."""

    os.makedirs(output_dir, exist_ok=True)

    # Group by dataset
    datasets = {}
    for result in results_list:
        if result is None:
            continue
        task = result["task"]
        if task not in datasets:
            datasets[task] = []
        datasets[task].append(result)

    # Create comprehensive plot for each dataset
    for task, task_results in datasets.items():

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        fig.suptitle(
            f'Threshold Analysis on {task}',
            fontsize=24,
            fontweight='bold',
        )

        # Plot F1 scores - separate hard and semantic
        averaging_methods = ['micro_f1', 'macro_f1', 'sample_f1']

        for idx, avg_method in enumerate(averaging_methods):
            ax = axes[idx // 2, idx % 2] if idx < 2 else axes[1, 0]

            for result in task_results:
                model_name = result["model"].split("/")[-1]
                thresholds = result["thresholds"]

                # Plot both hard and semantic F1
                hard_values = result["metrics"][f"hard_{avg_method}"]
                semantic_values = result["metrics"][f"semantic_{avg_method}"]

                ax.plot(
                    thresholds,
                    hard_values,
                    'o--',
                    label='Hard',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                    color=HARD_COLOR,
                )
                ax.plot(
                    thresholds,
                    semantic_values,
                    's-',
                    label='Semantic',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                    color=SEM_COLOR,
                )

            if idx // 2 == 1:  # Only label x-axis on bottom row
                ax.set_xlabel('Threshold', fontsize=16)
            # ax.set_ylabel(f'{avg_method.replace("_", " ").title()}')
            ax.set_title(f'{avg_method.replace("_", " ").title()}', fontsize=22)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=22)

        # Plot prediction rate
        ax = axes[1, 1]
        for result in task_results:
            model_name = result["model"].split("/")[-1]
            thresholds = result["thresholds"]
            pred_rates = result["pred_stats"]

            ax.plot(
                thresholds,
                pred_rates,
                '^-',
                label='Pred Rate',
                linewidth=2,
                markersize=6,
                alpha=0.8,
                color=THIRD_COLOR,
            )

        ax.set_xlabel('Threshold', fontsize=16)
        # ax.set_ylabel('Prediction Rate', fontsize=16)
        ax.set_title('Label Prediction Rate vs Threshold', fontsize=22)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=24)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'study_c_{task.lower()}_final.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.savefig(
            os.path.join(output_dir, f'study_c_{task.lower()}_final.pdf'),
            bbox_inches='tight',
        )
        plt.close()

    # Create smoothness and monotonicity comparison for both hard and semantic metrics
    smoothness_data = []
    smoothness_relative_data = []
    smoothness_range_norm_data = []
    smoothness_mean_norm_data = []
    monotonicity_data = []

    for result in results_list:
        if result is None:
            continue

        task = result["task"]
        model = result["model"].split("/")[-1]

        # Collect all smoothness variants
        for metric_name in result["smoothness"].keys():
            # Parse metric name to get type and averaging method
            if metric_name.startswith("hard_"):
                metric_type = "Hard F1"
                avg_method = metric_name[5:].replace("_", " ").title()
            elif metric_name.startswith("semantic_"):
                metric_type = "Semantic F1"
                avg_method = metric_name[9:].replace("_", " ").title()
            else:
                continue

            # Standard smoothness
            smoothness_data.append(
                {
                    "Task": task,
                    "Model": model,
                    "Metric Type": metric_type,
                    "Averaging": avg_method,
                    "Smoothness Index": result["smoothness"][metric_name],
                }
            )

            # Relative smoothness
            smoothness_relative_data.append(
                {
                    "Task": task,
                    "Model": model,
                    "Metric Type": metric_type,
                    "Averaging": avg_method,
                    "Relative Smoothness": result["smoothness_relative"][
                        metric_name
                    ],
                }
            )

            # Range-normalized smoothness
            smoothness_range_norm_data.append(
                {
                    "Task": task,
                    "Model": model,
                    "Metric Type": metric_type,
                    "Averaging": avg_method,
                    "Range-Normalized Smoothness": result[
                        "smoothness_range_normalized"
                    ][metric_name],
                }
            )

            # Mean-normalized smoothness
            smoothness_mean_norm_data.append(
                {
                    "Task": task,
                    "Model": model,
                    "Metric Type": metric_type,
                    "Averaging": avg_method,
                    "Mean-Normalized Smoothness": result[
                        "smoothness_mean_normalized"
                    ][metric_name],
                }
            )

            # Monotonicity
            monotonicity_data.append(
                {
                    "Task": task,
                    "Model": model,
                    "Metric Type": metric_type,
                    "Averaging": avg_method,
                    "Monotonicity Index": result["monotonicity"][metric_name],
                }
            )

    # Define smoothness variants for plotting
    smoothness_variants = [
        (
            smoothness_data,
            "Smoothness Index",
            "study_c_smoothness_absolute.pdf",
            "Absolute Smoothness",
        ),
        (
            smoothness_relative_data,
            "Relative Smoothness",
            "study_c_smoothness_relative.pdf",
            "Relative Smoothness",
        ),
        (
            smoothness_range_norm_data,
            "Range-Normalized Smoothness",
            "study_c_smoothness_range_normalized.pdf",
            "Range-Normalized Smoothness",
        ),
        (
            smoothness_mean_norm_data,
            "Mean-Normalized Smoothness",
            "study_c_smoothness_mean_normalized.pdf",
            "Mean-Normalized Smoothness",
        ),
    ]

    averaging_methods = ['Micro F1', 'Macro F1', 'Sample F1']

    # Create plots for all smoothness variants
    for data_list, y_label, filename, title_suffix in smoothness_variants:
        if data_list:
            df_smooth = pd.DataFrame(data_list)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            fig.suptitle(title_suffix, fontsize=28, fontweight='bold')

            for idx, avg_method in enumerate(averaging_methods):
                ax = axes[idx]
                avg_data = df_smooth[df_smooth['Averaging'] == avg_method]

                if not avg_data.empty:
                    palette = {
                        'Hard F1': HARD_COLOR,
                        'Semantic F1': SEM_COLOR,
                    }
                    sns.barplot(
                        data=avg_data,
                        x='Task',
                        y=y_label,
                        hue='Metric Type',
                        ax=ax,
                        alpha=0.8,
                        palette=palette,
                    )

                    ax.set_title(f'{avg_method}', fontsize=26)
                    ax.set_xlabel('')
                    ax.tick_params(axis='x', labelsize=24, rotation=20)
                    ax.set_title(f'{avg_method}', fontsize=26)

                    # Remove legend from all but the last subplot
                    if idx < len(averaging_methods) - 1:
                        ax.get_legend().remove()
                    else:
                        ax.legend(fontsize=26)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

        # Create monotonicity plots
        if monotonicity_data:
            df_mono = pd.DataFrame(monotonicity_data)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            fig.suptitle(
                'Negative Kendall Tau',
                fontsize=28,
                fontweight='bold',
            )

            for idx, avg_method in enumerate(averaging_methods):
                ax = axes[idx]
                avg_data = df_mono[df_mono['Averaging'] == avg_method]

                if not avg_data.empty:
                    sns.barplot(
                        data=avg_data,
                        x='Task',
                        y='Monotonicity Index',
                        hue='Metric Type',
                        ax=ax,
                        alpha=0.8,
                    )

                    ax.set_title(f'{avg_method}', fontsize=26)
                    ax.set_xlabel('')
                    if idx == 0:  # Only label y-axis on first subplot
                        ax.set_ylabel('Monotonicity Index')
                    else:
                        ax.set_ylabel('')
                    ax.tick_params(axis='x', rotation=20, labelsize=24)

                    # Remove legend from all but the last subplot
                    if idx < len(averaging_methods) - 1:
                        ax.get_legend().remove()
                    else:
                        ax.legend(fontsize=26)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, 'study_c_monotonicity_final.pdf'),
                bbox_inches='tight',
            )
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Study C: Final Threshold Analysis"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["SemEval"], help="Datasets to analyze"
    )
    parser.add_argument(
        "--logs_dir",
        default="logs_emnlp/DEMUX",
        help="Directory containing experiment logs",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/analysis/semantic_f1/study_c",
        help="Output directory",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds to test",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples for testing",
    )

    args = parser.parse_args()

    # Convert paths to absolute paths
    logs_dir = os.path.abspath(args.logs_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Study C: Final Threshold Analysis")
    print(f"Datasets: {args.datasets}")
    print(f"Logs directory: {logs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Thresholds: {args.thresholds}")
    if args.max_examples:
        print(f"Max examples per dataset: {args.max_examples}")
    print()

    # Find best experiment for each dataset
    results = []

    for dataset in args.datasets:
        try:
            # Find best experiment directory
            dataset_dir = os.path.join(logs_dir, dataset)
            if not os.path.exists(dataset_dir):
                print(f"Dataset directory not found: {dataset_dir}")
                continue

            best_exp_dir = None
            best_jaccard = -1.0

            # Look for normal experiments
            for item in os.listdir(dataset_dir):
                if item.startswith("normal-") and os.path.isdir(
                    os.path.join(dataset_dir, item)
                ):
                    exp_path = os.path.join(dataset_dir, item)
                    agg_metrics_path = os.path.join(
                        exp_path, "aggregated_metrics.yml"
                    )

                    if os.path.exists(agg_metrics_path):
                        try:
                            with open(agg_metrics_path, "r") as f:
                                agg_data = yaml.safe_load(f)

                            # Extract Jaccard score
                            for key, metrics in agg_data.items():
                                if (
                                    isinstance(metrics, dict)
                                    and "best_dev_jaccard_score" in metrics
                                ):
                                    jaccard_str = metrics[
                                        "best_dev_jaccard_score"
                                    ]
                                    jaccard_score = float(
                                        jaccard_str.split("+-")[0]
                                    )

                                    if jaccard_score > best_jaccard:
                                        best_jaccard = jaccard_score
                                        best_exp_dir = exp_path
                                    break
                        except Exception as e:
                            print(
                                f"Warning: Could not parse {agg_metrics_path}: {e}"
                            )

            if best_exp_dir:
                print(
                    f"Selected {dataset}: {os.path.basename(best_exp_dir)} (Jaccard: {best_jaccard:.4f})"
                )
                result = analyze_experiment_enhanced(
                    best_exp_dir, args.thresholds, args.max_examples
                )
                if result:
                    results.append(result)
            else:
                print(f"No valid experiments found for {dataset}")

        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    if not results:
        print("No successful analyses!")
        return

    # Create plots and save results
    create_enhanced_plots(results, output_dir)

    # Save detailed summary
    summary_rows = []
    for result in results:
        row = {
            "Task": result["task"],
            "Model": result["model"].split("/")[-1],
            "Exp_Dir": os.path.basename(result["exp_dir"]),
            "Num_Examples": result["num_examples"],
            "Num_Labels": result["num_labels"],
        }

        # Add all smoothness variants for both hard and semantic metrics
        for metric_name, smoothness_value in result["smoothness"].items():
            row[f"SI_abs_{metric_name}"] = f"{smoothness_value:.6f}"

        for metric_name, relative_smoothness in result[
            "smoothness_relative"
        ].items():
            row[f"SI_rel_{metric_name}"] = f"{relative_smoothness:.6f}"

        for metric_name, range_norm_smoothness in result[
            "smoothness_range_normalized"
        ].items():
            row[f"SI_range_{metric_name}"] = f"{range_norm_smoothness:.6f}"

        for metric_name, mean_norm_smoothness in result[
            "smoothness_mean_normalized"
        ].items():
            row[f"SI_mean_{metric_name}"] = f"{mean_norm_smoothness:.6f}"

        # Add monotonicity indices for both hard and semantic metrics
        for metric_name, monotonicity_value in result["monotonicity"].items():
            row[f"MI_{metric_name}"] = f"{monotonicity_value:.6f}"

        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'study_c_final_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\nDetailed results saved to: {summary_path}")

    # Create explanation file
    explanation_path = os.path.join(output_dir, 'METRICS_EXPLANATION.md')
    with open(explanation_path, 'w') as f:
        f.write(
            """# Study C Metrics Explanation

## Smoothness Index Variants

### 1. Absolute Smoothness Index (SI_abs)
- **What it measures**: Average absolute change between consecutive F1 scores across thresholds
- **Formula**: `mean(|F1[i+1] - F1[i]|)`
- **Interpretation**: 
  - **Lower is Better** - indicates more stable performance across thresholds
  - **Scale-dependent**: Higher F1 values will tend to have larger absolute changes
  - **Use case**: When comparing models on the same metric/scale

### 2. Relative Smoothness Index (SI_rel)
- **What it measures**: Standard deviation of changes normalized by mean F1 value
- **Formula**: `std(changes) / mean(F1_values)`
- **Interpretation**:
  - **Lower is Better** - indicates relative stability regardless of scale
  - **Scale-independent**: Accounts for different F1 ranges between metrics
  - **Use case**: Best for comparing Hard vs Semantic F1 or different datasets

### 3. Range-Normalized Smoothness (SI_range)
- **What it measures**: Absolute smoothness divided by the range of F1 values
- **Formula**: `mean(|changes|) / (max(F1) - min(F1))`
- **Interpretation**:
  - **Lower is Better** - smoothness relative to the total variation
  - **Range-normalized**: Accounts for how much the metric varies overall
  - **Use case**: When total variation differs significantly between metrics

### 4. Mean-Normalized Smoothness (SI_mean)
- **What it measures**: Absolute smoothness divided by mean F1 value
- **Formula**: `mean(|changes|) / mean(F1_values)`
- **Interpretation**:
  - **Lower is Better** - smoothness as a percentage of typical F1 values
  - **Mean-normalized**: Accounts for different baseline performance levels
  - **Use case**: When comparing metrics with different typical values

## Monotonicity Index (MI)  
- **What it measures**: Absolute value of Kendall's τ correlation between thresholds and F1 scores
- **Interpretation**:
  - **Higher is Better** - indicates more predictable threshold behavior
  - Higher values mean F1 consistently decreases (or increases) with threshold changes
  - Perfect monotonicity (MI = 1.0) means F1 always changes in the same direction
- **Range**: [0, 1], where 1 = perfectly monotonic relationship

## Key Insights
- **Scale Sensitivity**: Absolute smoothness is biased toward metrics with larger values
- **Normalization Benefits**: Relative and normalized smoothness allow fair comparison between different metrics
- **Best Practice**: Use relative smoothness (SI_rel) for comparing Hard vs Semantic F1
- **Semantic Expectation**: Semantic F1 should be smoother due to label similarity considerations

## CSV Column Prefixes
- `SI_abs_`: Absolute Smoothness Index
- `SI_rel_`: Relative Smoothness Index  
- `SI_range_`: Range-Normalized Smoothness Index
- `SI_mean_`: Mean-Normalized Smoothness Index
- `MI_`: Monotonicity Index
- `hard_`: Traditional F1 calculation
- `semantic_`: F1 with label similarity weighting

## Generated Plots
- `study_c_smoothness_absolute.png`: Standard absolute smoothness
- `study_c_smoothness_relative.png`: Scale-independent relative smoothness ⭐ **Recommended**
- `study_c_smoothness_range_normalized.png`: Range-normalized smoothness
- `study_c_smoothness_mean_normalized.png`: Mean-normalized smoothness
- `study_c_monotonicity_final.png`: Monotonicity comparison
"""
        )
    print(f"Metrics explanation saved to: {explanation_path}")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("METRICS SUMMARY:")
    print("• Absolute Smoothness (SI_abs): Lower = Better (scale-dependent)")
    print(
        "• Relative Smoothness (SI_rel): Lower = Better (scale-independent) ⭐ BEST"
    )
    print("• Range-Normalized (SI_range): Lower = Better (range-adjusted)")
    print("• Mean-Normalized (SI_mean): Lower = Better (mean-adjusted)")
    print(
        "• Monotonicity Index (MI): Higher = Better (more negative = more monotonic)"
    )
    print("• Generated 5 plots: 4 smoothness variants + 1 monotonicity")
    print("=" * 70)


if __name__ == "__main__":
    main()
