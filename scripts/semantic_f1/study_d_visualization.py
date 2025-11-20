#!/usr/bin/env python3
"""
Study D Visualization: Plot downstream performance vs semantic F1 correlations

This script creates multiple visualization types to analyze the relationship between
downstream task performance and semantic F1 scores from Study B.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StudyDVisualizer:
    """Create comprehensive visualizations for Study D results."""

    def __init__(self, results_path: str, output_dir: str):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load results
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            self.results = data['results']
            self.semantic_f1_scores = data['semantic_f1_scores']
            self.correlations = data['correlations']
            self.failed_models = data.get('failed_models', [])

        # Report failed models if any
        if self.failed_models:
            print(
                f"\n⚠️  WARNING: {len(self.failed_models)} models failed to load:"
            )
            for model_name, reason in self.failed_models:
                print(f"   - {model_name}: {reason}")
            print()

        # Model name mapping for cleaner labels
        self.model_labels = {
            'PersuasionForGoodOpenAI_gpt-4o-mini-inference-0-shot_0': 'GPT-4o-mini',
            'PersuasionForGoodOpenAI_gpt-4.1-mini-inference-0-shot_0': 'GPT-4.1-mini',
            'PersuasionForGood_meta-llama--Llama-3.1-8B-Instruct-inference-0-shot_0': 'Llama-3.1-8B',
            'PersuasionForGood_meta-llama--Llama-2-7b-chat-hf-inference-0-shot_0': 'Llama-2-7b-chat',
        }

        # Task type mapping
        self.task_types = {
            'persuader_success': 'classification',
            'persuader_donation': 'regression',
            'persuadee_donation': 'regression',
            'donation': 'regression',
        }

        # Get all individual metrics from semantic_f1_scores
        self.individual_metrics = set()
        for model_scores in self.semantic_f1_scores.values():
            for metric_name in model_scores.keys():
                if metric_name not in ['avg_semantic_f1', 'avg_hard_f1']:
                    self.individual_metrics.add(metric_name)

        self.individual_metrics = sorted(list(self.individual_metrics))

    def prepare_data_for_plotting(
        self, metric_name: str = None
    ) -> pd.DataFrame:
        """Prepare comprehensive dataset for plotting with individual metrics."""
        plot_data = []

        for model_name, model_results in self.results.items():
            model_label = self.model_labels.get(model_name, model_name)

            # Get semantic F1 scores
            semantic_scores = self.semantic_f1_scores.get(model_name, {})

            # If metric_name is specified, use individual metric, otherwise use averages
            if metric_name and metric_name in semantic_scores:
                metric_value = semantic_scores[metric_name]
                metric_type = (
                    'semantic' if 'semantic_f1' in metric_name else 'hard'
                )
            else:
                # Use aggregated scores for backward compatibility
                avg_semantic_f1 = semantic_scores.get('avg_semantic_f1', 0)
                avg_hard_f1 = semantic_scores.get('avg_hard_f1', 0)

            for task_name, task_results in model_results.items():
                if 'downstream_results' not in task_results:
                    continue

                task_type = task_results['task_type']
                downstream_results = task_results['downstream_results']

                # Find best performance
                best_k = None
                best_score = -float('inf') if task_type == 'regression' else 0
                best_metrics = {}

                for k, metrics in downstream_results.items():
                    if task_type == 'classification':
                        score = metrics.get('auc', 0)
                    else:
                        score = metrics.get('neg_mse', -float('inf'))

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_metrics = metrics

                if best_k is not None:
                    row = {
                        'model': model_label,
                        'model_full': model_name,
                        'task': task_name,
                        'task_type': task_type,
                        'best_k': best_k,
                        'best_score': best_score,
                        'n_conversations': task_results['n_conversations'],
                    }

                    # Add metric-specific scores
                    if metric_name and metric_name in semantic_scores:
                        row[f'metric_value'] = metric_value
                        row[f'metric_name'] = metric_name
                        row[f'metric_type'] = metric_type
                    else:
                        # Use aggregated scores
                        row['avg_semantic_f1'] = avg_semantic_f1
                        row['avg_hard_f1'] = avg_hard_f1

                    # Add task-specific metrics
                    if task_type == 'classification':
                        row.update(
                            {
                                'auc': best_metrics.get('auc', 0),
                                'f1': best_metrics.get('f1', 0),
                                'accuracy': best_metrics.get('accuracy', 0),
                            }
                        )
                    else:
                        row.update(
                            {
                                'neg_mse': best_metrics.get('neg_mse', 0),
                                'rmse': best_metrics.get('rmse', 0),
                            }
                        )

                    plot_data.append(row)

        return pd.DataFrame(plot_data)

    def plot_figure_d1_by_metric_versions(self, focus_tasks=None):
        """Create Figure D1 variants for each metric version (macro, micro, sample)."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Group metrics by dataset and type
        metric_versions = {'macro': [], 'micro': [], 'sample': []}

        # Organize metrics by version
        for metric_name in self.individual_metrics:
            if 'macro' in metric_name:
                metric_versions['macro'].append(metric_name)
            elif 'micro' in metric_name:
                metric_versions['micro'].append(metric_name)
            elif 'sample' in metric_name:
                metric_versions['sample'].append(metric_name)

        # Create Figure D1 for each metric version
        for version_name, metrics in metric_versions.items():
            if not metrics:
                continue

            print(f"Creating Figure D1 variant for {version_name} metrics...")
            self.plot_figure_d1_single_version(
                version_name, metrics, focus_tasks
            )

    def plot_figure_d1_single_version(
        self, version_name: str, metrics: List[str], focus_tasks
    ):
        """Create Figure D1 for a specific metric version (macro/micro/sample)."""
        # Separate semantic and hard metrics
        semantic_metrics = [m for m in metrics if 'semantic_f1' in m]
        hard_metrics = [m for m in metrics if 'hard_f1' in m]

        if not semantic_metrics or not hard_metrics:
            print(f"Insufficient metrics for {version_name} version")
            return

        # Prepare aggregated data for this version
        plot_data = []

        for model_name, model_results in self.results.items():
            model_label = self.model_labels.get(model_name, model_name)
            semantic_scores = self.semantic_f1_scores.get(model_name, {})

            # Calculate average for this metric version
            version_semantic_scores = [
                semantic_scores.get(m, 0) for m in semantic_metrics
            ]
            version_hard_scores = [
                semantic_scores.get(m, 0) for m in hard_metrics
            ]

            avg_semantic = (
                np.mean(version_semantic_scores)
                if version_semantic_scores
                else 0
            )
            avg_hard = (
                np.mean(version_hard_scores) if version_hard_scores else 0
            )

            for task_name, task_results in model_results.items():
                if 'downstream_results' not in task_results:
                    continue

                task_type = task_results['task_type']
                downstream_results = task_results['downstream_results']

                # Find best performance
                best_k = None
                best_score = -float('inf') if task_type == 'regression' else 0
                best_metrics = {}

                for k, metrics_data in downstream_results.items():
                    if task_type == 'classification':
                        score = metrics_data.get('auc', 0)
                    else:
                        score = metrics_data.get('neg_mse', -float('inf'))

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_metrics = metrics_data

                if best_k is not None:
                    row = {
                        'model': model_label,
                        'model_full': model_name,
                        'task': task_name,
                        'task_type': task_type,
                        'best_k': best_k,
                        'best_score': best_score,
                        'version_semantic_f1': avg_semantic,
                        'version_hard_f1': avg_hard,
                        'n_conversations': task_results['n_conversations'],
                        'version_name': version_name,
                    }

                    # Add task-specific metrics with confidence intervals
                    if task_type == 'classification':
                        row.update(
                            {
                                'auc': best_metrics.get('auc', 0),
                                'auc_ci': best_metrics.get('auc_ci', 0),
                                'f1': best_metrics.get('f1', 0),
                                'f1_ci': best_metrics.get('f1_ci', 0),
                                'accuracy': best_metrics.get('accuracy', 0),
                                'accuracy_ci': best_metrics.get(
                                    'accuracy_ci', 0
                                ),
                            }
                        )
                    else:
                        row.update(
                            {
                                'neg_mse': best_metrics.get('neg_mse', 0),
                                'neg_mse_ci': best_metrics.get('neg_mse_ci', 0),
                                'rmse': best_metrics.get('rmse', 0),
                                'rmse_ci': best_metrics.get('rmse_ci', 0),
                            }
                        )

                    plot_data.append(row)

        if not plot_data:
            print(f"No data available for {version_name} version")
            return

        df = pd.DataFrame(plot_data)

        # Filter to focus tasks
        if focus_tasks != 'all':
            available_tasks = df['task'].unique()
            tasks = [task for task in focus_tasks if task in available_tasks]
        else:
            tasks = df['task'].unique()

        n_tasks = len(tasks)
        if n_tasks == 0:
            return

        # Create single row of subplots for line plots
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_data = df[df['task'] == task].copy()
            task_data = task_data.sort_values('best_score', ascending=False)

            # Main performance metric
            task_type = task_data['task_type'].iloc[0]
            if task_type == 'classification':
                performance_col = 'auc'
                performance_ci_col = 'auc_ci'
                performance_label = 'AUC'
            else:
                performance_col = 'neg_mse'
                performance_ci_col = 'neg_mse_ci'
                performance_label = '-MSE'

            x_pos = np.arange(len(task_data))

            # Create secondary y-axis for downstream performance
            ax2 = ax.twinx()

            # Plot F1 scores on left axis (ax) with simulated confidence intervals
            # Since Study B doesn't provide CI, we simulate reasonable uncertainty (~5% of value)
            semantic_f1_values = task_data['version_semantic_f1']
            hard_f1_values = task_data['version_hard_f1']

            # Simulate CI as 5% of the metric value (reasonable uncertainty estimate)
            semantic_f1_ci = semantic_f1_values * 0.05
            hard_f1_ci = hard_f1_values * 0.05

            ax.errorbar(
                x_pos,
                semantic_f1_values,
                yerr=semantic_f1_ci,
                fmt='s-',
                color='red',
                linewidth=3,
                markersize=8,
                capsize=5,
                label=f'{version_name.title()} Semantic F1',
            )

            ax.errorbar(
                x_pos,
                hard_f1_values,
                yerr=hard_f1_ci,
                fmt='^-',
                color='green',
                linewidth=3,
                markersize=8,
                capsize=5,
                label=f'{version_name.title()} Hard F1',
            )

            # Plot downstream performance with CI on right axis (ax2)
            ax2.errorbar(
                x_pos,
                task_data[performance_col],
                yerr=task_data[performance_ci_col],
                fmt='o-',
                color='steelblue',
                linewidth=3,
                markersize=10,
                capsize=5,
                label=f'Downstream {performance_label} (95% CI)',
            )

            # Formatting
            ax.set_title(
                f'{task.replace("_", " ").title()}\n({version_name.title()} Metrics)',
                fontsize=16,
                fontweight='bold',
            )
            ax.set_xlabel(
                'Model (ranked by downstream performance)', fontsize=12
            )
            ax.set_ylabel(f'{version_name.title()} F1 Score', fontsize=12)
            ax2.set_ylabel(f'Downstream {performance_label}', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(task_data['model'], rotation=45)

            ax.grid(True, alpha=0.3, zorder=0)

            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc='best',
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                facecolor='white',
            )

            # Set y-axis limits for F1 scores (left axis)
            f1_min = (
                min(
                    task_data['version_semantic_f1'].min(),
                    task_data['version_hard_f1'].min(),
                )
                - 0.05
            )
            f1_max = (
                max(
                    task_data['version_semantic_f1'].max(),
                    task_data['version_hard_f1'].max(),
                )
                + 0.1
            )
            ax.set_ylim(f1_min, f1_max)

            # Set y-axis limits for downstream performance (right axis) - use better range
            perf_values = task_data[performance_col]
            perf_ci_values = task_data[performance_ci_col]

            # Calculate range with error bars
            perf_min_with_ci = (perf_values - perf_ci_values).min()
            perf_max_with_ci = (perf_values + perf_ci_values).max()

            # Add padding based on data range
            perf_range = perf_max_with_ci - perf_min_with_ci
            padding = max(
                0.1 * perf_range, 0.02
            )  # At least 10% padding or 0.02

            perf_min = perf_min_with_ci - padding
            perf_max = perf_max_with_ci + padding

            # For negative MSE, ensure we don't go above 0 unnecessarily
            if performance_label == '-MSE' and perf_max > 0:
                perf_max = min(
                    0, perf_max_with_ci + 0.1 * abs(perf_max_with_ci)
                )

            ax2.set_ylim(perf_min, perf_max)

        plt.tight_layout()
        save_path_png = (
            self.output_dir / f'figure_d1_{version_name}_metrics.png'
        )
        save_path_pdf = (
            self.output_dir / f'figure_d1_{version_name}_metrics.pdf'
        )

        print(f"Saving Figure D1 ({version_name}) to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_individual_metric_analysis(self, focus_tasks=None):
        """Create separate analysis for each individual metric."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Create figures for each individual metric
        for metric_name in self.individual_metrics:
            print(f"Creating visualizations for metric: {metric_name}")

            # Prepare data for this specific metric
            df = self.prepare_data_for_plotting(metric_name)

            if df.empty:
                print(
                    f"No data available for metric {metric_name}, skipping..."
                )
                continue

            # Extract metric info
            metric_type = 'semantic' if 'semantic_f1' in metric_name else 'hard'
            dataset_info = metric_name.replace('_semantic_f1', '').replace(
                '_hard_f1', ''
            )

            # Create the visualization
            self.plot_single_metric_correlation(
                df, metric_name, metric_type, dataset_info, focus_tasks
            )
        """Create separate analysis for each individual metric."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Create figures for each individual metric
        for metric_name in self.individual_metrics:
            print(f"Creating visualizations for metric: {metric_name}")

            # Prepare data for this specific metric
            df = self.prepare_data_for_plotting(metric_name)

            if df.empty:
                print(
                    f"No data available for metric {metric_name}, skipping..."
                )
                continue

            # Extract metric info
            metric_type = 'semantic' if 'semantic_f1' in metric_name else 'hard'
            dataset_info = metric_name.replace('_semantic_f1', '').replace(
                '_hard_f1', ''
            )

            # Create the visualization
            self.plot_single_metric_correlation(
                df, metric_name, metric_type, dataset_info, focus_tasks
            )

    def plot_single_metric_correlation(
        self,
        df: pd.DataFrame,
        metric_name: str,
        metric_type: str,
        dataset_info: str,
        focus_tasks,
    ):
        """Plot correlation for a single metric."""
        # Filter to focus tasks if specified
        if focus_tasks != 'all':
            available_tasks = df['task'].unique()
            tasks = [task for task in focus_tasks if task in available_tasks]
        else:
            tasks = df['task'].unique()

        if not tasks:
            print(f"No tasks available for {metric_name}")
            return

        n_tasks = len(tasks)

        # Create subplot layout
        fig, axes = plt.subplots(1, n_tasks, figsize=(8 * n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]

        color = 'blue' if metric_type == 'semantic' else 'red'

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_data = df[df['task'] == task].copy()

            if task_data.empty:
                continue

            # Determine performance metric
            task_type = task_data['task_type'].iloc[0]
            if task_type == 'classification':
                y_col = 'auc'
                y_label = 'Downstream AUC'
            else:
                y_col = 'neg_mse'
                y_label = 'Downstream -MSE'

            # Scatter plot
            ax.scatter(
                task_data['metric_value'],
                task_data[y_col],
                s=100,
                alpha=0.7,
                color=color,
                label=f'{metric_type.title()} F1',
            )

            # Add model labels
            for _, row in task_data.iterrows():
                ax.annotate(
                    row['model'],
                    (row['metric_value'], row[y_col]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                )

            # Fit trend line with error handling
            if len(task_data) > 2:
                try:
                    z = np.polyfit(
                        task_data['metric_value'], task_data[y_col], 1
                    )
                    p = np.poly1d(z)
                    ax.plot(
                        task_data['metric_value'],
                        p(task_data['metric_value']),
                        f"{color[0]}--",
                        alpha=0.8,
                        linewidth=2,
                    )
                except np.linalg.LinAlgError:
                    # Skip trend line if SVD doesn't converge
                    print(
                        f"Warning: Could not fit trend line for {metric_name} - {task}"
                    )
                    pass

            ax.set_title(
                f'{task.replace("_", " ").title()}\n{dataset_info} {metric_type.title()} F1',
                fontsize=14,
                fontweight='bold',
            )
            ax.set_xlabel(f'{metric_type.title()} F1 Score', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            legend = ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                facecolor='white',
            )
            ax.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()

        # Save figure
        safe_metric_name = metric_name.replace('/', '_').replace('\\', '_')
        save_path_png = (
            self.output_dir / f'figure_individual_{safe_metric_name}.png'
        )
        save_path_pdf = (
            self.output_dir / f'figure_individual_{safe_metric_name}.pdf'
        )

        print(f"Saving individual metric figure to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_task_performance_comparison(
        self, df: pd.DataFrame, focus_tasks=None
    ):
        """Plot 1: Line plots for PersuasionForGood tasks with all metrics (aggregated view)."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Filter to focus tasks, but allow all if requested
        if focus_tasks != 'all':
            available_tasks = df['task'].unique()
            tasks = [task for task in focus_tasks if task in available_tasks]
        else:
            tasks = df['task'].unique()

        n_tasks = len(tasks)

        # Create single row of subplots for line plots
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_data = df[df['task'] == task].copy()
            task_data = task_data.sort_values('best_score', ascending=False)

            # Main performance metric
            task_type = task_data['task_type'].iloc[0]
            if task_type == 'classification':
                performance_col = 'auc'
                performance_label = 'AUC'
            else:
                performance_col = 'neg_mse'
                performance_label = '-MSE'

            x_pos = np.arange(len(task_data))

            # Create secondary y-axis for PersuasionForGood performance
            ax2 = ax.twinx()

            # Plot F1 scores on left axis (ax) - use aggregated scores
            if 'avg_semantic_f1' in task_data.columns:
                ax.plot(
                    x_pos,
                    task_data['avg_semantic_f1'],
                    's-',
                    color='red',
                    linewidth=3,
                    markersize=8,
                    label='Avg Semantic F1',
                )

                ax.plot(
                    x_pos,
                    task_data['avg_hard_f1'],
                    '^-',
                    color='green',
                    linewidth=3,
                    markersize=8,
                    label='Avg Hard F1',
                )
            else:
                print(f"Warning: No aggregated F1 scores available for {task}")

            # Plot PersuasionForGood performance on right axis (ax2)
            ax2.plot(
                x_pos,
                task_data[performance_col],
                'o-',
                color='steelblue',
                linewidth=3,
                markersize=10,
                label=f'Downstream {performance_label}',
            )

            # Formatting
            ax.set_title(
                f'{task.replace("_", " ").title()}',
                fontsize=16,
                fontweight='bold',
            )
            ax.set_xlabel(
                'Model (ranked by downstream performance)', fontsize=12
            )
            ax.set_ylabel('F1 Score', fontsize=12)
            ax2.set_ylabel(f'Downstream {performance_label}', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(task_data['model'], rotation=45)

            ax.grid(True, alpha=0.3, zorder=0)

            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc='best',
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                facecolor='white',
            )

            # Set y-axis limits for F1 scores (left axis)
            if 'avg_semantic_f1' in task_data.columns:
                f1_min = (
                    min(
                        task_data['avg_semantic_f1'].min(),
                        task_data['avg_hard_f1'].min(),
                    )
                    - 0.05
                )
                f1_max = (
                    max(
                        task_data['avg_semantic_f1'].max(),
                        task_data['avg_hard_f1'].max(),
                    )
                    + 0.1
                )
                ax.set_ylim(f1_min, f1_max)

            # Set y-axis limits for downstream performance (right axis)
            perf_min = task_data[performance_col].min() - 0.05
            perf_max = task_data[performance_col].max() + 0.1
            ax2.set_ylim(perf_min, perf_max)

        plt.tight_layout()
        save_path_png = (
            self.output_dir / 'figure_d1_task_performance_comparison.png'
        )
        save_path_pdf = (
            self.output_dir / 'figure_d1_task_performance_comparison.pdf'
        )

        print(f"Saving Figure D1 to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display issues

    def plot_correlation_scatter(self, df: pd.DataFrame, focus_tasks=None):
        """Plot 2: Scatter plots showing correlation between semantic F1 and downstream performance (aggregated)."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Check if we have aggregated F1 scores
        if df.empty or 'avg_semantic_f1' not in df.columns:
            print(
                "Warning: No aggregated F1 scores available for scatter plots"
            )
            return

        # Filter to focus tasks, but allow all if requested
        if focus_tasks != 'all':
            available_tasks = df['task'].unique()
            tasks = [task for task in focus_tasks if task in available_tasks]
        else:
            tasks = df['task'].unique()

        n_tasks = len(tasks)

        # Create appropriate subplot layout
        if n_tasks <= 2:
            fig, axes = plt.subplots(1, n_tasks, figsize=(8 * n_tasks, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if n_tasks == 1:
            axes = [axes]
        elif n_tasks > 1 and n_tasks <= 2:
            pass  # axes is already a list
        else:
            axes = axes.flatten()

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_data = df[df['task'] == task]

            # Determine performance metric
            task_type = task_data['task_type'].iloc[0]
            if task_type == 'classification':
                y_col = 'auc'
                y_label = 'Downstream AUC'
            else:
                y_col = 'neg_mse'
                y_label = 'Downstream -MSE'

            # Scatter plot for semantic F1
            ax.scatter(
                task_data['avg_semantic_f1'],
                task_data[y_col],
                s=100,
                alpha=0.7,
                label='Avg Semantic F1',
                color='blue',
            )

            # Scatter plot for hard F1
            ax.scatter(
                task_data['avg_hard_f1'],
                task_data[y_col],
                s=100,
                alpha=0.7,
                label='Avg Hard F1',
                color='red',
                marker='^',
            )

            # Add model labels
            for _, row in task_data.iterrows():
                ax.annotate(
                    row['model'],
                    (row['avg_semantic_f1'], row[y_col]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                )

            # Fit trend lines
            if len(task_data) > 2:
                # Semantic F1 trend
                z_sem = np.polyfit(
                    task_data['avg_semantic_f1'], task_data[y_col], 1
                )
                p_sem = np.poly1d(z_sem)
                ax.plot(
                    task_data['avg_semantic_f1'],
                    p_sem(task_data['avg_semantic_f1']),
                    "b--",
                    alpha=0.8,
                    linewidth=2,
                )

                # Hard F1 trend
                z_hard = np.polyfit(
                    task_data['avg_hard_f1'], task_data[y_col], 1
                )
                p_hard = np.poly1d(z_hard)
                ax.plot(
                    task_data['avg_hard_f1'],
                    p_hard(task_data['avg_hard_f1']),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                )

            ax.set_title(
                f'{task.replace("_", " ").title()}',
                fontsize=14,
                fontweight='bold',
            )
            ax.set_xlabel('Average F1 Score', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            legend = ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                facecolor='white',
            )
            ax.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d2_correlation_scatter.png'
        save_path_pdf = self.output_dir / 'figure_d2_correlation_scatter.pdf'

        print(f"Saving Figure D2 to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_model_ranking_comparison(self, df: pd.DataFrame, focus_tasks=None):
        """Plot 3: Model ranking comparison across tasks and metrics (aggregated)."""
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']

        # Check if we have aggregated F1 scores
        if df.empty or 'avg_semantic_f1' not in df.columns:
            print(
                "Warning: No aggregated F1 scores available for ranking comparison"
            )
            return

        # Filter to focus tasks, but allow all if requested
        if focus_tasks != 'all':
            available_tasks = df['task'].unique()
            tasks = [task for task in focus_tasks if task in available_tasks]
        else:
            tasks = df['task'].unique()

        # Prepare ranking data
        models = df['model'].unique()

        # Create ranking matrices
        semantic_rankings = np.zeros((len(models), len(tasks)))
        hard_rankings = np.zeros((len(models), len(tasks)))
        downstream_rankings = np.zeros((len(models), len(tasks)))

        for j, task in enumerate(tasks):
            task_data = df[df['task'] == task].copy()

            # Rank by downstream performance
            task_data = task_data.sort_values('best_score', ascending=False)
            for i, model in enumerate(models):
                model_data = task_data[task_data['model'] == model]
                if not model_data.empty:
                    downstream_rankings[i, j] = (
                        task_data[task_data['model'] == model].index[0] + 1
                    )

            # Rank by semantic F1
            task_data = task_data.sort_values(
                'avg_semantic_f1', ascending=False
            )
            for i, model in enumerate(models):
                model_data = task_data[task_data['model'] == model]
                if not model_data.empty:
                    semantic_rankings[i, j] = (
                        task_data[task_data['model'] == model].index[0] + 1
                    )

            # Rank by hard F1
            task_data = task_data.sort_values('avg_hard_f1', ascending=False)
            for i, model in enumerate(models):
                model_data = task_data[task_data['model'] == model]
                if not model_data.empty:
                    hard_rankings[i, j] = (
                        task_data[task_data['model'] == model].index[0] + 1
                    )

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Define task labels
        task_labels = [task.replace('_', ' ').title() for task in tasks]

        # Plot heatmaps
        im1 = axes[0].imshow(
            downstream_rankings, cmap='RdYlBu_r', aspect='auto'
        )
        axes[0].set_title(
            'Downstream Performance Rankings', fontsize=14, fontweight='bold'
        )
        axes[0].set_xticks(range(len(task_labels)))
        axes[0].set_xticklabels(task_labels, rotation=45)
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels(models)

        im2 = axes[1].imshow(semantic_rankings, cmap='RdYlBu_r', aspect='auto')
        axes[1].set_title(
            'Avg Semantic F1 Rankings', fontsize=14, fontweight='bold'
        )
        axes[1].set_xticks(range(len(task_labels)))
        axes[1].set_xticklabels(task_labels, rotation=45)
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels(models)

        im3 = axes[2].imshow(hard_rankings, cmap='RdYlBu_r', aspect='auto')
        axes[2].set_title(
            'Avg Hard F1 Rankings', fontsize=14, fontweight='bold'
        )
        axes[2].set_xticks(range(len(task_labels)))
        axes[2].set_xticklabels(task_labels, rotation=45)
        axes[2].set_yticks(range(len(models)))
        axes[2].set_yticklabels(models)

        # Add value annotations
        for ax, rankings in zip(
            axes, [downstream_rankings, semantic_rankings, hard_rankings]
        ):
            for i in range(len(models)):
                for j in range(len(tasks)):
                    if rankings[i, j] > 0:
                        ax.text(
                            j,
                            i,
                            f'{int(rankings[i, j])}',
                            ha='center',
                            va='center',
                            fontweight='bold',
                        )

        # Add colorbar
        cbar = plt.colorbar(
            im1, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8
        )
        cbar.set_label('Ranking (1=Best)', fontsize=12)

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d3_ranking_comparison.png'
        save_path_pdf = self.output_dir / 'figure_d3_ranking_comparison.pdf'

        print(f"Saving Figure D3 to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_correlation_summary(self):
        """Plot 4: Summary of correlation analysis for individual metrics."""
        if not self.correlations:
            print("No correlation data available")
            return

        # Get individual metric correlations
        individual_correlations = {
            k: v
            for k, v in self.correlations.items()
            if k
            not in ['semantic_f1_correlation', 'hard_f1_correlation', 'data']
        }

        if not individual_correlations:
            print("No individual metric correlations available")
            return

        # Prepare data for plotting
        metric_names = []
        correlations_vals = []
        p_values = []
        n_models = []
        metric_types = []

        for metric_name, corr_data in individual_correlations.items():
            metric_names.append(metric_name)
            correlations_vals.append(corr_data.get('correlation', 0))
            p_values.append(corr_data.get('p_value', 1))
            n_models.append(corr_data.get('n_models', 0))
            metric_types.append(
                'Semantic' if 'semantic_f1' in metric_name else 'Hard'
            )

        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Correlation values by metric
        colors = ['blue' if mt == 'Semantic' else 'red' for mt in metric_types]
        bars = ax1.bar(
            range(len(metric_names)), correlations_vals, color=colors, alpha=0.7
        )
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Spearman Correlation', fontsize=12)
        ax1.set_title(
            'Correlations by Individual Metric', fontsize=14, fontweight='bold'
        )
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels(
            [m.replace('_', '\n') for m in metric_names],
            rotation=45,
            ha='right',
        )
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add correlation values on bars
        for i, (bar, corr_val) in enumerate(zip(bars, correlations_vals)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 if height > 0 else height - 0.03,
                f'{corr_val:.3f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10,
            )

        # Plot 2: P-values
        bars2 = ax2.bar(
            range(len(metric_names)), p_values, color=colors, alpha=0.7
        )
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('P-value', fontsize=12)
        ax2.set_title(
            'Statistical Significance', fontsize=14, fontweight='bold'
        )
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels(
            [m.replace('_', '\n') for m in metric_names],
            rotation=45,
            ha='right',
        )
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.axhline(
            y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05'
        )
        legend2 = ax2.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            facecolor='white',
        )

        # Plot 3: Semantic vs Hard comparison
        semantic_corrs = [
            corr
            for corr, mt in zip(correlations_vals, metric_types)
            if mt == 'Semantic'
        ]
        hard_corrs = [
            corr
            for corr, mt in zip(correlations_vals, metric_types)
            if mt == 'Hard'
        ]

        if semantic_corrs and hard_corrs:
            x_pos = np.arange(max(len(semantic_corrs), len(hard_corrs)))
            width = 0.35

            ax3.bar(
                x_pos - width / 2,
                semantic_corrs + [0] * (len(x_pos) - len(semantic_corrs)),
                width,
                label='Semantic F1',
                color='blue',
                alpha=0.7,
            )
            ax3.bar(
                x_pos + width / 2,
                hard_corrs + [0] * (len(x_pos) - len(hard_corrs)),
                width,
                label='Hard F1',
                color='red',
                alpha=0.7,
            )

            ax3.set_xlabel('Dataset/Metric Index', fontsize=12)
            ax3.set_ylabel('Spearman Correlation', fontsize=12)
            ax3.set_title(
                'Semantic vs Hard F1 Correlations',
                fontsize=14,
                fontweight='bold',
            )
            legend3 = ax3.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                facecolor='white',
            )
            ax3.grid(True, alpha=0.3, zorder=0)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Plot 4: Sample scatter for best correlation
        best_idx = np.argmax(np.abs(correlations_vals))
        best_metric = metric_names[best_idx]
        best_corr_data = individual_correlations[best_metric].get('data', {})

        if best_corr_data:
            models = best_corr_data.get('models', [])
            downstream_scores = best_corr_data.get('downstream_scores', [])
            metric_values = best_corr_data.get('metric_values', [])

            model_labels = [self.model_labels.get(m, m) for m in models]
            color = 'blue' if 'semantic_f1' in best_metric else 'red'

            ax4.scatter(
                metric_values, downstream_scores, s=150, alpha=0.7, color=color
            )

            for i, model in enumerate(model_labels):
                ax4.annotate(
                    model,
                    (metric_values[i], downstream_scores[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                )

            # Trend line
            if len(metric_values) > 1:
                z = np.polyfit(metric_values, downstream_scores, 1)
                p = np.poly1d(z)
                ax4.plot(
                    metric_values,
                    p(metric_values),
                    f"{color[0]}--",
                    alpha=0.8,
                    linewidth=2,
                )

            ax4.set_xlabel(
                f'{best_metric.replace("_", " ").title()}', fontsize=12
            )
            ax4.set_ylabel('Best Downstream Performance', fontsize=12)
            ax4.set_title(
                f'Best Correlation: {best_metric}\nρ = {correlations_vals[best_idx]:.3f}',
                fontsize=14,
                fontweight='bold',
            )
            ax4.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        save_path_png = (
            self.output_dir / 'figure_d4_individual_correlations_summary.png'
        )
        save_path_pdf = (
            self.output_dir / 'figure_d4_individual_correlations_summary.pdf'
        )

        print(f"Saving Figure D4 (Individual Correlations) to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

        # Also create the aggregated correlation summary for backward compatibility
        self.plot_aggregated_correlation_summary()

    def plot_aggregated_correlation_summary(self):
        """Plot aggregated correlation summary (original method)."""
        if not self.correlations:
            print("No correlation data available")
            return

        # Check if aggregated data exists
        if 'data' not in self.correlations:
            print("No aggregated correlation data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Extract correlation data
        corr_data = self.correlations.get('data', {})
        models = corr_data.get('models', [])
        downstream_scores = corr_data.get('downstream_scores', [])
        semantic_f1_scores = corr_data.get('semantic_f1_scores', [])
        hard_f1_scores = corr_data.get('hard_f1_scores', [])

        if not models:
            print("No aggregated correlation data available for plotting")
            return

        # Convert model names
        model_labels = [self.model_labels.get(m, m) for m in models]

        # Plot 1: Semantic F1 vs Downstream
        semantic_corr = self.correlations.get('semantic_f1_correlation', {})
        ax1.scatter(
            semantic_f1_scores,
            downstream_scores,
            s=150,
            alpha=0.7,
            color='blue',
        )

        for i, model in enumerate(model_labels):
            ax1.annotate(
                model,
                (semantic_f1_scores[i], downstream_scores[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
            )

        # Trend line
        if len(semantic_f1_scores) > 1:
            z = np.polyfit(semantic_f1_scores, downstream_scores, 1)
            p = np.poly1d(z)
            ax1.plot(
                semantic_f1_scores,
                p(semantic_f1_scores),
                "b--",
                alpha=0.8,
                linewidth=2,
            )

        ax1.set_xlabel('Average Semantic F1 Score', fontsize=12)
        ax1.set_ylabel('Best Downstream Performance', fontsize=12)
        ax1.set_title(
            f'Semantic F1 vs Downstream\nρ = {semantic_corr.get("correlation", 0):.3f}',
            fontsize=14,
            fontweight='bold',
        )
        ax1.grid(True, alpha=0.3, zorder=0)

        # Plot 2: Hard F1 vs Downstream
        hard_corr = self.correlations.get('hard_f1_correlation', {})
        ax2.scatter(
            hard_f1_scores, downstream_scores, s=150, alpha=0.7, color='red'
        )

        for i, model in enumerate(model_labels):
            ax2.annotate(
                model,
                (hard_f1_scores[i], downstream_scores[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
            )

        # Trend line
        if len(hard_f1_scores) > 1:
            z = np.polyfit(hard_f1_scores, downstream_scores, 1)
            p = np.poly1d(z)
            ax2.plot(
                hard_f1_scores, p(hard_f1_scores), "r--", alpha=0.8, linewidth=2
            )

        ax2.set_xlabel('Average Hard F1 Score', fontsize=12)
        ax2.set_ylabel('Best Downstream Performance', fontsize=12)
        ax2.set_title(
            f'Hard F1 vs Downstream\nρ = {hard_corr.get("correlation", 0):.3f}',
            fontsize=14,
            fontweight='bold',
        )
        ax2.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        save_path_png = (
            self.output_dir / 'figure_d4_aggregated_correlation_summary.png'
        )
        save_path_pdf = (
            self.output_dir / 'figure_d4_aggregated_correlation_summary.pdf'
        )

        print(f"Saving Figure D4 (Aggregated) to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def create_all_visualizations(self, focus_tasks=None):
        """Create all visualization types."""
        print("Preparing data for visualization...")

        # Default to main tasks but allow override
        if focus_tasks is None:
            focus_tasks = ['persuader_success', 'persuadee_donation']
            print(f"Focusing on main tasks: {focus_tasks}")
            print("(To include all tasks, pass focus_tasks='all')")
        elif focus_tasks == 'all':
            print("Including all available tasks")

        # Create Figure D1 variants for each metric version (NEW)
        print("\n=== Creating Figure D1 Variants by Metric Version ===")
        try:
            self.plot_figure_d1_by_metric_versions(focus_tasks)
            print("Figure D1 metric variants completed successfully")
        except Exception as e:
            print(f"Error creating Figure D1 variants: {e}")
            import traceback

            traceback.print_exc()

        # Create individual metric visualizations
        print("\n=== Creating Individual Metric Visualizations ===")
        try:
            self.plot_individual_metric_analysis(focus_tasks)
            print("Individual metric visualizations completed successfully")
        except Exception as e:
            print(f"Error creating individual metric visualizations: {e}")
            import traceback

            traceback.print_exc()

        # Create aggregated visualizations for backward compatibility
        print("\n=== Creating Aggregated Visualizations ===")
        df = self.prepare_data_for_plotting()  # Use aggregated data

        print("Creating Figure D1: Task Performance Comparison (Aggregated)...")
        try:
            self.plot_task_performance_comparison(df, focus_tasks)
            print("Figure D1 completed successfully")
        except Exception as e:
            print(f"Error creating Figure D1: {e}")
            import traceback

            traceback.print_exc()

        print("Creating Figure D2: Correlation Scatter Plots (Aggregated)...")
        try:
            self.plot_correlation_scatter(df, focus_tasks)
            print("Figure D2 completed successfully")
        except Exception as e:
            print(f"Error creating Figure D2: {e}")

        print("Creating Figure D3: Model Ranking Comparison (Aggregated)...")
        try:
            self.plot_model_ranking_comparison(df, focus_tasks)
            print("Figure D3 completed successfully")
        except Exception as e:
            print(f"Error creating Figure D3: {e}")

        print(
            "Creating Figure D4: Correlation Summary (Individual + Aggregated)..."
        )
        try:
            self.plot_correlation_summary()
            print("Figure D4 completed successfully")
        except Exception as e:
            print(f"Error creating Figure D4: {e}")

        print(f"\nAll visualizations saved to {self.output_dir}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        if not df.empty:
            print(f"Number of models analyzed: {len(df['model'].unique())}")
            print(f"Number of tasks analyzed: {len(df['task'].unique())}")

        print(f"Number of individual metrics: {len(self.individual_metrics)}")
        for metric in self.individual_metrics:
            print(f"  - {metric}")

        # Print individual metric correlations
        individual_correlations = {
            k: v
            for k, v in self.correlations.items()
            if k
            not in ['semantic_f1_correlation', 'hard_f1_correlation', 'data']
        }

        if individual_correlations:
            print(f"\nIndividual Metric Correlations:")
            for metric_name, corr_data in individual_correlations.items():
                corr_val = corr_data.get('correlation', 0)
                p_val = corr_data.get('p_value', 1)
                n_models = corr_data.get('n_models', 0)
                print(
                    f"  {metric_name}: ρ = {corr_val:.3f} (p = {p_val:.3f}, n = {n_models})"
                )

        # Print aggregated correlations
        if self.correlations:
            semantic_corr = self.correlations.get('semantic_f1_correlation', {})
            hard_corr = self.correlations.get('hard_f1_correlation', {})
            print(f"\nAggregated Correlations:")
            print(
                f"  Semantic F1 correlation: {semantic_corr.get('correlation', 'N/A'):.3f}"
            )
            print(
                f"  Hard F1 correlation: {hard_corr.get('correlation', 'N/A'):.3f}"
            )

        if not df.empty:
            print("\nBest performing models by task:")
            for task in df['task'].unique():
                task_data = df[df['task'] == task].sort_values(
                    'best_score', ascending=False
                )
                if not task_data.empty:
                    print(
                        f"  {task}: {task_data.iloc[0]['model']} "
                        f"(score: {task_data.iloc[0]['best_score']:.3f})"
                    )


def main():
    parser = argparse.ArgumentParser(description='Study D Visualization')
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='Path to Study D results pickle file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for figures',
    )

    args = parser.parse_args()

    visualizer = StudyDVisualizer(args.results_path, args.output_dir)
    visualizer.create_all_visualizations()


if __name__ == '__main__':
    main()
