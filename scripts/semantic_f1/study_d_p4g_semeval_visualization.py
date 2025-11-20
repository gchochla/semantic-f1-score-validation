#!/usr/bin/env python3
"""
Study D P4G-SemEval Visualization: Plot P4G downstream performance vs SemEval F1 correlations

This script creates visualizations to analyze the relationship between
downstream task performance on PersuasionForGood and SemEval 2-shot F1 scores.
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR


class StudyDP4GVisualizor:
    """Create comprehensive visualizations for Study D P4G vs SemEval results."""

    def __init__(self, results_path: str, output_dir: str):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load results
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            self.p4g_results = data['results']
            self.semeval_results = data['semeval_results']
            self.correlations = data['correlations']
            self.failed_models = data.get('failed_models', [])
            self.metadata = data.get('metadata', {})
            self.semantic_metric = self.metadata.get('semantic_metric', 'f1')

        # Report semantic metric being used
        print(f"\n📊 Semantic metric in use: {self.semantic_metric}")
        print(f"   Display label: {self.get_semantic_metric_label()}\n")

        # Report failed models if any
        if self.failed_models:
            print(
                f"\n⚠️  WARNING: {len(self.failed_models)} models failed to load:"
            )
            for model_name, reason in self.failed_models:
                print(f"  - {model_name}: {reason}")
            print()

        # Model name mapping for cleaner labels
        self.model_labels = {
            'gpt-4o-mini-inference-semeval-0-shot_0': 'GPT-4o-mini',
            'gpt-4.1-mini-inference-semeval-0-shot_0': 'GPT-4.1-mini',
            'meta-llama--Llama-3.1-8B-Instruct-inference-semeval-0-shot_0': 'Llama-3.1-8B',
            'meta-llama--Llama-2-7b-chat-hf-inference-semeval-0-shot_0': 'Llama-2-7b-chat',
            'meta-llama--Llama-2-70b-chat-hf-inference-semeval-0-shot_0': 'Llama-2-70b-chat',
            'meta-llama--Llama-3.3-70B-Instruct-inference-semeval-0-shot_0': 'Llama-3.3-70B',
        }

        # Task type mapping
        self.task_types = {
            'persuader_success': 'classification',
            'persuader_donation': 'regression',
            'persuadee_donation': 'regression',
            'donation': 'regression',
        }

    def get_semantic_metric_label(self) -> str:
        """Get the properly formatted semantic metric name for display."""
        return self.semantic_metric.title()

    def prepare_data_for_plotting(self) -> pd.DataFrame:
        """Prepare comprehensive dataset for plotting."""
        plot_data = []

        for model_name, model_results in self.p4g_results.items():
            model_label = self.model_labels.get(model_name, model_name)

            # Get SemEval scores
            semeval_scores = self.semeval_results.get(model_name, {})

            for task_name, task_results in model_results.items():
                # Extract best downstream performance
                if 'downstream_results' not in task_results:
                    continue

                downstream_results = task_results['downstream_results']
                if not downstream_results:
                    continue

                task_type = self.task_types[task_name]

                # Find best performance
                if task_type == 'classification':
                    best_k = max(
                        downstream_results.keys(),
                        key=lambda k: downstream_results[k].get('auc', 0),
                    )
                    best_score = downstream_results[best_k]['auc']
                    best_score_ci = downstream_results[best_k].get('auc_ci', 0)
                    performance_metric = 'AUC'
                else:
                    best_k = max(
                        downstream_results.keys(),
                        key=lambda k: downstream_results[k].get(
                            'neg_mse', -float('inf')
                        ),
                    )
                    best_score = downstream_results[best_k]['neg_mse']
                    best_score_ci = downstream_results[best_k].get(
                        'neg_mse_ci', 0
                    )
                    performance_metric = '-MSE'

                # Add row for each SemEval metric
                for (
                    semeval_metric,
                    semeval_score,
                ) in semeval_scores.items():
                    if 'f1' not in semeval_metric.lower():
                        continue

                    plot_data.append(
                        {
                            'model': model_label,
                            'model_full': model_name,
                            'task': task_name,
                            'task_type': task_type,
                            'best_score': best_score,
                            'best_score_ci': best_score_ci,
                            'performance_metric': performance_metric,
                            'best_k': best_k,
                            'semeval_metric': semeval_metric,
                            'semeval_score': semeval_score,
                        }
                    )

        return pd.DataFrame(plot_data)

    def plot_correlation_overview(self):
        """Plot overview of all correlations found."""
        if not self.correlations:
            print("No correlations found to plot")
            return

        # Extract correlation data
        metric_names = []
        correlation_values = []
        p_values = []
        n_models = []

        for metric_name, corr_data in self.correlations.items():
            metric_names.append(metric_name.replace('_', '\n'))
            correlation_values.append(corr_data['correlation'])
            p_values.append(corr_data['p_value'])
            n_models.append(corr_data['n_models'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Correlation values
        colors = ['red' if p < 0.05 else 'lightcoral' for p in p_values]
        bars = ax1.bar(
            range(len(metric_names)),
            correlation_values,
            color=colors,
            alpha=0.7,
        )

        ax1.set_xlabel('SemEval F1 Metrics', fontsize=12)
        ax1.set_ylabel('Spearman Correlation with P4G Performance', fontsize=12)
        ax1.set_title(
            'Correlations: SemEval F1 vs P4G Downstream Performance',
            fontsize=14,
            fontweight='bold',
        )
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add correlation values on bars
        for i, (bar, corr_val, p_val) in enumerate(
            zip(bars, correlation_values, p_values)
        ):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.02 if height >= 0 else -0.05),
                f'{corr_val:.3f}' + ('*' if p_val < 0.05 else ''),
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontweight='bold',
            )

        # Plot 2: P-values
        bars2 = ax2.bar(
            range(len(metric_names)), p_values, color=colors, alpha=0.7
        )
        ax2.set_xlabel('SemEval F1 Metrics', fontsize=12)
        ax2.set_ylabel('P-value', fontsize=12)
        ax2.set_title(
            'Statistical Significance', fontsize=14, fontweight='bold'
        )
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.axhline(
            y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05'
        )
        ax2.legend()

        # Add p-values on bars
        for i, (bar, p_val) in enumerate(zip(bars2, p_values)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f'{p_val:.3f}',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d1_correlation_overview.png'
        save_path_pdf = self.output_dir / 'figure_d1_correlation_overview.pdf'

        print(f"Saving correlation overview to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_correlation_summary(self):
        """Plot 4: Summary of correlation analysis for individual metrics (like original Study D)."""
        if not self.correlations:
            print("No correlations found to plot")
            return

        # Separate metrics by type
        semantic_metrics = {}
        hard_metrics = {}
        individual_emotion_metrics = {}

        for metric_name, corr_data in self.correlations.items():
            if pd.isna(corr_data['correlation']):
                continue

            if 'semantic_f1' in metric_name:
                semantic_metrics[metric_name] = corr_data
            elif any(
                x in metric_name for x in ['macro_f1', 'micro_f1', 'sample_f1']
            ):
                hard_metrics[metric_name] = corr_data
            elif '_f1' in metric_name and 'test_' in metric_name:
                # Individual emotion F1 scores
                individual_emotion_metrics[metric_name] = corr_data

        # Create Figure D4a: Semantic vs Hard F1 correlations
        self.plot_semantic_vs_hard_f1_correlations(
            semantic_metrics, hard_metrics
        )

        # Create Figure D4b: Individual emotion F1 correlations
        self.plot_individual_emotion_correlations(individual_emotion_metrics)

    def plot_semantic_vs_hard_f1_correlations(
        self, semantic_metrics: Dict, hard_metrics: Dict
    ):
        """Plot semantic vs hard F1 correlations (Figure D4a)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Semantic F1 correlations
        if semantic_metrics:
            metric_names = []
            correlations = []
            p_values = []

            for metric_name, corr_data in semantic_metrics.items():
                clean_name = (
                    metric_name.replace('test_', '')
                    .replace('_semantic_f1', '')
                    .replace('_f1', '')
                )
                metric_names.append(clean_name.replace('_', '\n'))
                correlations.append(corr_data['correlation'])
                p_values.append(corr_data['p_value'])

            colors = ['red' if p < 0.05 else 'lightcoral' for p in p_values]
            bars = ax1.bar(
                range(len(metric_names)), correlations, color=colors, alpha=0.7
            )

            metric_label = self.get_semantic_metric_label()
            ax1.set_xlabel(f'Semantic {metric_label} Metrics', fontsize=12)
            ax1.set_ylabel(
                'Spearman Correlation with P4G Performance', fontsize=12
            )
            ax1.set_title(
                f'Semantic {metric_label} vs P4G Downstream Performance',
                fontsize=14,
                fontweight='bold',
            )
            ax1.set_xticks(range(len(metric_names)))
            ax1.set_xticklabels(metric_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, zorder=0)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add correlation values on bars
            for i, (bar, corr_val, p_val) in enumerate(
                zip(bars, correlations, p_values)
            ):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (0.02 if height >= 0 else -0.05),
                    f'{corr_val:.3f}' + ('*' if p_val < 0.05 else ''),
                    ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontweight='bold',
                )

        # Plot 2: Hard F1 correlations
        if hard_metrics:
            metric_names = []
            correlations = []
            p_values = []

            for metric_name, corr_data in hard_metrics.items():
                clean_name = metric_name.replace('test_', '').replace('_f1', '')
                metric_names.append(clean_name.replace('_', '\n'))
                correlations.append(corr_data['correlation'])
                p_values.append(corr_data['p_value'])

            colors = ['blue' if p < 0.05 else 'lightblue' for p in p_values]
            bars = ax2.bar(
                range(len(metric_names)), correlations, color=colors, alpha=0.7
            )

            ax2.set_xlabel('Hard F1 Metrics', fontsize=12)
            ax2.set_ylabel(
                'Spearman Correlation with P4G Performance', fontsize=12
            )
            ax2.set_title(
                'Hard F1 vs P4G Downstream Performance',
                fontsize=14,
                fontweight='bold',
            )
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, zorder=0)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add correlation values on bars
            for i, (bar, corr_val, p_val) in enumerate(
                zip(bars, correlations, p_values)
            ):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (0.02 if height >= 0 else -0.05),
                    f'{corr_val:.3f}' + ('*' if p_val < 0.05 else ''),
                    ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontweight='bold',
                )

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d4a_semantic_vs_hard_f1.png'
        save_path_pdf = self.output_dir / 'figure_d4a_semantic_vs_hard_f1.pdf'

        print(f"Saving Figure D4a (Semantic vs Hard F1) to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_individual_emotion_correlations(self, individual_metrics: Dict):
        """Plot individual emotion F1 correlations (Figure D4b)."""
        if not individual_metrics:
            print("No individual emotion metrics found")
            return

        # Sort by absolute correlation value
        sorted_metrics = sorted(
            individual_metrics.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True,
        )

        # Take top 15 emotions for readability
        top_metrics = dict(sorted_metrics[:15])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Extract data
        emotion_names = []
        correlations = []
        p_values = []

        for metric_name, corr_data in top_metrics.items():
            emotion_name = metric_name.replace('test_', '').replace('_f1', '')
            emotion_names.append(emotion_name.title())
            correlations.append(corr_data['correlation'])
            p_values.append(corr_data['p_value'])

        # Plot 1: Correlation values
        colors = ['green' if c > 0 else 'red' for c in correlations]
        sig_colors = [
            c if p < 0.05 else 'lightgray' for c, p in zip(colors, p_values)
        ]

        bars = ax1.barh(
            range(len(emotion_names)), correlations, color=sig_colors, alpha=0.7
        )
        ax1.set_xlabel('Spearman Correlation with P4G Performance', fontsize=12)
        ax1.set_ylabel('Individual Emotion F1 Scores', fontsize=12)
        ax1.set_title(
            'Individual Emotion F1 vs P4G Downstream Performance (Top 15)',
            fontsize=14,
            fontweight='bold',
        )
        ax1.set_yticks(range(len(emotion_names)))
        ax1.set_yticklabels(emotion_names)
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Add correlation values on bars
        for i, (bar, corr_val, p_val) in enumerate(
            zip(bars, correlations, p_values)
        ):
            width = bar.get_width()
            ax1.text(
                width + (0.02 if width >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2.0,
                f'{corr_val:.3f}' + ('*' if p_val < 0.05 else ''),
                ha='left' if width >= 0 else 'right',
                va='center',
                fontweight='bold',
            )

        # Plot 2: P-values
        colors_p = [
            'red' if p < 0.05 else 'orange' if p < 0.1 else 'gray'
            for p in p_values
        ]
        bars2 = ax2.barh(
            range(len(emotion_names)), p_values, color=colors_p, alpha=0.7
        )
        ax2.set_xlabel('P-value', fontsize=12)
        ax2.set_ylabel('Individual Emotion F1 Scores', fontsize=12)
        ax2.set_title(
            'Statistical Significance of Correlations',
            fontsize=14,
            fontweight='bold',
        )
        ax2.set_yticks(range(len(emotion_names)))
        ax2.set_yticklabels(emotion_names)
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.axvline(
            x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05'
        )
        ax2.axvline(
            x=0.1, color='orange', linestyle='--', alpha=0.7, label='p=0.10'
        )
        ax2.legend()

        # Add p-values on bars
        for i, (bar, p_val) in enumerate(zip(bars2, p_values)):
            width = bar.get_width()
            ax2.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                f'{p_val:.3f}',
                ha='left',
                va='center',
                fontweight='bold',
            )

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d4b_individual_emotions.png'
        save_path_pdf = self.output_dir / 'figure_d4b_individual_emotions.pdf'

        print(f"Saving Figure D4b (Individual Emotions) to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_f1_type_correlations(self, f1_type='sample'):
        """Plot dual-axis line plots for a specific F1 type (sample, macro, micro) comparing semantic vs hard"""
        if not self.correlations:
            print(f"No correlations found for {f1_type} F1 type")
            return

        # Map F1 types to their correlation keys
        f1_type_mapping = {
            'sample': {
                'semantic': 'test_semantic_f1',
                'hard': 'test_sample_f1',
            },
            'macro': {
                'semantic': 'test_macro_semantic_f1',
                'hard': 'test_macro_f1',
            },
            'micro': {
                'semantic': 'test_micro_semantic_f1',
                'hard': 'test_micro_f1',
            },
        }

        if f1_type not in f1_type_mapping:
            print(f"Unknown F1 type: {f1_type}")
            return

        semantic_key = f1_type_mapping[f1_type]['semantic']
        hard_key = f1_type_mapping[f1_type]['hard']

        semantic_corr = self.correlations.get(semantic_key)
        hard_corr = self.correlations.get(hard_key)

        if not semantic_corr and not hard_corr:
            print(f"No semantic or hard {f1_type} F1 correlations found")
            return

        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Get P4G performance data across all models and tasks
        p4g_task_names = [
            'persuader_success',
            'persuader_donation',
            'persuadee_donation',
            'donation',
        ]
        model_data = []  # Store all data together for sorting

        if (
            hasattr(self, 'p4g_results')
            and self.p4g_results
            and hasattr(self, 'semeval_results')
            and self.semeval_results
        ):
            for model_name in self.p4g_results.keys():
                if model_name in self.semeval_results:
                    clean_model_name = model_name.replace(
                        'meta-llama--', ''
                    ).replace('-inference-semeval-0-shot_0', '')

                    # Get P4G downstream performance (use persuader_success AUC as representative)
                    p4g_data = self.p4g_results[model_name]
                    if (
                        'persuader_success' in p4g_data
                        and 'downstream_results'
                        in p4g_data['persuader_success']
                    ):
                        # Get best AUC across k values
                        best_auc = 0
                        for k, results in p4g_data['persuader_success'][
                            'downstream_results'
                        ].items():
                            if isinstance(results, dict) and 'auc' in results:
                                if results['auc'] > best_auc:
                                    best_auc = results['auc']
                        downstream_score = best_auc
                    else:
                        downstream_score = 0

                    # Get SemEval F1 scores
                    semeval_data = self.semeval_results[model_name]

                    # Extract semantic F1 score (keep the test_ prefix)
                    semantic_score = semeval_data.get(semantic_key, 0)

                    # Extract hard F1 score (keep the test_ prefix)
                    hard_score = semeval_data.get(hard_key, 0)

                    # Store all data together
                    model_data.append(
                        {
                            'model_name': clean_model_name,
                            'downstream_score': downstream_score,
                            'semantic_f1': semantic_score,
                            'hard_f1': hard_score,
                        }
                    )

        # Sort by downstream performance (descending order), with semantic F1 as tie-breaker
        model_data.sort(
            key=lambda x: (x['downstream_score'], x['semantic_f1']),
            reverse=True,
        )

        # Extract sorted data for plotting
        models = [item['model_name'] for item in model_data]
        downstream_scores = [item['downstream_score'] for item in model_data]
        semantic_f1_scores = [item['semantic_f1'] for item in model_data]
        hard_f1_scores = [item['hard_f1'] for item in model_data]

        if not models:
            print(f"No model data found for {f1_type} dual-axis plot")
            return

        # Create the dual-axis line plot
        x_pos = range(len(models))

        # Plot downstream performance on left axis (line plot)
        line1 = ax1.plot(
            x_pos,
            downstream_scores,
            'o-',
            color='blue',
            linewidth=3,
            markersize=10,
            label='P4G Downstream AUC',
            alpha=0.8,
        )

        ax1.set_ylabel(
            'P4G Downstream Performance (AUC)', fontsize=12, color='blue'
        )
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # Plot SemEval F1 scores on right axis (line plots)
        if semantic_f1_scores and any(s > 0 for s in semantic_f1_scores):
            line2 = ax2.plot(
                x_pos,
                semantic_f1_scores,
                's-',
                color='red',
                linewidth=3,
                markersize=8,
                label=f'{f1_type.title()} Semantic {self.get_semantic_metric_label()}',
                alpha=0.8,
            )

        if hard_f1_scores and any(s > 0 for s in hard_f1_scores):
            line3 = ax2.plot(
                x_pos,
                hard_f1_scores,
                '^-',
                color='orange',
                linewidth=3,
                markersize=8,
                label=f'{f1_type.title()} Hard F1',
                alpha=0.8,
            )

        ax2.set_ylabel('SemEval F1 Score', fontsize=20, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set x-axis
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)

        # Add title
        plt.title(
            f'P4G Downstream Performance vs SemEval {f1_type.title()} F1 Scores',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )

        # Add correlation values as text (positioned in upper left)
        y_pos = 0.98
        if semantic_corr:
            plt.text(
                0.02,
                y_pos,
                f'Semantic {self.get_semantic_metric_label()} Correlation: ρ = {semantic_corr["correlation"]:.3f}, p = {semantic_corr["p_value"]:.3f}',
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )
            y_pos -= 0.08

        if hard_corr:
            plt.text(
                0.02,
                y_pos,
                f'Hard F1 Correlation: ρ = {hard_corr["correlation"]:.3f}, p = {hard_corr["p_value"]:.3f}',
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        # Combine legends and position in upper right to avoid correlation text
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.9,
        )

        plt.tight_layout()

        # Save the figure
        save_path_png = self.output_dir / f'figure_d4_{f1_type}_dual_axis.png'
        save_path_pdf = self.output_dir / f'figure_d4_{f1_type}_dual_axis.pdf'

        print(f"Saving Figure D4 ({f1_type} dual-axis): {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_f1_type_correlations_mse(self, f1_type='sample'):
        """Plot dual-axis line plots for a specific F1 type (sample, macro, micro) comparing semantic vs hard with MSE instead of AUC"""
        if not self.correlations:
            print(f"No correlations found for {f1_type} F1 type (MSE)")
            return

        # Map F1 types to their correlation keys
        f1_type_mapping = {
            'sample': {
                'semantic': 'test_semantic_f1',
                'hard': 'test_sample_f1',
            },
            'macro': {
                'semantic': 'test_macro_semantic_f1',
                'hard': 'test_macro_f1',
            },
            'micro': {
                'semantic': 'test_micro_semantic_f1',
                'hard': 'test_micro_f1',
            },
        }

        if f1_type not in f1_type_mapping:
            print(f"Unknown F1 type: {f1_type}")
            return

        semantic_key = f1_type_mapping[f1_type]['semantic']
        hard_key = f1_type_mapping[f1_type]['hard']

        semantic_corr = self.correlations.get(semantic_key)
        hard_corr = self.correlations.get(hard_key)

        if not semantic_corr and not hard_corr:
            print(f"No semantic or hard {f1_type} F1 correlations found (MSE)")
            return

        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Get P4G performance data across all models and tasks
        model_data = []  # Store all data together for sorting

        if (
            hasattr(self, 'p4g_results')
            and self.p4g_results
            and hasattr(self, 'semeval_results')
            and self.semeval_results
        ):
            for model_name in self.p4g_results.keys():
                if model_name in self.semeval_results:
                    clean_model_name = model_name.replace(
                        'meta-llama--', ''
                    ).replace('-inference-semeval-0-shot_0', '')

                    # Get P4G downstream performance (use persuader_donation RMSE as representative)
                    p4g_data = self.p4g_results[model_name]
                    if (
                        'persuader_donation' in p4g_data
                        and 'downstream_results'
                        in p4g_data['persuader_donation']
                    ):
                        # Get best RMSE across k values, but filter out catastrophic outliers first
                        # (RMSE > 100 is unreasonable for [0,100] donation range)
                        best_rmse = float('inf')
                        valid_found = False
                        for k, results in p4g_data['persuader_donation'][
                            'downstream_results'
                        ].items():
                            if isinstance(results, dict) and 'rmse' in results:
                                # Filter out catastrophic outliers
                                if (
                                    results['rmse'] <= 100
                                    and results['rmse'] < best_rmse
                                ):
                                    best_rmse = results['rmse']
                                    valid_found = True
                        downstream_score = best_rmse if valid_found else 0
                    else:
                        downstream_score = 0

                    # Get SemEval F1 scores
                    semeval_data = self.semeval_results[model_name]

                    # Extract semantic F1 score (keep the test_ prefix)
                    semantic_score = semeval_data.get(semantic_key, 0)

                    # Extract hard F1 score (keep the test_ prefix)
                    hard_score = semeval_data.get(hard_key, 0)

                    # Store all data together
                    model_data.append(
                        {
                            'model_name': clean_model_name,
                            'downstream_score': downstream_score,
                            'semantic_f1': semantic_score,
                            'hard_f1': hard_score,
                        }
                    )

        # Sort by downstream performance (ascending order for MSE - lower is better), with semantic F1 as tie-breaker
        model_data.sort(
            key=lambda x: (x['downstream_score'], -x['semantic_f1'])
        )

        # Extract sorted data for plotting
        models = [item['model_name'] for item in model_data]
        downstream_scores = [item['downstream_score'] for item in model_data]
        semantic_f1_scores = [item['semantic_f1'] for item in model_data]
        hard_f1_scores = [item['hard_f1'] for item in model_data]

        if not models:
            print(f"No model data found for {f1_type} dual-axis plot (MSE)")
            return

        # Create the dual-axis line plot
        x_pos = range(len(models))

        # Plot downstream performance on left axis (line plot) - MSE
        line1 = ax1.plot(
            x_pos,
            downstream_scores,
            'o-',
            color='blue',
            linewidth=3,
            markersize=10,
            label='P4G Downstream RMSE',
            alpha=0.8,
        )

        ax1.set_ylabel(
            'P4G Downstream Performance (RMSE)', fontsize=12, color='blue'
        )
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # Plot SemEval F1 scores on right axis (line plots)
        if semantic_f1_scores and any(s > 0 for s in semantic_f1_scores):
            line2 = ax2.plot(
                x_pos,
                semantic_f1_scores,
                's-',
                color='red',
                linewidth=3,
                markersize=8,
                label=f'{f1_type.title()} Semantic {self.get_semantic_metric_label()}',
                alpha=0.8,
            )

        if hard_f1_scores and any(s > 0 for s in hard_f1_scores):
            line3 = ax2.plot(
                x_pos,
                hard_f1_scores,
                '^-',
                color='orange',
                linewidth=3,
                markersize=8,
                label=f'{f1_type.title()} Hard F1',
                alpha=0.8,
            )

        ax2.set_ylabel('SemEval F1 Score', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set x-axis
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)

        # Add title
        plt.title(
            f'P4G Downstream Performance (RMSE) vs SemEval {f1_type.title()} F1 Scores',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )

        # Add correlation values as text (positioned in upper left)
        y_pos = 0.98
        if semantic_corr:
            plt.text(
                0.02,
                y_pos,
                f'Semantic {self.get_semantic_metric_label()} Correlation: ρ = {semantic_corr["correlation"]:.3f}, p = {semantic_corr["p_value"]:.3f}',
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )
            y_pos -= 0.08

        if hard_corr:
            plt.text(
                0.02,
                y_pos,
                f'Hard F1 Correlation: ρ = {hard_corr["correlation"]:.3f}, p = {hard_corr["p_value"]:.3f}',
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        # Combine legends and position in upper right to avoid correlation text
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.9,
        )

        plt.tight_layout()

        # Save the figure with MSE suffix
        save_path_png = (
            self.output_dir / f'figure_d4_{f1_type}_dual_axis_rmse.png'
        )
        save_path_pdf = (
            self.output_dir / f'figure_d4_{f1_type}_dual_axis_rmse.pdf'
        )

        print(f"Saving Figure D4 ({f1_type} dual-axis RMSE): {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_f1_type_correlations_with_ci(
        self, f1_type='sample', metric_type='auc'
    ):
        """Plot dual-axis line plots with confidence intervals for a specific F1 type"""
        if not self.correlations:
            print(
                f"No correlations found for {f1_type} F1 type with CI ({metric_type})"
            )
            return

        # Map F1 types to their correlation keys
        f1_type_mapping = {
            'sample': {
                'semantic': 'test_semantic_f1',
                'hard': 'test_sample_f1',
            },
            'macro': {
                'semantic': 'test_macro_semantic_f1',
                'hard': 'test_macro_f1',
            },
            'micro': {
                'semantic': 'test_micro_semantic_f1',
                'hard': 'test_micro_f1',
            },
        }

        if f1_type not in f1_type_mapping:
            print(f"Unknown F1 type: {f1_type}")
            return

        semantic_key = f1_type_mapping[f1_type]['semantic']
        hard_key = f1_type_mapping[f1_type]['hard']

        semantic_corr = self.correlations.get(semantic_key)
        hard_corr = self.correlations.get(hard_key)

        if not semantic_corr and not hard_corr:
            print(
                f"No semantic or hard {f1_type} F1 correlations found with CI ({metric_type})"
            )
            return

        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Get P4G performance data across all models and tasks
        model_data = []  # Store all data together for sorting

        if (
            hasattr(self, 'p4g_results')
            and self.p4g_results
            and hasattr(self, 'semeval_results')
            and self.semeval_results
        ):
            for model_name in self.p4g_results.keys():
                if model_name in self.semeval_results:
                    clean_model_name = model_name.replace(
                        'meta-llama--', ''
                    ).replace('-inference-semeval-0-shot_0', '')

                    # Get P4G downstream performance
                    p4g_data = self.p4g_results[model_name]

                    if metric_type == 'auc':
                        # Use classification task for AUC
                        task_key = 'persuader_success'
                    else:  # rmse
                        # Use regression task for RMSE
                        task_key = 'persuader_donation'

                    if (
                        task_key in p4g_data
                        and 'downstream_results' in p4g_data[task_key]
                    ):
                        if metric_type == 'auc':
                            # Get best AUC across k values
                            best_score = 0
                            best_ci = 0
                            for k, results in p4g_data[task_key][
                                'downstream_results'
                            ].items():
                                if (
                                    isinstance(results, dict)
                                    and 'auc' in results
                                ):
                                    if results['auc'] > best_score:
                                        best_score = results['auc']
                                        best_ci = results.get('auc_ci', 0)
                        else:  # rmse
                            # Get best RMSE across k values (lowest RMSE is best)
                            best_score = float('inf')
                            best_ci = 0
                            for k, results in p4g_data[task_key][
                                'downstream_results'
                            ].items():
                                if (
                                    isinstance(results, dict)
                                    and 'rmse' in results
                                ):
                                    if results['rmse'] < best_score:
                                        best_score = results['rmse']
                                        best_ci = results.get('rmse_ci', 0)
                            best_score = (
                                best_score if best_score != float('inf') else 0
                            )

                        downstream_score = best_score
                        downstream_ci = best_ci
                    else:
                        downstream_score = 0
                        downstream_ci = 0

                    # Get SemEval F1 scores
                    semeval_data = self.semeval_results[model_name]

                    # Extract semantic F1 score (keep the test_ prefix)
                    semantic_score = semeval_data.get(semantic_key, 0)

                    # Extract hard F1 score (keep the test_ prefix)
                    hard_score = semeval_data.get(hard_key, 0)

                    # Store all data together
                    model_data.append(
                        {
                            'model_name': clean_model_name,
                            'downstream_score': downstream_score,
                            'downstream_ci': downstream_ci,
                            'semantic_f1': semantic_score,
                            'hard_f1': hard_score,
                        }
                    )

        # Sort by downstream performance, with semantic F1 as tie-breaker
        if metric_type == 'auc':
            model_data.sort(
                key=lambda x: (x['downstream_score'], x['semantic_f1']),
                reverse=True,
            )
        else:  # rmse - lower is better
            model_data.sort(
                key=lambda x: (x['downstream_score'], -x['semantic_f1'])
            )

        # Extract sorted data for plotting
        models = [item['model_name'] for item in model_data]
        downstream_scores = [item['downstream_score'] for item in model_data]
        downstream_cis = [item['downstream_ci'] for item in model_data]
        semantic_f1_scores = [item['semantic_f1'] for item in model_data]
        hard_f1_scores = [item['hard_f1'] for item in model_data]

        if not models:
            print(
                f"No model data found for {f1_type} dual-axis plot with CI ({metric_type})"
            )
            return

        # Create the dual-axis line plot with error bars
        x_pos = range(len(models))

        # Plot downstream performance on left axis with confidence intervals
        line1 = ax1.errorbar(
            x_pos,
            downstream_scores,
            yerr=downstream_cis,
            fmt='o-',
            color=THIRD_COLOR,
            linewidth=3,
            markersize=10,
            capsize=5,
            capthick=2,
            label=f'P4G Downstream {metric_type.upper()}',
            alpha=0.8,
        )

        metric_label = 'AUC' if metric_type == 'auc' else 'RMSE'

        ax1.set_ylabel(
            metric_label,
            fontsize=22,
            color=THIRD_COLOR,
        )
        ax1.tick_params(axis='y', labelcolor=THIRD_COLOR)
        ax1.grid(True, alpha=0.3)

        # Plot SemEval F1 scores on right axis (line plots)
        if semantic_f1_scores and any(s > 0 for s in semantic_f1_scores):
            line2 = ax2.plot(
                x_pos,
                semantic_f1_scores,
                's-',
                color=SEM_COLOR,
                linewidth=3,
                markersize=8,
                label=f'Semantic {self.get_semantic_metric_label()} (ρ = {semantic_corr["correlation"]:.3f})',
                alpha=0.8,
            )

        if hard_f1_scores and any(s > 0 for s in hard_f1_scores):
            line3 = ax2.plot(
                x_pos,
                hard_f1_scores,
                '^-',
                color=HARD_COLOR,
                linewidth=3,
                markersize=8,
                label=f'Hard F1 (ρ = {hard_corr["correlation"]:.3f})',
                alpha=0.8,
            )

        ax2.set_ylabel('F1 Score', fontsize=16, labelpad=10)
        ax2.tick_params(axis='y')

        # Set x-axis
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=18)

        # Add title
        plt.title(
            f'P4G {metric_label} vs SemEval {f1_type.title()} F1',
            fontsize=26,
            fontweight='bold',
            pad=10,
        )

        # Combine legends and position in upper right to avoid correlation text
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='center right',
            # bbox_to_anchor=(0.98, 0.98),
            framealpha=0.9,
            fontsize=18,
        )

        plt.tight_layout()

        # Save the figure with CI suffix
        save_path_png = (
            self.output_dir
            / f'figure_d4_{f1_type}_dual_axis_{metric_type}_ci.png'
        )
        save_path_pdf = (
            self.output_dir
            / f'figure_d4_{f1_type}_dual_axis_{metric_type}_ci.pdf'
        )

        print(
            f"Saving Figure D4 ({f1_type} dual-axis {metric_type.upper()} with CI): {save_path_png}"
        )
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_detailed_correlations(self):
        """Plot detailed scatter plots for significant correlations."""
        if not self.correlations:
            print("No correlations found to plot")
            return

        # Filter to significant correlations (p < 0.1 for visualization)
        significant_correlations = {
            k: v for k, v in self.correlations.items() if v['p_value'] < 0.1
        }

        if not significant_correlations:
            print("No significant correlations found")
            return

        n_correlations = len(significant_correlations)
        cols = min(3, n_correlations)
        rows = (n_correlations + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_correlations == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, (metric_name, corr_data) in enumerate(
            significant_correlations.items()
        ):
            ax = axes[i]

            data = corr_data['data']
            models = data['models']
            semeval_scores = data['semeval_scores']
            p4g_scores = data['p4g_scores']

            # Create scatter plot
            ax.scatter(
                semeval_scores,
                p4g_scores,
                s=150,
                alpha=0.7,
                color='steelblue',
            )

            # Add model labels
            model_labels = [self.model_labels.get(m, m) for m in models]
            for j, (x, y, label) in enumerate(
                zip(semeval_scores, p4g_scores, model_labels)
            ):
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                )

            # Add trend line
            if len(semeval_scores) > 1:
                z = np.polyfit(semeval_scores, p4g_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    min(semeval_scores), max(semeval_scores), 100
                )
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(
                f'{metric_name.replace("_", " ").title()} F1 Score', fontsize=11
            )
            ax.set_ylabel('Best P4G Downstream Performance', fontsize=11)
            ax.set_title(
                f'{metric_name.replace("_", " ").title()}\n'
                f'ρ = {corr_data["correlation"]:.3f}, p = {corr_data["p_value"]:.3f}',
                fontsize=12,
                fontweight='bold',
            )
            ax.grid(True, alpha=0.3, zorder=0)

        # Hide unused subplots
        for i in range(n_correlations, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d2_detailed_correlations.png'
        save_path_pdf = self.output_dir / 'figure_d2_detailed_correlations.pdf'

        print(f"Saving detailed correlations to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_model_performance_comparison(self):
        """Plot model performance across P4G tasks and SemEval metrics."""
        df = self.prepare_data_for_plotting()

        if df.empty:
            print("No data available for model performance comparison")
            return

        # Focus on main P4G tasks
        focus_tasks = ['persuader_success', 'persuadee_donation']
        df_filtered = df[df['task'].isin(focus_tasks)]

        if df_filtered.empty:
            print("No data available for focus tasks")
            return

        # Get unique models and SemEval metrics
        models = df_filtered['model'].unique()
        semeval_metrics = df_filtered['semeval_metric'].unique()

        # Create subplot for each SemEval metric
        n_metrics = len(semeval_metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, semeval_metric in enumerate(semeval_metrics):
            ax = axes[i]
            metric_data = df_filtered[
                df_filtered['semeval_metric'] == semeval_metric
            ]

            if metric_data.empty:
                continue

            # Create grouped bar plot
            x_pos = np.arange(len(models))
            width = 0.35

            # Group by task
            tasks = metric_data['task'].unique()
            colors = ['steelblue', 'orange', 'green', 'red']

            for j, task in enumerate(tasks):
                task_data = metric_data[metric_data['task'] == task]

                # Get scores for each model
                scores = []
                score_cis = []
                for model in models:
                    model_data = task_data[task_data['model'] == model]
                    if not model_data.empty:
                        scores.append(model_data['best_score'].iloc[0])
                        score_cis.append(model_data['best_score_ci'].iloc[0])
                    else:
                        scores.append(0)
                        score_cis.append(0)

                # Plot bars
                ax.bar(
                    x_pos + j * width,
                    scores,
                    width,
                    label=task.replace('_', ' ').title(),
                    color=colors[j % len(colors)],
                    alpha=0.7,
                    yerr=score_cis,
                    capsize=3,
                )

            ax.set_ylabel('Performance Score', fontsize=12)
            ax.set_title(
                f'P4G Performance vs {semeval_metric.replace("_", " ").title()}',
                fontsize=13,
                fontweight='bold',
            )
            ax.set_xticks(x_pos + width / 2)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, zorder=0)

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d3_model_performance.png'
        save_path_pdf = self.output_dir / 'figure_d3_model_performance.pdf'

        print(f"Saving model performance comparison to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def plot_dual_axis_comparison(self):
        """Plot dual-axis comparison showing both SemEval and P4G performance."""
        df = self.prepare_data_for_plotting()

        if df.empty:
            print("No data available for dual-axis comparison")
            return

        # Focus on persuader_success task and aggregate across SemEval metrics
        task_data = df[df['task'] == 'persuader_success'].copy()
        if task_data.empty:
            print("No data available for persuader_success task")
            return

        # Aggregate SemEval scores (take mean across all F1 metrics)
        model_summary = (
            task_data.groupby(['model', 'model_full'])
            .agg(
                {
                    'best_score': 'first',
                    'best_score_ci': 'first',
                    'semeval_score': 'mean',  # Average across all F1 metrics
                }
            )
            .reset_index()
        )

        # Sort by P4G performance
        model_summary = model_summary.sort_values('best_score', ascending=False)

        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        x_pos = np.arange(len(model_summary))

        # Plot P4G performance (left axis)
        ax1.errorbar(
            x_pos,
            model_summary['best_score'],
            yerr=model_summary['best_score_ci'],
            fmt='o-',
            color='steelblue',
            linewidth=3,
            markersize=8,
            capsize=5,
            label='P4G AUC',
        )

        ax1.set_xlabel('Model (ranked by P4G performance)', fontsize=12)
        ax1.set_ylabel('P4G AUC', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_summary['model'], rotation=45)

        # Plot SemEval performance (right axis)
        ax2 = ax1.twinx()
        ax2.plot(
            x_pos,
            model_summary['semeval_score'],
            's-',
            color='red',
            linewidth=3,
            markersize=8,
            label='SemEval Avg F1',
        )

        ax2.set_ylabel('SemEval Average F1', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.title(
            'P4G Persuader Success vs SemEval 2-shot Performance',
            fontsize=16,
            fontweight='bold',
        )
        ax1.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        save_path_png = self.output_dir / 'figure_d4_dual_axis.png'
        save_path_pdf = self.output_dir / 'figure_d4_dual_axis.pdf'

        print(f"Saving dual-axis comparison to: {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()

    def create_summary_table(self):
        """Create summary table of results."""
        # Create correlation summary
        if self.correlations:
            corr_df = pd.DataFrame(
                [
                    {
                        'SemEval_Metric': metric_name,
                        'Correlation': f"{corr_data['correlation']:.3f}",
                        'P_Value': f"{corr_data['p_value']:.3f}",
                        'Significant': (
                            'Yes' if corr_data['p_value'] < 0.05 else 'No'
                        ),
                        'N_Models': corr_data['n_models'],
                    }
                    for metric_name, corr_data in self.correlations.items()
                ]
            )

            # Save correlation table
            corr_path = self.output_dir / 'correlation_summary.csv'
            corr_df.to_csv(corr_path, index=False)
            print(f"Correlation summary saved to: {corr_path}")

        # Create model performance summary
        df = self.prepare_data_for_plotting()
        if not df.empty:
            # Aggregate by model and task
            model_summary = (
                df.groupby(['model', 'task'])
                .agg(
                    {
                        'best_score': 'first',
                        'best_score_ci': 'first',
                        'performance_metric': 'first',
                        'best_k': 'first',
                    }
                )
                .reset_index()
            )

            # Pivot to get tasks as columns
            performance_table = model_summary.pivot(
                index='model',
                columns='task',
                values=['best_score', 'best_score_ci', 'best_k'],
            )

            # Save performance table
            perf_path = self.output_dir / 'model_performance_summary.csv'
            performance_table.to_csv(perf_path)
            print(f"Model performance summary saved to: {perf_path}")

    def create_all_visualizations(self):
        """Create all visualization types."""
        print("Creating Study D P4G-SemEval visualizations...")

        print("Creating correlation overview...")
        try:
            self.plot_correlation_overview()
        except Exception as e:
            print(f"Error creating correlation overview: {e}")

        print("Creating detailed correlation plots...")
        try:
            self.plot_detailed_correlations()
        except Exception as e:
            print(f"Error creating detailed correlations: {e}")

        print("Creating model performance comparison...")
        try:
            self.plot_model_performance_comparison()
        except Exception as e:
            print(f"Error creating model performance comparison: {e}")

        print("Creating dual-axis comparison...")
        try:
            self.plot_dual_axis_comparison()
        except Exception as e:
            print(f"Error creating dual-axis comparison: {e}")

        print("Creating correlation summary (Figure D4)...")
        try:
            self.plot_correlation_summary()
        except Exception as e:
            print(f"Error creating correlation summary: {e}")

        print("Creating F1 type-specific correlations...")
        try:
            self.plot_f1_type_correlations('sample')
            self.plot_f1_type_correlations('macro')
            self.plot_f1_type_correlations('micro')
        except Exception as e:
            print(f"Error creating F1 type correlations: {e}")

        print("Creating F1 type-specific correlations with RMSE...")
        try:
            self.plot_f1_type_correlations_mse('sample')
            self.plot_f1_type_correlations_mse('macro')
            self.plot_f1_type_correlations_mse('micro')
        except Exception as e:
            print(f"Error creating F1 type correlations (RMSE): {e}")

        print(
            "Creating F1 type-specific correlations with confidence intervals (AUC)..."
        )
        try:
            self.plot_f1_type_correlations_with_ci('sample', 'auc')
            self.plot_f1_type_correlations_with_ci('macro', 'auc')
            self.plot_f1_type_correlations_with_ci('micro', 'auc')
        except Exception as e:
            print(f"Error creating F1 type correlations with CI (AUC): {e}")

        print(
            "Creating F1 type-specific correlations with confidence intervals (RMSE)..."
        )
        try:
            self.plot_f1_type_correlations_with_ci('sample', 'rmse')
            self.plot_f1_type_correlations_with_ci('macro', 'rmse')
            self.plot_f1_type_correlations_with_ci('micro', 'rmse')
        except Exception as e:
            print(f"Error creating F1 type correlations with CI (RMSE): {e}")

        print("Creating summary tables...")
        try:
            self.create_summary_table()
        except Exception as e:
            print(f"Error creating summary tables: {e}")

        print(f"\nAll visualizations saved to {self.output_dir}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        print(f"Models analyzed: {len(self.p4g_results)}")
        print(f"Models failed: {len(self.failed_models)}")
        print(f"Correlations found: {len(self.correlations)}")

        if self.correlations:
            print("\nCorrelation Summary:")
            for metric_name, corr_data in self.correlations.items():
                significance = (
                    "***"
                    if corr_data['p_value'] < 0.001
                    else (
                        "**"
                        if corr_data['p_value'] < 0.01
                        else "*" if corr_data['p_value'] < 0.05 else ""
                    )
                )
                print(
                    f"  {metric_name}: ρ = {corr_data['correlation']:.3f}, "
                    f"p = {corr_data['p_value']:.3f}{significance}"
                )

        if self.failed_models:
            print(f"\nFailed Models ({len(self.failed_models)}):")
            for model_name, reason in self.failed_models:
                print(f"  ✗ {model_name}: {reason}")


def main():
    parser = argparse.ArgumentParser(
        description='Study D P4G-SemEval Visualization'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='Path to Study D P4G-SemEval results pickle file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for figures',
    )

    args = parser.parse_args()

    visualizer = StudyDP4GVisualizor(args.results_path, args.output_dir)
    visualizer.create_all_visualizations()


if __name__ == '__main__':
    main()
