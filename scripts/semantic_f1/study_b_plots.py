#!/usr/bin/env python3
"""
Study B Plotting Script: Real Data (Convergent Validity)

This script creates figures for Study B as described in PROJECT.md:
- Tab B1: Model ranks by Hard vs Semantic F1
- Fig B1: Average score in controls vs subjective multilabel for all models
- Fig B2: Rank-correlation between capability ranking and each metric
- Tab B2: Macro/micro/sample numbers with bootstrap CIs

Note: IAA analysis (Fig B3) is not implemented yet as IAA has not been studied.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from semantic_f1 import HARD_COLOR, SEM_COLOR, THIRD_COLOR


class StudyBAnalyzer:
    def __init__(
        self,
        logs_dir,
        correlation_method="spearman",
    ):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path("logs/analysis/semantic_f1/study_b")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Define dataset categories based on PROJECT.md
        self.subjective_datasets = ['GoEmotions', 'SemEval', 'MFRC']
        self.control_datasets = ['MovieLens', 'Boxes', 'TREC']

        self.data = defaultdict(dict)
        self.shots = defaultdict(dict)
        self.model_names = set()
        allowed_methods = {'spearman', 'pearson', 'ccc', 'icc'}
        method = correlation_method.lower()
        if method not in allowed_methods:
            method = 'spearman'

        self.correlation_method = method

    @property
    def correlation_symbol(self):
        symbol_map = {
            'spearman': 'ρ',
            'pearson': 'r',
            'ccc': 'CCC',
            'icc': 'ICC',
        }
        return symbol_map.get(self.correlation_method, 'ρ')

    @property
    def correlation_display_name(self):
        display_map = {
            'spearman': 'Spearman',
            'pearson': 'Pearson',
            'ccc': 'Concordance Correlation Coefficient (CCC)',
            'icc': 'Intraclass Correlation (ICC 3,1)',
        }
        return display_map.get(self.correlation_method, 'Spearman')

    @property
    def correlation_min_points(self):
        return 3 if self.correlation_method == 'spearman' else 2

    @property
    def correlation_axis_label(self):
        axis_map = {
            'spearman': 'Spearman Correlation with Capability',
            'pearson': 'Pearson Correlation with Capability',
            'ccc': 'Concordance Correlation Coefficient (CCC) with Capability',
            'icc': 'Intraclass Correlation (ICC 3,1) with Capability',
        }
        return axis_map.get(
            self.correlation_method, 'Spearman Correlation with Capability'
        )

    @property
    def correlation_title(self):
        title_map = {
            'spearman': 'Spearman Correlation: Capability vs Performance Metrics',
            'pearson': 'Pearson Correlation: Capability vs Performance Metrics',
            'ccc': 'Concordance Correlation (CCC): Capability vs Performance Metrics',
            'icc': 'Intraclass Correlation (ICC 3,1): Capability vs Performance Metrics',
        }
        return title_map.get(
            self.correlation_method,
            'Spearman Correlation: Capability vs Performance Metrics',
        )

    def compute_correlation(self, x_vals, y_vals):
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)

        if x.size != y.size:
            return None

        if (
            x.size < self.correlation_min_points
            or y.size < self.correlation_min_points
        ):
            return None

        if self.correlation_method == 'pearson':
            corr, _ = pearsonr(x, y)
        elif self.correlation_method == 'spearman':
            corr, _ = spearmanr(x, y)
        elif self.correlation_method == 'ccc':
            corr = self.compute_ccc(x, y)
        elif self.correlation_method == 'icc':
            corr = self.compute_icc(x, y)
        else:
            corr, _ = spearmanr(x, y)

        if corr is None or np.isnan(corr):
            return None
        return corr

    def compute_ccc(self, x, y):
        if np.allclose(x, y):
            return 1.0

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        cov_xy = np.cov(x, y, ddof=1)[0, 1]

        denominator = var_x + var_y + (mean_x - mean_y) ** 2
        if denominator <= 0:
            return None

        return (2 * cov_xy) / denominator

    def compute_icc(self, x, y):
        data = np.vstack([x, y]).T  # subjects x raters
        n, k = data.shape
        if n < 2:
            return None

        grand_mean = np.mean(data)
        mean_subjects = np.mean(data, axis=1)
        mean_raters = np.mean(data, axis=0)

        ss_subjects = k * np.sum((mean_subjects - grand_mean) ** 2)
        ss_raters = n * np.sum((mean_raters - grand_mean) ** 2)
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_error = ss_total - ss_subjects - ss_raters

        df_subjects = n - 1
        df_raters = k - 1
        df_error = df_subjects * df_raters

        if df_subjects <= 0 or df_error <= 0:
            return None

        ms_subjects = ss_subjects / df_subjects
        ms_error = ss_error / df_error

        ms_error = max(ms_error, 0.0)

        denominator = ms_subjects + (k - 1) * ms_error
        if denominator <= 0:
            return None

        return (ms_subjects - ms_error) / denominator

    def extract_model_name(self, folder_name):
        """Extract clean model name from folder name"""
        # Handle different model naming patterns
        if 'gpt-' in folder_name:
            # Extract from patterns like gpt-4o-mini-inference-25-shot_0
            parts = folder_name.split('-')
            return f"{parts[0]}-{parts[1]}-{parts[2]}"  # e.g., gpt-4o-mini
        elif 'meta-llama' in folder_name:
            # Extract from patterns like meta-llama--Llama-3.1-8B-Instruct-inference-25-shot_0
            model_part = folder_name.split('--')[1]  # Get part after '--'
            if 'Llama-2-7b' in model_part:
                # meta-llama--Llama-2-7b-chat-hf-inference-25-shot_0 -> Llama-2-7b-chat
                return 'Llama-2-7b-chat'
            elif 'Llama-2-70b' in model_part:
                return 'Llama-2-70b-chat'
            elif 'Llama-3.1' in model_part:
                # meta-llama--Llama-3.1-8B-Instruct-inference-25-shot_0 -> Llama-3.1-8B
                return 'Llama-3.1-8B'
            elif 'Llama-3.3' in model_part:
                return 'Llama-3.3-70B'
        return folder_name

    def extract_shot(self, folder_name):
        """Extract shot number from folder name"""
        match = re.search(r'(\d+)-shot', folder_name)
        if match:
            return int(match.group(1))
        return 0  # Default to 0-shot if not found

    def load_results(self):
        """Load all results from logs directory"""
        print("Loading results...")

        for dataset_dir in self.logs_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name == 'analysis':
                continue

            dataset_name = dataset_dir.name.replace(
                'OpenAI', ''
            )  # Handle OpenAI suffix

            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = self.extract_model_name(model_dir.name)
                shot = self.extract_shot(model_dir.name)

                if self.shots[dataset_name].get(model_name, -1) > shot:
                    continue  # Skip if we already have a higher shot count

                self.model_names.add(model_name)

                metrics_file = model_dir / "aggregated_metrics.yml"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = yaml.safe_load(f)

                    # Extract the actual metrics (skip the empty key)
                    if '' in metrics:
                        metrics = metrics['']

                    # Store metrics - combine results from OpenAI and regular datasets
                    if dataset_name not in self.data:
                        self.data[dataset_name] = {}

                    # If model already exists, we could average or keep the latest
                    # For now, we'll keep the latest (overwrite)
                    self.data[dataset_name][model_name] = metrics
                    self.shots[dataset_name][model_name] = shot

        print(
            f"Loaded results for {len(self.data)} datasets and {len(self.model_names)} models"
        )
        print(f"Datasets: {list(self.data.keys())}")
        print(f"Models: {sorted(self.model_names)}")

        # Debug: print some sample data
        for dataset in list(self.data.keys())[:2]:
            print(f"\nSample data for {dataset}:")
            for model in list(self.data[dataset].keys())[:2]:
                metrics = self.data[dataset][model]
                sample_metrics = {k: v for k, v in list(metrics.items())[:5]}
                print(f"  {model}: {sample_metrics}")

    def extract_metric_value(self, metrics, metric_name):
        """Extract metric value from string like '0.3271+-0.0366'"""
        if metric_name not in metrics:
            return None, None

        value_str = str(metrics[metric_name])
        if '+-' in value_str:
            mean, std = value_str.split('+-')
            return float(mean), float(std)
        else:
            return float(value_str), 0.0

    def get_model_order(self):
        """Get models in chronological order (older to newer, smaller to larger within generation)"""
        model_order = [
            'Llama-2-7b-chat',  # 2023, 7B
            'Llama-2-70b-chat',  # 2023, 70B
            'Llama-3.1-8B',  # 2024, 8B
            'Llama-3.3-70B',  # 2024, 70B
            'gpt-4o-mini',  # 2024, mini
            'gpt-4.1-mini',  # 2024, small
        ]
        # Return only models that actually exist in our data
        return [model for model in model_order if model in self.model_names]

    def get_model_capability_ranking(self):
        """Create capability ranking based on control datasets (MMLU-Pro and TREC)"""
        print("Computing model capability ranking...")

        capability_scores = defaultdict(list)

        # Use control datasets for capability ranking
        for dataset in ['MMLUPro', 'TREC']:
            if dataset in self.data:
                for model in self.model_names:
                    if model in self.data[dataset]:
                        metrics = self.data[dataset][model]
                        # Use macro F1 for capability assessment
                        macro_f1, _ = self.extract_metric_value(
                            metrics, 'test_macro_f1'
                        )
                        if macro_f1 is not None:
                            capability_scores[model].append(macro_f1)

        # Average across control datasets
        model_capabilities = {}
        for model, scores in capability_scores.items():
            if scores:
                model_capabilities[model] = np.mean(scores)

        # Rank models by capability (higher is better)
        ranked_models = sorted(
            model_capabilities.items(), key=lambda x: x[1], reverse=True
        )
        capability_ranking = {
            model: rank for rank, (model, _) in enumerate(ranked_models)
        }

        print(f"Capability ranking: {ranked_models}")
        return capability_ranking, model_capabilities

    def create_table_b1(self):
        """Create Table B1: Model ranks by Hard vs Semantic F1"""
        print("Creating Table B1...")

        results = []

        for dataset in self.subjective_datasets + self.control_datasets:
            if dataset not in self.data:
                continue

            # Get model scores for this dataset
            model_scores = {}
            for model in self.model_names:
                if model in self.data[dataset]:
                    metrics = self.data[dataset][model]

                    # Extract different F1 metrics
                    macro_f1, _ = self.extract_metric_value(
                        metrics, 'test_macro_f1'
                    )
                    micro_f1, _ = self.extract_metric_value(
                        metrics, 'test_micro_f1'
                    )
                    sample_f1, _ = self.extract_metric_value(
                        metrics, 'test_sample_f1'
                    )

                    # Semantic F1 variants
                    macro_sem_f1, _ = self.extract_metric_value(
                        metrics, 'test_macro_semantic_f1'
                    )
                    micro_sem_f1, _ = self.extract_metric_value(
                        metrics, 'test_micro_semantic_f1'
                    )
                    sample_sem_f1, _ = self.extract_metric_value(
                        metrics, 'test_semantic_f1'
                    )  # samples semantic is just 'semantic_f1'

                    model_scores[model] = {
                        'macro_f1': macro_f1,
                        'micro_f1': micro_f1,
                        'sample_f1': sample_f1,
                        'macro_sem_f1': macro_sem_f1,
                        'micro_sem_f1': micro_sem_f1,
                        'sample_sem_f1': sample_sem_f1,
                    }

            # Rank models for each metric (higher score = better rank)
            for metric_type in ['macro', 'micro', 'sample']:
                if all(
                    model_scores[m][f'{metric_type}_f1'] is not None
                    for m in model_scores
                ):
                    hard_ranking = sorted(
                        model_scores.items(),
                        key=lambda x: x[1][f'{metric_type}_f1'],
                        reverse=True,
                    )

                    # Check if semantic F1 exists for this dataset
                    sem_scores = {}
                    for m in model_scores:
                        if metric_type == 'sample':
                            sem_f1 = model_scores[m]['sample_sem_f1']
                        else:
                            sem_f1 = model_scores[m][f'{metric_type}_sem_f1']
                        if sem_f1 is not None:
                            sem_scores[m] = sem_f1

                    if sem_scores:
                        sem_ranking = sorted(
                            sem_scores.items(), key=lambda x: x[1], reverse=True
                        )

                        # Create ranking comparison
                        hard_ranks = {
                            model: rank
                            for rank, (model, _) in enumerate(hard_ranking)
                        }
                        sem_ranks = {
                            model: rank
                            for rank, (model, _) in enumerate(sem_ranking)
                        }

                        for model in model_scores:
                            sem_rank = sem_ranks.get(model, 'N/A')
                            rank_diff = (
                                hard_ranks[model]
                                - sem_ranks.get(model, hard_ranks[model])
                                if sem_rank != 'N/A'
                                else 0
                            )

                            results.append(
                                {
                                    'Dataset': dataset,
                                    'Model': model,
                                    'Metric_Type': metric_type,
                                    'Hard_F1_Rank': hard_ranks[model]
                                    + 1,  # 1-indexed
                                    'Semantic_F1_Rank': (
                                        sem_rank + 1
                                        if sem_rank != 'N/A'
                                        else 'N/A'
                                    ),
                                    'Rank_Difference': rank_diff,
                                }
                            )

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "table_b1_model_rankings.csv", index=False)

        # Create a pivot table for better visualization
        if not df.empty:
            pivot_df = df.pivot_table(
                index=['Dataset', 'Model'],
                columns='Metric_Type',
                values=['Hard_F1_Rank', 'Semantic_F1_Rank'],
                aggfunc='first',
            )
            pivot_df.to_csv(self.output_dir / "table_b1_pivot.csv")

        print(f"Table B1 saved to {self.output_dir}")
        return df

    def create_figure_b1(self):
        """Create Figure B1: Average score comparison across model capabilities"""
        print("Creating Figure B1...")

        # Calculate average scores per model across dataset types
        model_scores = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for model in self.model_names:
            # Calculate averages for subjective datasets
            for dataset in self.subjective_datasets:
                if dataset in self.data and model in self.data[dataset]:
                    metrics = self.data[dataset][model]
                    for metric_type in ['macro', 'micro', 'sample']:
                        hard_f1, _ = self.extract_metric_value(
                            metrics, f'test_{metric_type}_f1'
                        )
                        if metric_type == 'sample':
                            sem_f1, _ = self.extract_metric_value(
                                metrics, 'test_semantic_f1'
                            )
                        else:
                            sem_f1, _ = self.extract_metric_value(
                                metrics, f'test_{metric_type}_semantic_f1'
                            )

                        if hard_f1 is not None:
                            model_scores[model]['subjective'][
                                f'{metric_type}_hard'
                            ].append(hard_f1)
                        if sem_f1 is not None:
                            model_scores[model]['subjective'][
                                f'{metric_type}_sem'
                            ].append(sem_f1)

            # Calculate averages for control datasets
            for dataset in self.control_datasets:
                if dataset in self.data and model in self.data[dataset]:
                    metrics = self.data[dataset][model]
                    for metric_type in ['macro', 'micro', 'sample']:
                        hard_f1, _ = self.extract_metric_value(
                            metrics, f'test_{metric_type}_f1'
                        )
                        if hard_f1 is not None:
                            model_scores[model]['control'][
                                f'{metric_type}_hard'
                            ].append(hard_f1)

        # Average the scores
        model_averages = defaultdict(lambda: defaultdict(dict))
        for model in model_scores:
            for dataset_type in ['subjective', 'control']:
                for metric_key, scores in model_scores[model][
                    dataset_type
                ].items():
                    if scores:
                        model_averages[model][dataset_type][metric_key] = (
                            np.mean(scores)
                        )

        # Create separate plots for each metric type
        for metric_type in ['macro', 'micro', 'sample']:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data for plotting - only models that have both control and subjective data
            models_with_data = []
            control_performance = []
            subjective_hard = []
            subjective_sem = []

            for model in self.model_names:
                control_score = model_averages[model]['control'].get(
                    f'{metric_type}_hard'
                )
                subj_hard_score = model_averages[model]['subjective'].get(
                    f'{metric_type}_hard'
                )
                subj_sem_score = model_averages[model]['subjective'].get(
                    f'{metric_type}_sem'
                )

                if (
                    control_score is not None
                    and subj_hard_score is not None
                    and subj_sem_score is not None
                ):
                    models_with_data.append(model)
                    control_performance.append(control_score)
                    subjective_hard.append(subj_hard_score)
                    subjective_sem.append(subj_sem_score)

            if models_with_data:
                # Use chronological ordering instead of capability ordering
                model_order = self.get_model_order()
                models_sorted = [
                    m for m in model_order if m in models_with_data
                ]

                # Get indices for the ordered models
                model_to_idx = {m: i for i, m in enumerate(models_with_data)}
                if not models_sorted:
                    models_sorted = list(models_with_data)
                sorted_indices = [model_to_idx[m] for m in models_sorted]

                x_pos = range(len(models_sorted))
                control_sorted = [
                    control_performance[i] for i in sorted_indices
                ]
                subj_hard_sorted = [subjective_hard[i] for i in sorted_indices]
                subj_sem_sorted = [subjective_sem[i] for i in sorted_indices]

                # Plot the three metrics as requested
                ax.plot(
                    x_pos,
                    control_sorted,
                    'o-',
                    label=f'Objective Hard',
                    linewidth=2,
                    markersize=8,
                    color=THIRD_COLOR,
                )

                corr_hard = self.compute_correlation(
                    control_sorted, subj_hard_sorted
                )
                corr_sem = self.compute_correlation(
                    control_sorted, subj_sem_sorted
                )

                ax.plot(
                    x_pos,
                    subj_hard_sorted,
                    's-',
                    label=f'Subjective Hard {self.correlation_symbol} = {corr_hard:.2f}',
                    linewidth=2,
                    markersize=8,
                    color=HARD_COLOR,
                )

                ax.plot(
                    x_pos,
                    subj_sem_sorted,
                    '^-',
                    label=f'Subjective Semantic {self.correlation_symbol} = {corr_sem:.2f}',
                    linewidth=2,
                    markersize=8,
                    color=SEM_COLOR,
                )

                ax.set_ylabel('Performance', fontsize=20, labelpad=15)
                ax.set_title(
                    f'{metric_type.capitalize()} F1: Objective vs. Subjective',
                    fontsize=24,
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    models_sorted,
                    rotation=45,
                    ha='right',
                    fontsize=18,
                )
                ax.legend(fontsize=18)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f"figure_b1_{metric_type}_performance_comparison.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.savefig(
                self.output_dir
                / f"figure_b1_{metric_type}_performance_comparison.pdf",
                bbox_inches='tight',
            )
            plt.close()

        print(f"Figure B1 (3 separate plots) saved to {self.output_dir}")

    def create_figure_b1_supp(self):
        """Create Figure B1 (supplementary): Grid of subplots for each dataset pair"""
        print("Creating Figure B1 (supplementary)...")

        # Calculate number of subplot rows and columns
        n_subjective = len(self.subjective_datasets)
        n_control = len(self.control_datasets)

        for metric_type in ['macro', 'micro', 'sample']:
            fig, axes = plt.subplots(
                n_subjective,
                n_control,
                figsize=(4 * n_control, 4 * n_subjective),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            fig.suptitle(
                f'{metric_type.capitalize()} F1 - Objective - Subjective Pair Comparison',
                fontsize=18,
                y=0.98,
            )

            for i, subj_dataset in enumerate(self.subjective_datasets):
                for j, control_dataset in enumerate(self.control_datasets):
                    ax = axes[i, j]

                    # Collect data for this specific dataset pair
                    models_with_data = []
                    control_performance = []
                    subjective_hard = []
                    subjective_sem = []

                    for model in self.model_names:
                        # Control dataset performance
                        control_score = None
                        if (
                            control_dataset in self.data
                            and model in self.data[control_dataset]
                        ):
                            metrics = self.data[control_dataset][model]
                            control_score, _ = self.extract_metric_value(
                                metrics, f'test_{metric_type}_f1'
                            )

                        # Subjective dataset performance
                        subj_hard_score = None
                        subj_sem_score = None
                        if (
                            subj_dataset in self.data
                            and model in self.data[subj_dataset]
                        ):
                            metrics = self.data[subj_dataset][model]
                            subj_hard_score, _ = self.extract_metric_value(
                                metrics, f'test_{metric_type}_f1'
                            )
                            if metric_type == 'sample':
                                subj_sem_score, _ = self.extract_metric_value(
                                    metrics, 'test_semantic_f1'
                                )
                            else:
                                subj_sem_score, _ = self.extract_metric_value(
                                    metrics, f'test_{metric_type}_semantic_f1'
                                )

                        if (
                            control_score is not None
                            and subj_hard_score is not None
                            and subj_sem_score is not None
                        ):
                            models_with_data.append(model)
                            control_performance.append(control_score)
                            subjective_hard.append(subj_hard_score)
                            subjective_sem.append(subj_sem_score)

                    if models_with_data:
                        # Use chronological ordering
                        model_order = self.get_model_order()
                        models_sorted = [
                            m for m in model_order if m in models_with_data
                        ]

                        # Get indices for the ordered models
                        model_to_idx = {
                            m: i for i, m in enumerate(models_with_data)
                        }
                        if not models_sorted:
                            models_sorted = list(models_with_data)
                        sorted_indices = [
                            model_to_idx[m] for m in models_sorted
                        ]

                        x_pos = range(len(models_sorted))
                        control_sorted = [
                            control_performance[k] for k in sorted_indices
                        ]
                        subj_hard_sorted = [
                            subjective_hard[k] for k in sorted_indices
                        ]
                        subj_sem_sorted = [
                            subjective_sem[k] for k in sorted_indices
                        ]

                        corr_hard = self.compute_correlation(
                            control_sorted, subj_hard_sorted
                        )
                        corr_sem = self.compute_correlation(
                            control_sorted, subj_sem_sorted
                        )

                        # Plot the three metrics
                        ax.plot(
                            x_pos,
                            control_sorted,
                            'o-',
                            label=f'{control_dataset} Hard',
                            linewidth=2,
                            markersize=6,
                            color=THIRD_COLOR,
                        )

                        ax.plot(
                            x_pos,
                            subj_hard_sorted,
                            's-',
                            label=f'{subj_dataset} Hard {self.correlation_symbol}={corr_hard:.2f}',
                            linewidth=2,
                            markersize=6,
                            color=HARD_COLOR,
                        )

                        ax.plot(
                            x_pos,
                            subj_sem_sorted,
                            '^-',
                            label=f'{subj_dataset} Semantic {self.correlation_symbol}={corr_sem:.2f}',
                            linewidth=2,
                            markersize=6,
                            color=SEM_COLOR,
                        )

                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(
                            models_sorted, rotation=45, ha='right', fontsize=10
                        )
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        ax.set_title(
                            f'{subj_dataset} vs {control_dataset}', fontsize=12
                        )

                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f'No data\n({len(models_with_data)} models)',
                            ha='center',
                            va='center',
                            transform=ax.transAxes,
                        )
                        ax.set_title(
                            f'{subj_dataset} vs {control_dataset}', fontsize=10
                        )

            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f"figure_b1_supp_{metric_type}_dataset_pairs.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.savefig(
                self.output_dir
                / f"figure_b1_supp_{metric_type}_dataset_pairs.pdf",
                bbox_inches='tight',
            )
            plt.close()

        print(f"Figure B1 (supplementary) saved to {self.output_dir}")

    def create_figure_b2(self):
        """Create Figure B2: Correlation between capability ranking and each metric"""
        print(
            f"Creating Figure B2 ({self.correlation_display_name} correlation)..."
        )

        capability_ranking, capability_scores = (
            self.get_model_capability_ranking()
        )

        correlations = defaultdict(list)

        # Calculate correlations for each dataset and metric type
        for dataset in self.subjective_datasets + self.control_datasets:
            if dataset not in self.data:
                continue

            for metric_type in ['macro', 'micro', 'sample']:
                # Get model scores for this metric
                model_scores_hard = {}
                model_scores_sem = {}

                for model in capability_ranking.keys():
                    if model in self.data[dataset]:
                        metrics = self.data[dataset][model]
                        hard_f1, _ = self.extract_metric_value(
                            metrics, f'test_{metric_type}_f1'
                        )

                        # Handle semantic F1 correctly for sample vs macro/micro
                        if metric_type == 'sample':
                            sem_f1, _ = self.extract_metric_value(
                                metrics, 'test_semantic_f1'
                            )
                        else:
                            sem_f1, _ = self.extract_metric_value(
                                metrics, f'test_{metric_type}_semantic_f1'
                            )

                        if hard_f1 is not None:
                            model_scores_hard[model] = hard_f1
                        if sem_f1 is not None:
                            model_scores_sem[model] = sem_f1

                # Calculate correlations with capability ranking
                if len(model_scores_hard) >= self.correlation_min_points:
                    capability_vals = [
                        capability_scores[m] for m in model_scores_hard.keys()
                    ]
                    hard_vals = list(model_scores_hard.values())

                    corr_hard = self.compute_correlation(
                        capability_vals, hard_vals
                    )
                    if corr_hard is not None:
                        correlations[f'{metric_type}_hard'].append(corr_hard)

                    if len(model_scores_sem) >= self.correlation_min_points:
                        sem_vals = [
                            model_scores_sem[m] for m in model_scores_sem.keys()
                        ]
                        cap_vals_sem = [
                            capability_scores[m]
                            for m in model_scores_sem.keys()
                        ]
                        corr_sem = self.compute_correlation(
                            cap_vals_sem, sem_vals
                        )
                        if corr_sem is not None:
                            correlations[f'{metric_type}_sem'].append(corr_sem)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        metric_types = ['macro', 'micro', 'sample']
        x_pos = np.arange(len(metric_types))
        width = 0.35

        hard_means = [
            (
                np.mean(correlations[f'{mt}_hard'])
                if correlations[f'{mt}_hard']
                else 0
            )
            for mt in metric_types
        ]
        sem_means = [
            (
                np.mean(correlations[f'{mt}_sem'])
                if correlations[f'{mt}_sem']
                else 0
            )
            for mt in metric_types
        ]

        hard_stds = [
            (
                np.std(correlations[f'{mt}_hard'])
                if len(correlations[f'{mt}_hard']) > 1
                else 0
            )
            for mt in metric_types
        ]
        sem_stds = [
            (
                np.std(correlations[f'{mt}_sem'])
                if len(correlations[f'{mt}_sem']) > 1
                else 0
            )
            for mt in metric_types
        ]

        bars1 = ax.bar(
            x_pos - width / 2,
            hard_means,
            width,
            yerr=hard_stds,
            label='Hard F1',
            alpha=0.8,
            capsize=5,
        )
        bars2 = ax.bar(
            x_pos + width / 2,
            sem_means,
            width,
            yerr=sem_stds,
            label='Semantic F1',
            alpha=0.8,
            capsize=5,
        )

        ax.set_xlabel('Metric Type')
        ax.set_ylabel(self.correlation_axis_label)
        ax.set_title(self.correlation_title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([mt.capitalize() for mt in metric_types])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figure_b2_rank_correlations.png",
            dpi=300,
            bbox_inches='tight',
        )
        plt.savefig(
            self.output_dir / "figure_b2_rank_correlations.pdf",
            bbox_inches='tight',
        )
        print(f"Figure B2 saved to {self.output_dir}")
        plt.close()

    def create_table_b2(self):
        """Create Table B2: Macro/micro/sample numbers with bootstrap CIs"""
        print("Creating Table B2...")

        results = []

        for dataset in self.subjective_datasets + self.control_datasets:
            if dataset not in self.data:
                continue

            dataset_type = (
                'Subjective'
                if dataset in self.subjective_datasets
                else 'Control'
            )

            for model in self.model_names:
                if model in self.data[dataset]:
                    metrics = self.data[dataset][model]

                    for metric_type in ['macro', 'micro', 'sample']:
                        # Hard F1
                        hard_f1, hard_std = self.extract_metric_value(
                            metrics, f'test_{metric_type}_f1'
                        )

                        # Semantic F1
                        if metric_type == 'sample':
                            sem_f1, sem_std = self.extract_metric_value(
                                metrics, 'test_semantic_f1'
                            )
                        else:
                            sem_f1, sem_std = self.extract_metric_value(
                                metrics, f'test_{metric_type}_semantic_f1'
                            )

                        # Identity check (semantic should equal hard when similarity = identity)
                        identity_check = "N/A"
                        if hard_f1 is not None and sem_f1 is not None:
                            if (
                                abs(hard_f1 - sem_f1) < 0.001
                            ):  # Very close values
                                identity_check = "✓"
                            else:
                                identity_check = f"Δ={sem_f1-hard_f1:.3f}"

                        results.append(
                            {
                                'Dataset': dataset,
                                'Dataset_Type': dataset_type,
                                'Model': model,
                                'Metric_Type': metric_type,
                                'Hard_F1_Mean': hard_f1,
                                'Hard_F1_Std': hard_std,
                                'Semantic_F1_Mean': sem_f1,
                                'Semantic_F1_Std': sem_std,
                                'Identity_Check': identity_check,
                            }
                        )

        df = pd.DataFrame(results)
        df.to_csv(
            self.output_dir / "table_b2_detailed_metrics.csv", index=False
        )

        # Create summary table
        summary_results = []
        for dataset_type in ['Subjective', 'Control']:
            subset = df[df['Dataset_Type'] == dataset_type]
            for metric_type in ['macro', 'micro', 'sample']:
                metric_subset = subset[subset['Metric_Type'] == metric_type]

                if not metric_subset.empty:
                    hard_mean = metric_subset['Hard_F1_Mean'].mean()
                    hard_std = metric_subset['Hard_F1_Std'].mean()
                    sem_mean = metric_subset['Semantic_F1_Mean'].mean()
                    sem_std = metric_subset['Semantic_F1_Std'].mean()

                    summary_results.append(
                        {
                            'Dataset_Type': dataset_type,
                            'Metric_Type': metric_type,
                            'Hard_F1': f"{hard_mean:.3f}±{hard_std:.3f}",
                            'Semantic_F1': (
                                f"{sem_mean:.3f}±{sem_std:.3f}"
                                if not pd.isna(sem_mean)
                                else "N/A"
                            ),
                        }
                    )

        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(self.output_dir / "table_b2_summary.csv", index=False)

        print(f"Table B2 saved to {self.output_dir}")
        return df, summary_df

    def run_analysis(self):
        """Run complete Study B analysis"""
        print("Starting Study B Analysis...")

        # Load all results
        self.load_results()

        if not self.data:
            print("No data loaded. Check logs directory.")
            return

        # Create all figures and tables
        try:
            self.create_table_b1()
        except Exception as e:
            print(f"Error creating Table B1: {e}")

        try:
            self.create_figure_b1()
        except Exception as e:
            print(f"Error creating Figure B1: {e}")

        try:
            self.create_figure_b1_supp()
        except Exception as e:
            print(f"Error creating Figure B1 (supplementary): {e}")

        try:
            self.create_figure_b2()
        except Exception as e:
            print(f"Error creating Figure B2: {e}")

        try:
            self.create_table_b2()
        except Exception as e:
            print(f"Error creating Table B2: {e}")

        print(f"\nAnalysis complete! All outputs saved to {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Study B figures and tables."
    )
    parser.add_argument(
        "--correlation",
        choices=['spearman', 'pearson', 'ccc', 'icc'],
        default='spearman',
        help="Correlation metric to use when comparing subjective and objective metrics.",
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Path to logs directory containing aggregated metrics.",
    )

    args = parser.parse_args()

    analyzer = StudyBAnalyzer(
        logs_dir=args.logs_dir, correlation_method=args.correlation
    )
    analyzer.run_analysis()
