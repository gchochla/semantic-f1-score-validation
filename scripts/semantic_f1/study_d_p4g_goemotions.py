#!/usr/bin/env python3
"""
Study D Implementation: P4G vs 2-shot GoEmotions Comparison

This script implements Study D by comparing downstream performance on PersuasionForGood
with 2-shot GoEmotions performance, reading results directly from the logs directory.
It handles model failures gracefully and provides comprehensive analysis.
"""

import argparse
import csv
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Emotion labels based on the instruction.txt file
EMOTION_LABELS = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
]


class EmotionExtractor:
    """Extract and process emotion predictions from model outputs."""

    def __init__(self):
        self.emotion_pattern = re.compile(
            r'EMOTIONS?:\s*([^,\n]+(?:,\s*[^,\n]+)*)', re.IGNORECASE
        )

    def extract_emotions_from_text(self, text: str) -> List[str]:
        """Extract emotions from model output text."""
        if not text:
            return []

        # Handle 'none' case
        if 'none' in text.lower():
            return []

        match = self.emotion_pattern.search(text)
        if not match:
            return []

        emotions_text = match.group(1).strip()

        # Split by comma and clean
        emotions = []
        for emotion in emotions_text.split(','):
            emotion = emotion.strip().lower()
            if emotion in [e.lower() for e in EMOTION_LABELS]:
                emotions.append(emotion)

        return emotions

    def emotions_to_vector(self, emotions: List[str]) -> np.ndarray:
        """Convert list of emotions to binary vector."""
        vector = np.zeros(len(EMOTION_LABELS))
        for emotion in emotions:
            emotion_lower = emotion.lower()
            for i, label in enumerate(EMOTION_LABELS):
                if label.lower() == emotion_lower:
                    vector[i] = 1
                    break
        return vector


class GoEmotionsProcessor:
    """Process GoEmotions 2-shot experiment results."""

    def __init__(
        self, emotion_extractor: EmotionExtractor, semantic_metric: str = 'f1'
    ):
        self.emotion_extractor = emotion_extractor
        self.semantic_metric = semantic_metric

    def load_goemotions_results(
        self, metrics_path: Path
    ) -> Optional[Dict[str, float]]:
        """Load GoEmotions results from metrics.yml file."""
        try:
            with open(metrics_path, 'r') as f:
                data = yaml.safe_load(f)

            # Extract metrics from the first experiment
            if not data:
                return None

            experiment_key = list(data.keys())[0]
            metrics = data[experiment_key]

            # Extract relevant metrics
            results = {}
            for metric_name, value in metrics.items():
                # For semantic metrics, use the specified metric type
                if 'semantic' in metric_name.lower():
                    if (
                        self.semantic_metric in metric_name.lower()
                        and isinstance(value, (int, float))
                    ):
                        # Rename to use 'f1' suffix for consistent downstream processing
                        normalized_name = metric_name.replace(
                            f'_{self.semantic_metric}', '_f1'
                        )
                        results[normalized_name] = float(value)
                # For non-semantic metrics, always use f1
                elif 'f1' in metric_name.lower() and isinstance(
                    value, (int, float)
                ):
                    results[metric_name] = float(value)

            return results

        except Exception as e:
            logger.warning(
                f"Failed to load GoEmotions results from {metrics_path}: {e}"
            )
            return None


class ConversationProcessor:
    """Process conversations and extract features for downstream prediction."""

    def __init__(self, emotion_extractor: EmotionExtractor):
        self.emotion_extractor = emotion_extractor

    def parse_conversation_id(self, conv_id: str) -> Tuple[str, int]:
        """Parse conversation ID into dialog_id and turn number."""
        parts = conv_id.split('--')
        if len(parts) == 2:
            return parts[0], int(parts[1])
        raise ValueError(f"Invalid conversation ID format: {conv_id}")

    def load_model_predictions(
        self, indexed_metrics_path: Path
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load model predictions from indexed_metrics.yml file."""
        try:
            with open(indexed_metrics_path, 'r') as f:
                data = yaml.safe_load(f)

            # Extract predictions from first experiment
            if not data:
                return None

            experiment_key = list(data.keys())[0]
            predictions = data[experiment_key]

            processed_predictions = {}
            for conv_id, metrics in predictions.items():
                try:
                    dialog_id, turn_idx = self.parse_conversation_id(conv_id)
                    emotions = (
                        self.emotion_extractor.extract_emotions_from_text(
                            metrics.get('test_outs', '')
                        )
                    )
                    emotion_vector = self.emotion_extractor.emotions_to_vector(
                        emotions
                    )

                    processed_predictions[conv_id] = {
                        'dialog_id': dialog_id,
                        'turn_idx': turn_idx,
                        'emotions': emotions,
                        'emotion_vector': emotion_vector,
                        'ground_truth': metrics.get('test_gt', ''),
                        'prediction': metrics.get('test_preds', ''),
                        'text': metrics.get('test_text', ''),
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping {conv_id}: {e}")
                    continue

            return processed_predictions

        except Exception as e:
            logger.warning(
                f"Failed to load P4G predictions from {indexed_metrics_path}: {e}"
            )
            return None


class DownstreamPredictor:
    """Train and evaluate downstream predictors for all label types."""

    def __init__(self, random_state: int = 0, n_runs: int = 5):
        self.random_state = random_state
        self.n_runs = n_runs
        self.scaler = StandardScaler()

    def prepare_data(
        self, conversation_features: Dict[str, Dict[str, Any]], k: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels for a given k."""
        X = []
        y = []
        dialog_ids = []

        for dialog_id, data in conversation_features.items():
            if k in data['features_by_k']:
                X.append(data['features_by_k'][k])
                y.append(data['outcome'])
                dialog_ids.append(dialog_id)

        return np.array(X), np.array(y), dialog_ids

    def evaluate_classification_task(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate classification task (persuader_success) with multiple runs for CI."""
        if len(X) == 0 or len(np.unique(y)) < 2:
            return {}

        # Store results from multiple runs
        auc_scores = []
        accuracy_scores = []
        f1_scores = []

        for run in range(self.n_runs):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.1,
                    random_state=self.random_state + run,
                    stratify=y,
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                max_iter = 2000

                # Train classifier
                clf = LogisticRegression(
                    random_state=self.random_state, max_iter=max_iter
                )
                clf.fit(X_train_scaled, y_train)

                # Predictions
                y_pred = clf.predict(X_test_scaled)
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

                # Metrics
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))

            except Exception as e:
                logger.warning(f"Classification run {run} failed: {e}")
                continue

        if not auc_scores:
            return {}

        # Calculate means and confidence intervals
        return {
            'auc': np.mean(auc_scores),
            'auc_ci': 1.96 * np.std(auc_scores) / np.sqrt(len(auc_scores)),
            'accuracy': np.mean(accuracy_scores),
            'accuracy_ci': 1.96
            * np.std(accuracy_scores)
            / np.sqrt(len(accuracy_scores)),
            'f1': np.mean(f1_scores),
            'f1_ci': 1.96 * np.std(f1_scores) / np.sqrt(len(f1_scores)),
            'n_samples': len(X),
        }

    def evaluate_regression_task(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression task (donations) with multiple runs for CI."""
        if len(X) == 0 or np.var(y) == 0:
            return {}

        # Store results from multiple runs
        neg_mse_scores = []
        rmse_scores = []
        r2_scores = []

        for run in range(self.n_runs):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=self.random_state + run
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train regressor
                reg = LinearRegression()
                reg.fit(X_train_scaled, y_train)

                # Predictions
                y_pred = reg.predict(X_test_scaled)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                neg_mse_scores.append(
                    -mse
                )  # Negative MSE for "higher is better"
                rmse_scores.append(np.sqrt(mse))
                r2_scores.append(r2_score(y_test, y_pred))

            except Exception as e:
                logger.warning(f"Regression run {run} failed: {e}")
                continue

        if not neg_mse_scores:
            return {}

        # Calculate means and confidence intervals
        return {
            'neg_mse': np.mean(neg_mse_scores),
            'neg_mse_ci': 1.96
            * np.std(neg_mse_scores)
            / np.sqrt(len(neg_mse_scores)),
            'rmse': np.mean(rmse_scores),
            'rmse_ci': 1.96 * np.std(rmse_scores) / np.sqrt(len(rmse_scores)),
            'r2': np.mean(r2_scores),
            'r2_ci': 1.96 * np.std(r2_scores) / np.sqrt(len(r2_scores)),
            'n_samples': len(X),
        }


class StudyDAnalyzer:
    """Main analyzer for Study D: P4G vs 2-shot GoEmotions comparison."""

    def __init__(
        self,
        logs_dir: str,
        output_dir: str,
        n_runs: int = 5,
        semantic_metric: str = 'f1',
    ):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_metric = semantic_metric

        print(f"Output directory created: {self.output_dir}")

        self.emotion_extractor = EmotionExtractor()
        self.conversation_processor = ConversationProcessor(
            self.emotion_extractor
        )
        self.goemotions_processor = GoEmotionsProcessor(
            self.emotion_extractor, semantic_metric=semantic_metric
        )
        self.downstream_predictor = DownstreamPredictor(n_runs=n_runs)

        # Define all label types and their task types
        self.label_types = {
            'persuader_success': 'classification',
            'persuader_donation': 'regression',
            'persuadee_donation': 'regression',
            'donation': 'regression',
        }

    def load_persuasion_data(self) -> Dict[str, Dict[str, float]]:
        """Load ground truth persuasion data for all label types."""
        data_file = "/path/to/persuasionforgood/data/FullData/full_info.csv"

        persuasion_data = {}
        try:
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in reader:
                    row_dict = {k: v for k, v in zip(headers, row)}
                    dialog_id = row_dict["B2"]

                    if dialog_id not in persuasion_data:
                        persuasion_data[dialog_id] = {}

                    role = "persuader" if row_dict["B4"] == "0" else "persuadee"
                    persuasion_data[dialog_id][f"{role}_donation"] = float(
                        row_dict["B6"]
                    )

        except FileNotFoundError:
            logger.warning(
                "Could not load persuasion data file, using simulated data"
            )
            return {}

        # Calculate additional label types with donation range validation
        for dialog_id in persuasion_data:
            data = persuasion_data[dialog_id]
            persuader_donation = data.get('persuader_donation', 0)
            persuadee_donation = data.get('persuadee_donation', 0)

            # Clamp individual donations to valid range [0, 2]
            persuader_donation = max(0, min(2, persuader_donation))
            persuadee_donation = max(0, min(2, persuadee_donation))

            # Update clamped values back to data
            data['persuader_donation'] = persuader_donation
            data['persuadee_donation'] = persuadee_donation

            # Calculate total donation and success
            total_donation = persuader_donation + persuadee_donation
            data['donation'] = max(0, min(2, total_donation))
            data['persuader_success'] = 1 if persuadee_donation > 0 else 0

        return persuasion_data

    def build_conversation_features_all_labels(
        self,
        predictions: Dict[str, Dict[str, Any]],
        persuasion_data: Dict[str, Dict[str, float]],
        max_turns: int = 10,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build features for each conversation and all label types."""
        conversations = defaultdict(list)

        # Group by dialog_id
        for conv_id, data in predictions.items():
            conversations[data['dialog_id']].append((data['turn_idx'], data))

        # Sort by turn index
        for dialog_id in conversations:
            conversations[dialog_id].sort(key=lambda x: x[0])

        all_features = {}

        for label_type in self.label_types:
            conversation_features = {}

            for dialog_id, turns in conversations.items():
                if not turns or dialog_id not in persuasion_data:
                    continue

                # Get the outcome for this label type
                outcome = persuasion_data[dialog_id].get(label_type, 0)

                # Build features for different values of k (last k turns)
                features_by_k = {}

                for k in range(1, min(max_turns + 1, len(turns) + 1)):
                    # Take last k turns
                    last_k_turns = turns[-k:]

                    # Aggregate emotion vectors
                    emotion_features = np.zeros(len(EMOTION_LABELS))
                    turn_count = len(last_k_turns)

                    for _, turn_data in last_k_turns:
                        emotion_features += turn_data['emotion_vector']

                    # Normalize by number of turns (mean emotions)
                    if turn_count > 0:
                        emotion_features = emotion_features / turn_count

                    features_by_k[k] = emotion_features

                conversation_features[dialog_id] = {
                    'outcome': outcome,
                    'features_by_k': features_by_k,
                    'num_turns': len(turns),
                }

            all_features[label_type] = conversation_features

        return all_features

    def find_model_experiments(self) -> List[Tuple[str, Path, Path]]:
        """Find all available model experiments with both P4G and GoEmotions 2-shot."""
        experiments = []
        failed_models = []

        print(f"Looking for models in {self.logs_dir}")

        # Define model name mappings between P4G and GoEmotions
        model_mappings = [
            # OpenAI models
            (
                'PersuasionForGoodOpenAI/gpt-4o-mini-inference-0-shot_0',
                'GoEmotionsOpenAI/gpt-4o-mini-inference-2-shot_0',
            ),
            (
                'PersuasionForGoodOpenAI/gpt-4.1-mini-inference-0-shot_0',
                'GoEmotionsOpenAI/gpt-4.1-mini-inference-2-shot_0',
            ),
            # Local models
            (
                'PersuasionForGood/meta-llama--Llama-2-7b-chat-hf-inference-0-shot_0',
                'GoEmotions/meta-llama--Llama-2-7b-chat-hf-inference-2-shot_0',
            ),
            (
                'PersuasionForGood/meta-llama--Llama-2-70b-chat-hf-inference-0-shot_0',
                'GoEmotions/meta-llama--Llama-2-70b-chat-hf-inference-2-shot_0',
            ),
            (
                'PersuasionForGood/meta-llama--Llama-3.1-8B-Instruct-inference-0-shot_0',
                'GoEmotions/meta-llama--Llama-3.1-8B-Instruct-inference-2-shot_0',
            ),
            (
                'PersuasionForGood/meta-llama--Llama-3.3-70B-Instruct-inference-0-shot_0',
                'GoEmotions/meta-llama--Llama-3.3-70B-Instruct-inference-2-shot_0',
            ),
        ]

        for p4g_path, goemotions_path in model_mappings:
            p4g_full_path = self.logs_dir / p4g_path
            goemotions_full_path = self.logs_dir / goemotions_path

            # Check if both experiments exist
            p4g_indexed = p4g_full_path / 'indexed_metrics.yml'
            goemotions_metrics = goemotions_full_path / 'metrics.yml'

            model_name = p4g_path.split('/')[-1]  # Extract model directory name

            if p4g_indexed.exists() and goemotions_metrics.exists():
                experiments.append(
                    (model_name, p4g_indexed, goemotions_metrics)
                )
                print(f"✓ Found complete experiment for: {model_name}")
            else:
                missing = []
                if not p4g_indexed.exists():
                    missing.append(f"P4G: {p4g_indexed}")
                if not goemotions_metrics.exists():
                    missing.append(f"GoEmotions: {goemotions_metrics}")

                failed_models.append(
                    (model_name, f"Missing files: {'; '.join(missing)}")
                )
                print(f"✗ Missing files for {model_name}: {'; '.join(missing)}")

        print(f"Found {len(experiments)} complete experiments")
        if failed_models:
            print(f"⚠️  {len(failed_models)} models failed to load:")
            for model_name, reason in failed_models:
                print(f"  - {model_name}: {reason}")

        return experiments, failed_models

    def run_analysis(self, max_k: int = 10) -> Dict[str, Any]:
        """Run the complete Study D analysis."""
        logger.info("Starting Study D analysis: P4G vs 2-shot GoEmotions...")

        experiments, failed_models = self.find_model_experiments()
        logger.info(f"Found {len(experiments)} complete experiments to analyze")

        # Load persuasion data
        persuasion_data = self.load_persuasion_data()
        logger.info(
            f"Loaded persuasion data for {len(persuasion_data)} conversations"
        )

        results = {}
        goemotions_results = {}

        for (
            model_name,
            p4g_indexed_path,
            goemotions_metrics_path,
        ) in experiments:
            logger.info(f"Processing model: {model_name}")

            try:
                # Load P4G predictions
                p4g_predictions = (
                    self.conversation_processor.load_model_predictions(
                        p4g_indexed_path
                    )
                )
                if p4g_predictions is None:
                    failed_models.append(
                        (model_name, "Failed to load P4G predictions")
                    )
                    continue

                # Load GoEmotions results
                goemotions_metrics = (
                    self.goemotions_processor.load_goemotions_results(
                        goemotions_metrics_path
                    )
                )
                if goemotions_metrics is None:
                    failed_models.append(
                        (model_name, "Failed to load GoEmotions results")
                    )
                    continue

                logger.info(f"Loaded {len(p4g_predictions)} P4G predictions")
                logger.info(
                    f"Loaded {len(goemotions_metrics)} GoEmotions metrics"
                )

                # Build conversation features for all label types
                all_conversation_features = (
                    self.build_conversation_features_all_labels(
                        p4g_predictions, persuasion_data, max_k
                    )
                )

                model_results = {}

                # Evaluate downstream tasks for each label type
                for label_type, task_type in self.label_types.items():
                    conversation_features = all_conversation_features[
                        label_type
                    ]

                    if not conversation_features:
                        logger.warning(
                            f"No conversation features for {label_type}"
                        )
                        continue

                    label_results = {'downstream_results': {}}

                    # Evaluate for different values of k
                    for k in range(1, min(max_k + 1, 11)):
                        X, y, dialog_ids = (
                            self.downstream_predictor.prepare_data(
                                conversation_features, k
                            )
                        )

                        if len(X) == 0:
                            continue

                        if task_type == 'classification':
                            metrics = self.downstream_predictor.evaluate_classification_task(
                                X, y
                            )
                        else:  # regression
                            metrics = self.downstream_predictor.evaluate_regression_task(
                                X, y
                            )

                        if metrics:
                            label_results['downstream_results'][k] = metrics

                    model_results[label_type] = label_results

                results[model_name] = model_results
                goemotions_results[model_name] = goemotions_metrics

                # Log best performance for each label type
                for label_type, label_results in model_results.items():
                    if 'downstream_results' in label_results:
                        best_k = max(
                            label_results['downstream_results'].keys(),
                            key=lambda k: (
                                label_results['downstream_results'][k].get(
                                    'auc', 0
                                )
                                if self.label_types[label_type]
                                == 'classification'
                                else label_results['downstream_results'][k].get(
                                    'neg_mse', -float('inf')
                                )
                            ),
                        )

                        metrics = label_results['downstream_results'][best_k]
                        if self.label_types[label_type] == 'classification':
                            logger.info(
                                f"{label_type}: Best k={best_k}, AUC={metrics['auc']:.3f}±{metrics['auc_ci']:.3f}"
                            )
                        else:
                            logger.info(
                                f"{label_type}: Best k={best_k}, -MSE={metrics['neg_mse']:.3f}±{metrics['neg_mse_ci']:.3f}"
                            )

            except Exception as e:
                error_msg = f"Error processing model: {str(e)}"
                logger.error(f"Failed to process {model_name}: {error_msg}")
                failed_models.append((model_name, error_msg))
                continue

        # Calculate correlations between P4G performance and GoEmotions metrics
        correlations = self.calculate_correlations(results, goemotions_results)

        # Save results
        output_data = {
            'results': results,
            'goemotions_results': goemotions_results,
            'correlations': correlations,
            'failed_models': failed_models,
            'metadata': {
                'n_models': len(results),
                'n_failed': len(failed_models),
                'max_k': max_k,
                'label_types': self.label_types,
                'semantic_metric': self.semantic_metric,
            },
        }

        output_file = self.output_dir / 'study_d_p4g_goemotions_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

        logger.info(f"Results saved to {output_file}")
        logger.info(
            f"Analysis complete: {len(results)} models processed, {len(failed_models)} failed"
        )

        return output_data

    def calculate_correlations(
        self, p4g_results: Dict[str, Any], goemotions_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate correlations between P4G downstream performance and GoEmotions metrics."""
        correlations = {}

        # Get all GoEmotions metric names
        all_goemotions_metrics = set()
        for model_metrics in goemotions_results.values():
            all_goemotions_metrics.update(model_metrics.keys())

        # Calculate correlations for each GoEmotions metric vs P4G performance
        for goemotions_metric in all_goemotions_metrics:
            if 'f1' not in goemotions_metric.lower():
                continue  # Only analyze F1 metrics

            # Collect data for correlation
            models = []
            goemotions_scores = []
            p4g_scores = []

            for model_name in p4g_results.keys():
                if model_name not in goemotions_results:
                    continue

                # Get GoEmotions score
                goemotions_score = goemotions_results[model_name].get(
                    goemotions_metric
                )
                if goemotions_score is None:
                    continue

                # Get best P4G downstream performance (prioritize classification over regression)
                best_p4g_score = None

                # First try to get best classification score
                for label_type in ['persuader_success']:
                    if label_type in p4g_results[model_name]:
                        downstream_results = p4g_results[model_name][
                            label_type
                        ].get('downstream_results', {})
                        if downstream_results:
                            best_auc = max(
                                (
                                    metrics.get('auc', 0)
                                    for metrics in downstream_results.values()
                                ),
                                default=0,
                            )
                            if best_auc > 0:
                                best_p4g_score = best_auc
                                break

                # If no classification score, try regression
                if best_p4g_score is None:
                    for label_type in ['persuadee_donation', 'donation']:
                        if label_type in p4g_results[model_name]:
                            downstream_results = p4g_results[model_name][
                                label_type
                            ].get('downstream_results', {})
                            if downstream_results:
                                best_neg_mse = max(
                                    (
                                        metrics.get('neg_mse', -float('inf'))
                                        for metrics in downstream_results.values()
                                    ),
                                    default=-float('inf'),
                                )
                                if best_neg_mse > -float('inf'):
                                    best_p4g_score = best_neg_mse
                                    break

                if best_p4g_score is not None:
                    models.append(model_name)
                    goemotions_scores.append(goemotions_score)
                    p4g_scores.append(best_p4g_score)

            # Calculate correlation if we have enough data points
            if len(models) >= 3:
                try:
                    corr_result = spearmanr(goemotions_scores, p4g_scores)
                    correlations[goemotions_metric] = {
                        'correlation': corr_result.correlation,
                        'p_value': corr_result.pvalue,
                        'n_models': len(models),
                        'data': {
                            'models': models,
                            'goemotions_scores': goemotions_scores,
                            'p4g_scores': p4g_scores,
                        },
                    }
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate correlation for {goemotions_metric}: {e}"
                    )

        return correlations


def main():
    parser = argparse.ArgumentParser(
        description='Study D: Compare P4G performance with 2-shot GoEmotions'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        required=True,
        help='Path to logs directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results',
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=10,
        help='Maximum number of turns to consider for feature building',
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=5,
        help='Number of runs for confidence interval estimation',
    )
    parser.add_argument(
        '--semantic-metric',
        type=str,
        choices=['f1', 'precision', 'recall'],
        default='f1',
        help='Which semantic metric to use for analysis (f1, precision, or recall). Non-semantic metrics always use f1.',
    )

    args = parser.parse_args()

    analyzer = StudyDAnalyzer(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        semantic_metric=args.semantic_metric,
    )

    results = analyzer.run_analysis(max_k=args.max_k)

    print("\n" + "=" * 80)
    print("STUDY D ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"Models analyzed: {results['metadata']['n_models']}")
    print(f"Models failed: {results['metadata']['n_failed']}")

    if results['failed_models']:
        print("\nFailed models:")
        for model_name, reason in results['failed_models']:
            print(f"  ✗ {model_name}: {reason}")

    print(f"\nCorrelations found: {len(results['correlations'])}")
    for metric_name, corr_data in results['correlations'].items():
        print(
            f"  {metric_name}: ρ = {corr_data['correlation']:.3f}, p = {corr_data['p_value']:.3f}"
        )

    print(
        f"\nResults saved to: {args.output_dir}/study_d_p4g_goemotions_results.pkl"
    )


if __name__ == '__main__':
    main()
