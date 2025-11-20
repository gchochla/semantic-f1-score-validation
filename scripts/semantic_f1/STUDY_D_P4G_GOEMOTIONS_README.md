# Study D: PersuasionForGood vs GoEmotions 2-shot Analysis

This directory contains scripts for analyzing the relationship between downstream performance on PersuasionForGood tasks and 2-shot GoEmotions emotion classification performance.

## Overview

Study D examines whether models that perform better on emotion classification (GoEmotions 2-shot) also produce better emotion features for downstream tasks in persuasion contexts (PersuasionForGood). This tests the ecological validity of emotion classification metrics.

## Scripts

### 1. `study_d_p4g_goemotions.py`
**Purpose**: Main analysis script that compares P4G downstream performance with GoEmotions 2-shot F1 scores
- Extracts emotions from P4G model predictions  
- Builds conversation-level features from emotion sequences
- Trains downstream predictors for all P4G tasks (persuader_success, donations)
- Loads GoEmotions 2-shot F1 metrics from logs
- Calculates correlations between emotion classification and downstream performance
- Handles model failures gracefully with notifications

**Usage**:
```bash
python scripts/semantic_f1/study_d_p4g_goemotions.py --max-k 10 --n-runs 5
```

**Arguments**:
- `--logs-dir`: Path to logs directory (default: logs/)
- `--output-dir`: Output directory for results  
- `--max-k`: Maximum number of conversation turns to consider (default: 10)
- `--n-runs`: Number of runs for confidence intervals (default: 5)

**Output**: 
- `study_d_p4g_goemotions_results.pkl`: Complete results with correlations

### 2. `study_d_p4g_goemotions_visualization.py`
**Purpose**: Generate comprehensive visualizations of P4G vs GoEmotions analysis
- Correlation overview showing all emotion F1 metrics vs P4G performance
- Detailed scatter plots for significant correlations  
- Model performance comparison across tasks
- Dual-axis plots showing both metrics together

**Usage**:
```bash
python scripts/semantic_f1/study_d_p4g_goemotions_visualization.py
```

**Output**:
- `figure_d1_correlation_overview.png/pdf`: Bar charts of all correlations
- `figure_d2_detailed_correlations.png/pdf`: Scatter plots for significant correlations
- `figure_d3_model_performance.png/pdf`: Model performance comparison
- `figure_d4_dual_axis.png/pdf`: Dual-axis comparison plot
- `correlation_summary.csv`: Tabular correlation results
- `model_performance_summary.csv`: Model performance across tasks

### 3. Same for SemEval

## Model Coverage

The analysis automatically finds and processes models with both P4G and GoEmotions 2-shot experiments:

**Supported models**:
- GPT-4o-mini
- GPT-4.1-mini  
- Llama-2-7b-chat
- Llama-2-70b-chat
- Llama-3.1-8B-Instruct
- Llama-3.3-70B-Instruct (if data available)

**Failed models are reported**: If any model lacks required data files, the script notifies the user and continues with available models.

## P4G Tasks Evaluated

1. **persuader_success** (classification): Whether persuadee donates (AUC metric)
2. **persuader_donation** (regression): Amount persuader donates (-MSE metric)  
3. **persuadee_donation** (regression): Amount persuadee donates (-MSE metric)
4. **donation** (regression): Total donation amount (-MSE metric)

## Key Findings

From the current analysis of 5 models:

**Significant Correlations Found**:
- **Annoyance F1**: Perfect positive correlation (ρ = 1.000, p < 0.001)
- **Gratitude F1**: Strong positive correlation (ρ = 0.975, p = 0.005)  
- **Admiration F1**: Strong positive correlation (ρ = 0.975, p = 0.005)
- **Semantic F1**: Strong positive correlation (ρ = 0.975, p = 0.005)
- **Grief F1**: Strong positive correlation (ρ = 0.918, p = 0.028)

**Best P4G Performance**:
- **Classification (Persuader Success)**: GPT-4o-mini (AUC = 0.570)
- **Regression**: Llama models and GPT-4.1-mini perform similarly well

## Dependencies

- Python packages: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, pyyaml
- Data: P4G and GoEmotions experiment results in `logs/` directories

## Workflow

1. **Run Analysis**: `python study_d_p4g_goemotions.py` 
2. **Create Visualizations**: `python study_d_p4g_goemotions_visualization.py`

## Error Handling

- **Missing model data**: Scripts gracefully skip models without required files and report them
- **Numerical issues**: Regression tasks may show extreme values for some models (noted in summary)
- **Insufficient data**: Correlations require ≥3 models; fewer models are handled gracefully

## Interpretation

This analysis provides evidence for whether emotion classification abilities transfer to downstream utility in conversational contexts. Significant positive correlations suggest that models better at detecting certain emotions (annoyance, gratitude, admiration) also produce more useful emotion features for predicting persuasion outcomes.

The perfect correlation for annoyance (ρ = 1.000) may indicate this emotion is particularly diagnostic for persuasion success, while the strong correlations for positive emotions (gratitude, admiration) suggest emotional intelligence transfers across tasks.
