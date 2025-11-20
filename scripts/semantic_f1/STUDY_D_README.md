# Study D Implementation - Final Scripts

## Overview
Study D tests whether models ranked higher by Semantic F1 produce emotion features that predict downstream outcomes better than models ranked by hard F1.

## Core Scripts

### 1. `extended_study_d_implementation.py`
**Purpose**: Complete Study D analysis implementation
- Extracts emotions from model predictions
- Builds conversation-level features from emotion sequences
- Trains downstream predictors for all PersuasionForGood tasks
- Correlates results with Study B semantic F1 scores
- Generates comprehensive results pickle file

**Usage**:
```bash
python scripts/extended_study_d_implementation.py
```

**Output**: 
- `logs/analysis/semantic_f1/study_d/extended_study_d_results.pkl`
- Console output with performance metrics

### 2. `study_d_visualization.py`
**Purpose**: Generate comprehensive visualizations of Study D results
- Dual y-axis line plots (F1 scores vs downstream performance)
- Correlation scatter plots
- Model ranking comparison heatmaps
- Correlation summary plots

**Usage**:
```bash
# Focus on main tasks (persuader_success, persuadee_donation)
python scripts/study_d_visualization.py

# Include all tasks by modifying the script or using:
# visualizer.create_all_visualizations(focus_tasks='all')
```

**Output**: 
- `logs/analysis/semantic_f1/study_d/figures/figure_d1_task_performance_comparison.png/pdf`
- `logs/analysis/semantic_f1/study_d/figures/figure_d2_correlation_scatter.png/pdf`
- `logs/analysis/semantic_f1/study_d/figures/figure_d3_ranking_comparison.png/pdf`
- `logs/analysis/semantic_f1/study_d/figures/figure_d4_correlation_summary.png/pdf`

### 3. `study_d_summary.py`
**Purpose**: Generate readable summary reports of Study D findings
- Model performance rankings
- Correlation analysis results
- Formatted output for reports

**Usage**:
```bash
python scripts/study_d_summary.py
```

**Output**: 
- Console summary with model rankings and correlations
- CSV files with detailed metrics

## Workflow

1. **Run Analysis**: Execute `extended_study_d_implementation.py` to generate results
2. **Create Visualizations**: Run `study_d_visualization.py` to generate plots
3. **Generate Summary**: Run `study_d_summary.py` for readable report

## Key Findings

- GPT-4o-mini achieves best downstream performance (AUC = 0.683)
- Negative correlation between semantic F1 and downstream performance (r = -0.770)
- Positive correlation between hard F1 and downstream performance (r = 0.405)
- Models with higher semantic F1 do not necessarily produce better emotion features for downstream tasks

## Dependencies

- Python packages: numpy, pandas, scikit-learn, matplotlib, seaborn, torch, pyyaml
- Data: Model predictions in `logs/` directories with `indexed_metrics.yml` files
- Study B results: `logs/analysis/semantic_f1/table_b2_detailed_metrics.csv`
