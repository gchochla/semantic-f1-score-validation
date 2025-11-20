import argparse
import os
import sys

import numpy as np
import yaml
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    mean_squared_error,
    mean_absolute_error,
)
from semantic_f1_score.semantic_f1 import (
    samples_semantic_f1_score,
    semantic_micro_f1_score,
    semantic_macro_f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


from llm_subj import DATASETS


def load_indexed_metrics(path: str) -> dict[str, dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Malformed YAML at {path}")
    return data


def extract_gt_pred_for_experiment(
    exp_data: dict,
) -> tuple[list[list[str]], list[list[str]]]:
    gts: list[list[str]] = []
    preds: list[list[str]] = []
    outs: list[list[str]] = []
    # Remove optional descriptive fields
    for k, v in exp_data.items():
        if k == "description":
            continue
        # Each v corresponds to an example entry
        try:
            gt = v["test_gt"]
            pr = v["test_preds"]
            out = v["test_outs"]
        except KeyError as e:
            continue  # dev example
        # Normalize to list of strings
        gt = gt if isinstance(gt, list) else [gt]
        pr = pr if isinstance(pr, list) else [pr]
        out = out if isinstance(out, list) else [out]
        gts.append([str(x) for x in gt])
        preds.append([str(x) for x in pr])
        outs.append([str(x) for x in out])
    return gts, preds, outs


def compute_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    metrics: list[str],
    semantic_matrix: pd.DataFrame | None = None,
    label_to_value_fn=lambda x: x,
) -> dict[str, float]:
    # Use sklearn for hard metrics
    all_labels = sorted({l for ex in (y_true + y_pred) for l in ex})
    mlb = MultiLabelBinarizer(classes=all_labels)
    Yt = mlb.fit_transform(y_true)
    Yp = mlb.transform(y_pred)

    # import pdb

    # pdb.set_trace()

    out: dict[str, float] = {}
    for m in metrics:
        if m == "sample_f1":
            out["test_sample_f1"] = float(
                f1_score(Yt, Yp, average="samples", zero_division=0)
            )
        elif m == "micro_f1":
            out["test_micro_f1"] = float(
                f1_score(Yt, Yp, average="micro", zero_division=0)
            )
        elif m == "macro_f1":
            out["test_macro_f1"] = float(
                f1_score(Yt, Yp, average="macro", zero_division=0)
            )
        elif m == "jaccard":
            out["test_jaccard_score"] = float(
                jaccard_score(Yt, Yp, average="samples", zero_division=1)
            )
        elif m == "mse":
            y_true_regr = [
                label_to_value_fn(l[0]) if l else 0.0 for l in y_true
            ]
            y_pred_regr = [
                label_to_value_fn(l[0]) if l else 0.0 for l in y_pred
            ]
            out["test_mse"] = float(
                mean_squared_error(y_true_regr, y_pred_regr)
            )
        elif m == "mae":
            y_true_regr = [
                label_to_value_fn(l[0]) if l else 0.0 for l in y_true
            ]
            y_pred_regr = [
                label_to_value_fn(l[0]) if l else 0.0 for l in y_pred
            ]
            out["test_mae"] = float(
                mean_absolute_error(y_true_regr, y_pred_regr)
            )
        elif m == "semantic_samples":
            if semantic_matrix is None:
                raise ValueError(
                    "semantic_samples requested but no semantic similarity matrix provided"
                )
            results = samples_semantic_f1_score(
                y_pred, y_true, semantic_matrix, return_components=True
            )
            out["test_semantic_f1"] = float(results["f1"])
            out["test_semantic_precision"] = float(results["precision"])
            out["test_semantic_recall"] = float(results["recall"])
        elif m == "semantic_micro":
            if semantic_matrix is None:
                raise ValueError(
                    "semantic_micro requested but no semantic similarity matrix provided"
                )
            results = semantic_micro_f1_score(
                y_pred, y_true, semantic_matrix, return_components=True
            )
            out["test_micro_semantic_f1"] = float(results["f1"])
            out["test_micro_semantic_precision"] = float(results["precision"])
            out["test_micro_semantic_recall"] = float(results["recall"])
        elif m == "semantic_macro":
            if semantic_matrix is None:
                raise ValueError(
                    "semantic_macro requested but no semantic similarity matrix provided"
                )
            results = semantic_macro_f1_score(
                y_pred, y_true, semantic_matrix, return_components=True
            )
            out["test_macro_semantic_f1"] = float(results["f1"])
            out["test_macro_semantic_precision"] = float(results["precision"])
            out["test_macro_semantic_recall"] = float(results["recall"])
        else:
            raise ValueError(f"Unknown metric: {m}")
    return out


def _dataset_class_from_task(task: str):
    try:
        return DATASETS[task]
    except KeyError:
        raise ValueError(
            f"Unknown task '{task}': not found in llm_subj.DATASETS"
        )


def _load_params_first_experiment(exp_dir: str) -> dict:
    p = os.path.join(exp_dir, "params.yml")
    if not os.path.exists(p):
        raise FileNotFoundError(f"params.yml not found in {exp_dir}")
    with open(p, "r") as f:
        params = yaml.safe_load(f)
    # pick smallest experiment_* key
    exp_keys = sorted(
        k for k in params.keys() if str(k).startswith("experiment_")
    )
    if not exp_keys:
        raise ValueError("No experiment_* entries in params.yml")
    return params[exp_keys[0]]


def _build_dataset_for_exp(exp_dir: str):
    exp_params = _load_params_first_experiment(exp_dir)
    task = exp_params.get("task")
    if not task:
        raise ValueError("'task' not found in params.yml experiment block")

    DatasetCls = _dataset_class_from_task(task)

    # Build kwargs using dataset argparse_args to avoid passing invalid keys
    arg_keys = set()
    if hasattr(DatasetCls, "argparse_args"):
        try:
            arg_keys = set(DatasetCls.argparse_args().keys())
        except Exception:
            arg_keys = set()

    kwargs = {}
    # Always try to pass root_dir if present
    if "root_dir" in exp_params:
        kwargs["root_dir"] = exp_params["root_dir"]
    # Use test_splits for constructing the dataset
    splits = exp_params.get("test_splits", "test")
    kwargs["splits"] = splits

    # Add any keys from params that the dataset declares in argparse_args
    if arg_keys:
        for k in arg_keys - {"root_dir", "splits"}:
            if k in exp_params:
                kwargs[k] = exp_params[k]

    for k in kwargs:
        if k == "text_preprocessor":
            # substitute the script processing with a simple identity function
            kwargs[k] = lambda x: x

    # Instantiate dataset; wrapper classes accept **kwargs and forward safely
    ds = DatasetCls(**kwargs)
    return ds


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    half_width = 1.96 * std / (len(arr) ** 0.5)
    return mean, half_width


def format_mean_ci(mean: float, hw: float) -> str:
    return f"{mean:.4f}+-{hw:.4f}"


def process_experiment_dir(
    exp_dir: str,
    metrics: list[str],
    overwrite: bool = True,
) -> None:
    """Process a single experiment directory containing `indexed_metrics.yml`.

    - Writes per-experiment results to `metrics.yml`.
    - Writes aggregated mean ± 95% CI per description to `aggregated_metrics.yml`.
    """

    idx_path = os.path.join(exp_dir, "indexed_metrics.yml")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No indexed_metrics.yml in {exp_dir}")

    ds = _build_dataset_for_exp(exp_dir)
    data = load_indexed_metrics(idx_path)

    # Compute per-experiment metrics
    per_exp_results: dict[str, dict[str, float]] = {}
    descriptions: dict[str, str] = {}
    for exp_key, exp_data in data.items():
        if not exp_key.startswith("experiment_"):
            # Skip stray fields
            continue
        desc = exp_data.get("description", "")
        descriptions[exp_key] = desc
        y_true, y_pred, _ = extract_gt_pred_for_experiment(exp_data)
        # Prepare semantic similarity matrix from the dataset if requested
        semantic_matrix = None
        if any(m.startswith("semantic_") for m in metrics):
            semantic_matrix = getattr(ds, "label_similarity", None)
            if semantic_matrix is None:
                raise ValueError(
                    "Semantic metrics requested but dataset provides no label_similarity"
                )

        res = compute_metrics(
            y_true,
            y_pred,
            metrics,
            semantic_matrix,
            ds.regression_value_for_label if ds.regression else (lambda x: x),
        )
        # Include description for later aggregation
        res_with_desc = {"description": desc} | res
        per_exp_results[exp_key] = res_with_desc

    # Write/merge metrics.yml
    metrics_path = os.path.join(exp_dir, "metrics.yml")
    if (not overwrite) and os.path.exists(metrics_path):
        print(f"metrics.yml exists, skipping write: {metrics_path}")
    else:
        existing_metrics: dict[str, dict[str, float]] = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        existing_metrics = loaded
            except Exception:
                existing_metrics = {}

        # Merge per-experiment results into existing metrics non-destructively
        merged_metrics = dict(existing_metrics)
        for exp_key, res in per_exp_results.items():
            merged = dict(existing_metrics.get(exp_key, {}))
            merged.update(res)  # add/overwrite only the computed keys
            merged_metrics[exp_key] = merged

        with open(metrics_path, "w") as f:
            yaml.dump(merged_metrics, f)

    # Aggregate by description: mean ± 95% CI for each metric
    # Collect metric keys from first experiment (excluding description)
    metric_keys = [
        k
        for k in next(iter(per_exp_results.values())).keys()
        if k != "description"
    ]

    # Group values by description
    grouped: dict[str, dict[str, list[float]]] = {}
    for _, res in per_exp_results.items():
        desc = res.get("description", "")
        for mk in metric_keys:
            grouped.setdefault(desc, {}).setdefault(mk, []).append(
                float(res[mk])
            )

    aggregated: dict[str, dict[str, str]] = {}
    for desc, mk_vals in grouped.items():
        aggregated[desc] = {}
        for mk, vals in mk_vals.items():
            mean, hw = mean_ci95(vals)
            aggregated[desc][mk] = format_mean_ci(mean, hw)

    # Write/merge aggregated_metrics.yml
    agg_path = os.path.join(exp_dir, "aggregated_metrics.yml")
    if (not overwrite) and os.path.exists(agg_path):
        print(f"aggregated_metrics.yml exists, skipping write: {agg_path}")
    else:
        existing_agg: dict[str, dict[str, str]] = {}
        if os.path.exists(agg_path):
            try:
                with open(agg_path, "r") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        existing_agg = loaded
            except Exception:
                existing_agg = {}

        merged_agg = dict(existing_agg)
        for desc, mkvals in aggregated.items():
            merged_desc = dict(existing_agg.get(desc, {}))
            merged_desc.update(mkvals)  # add/overwrite only computed metrics
            merged_agg[desc] = merged_desc

        with open(agg_path, "w") as f:
            yaml.dump(merged_agg, f)


def iter_experiment_dirs(root: str, recursive: bool) -> list[str]:
    if os.path.isfile(root):
        raise ValueError("Root path must be a directory")
    if not recursive:
        if os.path.exists(os.path.join(root, "indexed_metrics.yml")):
            return [root]
        else:
            raise FileNotFoundError(
                f"No indexed_metrics.yml found directly under {root}. Use --recursive to search."
            )
    # Recursive search: directories containing indexed_metrics.yml
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "indexed_metrics.yml" in filenames:
            out.append(dirpath)
    return sorted(out)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute metrics from indexed_metrics.yml and aggregate with 95% CIs."
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="One or more directories to process, or a single root with --recursive.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["sample_f1"],
        choices=[
            "sample_f1",
            "micro_f1",
            "macro_f1",
            "jaccard",
            "semantic_samples",
            "semantic_micro",
            "semantic_macro",
            "mse",
            "mae",
        ],
        help="Metrics to compute (default: sample_f1).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for experiment dirs under given paths.",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing metrics.yml or aggregated_metrics.yml.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    overwrite = not args.no_overwrite

    all_dirs: list[str] = []
    for p in args.paths:
        all_dirs.extend(iter_experiment_dirs(p, recursive=args.recursive))

    if not all_dirs:
        print("No experiment directories found.")
        return

    for d in all_dirs:
        try:
            print(f"Processing: {d}")
            process_experiment_dir(d, metrics=args.metrics, overwrite=overwrite)
        except Exception as e:
            raise e
            print(f"Error processing {d}: {e}")


if __name__ == "__main__":
    main()
