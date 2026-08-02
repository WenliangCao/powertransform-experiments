#!/usr/bin/env python3
"""Run the complete preprocessing and downstream-task experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.json"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
DML_DRIVER = ROOT / "dml" / "transform_split.dml"
WORK_DIR = ROOT / "work"
RESULTS_DIR = ROOT / "results"

METHODS = ["none", "standard", "robust", "yeo-johnson", "box-cox"]
TIMING_INDEX = {"standard": 0, "robust": 1, "yeo-johnson": 2, "box-cox": 3}


def git(systemds_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=systemds_root, text=True
    ).strip()


def locate_systemds_jar(systemds_root: Path) -> Path:
    candidates = sorted(
        (systemds_root / "target").glob("systemds-*-SNAPSHOT.jar"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    excluded = ("tests", "shaded", "unshaded", "perf", "ropt")
    candidates = [
        path for path in candidates if not any(token in path.name for token in excluded)
    ]
    if not candidates:
        raise FileNotFoundError("No SystemDS SNAPSHOT jar found; build SystemDS first")
    jar = candidates[0]
    required_sources = [
        systemds_root / "src" / "main" / "java" / "org" / "apache" / "sysds" / "common" / "Builtins.java",
        systemds_root / "scripts" / "builtin" / "powerTransform.dml",
        systemds_root / "scripts" / "builtin" / "powerTransformApply.dml",
    ]
    if any(source.stat().st_mtime > jar.stat().st_mtime for source in required_sources):
        raise RuntimeError(f"SystemDS jar is older than required sources: {jar}")
    return jar


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",", ndmin=2)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {path}")
    return matrix


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.17g")


def run_systemds_transformations(
    systemds_root: Path,
    jar: Path,
    dataset: str,
    seed: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    box_cox_eligible: bool,
    log_dir: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    run_dir = WORK_DIR / dataset / str(seed)
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    save_matrix(train_path, x_train)
    save_matrix(test_path, x_test)

    command = [
        str(systemds_root / "bin" / "systemds"),
        str(jar),
        "-f",
        str(DML_DRIVER),
        "-exec",
        "singlenode",
        "-nvargs",
        f"train={train_path}",
        f"test={test_path}",
        f"output={output_dir}",
        f"boxcox={'TRUE' if box_cox_eligible else 'FALSE'}",
    ]
    environment = os.environ.copy()
    environment["SYSTEMDS_ROOT"] = str(systemds_root)
    environment["SYSDS_QUIET"] = "1"
    completed = subprocess.run(
        command,
        cwd=systemds_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_dir / f"{dataset}-{seed}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or "An Error Occurred" in completed.stdout:
        raise RuntimeError(f"SystemDS failed for {dataset}/{seed}; see {log_path}")

    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "none": (x_train, x_test)
    }
    expected = ["standard", "robust", "yeo-johnson"]
    if box_cox_eligible:
        expected.append("box-cox")
    for method in expected:
        transformed_train = load_matrix(output_dir / f"{method}_train.csv")
        transformed_test = load_matrix(output_dir / f"{method}_test.csv")
        if (
            transformed_train.shape != x_train.shape
            or transformed_test.shape != x_test.shape
        ):
            raise ValueError(f"Unexpected {method} output shape for {dataset}/{seed}")
        transformed[method] = (transformed_train, transformed_test)

    timing_values = np.loadtxt(output_dir / "timings.csv", delimiter=",", ndmin=1)
    if timing_values.size != 4 or not np.isfinite(timing_values).all():
        raise ValueError(f"Invalid timing output for {dataset}/{seed}")
    timings = {"none": 0.0}
    for method in expected:
        timings[method] = float(timing_values[TIMING_INDEX[method]])
    return transformed, timings


def evaluate(
    task: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], float, str]:
    warning_message = ""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with np.errstate(all="ignore"):
            if task == "regression":
                model = KNeighborsRegressor(
                    n_neighbors=5,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                    n_jobs=1,
                )
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
            elif task == "classification":
                model = KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                    n_jobs=1,
                )
                model.fit(x_train, y_train.astype(int))
                predictions = model.predict(x_test)
            elif task == "clustering":
                cluster_count = int(np.unique(y_train).size)
                model = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed)
                predictions = model.fit_predict(x_train)
            else:
                raise ValueError(f"Unknown task: {task}")
        model_seconds = time.perf_counter() - start
        warning_message = " | ".join(str(item.message) for item in captured)

    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite predictions for {task}")

    if task == "regression":
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }
    elif task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }
    else:
        metrics = {
            "silhouette": float(silhouette_score(x_train, predictions)),
            "ari": float(adjusted_rand_score(y_train.astype(int), predictions)),
        }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite metrics for {task}")
    return metrics, model_seconds, warning_message


def split_data(
    task: str,
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split == "full_dataset":
        return x, x, y, y
    if split == "group_shuffle":
        if groups is None:
            raise ValueError("Group split requires group labels")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_index, test_index = next(splitter.split(x, y, groups))
        if set(groups[train_index]) & set(groups[test_index]):
            raise ValueError("Group split leaked labels between train and test")
        return x[train_index], x[test_index], y[train_index], y[test_index]
    if split not in {"random", "stratified_random"}:
        raise ValueError(f"Unknown split strategy: {split}")
    stratify = y if split == "stratified_random" else None
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "dataset",
        "method",
        "seed",
        "n_train",
        "n_test",
        "preprocess_seconds",
        "model_seconds",
        "rmse",
        "r2",
        "accuracy",
        "macro_f1",
        "silhouette",
        "ari",
        "model_warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systemds-root",
        type=Path,
        default=ROOT.parent / "systemds",
        help="Path to the SystemDS checkout under test",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Experiment configuration file",
    )
    parser.add_argument("--datasets", nargs="*", help="Optional dataset subset")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional seed subset")
    args = parser.parse_args()
    systemds_root = args.systemds_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_name = str(config.get("run_name", "core"))
    raw_results_path = RESULTS_DIR / "raw" / (
        "results.csv" if run_name == "core" else f"{run_name}_results.csv"
    )
    log_dir = RESULTS_DIR / "logs" if run_name == "core" else RESULTS_DIR / "logs" / run_name
    jar = locate_systemds_jar(systemds_root)
    systemds_dirty = bool(git(systemds_root, "status", "--porcelain"))
    if systemds_dirty:
        raise RuntimeError("SystemDS checkout must be clean before experiments")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if log_dir.exists():
        shutil.rmtree(log_dir)

    rows: list[dict[str, object]] = []
    selected_datasets = args.datasets or list(config["datasets"])
    unknown_datasets = set(selected_datasets) - set(config["datasets"])
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")
    selected_seeds = args.seeds or config["seeds"]
    for dataset in selected_datasets:
        dataset_config = config["datasets"][dataset]
        task = dataset_config["task"]
        data_dir = ROOT / "data" / "processed" / dataset
        x = load_matrix(data_dir / "X.csv")
        y = np.loadtxt(data_dir / "y.csv", delimiter=",", ndmin=1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/target row mismatch for {dataset}")
        box_cox_eligible = bool(manifest[dataset]["box_cox_eligible"])
        if task == "clustering":
            default_split = "full_dataset"
        elif task == "classification":
            default_split = "stratified_random"
        else:
            default_split = "random"
        split = str(dataset_config.get("split", default_split))
        groups_path = data_dir / "groups.csv"
        groups = (
            np.loadtxt(groups_path, delimiter=",", ndmin=1)
            if split == "group_shuffle"
            else None
        )

        for seed in selected_seeds:
            x_train, x_test, y_train, y_test = split_data(
                task, split, x, y, seed, float(config["test_size"]), groups
            )
            transformed, timings = run_systemds_transformations(
                systemds_root,
                jar,
                dataset,
                seed,
                x_train,
                x_test,
                box_cox_eligible,
                log_dir,
            )
            for method in METHODS:
                if method not in transformed:
                    continue
                transformed_train, transformed_test = transformed[method]
                metrics, model_seconds, warning_message = evaluate(
                    task,
                    transformed_train,
                    transformed_test,
                    y_train,
                    y_test,
                    seed,
                )
                row: dict[str, object] = {
                    "task": task,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "n_train": x_train.shape[0],
                    "n_test": x_test.shape[0],
                    "preprocess_seconds": f"{timings[method]:.9f}",
                    "model_seconds": f"{model_seconds:.9f}",
                    "rmse": "",
                    "r2": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "silhouette": "",
                    "ari": "",
                    "model_warning": warning_message,
                }
                row.update({name: f"{value:.12g}" for name, value in metrics.items()})
                rows.append(row)
                print(f"{dataset} seed={seed} method={method} {metrics}", flush=True)

    write_csv(raw_results_path, rows)
    print(f"wrote {len(rows)} rows to {raw_results_path}")


if __name__ == "__main__":
    main()
