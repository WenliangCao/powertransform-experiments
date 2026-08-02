#!/usr/bin/env python3
"""Aggregate experiment results and generate publication-ready figures."""

from __future__ import annotations

import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_RESULT_PATHS = [
    ROOT / "results" / "raw" / "results.csv",
    ROOT / "results" / "raw" / "stress_test_results.csv",
]
SUMMARY_PATH = ROOT / "results" / "summary.csv"
FIGURES_DIR = ROOT / "results" / "figures"

METHOD_ORDER = ["none", "standard", "robust", "yeo-johnson", "box-cox"]
METHOD_LABELS = {
    "none": "None",
    "standard": "Standard",
    "robust": "Robust",
    "yeo-johnson": "Yeo-Johnson",
    "box-cox": "Box-Cox",
}
METHOD_COLORS = {
    "none": "#F2F2F2",
    "standard": "#D9D9D9",
    "robust": "#BFBFBF",
    "yeo-johnson": "#8C8C8C",
    "box-cox": "#595959",
}
METHOD_HATCHES = {
    "none": "",
    "standard": "//",
    "robust": "..",
    "yeo-johnson": "xx",
    "box-cox": "++",
}
TASK_METRICS = {
    "regression": ("rmse", "r2", False),
    "classification": ("macro_f1", "accuracy", True),
    "clustering": ("ari", "silhouette", True),
}
DATASET_LABELS = {
    "california_housing": "California Housing",
    "abalone": "Abalone",
    "parkinsons_telemonitoring": "Parkinsons Telemonitoring",
    "breast_cancer": "Breast Cancer Wisconsin",
    "wine": "Wine",
    "htru2": "HTRU2",
    "iris": "Iris",
    "seeds": "Seeds",
    "wholesale_customers": "Wholesale Customers",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in RAW_RESULT_PATHS:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    if len(rows) != 205:
        raise ValueError(f"Expected 205 raw result rows, found {len(rows)}")
    if any(row["model_warning"] for row in rows):
        raise ValueError("Raw results contain model warnings")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["dataset"], row["method"])].append(row)

    summary: list[dict[str, object]] = []
    for (task, dataset, method), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], METHOD_ORDER.index(item[0][2]))
    ):
        primary, secondary, _ = TASK_METRICS[task]
        primary_mean, primary_std = mean_std([float(row[primary]) for row in group])
        secondary_mean, secondary_std = mean_std([float(row[secondary]) for row in group])
        preprocess_mean, preprocess_std = mean_std(
            [float(row["preprocess_seconds"]) for row in group]
        )
        model_mean, model_std = mean_std([float(row["model_seconds"]) for row in group])
        summary.append(
            {
                "task": task,
                "dataset": dataset,
                "method": method,
                "n_runs": len(group),
                "primary_metric": primary,
                "primary_mean": primary_mean,
                "primary_std": primary_std,
                "secondary_metric": secondary,
                "secondary_mean": secondary_mean,
                "secondary_std": secondary_std,
                "preprocess_mean_seconds": preprocess_mean,
                "preprocess_std_seconds": preprocess_std,
                "model_mean_seconds": model_mean,
                "model_std_seconds": model_std,
            }
        )
    if len(summary) != 41:
        raise ValueError(f"Expected 41 summary rows, found {len(summary)}")
    return summary


def write_summary(summary: list[dict[str, object]]) -> None:
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            formatted = dict(row)
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.12g}"
            writer.writerow(formatted)


def plot_dataset(dataset: str, rows: list[dict[str, object]]) -> None:
    task = str(rows[0]["task"])
    metric = str(rows[0]["primary_metric"])
    higher_is_better = TASK_METRICS[task][2]
    methods = [str(row["method"]) for row in rows]
    means = [float(row["primary_mean"]) for row in rows]
    standard_deviations = [float(row["primary_std"]) for row in rows]
    positions = np.arange(len(methods))

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=4,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="#000000",
        linewidth=0.7,
    )
    for bar, method in zip(bars, methods):
        bar.set_hatch(METHOD_HATCHES[method])
    axis.set_xticks(
        positions,
        [METHOD_LABELS[method] for method in methods],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel(metric.upper().replace("_", " "))
    direction = "higher is better" if higher_is_better else "lower is better"
    axis.set_title(f"{DATASET_LABELS[dataset]} - {metric.upper().replace('_', ' ')} ({direction})")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"{dataset}-{metric}.png", dpi=180)
    plt.close(figure)


def plot_runtime(summary: list[dict[str, object]]) -> None:
    methods = METHOD_ORDER[1:]
    datasets = list(DATASET_LABELS)
    figure, axis = plt.subplots(figsize=(12.2, 5.3))
    width = 0.18
    positions = np.arange(len(datasets))
    lookup = {(str(row["dataset"]), str(row["method"])): row for row in summary}
    for offset, method in enumerate(methods):
        values = []
        for dataset in datasets:
            row = lookup.get((dataset, method))
            values.append(float(row["preprocess_mean_seconds"]) if row else np.nan)
        axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="#000000",
            linewidth=0.7,
            hatch=METHOD_HATCHES[method],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Mean preprocessing pipeline time (seconds, log scale)")
    axis.set_xticks(positions, [DATASET_LABELS[name] for name in datasets], rotation=22, ha="right")
    axis.set_title("SystemDS preprocessing time")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "preprocessing-runtime.png", dpi=180)
    plt.close(figure)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FIGURES_DIR.exists():
        shutil.rmtree(FIGURES_DIR)
    FIGURES_DIR.mkdir(parents=True)
    write_summary(summary)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[str(row["dataset"])].append(row)
    for dataset, dataset_rows in grouped.items():
        plot_dataset(dataset, dataset_rows)
    plot_runtime(summary)
    print(f"wrote {len(summary)} summary rows and {len(grouped) + 1} figures")


if __name__ == "__main__":
    main()
