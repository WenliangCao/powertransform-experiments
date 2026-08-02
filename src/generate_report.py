#!/usr/bin/env python3
"""Generate LaTeX tables and compile the experiment report."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
PDF_DIR = ROOT / "output" / "pdf"
SUMMARY_PATH = ROOT / "results" / "summary.csv"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

METHOD_ORDER = ["none", "standard", "robust", "yeo-johnson", "box-cox"]
METHOD_LABELS = {
    "none": "No scaling",
    "standard": "Standard",
    "robust": "Robust",
    "yeo-johnson": "Yeo-Johnson",
    "box-cox": "Box-Cox",
}
DATASET_ORDER = [
    "california_housing",
    "abalone",
    "parkinsons_telemonitoring",
    "breast_cancer",
    "wine",
    "htru2",
    "iris",
    "seeds",
    "wholesale_customers",
]
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
TASK_DATASETS = {
    "regression": DATASET_ORDER[:3],
    "classification": DATASET_ORDER[3:6],
    "clustering": DATASET_ORDER[6:],
}
TASK_METRICS = {
    "regression": ("rmse", "r2", False, True),
    "classification": ("macro_f1", "accuracy", True, True),
    "clustering": ("ari", "silhouette", True, True),
}
STRESS_DATASETS = {
    "parkinsons_telemonitoring",
    "htru2",
    "wholesale_customers",
}


def escape(value: object) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def load_summary() -> list[dict[str, str]]:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 41:
        raise ValueError(f"Expected 41 summary rows, found {len(rows)}")
    if {int(row["n_runs"]) for row in rows} != {5}:
        raise ValueError("Every summary row must contain five runs")
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["dataset"]][row["method"]] = row
    return grouped


def metric_cell(row: dict[str, str], prefix: str, bold: bool = False) -> str:
    text = (
        f"${float(row[f'{prefix}_mean']):.4f} "
        f"\\pm {float(row[f'{prefix}_std']):.4f}$"
    )
    return f"\\textbf{{{text}}}" if bold else text


def write_dataset_table(manifest: dict[str, dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Dataset & Task & Rows & Features & Study set & Box-Cox \\",
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        metadata = manifest[dataset]
        study_set = "Skew-focused" if dataset in STRESS_DATASETS else "Core"
        eligible = "Yes" if metadata["box_cox_eligible"] else "No"
        lines.append(
            f"{escape(DATASET_LABELS[dataset])} & "
            f"{escape(str(metadata['task']).title())} & "
            f"{int(metadata['rows']):,} & {int(metadata['features'])} & "
            f"{study_set} & {eligible} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Dataset matrix. Box-Cox is used only for strictly positive feature matrices.}",
            r"\label{tab:datasets}",
            r"\end{table}",
        ]
    )
    (GENERATED_DIR / "datasets.tex").write_text("\n".join(lines) + "\n")


def write_result_table(
    grouped: dict[str, dict[str, dict[str, str]]],
    task: str,
    prefix: str,
    metric: str,
    higher_is_better: bool,
) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Dataset & "
        + " & ".join(escape(METHOD_LABELS[method]) for method in METHOD_ORDER)
        + r" \\",
        r"\midrule",
    ]
    for dataset in TASK_DATASETS[task]:
        dataset_rows = grouped[dataset]
        means = [float(row[f"{prefix}_mean"]) for row in dataset_rows.values()]
        best = (max if higher_is_better else min)(means)
        cells = []
        for method in METHOD_ORDER:
            row = dataset_rows.get(method)
            cells.append(
                "--"
                if row is None
                else metric_cell(
                    row,
                    prefix,
                    abs(float(row[f"{prefix}_mean"]) - best) <= 1e-12,
                )
            )
        lines.append(
            f"{escape(DATASET_LABELS[dataset])} & " + " & ".join(cells) + r" \\"
        )
    label = f"tab:{task}-{prefix}"
    caption = f"{task.title()} {escape(metric)} results across five fixed seeds (mean $\\pm$ standard deviation)."
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    (GENERATED_DIR / f"{task}-{prefix}.tex").write_text(
        "\n".join(lines) + "\n"
    )


def write_runtime_table(grouped: dict[str, dict[str, dict[str, str]]]) -> None:
    methods = METHOD_ORDER[1:]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Dataset & "
        + " & ".join(escape(METHOD_LABELS[method]) for method in methods)
        + r" \\",
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        cells = []
        for method in methods:
            row = grouped[dataset].get(method)
            cells.append(
                "--" if row is None else f"{float(row['preprocess_mean_seconds']):.4f}"
            )
        lines.append(
            f"{escape(DATASET_LABELS[dataset])} & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Mean SystemDS preprocessing pipeline time in seconds.}",
            r"\label{tab:runtime}",
            r"\end{table}",
        ]
    )
    (GENERATED_DIR / "runtime.tex").write_text("\n".join(lines) + "\n")


def generate() -> None:
    rows = load_summary()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    grouped = group_rows(rows)
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset_table(manifest)
    for task, metrics in TASK_METRICS.items():
        primary, secondary, primary_higher, secondary_higher = metrics
        write_result_table(grouped, task, "primary", primary.upper(), primary_higher)
        write_result_table(
            grouped,
            task,
            "secondary",
            secondary.upper(),
            secondary_higher,
        )
    write_runtime_table(grouped)


def compile_report() -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("Tectonic is required to compile the LaTeX report")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            tectonic,
            "powertransform-experiment-report.tex",
            "--outdir",
            str(PDF_DIR),
        ],
        cwd=REPORT_DIR,
        check=True,
    )


def main() -> None:
    generate()
    compile_report()
    print("wrote LaTeX tables and compiled the experiment report")


if __name__ == "__main__":
    main()
