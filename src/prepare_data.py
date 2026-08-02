#!/usr/bin/env python3
"""Download, normalize, and fingerprint the experiment datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
    load_wine,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ABALONE_SOURCE = "https://archive.ics.uci.edu/dataset/1/abalone"
SEEDS_SOURCE = "https://archive.ics.uci.edu/dataset/236/seeds"
PARKINSONS_SOURCE = "https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring"
HTRU2_SOURCE = "https://archive.ics.uci.edu/dataset/372/htru2"
WHOLESALE_SOURCE = "https://archive.ics.uci.edu/dataset/292/wholesale+customers"

ABALONE_DOWNLOAD = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
SEEDS_DOWNLOAD = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
PARKINSONS_DOWNLOAD = (
    "https://archive.ics.uci.edu/static/public/189/parkinsons%2Btelemonitoring.zip"
)
HTRU2_DOWNLOAD = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
WHOLESALE_DOWNLOAD = (
    "https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SystemDS-PowerTransformer-Experiment/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def load_abalone() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    archive = RAW_DIR / "abalone.zip"
    download(ABALONE_DOWNLOAD, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("abalone.data"))
        rows = list(csv.reader(io.TextIOWrapper(bundle.open(member), encoding="utf-8")))
    values = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    feature_names = [
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]
    x = values[:, :-1]
    y = values[:, -1]
    valid = np.all(x > 0.0, axis=1)
    if int(np.sum(~valid)) != 2:
        raise ValueError("Expected exactly two Abalone records with non-positive measurements")
    return x[valid], y[valid], feature_names, ABALONE_SOURCE


def load_seeds() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    archive = RAW_DIR / "seeds.zip"
    download(SEEDS_DOWNLOAD, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if "seeds_dataset" in name)
        text = bundle.read(member).decode("utf-8", errors="replace")
    values = np.loadtxt(io.StringIO(text), dtype=np.float64)
    feature_names = [
        "area",
        "perimeter",
        "compactness",
        "kernel_length",
        "kernel_width",
        "asymmetry_coefficient",
        "kernel_groove_length",
    ]
    return values[:, :-1], values[:, -1].astype(int) - 1, feature_names, SEEDS_SOURCE


def load_parkinsons() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    archive = RAW_DIR / "parkinsons_telemonitoring.zip"
    download(PARKINSONS_DOWNLOAD, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("parkinsons_updrs.data"))
        text = bundle.read(member).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    excluded = {"subject#", "motor_UPDRS", "total_UPDRS"}
    feature_names = [name for name in rows[0] if name not in excluded]
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row["total_UPDRS"]) for row in rows], dtype=np.float64)
    groups = np.asarray([int(row["subject#"]) for row in rows], dtype=np.int64)
    return x, y, groups, feature_names, PARKINSONS_SOURCE


def load_htru2() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    archive = RAW_DIR / "htru2.zip"
    download(HTRU2_DOWNLOAD, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("HTRU_2.csv"))
        text = bundle.read(member).decode("utf-8-sig")
    rows = csv.reader(io.StringIO(text, newline=""), delimiter=",")
    values = np.asarray(
        [[float(value) for value in row] for row in rows if row],
        dtype=np.float64,
    )
    feature_names = [
        "profile_mean",
        "profile_stdev",
        "profile_kurtosis",
        "profile_skewness",
        "dm_mean",
        "dm_stdev",
        "dm_kurtosis",
        "dm_skewness",
    ]
    return values[:, :-1], values[:, -1].astype(int), feature_names, HTRU2_SOURCE


def load_wholesale() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    archive = RAW_DIR / "wholesale_customers.zip"
    download(WHOLESALE_DOWNLOAD, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(".csv"))
        text = bundle.read(member).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    feature_names = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen",
    ]
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["Channel"]) - 1 for row in rows], dtype=np.int64)
    return x, y, feature_names, WHOLESALE_SOURCE


def datasets() -> dict[str, tuple[np.ndarray, np.ndarray, list[str], str, str, Optional[np.ndarray]]]:
    california = fetch_california_housing(data_home=RAW_DIR / "sklearn")
    breast_cancer = load_breast_cancer()
    wine = load_wine()
    iris = load_iris()
    abalone_x, abalone_y, abalone_features, abalone_source = load_abalone()
    seeds_x, seeds_y, seeds_features, seeds_source = load_seeds()
    parkinsons_x, parkinsons_y, parkinsons_groups, parkinsons_features, parkinsons_source = (
        load_parkinsons()
    )
    htru2_x, htru2_y, htru2_features, htru2_source = load_htru2()
    wholesale_x, wholesale_y, wholesale_features, wholesale_source = load_wholesale()

    return {
        "california_housing": (
            california.data,
            california.target,
            list(california.feature_names),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
            "regression",
            None,
        ),
        "abalone": (abalone_x, abalone_y, abalone_features, abalone_source, "regression", None),
        "breast_cancer": (
            breast_cancer.data,
            breast_cancer.target,
            list(breast_cancer.feature_names),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
            "classification",
            None,
        ),
        "wine": (
            wine.data,
            wine.target,
            list(wine.feature_names),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
            "classification",
            None,
        ),
        "iris": (
            iris.data,
            iris.target,
            list(iris.feature_names),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
            "clustering",
            None,
        ),
        "seeds": (seeds_x, seeds_y, seeds_features, seeds_source, "clustering", None),
        "parkinsons_telemonitoring": (
            parkinsons_x,
            parkinsons_y,
            parkinsons_features,
            parkinsons_source,
            "regression",
            parkinsons_groups,
        ),
        "htru2": (htru2_x, htru2_y, htru2_features, htru2_source, "classification", None),
        "wholesale_customers": (
            wholesale_x,
            wholesale_y,
            wholesale_features,
            wholesale_source,
            "clustering",
            None,
        ),
    }


def write_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    source: str,
    task: str,
    groups: Optional[np.ndarray],
) -> dict[str, object]:
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{name} contains non-finite values")
    destination = PROCESSED_DIR / name
    destination.mkdir(parents=True, exist_ok=True)
    x_path = destination / "X.csv"
    y_path = destination / "y.csv"
    np.savetxt(x_path, x, delimiter=",", fmt="%.17g")
    np.savetxt(y_path, y.reshape(-1, 1), delimiter=",", fmt="%.17g")
    metadata = {
        "name": name,
        "task": task,
        "source": source,
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "feature_names": feature_names,
        "target_values": int(np.unique(y).size),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "box_cox_eligible": bool(np.all(x > 0.0)),
        "x_sha256": sha256(x_path),
        "y_sha256": sha256(y_path),
    }
    if groups is not None:
        if groups.shape[0] != x.shape[0]:
            raise ValueError(f"{name} group/feature row mismatch")
        groups_path = destination / "groups.csv"
        np.savetxt(groups_path, groups.reshape(-1, 1), delimiter=",", fmt="%.17g")
        metadata["group_values"] = int(np.unique(groups).size)
        metadata["groups_sha256"] = sha256(groups_path)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        name: write_dataset(name, *values)
        for name, values in datasets().items()
    }
    manifest_path = ROOT / "data" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, metadata in manifest.items():
        print(
            f"{name}: {metadata['rows']}x{metadata['features']}, "
            f"box_cox={metadata['box_cox_eligible']}"
        )


if __name__ == "__main__":
    main()
