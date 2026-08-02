# SystemDS PowerTransformer Experiments

This repository contains the reproducible evaluation of the Apache SystemDS
`PowerTransformer` built-ins. It is maintained separately from the SystemDS
source repository so that experiment code, datasets, raw results, figures, and
the final report remain isolated from the upstream implementation.

## Experiment design

| Task | Datasets | Estimator | Primary metric |
| --- | --- | --- | --- |
| Regression | California Housing, Abalone, Parkinsons Telemonitoring | 5-nearest-neighbors regressor | RMSE (lower is better) |
| Classification | Breast Cancer Wisconsin, Wine, HTRU2 | 5-nearest-neighbors classifier | Macro-F1 (higher is better) |
| Clustering | Iris, Seeds, Wholesale Customers | K-means | ARI (higher is better) |

Each task compares no scaling, standard scaling, robust scaling,
Yeo-Johnson, and Box-Cox. Box-Cox is evaluated only when every feature is
strictly positive. The supervised estimators use five neighbors, uniform
weights, and Euclidean distance. K-means uses the known class count and 20
initializations.

The protocol uses the fixed seeds `13`, `37`, `73`, `101`, and `149`.
Supervised tasks use 80/20 train/test splits; classification is stratified and
Parkinsons Telemonitoring uses subject-level group splits. Every transformation
is fitted on training data and applied to test data. Clustering uses the full
dataset for each seed.

## Requirements

- Python 3.9 or later
- Apache SystemDS built from the PowerTransformer branch
- Java compatible with the SystemDS checkout
- [Tectonic](https://tectonic-typesetting.github.io/) for the PDF report

The default layout expects the SystemDS and experiment repositories to be
sibling directories.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_all.sh
```

Use a different SystemDS checkout with:

```bash
./run_all.sh --systemds-root /path/to/systemds
```

The runner requires a clean SystemDS worktree and a current SystemDS build. It
downloads or loads the public datasets, prepares deterministic CSV inputs, runs
both experiment configurations, aggregates 205 method-level results,
regenerates ten figures, and compiles the LaTeX report.

## Artifacts

- `results/raw/`: five-seed results for the core and skew-focused runs
- `results/summary.csv`: combined method-level means and standard deviations
- `results/figures/`: nine task figures and one preprocessing-time figure
- `output/pdf/powertransform-experiment-report.pdf`: final English report
- `output/spreadsheet/powertransform-leaderboard.xlsx`: English leaderboard
- `output/spreadsheet/powertransform-leaderboard-preview.png`: leaderboard preview

The dataset inventory, citations, and third-party terms are documented in
[`DATASETS.md`](DATASETS.md). Project code and documentation are licensed under
the Apache License 2.0; third-party datasets remain subject to their respective
terms.
