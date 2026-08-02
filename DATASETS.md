# Dataset Sources

The experiment uses public tabular datasets loaded by `src/prepare_data.py`.
Raw and processed snapshots are retained for exact reproduction; SHA-256
checksums are stored in `data/manifest.json`.

| Dataset | Loader or source | Citation |
| --- | --- | --- |
| California Housing | scikit-learn `fetch_california_housing` | Pace, R. K. and Barry, R. (1997), *Sparse Spatial Autoregressions*, Statistics & Probability Letters 33(3), 291-297 |
| Abalone | UCI Machine Learning Repository | Nash, W. et al. (1994), DOI: [10.24432/C55C7W](https://doi.org/10.24432/C55C7W) |
| Parkinsons Telemonitoring | UCI Machine Learning Repository | Tsanas, A. and Little, M. (2009), DOI: [10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N) |
| Breast Cancer Wisconsin (Diagnostic) | scikit-learn `load_breast_cancer` | Wolberg, W. et al. (1993), DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) |
| Wine | scikit-learn `load_wine` | Aeberhard, S. and Forina, M. (1992), DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) |
| HTRU2 | UCI Machine Learning Repository | Lyon, R. (2015), DOI: [10.24432/C5DK6R](https://doi.org/10.24432/C5DK6R) |
| Iris | scikit-learn `load_iris` | Fisher, R. A. (1936), DOI: [10.24432/C56C76](https://doi.org/10.24432/C56C76) |
| Seeds | UCI Machine Learning Repository | Charytanowicz, M. et al. (2010), DOI: [10.24432/C5H30K](https://doi.org/10.24432/C5H30K) |
| Wholesale Customers | UCI Machine Learning Repository | Cardoso, M. (2013), DOI: [10.24432/C5030X](https://doi.org/10.24432/C5030X) |

The eight UCI datasets listed above are published under the Creative Commons
Attribution 4.0 International license. The California Housing copy is obtained
through scikit-learn from the StatLib repository. The experiment code does not
alter target values. It removes the categorical sex feature and two records
with non-positive physical measurements from Abalone so that the retained
numeric feature matrix is valid for Box-Cox.
