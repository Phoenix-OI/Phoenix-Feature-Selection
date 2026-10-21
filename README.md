# Phoenix-OI: Cross-Phase Orthogonal Inference for Expensive High-Dimensional Wrapper Feature Selection

[![Dataset and benchmark results on Zenodo](https://img.shields.io/badge/Data%20%26%20results-Zenodo-blue)](https://doi.org/10.5281/zenodo.22503472)

**Phoenix-OI** is a particle swarm optimization (PSO)-based algorithm for high-dimensional wrapper feature selection. It applies a common **Orthogonal Inference (OI)** evaluation-to-decision procedure to three search phases: feature-level initialization, segment-level intelligent movement, and elite-only two-stage refinement.

**Where to find the resources**

| Resource | Location |
| --- | --- |
| Functional Phoenix-OI implementation and usage instructions | **This GitHub repository**: <https://github.com/Phoenix-OI/Phoenix-Feature-Selection> |
| **Dataset63**, original and standardized variants | **Zenodo**: <https://doi.org/10.5281/zenodo.22503472> |
| Dataset provenance, metadata, quality-control information, and file hashes | **Same Zenodo record** |
| Benchmark results and supporting experimental materials | **Same Zenodo record** |
| **Complete 63-dataset benchmark comparison table** (accuracy and selected features) | **[See the table below](#complete-63-dataset-benchmark-results)**; also in [`docs/benchmark_results.md`](docs/benchmark_results.md) |
| Detailed experimental protocols and statistics | Phoenix-OI manuscript and supplementary material (paper citation to be added when available) |

**Data and experimental output files are intentionally not stored in this GitHub repository.** Download them separately from Zenodo. The DOI above identifies the **dataset/benchmark archive**, not a software-release DOI.

## Algorithm at a glance

Phoenix-OI implements the orthogonal experimental design (OED), orthogonal-array (OA) trial evaluation, main-effect estimation, and main-effect-driven (MED) candidate construction used for OI across these phases:

1. **Initialization OI:** feature-inclusion/exclusion choices define two-level factors and generate a structured initial swarm.
2. **OI-guided Intelligent Move Mechanism (IMM):** global-learning and personal-learning proposals are combined through segment-level OI.
3. **Elite-only refinement:** incumbent/complement recombination followed by retain/remove reduction is guided by two-stage OI.

The default feature-subset objective is

$$
F(S) = \mathrm{Acc}_{\mathrm{CV}}(S) - \lambda |S|, \qquad \lambda = 0.001.
$$

where `Acc_CV` is the mean five-fold cross-validation accuracy and `|S|` is the number of selected features. The primary classifier is a **k-nearest-neighbors (KNN) classifier with k = 3**. Binarization selects a feature only when its position exceeds `0.5`.

**FFE definition:** one actual fitness-function evaluation of a **nonempty** candidate subset is one complete cross-validated wrapper evaluation (five folds by default), **not five FFEs**. Repeating an evaluation incurs another FFE; empty subsets are not counted. The script prints stage-wise and total search-FFE counts.

The strict **No-ANOVA** setting is the default. Constant-feature removal is enabled by default, whereas supervised ANOVA filtering is **optional and disabled by default**. The optional GPSISF-style score and post-hoc Pareto report do **not** constitute the full multi-objective/archive method in the supplementary experiments.

## Repository layout

```text
Phoenix-Feature-Selection/
├── Phoenix.py              # Functional optimizer + CSV command-line interface
├── oa_package/
│   ├── __init__.py
│   └── oa.py               # Orthogonal-array support module
├── docs/
│   ├── benchmark_results.md    # Complete 63-dataset table
│   ├── dataset_description.md
│   ├── method_overview.md
│   ├── release_checklist.md
│   └── usage.md
├── README.md
├── requirements.txt
├── CITATION.cff
├── ATTRIBUTION.md
├── LICENSE                 # Apache License 2.0: original software
├── LICENSE-DATA            # CC BY 4.0: original project documentation/results
├── LICENSE_SUMMARY.md
├── NOTICE
└── .gitignore
```

**Not in this repository:** `dataset63/`, `data/`, original/standardized dataset archives, benchmark `.csv` result tables, and run outputs. These are distributed through Zenodo or remain in a local, Git-ignored workspace. **The summary benchmark table is deliberately mirrored in README and `docs/benchmark_results.md` for convenient viewing.** This GitHub release also includes the provided `docs/` and `oa_package/` source files.

## Installation

For the primary paper benchmark, the reported environment uses **Python 3.9.21**. To reproduce the documented dependency versions as closely as the available packages permit:

```bash
git clone https://github.com/Phoenix-OI/Phoenix-Feature-Selection.git
cd Phoenix-Feature-Selection
conda create -n phoenix-oi python=3.9.21 -y
conda activate phoenix-oi
python -m pip install -r requirements.txt
```

The `requirements.txt` pins NumPy, pandas, SciPy, scikit-learn, and joblib to the versions reported in the manuscript. The manuscript does not specify a Matplotlib version; its version constraint is an installation convenience, not a claimed historically exact pin.

`Phoenix.py` imports `OrthogonalArray` from `oa_package.oa`. The included `oa_package/oa.py` is the OA implementation provided with this release. Keep it at that relative path. Its supplied source did not contain explicit upstream copyright or license metadata; **confirm provenance and reuse rights before publication**, and preserve any upstream notices as appropriate.

## Downloading Dataset63

1. Open the **[Dataset63 and benchmark archive on Zenodo](https://doi.org/10.5281/zenodo.22503472)**.
2. Download the dataset file(s) or archive appropriate for your use case, and extract them **outside the tracked Git repository**, or into a directory ignored by Git.
3. Use the **actual extracted path** as `--data` in the examples below. Archive filenames and extracted layouts may vary; this repository does not assume or generate a particular Zenodo filename.

The manuscript recommends the **original** datasets for unbiased evaluation, with any learned preprocessing fitted only on the relevant training data. A **standardized** dataset variant is provided in Zenodo for convenience and reproducibility; ensure the chosen preprocessing protocol is appropriate for your evaluation. A ready-standardized file alone does not establish leakage-free evaluation for new train/test splits.

### Expected CSV input

The standalone command-line interface expects a numeric CSV with a header row, numeric feature columns, and the **class label in the last column**:

```csv
feature_1,feature_2,feature_3,label
0.12,1.42,0.51,0
0.44,1.10,0.19,1
```

This is **an illustration of the expected schema**, not a downloadable Dataset63 file. Adapt or convert a Zenodo dataset to that format if necessary. Missing values, text-encoded features and scaling are not automatically handled by this lightweight script.

## Running Phoenix-OI

Provide a path to a real, local CSV downloaded/prepared separately:

```bash
python Phoenix.py --data "/path/to/dataset.csv" --seed 42
```

The following options adjust the **functional implementation**:

```bash
# Primary scalarized mode (default), with paper benchmark hyperparameters
python Phoenix.py --data "/path/to/dataset.csv" --mode original --Np 30 --max_iter 30 --max_conv_iter 10 --rho 3 --knn_k 3 --cv 5 --lambda_penalty 0.001 --seed 42

# Optional supervised ANOVA screening (NOT the primary no-ANOVA protocol)
python Phoenix.py --data "/path/to/dataset.csv" --use_anova_prefilter --search_space 200 --seed 42

# Optional GPSISF-style scoring and post-hoc Pareto outputs
python Phoenix.py --data "/path/to/dataset.csv" --mode gpsisf --save_pareto --output_dir results --seed 42

# Show all available CLI switches
python Phoenix.py --help
```

### Default parameter values

| Parameter | Default | Description |
| --- | --- | --- |
| `--mode` | `original` | Scalarized wrapper fitness |
| `--Np` | `30` | Number of particles |
| `--max_iter` | `30` | Maximum number of search iterations |
| `--max_conv_iter` | `10` | Early stopping patience |
| `--tol` | `0.001` | Global-best improvement tolerance |
| `--w`, `--c1`, `--c2` | `0.9`, `2.0`, `2.0` | PSO parameters |
| `--rho` | `3` | IMM particle allocation divisor |
| `--knn_k`, `--cv` | `3`, `5` | Primary KNN evaluator and CV folds |
| `--lambda_penalty` | `0.001` | Fixed feature-count penalty |
| `--use_anova_prefilter` | disabled | Optional supervised screening |
| `--no_constant_filter` | disabled | By default constant features are removed |
| `--seed` | unset | Pseudorandom seed; set for repeatable example runs |

The script prints the best selected-feature subset, its fitness and accuracy, elapsed time, and exact stage-wise FFE accounting. Pareto output files are written locally only if `--save_pareto` is enabled.

**Scope of reproducibility:** `Phoenix.py` is the **functional core optimizer**. It is **not** a bundled end-to-end reproduction of all paper experiments: baseline comparators, 30-run aggregations, ablation suites, classifier-substitution analysis, nested held-out tests, metabolomics analysis, and full multi-objective archive experiments require additional experimental code/protocols. Consult the manuscript/supplementary materials and Zenodo for the archived results and available resources; do not interpret one local CLI run as a reproduction of reported table statistics.

## Complete 63-dataset benchmark results

The tables below reproduce **all 63 per-dataset results** in **Supplementary Tables A5 (classification accuracy)** and **A6 (number of selected features)** of the accompanying Phoenix-OI manuscript. The accuracy columns report **mean ± standard deviation (%)**, and `#Feat` columns report the **mean number of selected features** across runs. The original paper benchmark uses **30 independent runs per dataset**, five-fold KNN (`k=3`) wrapper evaluation, and fixed `lambda = 0.001` under the strict No-ANOVA protocol.

The comparison covers Phoenix-OI, PSO, SFE and SFE-PSO. It is reproduced here as an easy-to-view results table; **these are reported experimental results, not values regenerated by running the small standalone `Phoenix.py` example**. A copy of this table appears in [`docs/benchmark_results.md`](docs/benchmark_results.md).

| Dataset | Phoenix-OI Acc. | Phoenix-OI #Feat | PSO Acc. | PSO #Feat | SFE Acc. | SFE #Feat | SFE-PSO Acc. | SFE-PSO #Feat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SRBCT | 100.00 ± 0.00% | 19.6 | 88.39 ± 1.80% | 539.47 | 85.34 ± 2.48% | 143.3 | 91.75 ± 2.51% | 64.1 |
| ORL | 94.90 ± 0.88% | 61.47 | 94.58 ± 0.59% | 204.1 | 93.65 ± 0.54% | 153.3 | 94.62 ± 0.55% | 88.17 |
| Yale | 82.22 ± 0.85% | 48.33 | 77.90 ± 0.92% | 242.2 | 73.62 ± 1.05% | 175 | 75.33 ± 1.12% | 92.07 |
| Colon | 98.13 ± 1.46% | 35.07 | 86.74 ± 2.34% | 459.53 | 85.30 ± 2.64% | 73.6 | 91.71 ± 2.59% | 27.53 |
| warpAR10P | 92.00 ± 1.37% | 30.53 | 61.18 ± 1.46% | 529.93 | 67.90 ± 4.77% | 111.23 | 77.54 ± 2.55% | 20 |
| Leukemia_1 | 99.82 ± 0.47% | 18.53 | 88.64 ± 1.75% | 1196.5 | 87.32 ± 2.58% | 205.5 | 93.72 ± 2.44% | 109.97 |
| lymphoma | 95.65 ± 0.36% | 33.53 | 94.75 ± 1.34% | 896.13 | 93.30 ± 1.27% | 280.97 | 95.92 ± 1.21% | 180.17 |
| lung | 99.07 ± 0.17% | 19.73 | 96.30 ± 0.62% | 708.2 | 96.39 ± 0.73% | 337.83 | 97.45 ± 0.49% | 222.3 |
| GLIOMA | 93.07 ± 2.17% | 30.47 | 84.27 ± 2.62% | 1017 | 85.00 ± 3.42% | 116.2 | 91.60 ± 3.07% | 40.97 |
| DLBCL | 100.00 ± 0.00% | 13.6 | 88.00 ± 2.09% | 1216.73 | 89.94 ± 2.76% | 180.57 | 94.15 ± 2.94% | 52.93 |
| bladder cancer | 99.67 ± 0.85% | 17 | 80.08 ± 3.56% | 1292.67 | 84.58 ± 4.52% | 288.27 | 93.50 ± 2.93% | 34.3 |
| 9_Tumors | 86.44 ± 3.89% | 55.47 | 48.50 ± 3.20% | 1308.03 | 48.06 ± 3.42% | 145 | 56.56 ± 4.21% | 68.53 |
| TOX_171 | 93.30 ± 1.38% | 64.07 | 62.49 ± 2.32% | 1291.23 | 63.19 ± 2.80% | 165.77 | 67.22 ± 1.68% | 111.83 |
| Brain_Tumor_1 | 93.04 ± 1.97% | 20.47 | 87.93 ± 1.21% | 1294.07 | 88.37 ± 1.56% | 709.33 | 90.19 ± 0.96% | 394.33 |
| Prostate_GE | 97.30 ± 1.11% | 11.27 | 85.21 ± 2.10% | 1331.47 | 85.30 ± 2.25% | 549.67 | 89.21 ± 1.75% | 126.47 |
| leukemia | 99.90 ± 0.36% | 13.07 | 87.12 ± 1.95% | 1580.8 | 90.42 ± 2.93% | 108.97 | 95.31 ± 2.18% | 57.53 |
| ALLAML | 99.90 ± 0.36% | 18.27 | 83.96 ± 2.22% | 1583.07 | 87.42 ± 2.66% | 246.57 | 94.01 ± 2.70% | 40.7 |
| CNS | 94.33 ± 2.71% | 37.4 | 64.28 ± 2.88% | 1631.67 | 68.44 ± 5.18% | 401.8 | 76.89 ± 4.88% | 33.97 |
| ALL_AML_3 | 99.02 ± 0.78% | 23 | 82.79 ± 2.06% | 1614.5 | 84.27 ± 2.96% | 242.83 | 89.04 ± 2.79% | 132.1 |
| ALL_AML_4 | 98.27 ± 1.28% | 25.53 | 77.22 ± 1.60% | 1603.7 | 78.04 ± 2.57% | 298.2 | 85.37 ± 2.16% | 116.33 |
| Leukemia_2 | 98.91 ± 1.42% | 23.4 | 77.22 ± 1.60% | 1603.7 | 78.83 ± 2.64% | 323.3 | 85.02 ± 3.03% | 201.1 |
| childhood_ALL | 75.33 ± 4.64% | 26.2 | 45.28 ± 3.01% | 1899.33 | 51.89 ± 5.79% | 182.2 | 61.56 ± 4.63% | 32.9 |
| Carcinom | 95.97 ± 0.95% | 40.33 | 84.50 ± 1.26% | 2043.07 | 84.66 ± 1.35% | 451 | 87.19 ± 1.87% | 239 |
| nci9 | 87.22 ± 2.25% | 48.67 | 50.00 ± 2.79% | 2197.93 | 52.22 ± 3.76% | 262.97 | 59.33 ± 3.89% | 122.47 |
| arcene | 96.73 ± 1.20% | 34.67 | 85.18 ± 1.84% | 2245.07 | 85.87 ± 1.41% | 1630.2 | 87.55 ± 0.95% | 803.33 |
| orlraws10P | 99.80 ± 0.75% | 26.07 | 95.93 ± 0.44% | 2273.27 | 96.17 ± 0.45% | 4345.87 | 99.50 ± 0.76% | 148.4 |
| Brain_Tumor_2 | 94.80 ± 1.76% | 28 | 70.60 ± 2.97% | 2389.63 | 74.33 ± 4.44% | 234.97 | 83.67 ± 3.39% | 48.4 |
| Leukemia_3 | 99.90 ± 0.36% | 19 | 79.67 ± 1.53% | 2527.63 | 82.05 ± 2.61% | 981.3 | 91.84 ± 2.41% | 95.57 |
| CLL_SUB_111 | 95.37 ± 2.20% | 36.93 | 65.58 ± 2.31% | 2596.27 | 68.35 ± 2.83% | 258.87 | 74.00 ± 2.41% | 171.2 |
| 11_Tumor | 96.69 ± 1.73% | 49.27 | 74.50 ± 1.15% | 2792.57 | 76.99 ± 1.87% | 316.9 | 80.70 ± 1.75% | 172.77 |
| MLL | 100.00 ± 0.00% | 17.6 | 83.80 ± 1.10% | 2838.57 | 85.60 ± 2.09% | 3164.27 | 92.88 ± 2.50% | 247.97 |
| Lung_2 | 96.84 ± 0.84% | 24.13 | 86.40 ± 1.10% | 2808.13 | 90.31 ± 1.29% | 134.13 | 92.55 ± 1.19% | 79.6 |
| glioblastoma | 94.67 ± 1.58% | 30.47 | 64.93 ± 3.57% | 2861.1 | 70.07 ± 3.98% | 480.5 | 80.87 ± 4.61% | 53.27 |
| AML_prognosis | 97.09 ± 2.20% | 47.27 | 64.76 ± 2.22% | 2853.4 | 70.98 ± 4.14% | 418.43 | 79.04 ± 4.32% | 76.83 |
| Prostate_cancer | 98.20 ± 0.75% | 18.53 | 83.89 ± 2.38% | 2371.97 | 84.54 ± 1.95% | 485.63 | 89.17 ± 1.84% | 155.23 |
| Ovarian cancer | 99.95 ± 0.13% | 6.13 | 93.63 ± 0.51% | 3403.13 | 95.58 ± 1.11% | 448.87 | 97.94 ± 0.70% | 78.67 |
| SMK_CAN_187 | 79.98 ± 3.54% | 28.2 | 59.15 ± 1.14% | 4505.57 | 63.81 ± 2.26% | 364.47 | 69.96 ± 2.34% | 78.03 |
| GLI_85 | 98.43 ± 0.82% | 27.33 | 90.12 ± 1.48% | 5068.43 | 92.20 ± 1.16% | 3070.4 | 94.67 ± 1.17% | 1471.4 |
| Breast Cancer | 95.72 ± 2.75% | 40.2 | 65.19 ± 1.78% | 5599.07 | 69.86 ± 2.52% | 571.67 | 76.17 ± 2.51% | 266.73 |
| lung_discrete | 95.68 ± 1.11% | 33.07 | 92.90 ± 1.17% | 77.63 | 85.07 ± 1.69% | 44.57 | 90.39 ± 2.32% | 25.67 |
| BASEHOCK | 86.36 ± 4.26% | 39.2 | 79.34 ± 1.19% | 1059.37 | 77.58 ± 1.07% | 655.47 | 78.94 ± 1.20% | 532.7 |
| CNAE9 | 90.99 ± 0.62% | 50.27 | 85.18 ± 1.31% | 250.93 | 75.03 ± 3.83% | 392.3 | 74.30 ± 3.42% | 357.73 |
| madelon | 90.22 ± 0.36% | 8.07 | 58.92 ± 1.68% | 118.67 | 57.19 ± 3.70% | 35.2 | 60.85 ± 5.42% | 16.83 |
| COIL20 | 97.28 ± 0.61% | 37.73 | 93.06 ± 0.85% | 201.27 | 91.90 ± 0.95% | 60.1 | 93.48 ± 0.90% | 36.43 |
| Chess_KRKPA7 | 93.62 ± 0.52% | 12.67 | 93.36 ± 0.79% | 13.7 | 72.17 ± 8.10% | 7.47 | 75.47 ± 8.72% | 5.67 |
| Driv_face | 92.19 ± 0.64% | 16 | 61.49 ± 0.82% | 1402.2 | 80.58 ± 2.99% | 3.67 | 82.67 ± 1.80% | 1.57 |
| GFE01 | 83.96 ± 0.76% | 20.33 | 78.46 ± 0.73% | 63.27 | 77.73 ± 1.61% | 9.5 | 78.43 ± 1.64% | 6.67 |
| MicroMass | 94.33 ± 1.51% | 22.93 | 80.97 ± 1.64% | 319.4 | 74.68 ± 3.15% | 124.77 | 79.37 ± 2.75% | 43.27 |
| MUSK1 | 88.77 ± 0.84% | 40.4 | 85.47 ± 0.95% | 49.13 | 78.66 ± 1.55% | 23.6 | 80.59 ± 1.56% | 16.27 |
| QAR | 90.29 ± 0.31% | 39.67 | 89.05 ± 0.52% | 197.73 | 88.73 ± 0.47% | 231.67 | 88.98 ± 0.38% | 136.1 |
| PCMAC | 84.90 ± 2.90% | 32.87 | 71.07 ± 1.60% | 715.53 | 69.85 ± 1.23% | 655.97 | 70.58 ± 1.09% | 448.63 |
| warpPIE10P | 100.00 ± 0.00% | 27.27 | 98.46 ± 0.40% | 488.77 | 98.76 ± 0.38% | 634 | 99.59 ± 0.40% | 126.47 |
| prostate_cancer_2 | 99.67 ± 1.25% | 11.8 | 62.83 ± 3.58% | 2856.67 | 72.17 ± 11.01% | 1999.73 | 90.83 ± 7.86% | 25.3 |
| RELATHE | 79.40 ± 2.85% | 43.67 | 76.59 ± 1.24% | 946.13 | 75.27 ± 1.12% | 719.53 | 75.74 ± 0.88% | 463.83 |
| USPS | 96.27 ± 0.18% | 35.67 | 96.19 ± 0.23% | 44.83 | 95.89 ± 0.20% | 79.87 | 96.03 ± 0.36% | 57.3 |
| DARWIN | 90.94 ± 1.31% | 28.8 | 80.79 ± 1.96% | 111.5 | 77.61 ± 3.80% | 11.73 | 80.47 ± 3.15% | 8.7 |
| dbworld | 89.93 ± 1.90% | 5.93 | 92.03 ± 1.95% | 56.43 | 75.91 ± 6.80% | 17.47 | 77.44 ± 7.11% | 16.53 |
| gisette | 95.52 ± 0.30% | 18.4 | 88.77 ± 0.75% | 1088 | 90.86 ± 0.70% | 1921.53 | 91.08 ± 0.55% | 1910.17 |
| Lung_cancer | 84.41 ± 3.27% | 16.67 | 81.37 ± 4.46% | 18.67 | 69.89 ± 7.86% | 6.03 | 66.62 ± 6.68% | 5.53 |
| Isolet | 91.92 ± 0.66% | 75.8 | 89.36 ± 0.73% | 144.73 | 86.42 ± 0.96% | 215.7 | 87.33 ± 0.97% | 159.07 |
| REJAFADA | 97.21 ± 0.33% | 13.87 | 87.64 ± 1.39% | 1519.17 | 87.21 ± 3.55% | 238.37 | 88.88 ± 4.69% | 157.57 |
| MUSK2 | 87.93 ± 0.70% | 17.67 | 81.51 ± 0.98% | 36.5 | 82.03 ± 0.99% | 5.7 | 81.32 ± 1.34% | 3.53 |
| Lymphography | 88.88 ± 0.48% | 12.93 | 85.89 ± 1.19% | 8.93 | 77.51 ± 3.57% | 4.97 | 76.89 ± 4.18% | 5.03 |

### Cross-dataset summary from Supplementary Tables A5–A6

| Metric | Phoenix-OI | Against PSO | Against SFE | Against SFE-PSO |
|---|---:|---:|---:|---:|
| Accuracy: dataset-wise win/tie/loss | — | 62 / 0 / 1 | 63 / 0 / 0 | 62 / 0 / 1 |
| Selected features: dataset-wise win/tie/loss (fewer is better) | — | 62 / 0 / 1 | 55 / 0 / 8 | 50 / 0 / 13 |

The 12-dataset mechanism-level sub-suite, ablation, FFE convergence, sensitivity, equal-budget, nested held-out, and external gastric-cancer metabolomics analyses are described in the main manuscript and supplementary material. The **dataset files**, standardized variants, provenance/metadata, quality-control and file-hash records, and archived benchmark results are separately provided on **[Zenodo](https://doi.org/10.5281/zenodo.22503472)**. The GitHub repository repeats only the **human-readable results table**, not the raw datasets or complete experiment outputs.

**Reproducibility boundary:** These published aggregate numbers cannot be exactly regenerated from one `Phoenix.py` CLI invocation without the original experiment orchestration, complete baselines, and dataset-specific protocols.

## Citation

For software use, cite the **[Phoenix-OI GitHub repository](https://github.com/Phoenix-OI/Phoenix-Feature-Selection)** (see [`CITATION.cff`](CITATION.cff)). For use of the curated datasets or benchmark resources, cite the **[Zenodo dataset record](https://doi.org/10.5281/zenodo.22503472)** (DOI: `10.5281/zenodo.22503472`) and honor any original dataset attribution/reuse restrictions. Cite the associated Phoenix-OI research paper **once bibliographic publication details are available**. See [`ATTRIBUTION.md`](ATTRIBUTION.md).

**The Zenodo dataset DOI should not be cited as a DOI for this source-code repository.** Consult the Zenodo record itself for its exact publication metadata, licensing, and recommended dataset citation.

## Licenses and provenance

- Original Phoenix-OI source code: **Apache License 2.0**, [`LICENSE`](LICENSE).
- Original project-authored documentation and benchmark result/metadata materials, to the extent the authors hold the necessary rights: **CC BY 4.0**, [`LICENSE-DATA`](LICENSE-DATA).
- Zenodo-hosted datasets, including incorporated third-party material: **the license and provenance specified for each resource on Zenodo and by its original provider control its reuse**. A GitHub documentation license does not override those rights.
- Vendored or third-party code in `oa_package/`: verify its upstream license and preserve any applicable attribution before a public release.

See [`LICENSE_SUMMARY.md`](LICENSE_SUMMARY.md) and [`NOTICE`](NOTICE) for a consolidated overview.

## Publication status

The associated Phoenix-OI research article is an unpublished manuscript as of this repository documentation revision. Replace this statement and add the final paper citation after acceptance/publication; this README does **not** assert a journal publication or a software-specific DOI.
