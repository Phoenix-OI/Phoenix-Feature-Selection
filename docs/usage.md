# Using Phoenix-OI

## Installation

```bash
git clone https://github.com/Phoenix-OI/Phoenix-Feature-Selection.git
cd Phoenix-Feature-Selection
conda create -n phoenix-oi python=3.9.21 -y
conda activate phoenix-oi
python -m pip install -r requirements.txt
```

The repository includes `oa_package/oa.py`, which is imported by `Phoenix.py`:

```python
from oa_package.oa import OrthogonalArray
```

## Getting a dataset

Download a dataset from [Dataset63 on Zenodo](https://doi.org/10.5281/zenodo.22503472) and prepare a headered numeric CSV with the target label in its last column. No `dataset63/example_dataset.csv` exists in this repository. The commands below use **placeholder paths** that must be replaced with real local files.

## Basic run

```bash
python Phoenix.py --data "/path/to/dataset.csv" --seed 42
```

## Strict No-ANOVA primary mode with explicit key defaults

```bash
python Phoenix.py --data "/path/to/dataset.csv" --mode original --knn_k 3 --cv 5 --lambda_penalty 0.001 --Np 30 --max_iter 30 --max_conv_iter 10 --tol 0.001 --rho 3 --seed 42
```

The default **IMM divisor is `rho=3`**, not `rho=1` (`rho=1` is an optional alternative where all particles are selected for IMM).

## Optional analyses

```bash
# Supervised ANOVA prefiltering (not the primary no-ANOVA setting)
python Phoenix.py --data "/path/to/dataset.csv" --use_anova_prefilter --search_space 200 --seed 42

# GPSISF-style scoring plus post-hoc Pareto reporting
python Phoenix.py --data "/path/to/dataset.csv" --mode gpsisf --save_pareto --seed 42

# List the supported arguments
python Phoenix.py --help
```

## Interpreting results

A **nonempty** candidate subset's one full cross-validated wrapper evaluation counts as **one FFE** even with five CV folds. The program reports its stage-wise search-FFE counts, selected features, accuracy, and elapsed time.

`Phoenix.py` is the functional method implementation. The complete baseline comparisons, ablation variants, 30-run benchmark orchestration, nested held-out protocols, metabolomics application code, and full multi-objective archive experiments are not all supplied by this standalone CLI. See the [main README](../README.md), [`benchmark_results.md`](benchmark_results.md), and manuscript/supplementary materials for the original published numbers.
