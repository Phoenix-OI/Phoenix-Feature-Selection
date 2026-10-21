# Method Overview

Phoenix-OI is a cross-phase OED/MED-guided PSO wrapper feature-selection framework for high-dimensional classification.

## 1. Binary Representation

Each particle is a candidate feature subset. A feature is selected when the binarized position is 1.

## 2. Wrapper Objective

The default single-objective fitness is:

```text
Fitness(S) = AccCV(S) - lambda * |S|
```

Default values:

```text
KNN k = 3
CV folds = 5
lambda = 0.001
```

## 3. Feature Preprocessing

The default protocol removes constant features and does not apply supervised ANOVA prefiltering.

ANOVA prefiltering is available as an optional switch:

```bash
--use_anova_prefilter
```

## 4. OED/MED Initialization

Phoenix-OI uses orthogonal-array trials and main-effect-driven candidate construction to initialize the swarm.

## 5. OI-Guided IMM Movement

Phoenix-OI constructs global-learning and personal-learning proposals, partitions them into segments, evaluates balanced OED trials, and applies MED inference to construct updated candidates.

## 6. Elite-Only Two-Stage Refinement

The elite solution is refined through:

1. incumbent-complement recombination;
2. retain/remove reduction.

## 7. Optional GPSISF/Pareto Analysis

The optional `gpsisf` mode provides GPSISF-style ranking and post-hoc Pareto output.

## 8. Fitness Function Evaluation (FFE) Accounting

In the default wrapper objective, one fitness-function evaluation (FFE) is one exact evaluation of a **nonempty** selected-feature subset. An evaluation contains the complete five-fold KNN cross-validation; folds are **not** counted separately. The implementation reports FFE counts by initialization OA trials, initial population, IMM OA trials, population-after-movement evaluation, and the two elite-refinement stages.

## 9. Reproducibility Scope and Data Availability

This repository contains the runnable functional optimizer and OA helper, but does not include the full baseline experiment and ablation orchestration. The full 63-dataset accuracy/feature-count table appears in [`README.md`](../README.md#complete-63-dataset-benchmark-results) and [`benchmark_results.md`](benchmark_results.md). Download the actual datasets and archived results from [Zenodo](https://doi.org/10.5281/zenodo.22503472); for detailed methods see the accompanying manuscript and supplementary material.
