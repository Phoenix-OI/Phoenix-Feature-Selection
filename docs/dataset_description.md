# Dataset63: Dataset Description and Availability

Phoenix-OI is evaluated on a curated set of **63 classification datasets**, referred to as **Dataset63** in the manuscript. Dataset names, numbers of samples, feature dimensions and class counts are given in **Supplementary Table A1**.

## Download

**The complete datasets are hosted on Zenodo, not directly in this GitHub repository:**

- DOI: **[10.5281/zenodo.22503472](https://doi.org/10.5281/zenodo.22503472)**
- URL: <https://doi.org/10.5281/zenodo.22503472>

The manuscript describes an **original-data collection**, a **standardized variant**, dataset summaries, provenance records, quality-control notes, file hashes and benchmark results in this archive. Confirm exact filenames, licenses and format against the Zenodo landing page and file metadata.

## Using a dataset with `Phoenix.py`

Download and extract the desired Zenodo archive to your machine; pass the actual file path using `--data`. The current standalone loader expects a CSV with a header, **numeric feature columns**, and the **target label in the final column**:

```bash
python Phoenix.py --data "/path/to/your_extracted_dataset.csv" --seed 42
```

No sample CSV is shipped with this repository. Convert nonconforming file formats to the expected schema before running it. The manuscript recommends original datasets for unbiased evaluation; fit any learned preprocessing using **training-only samples** for held-out assessments. Ready-standardized data do not automatically establish leakage-free training/test evaluation.

## Results without downloading raw datasets

The full per-dataset comparison of Phoenix-OI, PSO, SFE, and SFE-PSO is visible in [`README.md`](../README.md#complete-63-dataset-benchmark-results) and [`benchmark_results.md`](benchmark_results.md).

## Licensing and attribution

Software here is licensed as stated in [`LICENSE`](../LICENSE). Project-authored documentation and tables are covered as stated in [`LICENSE-DATA`](../LICENSE-DATA). Third-party dataset rights and any restrictions are governed by the particular dataset's provenance and license; check the original providers and Zenodo metadata. Cite the dataset archive separately from the GitHub code; see [`ATTRIBUTION.md`](../ATTRIBUTION.md).
