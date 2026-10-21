# Phoenix-OI GitHub Release: Complete 63-Dataset Table + Documentation

This package is intended to **merge over the existing GitHub working tree**, not to delete any unrelated tracked files or the `.git/` directory.

## Contents and changes

- `README.md`: includes the **full merged 63-dataset Table A5/A6** (Phoenix-OI, PSO, SFE, SFE-PSO: accuracy mean ± SD and feature count), links to docs and Zenodo, and an explicit reproducibility boundary.
- `docs/benchmark_results.md`: maintains the same complete table as a standalone page; obsolete references to local `data/*.csv` are removed.
- `docs/dataset_description.md`: DOI linked, no old `TBD` placeholders.
- `docs/usage.md`: replaces nonexistent dataset example path; documents actual default `rho=3` and switches.
- `docs/method_overview.md`: extends the original supplied description with FFE accounting and reproducibility scope.
- `docs/release_checklist.md`: updated to the GitHub-code/Zenodo-data policy and remaining verification tasks.
- `oa_package/oa.py`: exact user-provided OA helper copied as-is; `__init__.py` enables ordinary package importing.
- The 10 root-level files from the prior patch, including `Phoenix.py`, citation and licenses, are preserved (except the README additions).

## Validation

- Compared the uploaded 63-row result table against **Supplementary_V31.docx** Tables A5/A6: all **252 accuracy cells and 252 selected-feature cells** match after whitespace normalization.
- Validated all 63 unique dataset identifiers.
- `oa_package/oa.py` was supplied without a clear original copyright or license notice; confirm its origin and permission for redistribution before making the repository public.
- Zenodo DOI is used as provided in the manuscript/user request; landing-page contents and upstream dataset rights still require independent verification.

## GitHub update guidance

Extract into your repository root, keeping `.git/` untouched; then verify and commit. Do not upload dataset archives or generated results back into GitHub. The Markdown result tables are intentionally in both GitHub and the external Zenodo archive.
