# Public GitHub Release Checklist

This list distinguishes static packaging checks from manual provenance/publication checks. The code and documentation live in GitHub; datasets and experiment archives are provided separately through [Zenodo](https://doi.org/10.5281/zenodo.22503472).

## Files and references

- [x] Include `Phoenix.py` and `oa_package/oa.py` under the expected import path.
- [x] Include `README.md` with **all 63** comparison rows from Supplementary Tables A5/A6.
- [x] Keep an equivalent `docs/benchmark_results.md` table for independent viewing.
- [x] Replace old `TBD` DOI fields with `10.5281/zenodo.22503472` in docs/citation files.
- [x] Make usage examples require a **real local path**, not a fictitious sample under `dataset63/`.
- [x] Document paper-level defaults: KNN `k=3`, CV `5`, `lambda=0.001`, `rho=3`, no ANOVA by default.
- [x] Exclude raw dataset archives/CSVs and generated outputs from Git via `.gitignore`.
- [ ] Confirm the DOI page resolves publicly and its actual contents, recommended citation and exact license/provenance per third-party dataset.
- [ ] Confirm upstream **origin, license and redistribution permissions** for the provided `oa_package/oa.py` (no upstream attribution notice was present in the uploaded helper source).
- [ ] Confirm that all desired reproducibility scripts and instructions are actually public; do **not** claim that `Phoenix.py` alone reproduces complete paper experiments.
- [ ] Confirm GitHub Actions/environment and execute a smoke test under the paper's **Python 3.9.21** / documented dependency versions.
- [ ] Verify whether a Git tag/release `v0.1.0` exists before using it in publication-facing metadata.
- [ ] When the manuscript is accepted/published, update the publication status and paper DOI/citation.
- [ ] Check third-party licenses and original dataset usage conditions before public redistribution.

## Final publish commands

```bash
python Phoenix.py --help
git status --short
git add README.md docs/ oa_package/ CITATION.cff ATTRIBUTION.md LICENSE LICENSE-DATA LICENSE_SUMMARY.md NOTICE .gitignore requirements.txt Phoenix.py
git commit -m "Publish full 63-dataset comparison and align docs with Zenodo"
git push
```

`.gitignore` only affects untracked files. If large datasets/outputs were already committed, check `git ls-files` and untrack only the appropriate data paths before pushing.
