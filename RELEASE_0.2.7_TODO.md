# CryoSwath 0.2.7 release TODO

This file is the resumable handoff for the 0.2.7 release. Work is isolated in
`/tmp/cryoswath-release-0.2.7` on branch `release/0.2.7`; the original dirty
worktree at `/home/haacker/downscale_test` was left intact.

## Completed

- [x] Based the release branch on committed feature-chain HEAD `a1bf5d1`.
- [x] Set the package version and dated changelog section to `0.2.7` and
  `2026-07-11`.
- [x] Added curated notes for MAAP/PDS downloading, L1b credential preflight,
  finalized L3 merging, and dependency compatibility.
- [x] Diagnosed the dependency conflict:
  - `xarray 2025.3.1` with `zarr 3.1.5` hangs in
    `test_infer_extension_spec_from_store_attrs_and_path`.
  - `xarray 2025.3.1` with `zarr 2.18.7` passes all 171 runnable unit tests.
- [x] Set coherent runtime bounds: `xarray>=2025.3` and `zarr<3`.
- [x] Simplified the production extra to `xarray[accel]` plus
  `dask[distributed]`.
- [x] Updated compatibility CI to run the full unit suite for minimum xarray,
  latest xarray, and the non-blocking Python 3.14 observation job.
- [x] Synchronized `pixi.toml`, exported Conda environments, notebook-test
  environment files, and `pixi.lock`.
- [x] Verified the dependency-definition sync check.
- [x] Passed the locked/current unit suite before the final Zarr bound:
  171 passed, 1 skipped.
- [x] Passed the minimum compatibility suite with xarray 2025.3.1 and Zarr
  2.18.7: 171 passed, 1 skipped.
- [x] Documented the release and mandatory local conda-forge-image workflow.
- [x] Added and tested `CRYOSWATH_TEST_DATA_DIR` so isolated notebook tests can
  reuse a complete external CryoSwath data tree instead of downloading RGI/L1b.
- [x] Passed all four committed report notebooks with the external data/DEM
  hooks and no credential download.
- [x] Extended the external-data hook to source tutorial support files that are
  intentionally absent from the committed checkout.

## Completed validation

- [x] Passed the synchronized current-xarray/Zarr 2 unit suite: 171 passed,
  1 skipped.
- [x] Passed the installed-wheel suite: 172 passed, 1 skipped; wheel and sdist
  also passed `twine check` inside the helper.
- [x] Passed the strict Sphinx build with warnings treated as errors.

## Remaining repository checks

- [ ] Decide how to handle the repository-wide Ruff baseline: `pixi run -e test
  lint` reports 95 pre-existing errors across legacy modules and notebooks. The
  release changes introduce no Python source files, so unrelated mass-formatting
  is intentionally not part of this release.
- [x] Pre-commit configuration validation passed.
- [x] Dependency synchronization check passed.
- [x] `git diff --check` passed.
- [ ] Decide how to handle the repository-wide formatting baseline: 14
  pre-existing files would be reformatted. No release metadata/workflow file is
  among them, so unrelated formatting is intentionally deferred.
- [x] Build the wheel and sdist and run `twine check` (covered by
  `test-installed`; create persistent `dist/` artifacts before feedstock work).
- [ ] Run `pixi run -e test test-fresh-committed`, passing existing local
  DEM/RGI/auxiliary/L1b paths to avoid redundant large downloads. Do not print
  credential values; request credentials only if local resources are
  insufficient.
  - Unit and installed-wheel stages passed in the first run.
  - The first notebook attempt failed because Miniforge was absent from `PATH`.
  - The second attempt reached notebooks but proved the old setup only reused an
    external DEM; it attempted Earthdata RGI download and lacked local L1b/L2.
  - Report notebooks now pass with both `CRYOSWATH_TEST_DATA_DIR` and
    `CRYOSWATH_TEST_DEM_DIR`.
  - Rerun tutorial notebooks after amending external support-file discovery into
    committed HEAD.
  - Diagnostic hooks, first waveform, first swath, and POCA tutorial notebooks
    passed.
  - The general step-by-step notebook was stopped at cell 46/67 after about ten
    minutes because it expanded into a broad historical re-download. Local data
    was configured correctly, but one-second track/product timestamp variants
    bypassed the cache and used working netrc credentials. Resolve the cache-ID
    reconciliation or narrow this tutorial before treating `test-all` as a
    practical release gate.
- [ ] Review the final diff and commit the release preparation on
  `release/0.2.7`.

## Upstream and publishing blockers

- [ ] Refresh `origin/main` and confirm the upstream feature merges before
  proposing the release. The attempted SSH fetch failed because no usable
  GitHub SSH key/askpass was available.
- [ ] Rebase or rebuild `release/0.2.7` on the refreshed upstream `main` if its
  tree differs from committed feature-chain HEAD `a1bf5d1`.
- [ ] Push the release branch and open the release PR only after the user
  authorizes external publication actions.
- [ ] After the release PR passes and merges, create GitHub Release `v0.2.7`
  from that exact commit and verify wheel/sdist publication on PyPI.

## Conda-forge preflight and PR

- [ ] Locate or install a Docker-compatible runtime. `docker` was not found on
  `PATH` in the current environment.
- [ ] Clone/update a personal fork of
  `conda-forge/cryoswath-feedstock` locally without pushing a branch.
- [ ] Prepare recipe version `0.2.7`, build number `0`, Python/runtime
  dependencies, `xarray >=2025.3`, and `zarr <3`.
- [ ] Temporarily point the recipe at the locally built 0.2.7 sdist and run
  `python build-locally.py` using the conda-forge CI image.
- [ ] Require build, recipe tests, output validation, and installation/import
  from `build_artifacts` to pass before any feedstock push or PR.
- [ ] After PyPI publication, replace the temporary source with the PyPI sdist
  URL and SHA-256. If its bytes differ from the tested local sdist, rerun
  `build-locally.py`.
- [ ] Push the personal-fork branch and open the feedstock PR; if autotick opens
  a duplicate, retain the passing/more complete PR.
- [ ] After merge, verify availability with `conda search` or
  `mamba repoquery`.
