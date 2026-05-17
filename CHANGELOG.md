# Changelog

This changelog is intentionally lightweight. It records user-visible changes and release notes that are useful to remember, without listing every commit.

## 0.2.6 - 2026-05-17

- Added configurable project layouts and package-backed setup helpers so processing projects and notebooks no longer depend on branch-local bootstrap code.
- Added automatic RGI and auxiliary-data preparation paths, including notebook test project initialization.
- Added installed-wheel and fresh-environment test workflows to catch packaging issues and local-state assumptions.
- Simplified tutorial notebooks and docs to assume CryoSwath is installed.

## 0.2.5.post2 - 2026-04-19

- Fixed default path handling and repaired the CryoSat HTTPS download flow.
- Updated the Pixi lockfile for the patch release.

## 0.2.5.post1 - 2026-04-16

- Hardened CryoSat data downloads, ESA credential resolution, and atomic file writes.
- Added automatic download support for missing DEM and RGI reference datasets.
- Added provenance sidecars, CF history helpers, optional xzarrguard support, and an L3 dataset extension workflow.
- Tightened dependency synchronization, xarray compatibility handling, and release/test maintenance.

## 0.2.5 - 2026-02-15

- Improved robustness across L1B-L4 processing for empty data, stale caches, scalar coordinates, missing columns, and clearer failure paths.
- Improved multiprocessing and zarr chunk handling for more portable processing.
- Added Pixi-first environment support, locked test workflows, tutorial notebook automation, and CRISTAL portability notes.
- Pinned xarray below 2025.12 due to known upstream compatibility issues.

## Earlier Releases

- Earlier releases are not tracked here in detail. See the git tags and commit history when needed.
