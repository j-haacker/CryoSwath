# Changelog

All notable changes are documented in this file.

## v0.2.5 (2026-02-15)

- Range: `v0.2.4..ee58532`
- Snapshot commit date: 2026-02-15

### User impact summary

This change set is primarily a stability and portability update rather than a major new algorithm release.

#### What it means for CryoSwath usage

- Day-to-day processing in `l1b`, `l2`, `l3`, `l4`, `gis`, and `misc` is more robust on edge cases (empty chunks/windows, missing columns, unnamed indexes, scalar coordinates, and missing region matches).
- Error handling is clearer in several paths that previously relied on assertions; failures should now be easier to diagnose in production runs.
- `l2` multiprocessing is safer across platforms (`spawn` context + CPU core detection), which helps on macOS/Windows and mixed cluster environments.
- `l3` chunk-time reindexing for zarr should reduce write/read issues related to non-contiguous chunk regions.
- Tutorial notebooks were updated to current APIs and now have automated workflow coverage, making the tutorial path more reliable as a usage reference.

#### New features

- Pixi-first environment support and workflow hardening were added and expanded (including locked runtime solve strategy and feature-scoped tasks).
- A `production` optional dependency group was introduced.
- Tutorial workflow automation was expanded with Snakemake and tox entrypoints for notebook/pipeline validation.
- A CRISTAL mission portability checklist was added to docs.

#### Compatibility notes

- Dependency manifests now pin `xarray < 2025.12` due to upstream regression handling in this cycle.
- If your environment already uses newer `xarray`, expect to align to the pinned version before running CryoSwath reliably.

### Commits since v0.2.4

- 2025-08-31 `2629767` chore: add gitignores
- 2025-08-31 `4e8cb24` chore: bump version
- 2025-12-20 `6d039f5` misc: update rgi_code_translator
- 2025-12-20 `ea142fe` update pyproject.toml
- 2025-12-20 `d6c4bd8` misc: fix previous commit
- 2026-01-24 `a8a2313` fix(misc): allow scalar dimensions in fill_missing_coords
- 2026-01-28 `403658c` fix: revise download_dem
- 2026-01-29 `8c55ef1` chore(l1b): polish code a bit
- 2026-01-29 `296ec1a` fix(test_plots): correct label
- 2026-01-29 `1239d01` fix(misc): use make_valid in load_glacier_outlines
- 2026-01-29 `81be744` chore(misc): format
- 2026-01-29 `a9d55a7` chore: update readme
- 2026-01-29 `d87a946` chore: update docker
- 2026-01-29 `075adc2` Add pixi compatibility and installation instructions
- 2026-01-30 `09f7c90` fix(misc): fix index issue
- 2026-01-30 `bdd8a0c` Merge pull request #52 from Tanmay-Ts/add-pixi-support
- 2026-02-01 `436de23` !fix(misc): update default DEM names
- 2026-02-01 `5faa58a` Merge remote-tracking branch 'origin' into develop
- 2026-02-11 `117c755` chore: add opt.dep. group production
- 2026-02-11 `89d1447` fix(misc): fix building paths from config
- 2026-02-11 `3a94789` chore(l2): simplify code
- 2026-02-11 `27c7079` fix(l3): fix handling even agg. windows
- 2026-02-11 `bb4d0f4` fix(l2): guard grid chunk split for small data
- 2026-02-11 `a9a61d1` fix(misc): handle default dir in ESRI conversion
- 2026-02-11 `e67840b` fix(gis): robust ESRI feather target path
- 2026-02-11 `9973b3f` fix(l4): guard optional filled_flag updates
- 2026-02-11 `97c1f26` fix(l3): raise for unsupported joined L2 aggregation
- 2026-02-11 `a39ba02` fix(l4): raise reference deviation sanity check
- 2026-02-11 `bf819eb` fix(gis): accept scalar lon/lat in CRS selection
- 2026-02-11 `b2ccf6c` fix(l2): replace assert-based stale cache checks
- 2026-02-11 `eab85a5` fix(l2): handle unnamed index in l1b conversion
- 2026-02-11 `8e0f77a` fix(l2): guard empty grid aggregation paths
- 2026-02-11 `40ab2e4` fix(l1b): replace assert-only threshold checks
- 2026-02-11 `ba8ea86` fix(l4): call local elevation reference helper
- 2026-02-11 `f5860f6` fix(l4): use coefficient arrays in trend reconstruction
- 2026-02-11 `cf15928` fix(misc): fail clearly when region lookup is empty
- 2026-02-11 `4daf415` fix(l3): honor l2_type when reading cache chunks
- 2026-02-11 `3d652ba` fix(gis): fail clearly when no o2 region matches
- 2026-02-12 `59443cb` fix(l1b): narrow xarray timedelta patch version gate
- 2026-02-12 `a48cd87` docs(cryoswath): revise RTD pages and API docstrings
- 2026-02-12 `13a0e48` docs(sphinx): refine docs style and refresh README warnings
- 2026-02-12 `7af912d` docs(theme): switch to pydata and clean sidebar/header
- 2026-02-13 `d8ec019` docs(linkcode): resolve source refs from git metadata
- 2026-02-13 `de9fda1` docs: move env warning to install pages and align manager wording
- 2026-02-13 `d5e1cef` merge(main): integrate develop (docs refresh, linkcode hardening, and processing fixes)
- 2026-02-14 `8c86f40` fix(l2): detect CPU cores cross-platform (refs #38)
- 2026-02-14 `100418c` fix(l2): use spawn context for multiprocessing safety (refs #35)
- 2026-02-14 `eadd824` fix(l3): reindex chunk time to contiguous zarr regions (refs #37)
- 2026-02-14 `5ce48ec` docs: add CRISTAL portability checklist (refs #42)
- 2026-02-14 `4e49632` build: document and harden pixi workflow (refs #51)
- 2026-02-14 `bf771bc` refactor(l1b): clarify download status messages (refs #20)
- 2026-02-14 `7b1d3b8` Pin xarray below 2025.12 across dependency manifests
- 2026-02-14 `b66e084` chore: finalize xarray<2025.12 regression fix
- 2026-02-14 `0761831` build(deps): align dependency policy and add py3.13 CI coverage
- 2026-02-14 `a00e721` fix(test_plots): update dem_transect
- 2026-02-14 `e6aec9e` fix(l2): fix missing column issue
- 2026-02-14 `822a044` test(reports): rename notebooks and add Snakemake conda workflow
- 2026-02-15 `7f36903` fix(tests): remove debugging helpers
- 2026-02-15 `5d85e9e` chore(test): add non-failing anonymous FTP login probe
- 2026-02-15 `4e9d342` fix(misc): recover stale cache backup and guard writes with lock
- 2026-02-15 `3eec709` fix(l3): handle empty roll aggregations in build_dataset
- 2026-02-15 `f2da694` fix(tutorial): use misc.fill_missing_coords in step-by-step notebook
- 2026-02-15 `c376355` build(pixi): modernize pyproject metadata and dependency ownership
- 2026-02-15 `e8929b5` fix(tutorial): call l4.append_elevation_reference in step-by-step notebook
- 2026-02-15 `637377a` fix(tutorial): call l4.fill_voids in step-by-step notebook
- 2026-02-15 `9905f68` build: scope pixi tasks by feature and align setup docs
- 2026-02-15 `ad7410a` build(pixi): pin runtime solve strategy and add locked env workflow
- 2026-02-15 `6b98cc7` fix(tutorial): replace removed misc.load_basins in process-first-swath
- 2026-02-15 `8c725d4` test(tutorials): add snakemake workflow for tutorial notebooks
- 2026-02-15 `1840ad0` test(pipeline): add tox entrypoint, config templates, and netrc helper tests
- 2026-02-15 `5bf70af` test(pipeline): add tutorial notebook workflow and update tutorial API calls
- 2026-02-15 `ee58532` merge(build): integrate codex/pixi-followup-envdocs into main
