# CryoSwath

![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/cryoswath)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/cryoswath)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14825358.svg)](https://doi.org/10.5281/zenodo.14825358)
![GitHub License](https://img.shields.io/github/license/j-haacker/cryoswath)
![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/cryoswath?logo=anaconda)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cryoswath?logo=pypi)

CryoSwath is a Python package for processing CryoSat-2 SARIn data,
from waveform-level processing to gridded elevation products.

## What CryoSwath provides

- discovery of CryoSat-2 tracks over a region of interest
- L1b download and preprocessing
- swath and POCA elevation retrieval
- aggregation to regular spatial/temporal grids
- gap filling and trend estimation workflows

## Important notes

- Install CryoSwath in a dedicated environment (`pixi`, `conda`/`mamba`,
  `venv`, or `uv`). The dependency tree is broad, and future
  dependency conflicts are otherwise likely.
- Supported Python version: **>=3.12**.
- Downloading CryoSat resources
  requires an **[ESA EO account](https://eoiam-idp.eo.esa.int/)**.
- ESA credentials are resolved in this order:
  `EOIAM_USER`/`EOIAM_PASSWORD`, then
  keyring (preferred for interactive setup), then
  `~/.netrc` (plaintext fallback), then legacy `config.ini [user]`
  `name/password` (temporary fallback).
- Automatic RGI downloads from NSIDC require NASA Earthdata credentials;
  see the prerequisites docs for setup details.
- L1b file downloads are HTTPS-first. FTP remains a fallback path and is
  still used for metadata refresh flows (catalog/track updates).
- Anonymous FTP login is no longer supported.
- Install `xarray` and `zarr` together to avoid version mismatches.

## Dependency policy

- Flexible package bounds (for pip/uv users): `xarray>=2025.3,<2025.12`.
- Stable environment (recommended for reproducible runs): use the
  checked-in lock/environment files (`pixi.lock`, `environment.yml`).
- Compatibility window in this repository was last audited on
  **February 14, 2026**.

## Installation

For full setup details, see the docs:
[cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)

For development setup and contribution checks, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Option 1: reproducible setup with pixi (recommended)

```sh
git clone https://github.com/j-haacker/cryoswath.git
cd cryoswath
pixi install --locked -e test
pixi run -e test test-unit
```

For an interactive shell in the project environment:

```sh
pixi shell -e test
```

### Option 2: install from conda-forge

```sh
mamba create -n cryoswath conda-forge::cryoswath
mamba activate cryoswath
```

### Option 3: editable install from source

```sh
git clone https://github.com/j-haacker/cryoswath.git
mamba env create -n cryoswath -f cryoswath/environment.yml
mamba activate cryoswath
mamba install pip
pip install --editable cryoswath
```

### Option 4: reproducible Pixi environment

```sh
git clone https://github.com/j-haacker/cryoswath.git
cd cryoswath
pixi install --locked -e test
pixi shell -e test
```

This uses the lock file and is the most robust option when dependency
resolvers disagree.

### Contributor lockfile workflow

For regular development runs:

```sh
pixi install --locked -e test
```

If you change dependency manifests (`pyproject.toml` and/or `pixi.toml`):

```sh
bash tools/sync_dependency_definitions.sh
pixi lock
pixi run -e test test-unit
pixi run -e docs docs-build
```

### Optional: Docker image

If local dependency resolution fails, you can use Docker:

```sh
docker run -it -p 8888:8888 -v <proj_dir>:/home/jovyan/project_dir cryoswath/jupyterlab:nightly
```

## Configure a project

CryoSwath keeps project data outside the package install directory. If no
configuration file is present, paths default to `./data` relative to the
current working directory. For a reusable project configuration, run the
bootstrap commands inside a project folder:

```sh
mkdir <proj_dir>
cd <proj_dir>
cryoswath create-config
cryoswath download-aux-data
cryoswath get-tutorials
```

`cryoswath create-config` writes `cryoswath.cfg` with your base data path.
`cryoswath download-aux-data` installs the Zenodo auxiliary-data baseline.
`cryoswath get-tutorials` copies packaged tutorial notebooks to `tutorials/`.
The tutorial notebooks assume CryoSwath is installed in the active Python
environment and import it directly.
You can also set `CRYOSWATH_DATA` or more specific `CRYOSWATH_*` path
variables; environment variables override config files. Set `CRYOSWATH_CONFIG`
to select a config file explicitly. Legacy `config.ini` files are still read.

To avoid storing secrets in config files, use keyring (preferred) or
environment variables for ESA credentials. You can configure keyring
credentials interactively with: `cryoswath update-keyring`.
If you need a fallback, you can write `~/.netrc` (this stores the password in
plaintext) using `cryoswath update-netrc`.

## Tutorials and documentation

- Main docs: [cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)
- General workflow tutorial:
  [`tutorials/tutorial__general_step-by-step.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__general_step-by-step.ipynb)
- First waveform tutorial:
  [`tutorials/tutorial__process_first_waveform.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__process_first_waveform.ipynb)
- First swath tutorial:
  [`tutorials/tutorial__process_first_swath.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__process_first_swath.ipynb)

## Local testing

Run the fast unit tests against the editable checkout:

```sh
pixi run -e test test-unit
```

Run the installed-package check. This builds the wheel, installs it into a
temporary environment outside the repository, and runs the unit tests against
that installed package rather than the source checkout:

```sh
pixi run -e test test-installed
```

Run the full local test pipeline:

```sh
pixi run -e test test-all
```

Run the full pipeline from a copy of the current tracked worktree with a fresh
Pixi environment and fresh home directory:

```sh
pixi run -e test test-fresh
```

For a release-style check of committed `HEAD` only, use:

```sh
pixi run -e test test-fresh-committed
```

These commands pass credential environment variables such as `EOIAM_USER`,
`EOIAM_PASSWORD`, `EARTHDATA_USERNAME`, and `EARTHDATA_PASSWORD`, but they do
not pass local CryoSwath path variables unless requested explicitly.

Run report notebooks only:

```sh
pixi run -e test test-notebooks
```

This creates `tests/reports/artifacts/project/cryoswath.cfg`, uses that
isolated data tree through `CRYOSWATH_CONFIG`, and downloads the auxiliary-data
baseline if it is missing.

Run tutorial notebooks only:

```sh
pixi run -e test test-tutorial-notebooks
```

This creates `tests/tutorials/artifacts/project/cryoswath.cfg`, copies the
packaged tutorial notebooks into that project, and uses the same isolated aux
setup. If you pass a custom tutorial directory, keep it compatible with this
generated project layout.

Run selected GitHub Actions jobs locally with `act` through Pixi. This requires
Docker or a compatible container runtime and approximates the Ubuntu CI jobs:

```sh
pixi run -e ci local-ci-pixi-test
pixi run -e ci local-ci-docs
pixi run -e ci local-ci-dependency-matrix
```

Notebook tests may download required larger data from first-hand sources
at runtime, so network availability and valid ESA credentials matter.

## External dependencies and data

CryoSwath relies on:

- Python dependencies: [requirements.txt](https://github.com/j-haacker/cryoswath/blob/main/requirements.txt)
- reference elevation models
- RGI glacier outlines

The package points to required external resources during setup and use.

## Known limitations

- ESA's data server is not reachable from all internet service providers.
- Projected RGI basin geometries can sometimes be invalid;
  use `.make_valid()` where required.
- Most testing and validation has focused on the Arctic.

Further details: [open issues](https://github.com/j-haacker/cryoswath/issues)

## Citation and attribution

If you use CryoSwath, please cite:

```bibtex
@software{cryoswath,
  author       = {Haacker, Jan},
  title        = {CryoSwath 0.2.6},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.2.6},
  doi          = {10.5281/zenodo.14825358}
}
```

Please also acknowledge upstream data/resources used in your workflow:

- ESA L1b data terms:
  [Terms and Conditions for the use of ESA Data](https://github.com/j-haacker/cryoswath/blob/main/data/L1b/Terms-and-Conditions-for-the-use-of-ESA-Data.pdf)
- RGI data license:
  [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- PGC DEM acknowledgement guidance:
  [Acknowledgement Policy](https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)

## License

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/blob/main/LICENSE.txt).
