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
- Supported Python version: **>=3.11** (regularly tested on 3.11 and 3.12).
- Starting **Monday, February 16, 2026**, downloading CryoSat resources
  requires an **[ESA EO account](https://eoiam-idp.eo.esa.int/)**.
- FTP credentials are resolved in this order:
  `~/.netrc` (with explicit login/password), then
  `CRYOSWATH_FTP_USER`/`CRYOSWATH_FTP_PASSWORD`, then legacy `config.ini [user]`
  `name/password` (temporary fallback).
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
pixi lock
pixi run -e test test-unit
pixi run -e docs docs-build
```

### Optional: Docker image

If local dependency resolution fails, you can use Docker:

```sh
docker run -it -p 8888:8888 -v <proj_dir>:/home/jovyan/project_dir cryoswath/jupyterlab:nightly
```

## Initialize a project

CryoSwath expects project data outside the package install directory.
Run `cryoswath-init` inside a new project folder:

```sh
mkdir <proj_dir>
cd <proj_dir>
cryoswath-init
```

`cryoswath-init` sets up the expected data structure and writes
`scripts/config.ini` with your base data path. The paths can be
reconfigured in `config.ini` if you use a different layout.

To avoid storing secrets in `config.ini`, use `~/.netrc` (preferred) or
environment variables for FTP credentials and keep `config.ini` focused on
paths.
To create or update your `~/.netrc` entry interactively, run:
`cryoswath-update-netrc`.

## Tutorials and documentation

- Main docs: [cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)
- General workflow tutorial:
  [`scripts/tutorial__general_step-by-step.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__general_step-by-step.ipynb)
- First waveform tutorial:
  [`scripts/tutorial__process_first_waveform.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_waveform.ipynb)
- First swath tutorial:
  [`scripts/tutorial__process_first_swath.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_swath.ipynb)

## Local testing

Run the full local test pipeline:

```sh
pixi run -e test test-all
```

Run report notebooks only:

```sh
pixi run -e test test-notebooks
```

Run tutorial notebooks only:

```sh
pixi run -e test test-tutorial-notebooks
```

If tutorials are stored outside the current checkout, set
`CRYOSWATH_TUTORIAL_DIR` to the directory containing
`tutorial__*.ipynb` before running this task.

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
  title        = {CryoSwath: v0.2.5},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.2.5},
  doi          = {10.5281/zenodo.17011635}
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
