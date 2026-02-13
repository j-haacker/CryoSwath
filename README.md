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

- Install CryoSwath in a dedicated environment (`conda`/`mamba`,
  `venv`, or `uv`). The dependency tree is broad, and future
  dependency conflicts are otherwise likely.
- Starting **Monday, February 16, 2026**, downloading CryoSat resources
  requires an **[ESA EO account](https://eoiam-idp.eo.esa.int/)**.
- Install `xarray` and `zarr` together to avoid version mismatches.

## Installation

For full setup details, see the docs:
[cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)

### Option 1: install from conda-forge

```sh
mamba create -n cryoswath conda-forge::cryoswath
mamba activate cryoswath
```

### Option 2: editable install from source

```sh
git clone https://github.com/j-haacker/cryoswath.git
mamba env create -n cryoswath -f cryoswath/environment.yml
mamba activate cryoswath
mamba install pip
pip install --editable cryoswath
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

## Tutorials and documentation

- Main docs: [cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)
- General workflow tutorial:
  [`scripts/tutorial__general_step-by-step.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__general_step-by-step.ipynb)
- First waveform tutorial:
  [`scripts/tutorial__process_first_waveform.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_waveform.ipynb)
- First swath tutorial:
  [`scripts/tutorial__process_first_swath.ipynb`](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_swath.ipynb)

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
  title        = {CryoSwath: v0.2.4},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.2.4},
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
