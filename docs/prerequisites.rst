Prerequisites
=============

Installation
------------

The recommended setup for development and reproducible workflows is an
isolated Python environment (``pixi``, ``conda/mamba``, ``venv``, ``uv``, etc.).

.. warning::
   CryoSwath has a broad dependency tree. To avoid future dependency
   incompatibilities, install it in a dedicated environment.

Recommended: pixi-managed environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

   git clone https://github.com/j-haacker/cryoswath.git
   cd cryoswath
   pixi install --locked -e test
   pixi run -e test test-unit

For interactive work inside the environment:

.. code-block:: sh

   pixi shell -e test

Optional: Docker image
^^^^^^^^^^^^^^^^^^^^^^

If local dependency resolution fails, you can use Docker. The image provides
the CryoSwath runtime and a JupyterLab kernel named ``Python (CryoSwath)``;
initialize your project inside the mounted project directory:

.. code-block:: sh

   docker run -it -p 8888:8888 -v <proj_dir>:/home/jovyan/project_dir cryoswath/jupyterlab:nightly

In the docker case, do the configuration (also see below) in a
JupyterLab terminal inside the container:

.. code-block:: sh

   cryoswath create-config
   cryoswath download-aux-data
   cryoswath get-tutorials
   python -c "from cryoswath.misc import update_netrc as up; up(<user>, <pw>, netrc_file='.netrc')"


Here, to make use of the automatic RGI basin shape download in
notebooks, you will need to add a cell that exports your credentials to
the current environment:

.. code-block:: python

   import os
   os.environ["EARTHDATA_USERNAME"] = <user>
   os.environ["EARTHDATA_PASSWORD"] = <pw>

Alternative: install from source

.. code-block:: sh

   git clone https://github.com/j-haacker/cryoswath.git
   pip install --editable ./cryoswath

Then configure your project paths. Without a config file, CryoSwath uses
``./data`` relative to the current working directory. To create a reusable
project config and install the default auxiliary baseline, run:

.. code-block:: sh

   mkdir <project_dir>
   cd <project_dir>
   cryoswath create-config
   cryoswath download-aux-data
   cryoswath get-tutorials

``cryoswath create-config`` writes ``cryoswath.cfg`` with your base data path.
``cryoswath download-aux-data`` installs the Zenodo auxiliary-data baseline.
``cryoswath get-tutorials`` copies packaged tutorial notebooks into
``tutorials/``. The notebooks import CryoSwath from the active Python
environment. You can also set ``CRYOSWATH_DATA`` or more specific
``CRYOSWATH_*`` path variables; environment variables override config files.
Set ``CRYOSWATH_CONFIG`` to select a config file explicitly. Legacy
``config.ini`` and ``scripts/config.ini`` files are still read.

Access requirements
-------------------

.. warning::
   Downloading CryoSat resources via CryoSwath requires an
   `ESA EO account <https://eoiam-idp.eo.esa.int/>`_. MAAP assets require a
   personal offline token; Science Server FTP fallback uses ESA credentials.

MAAP catalog discovery is anonymous, but downloading a selected asset requires
an offline token. Generate a 90-day token through the
`ESA MAAP portal <https://portal.maap.eo.esa.int/ini/services/auth/token/90dToken.php>`_.

MAAP token setup
^^^^^^^^^^^^^^^^

CryoSwath resolves the MAAP offline token in this order:

1. ``ESA_MAAP_OFFLINE_TOKEN``.
2. Keyring (preferred for interactive setup).

Preferred interactive setup (keyring):

.. code-block:: sh

   cryoswath update-maap-token

Automation setup (environment variables):

.. code-block:: sh

   export ESA_MAAP_OFFLINE_TOKEN="your-personal-offline-token"

This token is distinct from an ESA username/password. Do not store it in
``cryoswath.cfg``, legacy ``config.ini``, or ``~/.netrc``.

Science Server credentials
^^^^^^^^^^^^^^^^^^^^^^^^^^

Science Server FTP fallback delivery uses ``EOIAM_USER`` and
``EOIAM_PASSWORD``, keyring, ``~/.netrc``, or legacy ``config.ini``. Those
credentials are not used for MAAP asset delivery.

Download protocol defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^

L1b workflows query MAAP STAC metadata (``CryoSatIceL110``) first. A selected
MAAP ``enclosure_nc`` or ``enclosure`` URL is downloaded with a short-lived
Bearer token. When MAAP has no usable selected asset or is unavailable,
CryoSwath scans the matching month on the authenticated Science Server using
FTP_TLS and selects the best same-track product. MAAP transfer failures are
reported and do not silently fall back. PDS HTTPS URLs are not constructed
from filenames.

An anonymous FTP connection can be established, but it cannot list or retrieve
CryoSat product directories. It is therefore not a discovery or download path.

Data dependencies
-----------------

CryoSwath requires:

1. A reference DEM (currently ArcticDEM/REMA via
   :func:`cryoswath.misc.get_dem_reader`).
2. RGI v7 glacier/region geometries for most region-based workflows.

Expected default locations:

- DEMs: ``data/auxiliary/DEM``
- RGI files: ``data/auxiliary/RGI``

You can override paths in ``cryoswath.cfg`` or with environment variables
such as ``CRYOSWATH_DATA``, ``CRYOSWATH_DEM``, and ``CRYOSWATH_RGI``.


Auxiliary-data baseline
^^^^^^^^^^^^^^^^^^^^^^^

The command ``cryoswath download-aux-data`` downloads the latest CryoSwath
auxiliary-data snapshot from Zenodo DOI ``10.5281/zenodo.20241526`` and
extracts it into ``data/auxiliary`` by default. The snapshot contains the
CryoSat-2 ground-track database, filename catalog, and static RGI metadata.
Newer installations may also contain the richer
``CryoSat-2_SARIn_L1B_track_catalog.feather`` cache, which stores MAAP item
IDs, product filenames, selected enclosure URLs, product versions,
processing dates, and track geometries for supported ``SIR_SIN_1B`` products.
Cached entries without a usable route are refreshed before delivery.

By default, :func:`cryoswath.misc.load_cs_ground_tracks` uses local track
caches when their latest timestamp covers the requested period. If the request
extends beyond local coverage and a network connection is available, CryoSwath
queries MAAP STAC metadata, caches the supported products, and then returns
the combined local result. Pass
``source="local"`` to force offline/local-only behavior, or ``source="stac"``
to force remote catalogue metadata discovery for the requested period.

The STAC-backed selector currently accepts validated CryoSat Baseline D/E/F
``SIR_SIN_1B`` products. If a newer unsupported baseline is seen, CryoSwath
warns and excludes those products rather than feeding unvalidated files into
the processing chain. Existing local NetCDF files are not replaced
automatically; newer-product replacement is intentionally left as an explicit
future workflow.

Run ``cryoswath update-tracks`` periodically to extend or refresh the local
track metadata after installing the baseline. This refreshes STAC-backed
metadata where possible and leaves the legacy filename catalog available for
older workflows.

DEM download behavior
^^^^^^^^^^^^^^^^^^^^^

If the default ArcticDEM or REMA 100 m ``*_dem.tif`` file is missing,
``get_dem_reader`` now attempts an automatic download and extraction before
raising ``FileNotFoundError``.

- Arctic source archive:
  ``https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v4.1/100m/arcticdem_mosaic_100m_v4.1.tar.gz``
- Antarctic source archive:
  ``https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/100m/rema_mosaic_100m_v2.0_filled_cop30.tar.gz``
- Downloaded archives are extracted into ``data/auxiliary/DEM`` and removed
  after successful extraction.

RGI download behavior
^^^^^^^^^^^^^^^^^^^^^

If a required RGI o1 region file is missing, CryoSwath now attempts an
automatic NSIDC download before raising ``FileNotFoundError``.

RGI downloads use NASA Earthdata credentials through ``earthaccess``, not ESA
credentials. For non-interactive downloads, provide either:

1. ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD``.
2. ``EARTHDATA_TOKEN`` instead of username/password.

Downloaded zip archives are extracted into ``data/auxiliary/RGI`` using a
directory named like the archive stem, for example
``RGI2000-v7.0-C-09_svalbard``. Zip archives are removed after successful
extraction.

You can also prefetch a region explicitly:

.. code-block:: sh

   cryoswath download-rgi --o1 09 --product complexes

Software dependencies
---------------------

Python package dependencies are defined in ``pyproject.toml``.

- Runtime dependencies: ``[project.dependencies]``
- Optional docs/dev extras: ``[project.optional-dependencies]``
- Supported Python version: ``>=3.12``

The root ``requirements.txt`` is kept for compatibility but is not the
primary dependency source.

Dependency strategy
-------------------

CryoSwath supports two installation modes:

1. Stable/reproducible: use ``pixi.lock`` or ``environment.yml``.
2. Flexible: install from ``pyproject.toml`` bounds (pip/uv workflows).

Use the stable mode for tutorials, bug reports, and scientific
reproducibility. Use the flexible mode when integrating CryoSwath into
an existing environment.

Contributor lockfile workflow
-----------------------------

For regular development runs:

.. code-block:: sh

   pixi install --locked -e test

If you change dependency manifests (``pyproject.toml`` and/or ``pixi.toml``):

.. code-block:: sh

   pixi lock
   pixi run -e test test-unit
   pixi run -e docs docs-build
