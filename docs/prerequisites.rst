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
   `ESA EO account <https://eoiam-idp.eo.esa.int/>`_.

Set up your ESA credentials before running download workflows.

Anonymous FTP login is no longer supported.

Credential resolution order
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CryoSwath resolves ESA credentials in this order:

1. ``EOIAM_USER`` and ``EOIAM_PASSWORD``.
2. Keyring (preferred for interactive setup).
3. ``~/.netrc`` entry for ``science-pds.cryosat.esa.int`` with
   explicit ``login`` and ``password`` (plaintext fallback).
4. Legacy ``config.ini`` values in ``[user]`` using ``name`` and
   ``password`` (temporary fallback).

Preferred interactive setup (keyring):

.. code-block:: sh

   cryoswath update-keyring

Automation setup (environment variables):

.. code-block:: sh

   export EOIAM_USER="your-esa-user"
   export EOIAM_PASSWORD="your-esa-password"

Plaintext fallback setup (``~/.netrc``):

.. code-block:: sh

   cryoswath update-netrc

``~/.netrc`` stores the password in plaintext and should only be used as a
fallback if keyring is not available.

Legacy ``config.ini [user] name/password`` credentials still work for
now, but are deprecated and should be replaced.

Download protocol defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^

L1b data downloads are HTTPS-first by default. FTP remains available as an
automatic fallback for download failures and is still used for metadata
refresh flows that compare local catalogs with upstream.

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
Run ``cryoswath update-tracks`` periodically to extend or refresh the local
track database after installing the baseline.

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
- Supported xarray window: ``>=2025.3,<2025.12``

The root ``requirements.txt`` is generated from ``pyproject.toml`` by
``tools/sync_dependency_definitions.sh``. It is kept for compatibility but is
not the primary dependency source.

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

   bash tools/sync_dependency_definitions.sh
   pixi lock
   pixi run -e test test-unit
   pixi run -e docs docs-build
