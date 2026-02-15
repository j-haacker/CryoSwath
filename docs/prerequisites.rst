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
   pixi install
   pixi run -e test test-unit

For interactive work inside the environment:

.. code-block:: sh

   pixi shell -e test

Alternative: install from source

.. code-block:: sh

   git clone https://github.com/j-haacker/cryoswath.git
   pip install --editable ./cryoswath

Then initialize your project directory:

.. code-block:: sh

   mkdir <project_dir>
   cd <project_dir>
   cryoswath-init

``cryoswath-init`` creates a project layout (``data/``, ``scripts/``) and
writes ``scripts/config.ini`` that stores your base data path.

Access requirements
-------------------

.. warning::
   Starting **Monday, February 16, 2026**, downloading CryoSat resources
   via CryoSwath requires an
   `ESA EO account <https://eoiam-idp.eo.esa.int/>`_.

Set up your ESA credentials before running download workflows.

Data dependencies
-----------------

CryoSwath requires:

1. A reference DEM (currently ArcticDEM/REMA via
   :func:`cryoswath.misc.get_dem_reader`).
2. RGI v7 glacier/region geometries for most region-based workflows.

Expected default locations:

- DEMs: ``data/auxiliary/DEM``
- RGI files: ``data/auxiliary/RGI``

You can override paths in ``config.ini`` or by adapting path handling in
:mod:`cryoswath.misc`.

Software dependencies
---------------------

Python package dependencies are defined in ``pyproject.toml``.

- Runtime dependencies: ``[project.dependencies]``
- Optional docs/dev extras: ``[project.optional-dependencies]``
- Supported Python version: ``>=3.11`` (regularly tested on 3.11 and 3.12)
- Supported xarray window: ``>=2025.3,<2025.12``

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
