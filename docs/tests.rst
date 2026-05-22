Tests
=====

Validation notebooks are located in ``tests/reports`` and provide
regression/sanity checks for key processing steps.

Recommended checks after changing core processing logic:

1. `tests/reports/l1b_swath_start.py.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_swath_start.py.ipynb>`_
   Edge cases for identifying swath start.
2. `tests/reports/l1b_waveform.py.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_waveform.py.ipynb>`_
   Waveform-level geometry and elevation sanity checks.
3. `tests/reports/l2_dem_comparison.py.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_dem_comparison.py.ipynb>`_
   L2 elevation comparison against the reference DEM.
4. `tests/reports/l2_tested_data_comparison.py.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_tested_data_comparison.py.ipynb>`_
   Comparison against validated reference output.
5. `cryoswath/tutorials/tutorial__diagnostic_hooks.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__diagnostic_hooks.ipynb>`_
   Diagnostic-hook exploration tutorial executed directly in tests.
6. `cryoswath/tutorials/tutorial__general_step-by-step.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__general_step-by-step.ipynb>`_
   End-to-end processing tutorial executed directly in tests.
7. `cryoswath/tutorials/tutorial__process_first_waveform.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__process_first_waveform.ipynb>`_
   Waveform tutorial executed directly in tests.
8. `cryoswath/tutorials/tutorial__process_first_swath.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__process_first_swath.ipynb>`_
   Swath tutorial executed directly in tests.
9. `cryoswath/tutorials/tutorial__poca.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/cryoswath/tutorials/tutorial__poca.ipynb>`_
   POCA tutorial executed directly in tests.

Run all report notebooks through Snakemake with Pixi:

.. code-block:: bash

   pixi run -e test test-notebooks

This task prepares ``tests/reports/artifacts/project/cryoswath.cfg``, sets
``CRYOSWATH_CONFIG`` to that file for the notebook run, and downloads the
auxiliary-data baseline if the required catalog files are missing.

Run tutorial notebooks through Snakemake with Pixi:

.. code-block:: bash

   pixi run -e test test-tutorial-notebooks

This task prepares ``tests/tutorials/artifacts/project/cryoswath.cfg``, copies
the packaged ``tutorial__*.ipynb`` files into that project with
``cryoswath get-tutorials`` semantics, and uses the generated project as the
default Snakemake ``--directory``. If you pass a custom tutorial directory,
keep it compatible with the generated project layout and ``CRYOSWATH_CONFIG``.

Run unit tests + report notebooks + tutorial notebooks:

.. code-block:: bash

   pixi run -e test test-all

Run the same pipeline from a copy of the current tracked worktree with a fresh
Pixi test environment and fresh home directory:

.. code-block:: bash

   pixi run -e test test-fresh

For a release-style check of committed ``HEAD`` only, use:

.. code-block:: bash

   pixi run -e test test-fresh-committed

``test-fresh`` passes credential variables such as ``EOIAM_USER``,
``EOIAM_PASSWORD``, ``EARTHDATA_USERNAME``, and ``EARTHDATA_PASSWORD`` through
to the temporary environment. It intentionally drops local CryoSwath path
overrides by default; pass extra variables with ``--pass-env NAME`` when needed.
Untracked files are not copied into the fresh checkout.

Run the fast unit tests against the editable checkout:

.. code-block:: bash

   pixi run -e test test-unit

Run the installed-package unit tests, which build the wheel, install it into a
temporary environment outside the repository, and guard against importing from
the source checkout:

.. code-block:: bash

   pixi run -e test test-installed

Run selected GitHub Actions jobs locally with ``act`` through Pixi. These
commands require Docker or a compatible container runtime and approximate the
hosted Ubuntu CI jobs:

.. code-block:: bash

   pixi run -e ci local-ci-pixi-test
   pixi run -e ci local-ci-docs
   pixi run -e ci local-ci-dependency-matrix

Notebooks starting with ``0-l4_`` are intentionally excluded from this workflow.

These notebooks are smoke/regression tests, not a full scientific
validation campaign.

Some notebook tests fetch larger datasets from first-hand online sources
at runtime. External network outages and credential issues can therefore
cause test failures.
