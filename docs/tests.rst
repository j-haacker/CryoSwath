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
5. `scripts/tutorial__general_step-by-step.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__general_step-by-step.ipynb>`_
   End-to-end processing tutorial executed directly in tests.
6. `scripts/tutorial__process_first_waveform.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_waveform.ipynb>`_
   Waveform tutorial executed directly in tests.
7. `scripts/tutorial__process_first_swath.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__process_first_swath.ipynb>`_
   Swath tutorial executed directly in tests.
8. `scripts/tutorial__poca.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial__poca.ipynb>`_
   POCA tutorial executed directly in tests.

Run all report notebooks through Snakemake with Pixi:

.. code-block:: bash

   pixi run -e test test-notebooks

Run tutorial notebooks through Snakemake with Pixi:

.. code-block:: bash

   pixi run -e test test-tutorial-notebooks

If tutorials are stored outside the current checkout, set
``CRYOSWATH_TUTORIAL_DIR`` to the directory that contains
``tutorial__*.ipynb`` before running ``test-tutorial-notebooks``.

Run unit tests + report notebooks + tutorial notebooks:

.. code-block:: bash

   pixi run -e test test-all

Notebooks starting with ``0-l4_`` are intentionally excluded from this workflow.

These notebooks are smoke/regression tests, not a full scientific
validation campaign.

Some notebook tests fetch larger datasets from first-hand online sources
at runtime. External network outages and credential issues can therefore
cause test failures.
