Tests
=====

Validation notebooks are located in ``tests/reports`` and provide
regression/sanity checks for key processing steps.

Recommended checks after changing core processing logic:

1. `tests/reports/l1b_swath_start.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_swath_start.ipynb>`_
   Edge cases for identifying swath start.
2. `tests/reports/l1b_waveform.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_waveform.ipynb>`_
   Waveform-level geometry and elevation sanity checks.
3. `tests/reports/l2_dem_comparison.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_dem_comparison.ipynb>`_
   L2 elevation comparison against the reference DEM.
4. `tests/reports/l2_tested_data_comparison.ipynb
   <https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_tested_data_comparison.ipynb>`_
   Comparison against validated reference output.

Run all report notebooks through Snakemake with Pixi:

.. code-block:: bash

   pixi run -e test test-notebooks

Notebooks starting with ``0-l4_`` are intentionally excluded from this workflow.

These notebooks are smoke/regression tests, not a full scientific
validation campaign.
