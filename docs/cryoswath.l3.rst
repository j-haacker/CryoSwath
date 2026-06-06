cryoswath.l3 module
===================

Finalized NetCDF Extension
--------------------------

Use :func:`cryoswath.l3.merge_finalized_dataset_extension` for CF-style
finalized L3 NetCDF products. It validates matching schema, static variables,
non-time coordinates, CRS metadata, and all overlapping time-dependent values
before appending the extension suffix and adding a provenance line to
``history``.

.. automodule:: cryoswath.l3
   :members:
   :undoc-members:
   :show-inheritance:
