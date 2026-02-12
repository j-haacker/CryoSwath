API Overview
============

Processing levels
-----------------

CryoSwath is organized around data processing levels:

1. :mod:`cryoswath.l1b`
   Load and preprocess ESA CryoSat-2 SARIn L1b tracks, including
   waveform filtering and ambiguity handling.
2. :mod:`cryoswath.l2`
   Convert processed tracks to geolocated point elevations (swath/POCA)
   and optionally cache/export.
3. :mod:`cryoswath.l3`
   Aggregate L2 point observations into regular spatio-temporal grids.
4. :mod:`cryoswath.l4`
   Gap-fill gridded products and derive trend/time-series products.

Supporting modules
------------------

- :mod:`cryoswath.misc`
  Shared utilities (paths, I/O, interpolation helpers, patches, data
  access, and CLI helpers).
- :mod:`cryoswath.gis`
  Geospatial helper functions for CRS handling and geometry operations.
- :mod:`cryoswath.test_plots`
  Plotting utilities for quick validation and diagnostics.

Typical workflow
----------------

1. Discover/select tracks for a region/time range.
2. Process L1b to L2 with :func:`cryoswath.l2.from_id`.
3. Build gridded products with :func:`cryoswath.l3.build_dataset`.
4. Produce higher-level products with :mod:`cryoswath.l4`.
