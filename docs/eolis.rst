Compare a track with EOLIS
==========================

EOLIS provides CryoSat swath elevation points. Compare them with CryoSwath points
from the same track to examine differences in elevation and position. The result
includes coverage counts, matched point differences and summary statistics.

Run a comparison
----------------

Use a configured CryoSwath processing project with ESA PDS credentials. Relevant
EOLIS files are retrieved automatically and cached; existing CryoSwath L2 products
are reused, and missing L2 is processed automatically.

.. code-block:: python

   from cryoswath import eolis

   result = eolis.compare_track("20240215T195335")
   result["coverage"]
   result["statistics"].loc[
       ["height_difference_m", "along_difference_m",
        "across_difference_m", "horizontal_distance_m"]
   ]
   matched = result["points"].loc[result["points"].matched]

Points are associated using satellite geometry and the nearest native waveform
range sample. Elevation agreement is not a condition for matching, so elevation
differences remain available for comparison.

Interpret the results
---------------------

Check ``coverage`` before interpreting the differences:

* ``cryoswath_points`` counts CryoSwath points with finite heights;
  ``eolis_observations`` counts EOLIS observations read for this track after
  removing exact duplicates.
* ``selected_observations`` counts distinct EOLIS observations within the
  along-track selection windows; ``matched_observations`` counts those associated
  with an available CryoSwath sample and accepted by the range tolerance.
* ``missing_cached_samples`` and ``outside_radial_tolerance`` count unsuccessful
  candidate associations. One EOLIS observation can be considered for several
  waveforms, so these counts do not partition the distinct selected observations.

These counts describe the available EOLIS comparison, not the fraction of all
CryoSwath points covered. ``points`` contains both matched and unmatched candidate
associations; ``statistics`` summarizes only matched observations.

Elevation, along-track and across-track differences are CryoSwath minus EOLIS,
in metres. Along-track is forward and across-track is right of flight. Horizontal
distance is the norm of the along/across difference. Positive elevation differences
mean CryoSwath is higher. Check the location and spread of the distributions and
whether elevation differences vary spatially.

The statistics describe agreement between products, not independent accuracy
estimates. Height datum and processing choices must be compatible. No extra quality
or uncertainty filtering is applied, and the reported percentiles are distribution
bounds, not confidence intervals.

Options and output reference
----------------------------

``reprocess=True`` regenerates CryoSwath processing results. Additional processing
keywords are forwarded to ``l2.from_id`` when processing is needed. EOLIS downloads
are cached under the configured auxiliary directory's ``EOLIS`` folder; use
``cache_dir`` to select another location. If multiple EOLIS versions are available,
select one with ``version``, for example ``"V301"``. Unsupported RGI regions are
listed in the point table's attributes; an entirely unsupported track raises an
error.

Association uses a satellite-centred frame derived from L1b position and velocity.
The ``along_track`` window defaults to ±170 m. The nearest native sample must be
within half a range bin (about 11.71 cm) of the EOLIS geometric range. Elevation,
across-track distance and EOLIS timestamps do not gate matches. Missing CryoSwath
samples remain unmatched; ambiguous matches raise an error.

``points`` is a GeoDataFrame in EPSG:4326 at EOLIS locations. Its attributes retain
source files, regions and the difference convention. ``statistics`` reports count,
mean, median, sample standard deviation, RMSE, NMAD and 2.5/97.5 percentiles for each
metric. ``coverage`` also reports ``exact_duplicates_removed``.

Two additional metrics diagnose range consistency, in metres:

* ``radial_difference_m`` is native sample range minus EOLIS geometric range.
  For matched points it is constrained by the half-bin matching tolerance and
  does not independently measure product agreement.
* ``geometry_range_bias_m`` is cached CryoSwath geometric range minus native
  sample range. It checks consistency of the cached geometry with the sample
  used for association.

.. autofunction:: cryoswath.eolis.compare_track
