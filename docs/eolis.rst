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

Points are associated using satellite geometry and half-open native waveform
range intervals before either version's L2 points are evaluated. Elevation agreement is not a condition for matching, so elevation
differences remain available for comparison.

Interpret the results
---------------------

Check ``coverage`` before interpreting the differences:

* ``cryoswath_points`` counts CryoSwath points with finite heights;
  ``eolis_observations`` counts EOLIS observations read for this track after
  removing exact duplicates.
* ``selected_observations`` counts distinct EOLIS observations within the
  along-track selection windows; ``matched_observations`` counts those associated
  with an available CryoSwath sample and an unambiguous raw range interval.
* ``matched_samples`` counts distinct waveform/sample IDs;
  ``records_sharing_samples`` counts matched records in many-to-one groups.
* ``ambiguous_observations`` counts records assigned to multiple raw waveforms.
  Those candidates remain in the output but are excluded from statistics.
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
The ``along_track`` window defaults to ±140 m. Sample intervals are
``[centre - half_bin, centre + half_bin)`` (half a range bin is about 11.71 cm).
The upper edge belongs to the next sample. The first and last bounds are
respected, and sample count comes from the raw waveform dimension. Elevation,
across-track distance and EOLIS timestamps do not gate matches. Missing CryoSwath
samples remain unmatched; ambiguous waveform assignments are retained and flagged.

``points`` is a GeoDataFrame at EOLIS locations. ``own_height_m``,
``own_longitude`` and ``own_latitude`` retain the evaluated processor's own
position and elevation. Its attributes retain
source files, regions and the difference convention. ``statistics`` reports count,
mean, median, sample standard deviation, RMSE, NMAD and 2.5/97.5 percentiles for each
metric. ``coverage`` also reports ``exact_duplicates_removed``.

The comparison statistic ``range_difference_m`` is reconstructed CryoSwath
satellite-to-point range minus EOLIS satellite-to-point range. Positive means
CryoSwath is farther from the satellite. It remains conditional on association.

Two metrics are retained in point data and the separate ``diagnostics`` summary,
not in the headline ``statistics``, in metres:

* ``sample_assignment_residual_m`` is native sample range minus EOLIS geometric range.
  For matched points it is constrained by the half-bin matching tolerance and
  does not independently measure product agreement.
* ``range_reconstruction_error_m`` is cached CryoSwath geometric range minus native
  sample range. It checks consistency of the cached geometry with the sample
  used for association.

.. autofunction:: cryoswath.eolis.compare_track

Reuse associations across processing versions
--------------------------------------------

``sample_ranges(raw)`` returns centre/lower/upper ranges in metres on dimensions
``time_20_ku`` and ``sample``. Sample IDs are zero-based native positions, even
when the raw coordinate has other labels. Open raw L1b with
``decode_timedelta=False`` to preserve window-delay precision.

``associate_points(raw, reference)`` uses every raw waveform, independently of
L2 filtering. ``reference`` is an EOLIS GeoDataFrame with an ``elevation`` column.
The returned table retains original reference columns, ``reference_position``,
``waveform_time``, ``sample``, ``native_range_m``, ``sample_assignment_residual_m``,
``in_range``, ``ambiguous`` and ``candidate_matched``. Distinct EOLIS records
assigned to one sample are retained. ``reference_position`` is a temporary
row index; archive source product/checksum/row identities for durable reuse.

``evaluate_associations(raw, swath, associations)`` joins another version's L2
points by its ``(time, sample)`` index, recomputes position/elevation differences,
and returns ``points``, ``coverage`` and ``statistics``. Empty L2 is permitted;
missing samples are never replaced with neighbours. Legacy UTC labels on L2
indices are removed for the native TAI identity join without moving timestamps.

``compare_track`` returns the raw ``associations`` table alongside its evaluation.
Raw product identity, geometry fingerprint and available source-file checksum
are stored in portable columns and validated when evaluating saved associations.
Keep the association attributes and parameter/source manifest separately when
saving an empty table, because Feather/Parquet do not preserve dataframe attrs.
The caller must also preserve and verify EOLIS source checksums and processing
configuration in that manifest.

.. autofunction:: cryoswath.eolis.sample_ranges
.. autofunction:: cryoswath.eolis.associate_points
.. autofunction:: cryoswath.eolis.evaluate_associations
