Mission Portability (CRISTAL Readiness)
=======================================

This page lists the main CryoSat-2 assumptions in CryoSwath and where to
adapt them when porting to a different mission such as CRISTAL.

Code locations to review first
------------------------------

1. Mission constants and ID parsing in :mod:`cryoswath.misc`.

   - ``antenna_baseline``, ``Ku_band_freq``, ``sample_width``
   - ``cryosat_id_pattern``

2. L1b filename and track ID assumptions in :mod:`cryoswath.l1b`.

   - ``read_esa_l1b`` filename match: ``*CS_????_SIR_SIN_1B_*.nc``
   - track time slicing from filenames, e.g. ``remote_file[19:34]``
   - local path conventions for ``L1b`` products

3. Remote data endpoint and folder layout in :mod:`cryoswath.misc` and
   :mod:`cryoswath.l1b`.

   - ``ftp_cs2_server`` host: ``science-pds.cryosat.esa.int``
   - remote directories under ``/SIR_SIN_L1/<year>/<month>``
   - HTTP fallback URL templates in ``download_single_file``

4. L1b flag semantics in :mod:`cryoswath.l1b` and :mod:`cryoswath.misc`.

   - ``build_flag_mask`` and ``flag_translator`` rely on ``flag_masks`` or
     ``flag_values`` metadata and current CryoSat naming.
   - Ensure CRISTAL flag attributes and meanings are mapped consistently.

5. Auxiliary databases and paths in :mod:`cryoswath.misc`.

   - file name catalog: ``CryoSat-2_SARIn_file_names.pkl``
   - track table: ``CryoSat-2_SARIn_ground_tracks.feather``
   - default path layout rooted at ``data/`` and ``data/auxiliary/``

CRISTAL adaptation checklist
----------------------------

1. Define mission-specific constants (frequency/baseline/sample geometry).
2. Replace filename pattern checks and timestamp extraction logic.
3. Implement mission-specific remote discovery/download paths.
4. Validate flag decoding against CRISTAL metadata fields.
5. Introduce mission-specific auxiliary catalog file names if needed.
6. Run L1b -> L2 smoke tests on a small set of known CRISTAL tracks.

Notes
-----

- The current implementation is explicitly CryoSat-2 focused. Portability is
  practical, but not yet abstracted behind a mission interface.
- Start by isolating mission-specific literals, then move them into a small
  mission configuration layer before broader refactors.
