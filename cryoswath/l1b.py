"""cryoswath.l1b module

Read and preprocess ESA CryoSat-2 SARIn L1b tracks.

This module covers waveform-level preprocessing, ambiguity handling, DEM
referencing, and conversion-ready outputs for L2 generation.
"""

__all__ = [
    "from_id",
    "read_esa_l1b",
    "download_wrapper",
    "append_ambiguous_reference_elevation",
    "append_best_fit_phase_index",
    "append_elev_diff_to_ref",
    "append_exclude_mask",
    "append_poca_and_swath_idxs",
    "append_smoothed_complex_phase",
    "build_flag_mask",
    "download_files",
    "download_single_file",
    "drop_waveform",
    "get_phase_jump",
    "get_phase_outlier",
    "get_rgi_o2",
    "if_not_empty",
    "locate_ambiguous_origin",
    "noise_val",
    "ref_range",
    "tag_groups",
    "unwrap_phase_diff",
]

import fnmatch
import ftplib
import numbers
import operator
import os
import tempfile
import time
import warnings
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urljoin, urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray as rioxr
import shapely
import xarray as xr
from numpy.typing import ArrayLike
from pyproj import Transformer
from scipy.stats import median_abs_deviation, ttest_ind

from cryoswath.gis import (
    buffer_4326_shp,
    ensure_pyproj_crs,
    find_planar_crs,
    subdivide_region,
)
from cryoswath.l2 import from_processed_l1b as l2_from_processed_l1b
from cryoswath.misc import (
    Ku_band_freq,
    WGS84_ellpsoid,
    _cryosat_l1b_product_sort_key,
    _preferred_cryosat_l1b_name,
    _resolve_esa_ftp_credentials,
    antenna_baseline,
    cs_time_to_id,
    empty_GeoDataFrame,
    ftp_cs2_server,
    gauss_filter_DataArray,
    get_dem_reader,
    l1b_path,
    load_cs_full_file_names,
    load_cs_ground_tracks,
    load_cs_l1b_track_catalog,
    load_glacier_outlines,
    monkeypatch,
    nan_unique,
    patched_xr_decode_scaling,
    patched_xr_decode_tDel,
    request_workers,
    rgi_path,
    sample_width,
    speed_of_light,
)

# requires implicitly rasterio(?), flox(?), dask(?)

_ESA_HTTPS_LOGIN_URL = "https://science-pds.cryosat.esa.int/?do=login"
_ESA_LOGIN_FAILURE_MARKERS = ("authFailure=true", "login.fail.message")


def _status(message: str) -> None:
    """Print one standardized user-facing status line."""
    print(f"[l1b] {message}")


def if_not_empty(func):
    """Skip processing helpers for tracks without waveforms."""

    def wrapper(l1b_data, *args, **kwargs):
        if len(l1b_data.time_20_ku) == 0:
            return l1b_data
        return func(l1b_data, *args, **kwargs)

    return wrapper


def noise_val(vec: ArrayLike) -> float:
    """calculate average noise values for waveform

    Args:
        vec (ArrayLike): First few (well more than 30) samples of power waveform.

    Returns:
        float: Noise power
    """
    # use sufficiently large slices (well more than 6 members)
    n = 30  # slice_thickness
    # iterate over slices: use those of which the average
    # does not significantly differ from previous slices
    # collectively
    for i in range(round(len(vec) / n) - 1):  # look at first quarter samples
        if (
            ttest_ind(
                vec[: (i + 1) * n], vec[(i + 1) * n : (i + 2) * n], equal_var=False
            ).pvalue
            < 0.001
        ):
            return np.mean(vec[: (i + 1) * n])
    return np.mean(vec)


def read_esa_l1b(
    l1b_filename: str,
    *,
    waveform_selection: int | pd.Timestamp | list[int | pd.Timestamp] | slice = None,
    drop_waveforms_by_flag: dict[str, list] = {
        "flag_mcd_20_ku": [
            "block_degraded",
            "blank_block",
            "datation_degraded",
            "orbit_prop_error",
            "echo_saturated",
            "other_echo_error",
            "sarin_rx1_error",
            "sarin_rx2_error",
            "window_delay_error",
            "agc_error",
            "trk_echo_error",
            "echo_rx1_error",
            "echo_rx2_error",
            "npm_error",
            "power_scale_error",
        ]
    },
    mask_coherence_gt1: bool = True,
    drop_outside: float = 30_000,
    coherence_threshold: float = 0.6,
    power_threshold: tuple = ("snr", 10),
    smooth_phase_difference: bool = True,
    use_original_noise_estimates: bool = False,
    dem_file_name_or_path: str | None = None,
    swath_start_kwargs: dict = {},
) -> xr.Dataset:
    """Loads ESA SARIn L1b and does initial processing

    Args to init:
        l1b_filename (str): File to read data from.

        waveform_selection (int | pd.Timestamp | list[int |
            pd.Timestamp] | slice, optional): Waveforms to retrieve data
            from. If none provided, retrieve all data. Defaults to None.

        drop_waveforms_by_flag (dict[str, list], optional):
            Exclude waveform based on flags. Defaults to
            {"flag_mcd_20_ku", [ 'block_degraded', 'blank_block',
            'datation_degraded', 'orbit_prop_error', 'echo_saturated',
            'other_echo_error', 'sarin_rx1_error', 'sarin_rx2_error',
            'window_delay_error', 'agc_error', 'trk_echo_error',
            'echo_rx1_error', 'echo_rx2_error', 'npm_error',
            'power_scale_error']}.

        mask_coherence_gt1 (bool, optional): Defaults to True.

        drop_outside (float, optional): Exclude waveforms where nadir is
            a chosen distance in meters outside of any RGI glacier. If
            None, no waveforms are excluded. Defaults to 30_000.

        coherence_threshold (float, optional): Exclude waveform samples
            with a lower coherence. This choice also affects the
            grouping, start sample for swath processing per waveform,
            and the POCA retrieval. Defaults to 0.6.

        power_threshold (tuple, optional): Similar to the coherence
            threshold, but does not affect swath start or POCA
            retrieval. Defaults to ("snr", 10).
    """
    # ! tbi customize or drop misleading attributes of xr.Dataset
    # currently only originally named CryoSat-2 SARIn files implemented
    if not fnmatch.fnmatch(l1b_filename, "*CS_????_SIR_SIN_1B_*.nc"):
        raise ValueError(
            "Provided filename deviates from standard form. That is currently not "
            "permitted, but feel free to disable this requirement."
        )
    patchdicts = [
        {
            "module": xr.coding.variables,
            "target": "_scale_offset_decoding",
            "replacement": patched_xr_decode_scaling,
            "version": xr.__version__,
            "rules": [
                {"version": "2024.3", "comperator": operator.lt, "action": "skip"},
                {"version": "2025.3", "comperator": operator.ge, "action": "skip"},
            ],
        },
        {
            "module": xr.coding.times,
            "target": "decode_cf_timedelta",
            "replacement": patched_xr_decode_tDel,
            "version": xr.__version__,
            "rules": [
                {"version": "2025.3", "comperator": operator.lt, "action": "skip"},
                {"version": "2025.4", "comperator": operator.ge, "action": "skip"},
            ],
        },
    ]
    for i in range(2):
        try:
            with monkeypatch(patchdicts):
                ds = xr.open_dataset(l1b_filename, decode_timedelta=True)
            break
        except (OSError, ValueError) as err:
            if i == 1:
                print("Renewing the file didn't help or failed again.")
                raise err
            if isinstance(err, OSError):
                if not err.errno == -101:
                    raise err
                else:
                    warnings.warn(err.strerror + " was raised. Downloading file again.")
            else:
                warnings.warn(str(err) + " was raised. Downloading file again.")
            os.remove(l1b_filename)
            download_single_file(os.path.split(l1b_filename)[-1][19:34])
    # at least until baseline E ns_20_ku needs to be made a coordinate
    ds = ds.assign_coords(ns_20_ku=("ns_20_ku", np.arange(len(ds.ns_20_ku))))  # pyright: ignore[reportPossiblyUnboundVariable]
    # remove data that will not be used to reduce memory footprint
    for dim in ["time_plrm_01_ku", "time_plrm_20_ku", "nlooks_ku", "space_3d"]:
        if dim in ds.dims:
            ds = ds.drop_dims(dim)
    # first: get azimuth bearing from smoothed incremental azimuths.
    # this needs to be done before dropping part of the recording
    poly3fit_params = np.polyfit(
        np.arange(len(ds.time_20_ku) - 1),
        WGS84_ellpsoid.inv(
            lats1=ds.lat_20_ku[:-1],
            lons1=ds.lon_20_ku[:-1],
            lats2=ds.lat_20_ku[1:],
            lons2=ds.lon_20_ku[1:],
        )[0],
        3,
    )
    ds = ds.assign(
        azimuth=(
            "time_20_ku",
            np.poly1d(poly3fit_params)(np.arange(len(ds.time_20_ku) - 0.5)) % 360,
        )
    )
    ds["power_waveform_20_ku"] = (
        ds.pwr_waveform_20_ku * ds.echo_scale_factor_20_ku * 2**ds.echo_scale_pwr_20_ku
    )
    if not use_original_noise_estimates:
        # consider noise estimates over periods on the scale of
        # multiple tracking cycles to avoid loss-of-lock issues
        tracking_cycles = 5
        # the implemented algorithm uses a forward and a backward
        # rolling minimum. to work it needs at least twice the
        # window width (however, it is designed for much longer
        # tracks)
        if len(ds.time_20_ku) > 2 * (tracking_cycles * 20):
            noise = xr.apply_ufunc(
                noise_val,
                ds.power_waveform_20_ku.isel(ns_20_ku=slice(int(len(ds.ns_20_ku) / 4))),
                input_core_dims=[["ns_20_ku"]],
                output_core_dims=[[]],
                vectorize=True,
            )

            def noise_floor(noise):
                # construct a lower envelope of the noise values
                window_size = 5 * 20  # on the scale of the tracking loop (1 Hz)
                fwd = noise.rolling(time_20_ku=window_size).min()
                bwd = (
                    noise.isel(time_20_ku=slice(None, None, -1))
                    .rolling(time_20_ku=window_size)
                    .min()
                    .isel(time_20_ku=slice(None, None, -1))
                )
                # the upper envelope of the two lower envelope builds
                # the collective lower envelope
                upper_envelope = xr.concat([fwd, bwd], "ds").max("ds")
                return upper_envelope.fillna(upper_envelope.max())

            ds["noise_power_20_ku"] = noise_floor(noise)
    else:
        ds["noise_power_20_ku"] = ds.transmit_pwr_20_ku * 10 ** (
            ds.noise_power_20_ku / 10
        )
    # waveform selection is meant to be versatile. however the handling seems fragile
    if waveform_selection is not None:
        if (
            not isinstance(waveform_selection, slice)
            and not isinstance(waveform_selection, list)
            and not isinstance(waveform_selection, pd.Index)
        ):
            waveform_selection = [waveform_selection]
        if (
            isinstance(waveform_selection, slice)
            and isinstance(waveform_selection.start, numbers.Integral)
        ) or isinstance(waveform_selection[0], numbers.Integral):
            ds = ds.isel(time_20_ku=waveform_selection)
        else:
            # for compatibility with lower precision timestamps, use backfill or
            # nearest. I prefer nearest because it should also work in cases where
            # the timestamp was rounded instead of floored. for cryosat it should be
            # safe to allow a mismatch of up to +-25 milliseconds (20 Hz).
            ds = ds.sel(
                time_20_ku=waveform_selection,
                method="nearest",
                tolerance=np.timedelta64(25, "ms"),
            )
    if mask_coherence_gt1:
        ds["coherence_waveform_20_ku"] = ds.coherence_waveform_20_ku.where(
            ds.coherence_waveform_20_ku <= 1
        )
    if drop_waveforms_by_flag:
        # see available flags using data.flag.attrs["flag_meanings"]
        # print("drop bad. cur buf:", buffer)
        for flag_var, flag_val_list in drop_waveforms_by_flag.items():
            ds = drop_waveform(ds, build_flag_mask(ds[flag_var], flag_val_list))
    if drop_outside is not None and drop_outside is not False:
        # ! needs to be tidied up:
        # (also: simplify needed?)
        planar_crs = find_planar_crs(lon=ds.lon_20_ku, lat=ds.lat_20_ku)
        ground_track_points_4326 = gpd.GeoSeries(
            gpd.points_from_xy(ds.lon_20_ku, ds.lat_20_ku), crs=4326
        )
        try:
            if isinstance(drop_outside, (int, float)):
                o2regions = gpd.read_feather(
                    os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather")
                )
                intersected_o2 = o2regions.geometry.intersects(
                    ground_track_points_4326.union_all(method="unary")
                )
                if sum(intersected_o2) == 0:
                    raise IndexError
                else:
                    o2codes = o2regions.loc[intersected_o2, "o2region"].values
                o2region_complexes = []
                for o2 in np.unique(o2codes):
                    if o2 != "05-01":  # Greenland periphery is too large
                        o2region_complexes.append(
                            load_glacier_outlines(o2, union=False)
                        )
                    else:  # cut into 10 subregions, append if crossed
                        # !tbi: instead of using the arbitrary chunks, use the custom
                        # subregions 05-11--05-15 (added in commit 2265523)
                        for grnlnd_part in subdivide_region(
                            load_glacier_outlines("05-01"),
                            lat_bin_width_degree=4.5,
                            lon_bin_width_degree=4.5,
                        ):
                            if buffer_4326_shp(
                                grnlnd_part.union_all(method="coverage").envelope,
                                drop_outside,
                            ).intersects(
                                ground_track_points_4326.union_all(method="unary")
                            ):
                                o2region_complexes.append(grnlnd_part)
                # below, using geopandas as shapely wrapper for readability
                buffered_complexes = (
                    gpd.GeoSeries(
                        buffer_4326_shp(
                            pd.concat(o2region_complexes).union_all(method="coverage"),
                            drop_outside,
                        ),
                        crs=4326,
                    )
                    .to_crs(planar_crs)
                    .clip_by_rect(
                        *ground_track_points_4326.to_crs(planar_crs).total_bounds
                    )
                    .to_crs(4326)
                    .make_valid()
                    .iloc[0]
                )
            else:
                buffered_complexes = drop_outside
            retain_indeces = ground_track_points_4326.intersects(buffered_complexes)
            ds = ds.isel(time_20_ku=retain_indeces[retain_indeces].index)
        except IndexError:
            warnings.warn(
                "No waveforms left on glacier. Proceeding with empty dataset."
            )
            ds = ds.isel(time_20_ku=[])
    ds = ds.assign_attrs(
        coherence_threshold=coherence_threshold,
        power_threshold=power_threshold,
        smooth_phase_difference=smooth_phase_difference,
    )
    # add potential phase wrap factor for later use
    ds = ds.assign_coords({"phase_wrap_factor": np.arange(-3, 4)})
    # create a working version phase difference
    ds["ph_diff"] = ds.ph_diff_waveform_20_ku
    if len(ds.time_20_ku) > 0:
        # find and store POCAs and swath-starts
        ds = append_poca_and_swath_idxs(ds, **swath_start_kwargs)
        ds = append_smoothed_complex_phase(ds)
        if smooth_phase_difference:
            ds["ph_diff"] = ds.ph_diff.where(
                ds.ph_diff_complex_smoothed.isnull(),
                xr.apply_ufunc(np.angle, ds.ph_diff_complex_smoothed)
                if not isinstance(smooth_phase_difference, dict)
                else xr.apply_ufunc(
                    np.angle,
                    ds.pipe(
                        append_smoothed_complex_phase, **smooth_phase_difference
                    ).ph_diff_complex_smoothed,
                ),
            )
        else:
            # always use lowpass-filtered phase difference at POCA
            ds["ph_diff"] = ds.ph_diff.where(
                ds.ns_20_ku != ds.poca_idx,
                xr.apply_ufunc(np.angle, ds.ph_diff_complex_smoothed),
            )
    return ds


@if_not_empty
def append_ambiguous_reference_elevation(ds, dem_file_name_or_path: str | None = None):
    """Sample DEM elevation for each ambiguous phase-wrapping solution."""
    # !! This function causes much of the computation time. I suspect that
    # sparse memory accessing can be minimized with some tricks. However,
    # first tries sorting the spatial data, took even (much) longer.
    if "xph_lats" not in ds.data_vars:
        ds = locate_ambiguous_origin(ds)
    # ! tbi: auto download ref dem if not present
    with get_dem_reader(
        (ds if dem_file_name_or_path is None else dem_file_name_or_path)
    ) as dem_reader:
        if isinstance(dem_reader, xr.DataArray):
            crs = dem_reader.rio.crs
        else:
            crs = dem_reader.crs
            dem_reader = rioxr.open_rasterio(dem_reader)
        trans_4326_to_dem_crs = Transformer.from_crs("EPSG:4326", crs)
        x, y = trans_4326_to_dem_crs.transform(ds.xph_lats, ds.xph_lons)
        ds = ds.assign(
            xph_x=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), x),
            xph_y=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), y),
        )
        ds.attrs.update({"CRS": ensure_pyproj_crs(crs)})
        # ! huge improvement potential: instead of the below,
        # rasterio.sample could be used
        # [edit] use postgis
        try:
            ref_dem = dem_reader.rio.clip_box(
                np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)
            ).squeeze()
        except rioxr.exceptions.NoDataInBounds:
            warnings.warn(
                f"couldn't find ref dem data in box: {np.nanmin(x)}, {np.nanmin(y)}, "
                f"{np.nanmax(x)}, {np.nanmax(y)}\nouter lat lon coords: "
                f"{ds.lat_20_ku.values[[0, -1]]}, {ds.lon_20_ku.values[[0, -1]]}"
            )
            raise
        ds["xph_ref_elevs"] = ref_dem.sel(x=ds.xph_x, y=ds.xph_y, method="nearest")
    # rasterio suggests sorting like
    #   `for ind in np.lexsort([y, x]): rv.append((x[ind], y[ind]))`
    # sort_key = np.lexsort([y, x])
    # planar_coords = zip(x[sort_key], y[sort_key])
    # ref_elev_vector = np.fromiter(dem_reader.sample(planar_coords),
    #                               "float32")[sort_key.argsort()]
    # return ds.assign(xph_ref_elevs=(ds.xph_lats.dims,
    #                                   np.reshape(ref_elev_vector,
    #                                              ds.xph_lats.shape)))
    return ds


@if_not_empty
def append_best_fit_phase_index(ds, best_column: callable = None) -> xr.Dataset:
    """Resolve phase difference ambiguity

    The phase difference is ambiguous and only know except for a multiple
    of 2 pi. This method finds the best fitting factor of 2 pi wrt. a
    digital elevation model (DEM). By default, the summed distance to the
    DEM per group is minimized.

    Args:
        best_column (callable, optional): Function that takes a k*n matrix of
            difference to the DEM as first argument, where k are the number
            of group members (waveform samples) and n the number of possible
            wrapping factors. The function needs to return the chosen index
            along the second axis. Visit the source code to get a template
            for an excepted function. Defaults to None.

    Returns:
        L1bData
    """
    # ! Implement opt-out or/and grouping alternatives
    # before locating echos, find groups because also phase is unwrapped
    if "group_id" not in ds.data_vars:
        ds = tag_groups(ds)
        # it makes sense to always unwrap the phases immediately after finding
        # the groups. assigning the best fitting indices otherwise messes up
        # your data
        ds = unwrap_phase_diff(ds)
    if "xph_elev_diffs" not in ds.data_vars:
        ds = append_elev_diff_to_ref(ds)
    ds = ds.assign(
        ph_idx=(
            ("time_20_ku", "ns_20_ku"),
            np.empty((len(ds.time_20_ku), len(ds.ns_20_ku)), dtype="int"),
        )
    )
    if best_column is None:

        def best_column(elev_diff):
            return np.argmin(
                np.abs(np.median(elev_diff, axis=0)) ** 2
                + median_abs_deviation(elev_diff, axis=0) ** 2
            )

    def find_group_ph_idx(elev_diff, group_ids):
        out = np.zeros_like(group_ids)
        for i in nan_unique(group_ids):
            mask = group_ids == i
            out[mask] = best_column(elev_diff[mask, :]) - len(ds.phase_wrap_factor) // 2
        return out

    ds["ph_idx"] = xr.apply_ufunc(
        find_group_ph_idx,
        ds.xph_elev_diffs,
        ds.group_id,
        input_core_dims=[["ns_20_ku", "phase_wrap_factor"], ["ns_20_ku"]],
        output_core_dims=[["ns_20_ku"]],
    )
    ds["ph_idx"] = xr.where(
        ds.group_id.isnull(),
        np.abs(ds.xph_elev_diffs).idxmin("phase_wrap_factor"),
        ds.ph_idx,
    )
    return ds


@if_not_empty
def append_elev_diff_to_ref(ds):
    """Append elevation differences between candidates and reference DEM."""
    if "xph_ref_elevs" not in ds.data_vars:
        ds = append_ambiguous_reference_elevation(ds)
    ds["xph_elev_diffs"] = ds.xph_elevs - ds.xph_ref_elevs
    return ds


def from_id(track_id: str | pd.Timestamp, **kwargs) -> xr.Dataset:
    """Load and preprocess a single track by CryoSat time ID."""
    track_id = pd.Timestamp(track_id)
    # edge cases with exactly 0 nanoseconds may fail. however, since this is
    # only relevant for detail inspection, edge cases are ignored
    if track_id.nanosecond != 0:
        if kwargs is None:
            kwargs = {}
        if "waveform_selection" not in kwargs:
            kwargs["waveform_selection"] = track_id
        # file name list as look up table
        full_file_names = load_cs_full_file_names(update="no")
        idx_loc = full_file_names.index.get_indexer([track_id], method="pad")[0]
        track_id = full_file_names.index[idx_loc]
    l1b_data_dir = os.path.join(l1b_path, track_id.strftime(f"%Y{os.path.sep}%m"))
    track_id = cs_time_to_id(track_id)
    if os.path.isdir(l1b_data_dir):
        for file_name in os.listdir(l1b_data_dir):
            if (
                fnmatch.fnmatch(file_name, "*CS_????_SIR_SIN_1B_*")
                and os.path.split(file_name)[-1][19:34] == track_id
                and file_name.endswith(".nc")
            ):
                return read_esa_l1b(os.path.join(l1b_data_dir, file_name), **kwargs)
    return read_esa_l1b(download_single_file(track_id), **kwargs)


def get_rgi_o2(ds) -> str:
    """Finds RGIv7 o2 region that contains the track's central lat,
    lon.

    Returns:
        str: RGI v7 `long_code`
    """
    if len(ds.time_20_ku) == 0:
        return "no region; empty track"
    rgi_o2_gpdf = gpd.read_feather(
        os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather")
    )
    return rgi_o2_gpdf[
        rgi_o2_gpdf.contains(
            gpd.points_from_xy(ds.lon_20_ku, ds.lat_20_ku, crs=4326)
            .unary_all(method="coverage")
            .centroid
        )
    ].long_code.values[0]


@if_not_empty
def get_phase_jump(ds):
    """Detect phase jumps along waveform samples."""
    ph_diff_diff = ds.ph_diff_complex_smoothed.diff("ns_20_ku")
    # ! implement choosing tolerance
    ph_diff_diff_tolerance = 0.1
    # ! implement find loc. max. + cmp. to threshold (prevents multiple jump bins)
    jump_mask = np.logical_or(
        np.abs(ph_diff_diff) > ph_diff_diff_tolerance,
        np.abs(ph_diff_diff).rolling(ns_20_ku=2).sum()
        > 2 * 0.8 * ph_diff_diff_tolerance,
    )
    if "exclude_mask" not in ds.data_vars:
        ds = append_exclude_mask(ds)
    return xr.where(ds.exclude_mask.sel(ns_20_ku=jump_mask.ns_20_ku), False, jump_mask)


@if_not_empty
def get_phase_outlier(ds, tol: float | None = None):
    """Flag phase samples that deviate from the smoothed complex phase."""
    # inputs have to be complex unit vectors
    # if no tol provided calc equivalent of 300 m at nadir
    if tol is None:
        temp_x_width = 300  # [m] allow ph_diff to jump by this value (roughly)
        temp_H = 720e3  # [m] rough altitude of CS2
        # 0s below: set to an arbitrary off nadir angle at which the x_width should
        # actually have the defined value
        tol = (
            (np.arctan(np.tan(np.deg2rad(0)) + temp_x_width / temp_H) - np.deg2rad(0))
            * 2
            * np.pi
            / np.tan(speed_of_light / Ku_band_freq / antenna_baseline)
        )
    # ph_diff_tol is small, so approx equal to secant length
    return (
        np.abs(np.exp(1j * ds.ph_diff_waveform_20_ku) - ds.ph_diff_complex_smoothed)
        > tol
    )


@if_not_empty
# ! rename to something like retrieve_ambiguous_origins
def locate_ambiguous_origin(ds):
    """Calculates all "possible" echo origins.

    Adds for the 7 look angles `xph_thetas` the variables xph_lats,
    xph_lons, xph_elevs, and xph_dists.

    Returns:
        Dataset: l1b_data including the calculated coordinates.
    """
    # Calculate normal distance: position on ellipsoid surface <--> major axis
    r_N = WGS84_ellpsoid.a / np.sqrt(
        1 - WGS84_ellpsoid.es * np.sin(np.deg2rad(ds.lat_20_ku)) ** 2
    )
    # Add satellite height
    r_cs2 = r_N + ds.alt_20_ku
    # Calculate distance: satellite <--> echo origin
    range_to_scat = ref_range(ds) + (ds.ns_20_ku - 512) * sample_width
    theta = np.arcsin(
        -(ds.ph_diff + ds.phase_wrap_factor * 2 * np.pi)
        * (speed_of_light / Ku_band_freq)
        / (2 * np.pi * antenna_baseline)
    ) - np.deg2rad(ds.off_nadir_roll_angle_str_20_ku)
    # Calculate distance: echo origin <--> major axis (from scalar product)
    r_x = np.sqrt(
        range_to_scat**2 + r_cs2**2 - (2 * range_to_scat * r_cs2 * np.cos(theta))
    )
    dist_off_groundtrack = r_N * np.arctan(
        range_to_scat * np.sin(theta) / (r_cs2 - range_to_scat * np.cos(theta))
    )
    lons, lats = WGS84_ellpsoid.fwd(
        lons=ds.lon_20_ku.expand_dims(
            {
                "ns_20_ku": ds.ns_20_ku.size,
                "phase_wrap_factor": ds.phase_wrap_factor.size,
            },
            [-2, -1],
        ),
        lats=ds.lat_20_ku.expand_dims(
            {
                "ns_20_ku": ds.ns_20_ku.size,
                "phase_wrap_factor": ds.phase_wrap_factor.size,
            },
            [-2, -1],
        ),
        az=ds.azimuth.expand_dims(
            {
                "ns_20_ku": ds.ns_20_ku.size,
                "phase_wrap_factor": ds.phase_wrap_factor.size,
            },
            [-2, -1],
        )
        + 90,
        dist=dist_off_groundtrack,
    )[:2]
    return ds.assign(
        xph_lons=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), lons),
        xph_lats=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), lats),
        # Assuming the local ellipsoid radius changes slowly:
        xph_elevs=(
            ("time_20_ku", "ns_20_ku", "phase_wrap_factor"),
            (r_x - r_N).transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values,
        ),
        xph_thetas=(
            ("time_20_ku", "ns_20_ku", "phase_wrap_factor"),
            theta.transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values,
        ),
        xph_dists=(
            ("time_20_ku", "ns_20_ku", "phase_wrap_factor"),
            dist_off_groundtrack.transpose(
                "time_20_ku", "ns_20_ku", "phase_wrap_factor"
            ).values,
        ),
    )


def ref_range(ds) -> xr.DataArray:
    """Calculate distance to center of range window.

    Returns:
        xr.DataArray: Reference ranges.
    """
    # make property?
    corrections = (
        ds.mod_dry_tropo_cor_01
        + ds.mod_wet_tropo_cor_01
        + ds.iono_cor_gim_01
        + ds.pole_tide_01
        + ds.solid_earth_tide_01
        + ds.load_tide_01
    )
    return ds.window_del_20_ku / np.timedelta64(
        1, "s"
    ) / 2 * speed_of_light + np.interp(ds.time_20_ku, ds.time_cor_01, corrections)


@if_not_empty
def tag_groups(ds) -> xr.Dataset:
    """Identifies and tags wafeform sample groups.

    Returns:
        xr.Dataset: l1b_ds.
    """
    phase_outlier = get_phase_outlier(ds)
    ignore_mask = (ds.exclude_mask + phase_outlier) != 0
    gap_separator = ignore_mask.rolling(ns_20_ku=3).sum() == 3
    any_separator = np.logical_or(
        *xr.align(get_phase_jump(ds), gap_separator, join="outer")
    )
    rising_edge_per_waveform_counter = (
        any_separator.astype("int32").diff("ns_20_ku") == -1
    ).cumsum("ns_20_ku") + 1
    group_tags = rising_edge_per_waveform_counter + xr.DataArray(
        data=np.arange(len(ds.time_20_ku)) * len(ds.ns_20_ku), dims="time_20_ku"
    )
    group_tags = xr.align(group_tags, ds.power_waveform_20_ku, join="right")[0].where(
        ~ignore_mask
    )

    def filter_small_groups(group_ids):
        out = group_ids
        for i in nan_unique(group_ids):
            mask = group_ids == i
            if mask.sum() < 3:
                out[mask] = 0
        return out

    group_tags = xr.apply_ufunc(
        filter_small_groups,
        group_tags,
        input_core_dims=[["ns_20_ku"]],
        output_core_dims=[["ns_20_ku"]],
    )
    group_tags = group_tags.where(group_tags != 0)
    ds["group_id"] = group_tags
    return ds


def to_l2(
    ds,
    out_vars: list | dict = None,
    *,
    retain_vars: list | dict = None,
    swath_or_poca: str = "swath",
    group_best_column_func: callable = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Converts l1b data to l2 data (point elevations).

    Args:
        out_vars (list | dict, optional): Return values. If none provided,
            returns time, x, y, height, reference elevation, and difference
            wrt. reference. Provide a dictionary to assign custom names.
            Defaults to None.
        retain_vars (list | dict, optional): Additional to `out_vars`.
            Defaults to None.
        swath_or_poca (str, optional): Either "swath", "poca", or "both".
            Decides what data is returned. Defaults to "swath".
        group_best_column_func (callable, optional): Optimization function to
            resolve phase difference ambiguity. View
            :func:`append_best_fit_phase_index` for details.

    Raises:
        ValueError: If `swath_or_poca` cannot be interpreted.

    Returns:
        gpd.GeoDataFrame: Elevation estimates and requested variables. If
        `swath_or_poca` is "both", a tuple with separate tables is
        returned.
    """
    if len(ds.time_20_ku) == 0:
        if swath_or_poca == "both":
            return empty_GeoDataFrame, empty_GeoDataFrame
        else:
            return empty_GeoDataFrame
    if out_vars is None:
        out_vars = dict(
            time_20_ku="time",
            xph_x="x",
            xph_y="y",
            xph_elevs="height",
            xph_ref_elevs="h_ref",
            xph_elev_diffs="h_diff",
        )
    # implicitly test whether data was processed. if not, do so
    if "ph_idx" not in ds.data_vars:
        ds = append_best_fit_phase_index(ds, group_best_column_func)
    if isinstance(out_vars, dict):
        ds = ds.drop_vars(list(out_vars.values()), errors="ignore")
        ds = ds.rename_vars(out_vars)
        out_vars = list(out_vars.values())
    if isinstance(retain_vars, dict):
        ds = ds.drop_vars(list(retain_vars.values()), errors="ignore")
        ds = ds.rename_vars(retain_vars)
        retain_vars = list(retain_vars.values())
    elif retain_vars is None:
        retain_vars = []
    if swath_or_poca == "swath":
        tmp = (
            ds[out_vars + retain_vars]
            .where(~ds.exclude_mask)
            .sel(phase_wrap_factor=ds.ph_idx)
            .dropna("time_20_ku", how="all")
        )
    elif swath_or_poca == "poca":
        waveforms_with_poca = ds.time_20_ku[~ds.poca_idx.isnull()]
        if len(waveforms_with_poca) == 0:
            return empty_GeoDataFrame
        tmp = (
            ds[out_vars + retain_vars + ["ph_idx"]]
            .sel(time_20_ku=waveforms_with_poca)
            .sel(ns_20_ku=ds.poca_idx[~ds.poca_idx.isnull()])
        )
        tmp = (
            tmp[out_vars + retain_vars]
            .sel(phase_wrap_factor=tmp.ph_idx)
            .dropna("time_20_ku", how="all")
        )
    elif swath_or_poca == "both":
        swath = to_l2(
            ds, out_vars, retain_vars=retain_vars, swath_or_poca="swath", **kwargs
        )
        poca = to_l2(
            ds, out_vars, retain_vars=retain_vars, swath_or_poca="poca", **kwargs
        )
        return swath, poca
    else:
        raise ValueError(
            f'You provided "swath_or_poca={swath_or_poca}". Choose "swath", "poca",',
            'or "both".',
        )
    drop_coords = [coord for coord in tmp.coords if coord not in ["time", "sample"]]

    # ! dropped .squeeze() below to handle issue #19. not sure about 2nd
    # degree consequences.
    # l2_data = l2.from_processed_l1b(tmp.squeeze().drop_vars(drop_coords), **kwargs)
    l2_data = l2_from_processed_l1b(tmp.drop_vars(drop_coords), **kwargs)
    return l2_data


@if_not_empty
def unwrap_phase_diff(ds) -> xr.Dataset:
    """Replaces phase difference by unwrapped version.

    Unwrapping is done per group of waveform samples.

    Returns:
        xr.Dataset: l1b_ds.
    """

    def unwrap(ph_diff, group_ids):
        out = ph_diff
        for i in nan_unique(group_ids):
            mask = group_ids == i
            out[mask] = np.unwrap(ph_diff[mask])
        return out

    ds["ph_diff"] = xr.apply_ufunc(
        unwrap,
        ds.ph_diff,
        ds.group_id,
        input_core_dims=[["ns_20_ku"], ["ns_20_ku"]],
        output_core_dims=[["ns_20_ku"]],
    )
    return ds


# helper functions ####################################################


def append_exclude_mask(cs_l1b_ds: xr.Dataset) -> xr.Dataset:
    """Adds mask indicating samples below threshold.

    Waveform samples that don't fulfill power and/or coherence requirements
    are flagged. The thresholds have to be included in the provided
    dataset. By default, they are assigned on creation.

    Args:
        cs_l1b_ds (l1b_data): Input data.

    Returns:
        l1b_data: Data including mask.
    """
    # for now require tuple. could be some auto recognition in future.
    if not isinstance(cs_l1b_ds.power_threshold, tuple):
        raise TypeError("power_threshold must be a tuple.")
    # only signal-to-noise-ratio implemented
    if cs_l1b_ds.power_threshold[0] != "snr":
        raise NotImplementedError(
            'Only power_threshold mode "snr" is currently implemented.'
        )
    power_threshold = cs_l1b_ds.noise_power_20_ku * cs_l1b_ds.power_threshold[1]
    cs_l1b_ds["exclude_mask"] = np.logical_or(
        cs_l1b_ds.power_waveform_20_ku < power_threshold,
        cs_l1b_ds.coherence_waveform_20_ku < cs_l1b_ds.coherence_threshold,
    )
    return cs_l1b_ds


def append_poca_and_swath_idxs(
    cs_l1b_ds: xr.Dataset,
    poca_upper: float = 10,
    swath_start_window: tuple[float, float] = (5, 50),
) -> xr.Dataset:
    """Adds indices for estimated POCA and begin of swath.

    Args:
        cs_l1b_ds (l1b_data): Input data.
        poca_upper (float): Maximum distance in meter of POCA sample to
            first sample above coherence threshold. Defaults to 10.
        swath_start_window (tuple[float, float]): Tuple of minimum and maximum
            distance in meter of swath start to POCA sample. If the
            maximum is negative, swath start is set to 0, implying all
            samples will be swath processed. Defaults to (5, 50).

    Returns:
        l1b_data: Data including mask.
    """
    if len(cs_l1b_ds.time_20_ku) == 0:
        return cs_l1b_ds.assign(
            swath_start=(("time_20_ku"), []),
            poca_idx=(("time_20_ku"), []),
            exclude_mask=(
                ("time_20_ku", "ns_20_ku"),
                np.empty_like(cs_l1b_ds.power_waveform_20_ku),
            ),
        )

    # ! performance improvement potential
    # should be possible to accelerate with numba
    def find_poca_idx_and_swath_start_idx(smooth_coh, coh_thr):
        # if smooth coherence exceeds threshold in the first 10 m, its
        # unreasonable to assume that the tracking loop did not fail
        # (the POCA may have been before the waveform even starts, but
        # we can't tell).
        poca_idx = np.argmax(smooth_coh > coh_thr)
        if poca_idx < int(10 / sample_width):
            # I opted for nan if no poca for transparency. this requires
            # dtype float and is slower
            return np.nan, 0
        # poca expected `poca_upper` m after coherence exceeds threshold
        # (no solid basis)
        poca_idx = (
            np.argmax(
                smooth_coh[poca_idx : poca_idx + max(1, int(poca_upper / sample_width))]
            )
            + poca_idx
        )
        if swath_start_window[1] < 0:
            swath_start = 0
        else:
            try:
                swath_start = poca_idx + int(swath_start_window[0] / sample_width)
                diff_smooth_coh = np.diff(
                    smooth_coh[
                        swath_start : swath_start
                        + int(swath_start_window[1] / sample_width)
                    ]
                )
                # swath can safest be used after the coherence dip
                swath_start = (
                    np.argmax(
                        diff_smooth_coh[np.argmax(np.abs(diff_smooth_coh) > 0.001) :]
                        > 0
                    )
                    + swath_start
                )
            # if swath doesn't start in range window, just indeed set the
            # index behind last element
            except ValueError:
                swath_start = len(smooth_coh)
        return float(poca_idx), swath_start

    cs_l1b_ds[["poca_idx", "swath_start"]] = xr.apply_ufunc(
        find_poca_idx_and_swath_start_idx,
        gauss_filter_DataArray(cs_l1b_ds.coherence_waveform_20_ku, "ns_20_ku", 35, 35),
        kwargs=dict(coh_thr=cs_l1b_ds.coherence_threshold),
        input_core_dims=[["ns_20_ku"]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    if "exclude_mask" not in cs_l1b_ds.data_vars:
        cs_l1b_ds = append_exclude_mask(cs_l1b_ds)
    cs_l1b_ds["exclude_mask"] = cs_l1b_ds.exclude_mask.where(
        cs_l1b_ds.ns_20_ku >= cs_l1b_ds.swath_start, True
    )
    return cs_l1b_ds


def append_smoothed_complex_phase(
    cs_l1b_ds: xr.Dataset, window_extent: int = 21, std: float = 5
) -> xr.Dataset:
    """Append low-pass filtered complex phase representation."""
    cs_l1b_ds["ph_diff_complex_smoothed"] = gauss_filter_DataArray(
        np.exp(1j * cs_l1b_ds.ph_diff_waveform_20_ku),
        dim="ns_20_ku",
        window_extent=window_extent,
        std=std,
    )
    return cs_l1b_ds


def build_flag_mask(cs_l1b_flag: xr.DataArray, flag_val_list: list) -> xr.DataArray:
    """Function returns a waveform mask based on flag values.

    This function can handle two types of flags: those that take the form
    of a checklist with multiple allowed ticks, and those that indicate
    one of more possible selections.

    It is designed for CryoSat-2 SARIn L1b Baseline D or E data and
    relies on an attribute "flag_masks" or "flag_values". For CRISTAL or
    if the attributes change, this function needs an update.

    Args:
        cs_l1b_flag (xr.DataArray): L1bData flag variable.
        flag_val_list (list, optional): List of flag values to mask.

    Returns:
        xr.DataArray: Mask that is True where flag matched provided list.
    """
    if "flag_masks" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(
            data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
            index=np.log2(
                np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64"))
            ).astype("int"),
        ).sort_index()

        def flag_func(int_code: int) -> bool:
            for i, b in enumerate(reversed(bin(int_code)[2:])):
                if b == "0":
                    continue
                try:
                    if flag_dictionary.loc[i] in flag_val_list:
                        return True
                except KeyError:
                    print(
                        "Flag not found in attributes! Pointing to a bug or an issue "
                        "in the data."
                    )
                    raise
            return False

    elif "flag_values" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(
            data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
            index=cs_l1b_flag.attrs["flag_values"],
        )

        def flag_func(int_code: int):
            return flag_dictionary.loc[int_code] in flag_val_list

    else:
        raise NotImplementedError
    return xr.apply_ufunc(
        np.vectorize(flag_func), cs_l1b_flag.astype(int), dask="allowed"
    )


def _existing_local_l1b_track_ids(year_month_str: str) -> set[str]:
    """Return track IDs encoded in local L1b filenames for one month."""
    try:
        return {
            filename[19:34]
            for filename in os.listdir(os.path.join(l1b_path, year_month_str))
        }
    except FileNotFoundError:
        return set()


def _missing_local_l1b_tracks(track_idx: pd.DatetimeIndex | str) -> pd.DatetimeIndex:
    """Return tracks absent under the downloader's current filename-ID rule."""
    track_idx = pd.DatetimeIndex(track_idx).sort_values()
    year_month = track_idx.strftime(f"%Y{os.path.sep}%m")
    missing_tracks = []
    for year_month_str in year_month.unique():
        month_tracks = track_idx[year_month == year_month_str]
        existing_track_ids = _existing_local_l1b_track_ids(year_month_str)
        missing_tracks.extend(
            track
            for track in month_tracks
            if track.strftime("%Y%m%dT%H%M%S") not in existing_track_ids
        )
    return pd.DatetimeIndex(missing_tracks)


# ! name is not intuitive
def download_wrapper(
    region_of_interest: str | shapely.Polygon = None,
    start_datetime: str | pd.Timestamp = "2010",
    end_datetime: str | pd.Timestamp = "2035",
    *,
    buffer_region_by: float = None,
    track_idx: pd.DatetimeIndex | str = None,
    stop_event: Event = None,
    n_threads: int = 8,
    # baseline: str = "latest",
) -> int:
    """Download ESA's L1b product.

    Args:
        region_of_interest (str | shapely.Polygon, optional): Provide a RGI
            identifier or lon/lat polygon to subset downloaded data.
            Defaults to None.
        start_datetime (str | pd.Timestamp, optional): Defaults to "2010".
        end_datetime (str | pd.Timestamp, optional): Defaults to "2035".
        buffer_region_by (float, optional): Use a buffer in meter around
            provided region (also RGI identifier). Defaults to None.
        track_idx (pd.DatetimeIndex | str, optional): Download only tracks
            at known times. Defaults to None.
        stop_event (Event, optional): Define when to terminate threads.
            Defaults to None.
        n_threads (int, optional): Number of download threads. Defaults to 8.

    Returns:
        int: 0 on success, 1 on graceful exit after error, and 2 on being
        aborted.
    """
    if track_idx is None:
        start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
        track_idx = load_cs_ground_tracks(
            region_of_interest,
            start_datetime,
            end_datetime,
            buffer_region_by=buffer_region_by,
        ).index
    else:
        track_idx = track_idx.sort_values()
    if stop_event is None:
        stop_event = Event()
    missing_track_idx = _missing_local_l1b_tracks(track_idx)
    if missing_track_idx.empty:
        _status("All selected L1b files are already present.")
        return 0
    try:
        user, password, _ = _resolve_esa_ftp_credentials()
        https_auth = (user, password)
    except RuntimeError as err:
        warnings.warn(
            "Could not configure PDS HTTPS credentials: "
            f"{err}. No download workers were started.",
            category=UserWarning,
            stacklevel=2,
        )
        _status("PDS HTTPS credentials unavailable; no download workers started.")
        return 1

    task_queue = request_workers(_download_files_with_auth, n_threads)
    months = missing_track_idx.to_period("M")
    for month in months.unique():
        idx_selection = missing_track_idx[months == month]
        task_queue.put((idx_selection, stop_event, https_auth))
    for _ in range(n_threads):
        task_queue.put(None)
    # wait for threads to finish
    try:
        task_queue.join()
    except Exception:
        stop_event.set()
        with task_queue.mutex:
            task_queue.queue.clear()
        _status("Aborting download because an error occurred (possibly an interrupt).")
        for i in range(3):
            time.sleep(10)
            if task_queue.empty():
                _status("Closed download threads. Some files may still be missing.")
                return 1
        _status(
            "Forcing download thread shutdown. Partially written NetCDF "
            "files may exist."
        )
        return 2
    else:
        worker_errors = getattr(task_queue, "worker_errors", [])
        if worker_errors:
            stop_event.set()
            first_error, first_traceback = worker_errors[0]
            warnings.warn(
                "One or more L1B download workers failed. Some files may still "
                f"be missing. First error: {first_error!r}",
                category=UserWarning,
            )
            _status(
                f"{len(worker_errors)} download task(s) failed. Some files may "
                "still be missing."
            )
            _status("First download worker traceback follows.")
            print(first_traceback)
            return 1
        _status("All downloads finished.")
        return 0


def _https_l1b_base_url(track_id: pd.Timestamp) -> str:
    """Return base HTTPS URL for one month of CryoSat L1b files."""
    return (
        r"https://science-pds.cryosat.esa.int/?do=download&file=Cry0Sat2_data"
        r"%2FSIR_SIN_L1%2F" + track_id.strftime("%Y%%2F%m") + "%2F"
    )


def _pds_l1b_download_url(remote_file: str) -> str:
    """Build the authenticated PDS HTTPS URL for a selected L1b filename."""
    filename = Path(remote_file).name
    if len(filename) < 34:
        raise ValueError(f"Could not extract a CryoSat timestamp from {remote_file!r}.")
    timestamp = pd.to_datetime(filename[19:34])
    return _https_l1b_base_url(timestamp) + filename


def _pds_download_error(failures: list[tuple[str, str]]) -> RuntimeError:
    """Return a concise error for PDS failures without automatic FTP fallback."""
    details = "; ".join(f"{track_id}: {reason}" for track_id, reason in failures)
    return RuntimeError(
        "CryoSat L1b download via PDS HTTPS failed; automatic FTP fallback is "
        f"disabled. Failed product(s): {details}. For future MAAP data delivery, "
        f"see enhancement #76."
    )


def _validate_netcdf_payload(path: str | Path) -> None:
    """Raise if downloaded payload does not look like NetCDF."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(512)
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return
    if header.startswith((b"CDF\x01", b"CDF\x02", b"CDF\x05")):
        return

    header_lstrip = header.lstrip().lower()
    if header_lstrip.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise RuntimeError(
            f"HTTPS endpoint returned HTML/XML instead of NetCDF for {path.name}."
        )
    raise RuntimeError(
        f"Downloaded payload for {path.name} is not recognized as NetCDF."
    )


def _l1b_product_name_candidates(remote_file: str) -> list[str]:
    """Return preferred remote filename candidates."""
    lta_candidate = remote_file.replace("OFFL", "LTA_")
    offl_candidate = remote_file.replace("LTA_", "OFFL")
    candidates = []
    for candidate in (lta_candidate, offl_candidate):
        if candidate not in candidates:
            candidates.append(candidate)
    if remote_file not in candidates:
        candidates.append(remote_file)
    return sorted(
        candidates,
        key=_cryosat_l1b_product_sort_key,
        reverse=True,
    )


def _select_lta_then_offl_for_track(track_id: str, remote_files: list[str]) -> str:
    """Select the preferred available product for ``track_id``."""
    matching_files = [
        name
        for name in remote_files
        if name.endswith(".nc") and len(name) >= 34 and name[19:34] == track_id
    ]
    if matching_files:
        return _preferred_cryosat_l1b_name(*matching_files)
    raise FileNotFoundError(f"No LTA_ or OFFL product found for track id {track_id}.")


def _esa_login_failed(response: requests.Response) -> bool:
    """Return whether ESA login flow reports an authentication failure."""
    urls = [response.url] + [item.url for item in response.history]
    locations = [item.headers.get("location", "") for item in response.history]
    return any(
        marker in value
        for marker in _ESA_LOGIN_FAILURE_MARKERS
        for value in urls + locations
    )


def _create_esa_https_session(
    auth: tuple[str, str],
    timeout: int | float = 120,
) -> requests.Session:
    """Create an authenticated ESA HTTPS session for CryoSat downloads."""
    user, password = auth
    session = requests.Session()
    try:
        login_response = session.get(_ESA_HTTPS_LOGIN_URL, timeout=timeout)
        session_data_key = parse_qs(urlparse(login_response.url).query).get(
            "sessionDataKey",
            [None],
        )[0]
        if session_data_key is None:
            raise RuntimeError(
                "ESA HTTPS login flow did not expose sessionDataKey for authentication."
            )
        auth_response = session.post(
            urljoin(login_response.url, "../commonauth"),
            data={
                "username": user,
                "password": password,
                "sessionDataKey": session_data_key,
            },
            timeout=timeout,
        )
        if _esa_login_failed(auth_response):
            raise RuntimeError("ESA HTTPS login failed.")
        return session
    except Exception:
        session.close()
        raise


def _download_https_url_atomic(
    session: requests.Session,
    url: str,
    local_path: str | Path,
    timeout: int | float = 120,
) -> str:
    """Download one HTTPS URL to a temporary file, then atomically move."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{local_path.name}.",
            suffix=".part",
            dir=local_path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
        os.replace(temp_path, local_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return str(local_path)


def _download_named_file_https(
    remote_file: str,
    local_path: str | Path,
    session: requests.Session,
    href: str | None = None,
) -> str:
    """Download one selected L1b filename from authenticated PDS HTTPS."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _ = href  # MAAP enclosure URLs remain catalog provenance, not delivery URLs.
    _status(f"Downloading {remote_file} via PDS https.")
    try:
        downloaded = _download_https_url_atomic(
            session=session,
            url=_pds_l1b_download_url(remote_file),
            local_path=local_path,
            timeout=120,
        )
        _validate_netcdf_payload(downloaded)
        return downloaded
    except Exception:
        if local_path.is_file():
            local_path.unlink()
        raise


def _download_remote_file_via_ftp_atomic(
    ftp: ftplib.FTP,
    remote_file: str,
    local_path: str | Path,
) -> str:
    """Download one file via FTP to a temporary file, then atomically move."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{local_path.name}.",
            suffix=".part",
            dir=local_path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            ftp.retrbinary("RETR " + remote_file, tmp_file.write)
        os.replace(temp_path, local_path)
        return str(local_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _load_cs_l1b_track_catalog_for(
    track_idx: pd.DatetimeIndex,
) -> gpd.GeoDataFrame | None:
    """Load and refresh the rich STAC catalog for requested tracks."""
    track_idx = pd.DatetimeIndex(track_idx).sort_values()
    try:
        catalog = load_cs_l1b_track_catalog(update="no")
    except Exception as err:
        warnings.warn(
            f"Could not load local STAC L1B catalog: {err}. "
            "Falling back to legacy filename lookup.",
            category=UserWarning,
        )
        return None
    if track_idx.empty:
        return catalog
    missing = track_idx.difference(catalog.index) if not catalog.empty else track_idx
    if len(missing) > 0:
        try:
            load_cs_ground_tracks(
                start_datetime=missing.min(),
                end_datetime=missing.max() + pd.Timedelta(seconds=1),
                source="stac",
            )
            catalog = load_cs_l1b_track_catalog(update="no")
        except Exception as err:
            warnings.warn(
                "Could not refresh STAC L1B catalog for missing tracks: "
                f"{err}. Falling back to legacy filename lookup.",
                category=UserWarning,
            )
    return catalog


def _load_cs_full_file_names_for(track_idx: pd.DatetimeIndex) -> pd.Series | None:
    """Load the local legacy file-name catalog without refreshing it over FTP."""
    try:
        return load_cs_full_file_names(update="no")
    except Exception as err:
        warnings.warn(
            f"Could not load local legacy file-name catalog: {err}.",
            category=UserWarning,
        )
        return None


def _download_files_via_ftp(
    track_idx: pd.DatetimeIndex | str,
    stop_event: Event = None,
) -> None:
    """Download all missing monthly L1b files via FTP."""
    year_month_str_list = track_idx.strftime(f"%Y{os.path.sep}%m").unique()
    for year_month_str in year_month_str_list:
        _status(f"Scanning {year_month_str} for missing files.")
        if stop_event is not None and stop_event.is_set():
            return
        try:
            currently_present_files = [
                x[19:] for x in os.listdir(os.path.join(l1b_path, year_month_str))
            ]
        except FileNotFoundError:
            os.makedirs(os.path.join(l1b_path, year_month_str))
            currently_present_files = []
        existing_track_ids = {name[:15] for name in currently_present_files}
        month_tracks = pd.DatetimeIndex(
            track_idx[track_idx.strftime(f"%Y{os.path.sep}%m") == year_month_str]
        )
        month_track_ids = month_tracks.strftime("%Y%m%dT%H%M%S")
        with ftp_cs2_server(timeout=120) as ftp:
            try:
                ftp.cwd("/SIR_SIN_L1/" + year_month_str)
            except ftplib.error_perm:
                warnings.warn(
                    "Directory /SIR_SIN_L1/" + year_month_str + " couldn't be accessed."
                )
                continue
            remote_listing = ftp.nlst()
            for track_id in month_track_ids:
                if stop_event is not None and stop_event.is_set():
                    return
                if track_id in existing_track_ids:
                    continue
                remote_file = _select_lta_then_offl_for_track(track_id, remote_listing)
                local_path = os.path.join(l1b_path, year_month_str, remote_file)
                try:
                    _status(f"Downloading {remote_file}.")
                    _download_remote_file_via_ftp_atomic(ftp, remote_file, local_path)
                except Exception:
                    _status(f"Download failed for {remote_file}.")
                    raise
                currently_present_files.append(remote_file[19:])
                existing_track_ids.add(track_id)


def _download_single_file_via_ftp(track_id: str) -> str:
    """Download one L1b file for a single CryoSat track ID via FTP."""
    retries = 10
    while retries > 0:
        try:
            with ftp_cs2_server() as ftp:
                ftp.cwd("/SIR_SIN_L1/" + pd.to_datetime(track_id).strftime("%Y/%m"))
                remote_file = _select_lta_then_offl_for_track(track_id, ftp.nlst())
                local_path = os.path.join(
                    l1b_path, pd.to_datetime(track_id).strftime("%Y/%m")
                )
                if not os.path.isdir(local_path):
                    os.makedirs(local_path)
                local_path = os.path.join(local_path, remote_file)
                try:
                    _status(f"Downloading {remote_file}.")
                    return _download_remote_file_via_ftp_atomic(
                        ftp, remote_file, local_path
                    )
                except Exception:
                    _status(f"Download failed for {remote_file}.")
                    raise
        except ftplib.error_temp as err:
            _status(
                f"{err} raised. Retrying track id {track_id} in 10 s "
                f"(attempt {11 - retries}/10)."
            )
            time.sleep(10)
            retries -= 1
    raise RuntimeError(f"FTP retries exhausted for track id {track_id}.")


def _download_files_with_auth(
    track_idx: pd.DatetimeIndex | str,
    stop_event: Event | None,
    https_auth: tuple[str, str],
) -> None:
    """Download a batch of missing L1b files with resolved PDS credentials."""
    track_idx = pd.DatetimeIndex(track_idx).sort_values()
    year_month_str_list = track_idx.strftime(f"%Y{os.path.sep}%m").unique()
    https_session = None
    track_catalog = _load_cs_l1b_track_catalog_for(track_idx)
    file_names = _load_cs_full_file_names_for(track_idx)
    if file_names is None and (track_catalog is None or track_catalog.empty):
        raise _pds_download_error(
            [
                (track.strftime("%Y%m%dT%H%M%S"), "no MAAP or local filename entry")
                for track in track_idx
            ]
        )
    try:
        https_session = _create_esa_https_session(https_auth)
    except Exception as err:
        raise RuntimeError(f"Could not initialize PDS HTTPS session: {err}") from err
    failures = []
    try:
        for year_month_str in year_month_str_list:
            _status(f"Scanning {year_month_str} for missing files.")
            if stop_event is not None and stop_event.is_set():
                return
            existing_track_ids = _existing_local_l1b_track_ids(year_month_str)
            month_tracks = track_idx[
                track_idx.strftime(f"%Y{os.path.sep}%m") == year_month_str
            ]
            for track_id in month_tracks:
                if stop_event is not None and stop_event.is_set():
                    return
                track_id_str = track_id.strftime("%Y%m%dT%H%M%S")
                if track_id_str in existing_track_ids:
                    continue
                catalog_row = None
                if track_catalog is not None and track_id in track_catalog.index:
                    catalog_row = track_catalog.loc[track_id]
                    if isinstance(catalog_row, pd.DataFrame):
                        catalog_row = catalog_row.iloc[-1]
                if catalog_row is not None:
                    remote_file = catalog_row["filename"]
                    href = catalog_row.get("href")
                elif file_names is not None and track_id in file_names.index:
                    remote_file = file_names.loc[track_id] + ".nc"
                    href = None
                else:
                    failures.append((track_id_str, "no MAAP or local filename entry"))
                    continue
                local_path = Path(l1b_path, year_month_str, remote_file)
                try:
                    _download_named_file_https(
                        remote_file=remote_file,
                        local_path=local_path,
                        session=https_session,
                        href=href,
                    )
                    existing_track_ids.add(track_id_str)
                except Exception as err:
                    failures.append((track_id_str, f"{remote_file}: {err}"))
        if failures:
            raise _pds_download_error(failures)
        _status(
            "Finished downloading tracks for months: "
            + ", ".join(str(x) for x in year_month_str_list)
        )
    finally:
        if https_session is not None:
            https_session.close()


def download_files(
    track_idx: pd.DatetimeIndex | str,
    stop_event: Event = None,
    # baseline: str = "latest",
):
    """Download all missing monthly L1b files for ``track_idx``."""
    try:
        user, password, _ = _resolve_esa_ftp_credentials()
    except RuntimeError as err:
        raise RuntimeError(f"Could not configure PDS HTTPS credentials: {err}") from err
    return _download_files_with_auth(track_idx, stop_event, (user, password))


def download_single_file(track_id: str) -> str:
    """Download one L1b file for a single CryoSat track ID."""
    track_id_timestamp = pd.to_datetime(track_id)
    track_id = track_id_timestamp.strftime("%Y%m%dT%H%M%S")
    try:
        user, password, _ = _resolve_esa_ftp_credentials()
        https_auth = (user, password)
    except RuntimeError as err:
        raise RuntimeError(f"Could not configure PDS HTTPS credentials: {err}") from err
    requested_idx = pd.DatetimeIndex([track_id_timestamp])
    track_catalog = _load_cs_l1b_track_catalog_for(requested_idx)
    catalog_row = None
    if track_catalog is not None and track_id_timestamp in track_catalog.index:
        catalog_row = track_catalog.loc[track_id_timestamp]
        if isinstance(catalog_row, pd.DataFrame):
            catalog_row = catalog_row.iloc[-1]
    file_names = _load_cs_full_file_names_for(requested_idx)
    if catalog_row is not None:
        filename = catalog_row["filename"]
        href = catalog_row.get("href")
    elif file_names is not None and track_id_timestamp in file_names.index:
        filename = file_names.loc[track_id_timestamp] + ".nc"
        href = None
    else:
        filename = None
    if filename is not None:
        local_path = Path(l1b_path, track_id_timestamp.strftime("%Y/%m"), filename)
        https_session = None
        try:
            https_session = _create_esa_https_session(https_auth)
            return _download_named_file_https(
                remote_file=filename,
                local_path=local_path,
                session=https_session,
                href=href,
            )
        except Exception as err:
            raise _pds_download_error([(track_id, f"{filename}: {err}")]) from err
        finally:
            if https_session is not None:
                https_session.close()
    raise _pds_download_error([(track_id, "no MAAP or local filename entry")])


def drop_waveform(cs_l1b_ds, time_20_ku_mask):
    """Use mask along time dim to drop waveforms.

    Args:
        time_20_ku_mask (1-dim bool): Mask: drop where True.

    Returns:
        xr.Dataset or DataArray: Input dataset without marked waveforms.
    """
    return cs_l1b_ds.sel(time_20_ku=cs_l1b_ds.time_20_ku[~time_20_ku_mask])


# left here for improvement ideas
# def choose_group_phase_wrap(waveform):
#     # this should be possible for all waveforms in parallel (see below).
#     # However, this takes much longer for some reason. Check again, when
#     # using dask (looks like a sparse memory accessing issue).
#     # ds["ph_idx"][~ds.group_id.isnull()] = ds.xph_swath_h_diff.groupby(
#     #       ds.group_id
#     # ).map(lambda x: x.ns_20_ku*0+x.mean("stacked_time_20_ku_ns_20_ku").idxmin(
#     #       "phase_wrap_factor"
#     # ))
#     return waveform.xph_elev_diffs.groupby(waveform.group_id).map(
#         lambda x: x.ns_20_ku * 0 + x.mean("ns_20_ku").idxmin("phase_wrap_factor")
#     )
