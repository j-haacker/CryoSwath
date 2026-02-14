"""CryoSat L1b to L2 processing auxiliary functions"""

import fnmatch
import json
import numpy as np
import rioxarray
import warnings
import xarray as xr
from numpy.typing import ArrayLike
from pyproj import Geod, Transformer
from scipy.stats import median_abs_deviation, ttest_ind, norm
import scipy
from typing import Literal
from pystac_client import Client
from rasterio.enums import Resampling
from pathlib import Path
import rasterio 
import os
import glob

# Define DEM path
dem_path = Path('data/auxiliary/DEM')


def _json_safe_attr(value):
    """Convert xarray attrs to JSON-serializable values for zarr metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(_json_safe_attr(v) for v in value)
    if isinstance(value, tuple):
        return [_json_safe_attr(v) for v in value]
    if isinstance(value, list):
        return [_json_safe_attr(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_attr(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _sanitize_attrs_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Sanitize all attrs in dataset variables/coords before writing to zarr."""
    ds = ds.copy(deep=False)
    ds.attrs = {str(k): _json_safe_attr(v) for k, v in ds.attrs.items()}
    for name in ds.variables:
        ds[name].attrs = {
            str(k): _json_safe_attr(v) for k, v in ds[name].attrs.items()
        }
    return ds


def _is_nan_scalar(value) -> bool:
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _extract_nodata_values(da: xr.DataArray) -> list:
    """Collect nodata/fill-value candidates from attrs/encoding/rio metadata."""
    keys = ("_FillValue", "fill_value", "nodata")
    out = []

    for key in keys:
        if key in da.attrs and da.attrs[key] is not None:
            out.append(da.attrs[key])

    encoding_attr = da.attrs.get("encoding")
    if isinstance(encoding_attr, dict):
        for key in keys:
            if key in encoding_attr and encoding_attr[key] is not None:
                out.append(encoding_attr[key])

    if isinstance(da.encoding, dict):
        for key in keys:
            if key in da.encoding and da.encoding[key] is not None:
                out.append(da.encoding[key])

    try:
        rio_nodata = da.rio.nodata
        if rio_nodata is not None:
            out.append(rio_nodata)
    except Exception:
        pass

    flat = []
    for val in out:
        if isinstance(val, (list, tuple, set)):
            flat.extend(val)
        else:
            flat.append(val)

    deduped = []
    seen = set()
    for val in flat:
        if val is None:
            continue
        key = ("nan",) if _is_nan_scalar(val) else (type(val).__name__, repr(val))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(val)
    return deduped


def _primary_nodata(da: xr.DataArray):
    vals = _extract_nodata_values(da)
    return vals[0] if vals else None


def _mask_nodata(da: xr.DataArray) -> xr.DataArray:
    masked = da
    for nodata in _extract_nodata_values(da):
        if _is_nan_scalar(nodata):
            try:
                masked = masked.where(~np.isnan(masked))
            except Exception:
                continue
        else:
            masked = masked.where(masked != nodata)
    return masked


def _normalize_initial_dem_array(da: xr.DataArray) -> xr.DataArray:
    """Prepare DEM arrays for zarr init with robust dtype/nodata handling."""
    data_type = da.attrs.get("data_type", da.dtype)
    clean = da.drop_attrs().astype(data_type)
    nodata = _primary_nodata(da)
    if nodata is not None:
        clean = clean.assign_attrs(encoding={"_FillValue": nodata})
    return clean


def _compute_track_azimuth(ds: xr.Dataset) -> np.ndarray:
    """Compute along-track azimuth robustly, even for short selections."""
    n_time = len(ds.time_20_ku)
    if n_time == 0:
        return np.empty(0, dtype="float64")
    if n_time == 1:
        return np.zeros(1, dtype="float64")

    bearings = np.asarray(
        WGS84_ellpsoid.inv(
            lats1=ds.lat_20_ku[:-1],
            lons1=ds.lon_20_ku[:-1],
            lats2=ds.lat_20_ku[1:],
            lons2=ds.lon_20_ku[1:],
        )[0],
        dtype="float64",
    )
    x = np.arange(bearings.size, dtype="float64")
    valid = np.isfinite(bearings)
    if valid.sum() == 0:
        return np.zeros(n_time, dtype="float64")
    if valid.sum() == 1:
        return np.full(n_time, float(bearings[valid][0] % 360), dtype="float64")

    x = x[valid]
    bearings = bearings[valid]
    degree = int(min(3, valid.sum() - 1))
    poly3fit_params = np.polyfit(x, bearings, degree)
    azimuth = np.poly1d(poly3fit_params)(np.arange(n_time, dtype="float64")) % 360
    return np.asarray(azimuth, dtype="float64")


def _bounds_for_clip_box(dem_da: xr.DataArray, xs: np.ndarray, ys: np.ndarray):
    """Build clip_box bounds respecting raster axis orientation."""
    xmin = float(np.nanmin(xs))
    xmax = float(np.nanmax(xs))
    ymin = float(np.nanmin(ys))
    ymax = float(np.nanmax(ys))

    x_res, y_res = dem_da.rio.resolution()
    left, right = (xmin, xmax) if x_res >= 0 else (xmax, xmin)
    bottom, top = (ymin, ymax) if y_res < 0 else (ymax, ymin)
    return left, bottom, right, top

def get_dem_reader(data: any = None):
    """Determines which DEM to use based on location or filename.
    
    Args:
        data: Data to determine DEM location from, or path to DEM
        
    Returns:
        Reader for the appropriate DEM file
    """
    raster_extensions = ["tif", "nc", "zarr"]

    def reader_or_store(path: Path):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".tif":
            return rasterio.open(path)
        elif path.suffix == ".nc":
            return xr.open_dataset(path, decode_coords="all", engine="h5netcdf").dem
        elif path.suffix == ".zarr":
            return xr.open_dataset(path, decode_coords="all", engine="zarr").dem
        else:
            raise Exception(str(path) + " cant be read.")

    if (
        isinstance(data, float)
        or isinstance(data, int)
        or (isinstance(data, np.ndarray) and data.size == 1)
    ):
        lat = data
    elif "lat_20_ku" in data:
        lat = data.lat_20_ku.values[0]
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        lat = np.mean(data.rio.transform_bounds("EPSG:4326")[1::2])
    elif isinstance(data, str):
        if data.lower() in ["arctic", "arcticdem"]:
            lat = 90
        elif data.lower() in ["antarctic", "rema"]:
            lat = -90
        elif os.path.sep in data:
            return reader_or_store(data)
        elif any([data.split(".")[-1] in raster_extensions]):
            return reader_or_store(dem_path / data)
    if "lat" not in locals():
        raise NotImplementedError(
            f"`get_dem_reader` could not handle the input of type {data.__class__}. "
            "See doc for further info."
        )
    if lat > 0:
        dem_filename = "arcticdem-mosaics-v4.1-32m.zarr"
    else:
        dem_filename = "rema_mosaic_100m_v2.0_filled_cop30_dem.tif"
        
    if not (dem_path / dem_filename).exists():
        raster_file_list = []
        for ext in raster_extensions:
            raster_file_list.extend(glob.glob("*." + ext, root_dir=dem_path))
        print(
            "DEM not found with default filename. Please select from the following:\n",
            ", ".join(raster_file_list),
            flush=True,
        )
        dem_filename = input("Enter filename:")
    return reader_or_store(dem_path / dem_filename)

# Constants from cryoswath.misc
WGS84_ellpsoid = Geod(ellps="WGS84")
antenna_baseline = 1.1676
Ku_band_freq = 13.575e9
speed_of_light = 299792458.0
sample_width = speed_of_light / (320e6 * 2) / 2

def sel_chunk_range(ds, **coord_ranges):
    """Helper for zarr chunk boundary aligned selection.
    
    Args:
        ds: Dataset to select from 
        **coord_ranges: Dictionary mapping coordinate names to [min, max] lists
        
    Returns:
        Selected Dataset with chunk-aligned bounds
    """
    chunks_dict = {}
    for key, (val1, val2) in coord_ranges.items():
        if key in ds.dims:
            chunks_dict[key] = slice(
                ds[key].values.searchsorted(val1),
                ds[key].values.searchsorted(val2)
            )
    return ds.isel(**chunks_dict)

def if_not_empty(func):
    def wrapper(l1b_data, *args, **kwargs):
        if len(l1b_data.time_20_ku) == 0:
            return l1b_data
        return func(l1b_data, *args, **kwargs)
    return wrapper

def noise_val(vec: ArrayLike) -> float:
    """Calculate average noise values for waveform
    Args:
        vec (ArrayLike): First few (well more than 30) samples of power waveform.
    Returns:
        float: Noise power
    """
    n = 30  # slice_thickness
    for i in range(round(len(vec) / n) - 1):  # look at first quarter samples
        if (ttest_ind(vec[: (i + 1) * n], vec[(i + 1) * n : (i + 2) * n], equal_var=False).pvalue < 0.001):
            return np.mean(vec[: (i + 1) * n])
    return np.mean(vec)

def gauss_filter_DataArray(
    da: xr.DataArray, dim: str, window_extent: int, std: int
) -> xr.DataArray:
    """Low-pass filters input array.

    Convolves each vector of an array along the specified dimension with a
    normalized gauss-function having the specified standard deviation.

    Args:
        da (xr.DataArray): Data to be filtered.
        dim (str): Dimension to apply filter along.
        window_extent (int): Window width. If not uneven, it is increased.
        std (int): Standard deviation of gauss-filter.

    Returns:
        xr.DataArray: _description_
    """
    # force window_extent to be uneven to ensure center to be where expected
    half_window_extent = window_extent // 2
    window_extent = 2 * half_window_extent + 1
    gauss_weights = scipy.stats.norm.pdf(
        np.arange(-half_window_extent, half_window_extent + 1), scale=std
    )
    gauss_weights = xr.DataArray(
        gauss_weights / np.sum(gauss_weights), dims=["window_dim"]
    )
    if np.iscomplexobj(da):
        helper = (
            da.rolling({dim: window_extent}, center=True, min_periods=1)
            .construct("window_dim")
            .dot(gauss_weights)
        )
        return helper / np.abs(helper)
    else:
        return (
            da.rolling({dim: window_extent}, center=True, min_periods=1)
            .construct("window_dim")
            .dot(gauss_weights)
        )

def nan_unique(x):
    """Find unique values considering NaN values"""
    return np.unique(x[~np.isnan(x)])

def download_dem(bounds, *, crs=3413, provider: Literal["PGC"] = "PGC"):
    """Download DEM data using STAC API from a provider.
    
    Args:
        bounds: Bounding box tuple as ``(xmin, ymin, xmax, ymax)``.
        crs: CRS of the input bounds. Defaults to ``3413``.
        provider (Literal["PGC"]): The DEM provider, currently only PGC is supported
        
    Returns:
        Path to downloaded DEM zarr store
    """
    if len(bounds) != 4:
        raise ValueError("bounds must be a 4-item tuple: (xmin, ymin, xmax, ymax)")
    transform_to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = [bounds[0], bounds[2], bounds[0], bounds[2]]
    ys = [bounds[1], bounds[1], bounds[3], bounds[3]]
    lons, lats = transform_to_wgs84.transform(xs, ys)
    bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))

    if provider == "PGC":
        catalog = Client.open("https://stac.pgc.umn.edu/api/v1/")
        collections = catalog.collection_search(q="((arcticdem AND v4+1) OR (rema AND v2)) AND 32m").collections()
        # transforming collection extent is difficult, maybe the code behind rioxr transform_bounds helps
        limits = {"x": (-3_500_000, 3_500_000), "y": (-3_500_000, 3_500_000)}

    items = list(catalog.search(
        collections=[coll.id for coll in collections],
        # not sure how this behaves if it covers the poles
        bbox=bbox_wgs84,
    ).items())

    this_dem_path = dem_path / (items[0].get_collection().id + ".zarr")  # don't .with_suffix; . in name!
    this_dem_path.parent.mkdir(parents=True, exist_ok=True)

    if not this_dem_path.exists():
        initial = (  # init dem store
            xr.full_like(
                xr.open_dataset(items[0], engine="stac", epsg=3413).load().squeeze(),
                np.nan,
            )
            .reindex({xy: np.arange(limits[xy][0], limits[xy][1] + 1, 100) for xy in ["x", "y"]})
            .map(_normalize_initial_dem_array)
            .drop_attrs(deep=False)
            .drop_vars(["time", "id"])
            .rio.write_crs(3413)
        )
        initial = _sanitize_attrs_for_zarr(initial)
        initial.to_zarr(this_dem_path, mode="w")

    for item in items:
        parent = xr.open_zarr(this_dem_path, decode_coords="all", mask_and_scale=True)
        ds = xr.open_dataset(item, engine="stac", epsg=3413).squeeze()
        x0, y0, x1, y1 = ds.rio.bounds()
        excerpt = parent.pipe(sel_chunk_range, x=[x0, x1], y=[y0, y1]).load()
        add = ds.map(lambda da: da.rio.reproject_match(excerpt, resampling=Resampling.average).astype(da.attrs["data_type"]))
        add = add.map(_mask_nodata)
        add = add.map(lambda da: da.fillna(excerpt[da.name]))
        add = add.drop_attrs().drop_vars(['time', 'id', 'spatial_ref'])
        add = _sanitize_attrs_for_zarr(add)
        add.to_zarr(this_dem_path, region="auto")

def read_esa_l1b(
    l1b_filename: str,
    *,
    waveform_selection=None,
    drop_outside=False,
    coherence_threshold: float = 0.6,
    power_threshold: tuple = ("snr", 10),
    smooth_phase_difference: bool = True,
    use_original_noise_estimates: bool = False,
    swath_start_kwargs: dict = {},
):
    """Loads ESA SARIn L1b and does initial processing"""
    if not fnmatch.fnmatch(l1b_filename, "*CS_????_SIR_SIN_1B_*.nc"):
        raise ValueError(
            "Provided filename deviates from standard form. That is currently not "
            "permitted, but feel free to disable this requirement."
        )

    ds = xr.open_dataset(l1b_filename)
    ds = ds.assign_coords(ns_20_ku=("ns_20_ku", np.arange(len(ds.ns_20_ku))))

    # Calculate azimuth bearing before optional subsetting so single-waveform
    # selections still get a valid local track direction.
    ds = ds.assign(azimuth=("time_20_ku", _compute_track_azimuth(ds)))

    # Handle waveform selection exactly as in original cryoswath
    if waveform_selection is not None:
        if (
            not isinstance(waveform_selection, slice) 
            and not isinstance(waveform_selection, list)
        ):
            waveform_selection = [waveform_selection]
        
        if (
            isinstance(waveform_selection, slice) 
            and isinstance(waveform_selection, int)
        ) or isinstance(waveform_selection[0], int):
            ds = ds.isel(time_20_ku=waveform_selection)
        else:
            ds = ds.sel(time_20_ku=waveform_selection)

    # Power waveform calculation
    ds["power_waveform_20_ku"] = (
        ds.pwr_waveform_20_ku * ds.echo_scale_factor_20_ku * 2**ds.echo_scale_pwr_20_ku
    )
    
    # Noise calculation
    if not use_original_noise_estimates:
        tracking_cycles = 5
        if len(ds.time_20_ku) > 2 * (tracking_cycles * 20):
            noise = xr.apply_ufunc(
                noise_val,
                ds.power_waveform_20_ku.isel(ns_20_ku=slice(int(len(ds.ns_20_ku) / 4))),
                input_core_dims=[["ns_20_ku"]],
                output_core_dims=[[]],
                vectorize=True,
            )

            def noise_floor(noise):
                window_size = 5 * 20
                fwd = noise.rolling(time_20_ku=window_size).min()
                bwd = (
                    noise.isel(time_20_ku=slice(None, None, -1))
                    .rolling(time_20_ku=window_size)
                    .min()
                    .isel(time_20_ku=slice(None, None, -1))
                )
                upper_envelope = xr.concat([fwd, bwd], "ds").max("ds")
                return upper_envelope.fillna(upper_envelope.max())

            ds["noise_power_20_ku"] = noise_floor(noise)
    else:
        ds["noise_power_20_ku"] = ds.transmit_pwr_20_ku * 10 ** (ds.noise_power_20_ku / 10)

    ds = ds.assign_attrs(
        coherence_threshold=coherence_threshold,
        power_threshold=power_threshold,
        smooth_phase_difference=smooth_phase_difference,
    )

    ds = ds.assign_coords({"phase_wrap_factor": np.arange(-3, 4)})
    ds["ph_diff"] = ds.ph_diff_waveform_20_ku

    if len(ds.time_20_ku) > 0:
        ds = append_poca_and_swath_idxs(ds, **swath_start_kwargs)
        ds = append_smoothed_complex_phase(ds)
        if smooth_phase_difference:
            ds["ph_diff"] = ds.ph_diff.where(
                ds.ph_diff_complex_smoothed.isnull(),
                xr.apply_ufunc(np.angle, ds.ph_diff_complex_smoothed)
                if not isinstance(smooth_phase_difference, dict)
                else xr.apply_ufunc(
                    np.angle,
                    ds.pipe(append_smoothed_complex_phase, **smooth_phase_difference).ph_diff_complex_smoothed,
                )
            )
        else:
            # always use lowpass-filtered phase difference at POCA
            ds["ph_diff"] = ds.ph_diff.where(
                ds.ns_20_ku != ds.poca_idx,
                xr.apply_ufunc(np.angle, ds.ph_diff_complex_smoothed)
            )
    return ds

def append_smoothed_complex_phase(ds: xr.Dataset, window_extent: int = 21, std: float = 5) -> xr.Dataset:
    """Append smoothed complex phase to dataset.

    Args:
        ds (xr.Dataset): The input dataset
        window_extent (int, optional): Window width for filtering. Defaults to 21.
        std (float, optional): Standard deviation for Gaussian filter. Defaults to 5.

    Returns:
        xr.Dataset: Dataset with smoothed complex phase added
    """
    ds["ph_diff_complex_smoothed"] = gauss_filter_DataArray(
        np.exp(1j * ds.ph_diff_waveform_20_ku),
        dim="ns_20_ku",
        window_extent=window_extent,
        std=std
    )
    return ds

def append_exclude_mask(ds):
    """Adds mask indicating samples below threshold."""
    assert isinstance(ds.power_threshold, tuple)
    assert ds.power_threshold[0] == "snr"
    power_threshold = ds.noise_power_20_ku * ds.power_threshold[1]
    ds["exclude_mask"] = np.logical_or(
        ds.power_waveform_20_ku < power_threshold,
        ds.coherence_waveform_20_ku < ds.coherence_threshold,
    )
    return ds

def append_poca_and_swath_idxs(ds, **kwargs):
    """Adds indices for estimated POCA and begin of swath."""
    coherence_threshold = ds.attrs.get('coherence_threshold', 0.6)
    swath_start_window = kwargs.get('swath_start_window', (5, 50))
    
    if len(ds.time_20_ku) == 0:
        return ds.assign(
            swath_start=(("time_20_ku"), []),
            poca_idx=(("time_20_ku"), []),
            exclude_mask=(
                ("time_20_ku", "ns_20_ku"),
                np.empty_like(ds.power_waveform_20_ku),
            ),
        )

    def find_poca_idx_and_swath_start_idx(smooth_coh, coh_thr):
        poca_idx = np.argmax(smooth_coh > coh_thr)
        if poca_idx < int(10 / sample_width):
            return np.nan, 0
            
        poca_idx = np.argmax(smooth_coh[poca_idx:]) + poca_idx
        
        if swath_start_window[1] < 0:
            swath_start = 0
        else:
            try:
                swath_start = poca_idx + int(swath_start_window[0] / sample_width)
                diff_smooth_coh = np.diff(
                    smooth_coh[swath_start : swath_start + int(swath_start_window[1] / sample_width)]
                )
                swath_start = (
                    np.argmax(
                        diff_smooth_coh[np.argmax(np.abs(diff_smooth_coh) > 0.001) :] > 0
                    )
                    + swath_start
                )
            except ValueError:
                swath_start = len(smooth_coh)
        return float(poca_idx), swath_start

    ds[["poca_idx", "swath_start"]] = xr.apply_ufunc(
        find_poca_idx_and_swath_start_idx,
        gauss_filter_DataArray(ds.coherence_waveform_20_ku, "ns_20_ku", 35, 35),
        kwargs=dict(coh_thr=coherence_threshold),
        input_core_dims=[["ns_20_ku"]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    
    if "exclude_mask" not in ds.data_vars:
        ds = append_exclude_mask(ds)
    ds["exclude_mask"] = ds.exclude_mask.where(
        ds.ns_20_ku >= ds.swath_start, True
    )
    return ds

@if_not_empty
def append_ambiguous_reference_elevation(ds, dem_file_name_or_path: str = None):
    """Append reference elevation from DEM considering phase ambiguity.
    
    Args:
        ds: Dataset to process
        dem_file_name_or_path: Path to DEM file, can be tif or zarr

    Returns:
        Dataset with reference elevations added
    """
    if "xph_lats" not in ds.data_vars:
        ds = locate_ambiguous_origin(ds)

    with get_dem_reader(
        (ds if dem_file_name_or_path is None else dem_file_name_or_path)
    ) as dem_reader:
        if isinstance(dem_reader, xr.DataArray):
            crs = dem_reader.rio.crs
        else:
            crs = dem_reader.crs
            dem_reader = rioxarray.open_rasterio(dem_reader)
        trans_4326_to_dem_crs = Transformer.from_crs(
            "EPSG:4326", crs, always_xy=True
        )
        x, y = trans_4326_to_dem_crs.transform(
            np.asarray(ds.xph_lons), np.asarray(ds.xph_lats)
        )
        x = np.asarray(x)
        y = np.asarray(y)
        if not (np.isfinite(x).any() and np.isfinite(y).any()):
            raise ValueError("Ambiguous reflection coordinates could not be transformed.")
        
        ds = ds.assign(
            xph_x=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), x),
            xph_y=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), y),
        )
        ds.attrs.update({"CRS": crs})

        try:
            left, bottom, right, top = _bounds_for_clip_box(dem_reader, x, y)
            ref_dem = (
                dem_reader
                .rio.clip_box(left, bottom, right, top)
                .squeeze()
            )
        except rioxarray.exceptions.NoDataInBounds:
            warnings.warn(
                f"couldn't find ref dem data in box: {np.nanmin(x)}, {np.nanmin(y)}, "
                f"{np.nanmax(x)}, {np.nanmax(y)}\nouter lat lon coords: "
                f"{ds.lat_20_ku.values[[0, -1]]}, {ds.lon_20_ku.values[[0, -1]]}"
            )
            raise
        ds["xph_ref_elevs"] = ref_dem.sel(x=ds.xph_x, y=ds.xph_y, method="nearest")
            
    return ds

@if_not_empty 
def append_best_fit_phase_index(ds, best_column = None):
    """Resolve phase difference ambiguity"""
    if "group_id" not in ds.data_vars:
        ds = tag_groups(ds)
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

def to_l2(ds, retain_vars=None):
    """Convert processed L1b data to L2 format."""
    selected_phase = ds.sel(phase_wrap_factor=ds.ph_idx)
    result = xr.Dataset({
        "height": selected_phase.xph_elevs,
        "off_nadir": selected_phase.xph_dists,
    })
    
    if retain_vars:
        for old_name, new_name in retain_vars.items():
            result[new_name] = ds[old_name]
            
    return result

@if_not_empty
def tag_groups(ds):
    """Identifies and tags waveform sample groups."""
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

@if_not_empty
def get_phase_jump(ds):
    """Get locations of phase jumps."""
    ph_diff_diff = ds.ph_diff_complex_smoothed.diff("ns_20_ku")
    ph_diff_diff_tolerance = 0.1
    jump_mask = np.logical_or(
        np.abs(ph_diff_diff) > ph_diff_diff_tolerance,
        np.abs(ph_diff_diff).rolling(ns_20_ku=2).sum()
        > 2 * 0.8 * ph_diff_diff_tolerance,
    )
    if "exclude_mask" not in ds.data_vars:
        ds = append_exclude_mask(ds)
    return xr.where(ds.exclude_mask.sel(ns_20_ku=jump_mask.ns_20_ku), False, jump_mask)

def get_phase_outlier(ds, tol=None):
    """Get locations of phase outliers."""
    if tol is None:
        temp_x_width = 300
        temp_H = 720e3
        tol = (
            (np.arctan(np.tan(np.deg2rad(0)) + temp_x_width / temp_H) - np.deg2rad(0))
            * 2
            * np.pi
            / np.tan(speed_of_light / Ku_band_freq / antenna_baseline)
        )
    return (
        np.abs(np.exp(1j * ds.ph_diff_waveform_20_ku) - ds.ph_diff_complex_smoothed)
        > tol
    )

@if_not_empty
def unwrap_phase_diff(ds):
    """Unwrap phase differences within groups."""
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

def locate_ambiguous_origin(ds):
    """Locates possible reflection points considering phase ambiguity"""
    r_N = WGS84_ellpsoid.a / np.sqrt(
        1 - WGS84_ellpsoid.es * np.sin(np.deg2rad(ds.lat_20_ku)) ** 2
    )
    r_cs2 = r_N + ds.alt_20_ku
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
            dist_off_groundtrack.transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values,
        ),
    )

def ref_range(ds):
    """Calculate distance to center of range window."""
    corrections = (
        ds.mod_dry_tropo_cor_01
        + ds.mod_wet_tropo_cor_01
        + ds.iono_cor_gim_01
        + ds.pole_tide_01
        + ds.solid_earth_tide_01
        + ds.load_tide_01
    )
    return ds.window_del_20_ku / np.timedelta64(1, "s") / 2 * speed_of_light + np.interp(
        ds.time_20_ku, ds.time_cor_01, corrections
    )

@if_not_empty
def append_elev_diff_to_ref(ds):
    """Calculate elevation differences to reference DEM."""
    if "xph_ref_elevs" not in ds.data_vars:
        ds = append_ambiguous_reference_elevation(ds)
    ds["xph_elev_diffs"] = ds.xph_elevs - ds.xph_ref_elevs
    return ds

def dem_transect(waveform, ax=None, selected_phase_only=True, dem_file_name_or_path=None):
    """Plot DEM transect with CryoSat measurements."""
    import matplotlib.pyplot as plt
    
    line_properties = {
        "swath": dict(color="tab:blue", marker='.', markersize=5, linewidth=1),
        "poca": dict(color="tab:green", marker='o', markersize=5, linewidth=1),
        "excluded": dict(color="tab:pink", marker='x', markersize=5, linewidth=1),
        "dem": dict(color="black", linestyle="solid", linewidth=0.6, facecolor="xkcd:ice")
    }
    
    if ax is None:
        _, ax = plt.subplots()

    # Get DEM sampling along transect
    dem_reader = get_dem_reader((waveform if dem_file_name_or_path is None else dem_file_name_or_path))
    dem_crs = dem_reader.rio.crs if isinstance(dem_reader, xr.DataArray) else dem_reader.crs
    trans_4326_to_dem_crs = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    sampling_dist = np.arange(-30000, 30000+1, 100)
    num_samples = len(sampling_dist)
    # Extract scalar values for the first point
    lon = float(waveform.lon_20_ku[0])
    lat = float(waveform.lat_20_ku[0])
    az = float(waveform.azimuth[0])
    lons, lats = WGS84_ellpsoid.fwd(
        lons=[lon] * num_samples,
        lats=[lat] * num_samples,
        az=[az + 90] * num_samples,
        dist=sampling_dist,
    )[:2]
    xs, ys = trans_4326_to_dem_crs.transform(lons, lats)
    if isinstance(dem_reader, xr.DataArray):
        ref_elevs = np.fromiter(
            (dem_reader.sel(x=x, y=y, method="nearest").item() for x, y in zip(xs, ys)),
            "float32",
        )
        fill_value = dem_reader.attrs.get("_FillValue")
        if fill_value is not None:
            ref_elevs = np.where(ref_elevs != fill_value, ref_elevs, np.nan)
    else:
        ref_elevs = np.fromiter(
            (vals[0] for vals in dem_reader.sample([(x, y) for x, y in zip(xs, ys)])),
            "float32",
        )
        ref_elevs = np.where(ref_elevs != dem_reader.nodata, ref_elevs, np.nan)
    ax.fill_between(sampling_dist, ref_elevs, **line_properties["dem"], label="DEM")
    h_dem = ax.get_legend_handles_labels()[0][0]

    # Plot all possible solutions if requested
    if not selected_phase_only:
        for ph_idx in waveform.phase_wrap_factor.values:
            temp = waveform.sel(phase_wrap_factor=ph_idx)
            temp = temp.where(waveform.ph_idx!=ph_idx).squeeze()
            ax.plot(temp.xph_dists, temp.xph_elevs, '.', c=f"{(.2+.2*np.abs(ph_idx)):.1f}")

    # Plot selected solution
    best_phase = waveform.sel(phase_wrap_factor=waveform.ph_idx)
    try:
        excluded = best_phase.where(best_phase.exclude_mask).transpose("ns_20_ku", ...)
        h_excl, = ax.plot(excluded.xph_dists, excluded.xph_elevs, ls='', **line_properties["excluded"], label="excluded")
        swath = best_phase.where(~best_phase.exclude_mask).transpose("ns_20_ku", ...)
        h_swath, = ax.plot(swath.xph_dists, swath.xph_elevs, ls='', **line_properties["swath"], label="swath")
        h_list = [h_swath, h_excl]
    except KeyError:
        h_all, = ax.plot(best_phase.xph_dists, best_phase.xph_elevs, ls='', **line_properties["swath"], label="all")
        h_list = [h_all]

    try:
        poca = best_phase.sel(ns_20_ku=best_phase.poca_idx)
        h_poca, = ax.plot(poca.xph_dists, poca.xph_elevs, ls='', **line_properties["poca"], label="POCA")
        h_list.insert(0, h_poca)
    except KeyError:
        pass

    h_list.append(h_dem)
    ax.legend(handles=h_list)
    ax.set_xlabel("across-track distance to nadir, km")
    ax.set_ylabel("elevation, m")
    ax.set_title(f"id: {waveform.time_20_ku.values[0]}")
    
    return ax
