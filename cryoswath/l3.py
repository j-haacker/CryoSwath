"""Aggregate L2 point elevations into regular spatio-temporal L3 grids."""

__all__ = [
    "cache_l2_data",
    "build_dataset",
    "extend_dataset",
    "merge_finalized_dataset_extension",
]

from collections.abc import Mapping
import dask.array
import datetime
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import h5py
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from pyproj.crs import CRS
import shapely
import shutil
import warnings
from pathlib import Path
import tempfile
import xarray as xr

from cryoswath import l1b, l2, provenance
from cryoswath.misc import (
    l3_path,
    dataframe_to_rioxr,
    filter_kwargs,
    find_region_id,
    load_cs_ground_tracks,
    load_glacier_outlines,
    sandbox_write_to,
    tmp_path,
)
from cryoswath.gis import buffer_4326_shp, ensure_pyproj_crs, find_planar_crs


@dataclass(frozen=True, slots=True)
class L3ExtensionSpec:
    """Inferred metadata for extending an L3 store."""

    base_store_path: Path | None
    output_path: Path
    region_id: str
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    timestep_months: int
    window_ntimesteps: int
    spatial_res_meter: float
    crs: CRS | None
    recompute_start_datetime: pd.Timestamp


def _normalize_l3_end_datetime(end_datetime: str | pd.Timestamp) -> pd.Timestamp:
    """Match build_dataset's inclusive month-end normalization."""
    end_datetime = pd.to_datetime(end_datetime)
    return end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")


def _infer_l3_month_step(data: xr.Dataset) -> int:
    """Infer the monthly cadence from attrs or the time coordinate."""
    for key in ["cryoswath_timestep_months", "timestep_months"]:
        if key in data.attrs:
            return int(data.attrs[key])
    if "time" not in data.coords or data.sizes.get("time", 0) <= 1:
        return 1
    freq = pd.infer_freq(pd.DatetimeIndex(data.indexes["time"]))
    if isinstance(freq, str) and freq.endswith("MS"):
        step = freq[:-2]
        return int(step) if step else 1
    time_index = pd.DatetimeIndex(data.indexes["time"]).sort_values()
    delta = relativedelta(time_index[1].to_pydatetime(), time_index[0].to_pydatetime())
    return max(1, int(delta.years * 12 + delta.months))


def _infer_l3_spatial_res_meter(data: xr.Dataset) -> float:
    """Infer the spatial grid spacing from attrs or coordinate spacing."""
    for key in ["cryoswath_spatial_res_meter", "spatial_res_meter"]:
        if key in data.attrs:
            return float(data.attrs[key])
    if "x" in data.coords and data.sizes.get("x", 0) > 1:
        x_vals = np.asarray(data.x.values)
        spacing = np.diff(np.sort(np.unique(x_vals)))
        if spacing.size:
            return float(np.nanmedian(np.abs(spacing)))
    if "y" in data.coords and data.sizes.get("y", 0) > 1:
        y_vals = np.asarray(data.y.values)
        spacing = np.diff(np.sort(np.unique(y_vals)))
        if spacing.size:
            return float(np.nanmedian(np.abs(spacing)))
    return 500.0


def _infer_l3_region_id(data: xr.Dataset, source: Path | None = None) -> str:
    """Infer the region id from attrs, the source path, or geometry."""
    for key in ["cryoswath_region_id", "region_id"]:
        if key in data.attrs:
            return str(data.attrs[key])
    if source is not None:
        stem = source.name
        if stem.endswith(".zarr"):
            stem = stem[: -len(".zarr")]
        if "_monthly_" in stem:
            return stem.split("_monthly_", 1)[0]
    return find_region_id(data)


def _infer_l3_window_ntimesteps(data: xr.Dataset) -> int:
    """Infer the rolling-window width if it was stored."""
    for key in ["cryoswath_window_ntimesteps", "window_ntimesteps"]:
        if key in data.attrs:
            return int(data.attrs[key])
    return 3


def _infer_l3_output_path(
    data: xr.Dataset,
    *,
    region_id: str,
    timestep_months: int,
    spatial_res_meter: float,
    end_datetime: pd.Timestamp | None = None,
    source_path: Path | None = None,
) -> Path:
    """Infer a new output path for an extension run."""
    if source_path is None and "cryoswath_store_path" in data.attrs:
        source_path = Path(data.attrs["cryoswath_store_path"])
    if source_path is None:
        source_path = Path(_build_path(region_id, timestep_months, spatial_res_meter))
    if end_datetime is None:
        suffix = "__extended"
    else:
        suffix = f"__extended_to_{pd.Timestamp(end_datetime).strftime('%Y%m')}"
    return source_path.with_name(f"{source_path.stem}{suffix}.zarr")


def _l3_build_attrs(
    *,
    region_id: str,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    timestep_months: int,
    window_ntimesteps: int,
    spatial_res_meter: float,
    outfilepath: str | Path,
) -> dict[str, object]:
    """Store light-weight provenance that helps future inference."""
    return {
        "cryoswath_region_id": region_id,
        "cryoswath_store_path": str(outfilepath),
        "cryoswath_build_start_datetime": pd.Timestamp(start_datetime).isoformat(),
        "cryoswath_build_end_datetime": pd.Timestamp(end_datetime).isoformat(),
        "cryoswath_timestep_months": int(timestep_months),
        "cryoswath_window_ntimesteps": int(window_ntimesteps),
        "cryoswath_spatial_res_meter": float(spatial_res_meter),
    }


def _l3_extension_attrs(
    base_attrs: dict[str, object],
    *,
    output_path: str | Path,
    source_path: str | Path,
    recompute_start_datetime: pd.Timestamp,
    overlap_time_steps: int,
    overlap_policy: str,
) -> dict[str, object]:
    """Update provenance after an extension run."""
    attrs = dict(base_attrs)
    attrs.update(
        {
            "cryoswath_store_path": str(output_path),
            "cryoswath_extended_from_store_path": str(source_path),
            "cryoswath_extension_overlap_time_steps": int(overlap_time_steps),
            "cryoswath_extension_policy": overlap_policy,
            "cryoswath_recompute_start_datetime": pd.Timestamp(
                recompute_start_datetime
            ).isoformat(),
        }
    )
    return attrs


def _open_l3_dataset(dataset_or_path: xr.Dataset | str | Path) -> xr.Dataset:
    """Open an L3 dataset from a store path or pass through an in-memory dataset."""
    if isinstance(dataset_or_path, xr.Dataset):
        return dataset_or_path
    return xr.open_zarr(dataset_or_path, decode_coords="all")


def _infer_l3_extension_spec(
    source: xr.Dataset | str | Path,
    *,
    region_id: str | None = None,
    start_datetime: str | pd.Timestamp | None = None,
    end_datetime: str | pd.Timestamp | None = None,
    timestep_months: int | None = None,
    window_ntimesteps: int | None = None,
    spatial_res_meter: float | None = None,
    crs: CRS | int | None = None,
    output_path: str | Path | None = None,
    recompute_start_datetime: str | pd.Timestamp | None = None,
) -> L3ExtensionSpec:
    """Infer the choices needed to extend an existing L3 dataset."""
    source_path = Path(source) if not isinstance(source, xr.Dataset) else None
    data = _open_l3_dataset(source)
    if region_id is None:
        region_id = _infer_l3_region_id(data, source_path)
    if start_datetime is None:
        if "time" not in data.coords or data.sizes.get("time", 0) == 0:
            raise ValueError("L3 extension requires a time coordinate.")
        start_datetime = pd.Timestamp(data.time.values[0])
    if end_datetime is None:
        if "time" not in data.coords or data.sizes.get("time", 0) == 0:
            raise ValueError("L3 extension requires a time coordinate.")
        end_datetime = pd.Timestamp(data.time.values[-1])
    if timestep_months is None:
        timestep_months = _infer_l3_month_step(data)
    if window_ntimesteps is None:
        window_ntimesteps = _infer_l3_window_ntimesteps(data)
    if spatial_res_meter is None:
        spatial_res_meter = _infer_l3_spatial_res_meter(data)
    if crs is None:
        crs = data.rio.crs if hasattr(data, "rio") else None
    if output_path is None:
        output_path = _infer_l3_output_path(
            data,
            region_id=region_id,
            timestep_months=timestep_months,
            spatial_res_meter=spatial_res_meter,
            end_datetime=end_datetime,
            source_path=source_path,
        )
    if recompute_start_datetime is None:
        recompute_start_datetime = pd.Timestamp(end_datetime) - pd.DateOffset(
            months=timestep_months
        )
    recompute_start_datetime = pd.Timestamp(recompute_start_datetime)
    return L3ExtensionSpec(
        base_store_path=source_path,
        output_path=Path(output_path),
        region_id=region_id,
        start_datetime=pd.Timestamp(start_datetime),
        end_datetime=_normalize_l3_end_datetime(end_datetime),
        timestep_months=int(timestep_months),
        window_ntimesteps=int(window_ntimesteps),
        spatial_res_meter=float(spatial_res_meter),
        crs=crs,
        recompute_start_datetime=recompute_start_datetime,
    )


def _dataset_time_slice(ds: xr.Dataset, times: pd.DatetimeIndex) -> xr.Dataset:
    """Select one or more time steps while keeping an empty result valid."""
    if len(times) == 0:
        return ds.isel(time=slice(0, 0))
    return ds.sel(time=times)


def _dataset_values_match(
    left: xr.Dataset,
    right: xr.Dataset,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool:
    """Compare values and coordinates, ignoring attributes."""
    if set(left.dims) != set(right.dims) or any(left.sizes[k] != right.sizes[k] for k in left.dims):
        return False
    if set(left.data_vars) != set(right.data_vars):
        return False
    if set(left.coords) != set(right.coords):
        return False
    for name in left.data_vars:
        a = left[name].values
        b = right[name].values
        if np.issubdtype(np.asarray(a).dtype, np.number) or np.issubdtype(np.asarray(b).dtype, np.number):
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    for name in left.coords:
        a = left.coords[name].values
        b = right.coords[name].values
        if np.issubdtype(np.asarray(a).dtype, np.number) or np.issubdtype(np.asarray(b).dtype, np.number):
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


_FINALIZED_NETCDF_ENCODING_KEYS = frozenset(
    {
        "_FillValue",
        "chunksizes",
        "complevel",
        "compression",
        "compression_opts",
        "contiguous",
        "dtype",
        "fletcher32",
        "shuffle",
        "zlib",
    }
)
_FINALIZED_TIME_ENCODING_KEYS = frozenset({"calendar", "units"})


def _array_values_match(
    left,
    right,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool:
    """Compare array-like values, treating NaNs as equal for numeric arrays."""
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        return False
    if np.issubdtype(left_values.dtype, np.number) or np.issubdtype(
        right_values.dtype, np.number
    ):
        return bool(
            np.allclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )
    return bool(np.array_equal(left_values, right_values))


def _attribute_values_match(left, right) -> bool:
    """Compare metadata values without tripping over numpy arrays."""
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_attribute_values_match(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return _array_values_match(left, right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(_attribute_values_match(a, b) for a, b in zip(left, right))
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _require_attrs_match(
    left: Mapping,
    right: Mapping,
    *,
    context: str,
    ignore: set[str] | None = None,
) -> None:
    """Raise if two xarray attribute mappings differ."""
    ignore = set() if ignore is None else ignore
    left_keys = set(left) - ignore
    right_keys = set(right) - ignore
    if left_keys != right_keys:
        missing = sorted(right_keys - left_keys)
        extra = sorted(left_keys - right_keys)
        raise ValueError(
            f"{context} attributes differ; missing from base: {missing}, "
            f"missing from extension: {extra}."
        )
    mismatched = [
        key
        for key in sorted(left_keys)
        if not _attribute_values_match(left[key], right[key])
    ]
    if mismatched:
        raise ValueError(f"{context} attributes differ for {mismatched}.")


def _require_variable_schema_match(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    context: str,
) -> None:
    """Require matching dimensions, dtype, and variable attributes."""
    if left.dims != right.dims:
        raise ValueError(
            f"{context} dimensions differ: {left.dims!r} != {right.dims!r}."
        )
    if left.dtype != right.dtype:
        raise ValueError(f"{context} dtype differs: {left.dtype!r} != {right.dtype!r}.")
    _require_attrs_match(left.attrs, right.attrs, context=context)


def _time_dependent_data_vars(ds: xr.Dataset) -> list[str]:
    """Return finalized-product data variables with a time dimension."""
    return [name for name, data in ds.data_vars.items() if "time" in data.dims]


def _static_data_vars(ds: xr.Dataset) -> list[str]:
    """Return finalized-product data variables without a time dimension."""
    return [name for name, data in ds.data_vars.items() if "time" not in data.dims]


def _open_finalized_l3_dataset(dataset_or_path: xr.Dataset | str | Path) -> xr.Dataset:
    """Open a finalized L3 NetCDF product or pass through an in-memory dataset."""
    if isinstance(dataset_or_path, xr.Dataset):
        return dataset_or_path
    return xr.open_dataset(dataset_or_path, decode_coords="all")


def _require_monotonic_unique_time(ds: xr.Dataset, *, context: str) -> pd.DatetimeIndex:
    """Return a validated monotonic, unique time index."""
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        raise ValueError(f"{context} must have a non-empty time coordinate.")
    time_index = pd.DatetimeIndex(ds.time.values)
    if not time_index.is_monotonic_increasing:
        raise ValueError(f"{context} time coordinate must be monotonic increasing.")
    if not time_index.is_unique:
        raise ValueError(f"{context} time coordinate must not contain duplicates.")
    return time_index


def _require_finalized_schema_match(base: xr.Dataset, extension: xr.Dataset) -> None:
    """Validate finalized-product schema and static content."""
    if set(base.data_vars) != set(extension.data_vars):
        raise ValueError("Finalized datasets must have the same data variables.")
    if set(base.coords) != set(extension.coords):
        raise ValueError("Finalized datasets must have the same coordinates.")
    if set(base.dims) != set(extension.dims):
        raise ValueError("Finalized datasets must have the same dimensions.")
    for dim, size in base.sizes.items():
        if dim != "time" and size != extension.sizes[dim]:
            raise ValueError(
                f"Finalized datasets differ along non-time dimension {dim!r}: "
                f"{size} != {extension.sizes[dim]}."
            )

    _require_attrs_match(
        base.attrs,
        extension.attrs,
        context="global",
        ignore={"history"},
    )

    for name in sorted(base.coords):
        _require_variable_schema_match(
            base.coords[name],
            extension.coords[name],
            context=f"coordinate {name!r}",
        )
        if "time" not in base.coords[name].dims and not _array_values_match(
            base.coords[name].values,
            extension.coords[name].values,
        ):
            raise ValueError(f"Non-time coordinate {name!r} differs.")

    for name in sorted(base.data_vars):
        _require_variable_schema_match(
            base[name],
            extension[name],
            context=f"data variable {name!r}",
        )

    for name in _static_data_vars(base):
        if not _array_values_match(base[name].values, extension[name].values):
            raise ValueError(f"Static data variable {name!r} differs.")


def _finalized_overlap_times(
    base_times: pd.DatetimeIndex,
    extension_times: pd.DatetimeIndex,
    *,
    overlap_time_steps: int,
) -> pd.DatetimeIndex:
    """Return validated contiguous tail/head overlap times."""
    if overlap_time_steps < 1:
        raise ValueError("overlap_time_steps must be at least 1.")
    overlap_times = base_times.intersection(extension_times).sort_values()
    if len(overlap_times) == 0:
        raise ValueError("The extension does not overlap the base dataset.")
    if len(overlap_times) < overlap_time_steps:
        raise ValueError(
            "The extension does not cover the requested overlap_time_steps."
        )

    base_tail = base_times[base_times >= extension_times[0]]
    extension_head = extension_times[extension_times <= base_times[-1]]
    if not overlap_times.equals(base_tail) or not overlap_times.equals(extension_head):
        raise ValueError(
            "The overlap must be the contiguous tail of the base dataset and "
            "the contiguous head of the extension dataset."
        )
    return overlap_times


def _require_finalized_overlap_match(
    base: xr.Dataset,
    extension: xr.Dataset,
    *,
    overlap_times: pd.DatetimeIndex,
    overlap_rtol: float,
    overlap_atol: float,
) -> None:
    """Compare all overlapping months for every time-dependent variable."""
    mismatched = []
    for name in _time_dependent_data_vars(base):
        if not _array_values_match(
            base[name].sel(time=overlap_times).values,
            extension[name].sel(time=overlap_times).values,
            rtol=overlap_rtol,
            atol=overlap_atol,
        ):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "Finalized extension overlap differs for time-dependent variables: "
            + ", ".join(sorted(mismatched))
            + "."
        )


def _copy_variable_encodings(target: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Carry decoded xarray encoding metadata such as scalar coord links."""
    for name in target.variables:
        if name in template.variables:
            target[name].encoding = dict(template[name].encoding)
    return target


def _finalized_netcdf_encoding_from(base: xr.Dataset) -> dict[str, dict[str, object]]:
    """Build a h5netcdf-safe encoding from a finalized base product."""
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in base.variables.items():
        allowed_keys = set(_FINALIZED_NETCDF_ENCODING_KEYS)
        if np.issubdtype(variable.dtype, np.datetime64) or name == "time":
            allowed_keys.update(_FINALIZED_TIME_ENCODING_KEYS)
        variable_encoding = {}
        for key in allowed_keys:
            if key in variable.encoding and variable.encoding[key] is not None:
                variable_encoding[key] = variable.encoding[key]
        if name in base.coords and "_FillValue" not in variable_encoding:
            variable_encoding["_FillValue"] = None
        if variable_encoding:
            encoding[name] = variable_encoding
    return encoding


def merge_finalized_dataset_extension(
    base: xr.Dataset | str | Path,
    extension: xr.Dataset | str | Path,
    *,
    output_path: str | Path | None = None,
    overlap_time_steps: int = 12,
    overlap_policy: str = "abort",
    overlap_rtol: float = 0.0,
    overlap_atol: float = 0.0,
) -> xr.Dataset:
    """Merge a finalized CF-style L3 NetCDF product with a finalized extension.

    The merge is intentionally stricter than :func:`extend_dataset`: it expects
    finalized products with matching schema, static variables, spatial grid, and
    non-time coordinates. In ``overlap_policy="abort"`` mode every overlapping
    month is compared for every time-dependent variable before the extension
    suffix is appended.
    """
    if overlap_policy != "abort":
        raise ValueError(
            "Only overlap_policy='abort' is supported for finalized products."
        )

    base_ds = _open_finalized_l3_dataset(base)
    extension_ds = _open_finalized_l3_dataset(extension)
    base_times = _require_monotonic_unique_time(base_ds, context="base")
    extension_times = _require_monotonic_unique_time(
        extension_ds,
        context="extension",
    )
    _require_finalized_schema_match(base_ds, extension_ds)
    overlap_times = _finalized_overlap_times(
        base_times,
        extension_times,
        overlap_time_steps=overlap_time_steps,
    )
    _require_finalized_overlap_match(
        base_ds,
        extension_ds,
        overlap_times=overlap_times,
        overlap_rtol=overlap_rtol,
        overlap_atol=overlap_atol,
    )

    prefix = base_ds.sel(time=base_times[base_times < overlap_times[0]])
    base_overlap = base_ds.sel(time=overlap_times)
    suffix = extension_ds.sel(time=extension_times[extension_times > overlap_times[-1]])
    pieces = [piece for piece in (prefix, base_overlap, suffix) if piece.sizes["time"]]
    merged = xr.concat(
        pieces,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

    base_path = Path(base) if not isinstance(base, xr.Dataset) else None
    extension_path = Path(extension) if not isinstance(extension, xr.Dataset) else None
    inputs = []
    if base_path is not None:
        inputs.append({"path": base_path, "role": "base-finalized-l3"})
    if extension_path is not None:
        inputs.append({"path": extension_path, "role": "extension-finalized-l3"})
    step = provenance.build_provenance_step(
        "merge finalized L3 dataset extension",
        "cryoswath.l3.merge_finalized_dataset_extension",
        base=str(base_path) if base_path is not None else "<xarray.Dataset>",
        extension=(
            str(extension_path) if extension_path is not None else "<xarray.Dataset>"
        ),
        output_path=str(output_path) if output_path is not None else None,
        overlap_time_steps=overlap_time_steps,
        overlap_policy=overlap_policy,
        overlap_rtol=overlap_rtol,
        overlap_atol=overlap_atol,
        inputs=inputs,
        metadata={
            "overlap_start": overlap_times[0].isoformat(),
            "overlap_end": overlap_times[-1].isoformat(),
            "overlap_count": int(len(overlap_times)),
        },
    )
    merged_attrs = dict(base_ds.attrs)
    merged_attrs["history"] = provenance.append_history(
        base_ds.attrs.get("history"),
        step,
    )
    merged = merged.assign_attrs(merged_attrs)
    merged = _copy_variable_encodings(merged, base_ds)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(
            output_path,
            engine="h5netcdf",
            encoding=_finalized_netcdf_encoding_from(base_ds),
        )

    return merged


def _merge_l3_extension_segments(
    base: xr.Dataset,
    recomputed: xr.Dataset,
    *,
    overlap_time_steps: int = 2,
    overlap_policy: str = "keep_original",
    overlap_rtol: float = 0.0,
    overlap_atol: float = 0.0,
) -> xr.Dataset:
    """Merge the base dataset with a recomputed tail segment."""
    if "time" not in base.coords or "time" not in recomputed.coords:
        raise ValueError("Both datasets must have a time coordinate.")
    if overlap_time_steps < 1:
        raise ValueError("overlap_time_steps must be at least 1.")
    base_times = pd.DatetimeIndex(base.time.values)
    recomputed_times = pd.DatetimeIndex(recomputed.time.values)
    overlap_times = base_times.intersection(recomputed_times)
    if len(overlap_times) == 0:
        raise ValueError("The recomputed segment does not overlap the base dataset.")
    overlap_times = overlap_times.sort_values()
    if len(overlap_times) < overlap_time_steps:
        raise ValueError(
            "The recomputed segment does not cover the requested overlap_time_steps."
        )
    validate_base = _dataset_time_slice(base, overlap_times[:1])
    validate_recomputed = _dataset_time_slice(recomputed, overlap_times[:1])
    exact_match = validate_base.equals(validate_recomputed)
    almost_match = exact_match or _dataset_values_match(
        validate_base, validate_recomputed, rtol=overlap_rtol, atol=overlap_atol
    )
    if not exact_match:
        warnings.warn(
            "The earliest overlapping time step does not match exactly between the "
            "base and recomputed datasets.",
            RuntimeWarning,
            stacklevel=2,
        )
    if overlap_policy == "abort" and not exact_match:
        raise RuntimeError(
            "The earliest overlapping time step changed; aborting to request a full reprocessing."
        )

    step = pd.DateOffset(months=_infer_l3_month_step(recomputed))
    prefix = base.sel(time=slice(None, overlap_times[0] - step))
    base_overlap = _dataset_time_slice(base, overlap_times)
    recomputed_overlap = _dataset_time_slice(recomputed, overlap_times)
    suffix = recomputed.sel(time=slice(overlap_times[-1] + step, None))

    if overlap_policy == "keep_original":
        pieces = [prefix, base_overlap, suffix]
    elif overlap_policy == "use_new":
        pieces = [prefix, recomputed_overlap, suffix]
    elif overlap_policy == "mixed":
        if almost_match:
            pieces = [prefix, base_overlap.isel(time=slice(0, 1)), recomputed_overlap.isel(time=slice(1, None)), suffix]
        else:
            pieces = [prefix, recomputed_overlap, suffix]
    elif overlap_policy == "abort":
        pieces = [prefix, base_overlap, suffix]
    else:
        raise ValueError(
            "overlap_policy must be one of 'keep_original', 'mixed', 'use_new', or 'abort'."
        )

    pieces = [piece for piece in pieces if piece.sizes.get("time", 0) > 0]
    if len(pieces) == 1:
        return pieces[0]
    return xr.concat(
        pieces,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )


# numba does not do help here easily. using the numpy functions is as fast as it gets.
def _med_iqr_cnt(data):
    """Return median, IQR, and sample count for one grouped series."""
    quartiles = np.quantile(data, [0.25, 0.5, 0.75])
    return pd.DataFrame(
        [[quartiles[1], quartiles[2] - quartiles[0], len(data)]],
        columns=["_median", "_iqr", "_count"],
    )


def _ensure_odd_window_ntimesteps(window_ntimesteps: int) -> int:
    """Ensure rolling-window width is odd."""
    if window_ntimesteps % 2 == 0:
        old_window = window_ntimesteps
        window_ntimesteps = window_ntimesteps + 1
        warnings.warn(
            "The window should be a uneven number of time steps. You asked for "
            f"{old_window}, but it has been changed to {window_ntimesteps}."
        )
    return window_ntimesteps


def _ensure_contiguous_time_coord(data: xr.Dataset, timestep_months: int) -> xr.Dataset:
    """Reindex to a contiguous monthly timeline for region writes."""
    if "time" not in data.coords or data.sizes.get("time", 0) <= 1:
        return data
    time_index = data.indexes["time"].sort_values()
    full_time_index = pd.date_range(
        time_index.min(),
        time_index.max(),
        freq=f"{timestep_months}MS",
    )
    if full_time_index.equals(time_index):
        return data
    return data.reindex(time=full_time_index)


def cache_l2_data(
    region_of_interest: str | shapely.Polygon,
    start_datetime: str | pd.Timestamp,
    end_datetime: str | pd.Timestamp,
    *,
    buffer_region_by: float = None,
    max_elev_diff: float = 150,
    timestep_months: int = 1,
    window_ntimesteps: int = 3,
    cache_filename: str = None,
    cache_filename_extra: str = None,
    crs: CRS | int = None,
    reprocess: bool = False,
    **l2_from_id_kwargs,
) -> None:
    """
    Cache Level-2 (L2) data for a specified region and time period.

    This function processes and stores essential L2 data in an HDF5
    file, downloading and processing Level-1b (L1b) files if they are
    not available. It supports buffering the region and time period to
    ensure no data is missed.

    Parameters:
        region_of_interest (str | shapely.Polygon): The region to process,
            specified as a RGI region ID (string) or a custom shapely Polygon.
        start_datetime (str | pd.Timestamp): The start date for the data
            to be cached.
        end_datetime (str | pd.Timestamp): The end date for the data to be cached.
        buffer_region_by (float, optional): Buffer distance (in meters) to
            expand the region of interest. Defaults to 30,000 meters if not provided.
        max_elev_diff (float, optional): Maximum elevation difference to filter
            the data. Defaults to 150 meters.
        timestep_months (int, optional): Time step in months. Defaults to 1 month.
        window_ntimesteps (int, optional): Number of time steps for the rolling
            window data aggregation. Must be an odd number. Defaults to 3.
        cache_filename (str, optional): Custom filename for the cached data.
            Defaults to a name derived from the region ID.
        cache_filename_extra (str, optional): Additional string to append to
            the cache filename. Defaults to None.
        crs (CRS | int, optional): Coordinate reference system for the data.
            If None, a planar CRS is determined automatically. Defaults to None.
        reprocess (bool, optional): Whether to reprocess existing data.
            Defaults to False.
        **l2_from_id_kwargs: Additional keyword arguments passed to the
            :func:`cryoswath.l2.from_id` function.

    Returns:
        None: The function saves the processed data to an HDF5 file and does
        not return any value.

    Raises:
        Warning: If the `window_ntimesteps` is not an odd number, it is adjusted
            and a warning is issued.
    """
    window_ntimesteps = _ensure_odd_window_ntimesteps(window_ntimesteps)
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    # this function only makes sense for multiple months, so assume input
    # was on the month scale and set end_datetime to end of month
    end_datetime = (
        end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")
    )
    if buffer_region_by is None:
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer_months = (window_ntimesteps * timestep_months) // 2
    print(
        "Caching l2 data for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime}",
        f"+-{relativedelta(months=time_buffer_months)}.",
    )
    cs_tracks = load_cs_ground_tracks(
        region_of_interest,
        start_datetime,
        end_datetime,
        buffer_period_by=relativedelta(months=time_buffer_months),
        buffer_region_by=buffer_region_by,
    )
    print(
        "First and last available ground tracks are on",
        f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
        f"{cs_tracks.shape[0]} tracks in total."
        "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
        "`incremental=True`, if you local ground tracks store is not up to",
        "date. Consider pulling the latest version from the repository.",
    )

    # ! exclude data out of regions total_bounds in l2.from_id
    # (?possible/logically consistent?)
    print(
        "Storing the essential L2 data in hdf5, downloading and",
        "processing L1b files if not available...",
    )
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
        region_of_interest = load_glacier_outlines(region_id, "glaciers")
    else:
        region_id = "_".join(
            [
                f"{region_of_interest.centroid.x:.0f}",
                f"{region_of_interest.centroid.y:.0f}",
            ]
        )
    if cache_filename is None:
        cache_filename = region_id
    if cache_filename_extra is not None:
        cache_filename += "_" + cache_filename_extra
    cache_fullname = os.path.join(tmp_path, cache_filename)
    if crs is None:
        crs = find_planar_crs(shp=region_of_interest)
    else:
        crs = ensure_pyproj_crs(crs)
    # cutting to actual glacier outlines takes very long. if needed,
    # implement multiprocessing.
    # bbox = gpd.GeoSeries(
    #     shapely.box(*gpd.GeoSeries(region_of_interest,
    #                 crs=4326).to_crs(crs).bounds.values[0]),
    #     crs=crs)
    # below tries to balance a large cache file with speed. it is not meant
    # to retain data in the suroundings - this is merely needed for the
    # implicit `simplify`` which would come at the cost of data if not
    # buffered
    bbox = gpd.GeoSeries(buffer_4326_shp(region_of_interest, 3_000), crs=4326).to_crs(
        crs
    )
    with sandbox_write_to(cache_fullname) as target:
        l2.from_id(
            cs_tracks.index,
            reprocess=reprocess,
            save_or_return="save",
            cache_fullname=target,
            crs=crs,
            bbox=bbox,
            max_elev_diff=max_elev_diff,
            **filter_kwargs(
                l2.from_id,
                l2_from_id_kwargs,
                blacklist=["cache", "max_elev_diff", "save_or_return", "reprocess"],
            ),
        )
    print(
        "Successfully finished caching for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime}",
        f"+-{relativedelta(months=time_buffer_months)}.",
    )


def _preallocate_zarr(path, bbox, crs, time_index, data_vars, attrs=None) -> None:
    """Create an empty chunked zarr layout for future L3 writes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x_dummy = np.arange(
        (bbox.bounds[0] // 500 + 0.5) * 500, bbox.bounds[2], 500, dtype="i4"
    )
    y_dummy = np.arange(
        (bbox.bounds[1] // 500 + 0.5) * 500, bbox.bounds[3], 500, dtype="i4"
    )
    array_dummy = xr.DataArray(
        dask.array.full(
            shape=(len(time_index), len(x_dummy), len(y_dummy)),
            fill_value=np.nan,
            dtype="f4",
        ),
        coords={"time": time_index, "x": x_dummy, "y": y_dummy},
    )
    (
        xr.merge([array_dummy.rename(stat) for stat in data_vars])
        .assign_attrs({} if attrs is None else attrs)
        .rio.write_crs(crs)
        .to_zarr(path, compute=False)
    )


def build_dataset(
    region_of_interest: str | shapely.Polygon,
    start_datetime: str | pd.Timestamp,
    end_datetime: str | pd.Timestamp,
    *,
    l2_type: str = "swath",
    buffer_region_by: float = None,
    max_elev_diff: float = 150,
    timestep_months: int = 1,
    window_ntimesteps: int = 3,
    spatial_res_meter: float = 500,
    agg_func_and_meta: tuple[callable, dict] = (
        _med_iqr_cnt,
        {"_median": "f8", "_iqr": "f8", "_count": "i8"},
    ),
    cache_filename: str = None,
    cache_filename_extra: str = None,
    outfilepath: str | Path = None,
    crs: CRS | int = None,
    reprocess: bool = False,
    **l2_from_id_kwargs,
):
    """
    Build a gridded dataset of elevation estimates.

    This function aggregates Level-2 (L2) elevation data into a regular
    grid using a rolling window approach. The resulting dataset is
    stored in a Zarr format for efficient access and analysis.

    Parameters:
        region_of_interest (str | shapely.Polygon): The region to process,
            specified as a RGI region ID (string) or a custom shapely Polygon.
        start_datetime (str | pd.Timestamp): The start date for the dataset.
        end_datetime (str | pd.Timestamp): The end date for the dataset.
        l2_type (str, optional): Type of L2 data to process ("swath", "poca", or
            "both"). Defaults to "swath".
        buffer_region_by (float, optional): Buffer distance (in meters) to expand
            the region of interest. Defaults to 30,000 meters if not provided.
        max_elev_diff (float, optional): Maximum elevation difference to filter
            the data. Defaults to 150 meters.
        timestep_months (int, optional): Time step in months. Defaults to 1 month.
        window_ntimesteps (int, optional): Number of time steps for the rolling
            window data aggregation. Must be an odd number. Defaults to 3.
        spatial_res_meter (float, optional): Spatial resolution of the output grid
            in meters. Defaults to 500 meters.
        agg_func_and_meta (tuple[callable, dict], optional): Aggregation function
            and metadata for the output variables. Defaults to calculating the
            median, interquartile range, and data count.
        cache_filename (str, optional): Custom filename for the cached L2 data.
            Defaults to a name derived from the region ID.
        cache_filename_extra (str, optional): Additional string to append to
            the cache filename. Defaults to None.
        outfilepath (str | Path, optional): Output zarr path. If omitted, it is
            inferred from the region and grid settings.
        crs (CRS | int, optional): Coordinate reference system for the data.
            If None, a planar CRS is determined automatically. Defaults to None.
        reprocess (bool, optional): Whether to reprocess existing data.
            Defaults to False.
        **l2_from_id_kwargs: Additional keyword arguments passed to the
            :func:`cryoswath.l2.from_id` function.

    Returns:
        xarray.Dataset: The gridded dataset of elevation estimates.

    Raises:
        Warning: If the `window_ntimesteps` is not an odd number, it is adjusted
            and a warning is issued.
        Exception: If joined swath and poca aggregation is requested (not implemented).

    Notes:
        - The function requires significant amounts of working memory.
        - Intermediate results are saved to ensure progress is not lost in case
          of interruptions.
    """
    window_ntimesteps = _ensure_odd_window_ntimesteps(window_ntimesteps)
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    # this function only makes sense for multiple months, so assume input
    # was on the month scale and set end_datetime to end of month
    end_datetime = (
        end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")
    )
    print(
        "Building a gridded dataset of elevation estimates for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime} every {timestep_months} months for",
        f"a rolling window of {window_ntimesteps} time steps.",
    )
    if buffer_region_by is None:
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer_months = (window_ntimesteps * timestep_months) // 2
    cs_tracks = load_cs_ground_tracks(
        region_of_interest,
        start_datetime,
        end_datetime,
        buffer_period_by=relativedelta(months=time_buffer_months),
        buffer_region_by=buffer_region_by,
    )
    print(
        "First and last available ground tracks are on",
        f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
        f"{cs_tracks.shape[0]} tracks in total."
        "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
        "`incremental=True`, if you local ground tracks store is not up to",
        "date. Consider pulling the latest version from the repository.",
    )

    # ! exclude data out of regions total_bounds in l2.from_id
    # (?possible/logically consistent?)
    print(
        "Storing the essential L2 data in hdf5, downloading and",
        "processing L1b files if not available...",
    )
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
        region_of_interest = load_glacier_outlines(region_id, "glaciers")
    else:
        region_id = "_".join(
            [
                f"{region_of_interest.centroid.x:.0f}",
                f"{region_of_interest.centroid.y:.0f}",
            ]
        )
    if cache_filename is None:
        cache_filename = region_id
    if cache_filename_extra is not None:
        cache_filename += "_" + cache_filename_extra
    cache_fullname = os.path.join(tmp_path, cache_filename)
    if crs is None:
        crs = find_planar_crs(shp=region_of_interest)
    else:
        crs = ensure_pyproj_crs(crs)
    # cutting to actual glacier outlines takes very long. if needed,
    # implement multiprocessing.
    # bbox = gpd.GeoSeries(
    #     shapely.box(*gpd.GeoSeries(region_of_interest,
    #                 crs=4326).to_crs(crs).bounds.values[0]),
    #     crs=crs)
    # below tries to balance a large cache file with speed. it is not meant
    # to retain data in the suroundings - this is merely needed for the
    # implicit `simplify` which would come at the cost of data if not
    # buffered
    region_of_interest = (
        gpd.GeoSeries(buffer_4326_shp(region_of_interest, 3_000), crs=4326)
        .to_crs(crs)
        .make_valid()
    )
    if outfilepath is None:
        outfilepath = _build_path(region_id, timestep_months, spatial_res_meter)
    outfilepath = Path(outfilepath)
    build_attrs = _l3_build_attrs(
        region_id=region_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timestep_months=timestep_months,
        window_ntimesteps=window_ntimesteps,
        spatial_res_meter=spatial_res_meter,
        outfilepath=outfilepath,
    )

    with sandbox_write_to(cache_fullname) as target:
        l2.from_id(
            cs_tracks.index,
            reprocess=reprocess,
            save_or_return="save",
            cache_fullname=target,
            crs=crs,
            bbox=region_of_interest,
            max_elev_diff=max_elev_diff,
            **filter_kwargs(
                l2.from_id,
                l2_from_id_kwargs,
                blacklist=["cache", "max_elev_diff", "save_or_return", "reprocess"],
            ),
            **filter_kwargs(
                l1b.read_esa_l1b,
                l2_from_id_kwargs,
            ),
        )
    ext_t_axis = pd.date_range(
        start_datetime - pd.DateOffset(months=time_buffer_months),
        end_datetime + pd.DateOffset(months=time_buffer_months),
        freq=f"{timestep_months}MS",
    )
    # strip GeoSeries-container -> shapely.Geometry
    region_of_interest = region_of_interest.iloc[0]
    if reprocess and os.path.isdir(outfilepath):
        shutil.rmtree(outfilepath)
    if os.path.isdir(outfilepath):
        previously_processed_l3 = xr.open_zarr(outfilepath, decode_coords="all")
    else:
        _preallocate_zarr(
            outfilepath,
            region_of_interest,
            crs,
            ext_t_axis,
            agg_func_and_meta[1].keys(),
            attrs=build_attrs,
        )
    ext_t_axis = ext_t_axis.astype("int64")
    node_list = []

    def collect_chunk_names(name, node):
        nonlocal node_list
        name_parts = name.split("/")
        if (
            not isinstance(node, h5py.Group)
            or len(name_parts) < 2
            or not name_parts[-2].startswith("x_")
        ):
            return
        chunk_name = name_parts[:2]
        if chunk_name not in node_list:
            x_range, y_range = chunk_name
            x0, x1 = [int(item) for item in x_range.split("_")[-2:]]
            y0, y1 = [int(item) for item in y_range.split("_")[-2:]]
            if (
                x0 > region_of_interest.bounds[2]
                or x1 < region_of_interest.bounds[0]
                or y0 > region_of_interest.bounds[3]
                or y1 < region_of_interest.bounds[1]
            ):
                return
            if not shapely.box(x0, y0, x1, y1).intersects(region_of_interest):
                print(
                    "cell",
                    chunk_name,
                    "will be skipped. cell does not intersect current region",
                )
            elif (
                "previously_processed_l3" in locals()
                and not previously_processed_l3._median.sel(
                    x=slice(x0, x1), y=slice(y0, y1)
                )
                .isnull()
                .all()
            ):
                print("cell", chunk_name, "will be skipped. data is present")
            else:
                node_list.append(chunk_name)

    with h5py.File(cache_fullname, "r") as h5:
        if l2_type == "swath":
            h5["swath"].visititems(collect_chunk_names)
        elif l2_type == "poca":
            h5["poca"].visititems(collect_chunk_names)
        elif l2_type in ["all", "both"]:
            raise NotImplementedError(
                "Joined swath and poca aggregation is not completely implemented."
            )
    print("processing queue contains:\n", node_list)
    print("\nGridding the data. Each chunk at a time...")
    # for the loop below, multiprocessing could be used. however, the
    # implementation should save intermediate results if interupted.
    for chunk_name in node_list:
        print("-----\n\nnext chunk:", chunk_name)
        with h5py.File(cache_fullname, "r") as h5:
            period_list = list(h5["/".join([l2_type] + chunk_name)].keys())
        l2_df = pd.concat(
            [
                pd.read_hdf(
                    cache_fullname,
                    "/".join([l2_type] + chunk_name + [period]),
                    mode="r",
                )
                for period in sorted(period_list)
            ],
            axis=0,
        )
        # one could drop some of the data before gridding. however, excluding
        # off-glacier data is expensive and filtering large differences to the
        # DEM can hide issues while statistics like the median and the IQR
        # should be fairly robust.
        if len(l2_df.index) != 0:
            l2_df = l2_df.loc[ext_t_axis[0] : ext_t_axis[-1]]
            l2_df[["x", "y"]] = (
                (l2_df[["x", "y"]] // spatial_res_meter + 0.5) * spatial_res_meter
            ).astype("i4")
            l2_df["roll_0"] = pd.cut(
                l2_df.index,
                bins=ext_t_axis,
                right=False,
                labels=False,
                include_lowest=True,
            )
            # note on the for-loops:
            #     because of the late-binding python behavior, one or the other way the
            #     counting index must be defined at place (as opposed to when dask tries
            #     to calculate the values because then all the indeces have the same
            #     value). the chosen way is defining a function which creates a new
            #     namespace (in which the index is copied and will not be changed from
            #     outside).
            for i in range(1, window_ntimesteps):
                l2_df[f"roll_{i}"] = l2_df.roll_0 - i
            for i in range(window_ntimesteps):
                l2_df[f"roll_{i}"] = (
                    l2_df[f"roll_{i}"].astype("i4") // window_ntimesteps
                )
            results_list = [None] * window_ntimesteps
            expected_stats = list(agg_func_and_meta[1].keys())
            for i in range(window_ntimesteps):

                def local_closure(roll_iteration):
                    # note: consider calculating the kurtosis of the data between the
                    #       25th and the 75th percentile. this could help later on to
                    #       identify the approximate distribution shape
                    return (
                        l2_df.rename(columns={f"roll_{i}": "time_idx"})
                        .groupby(["time_idx", "x", "y"])
                        .h_diff.apply(agg_func_and_meta[0])
                    )

                results_list[i] = local_closure(i)
            del l2_df
            for i in range(window_ntimesteps):
                result = results_list[i]
                if isinstance(result, pd.Series):
                    if result.empty:
                        result = pd.DataFrame(
                            index=pd.MultiIndex.from_arrays(
                                [
                                    np.array([], dtype="i8"),
                                    np.array([], dtype="i8"),
                                    np.array([], dtype="i8"),
                                ],
                                names=["time_idx", "x", "y"],
                            ),
                            columns=expected_stats,
                        )
                    else:
                        result = result.unstack(level=-1)
                if result.index.nlevels == 4:
                    result = result.droplevel(3, axis=0)
                elif result.index.nlevels != 3:
                    raise ValueError(
                        "Unexpected grouped result index depth in l3 aggregation: "
                        f"{result.index.nlevels}."
                    )
                results_list[i] = result.reindex(columns=expected_stats)
                results_list[i].index = (
                    results_list[i]
                    .index.set_levels(
                        (results_list[i].index.levels[0] * window_ntimesteps + i + 1),
                        level=0,
                    )
                    .rename("time", level=0)
                )
            l3_data = (
                pd.concat(results_list)
                .sort_index()
                .loc[(slice(0, len(ext_t_axis) - 1), slice(None), slice(None)), :]
            )
            for df in results_list:
                del df
            l3_data.index = l3_data.index.remove_unused_levels()
            l3_data.index = l3_data.index.set_levels(
                ext_t_axis[l3_data.index.levels[0]].astype("datetime64[ns]"), level=0
            )
            l3_data = l3_data.sort_index()
            l3_data = l3_data.query(
                f"time >= '{start_datetime}' and time <= '{end_datetime}'"
            )

            try:
                l3_chunk = (
                    dataframe_to_rioxr(l3_data, crs)
                    .rio.clip([region_of_interest])
                    .drop_vars(["spatial_ref"])
                )
                l3_chunk = _ensure_contiguous_time_coord(l3_chunk, timestep_months)
                l3_chunk.to_zarr(
                    outfilepath, region="auto"
                )  # [["_median", "_iqr", "_count"]]
            except Exception as err:
                print("\n")
                warnings.warn(
                    "Failed to write to zarr! Attempting to dump current dataframe."
                )
                try:
                    safety_net_tmp_file_path = os.path.join(
                        tmp_path,
                        "__".join(
                            [
                                f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}",
                                "l3_dfdump",
                                f"region_{region_id}",
                                "_".join(chunk_name),
                            ]
                        )
                        + ".feather",
                    )
                    Path(safety_net_tmp_file_path).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    l3_data.to_feather(safety_net_tmp_file_path)
                except Exception as err_inner:
                    print("\n")
                    warnings.warn(
                        "Failed to do an emergency dump!" + " Rethrowing errors:"
                    )
                    raise err_inner
                else:
                    print(
                        "\n",
                        "Managed to dump current dataframe to",
                        safety_net_tmp_file_path,
                    )
                    print("\n", "Original error is printed below.", str(err), "\n")
            else:
                print(datetime.datetime.now())
                print("processed and stored cell", chunk_name)
                print(l3_data.head())
    print("\n\n+++++++++++++ successfully build dataset ++++++++++++++\n\n")
    result = xr.open_zarr(outfilepath, decode_coords="all")
    result.attrs.update(build_attrs)
    return result


def extend_dataset(
    dataset_or_path: xr.Dataset | str | Path,
    end_datetime: str | pd.Timestamp,
    *,
    recompute_start_datetime: str | pd.Timestamp = None,
    output_path: str | Path = None,
    overlap_time_steps: int = 2,
    overlap_policy: str = "keep_original",
    overlap_rtol: float = 1e-5,
    overlap_atol: float = 1e-8,
    region_of_interest: str | shapely.Polygon = None,
    l2_type: str = "swath",
    buffer_region_by: float = None,
    max_elev_diff: float = 150,
    timestep_months: int = None,
    window_ntimesteps: int = None,
    spatial_res_meter: float = None,
    agg_func_and_meta: tuple[callable, dict] = (
        _med_iqr_cnt,
        {"_median": "f8", "_iqr": "f8", "_count": "i8"},
    ),
    cache_filename: str = None,
    cache_filename_extra: str = None,
    crs: CRS | int = None,
    overwrite: bool = False,
    reprocess: bool = True,
    **l2_from_id_kwargs,
):
    """Extend an existing L3 dataset by recomputing a tail segment."""
    base = _open_l3_dataset(dataset_or_path)
    source_path = (
        Path(dataset_or_path)
        if not isinstance(dataset_or_path, xr.Dataset)
        else Path(base.attrs["cryoswath_store_path"])
        if "cryoswath_store_path" in base.attrs
        else None
    )
    if region_of_interest is None:
        region_of_interest = _infer_l3_region_id(base, source_path)
    if timestep_months is None:
        timestep_months = _infer_l3_month_step(base)
    if window_ntimesteps is None:
        window_ntimesteps = _infer_l3_window_ntimesteps(base)
    if spatial_res_meter is None:
        spatial_res_meter = _infer_l3_spatial_res_meter(base)
    if crs is None and hasattr(base, "rio"):
        crs = base.rio.crs
    end_datetime = _normalize_l3_end_datetime(end_datetime)
    if output_path is None:
        output_path = _infer_l3_output_path(
            base,
            region_id=region_of_interest,
            timestep_months=timestep_months,
            spatial_res_meter=spatial_res_meter,
            end_datetime=end_datetime,
            source_path=source_path,
        )
    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "time" not in base.coords or base.sizes.get("time", 0) == 0:
        raise ValueError("The source L3 dataset must have a time coordinate.")
    base_times = pd.DatetimeIndex(base.time.values).sort_values()
    base_start = base_times[0]
    base_end = base_times[-1]
    if recompute_start_datetime is None:
        recompute_start_datetime = base_end - pd.DateOffset(
            months=timestep_months * (overlap_time_steps - 1)
        )
    recompute_start_datetime = pd.Timestamp(recompute_start_datetime)
    if recompute_start_datetime < base_start:
        recompute_start_datetime = base_start
    if recompute_start_datetime > base_end:
        raise ValueError(
            "The recompute start must overlap the source dataset; use build_dataset "
            "for a full reprocessing run."
        )

    temp_cache_extra = cache_filename_extra
    if temp_cache_extra is None:
        temp_cache_extra = (
            f"extend_{base_start.strftime('%Y%m')}_{end_datetime.strftime('%Y%m')}"
        )

    with tempfile.TemporaryDirectory(prefix="cryoswath-l3-extend-") as tmpdir:
        recompute_store = Path(tmpdir) / f"{output_path.stem}__segment.zarr"
        recomputed = build_dataset(
            region_of_interest,
            recompute_start_datetime,
            end_datetime,
            l2_type=l2_type,
            buffer_region_by=buffer_region_by,
            max_elev_diff=max_elev_diff,
            timestep_months=timestep_months,
            window_ntimesteps=window_ntimesteps,
            spatial_res_meter=spatial_res_meter,
            agg_func_and_meta=agg_func_and_meta,
            cache_filename=cache_filename,
            cache_filename_extra=temp_cache_extra,
            outfilepath=recompute_store,
            crs=crs,
            reprocess=reprocess,
            **l2_from_id_kwargs,
        )
        merged = _merge_l3_extension_segments(
            base,
            recomputed,
            overlap_time_steps=overlap_time_steps,
            overlap_policy=overlap_policy,
            overlap_rtol=overlap_rtol,
            overlap_atol=overlap_atol,
        )
        final_attrs = dict(base.attrs)
        final_attrs.update(
            _l3_build_attrs(
                region_id=str(region_of_interest),
                start_datetime=base_start,
                end_datetime=end_datetime,
                timestep_months=timestep_months,
                window_ntimesteps=window_ntimesteps,
                spatial_res_meter=spatial_res_meter,
                outfilepath=output_path,
            )
        )
        final_attrs.update(
            _l3_extension_attrs(
                final_attrs,
                output_path=output_path,
                source_path=(source_path or output_path),
                recompute_start_datetime=recompute_start_datetime,
                overlap_time_steps=overlap_time_steps,
                overlap_policy=overlap_policy,
            )
        )
        merged = merged.assign_attrs(final_attrs)
        merged.to_zarr(output_path, mode="w")

    result = xr.open_zarr(output_path, decode_coords="all")
    result.attrs.update(final_attrs)
    return result


def _build_path(
    region_of_interest, timestep_months, spatial_res_meter, aggregation_period=None
):
    """Build output zarr path for an L3 product."""
    # ! implement parsing aggregation period
    if not isinstance(region_of_interest, str):
        region_id = find_region_id(region_of_interest)
    else:
        region_id = region_of_interest
    if timestep_months != 1:
        timestep_str = str(timestep_months) + "-"
    else:
        timestep_str = ""
    timestep_str += "monthly"
    if spatial_res_meter == 1000:
        spatial_res_str = "1km"
    elif np.floor(spatial_res_meter / 1000) < 2:
        spatial_res_str = f"{spatial_res_meter}m"
    else:
        # if the trailing ".0" should be omitted, that needs to be implemented here
        spatial_res_str = f"{round(spatial_res_meter / 1000, 1)}km"
    return os.path.join(
        l3_path, "_".join([region_id, timestep_str, spatial_res_str + ".zarr"])
    )
