from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cryoswath.l3 as l3


def _tiny_finalized(times: pd.DatetimeIndex | list[str]) -> xr.Dataset:
    times = pd.DatetimeIndex(times)
    values = np.arange(len(times) * 2 * 3, dtype=float).reshape(len(times), 2, 3)
    values[1, 0, 1] = np.nan
    ds = xr.Dataset(
        data_vars={
            "elev_diff": (("time", "x", "y"), values),
            "elev_diff_error": (("time", "x", "y"), values + 0.25),
            "elev_diff_ref": (("x", "y"), np.arange(6, dtype=float).reshape(2, 3)),
        },
        coords={
            "time": times,
            "x": np.array([0.0, 500.0]),
            "y": np.array([1000.0, 1500.0, 2000.0]),
            "spatial_ref": xr.DataArray(
                0.0,
                attrs={
                    "grid_mapping_name": "polar_stereographic",
                    "spatial_ref": "EPSG:3413",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.12",
            "title": "Tiny finalized L3 product",
            "source": "unit test",
            "history": "first line",
        },
    )
    ds["elev_diff"].attrs.update(
        {
            "standard_name": "land_ice_surface_height_above_reference",
            "units": "m",
        }
    )
    ds["elev_diff_error"].attrs.update(
        {
            "standard_name": "land_ice_surface_height_above_reference standard_error",
            "units": "m",
        }
    )
    ds["elev_diff_ref"].attrs.update(
        {
            "standard_name": "land_height_reference_above_WGS84",
            "units": "m",
        }
    )
    for name in ds.data_vars:
        ds[name].encoding["_FillValue"] = np.nan
    return ds


def _assert_same_except_history(actual: xr.Dataset, expected: xr.Dataset) -> None:
    expected = expected.copy()
    expected.attrs["history"] = actual.attrs["history"]
    xr.testing.assert_identical(actual, expected)


def test_finalized_exact_split_remerge_appends_history():
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-03-01", "2020-06-01"))

    merged = l3.merge_finalized_dataset_extension(
        base,
        extension,
        overlap_time_steps=2,
    )

    _assert_same_except_history(merged, truth)
    history_lines = merged.attrs["history"].splitlines()
    assert len(history_lines) == 2
    assert "merge finalized L3 dataset extension" in history_lines[-1]


def test_finalized_overlap_value_mismatch_raises_in_abort_mode():
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-03-01", "2020-06-01")).copy(deep=True)
    extension["elev_diff"].loc[dict(time="2020-04-01", x=0.0, y=1000.0)] += 1

    with pytest.raises(RuntimeError, match="overlap differs"):
        l3.merge_finalized_dataset_extension(
            base,
            extension,
            overlap_time_steps=2,
        )


def test_finalized_static_variable_mismatch_raises():
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-03-01", "2020-06-01")).copy(deep=True)
    extension["elev_diff_ref"].loc[dict(x=0.0, y=1000.0)] += 1

    with pytest.raises(ValueError, match="Static data variable"):
        l3.merge_finalized_dataset_extension(
            base,
            extension,
            overlap_time_steps=2,
        )


def test_finalized_spatial_coordinate_mismatch_raises():
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-03-01", "2020-06-01")).copy(deep=True)
    extension = extension.assign_coords(x=[0.0, 501.0])

    with pytest.raises(ValueError, match="Non-time coordinate"):
        l3.merge_finalized_dataset_extension(
            base,
            extension,
            overlap_time_steps=2,
        )


def test_finalized_insufficient_overlap_raises():
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-04-01", "2020-06-01"))

    with pytest.raises(ValueError, match="requested overlap_time_steps"):
        l3.merge_finalized_dataset_extension(
            base,
            extension,
            overlap_time_steps=2,
        )


def test_finalized_path_write_reopen_preserves_data_and_schema(tmp_path: Path):
    truth = _tiny_finalized(pd.date_range("2020-01-01", periods=6, freq="MS"))
    base = truth.sel(time=slice("2020-01-01", "2020-04-01"))
    extension = truth.sel(time=slice("2020-03-01", "2020-06-01"))
    base_path = tmp_path / "base.nc"
    extension_path = tmp_path / "extension.nc"
    output_path = tmp_path / "merged.nc"
    base.to_netcdf(base_path, engine="h5netcdf")
    extension.to_netcdf(extension_path, engine="h5netcdf")

    l3.merge_finalized_dataset_extension(
        base_path,
        extension_path,
        output_path=output_path,
        overlap_time_steps=2,
    )

    with xr.open_dataset(output_path, decode_coords="all") as merged:
        _assert_same_except_history(merged, truth)
        assert "merge finalized L3 dataset extension" in merged.attrs["history"]
