from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cryoswath.l3 as l3


def _tiny_l3(times, values, *, source_path: Path | None = None) -> xr.Dataset:
    data = np.asarray(values, dtype=float)[:, None, None]
    ds = xr.Dataset(
        {
            "_median": (("time", "x", "y"), data),
            "_iqr": (("time", "x", "y"), data + 10),
            "_count": (("time", "x", "y"), data + 20),
        },
        coords={"time": pd.DatetimeIndex(times), "x": [0], "y": [0]},
    )
    if source_path is not None:
        ds.attrs.update(
            {
                "cryoswath_region_id": "05-01",
                "cryoswath_store_path": str(source_path),
                "cryoswath_timestep_months": 1,
                "cryoswath_window_ntimesteps": 3,
                "cryoswath_spatial_res_meter": 500,
            }
        )
    return ds


def test_infer_extension_spec_from_store_attrs_and_path(tmp_path: Path):
    source_path = tmp_path / "05-01_monthly_500m.zarr"
    ds = _tiny_l3(["2020-01-01", "2020-02-01", "2020-03-01"], [1, 2, 3], source_path=source_path)
    ds.to_zarr(source_path, mode="w")

    spec = l3._infer_l3_extension_spec(source_path, end_datetime="2020-05-01")

    assert spec.base_store_path == source_path
    assert spec.region_id == "05-01"
    assert spec.timestep_months == 1
    assert spec.window_ntimesteps == 3
    assert spec.spatial_res_meter == 500
    assert spec.output_path == tmp_path / "05-01_monthly_500m__extended_to_202005.zarr"
    assert spec.end_datetime == pd.Timestamp("2020-05-31 23:59:59")


def test_keep_original_policy_preserves_base_boundary():
    base = _tiny_l3(["2020-01-01", "2020-02-01", "2020-03-01"], [1, 2, 3])
    recomputed = _tiny_l3(
        ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        [2.1, 30, 40, 50],
    )

    with pytest.warns(RuntimeWarning, match="does not match exactly"):
        merged = l3._merge_l3_extension_segments(
            base,
            recomputed,
            overlap_time_steps=2,
            overlap_policy="keep_original",
        )

    assert list(pd.to_datetime(merged.time.values)) == list(
        pd.to_datetime(
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
        )
    )
    assert merged["_median"].sel(time="2020-02-01").item() == 2
    assert merged["_median"].sel(time="2020-03-01").item() == 3
    assert merged["_median"].sel(time="2020-04-01").item() == 40


def test_use_new_policy_replaces_overlapping_boundary():
    base = _tiny_l3(["2020-01-01", "2020-02-01", "2020-03-01"], [1, 2, 3])
    recomputed = _tiny_l3(
        ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        [2.1, 30, 40, 50],
    )

    with pytest.warns(RuntimeWarning, match="does not match exactly"):
        merged = l3._merge_l3_extension_segments(
            base,
            recomputed,
            overlap_time_steps=2,
            overlap_policy="use_new",
        )

    assert merged["_median"].sel(time="2020-02-01").item() == 2.1
    assert merged["_median"].sel(time="2020-03-01").item() == 30
    assert merged["_median"].sel(time="2020-04-01").item() == 40


def test_mixed_policy_keeps_original_earliest_overlap_when_almost_matching():
    base = _tiny_l3(["2020-01-01", "2020-02-01", "2020-03-01"], [1, 2, 3])
    recomputed = _tiny_l3(
        ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        [2.0000001, 30, 40, 50],
    )

    with pytest.warns(RuntimeWarning, match="does not match exactly"):
        merged = l3._merge_l3_extension_segments(
            base,
            recomputed,
            overlap_time_steps=2,
            overlap_policy="mixed",
            overlap_rtol=1e-5,
            overlap_atol=1e-8,
        )

    assert merged["_median"].sel(time="2020-02-01").item() == 2
    assert merged["_median"].sel(time="2020-03-01").item() == 30
    assert merged["_median"].sel(time="2020-04-01").item() == 40


def test_abort_policy_raises_on_mismatch():
    base = _tiny_l3(["2020-01-01", "2020-02-01", "2020-03-01"], [1, 2, 3])
    recomputed = _tiny_l3(
        ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        [2.1, 30, 40, 50],
    )

    with pytest.warns(RuntimeWarning, match="does not match exactly"):
        with pytest.raises(RuntimeError, match="full reprocessing"):
            l3._merge_l3_extension_segments(
                base,
                recomputed,
                overlap_time_steps=2,
                overlap_policy="abort",
            )
