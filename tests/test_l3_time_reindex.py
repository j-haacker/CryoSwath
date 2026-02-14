import numpy as np
import pandas as pd
import xarray as xr

import cryoswath.l3 as l3


def _tiny_dataset(times):
    return xr.Dataset(
        data_vars={"_median": (("time", "x", "y"), np.arange(len(times), dtype=float)[:, None, None])},
        coords={"time": pd.DatetimeIndex(times), "x": [0], "y": [0]},
    )


def test_non_contiguous_monthly_time_is_filled():
    ds = _tiny_dataset(["2020-01-01", "2020-03-01"])
    out = l3._ensure_contiguous_time_coord(ds, timestep_months=1)
    assert list(pd.to_datetime(out.time.values)) == list(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    )
    assert np.isnan(out["_median"].sel(time="2020-02-01").item())


def test_already_contiguous_monthly_time_is_unchanged():
    ds = _tiny_dataset(pd.date_range("2020-01-01", "2020-03-01", freq="MS"))
    out = l3._ensure_contiguous_time_coord(ds, timestep_months=1)
    xr.testing.assert_identical(out, ds)


def test_single_timestep_is_unchanged():
    ds = _tiny_dataset(["2020-01-01"])
    out = l3._ensure_contiguous_time_coord(ds, timestep_months=1)
    xr.testing.assert_identical(out, ds)
