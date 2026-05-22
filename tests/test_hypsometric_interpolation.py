import warnings

import numpy as np
import xarray as xr

import cryoswath.misc as misc


def test_interpolate_hypsometrically_avoids_multiindex_drop_warning():
    x = np.arange(6)
    y = np.arange(6)
    ref_elev = (x[:, None] * 10 + y[None, :]).astype(float)
    values = ref_elev * 0.1 + 5
    values[1, 1] = np.nan
    values[2, 3] = np.nan
    values[4, 4] = np.nan
    ds = xr.Dataset(
        {
            "dh": (("x", "y"), values),
            "dh_std": (("x", "y"), np.ones_like(values)),
            "ref_elev": (("x", "y"), ref_elev),
        },
        coords={"x": x, "y": y},
    ).stack(stacked_x_y=["x", "y"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Deleting a single level of a MultiIndex is deprecated.*",
            category=Warning,
        )
        result = misc.interpolate_hypsometrically(ds, "dh", "dh_std")

    assert "stacked_x_y" in result.dims
    assert result["dh"].sizes["stacked_x_y"] == ds["dh"].sizes["stacked_x_y"]
