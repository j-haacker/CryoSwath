import numpy as np
import xarray as xr

import cryoswath.misc as misc


def _fixed_elev_band_edges(_elevations):
    return np.array([0, 50, 100, 150, 200], dtype=float)


def _retreat_zone_dataset(median_values):
    ref_elev = np.array(
        [
            10,
            20,
            30,
            40,
            45,
            60,
            70,
            80,
            90,
            95,
            110,
            120,
            130,
            140,
            145,
            160,
            170,
            180,
            190,
            195,
        ],
        dtype=float,
    )
    median_values = np.asarray(median_values, dtype=float)
    return xr.Dataset(
        {
            "_median": ("cell", median_values),
            "_count": ("cell", np.arange(median_values.size, dtype=float)),
        },
        coords={"ref_elev": ("cell", ref_elev)},
    )


def test_discard_frontal_retreat_zone_avoids_intervalindex_idxmax_regression(
    monkeypatch,
):
    monkeypatch.setattr(misc, "define_elev_band_edges", _fixed_elev_band_edges)
    ds = _retreat_zone_dataset(
        [
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )

    out = misc.discard_frontal_retreat_zone(
        ds.copy(deep=True),
        replace_vars=["_median", "_count"],
        mode="trend",
        threshold=1,
    )

    expected = ds.copy(deep=True)
    expected["_median"] = xr.where(expected["ref_elev"] < 50, np.nan, expected["_median"])
    expected["_count"] = xr.where(expected["ref_elev"] < 50, np.nan, expected["_count"])

    xr.testing.assert_identical(out, expected)


def test_discard_frontal_retreat_zone_returns_input_when_front_mask_has_no_true_bins(
    monkeypatch,
):
    monkeypatch.setattr(misc, "define_elev_band_edges", _fixed_elev_band_edges)
    ds = _retreat_zone_dataset(np.zeros(20, dtype=float))

    out = misc.discard_frontal_retreat_zone(
        ds.copy(deep=True),
        replace_vars=["_median", "_count"],
        mode="trend",
        threshold=-1,
    )

    xr.testing.assert_identical(out, ds)
