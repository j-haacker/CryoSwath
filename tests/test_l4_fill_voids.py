import numpy as np
import xarray as xr

import cryoswath.l4 as l4


def test_fill_voids_uses_existing_groupers_without_basin_shapes(monkeypatch):
    x = np.arange(3)
    y = np.arange(3)
    values = np.ones((3, 3), dtype=float)
    values[0, 0] = np.nan
    values[1, 1] = np.nan
    basin_id = np.ones((3, 3), dtype=float)
    basin_id[1, 1] = np.nan
    group_id = np.ones((3, 3), dtype=float)
    group_id[1, 1] = np.nan
    ds = xr.Dataset(
        {
            "dh": (("x", "y"), values),
            "dh_std": (("x", "y"), np.ones_like(values)),
            "ref_elev": (("x", "y"), np.arange(9).reshape(3, 3).astype(float)),
            "basin_id": (("x", "y"), basin_id),
            "group_id": (("x", "y"), group_id),
        },
        coords={"x": x, "y": y},
    )

    def fail(*args, **kwargs):
        raise AssertionError("shape loading or ID assignment should not be called")

    def fake_interpolate(ds, main_var, *args, **kwargs):
        return ds.assign({main_var: ds[main_var].fillna(99)})

    monkeypatch.setattr(l4, "find_region_id", fail)
    monkeypatch.setattr(l4, "load_glacier_outlines", fail)
    monkeypatch.setattr(l4, "append_basin_id", fail)
    monkeypatch.setattr(l4, "append_basin_group", fail)
    monkeypatch.setattr(l4, "interpolate_hypsometrically", fake_interpolate)

    result = l4.fill_voids(
        ds,
        "dh",
        "dh_std",
        per=("basin", "basin_group"),
        basin_shapes=None,
        discard_deglaciated=False,
    )

    assert result["dh"].sel(x=0, y=0) == 99
    assert np.isnan(result["dh"].sel(x=1, y=1))
