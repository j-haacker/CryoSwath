import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

import cryoswath.l2 as l2


def test_detect_available_cores_prefers_affinity(monkeypatch):
    monkeypatch.setattr(
        l2.os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False
    )
    monkeypatch.setattr(l2.os, "cpu_count", lambda: 8)
    assert l2._detect_available_cores() == 3


def test_detect_available_cores_falls_back_to_cpu_count(monkeypatch):
    def _raise_attribute_error(_pid):
        raise AttributeError("no affinity support")

    monkeypatch.setattr(
        l2.os, "sched_getaffinity", _raise_attribute_error, raising=False
    )
    monkeypatch.setattr(l2.os, "cpu_count", lambda: 6)
    assert l2._detect_available_cores() == 6


def test_detect_available_cores_warns_and_defaults_to_one(monkeypatch):
    def _raise_not_implemented(_pid):
        raise NotImplementedError("unsupported")

    monkeypatch.setattr(
        l2.os, "sched_getaffinity", _raise_not_implemented, raising=False
    )
    monkeypatch.setattr(l2.os, "cpu_count", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cores = l2._detect_available_cores()
    assert cores == 1
    assert any("Failed to find number of CPU cores" in str(w.message) for w in caught)


def test_from_processed_l1b_handles_removed_time_rows():
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {
            "time": ("time_20_ku", times),
            "x": (("time_20_ku", "ns_20_ku"), np.zeros((3, 2))),
            "y": (("time_20_ku", "ns_20_ku"), np.ones((3, 2))),
            "height": (
                ("time_20_ku", "ns_20_ku"),
                [[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0]],
            ),
        },
        coords={"time_20_ku": [0, 1, 2], "ns_20_ku": [0, 1]},
        attrs={"CRS": CRS.from_epsg(3413)},
    )

    result = l2.from_processed_l1b(ds)

    assert len(result) == 4
    assert result.index.get_level_values("time").nunique() == 2
