from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cryoswath.l4 as l4
import cryoswath.misc as misc


def _hypsometry_dataset():
    ny, nx = 12, 12
    y = np.arange(ny)
    x = np.arange(nx)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    elev = xx * 45 + yy * 8.0
    values = 0.01 * elev + 0.00002 * elev**2 + np.sin(xx / 2) * 0.05
    errors = np.ones_like(values) * 0.5
    missing = (xx + yy) % 7 == 0
    values = values.astype(float)
    errors = errors.astype(float)
    values[missing] = np.nan
    errors[missing] = np.nan
    return xr.Dataset(
        {
            "h": (("y", "x"), values),
            "h_std": (("y", "x"), errors),
            "ref_elev": (("y", "x"), elev),
        },
        coords={"x": x, "y": y},
    ).stack(stacked_x_y=["x", "y"])


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
    return xr.Dataset(
        {
            "_median": ("cell", np.asarray(median_values, dtype=float)),
            "_count": ("cell", np.arange(len(ref_elev), dtype=float)),
        },
        coords={"ref_elev": ("cell", ref_elev)},
    )


def _timeseries_dataset():
    time = pd.date_range("2020-01-01", periods=36, freq="MS")
    x = np.arange(4)
    y = np.arange(4)
    trend = np.linspace(0, 1, len(time))[:, None, None]
    grid = np.zeros((len(time), len(y), len(x)), dtype=float) + trend
    return xr.Dataset(
        {
            "_median": (("time", "y", "x"), grid),
            "_iqr": (("time", "y", "x"), np.ones_like(grid)),
            "filled_flag": (("time", "y", "x"), np.zeros_like(grid)),
        },
        coords={"time": time, "x": x, "y": y},
    )


def test_interpolate_hypsometrically_default_preserves_behavior():
    ds = _hypsometry_dataset()

    out = misc.interpolate_hypsometrically(ds, "h", "h_std")

    assert int(out.h.isnull().sum()) == 0


def test_interpolate_hypsometrically_emits_stage_diagnostics():
    ds = _hypsometry_dataset()
    events = []

    misc.interpolate_hypsometrically(
        ds,
        "h",
        "h_std",
        outlier_replace=True,
        diagnostic_hook=lambda name, payload: events.append((name, payload)),
    )

    assert [name for name, _ in events] == [
        "hypsometry.outlier_neighbour_check",
        "hypsometry.model_preview",
        "hypsometry.local_deviation",
        "hypsometry.fit_fill_mask",
    ]
    payload = events[-1][1]
    assert {
        "ds",
        "main_var",
        "elev",
        "fill_mask",
        "modelled",
        "x_vals",
        "elev_bin_means",
        "elev_bin_errs",
        "neighbour_std",
        "diagnostic_context",
    }.issubset(payload)


def test_diagnostic_hook_exceptions_propagate():
    def raise_from_hook(_name, _payload):
        raise RuntimeError("diagnostic failed")

    with pytest.raises(RuntimeError, match="diagnostic failed"):
        misc.interpolate_hypsometrically(
            _hypsometry_dataset(),
            "h",
            "h_std",
            diagnostic_hook=raise_from_hook,
        )


def test_discard_frontal_retreat_zone_emits_threshold_event(monkeypatch):
    monkeypatch.setattr(misc, "define_elev_band_edges", _fixed_elev_band_edges)
    ds = _retreat_zone_dataset(
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    )
    events = []

    misc.discard_frontal_retreat_zone(
        ds,
        replace_vars=["_median", "_count"],
        mode="trend",
        threshold=1,
        diagnostic_hook=lambda name, payload: events.append((name, payload)),
    )

    assert [name for name, _ in events] == ["frontal_retreat_zone.threshold"]
    assert events[0][1]["front_bin"].left == 50


def test_discard_frontal_retreat_zone_skips_hook_without_front(monkeypatch):
    monkeypatch.setattr(misc, "define_elev_band_edges", _fixed_elev_band_edges)
    events = []

    misc.discard_frontal_retreat_zone(
        _retreat_zone_dataset(np.zeros(20, dtype=float)),
        replace_vars=["_median", "_count"],
        mode="trend",
        threshold=-1,
        diagnostic_hook=lambda name, payload: events.append((name, payload)),
    )

    assert events == []


def test_diagnostic_hook_with_context_preserves_event_name_and_adds_context():
    events = []
    hook = l4._diagnostic_hook_with_context(
        lambda name, payload: events.append((name, payload)),
        source="fill_voids",
        stage="basin",
        group_label=7,
    )

    hook("hypsometry.fit_fill_mask", {"diagnostic_context": {"time": "2020-01"}})

    assert events[0][0] == "hypsometry.fit_fill_mask"
    assert events[0][1]["diagnostic_context"] == {
        "source": "fill_voids",
        "stage": "basin",
        "group_label": 7,
        "time": "2020-01",
    }


def test_timeseries_from_gridded_emits_result_and_writes_no_debug_png(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    events = []

    results = l4.timeseries_from_gridded(
        _timeseries_dataset(),
        diagnostic_hook=lambda name, payload: events.append((name, payload)),
    )

    assert [name for name, _ in events] == ["timeseries_from_gridded.result"]
    assert events[0][1]["results"].equals(results)
    assert not list(tmp_path.glob("tmp__quick_view_elev_ts_with_unc__*.png"))
    assert capsys.readouterr().out == ""
