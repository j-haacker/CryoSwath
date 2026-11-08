import numpy as np
import pytest
import xarray as xr

import cryoswath.l1b as l1b
from cryoswath.l1b import noise_val
from cryoswath.test_plots import waveform as waveform_plot


@pytest.mark.filterwarnings("error:Precision loss occurred:RuntimeWarning")
@pytest.mark.parametrize("baseline", [0.0, 100.0, 1e12])
def test_noise_val_constant_segments(baseline):
    assert noise_val(np.full(90, baseline)) == baseline
    assert (
        noise_val(np.r_[np.full(30, baseline), np.full(60, baseline + 1)]) == baseline
    )


@pytest.mark.filterwarnings("error:Precision loss occurred:RuntimeWarning")
def test_noise_val_nearly_constant_segments():
    noise = 1e12 + np.tile([-0.001, 0.0, 0.001], 10)
    assert noise_val(np.r_[noise, noise + 1, noise + 1]) == np.mean(noise)


def test_noise_val():
    n = 30  # noise_val considers slices with 30 sample thickness
    test_vec_len = 256
    # model thermal noise approximately using normal distribution
    np.random.seed(0)
    test_vec__pure_noise = np.random.normal(size=(test_vec_len,))
    test_vec__linear_trend = test_vec__pure_noise + np.linspace(0, 1, test_vec_len)
    # distance between noise and signal are +- 10 standard deviabtions
    test_vec__step_start_3rd_slice = test_vec__pure_noise + 10 * (
        np.arange(test_vec_len) >= 2 * n
    )
    assert noise_val(test_vec__pure_noise) == np.mean(test_vec__pure_noise)
    assert noise_val(test_vec__linear_trend) == np.mean(test_vec__linear_trend)
    assert noise_val(test_vec__step_start_3rd_slice) == np.mean(
        test_vec__step_start_3rd_slice[: 2 * n]
    )
    # edge cases
    test_vec__step_start_2nd_slice = test_vec__pure_noise + 10 * (
        np.arange(test_vec_len) >= n
    )
    assert noise_val(test_vec__step_start_2nd_slice) == np.mean(
        test_vec__step_start_2nd_slice[:n]
    )
    test_vec__step_mid_2nd_slice = test_vec__pure_noise + 10 * (
        np.arange(test_vec_len) >= 1.5 * n
    )
    assert noise_val(test_vec__step_mid_2nd_slice) <= np.mean(
        test_vec__step_mid_2nd_slice[: 2 * n]
    )
    test_vec__exp_increase = test_vec__pure_noise + 10 ** (
        np.linspace(0, 1, test_vec_len)
    )
    assert noise_val(test_vec__exp_increase) < np.mean(test_vec__exp_increase)


def test_tag_groups_accelerates_rolling_without_changing_reduction_options(monkeypatch):
    from xarray.computation.rolling import DataArrayRolling

    dims = ("time_20_ku", "ns_20_ku")
    excluded = np.array([[False] * 3 + [True] * 3 + [False] * 6])
    ds = xr.Dataset(
        {
            "exclude_mask": (dims, excluded),
            "power_waveform_20_ku": (dims, np.ones(excluded.shape)),
        },
        coords={"time_20_ku": [0], "ns_20_ku": np.arange(12)},
    )
    monkeypatch.setattr(
        l1b, "get_phase_outlier", lambda _: xr.zeros_like(ds.exclude_mask)
    )
    monkeypatch.setattr(l1b, "get_phase_jump", lambda _: xr.zeros_like(ds.exclude_mask))
    calls = []
    original_reduce = DataArrayRolling._bottleneck_reduce

    def record_reduce(self, func, *args, **kwargs):
        calls.append(func.__name__)
        return original_reduce(self, func, *args, **kwargs)

    monkeypatch.setattr(DataArrayRolling, "_bottleneck_reduce", record_reduce)
    with xr.set_options(use_bottleneck=False, use_numbagg=True):
        result = l1b.tag_groups(ds)
        assert not xr.get_options()["use_bottleneck"]
        assert xr.get_options()["use_numbagg"]

    assert calls == ["move_sum"]
    assert result.group_id.isel(ns_20_ku=slice(3, 6)).isnull().all()
    assert result.group_id.isel(ns_20_ku=1) != result.group_id.isel(ns_20_ku=7)


def test_to_l2_omits_samples_without_a_valid_phase_candidate(monkeypatch):
    phase_wrap_factor = np.arange(-3, 4)
    dims = ("time_20_ku", "ns_20_ku", "phase_wrap_factor")
    elev_diffs = np.full((2, 2, len(phase_wrap_factor)), np.nan)
    elev_diffs[0, 0] = np.arange(len(phase_wrap_factor))
    ds = xr.Dataset(
        data_vars={
            "xph_elev_diffs": (dims, elev_diffs),
            "xph_elevs": (dims, np.ones_like(elev_diffs)),
            "exclude_mask": (("time_20_ku", "ns_20_ku"), np.zeros((2, 2), bool)),
            "group_id": (("time_20_ku", "ns_20_ku"), [[1, 2], [2, np.nan]]),
            "poca_idx": ("time_20_ku", [0, 1]),
        },
        coords={
            "time_20_ku": [0, 1],
            "ns_20_ku": [0, 1],
            "phase_wrap_factor": phase_wrap_factor,
        },
    )

    processed = l1b.append_best_fit_phase_index(ds)

    assert processed.ph_idx.notnull().sum() == 1
    assert processed.ph_idx.isnull().sum() == 3

    captured = []
    monkeypatch.setattr(
        l1b, "l2_from_processed_l1b", lambda data: captured.append(data)
    )

    l1b.to_l2(processed, out_vars=["xph_elevs"], swath_or_poca="both")

    assert len(captured) == 2
    for data in captured:
        assert data.xph_elevs.notnull().sum() == 1


def test_dem_transect_samples_partial_xarray_dem(monkeypatch):
    dem = xr.DataArray(
        np.full((3, 3), 100.0),
        dims=("y", "x"),
        coords={"x": [-1, 0, 1], "y": [-1, 0, 1]},
    ).rio.write_crs(4326)
    waveform = xr.Dataset(
        {
            "lon_20_ku": ("time_20_ku", [0.0]),
            "lat_20_ku": ("time_20_ku", [0.0]),
            "azimuth": ("time_20_ku", [0.0]),
            "xph_dists": (
                ("time_20_ku", "ns_20_ku", "phase_wrap_factor"),
                [[[0.0], [100.0]]],
            ),
            "xph_elevs": (
                ("time_20_ku", "ns_20_ku", "phase_wrap_factor"),
                [[[1.0], [2.0]]],
            ),
            "ph_idx": (("time_20_ku", "ns_20_ku"), [[0, 0]]),
            "exclude_mask": (("time_20_ku", "ns_20_ku"), [[False, False]]),
            "poca_idx": ("time_20_ku", [0]),
        },
        coords={
            "time_20_ku": [np.datetime64("2020-01-01")],
            "ns_20_ku": [0, 1],
            "phase_wrap_factor": [0],
        },
    )
    monkeypatch.setattr(waveform_plot, "get_dem_reader", lambda _: dem)

    ax = waveform_plot.dem_transect(waveform)

    assert "DEM" in ax.get_legend_handles_labels()[1]
    waveform_plot.plt.close(ax.figure)


test_noise_val()
