"""Value regressions for native-range association and actual EOLIS schemas."""

from contextlib import contextmanager
from decimal import Decimal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

from cryoswath import eolis, l1b, misc


def raw_track(n=1):
    times = pd.date_range("2024-02-15T19:54:22", periods=n, freq="50ms")
    ds = xr.Dataset(
        {
            "lon_20_ku": ("time_20_ku", np.zeros(n)),
            "lat_20_ku": ("time_20_ku", np.zeros(n)),
            "alt_20_ku": ("time_20_ku", np.full(n, 700000.0)),
            "sat_vel_vec_20_ku": (
                ("time_20_ku", "space_3d"),
                np.tile([0.0, 0.0, 7500.0], (n, 1)),
            ),
            "window_del_20_ku": (
                "time_20_ku",
                np.full(n, 1400000 / misc.speed_of_light),
            ),
            **{
                name: ("time_cor_01", np.zeros(n))
                for name in [
                    "mod_dry_tropo_cor_01",
                    "mod_wet_tropo_cor_01",
                    "iono_cor_gim_01",
                    "pole_tide_01",
                    "solid_earth_tide_01",
                    "load_tide_01",
                ]
            },
        },
        coords={
            "time_20_ku": times,
            "time_cor_01": times,
            "space_3d": [0, 1, 2],
        },
        attrs={
            "product_name": "CS_OFFL_SIR_SIN_1B_20240215T195335_20240215T195420_E001"
        },
    )
    own = gpd.GeoDataFrame(
        {"height": np.zeros(n)},
        geometry=gpd.points_from_xy(np.zeros(n), np.zeros(n)),
        crs=4326,
        index=pd.MultiIndex.from_arrays(
            [times.tz_localize("UTC"), np.full(n, misc.reference_bin)],
            names=["time", "sample"],
        ),
    )
    ref = gpd.GeoDataFrame(
        {"row": [10], "elevation": [0.0], "eolis_file": ["tile.nc"]},
        geometry=[shapely.Point(0, 0)],
        crs=4326,
    )
    return ds, own, ref


def test_radial_identity_and_no_height_or_across_gate():
    raw, own, ref = raw_track()
    # On the equator, solve for ellipsoidal height giving exactly 700 km slant
    # range at a different longitude. It is >2 km across track and >3 m higher.
    angle = np.deg2rad(0.02)
    sat_radius = misc.WGS84_ellpsoid.a + 700000
    radius = sat_radius * np.cos(angle) - np.sqrt(
        700000**2 - sat_radius**2 * np.sin(angle) ** 2
    )
    ref["elevation"] = radius - misc.WGS84_ellpsoid.a
    ref.geometry = [shapely.Point(0.02, 0)]
    result = eolis._associate(raw, own, ref)
    point = result["points"].iloc[0]
    assert point.matched and point["sample"] == misc.reference_bin
    assert abs(point.radial_difference_m) < 1e-8
    assert abs(point.across_difference_m) > 2000
    assert abs(point.height_difference_m) > 3
    assert result["statistics"].loc["height_difference_m", "mean"] == pytest.approx(
        -ref.elevation.iloc[0]
    )
    assert result["points"].attrs["waveform_time_scale"] == "TAI"


def test_missing_sample_is_not_replaced():
    raw, own, ref = raw_track()
    own.index = own.index.set_levels([misc.reference_bin + 1], level="sample")
    result = eolis._associate(raw, own, ref)
    assert result["coverage"].missing_cached_samples == 1
    assert not result["points"].matched.any()
    assert result["points"]["sample"].iloc[0] == misc.reference_bin
    assert result["statistics"].loc["height_difference_m", "count"] == 0


def test_along_gate_and_empty_reference():
    raw, own, ref = raw_track()
    ref.geometry = [shapely.Point(0, 0.002)]  # About 221 m along northward flight.
    result = eolis._associate(raw, own, ref)
    assert result["coverage"].selected_observations == 0
    assert eolis._associate(raw, own, ref.iloc[:0])["points"].empty


def test_multiple_waveforms_raise_actionable_error():
    raw, own, ref = raw_track(2)
    with pytest.raises(ValueError, match="GitHub issue") as error:
        eolis._associate(raw, own, ref)
    assert "20240215T195335" in str(error.value)
    assert "2024-02-15" in str(error.value)
    assert str(misc.reference_bin) in str(error.value)


def test_full_precision_reference_range_and_legacy_timedelta(tmp_path):
    ds, _, _ = raw_track()
    value = 0.004851310927
    ds["window_del_20_ku"][:] = value
    ds.window_del_20_ku.attrs["units"] = "seconds"
    ds["other_delay"] = xr.DataArray(
        [0.1], dims="time_20_ku", attrs={"units": "seconds"}
    )
    path = tmp_path / "raw.nc"
    ds.to_netcdf(path)
    expected = float(Decimal("0.004851310927") * Decimal(299792458) / 2)
    with xr.open_dataset(path, decode_cf=False) as encoded:
        opened = xr.decode_cf(
            encoded,
            decode_timedelta={
                name: name != "window_del_20_ku" for name in encoded.variables
            },
        )
        assert np.issubdtype(opened.window_del_20_ku.dtype, np.floating)
        assert np.issubdtype(opened.other_delay.dtype, np.timedelta64)
        assert l1b.ref_range(opened).item() == pytest.approx(expected, abs=1e-9, rel=0)
    with xr.open_dataset(path, decode_timedelta=True) as opened:
        seconds = opened.window_del_20_ku.item() / 1e9
        assert l1b.ref_range(opened).item() == pytest.approx(
            seconds * misc.speed_of_light / 2
        )


def test_region_finder_all_matches_preserves_default(monkeypatch):
    regions = gpd.GeoDataFrame(
        {"o1region": ["06", "08", "08"], "o2region": ["06-01", "08-01", "08-02"]},
        geometry=[
            shapely.box(-20, 60, -10, 70),
            shapely.box(10, 60, 20, 70),
            shapely.box(20, 60, 30, 70),
        ],
        crs=4326,
    )
    monkeypatch.setattr(misc.gpd, "read_feather", lambda _: regions)
    assert misc.find_region_id((65, -16), scope="o1") == "06"
    trail = shapely.MultiPoint([(-16, 65), (15, 65), (25, 65)])
    assert misc.find_region_id(trail, scope="o1", all_matches=True) == ["06", "08"]
    assert misc.find_region_id(trail.envelope, scope="o1", all_matches=True) == [
        "06",
        "08",
    ]
    points = gpd.GeoSeries(list(trail.geoms), crs=4326)
    assert misc.find_region_id(points, scope="o1", all_matches=True) == ["06", "08"]


def test_real_schema_source_filter_and_exact_overlap_dedup(tmp_path):
    paths = []
    for name in ["tile1.nc", "tile2.nc"]:
        path = tmp_path / name
        ds = xr.Dataset(
            {
                "x": ("row", [1300000.0, 1300001.0, 1300002.0]),
                "y": ("row", [-2500000.0] * 3),
                "elevation": ("row", [1000.0] * 3),
                "input_file_id": ("row", [-7, -7, 42]),
                "is_swath": ("row", [1, 0, 1]),
                "time": ("row", np.array([1707998015] * 3, dtype="int32")),
            },
            coords={"row": [1, 2, 3]},
            attrs={
                "geospatial_projection": "EPSG:3413",
                "fileids": ["-7 : CS_OFFL_20240215T195335.nc", "42 : other_pass.nc"],
            },
        )
        ds.time.attrs["units"] = "Seconds from 1970 in the UTC timezone."
        ds.to_netcdf(path)
        paths.append(path)
    points, removed = eolis._read_points(paths, "20240215T195335")
    assert len(points) == 1 and removed == 1
    assert points.row.iloc[0] == 1
    assert points.time.iloc[0] == pd.Timestamp("2024-02-15T11:53:35Z")
    assert points.crs.to_epsg() == 4326


class FTP:
    def __init__(self, names):
        self.names, self.transfers, self.folders = names, [], []

    def nlst(self, folder):
        self.folders.append(folder)
        return [folder + "/" + n for n in self.names]

    def retrbinary(self, command, write):
        self.transfers.append(command)
        write(b"complete")


def test_chunk_selection_version_cache_and_padded_index_path(monkeypatch, tmp_path):
    names = [
        "CS_OFFL_THEM_POINT_ICELAND___2020_01_0_0_V301.nc",
        "CS_OFFL_THEM_POINT_ICELAND___2020_01_0_2000000_V301.nc",
        "index.tgz",
    ]
    ftp = FTP(names)

    @contextmanager
    def connection():
        yield ftp

    monkeypatch.setattr(eolis.misc, "ftp_cs2_server", connection)
    crs = "+proj=aeqd +lat_0=65 +lon_0=-16 +datum=WGS84 +units=m"
    tiles = gpd.GeoDataFrame(
        {"pdgs_path": [f"/2020/01/ICELAND__/{n}" for n in names[:2]]},
        geometry=[
            shapely.box(-50000, -50000, 50000, 50000),
            shapely.box(-50000, 2000000, 50000, 2100000),
        ],
        crs=crs,
    )
    monkeypatch.setattr(eolis.gpd, "read_file", lambda _: tiles)
    ground = gpd.GeoDataFrame(
        geometry=[shapely.Point(-16, 65), shapely.Point(-16, 65.01)], crs=4326
    )
    filename = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000100_E001.nc"
    for _ in range(2):
        paths = eolis._retrieve(filename, ["ICELAND"], ground, tmp_path, None, 170)
        assert [p.name for p in paths] == names[:1]
    assert len(ftp.transfers) == 2
    assert ftp.folders == ["/TEMPO_SWATH_POINT/2020/01/ICELAND"] * 2
    assert all("/ICELAND__/" not in command for command in ftp.transfers)
    ftp.names.append(names[0].replace("V301", "V302"))
    with pytest.raises(ValueError, match="Multiple EOLIS versions"):
        eolis._retrieve(filename, ["ICELAND"], ground, tmp_path, None, 170)


def test_failed_transfer_leaves_no_partial_file(tmp_path):
    class BrokenFTP:
        def retrbinary(self, command, write):
            write(b"partial")
            raise ConnectionError("interrupted")

    with pytest.raises(ConnectionError):
        eolis._download(BrokenFTP(), "/folder/product.nc", tmp_path)
    assert list((tmp_path / "folder").iterdir()) == []


def test_cached_l1b_resolver_does_not_invoke_downloader(monkeypatch, tmp_path):
    folder = tmp_path / "2024/02"
    folder.mkdir(parents=True)
    path = folder / "CS_OFFL_SIR_SIN_1B_20240215T195335_20240215T195420_E001.nc"
    path.touch()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "download_single_file", lambda _: pytest.fail("cached file downloaded")
    )
    assert l1b._resolve_l1b_file("20240215T195335") == str(path)


def test_compare_track_reuses_processing_and_reports_unsupported_region(
    monkeypatch, tmp_path
):
    raw, own, _ = raw_track()
    filename = tmp_path / (raw.attrs["product_name"] + ".nc")
    raw.to_netcdf(filename)
    tile = tmp_path / "tile.nc"
    ds = xr.Dataset(
        {
            "lon": ("row", [0.0]),
            "lat": ("row", [0.0]),
            "elevation": ("row", [0.0]),
            "input_file_id": ("row", [7]),
            "is_swath": ("row", [1]),
            "time": ("row", [np.datetime64("2024-02-15")]),
        },
        coords={"row": [42]},
        attrs={"fileids": "7 : CS_OFFL_20240215T195335.nc"},
    )
    ds.to_netcdf(tile)
    calls = []
    monkeypatch.setattr(l1b, "_resolve_l1b_file", lambda _: str(filename))

    def process(track, **kwargs):
        calls.append(kwargs)
        return own, None

    monkeypatch.setattr(eolis.l2, "from_id", process)

    def find(location, **kwargs):
        assert location.geom_type == "MultiPoint"
        assert kwargs == {"scope": "o1", "all_matches": True}
        return ["06", "10"]

    monkeypatch.setattr(misc, "find_region_id", find)

    def retrieve(filename, regions, ground, cache, version, along_track):
        assert regions == ["ICELAND"]
        return [tile]

    monkeypatch.setattr(eolis, "_retrieve", retrieve)
    result = eolis.compare_track("20240215T195335", coherence_threshold=0.8)
    assert result["coverage"].matched_observations == 1
    assert result["points"].attrs["unsupported_rgi_regions"] == ["10"]
    assert calls == [
        {
            "reprocess": False,
            "save_or_return": "both",
            "cores": 1,
            "coherence_threshold": 0.8,
        }
    ]


def test_delay_precision_shifts_whole_waveform_not_bin_spacing():
    raw, _, _ = raw_track()
    raw["window_del_20_ku"][:] = 0.004851310927
    raw = raw.assign_coords(
        ns_20_ku=[misc.reference_bin, misc.reference_bin + 1], phase_wrap_factor=[0]
    )
    raw["ph_diff"] = xr.DataArray([[0.0, 0.0]], dims=("time_20_ku", "ns_20_ku"))
    raw["off_nadir_roll_angle_str_20_ku"] = xr.DataArray([0.0], dims="time_20_ku")
    raw["azimuth"] = xr.DataArray([0.0], dims="time_20_ku")
    legacy = raw.copy(deep=True)
    legacy["window_del_20_ku"] = xr.DataArray(
        np.array([4851310], dtype="timedelta64[ns]"), dims="time_20_ku"
    )
    precise = l1b.locate_ambiguous_origin(raw).xph_elevs.values.ravel()
    quantized = l1b.locate_ambiguous_origin(legacy).xph_elevs.values.ravel()
    expected_shift = float(Decimal("0.000000000927") * Decimal(299792458) / 2)
    np.testing.assert_allclose(quantized - precise, expected_shift, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.diff(precise), -misc.sample_width, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        np.diff(quantized), -misc.sample_width, rtol=0, atol=2e-6
    )
