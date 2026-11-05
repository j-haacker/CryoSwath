import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pystac
import pytest
import rasterio
import xarray as xr
from pyproj import Geod
from rasterio.transform import from_origin

import cryoswath.l1b as l1b
import cryoswath.l4 as l4
import cryoswath.misc as misc


def _write_tar_with_single_file(dest: Path, arcname: str) -> None:
    source = dest.parent / ".source"
    source.mkdir(exist_ok=True)
    payload = source / Path(arcname).name
    payload.write_bytes(b"dem")
    with tarfile.open(dest, mode="w:gz") as archive:
        archive.add(payload, arcname=arcname)


def test_get_dem_reader_auto_downloads_missing_arcticdem(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    calls = []

    def fake_download_file(url, dest, auth=None, timeout=120):
        calls.append((url, Path(dest), auth, timeout))
        _write_tar_with_single_file(
            Path(dest),
            "arcticdem_mosaic_100m_v4.1_dem.tif",
        )
        return str(dest)

    monkeypatch.setattr(misc, "download_file", fake_download_file)
    monkeypatch.setattr(misc.rasterio, "open", lambda path: ("reader", Path(path).name))

    with pytest.warns(UserWarning, match="Full DEM archive download requested"):
        out = misc.get_dem_reader(80, missing_dem="full")

    assert out == ("reader", "arcticdem_mosaic_100m_v4.1_dem.tif")
    assert calls[0][0] == misc._ARCTICDEM_100M_V41_ARCHIVE_URL
    assert calls[0][2] is None
    assert calls[0][3] == 120
    assert (tmp_path / "arcticdem_mosaic_100m_v4.1_dem.tif").is_file()
    assert not (tmp_path / "arcticdem_mosaic_100m_v4.1.tar.gz").exists()


def test_get_dem_reader_auto_downloads_missing_rema(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    calls = []

    def fake_download_file(url, dest, auth=None, timeout=120):
        calls.append((url, Path(dest), auth, timeout))
        _write_tar_with_single_file(
            Path(dest),
            "rema_mosaic_100m_v2.0_filled_cop30_dem.tif",
        )
        return str(dest)

    monkeypatch.setattr(misc, "download_file", fake_download_file)
    monkeypatch.setattr(misc.rasterio, "open", lambda path: ("reader", Path(path).name))

    with pytest.warns(UserWarning, match="Full DEM archive download requested"):
        out = misc.get_dem_reader(-80, missing_dem="full")

    assert out == ("reader", "rema_mosaic_100m_v2.0_filled_cop30_dem.tif")
    assert calls[0][0] == misc._REMA_100M_V20_FILLED_COP30_ARCHIVE_URL
    assert calls[0][2] is None
    assert calls[0][3] == 120
    assert (tmp_path / "rema_mosaic_100m_v2.0_filled_cop30_dem.tif").is_file()
    assert not (tmp_path / "rema_mosaic_100m_v2.0_filled_cop30.tar.gz").exists()


def test_get_dem_reader_raises_when_auto_download_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    monkeypatch.setattr(
        misc,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    monkeypatch.setattr(misc.sys, "stdin", io.StringIO(""))

    with pytest.warns(UserWarning, match="Automatic DEM download failed"):
        with pytest.raises(
            FileNotFoundError, match="Automatic download was unsuccessful"
        ):
            misc.get_dem_reader(80, missing_dem="full")


def test_get_dem_reader_uses_existing_default_dem_without_download(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    existing_dem = tmp_path / "arcticdem_mosaic_100m_v4.1_dem.tif"
    existing_dem.write_bytes(b"present")
    monkeypatch.setattr(
        misc,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download_file should not be called")
        ),
    )
    monkeypatch.setattr(misc.rasterio, "open", lambda path: ("reader", Path(path).name))

    out = misc.get_dem_reader(80)
    assert out == ("reader", "arcticdem_mosaic_100m_v4.1_dem.tif")


def test_get_dem_reader_targeted_provisions_spatial_input(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    captured = []
    monkeypatch.setattr(
        misc,
        "download_dem",
        lambda geometry: captured.append(geometry) or tmp_path / "target.zarr",
    )
    reader = xr.Dataset({"dem": xr.DataArray(1)})
    monkeypatch.setattr(misc.xr, "open_dataset", lambda path, **kwargs: reader)

    out = misc.get_dem_reader(misc.shapely.Point(10, 80))

    assert out.identical(reader.dem)
    assert len(captured) == 1
    assert captured[0].crs == "EPSG:4326"


def test_get_dem_reader_targeted_rejects_extent_free_input(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    monkeypatch.setattr(
        misc,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("targeted provisioning must not download the full archive")
        ),
    )

    with pytest.raises(FileNotFoundError, match="requires a spatial input"):
        misc.get_dem_reader(80)


@pytest.mark.parametrize(
    "spatial_input",
    [
        lambda: misc.shapely.Point(10, 80),
        lambda: misc.gpd.GeoSeries([misc.shapely.box(9, 79, 11, 81)], crs="EPSG:4326"),
        lambda: misc.gpd.GeoDataFrame(
            geometry=[misc.shapely.box(9, 79, 11, 81)], crs="EPSG:4326"
        ),
        lambda: xr.DataArray(
            [[1, 1], [1, 1]],
            dims=("y", "x"),
            coords={"x": [9, 11], "y": [79, 81]},
        ).rio.write_crs(4326),
        lambda: xr.Dataset(
            {"value": (("y", "x"), [[1, 1], [1, 1]])},
            coords={"x": [9, 11], "y": [79, 81]},
        ).rio.write_crs(4326),
    ],
    ids=["shapely", "geoseries", "geodataframe", "dataarray", "dataset"],
)
def test_get_dem_reader_refreshes_existing_regional_cache(
    monkeypatch, tmp_path, spatial_input
):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    cache = tmp_path / "arcticdem-mosaics-v4.1-32m_100m-mean.zarr"
    cache.mkdir()
    reader = xr.Dataset({"dem": xr.DataArray(1)})
    monkeypatch.setattr(misc.xr, "open_dataset", lambda path, **kwargs: reader)
    captured = []
    monkeypatch.setattr(
        misc,
        "download_dem",
        lambda geometry: captured.append(geometry) or cache,
    )

    assert misc.get_dem_reader(spatial_input()).identical(reader.dem)
    assert len(captured) == 1


def test_get_dem_reader_refreshes_partial_cache_for_buffered_l1b_track(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    cache = tmp_path / "arcticdem-mosaics-v4.1-32m_100m-mean.zarr"
    cache.mkdir()
    reader = xr.Dataset({"dem": xr.DataArray(1)})
    captured = []
    track = xr.Dataset(
        {
            "lat_20_ku": ("time_20_ku", [80.0]),
            "lon_20_ku": ("time_20_ku", [10.0]),
        }
    )
    monkeypatch.setattr(
        misc,
        "download_dem",
        lambda geometry: captured.append(geometry) or cache,
    )
    monkeypatch.setattr(misc.xr, "open_dataset", lambda path, **kwargs: reader)

    assert misc.get_dem_reader(track).identical(reader.dem)
    assert len(captured) == 1
    west, south, east, north = captured[0].total_bounds
    assert east - west > 1
    assert north - south > 0.4
    _, _, distance_to_south = Geod(ellps="WGS84").inv(10, 80, 10, south)
    assert distance_to_south == pytest.approx(30_000, rel=0.05)


def test_get_dem_reader_does_not_refresh_full_or_specified_dem(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    full_dem = tmp_path / "arcticdem_mosaic_100m_v4.1_dem.tif"
    full_dem.write_bytes(b"present")
    specified_dem = tmp_path / "specified.tif"
    specified_dem.write_bytes(b"present")
    track = xr.Dataset(
        {
            "lat_20_ku": ("time_20_ku", [80.0]),
            "lon_20_ku": ("time_20_ku", [10.0]),
        }
    )
    monkeypatch.setattr(
        misc,
        "download_dem",
        lambda geometry: (_ for _ in ()).throw(
            AssertionError("DEM provisioning should not run")
        ),
    )
    monkeypatch.setattr(misc.rasterio, "open", lambda path: ("reader", Path(path).name))

    assert misc.get_dem_reader(track) == ("reader", full_dem.name)
    assert misc.get_dem_reader(str(specified_dem)) == ("reader", specified_dem.name)


def test_l1b_forwards_missing_dem(monkeypatch):
    class ReaderRequested(Exception):
        pass

    calls = []
    ds = xr.Dataset(coords={"time_20_ku": [0]})
    spatial_ds = ds.assign(
        xph_lats=xr.DataArray([[80.0]], dims=("time_20_ku", "ns_20_ku")),
        xph_lons=xr.DataArray([[10.0]], dims=("time_20_ku", "ns_20_ku")),
    )
    monkeypatch.setattr(l1b, "locate_ambiguous_origin", lambda _: spatial_ds)

    def fake_get_dem_reader(data, *, missing_dem):
        calls.append((data, missing_dem))
        raise ReaderRequested

    monkeypatch.setattr(l1b, "get_dem_reader", fake_get_dem_reader)

    with pytest.raises(ReaderRequested):
        l1b.append_ambiguous_reference_elevation(ds, missing_dem="full")

    assert calls == [(spatial_ds, "full")]


def test_l4_forwards_missing_dem(monkeypatch):
    class ReaderRequested(Exception):
        pass

    calls = []

    def fake_get_dem_reader(data, *, missing_dem):
        calls.append((data, missing_dem))
        raise ReaderRequested

    monkeypatch.setattr(l4, "get_dem_reader", fake_get_dem_reader)
    ds = xr.Dataset()

    with pytest.raises(ReaderRequested):
        l4.append_elevation_reference(ds, missing_dem="full")

    assert calls == [(ds, "full")]


def test_l4_uses_encoded_fill_value_from_xarray_dem(monkeypatch):
    dem = xr.DataArray(
        [[100.0, -9999.0], [200.0, 300.0]],
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    ).rio.write_crs(4326)
    dem.encoding["_FillValue"] = -9999.0
    target = xr.Dataset(coords={"x": [0.0, 1.0], "y": [1.0, 0.0]}).rio.write_crs(4326)
    monkeypatch.setattr(l4, "get_dem_reader", lambda *args, **kwargs: dem)

    result = l4.append_elevation_reference(target)

    assert np.isnan(result.ref_elev.sel(x=1.0, y=1.0))


@pytest.mark.parametrize("target_crs", [4326, 3857])
@pytest.mark.parametrize("nodata", [-9999.0, np.nan])
def test_l4_reprojects_reference_elevations_with_average(
    monkeypatch, target_crs, nodata
):
    # Three-by-three footprints: a partial void, a full void, and an outlier.
    values = np.array(
        [
            [1, 2, 3, nodata, nodata, nodata],
            [4, nodata, 6, nodata, nodata, nodata],
            [7, 8, 18, nodata, nodata, nodata],
            [10, 10, 10, 20, 20, 20],
            [10, 100, 10, 20, 20, 20],
            [10, 10, 10, 20, 20, 20],
        ],
        dtype=float,
    )
    dem = (
        xr.DataArray(
            values,
            dims=("y", "x"),
            coords={"x": np.arange(6), "y": np.arange(5, -1, -1)},
        )
        .rio.write_crs(4326)
        .rio.write_nodata(nodata)
    )
    bounds = rasterio.warp.transform_bounds(4326, target_crs, *dem.rio.bounds())
    transform = rasterio.transform.from_bounds(*bounds, width=2, height=2)
    target = xr.Dataset(
        coords={
            "x": transform.c + (np.arange(2) + 0.5) * transform.a,
            "y": transform.f + (np.arange(2) + 0.5) * transform.e,
        }
    ).rio.write_crs(target_crs)
    expected = np.full((2, 2), np.nan)
    rasterio.warp.reproject(
        source=values,
        destination=expected,
        src_transform=dem.rio.transform(),
        src_crs=dem.rio.crs,
        src_nodata=nodata,
        dst_transform=transform,
        dst_crs=target_crs,
        dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.average,
    )
    if target_crs == 4326:
        # Target centers coincide with source centers, whose values are wrong
        # without resampling (the lower-left center is 100 rather than 20).
        np.testing.assert_allclose(expected, [[49 / 8, np.nan], [20, 20]])
    assert np.isnan(expected[0, 1])
    monkeypatch.setattr(l4, "get_dem_reader", lambda *args, **kwargs: dem)

    result = l4.append_elevation_reference(target)

    np.testing.assert_allclose(result.ref_elev.values, expected, equal_nan=True)
    assert np.isnan(result.ref_elev.rio.nodata)


@pytest.mark.parametrize("dtype", ["float32", "int16"])
@pytest.mark.filterwarnings("error:invalid value encountered in cast:RuntimeWarning")
def test_read_stac_accepts_static_item_with_numpy_python_scalars(tmp_path, dtype):
    """Exercise stackstac item construction with NumPy Python scalars locally."""
    raster_path = tmp_path / "tile.tif"
    transform = from_origin(10, 80, 0.1, 0.1)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as raster:
        raster.write(np.array([[1, -9999], [3, 4]], dtype=dtype), 1)

    item = pystac.Item(
        id="static-tile",
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[10, 79.8], [10.2, 79.8], [10.2, 80], [10, 80], [10, 79.8]]
            ],
        },
        bbox=[10, 79.8, 10.2, 80],
        datetime=datetime(2020, 1, 1, tzinfo=timezone.utc),
        properties={"proj:code": "EPSG:4326"},
    )
    item.add_asset(
        "dem",
        pystac.Asset(
            href=str(raster_path),
            media_type=pystac.MediaType.COG,
            roles=["data"],
            extra_fields={
                "data_type": dtype,
                "nodata": -9999,
                "proj:shape": [2, 2],
                "proj:transform": list(transform)[:6],
                "raster:bands": [{"data_type": dtype, "nodata": -9999}],
            },
        ),
    )

    result = misc._read_stac(item).compute()

    np.testing.assert_allclose(result["dem"].values[:2, :2], [[1, np.nan], [3, 4]])


@pytest.mark.filterwarnings("error:Behaviour of argmin/argmax:FutureWarning")
def test_chunk_idx_uses_explicit_coordinate_dimension():
    ds = xr.Dataset({"data": ("x", np.zeros(6))}, coords={"x": np.arange(6)}).chunk(x=2)
    assert misc.chunk_idx(ds, "x", [0, 2, 4]) == [0, 1, 2]


@pytest.mark.filterwarnings("error:invalid value encountered in cast:RuntimeWarning")
def test_read_stac_preserves_missing_integer_band_values(monkeypatch):
    stacked = xr.DataArray(
        da.from_array(np.array([[[[1, np.nan], [3, 4]]]]), chunks=(1, 1, 2, 2)),
        dims=("time", "band", "y", "x"),
        coords={
            "time": [0],
            "id": ("time", ["tile"]),
            "band": ["count"],
            "y": [1, 0],
            "x": [0, 1],
            "data_type": ("band", ["int16"]),
            "nodata": ("band", [-9999]),
        },
    )
    monkeypatch.setattr(misc.stackstac, "stack", lambda *args, **kwargs: stacked)
    item = SimpleNamespace(properties={"proj:code": "EPSG:3413"})
    result = misc._read_stac(item).compute()
    np.testing.assert_allclose(result["count"], [[1, np.nan], [3, 4]])
    assert np.isnan(xr.full_like(result, np.nan)["count"]).all()
    target = xr.Dataset(coords={"x": [0.5, 2.5], "y": [0.5, -1.5]}).rio.write_crs(3413)
    averaged = result["count"].rio.reproject_match(
        target, resampling=rasterio.warp.Resampling.average, nodata=np.nan
    )
    np.testing.assert_allclose(averaged, [[8 / 3, np.nan], [np.nan, np.nan]])


def test_download_dem_wraps_pgc_stac_timeout(monkeypatch):
    calls = []

    def fake_open(url, *args, **kwargs):
        calls.append((url, kwargs))
        raise misc.requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(misc.Client, "open", fake_open)

    with pytest.raises(RuntimeError, match="PGC STAC API.*did not respond"):
        misc.download_dem(object())

    assert calls[0][0] == misc._PGC_STAC_API_URL
    assert calls[0][1]["timeout"] == misc._PGC_STAC_TIMEOUT


def test_download_dem_reraises_non_connectivity_pgc_stac_api_error(monkeypatch):
    error = misc.APIError("server returned HTTP 400")

    def fake_open(*args, **kwargs):
        raise error

    monkeypatch.setattr(misc.Client, "open", fake_open)

    with pytest.raises(misc.APIError) as excinfo:
        misc.download_dem(object())

    assert excinfo.value is error


def test_download_dem_repairs_partial_item_coverage(monkeypatch, tmp_path):
    """A finite cache sliver must not mark an entire source item complete."""
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    monkeypatch.setattr(
        misc,
        "_DEM_CACHE_LIMITS",
        {"x": (-3_500_000, -3_399_900), "y": (-3_450_100, -3_449_900)},
    )
    bbox = [-3_400_100, -3_450_100, -3_399_900, -3_449_900]

    class Item:
        properties = {"proj:bbox": bbox}

        @staticmethod
        def get_collection():
            return SimpleNamespace(id="arcticdem-mosaics-v4.1-32m")

    item = Item()
    source = xr.Dataset(
        {
            "dem": (("y", "x"), np.full((3, 3), 100, dtype="float32")),
            "count": (("y", "x"), np.full((3, 3), 19, dtype="float32")),
        },
        coords={
            "x": [-3_400_100, -3_400_000, -3_399_900],
            "y": [-3_450_100, -3_450_000, -3_449_900],
        },
    ).rio.write_crs(3413)
    for var in source.data_vars.values():
        var.attrs["encoding"] = {"_FillValue": 0}

    calls = []
    monkeypatch.setattr(misc, "_open_pgc_stac_catalog", lambda: object())
    monkeypatch.setattr(misc, "_pgc_stac_items", lambda *args: [item])
    monkeypatch.setattr(
        misc,
        "_read_stac",
        lambda requested_item: calls.append(requested_item) or source,
    )

    cache = misc.download_dem(object())
    far_side = {"x": -3_399_900, "y": -3_450_000}
    cached = xr.open_zarr(cache, mask_and_scale=True)
    assert cached["dem"].sel(far_side).compute().item() == 100

    xr.Dataset(
        {
            name: xr.DataArray(
                [[np.nan]],
                dims=("y", "x"),
                coords={key: [value] for key, value in far_side.items()},
            )
            for name in ["dem", "count"]
        }
    ).to_zarr(cache, mode="r+", region="auto")

    calls.clear()
    misc.download_dem(object())
    assert calls == [item]
    assert (
        xr.open_zarr(cache, mask_and_scale=True)["dem"].sel(far_side).compute().item()
        == 100
    )

    calls.clear()
    misc.download_dem(object())
    assert calls == []
