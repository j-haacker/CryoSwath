import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
import xarray as xr
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


def test_get_dem_reader_reuses_existing_regional_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "dem_path", tmp_path)
    cache = tmp_path / "arcticdem-mosaics-v4.1-32m_100m-mean.zarr"
    cache.mkdir()
    reader = xr.Dataset({"dem": xr.DataArray(1)})
    monkeypatch.setattr(misc.xr, "open_dataset", lambda path, **kwargs: reader)
    monkeypatch.setattr(
        misc,
        "download_dem",
        lambda geometry: (_ for _ in ()).throw(
            AssertionError("cache should be reused")
        ),
    )

    assert misc.get_dem_reader(misc.shapely.Point(10, 80)).identical(reader.dem)


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



def test_read_stac_accepts_static_item_with_numpy_python_scalars(tmp_path):
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
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as raster:
        raster.write(np.array([[1, 2], [3, 4]], dtype="float32"), 1)

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
                "proj:shape": [2, 2],
                "proj:transform": list(transform)[:6],
                "raster:bands": [{"data_type": "float32", "nodata": -9999}],
            },
        ),
    )

    result = misc._read_stac(item).compute()

    np.testing.assert_array_equal(result["dem"].values[:2, :2], [[1, 2], [3, 4]])


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
