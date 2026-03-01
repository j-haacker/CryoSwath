import io
from pathlib import Path
import tarfile

import pytest

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

    with pytest.warns(UserWarning, match="Attempting automatic download now"):
        out = misc.get_dem_reader(80)

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

    with pytest.warns(UserWarning, match="Attempting automatic download now"):
        out = misc.get_dem_reader(-80)

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
            misc.get_dem_reader(80)


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
