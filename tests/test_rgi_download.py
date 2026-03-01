from pathlib import Path
import sys
import zipfile

import pandas as pd
import pytest

import cryoswath.misc as misc


def test_rgi_o1_archive_stem_uses_long_code_table(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))

    def fake_read_feather(path, columns):
        assert str(path).endswith("RGI2000-v7.0-o1regions.feather")
        assert columns == ["o1region", "long_code"]
        return pd.DataFrame({"o1region": ["09"], "long_code": ["09_svalbard"]})

    monkeypatch.setattr(misc.pd, "read_feather", fake_read_feather)
    assert misc._rgi_o1_archive_stem("09", "C") == "RGI2000-v7.0-C-09_svalbard"


def test_download_rgi_o1region_uses_auth_extracts_and_cleans_zip(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))
    monkeypatch.setattr(
        misc,
        "_resolve_esa_ftp_credentials",
        lambda: ("esa-user", "esa-password", "env"),
    )
    monkeypatch.setattr(
        misc,
        "_rgi_o1_archive_stem",
        lambda o1code, product: "RGI2000-v7.0-C-09_svalbard",
    )
    monkeypatch.setattr(misc, "_find_rgi_o1region_source", lambda o1, p: None)
    calls = []

    def fake_download_file(url, dest, auth, timeout):
        calls.append((url, Path(dest), auth, timeout))
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr("RGI2000-v7.0-C-09_svalbard/data.shp", "payload")
        return str(dest)

    monkeypatch.setattr(misc, "download_file", fake_download_file)
    out = Path(misc.download_rgi_o1region("09", product="complexes", timeout=42))
    assert out == tmp_path / "RGI2000-v7.0-C-09_svalbard"
    assert out.is_dir()
    assert (out / "data.shp").is_file()
    assert calls[0][0].endswith(
        "/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-C-09_svalbard.zip"
    )
    assert calls[0][2] == ("esa-user", "esa-password")
    assert calls[0][3] == 42
    assert not (tmp_path / "RGI2000-v7.0-C-09_svalbard.zip").exists()


def test_load_o1region_triggers_download_when_missing(monkeypatch):
    calls = {"find": 0, "download": 0}
    expected = object()

    def fake_find(o1code, product):
        calls["find"] += 1
        if calls["find"] == 1:
            return None
        return Path("/tmp/RGI2000-v7.0-C-09_svalbard.feather")

    def fake_download(o1code, product="complexes", force=False, timeout=120):
        calls["download"] += 1
        assert o1code == "09"
        assert product == "C"
        assert force is False
        assert timeout == 120
        return "/tmp/downloaded"

    monkeypatch.setattr(misc, "_find_rgi_o1region_source", fake_find)
    monkeypatch.setattr(misc, "download_rgi_o1region", fake_download)
    monkeypatch.setattr(misc, "_read_rgi_o1region_source", lambda _: expected)
    with pytest.warns(UserWarning, match="Attempting automatic download now"):
        out = misc._load_o1region("09", product="complexes")
    assert out is expected
    assert calls["download"] == 1
    assert calls["find"] == 2


def test_load_o1region_raises_filenotfound_after_download_failure(monkeypatch):
    monkeypatch.setattr(misc, "_find_rgi_o1region_source", lambda o1, p: None)
    monkeypatch.setattr(
        misc,
        "download_rgi_o1region",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.warns(UserWarning, match="Automatic download failed: boom"):
        with pytest.raises(FileNotFoundError):
            misc._load_o1region("09", product="complexes")


def test_load_o1region_does_not_retry_download_for_read_errors(monkeypatch):
    monkeypatch.setattr(
        misc, "_find_rgi_o1region_source", lambda o1, p: Path("/tmp/local.feather")
    )
    monkeypatch.setattr(
        misc,
        "download_rgi_o1region",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download should not be retried on read errors")
        ),
    )
    monkeypatch.setattr(
        misc,
        "_read_rgi_o1region_source",
        lambda path: (_ for _ in ()).throw(RuntimeError("corrupt local file")),
    )
    with pytest.raises(RuntimeError, match="corrupt local file"):
        misc._load_o1region("09", product="complexes")


def test_download_rgi_cli_dispatches_and_prints_path(monkeypatch, capsys):
    observed = {}

    def fake_download(o1code, product="complexes", force=False, timeout=120):
        observed["o1code"] = o1code
        observed["product"] = product
        observed["force"] = force
        observed["timeout"] = timeout
        return "/tmp/RGI2000-v7.0-C-09_svalbard"

    monkeypatch.setattr(misc, "download_rgi_o1region", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath-download-rgi",
            "--o1",
            "09",
            "--product",
            "C",
            "--force",
            "--timeout",
            "90",
        ],
    )
    misc.download_rgi_cli()
    assert observed == {
        "o1code": "09",
        "product": "C",
        "force": True,
        "timeout": 90.0,
    }
    assert capsys.readouterr().out.strip() == "/tmp/RGI2000-v7.0-C-09_svalbard"
