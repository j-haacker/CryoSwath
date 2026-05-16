from pathlib import Path
import sys
import zipfile

import pytest

import cryoswath.misc as misc


def test_normalize_rgi_product_aliases():
    assert misc._normalize_rgi_product("complexes") == "C"
    assert misc._normalize_rgi_product("C") == "C"
    assert misc._normalize_rgi_product("c") == "C"
    assert misc._normalize_rgi_product("glaciers") == "G"
    assert misc._normalize_rgi_product("basins") == "G"
    assert misc._normalize_rgi_product("G") == "G"


def test_normalize_rgi_product_rejects_invalid_product():
    with pytest.raises(ValueError, match="glaciers.*complexes"):
        misc._normalize_rgi_product("ice")


def test_normalize_rgi_o1code_accepts_short_and_long_codes():
    assert misc._normalize_rgi_o1code(9) == "09"
    assert misc._normalize_rgi_o1code("9") == "09"
    assert misc._normalize_rgi_o1code("09_russian_arctic") == "09"


def test_normalize_rgi_o1code_rejects_invalid_code():
    with pytest.raises(ValueError, match="o1code should start"):
        misc._normalize_rgi_o1code("x09")


def test_rgi_remote_product_url_normalizes_product():
    assert misc._rgi_remote_product_url("complexes").endswith("/RGI2000-v7.0-C/")
    assert misc._rgi_remote_product_url("glaciers").endswith("/RGI2000-v7.0-G/")


def test_rgi_o1_archive_stem_uses_long_code_translator(monkeypatch):
    calls = []

    def fake_translator(code, out_type="full_name"):
        calls.append((code, out_type))
        return "09_russian_arctic"

    monkeypatch.setattr(misc, "rgi_code_translator", fake_translator)
    assert (
        misc._rgi_o1_archive_stem(9, "complexes")
        == "RGI2000-v7.0-C-09_russian_arctic"
    )
    assert calls == [("09", "long_code")]


def test_find_rgi_o1region_source_finds_supported_local_sources(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))

    feather = tmp_path / "RGI2000-v7.0-C-09_russian_arctic.feather"
    feather.write_bytes(b"placeholder")
    assert misc._find_rgi_o1region_source(9, "complexes") == feather

    glacier_dir = tmp_path / "RGI2000-v7.0-G-09_russian_arctic"
    glacier_dir.mkdir()
    assert misc._find_rgi_o1region_source("09", "glaciers") == glacier_dir

    shp = tmp_path / "RGI2000-v7.0-C-10_north_asia.shp"
    shp.write_bytes(b"placeholder")
    assert misc._find_rgi_o1region_source(10, "C") == shp


def test_find_rgi_o1region_source_ignores_unsupported_sources(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))
    (tmp_path / "RGI2000-v7.0-C-09_russian_arctic.zip").write_bytes(b"zip")
    assert misc._find_rgi_o1region_source("09", "complexes") is None


def test_find_rgi_o1region_source_returns_none_for_missing_root(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path / "missing"))
    assert misc._find_rgi_o1region_source("09", "complexes") is None


def test_download_rgi_o1region_skips_download_when_present(monkeypatch, tmp_path):
    existing = tmp_path / "RGI2000-v7.0-C-09_russian_arctic.feather"
    existing.write_bytes(b"present")
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))
    monkeypatch.setattr(
        misc,
        "_resolve_earthdata_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("credentials should not be resolved")
        ),
    )
    monkeypatch.setattr(
        misc,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download should not run")
        ),
    )

    assert misc.download_rgi_o1region("09", product="complexes") == str(existing)


def test_download_rgi_o1region_uses_earthdata_extracts_and_cleans_zip(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))
    monkeypatch.setattr(
        misc,
        "_resolve_earthdata_credentials",
        lambda: ("earth-user", "earth-password", "env"),
    )
    monkeypatch.setattr(
        misc,
        "_resolve_esa_ftp_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("ESA credentials must not be used for RGI")
        ),
    )
    monkeypatch.setattr(
        misc,
        "_rgi_o1_archive_stem",
        lambda o1code, product: "RGI2000-v7.0-C-09_russian_arctic",
    )
    calls = []

    def fake_download_file(url, dest, auth, timeout):
        calls.append((url, Path(dest), auth, timeout))
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr("RGI2000-v7.0-C-09_russian_arctic/data.shp", "payload")
        return str(dest)

    monkeypatch.setattr(misc, "download_file", fake_download_file)
    out = Path(misc.download_rgi_o1region("09", product="complexes", timeout=42))

    assert out == tmp_path / "RGI2000-v7.0-C-09_russian_arctic"
    assert out.is_dir()
    assert (out / "data.shp").is_file()
    assert calls == [
        (
            misc._RGI_DOWNLOAD_BASE_URL
            + "/RGI2000-v7.0-C/RGI2000-v7.0-C-09_russian_arctic.zip",
            tmp_path / "RGI2000-v7.0-C-09_russian_arctic.zip",
            ("earth-user", "earth-password"),
            42,
        )
    ]
    assert not (tmp_path / "RGI2000-v7.0-C-09_russian_arctic.zip").exists()


def test_download_rgi_o1region_replaces_existing_target_after_download(
    monkeypatch, tmp_path
):
    archive_stem = "RGI2000-v7.0-C-09_russian_arctic"
    target = tmp_path / archive_stem
    target.mkdir()
    (target / "old.shp").write_text("old")
    monkeypatch.setattr(misc, "rgi_path", str(tmp_path))
    monkeypatch.setattr(
        misc,
        "_resolve_earthdata_credentials",
        lambda: ("earth-user", "earth-password", "env"),
    )
    monkeypatch.setattr(misc, "_rgi_o1_archive_stem", lambda o1code, product: archive_stem)

    def fake_download_file(url, dest, auth, timeout):
        assert (target / "old.shp").exists()
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr(f"{archive_stem}/new.shp", "new")
        return str(dest)

    monkeypatch.setattr(misc, "download_file", fake_download_file)

    out = Path(misc.download_rgi_o1region("09", product="complexes", force=True))

    assert out == target
    assert not (target / "old.shp").exists()
    assert (target / "new.shp").is_file()


def test_load_o1region_triggers_download_when_missing(monkeypatch):
    calls = {"find": 0, "download": 0}
    expected = object()

    def fake_find(o1code, product):
        calls["find"] += 1
        if calls["find"] == 1:
            return None
        return Path("/tmp/RGI2000-v7.0-C-09_russian_arctic.feather")

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


def test_load_o1region_raises_filenotfound_when_download_does_not_create_source(
    monkeypatch,
):
    calls = {"download": 0}
    monkeypatch.setattr(misc, "_find_rgi_o1region_source", lambda o1, p: None)

    def fake_download(*args, **kwargs):
        calls["download"] += 1
        return "/tmp/downloaded"

    monkeypatch.setattr(misc, "download_rgi_o1region", fake_download)
    with pytest.warns(UserWarning, match="Attempting automatic download now"):
        with pytest.raises(FileNotFoundError):
            misc._load_o1region("09", product="complexes")
    assert calls["download"] == 1


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


def test_resolve_earthdata_credentials_prefers_environment_variables(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USER", "env-user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "env-password")
    monkeypatch.setattr(
        misc,
        "_resolve_earthdata_keyring_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("keyring should not be used")),
    )

    user, password, source = misc._resolve_earthdata_credentials()

    assert user == "env-user"
    assert password == "env-password"
    assert source == "environment variables"


def test_resolve_earthdata_credentials_uses_keyring_before_netrc(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USER", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    store = {
        (misc._EARTHDATA_HOST, misc._ESA_KEYRING_DEFAULT_USER_KEY): "keyring-user",
        (misc._EARTHDATA_HOST, "keyring-user"): "keyring-password",
    }

    class FakeKeyring:
        def get_password(self, service, user):
            return store.get((service, user))

    class NetrcNotExpected:
        def authenticators(self, machine):
            raise AssertionError("netrc should not be used when keyring is available")

    monkeypatch.setattr(misc, "keyring", FakeKeyring())
    monkeypatch.setattr(misc.netrc, "netrc", lambda: NetrcNotExpected())

    user, password, source = misc._resolve_earthdata_credentials()

    assert user == "keyring-user"
    assert password == "keyring-password"
    assert source == f"keyring service {misc._EARTHDATA_HOST}"


def test_resolve_earthdata_credentials_uses_netrc_when_keyring_missing(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USER", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr(misc, "_resolve_earthdata_keyring_credentials", lambda: None)

    class FakeNetrc:
        def authenticators(self, machine):
            assert machine == misc._EARTHDATA_HOST
            return ("netrc-user", None, "netrc-password")

    monkeypatch.setattr(misc.netrc, "netrc", lambda: FakeNetrc())

    user, password, source = misc._resolve_earthdata_credentials()

    assert user == "netrc-user"
    assert password == "netrc-password"
    assert source == "~/.netrc"


def test_download_rgi_cli_dispatches_and_prints_path(monkeypatch, capsys):
    observed = {}

    def fake_download(o1code, product="complexes", force=False, timeout=120):
        observed["o1code"] = o1code
        observed["product"] = product
        observed["force"] = force
        observed["timeout"] = timeout
        return "/tmp/RGI2000-v7.0-C-09_russian_arctic"

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
    assert capsys.readouterr().out.strip() == "/tmp/RGI2000-v7.0-C-09_russian_arctic"
