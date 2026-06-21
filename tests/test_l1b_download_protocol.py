import os
from pathlib import Path

import pandas as pd
import pytest

import cryoswath.l1b as l1b


class DummyResponse:
    def __init__(
        self,
        *,
        json_data=None,
        content: bytes = b"",
        headers: dict | None = None,
        url: str = "https://example.com",
        history: list | None = None,
        status_code: int = 200,
    ):
        self._json_data = json_data
        self._content = content
        self.headers = headers or {}
        self.url = url
        self.history = history or []
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=8192):
        if self._content:
            yield self._content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, get_responses=None, post_responses=None):
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_responses.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.post_responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_implicit_stac_catalog_refresh(monkeypatch):
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(),
    )


def test_pds_l1b_download_url_uses_selected_filename():
    filename = "CS_OFFL_SIR_SIN_1B_20220917T082319_20220917T082404_E001.nc"

    assert l1b._pds_l1b_download_url(filename) == (
        "https://science-pds.cryosat.esa.int/?do=download&file="
        "Cry0Sat2_data%2FSIR_SIN_L1%2F2022%2F09%2F" + filename
    )


def test_create_esa_https_session_success(monkeypatch):
    session = DummySession(
        get_responses=[
            DummyResponse(
                url=(
                    "https://eoiam-idp.eo.esa.int/authenticationendpoint/login.do"
                    "?sessionDataKey=test-session-key"
                )
            )
        ],
        post_responses=[DummyResponse(url="https://science-pds.cryosat.esa.int/")],
    )
    monkeypatch.setattr(l1b.requests, "Session", lambda: session)

    result = l1b._create_esa_https_session(("esa-user", "esa-password"))

    assert result is session
    assert session.get_calls[0][0] == l1b._ESA_HTTPS_LOGIN_URL
    assert session.post_calls[0][0].endswith("/commonauth")
    assert session.post_calls[0][1]["data"] == {
        "username": "esa-user",
        "password": "esa-password",
        "sessionDataKey": "test-session-key",
    }


def test_create_esa_https_session_raises_on_auth_failure(monkeypatch):
    session = DummySession(
        get_responses=[
            DummyResponse(
                url=(
                    "https://eoiam-idp.eo.esa.int/authenticationendpoint/login.do"
                    "?sessionDataKey=test-session-key"
                )
            )
        ],
        post_responses=[
            DummyResponse(
                url=(
                    "https://eoiam-idp.eo.esa.int/authenticationendpoint/login.do"
                    "?authFailure=true&authFailureMsg=login.fail.message"
                )
            )
        ],
    )
    monkeypatch.setattr(l1b.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="login failed"):
        l1b._create_esa_https_session(("esa-user", "esa-password"))
    assert session.closed


def test_from_id_reads_from_configured_l1b_path(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    local_dir = tmp_path / "custom-l1b" / "2020" / "01"
    local_dir.mkdir(parents=True)
    local_file = local_dir / f"CS_OFFL_SIR_SIN_1B_{track_id}_TEST.nc"
    local_file.write_text("placeholder")

    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path / "custom-l1b"))
    monkeypatch.setattr(l1b, "read_esa_l1b", lambda path, **kwargs: Path(path))
    monkeypatch.setattr(
        l1b,
        "download_single_file",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("existing L1b file should be read")
        ),
    )

    assert l1b.from_id(pd.Timestamp(track_id)) == local_file


def test_download_single_file_prefers_https(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_base_name = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series({track_time: remote_base_name}),
    )
    calls = []
    session = DummySession()

    def fake_https(remote_file, local_path, session, href=None):
        calls.append((remote_file, Path(local_path), session))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_esa_https_session", lambda auth: session)
    monkeypatch.setattr(l1b, "_download_named_file_https", fake_https)
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("FTP fallback should not be used")
        ),
    )
    result = l1b.download_single_file(track_id)
    assert result.endswith(remote_base_name + ".nc")
    assert calls[0][0] == remote_base_name + ".nc"
    assert calls[0][1] == tmp_path / "2020" / "01" / (remote_base_name + ".nc")
    assert calls[0][2] is session
    assert session.closed


def test_download_single_file_uses_stac_catalog_href(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001.nc"
    href = "https://science-pds.cryosat.esa.int/?do=download&file=test.nc"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(
            {"filename": [remote_file], "href": [href]}, index=[track_time]
        ),
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series(dtype="object"),
    )
    session = DummySession()
    calls = []

    def fake_https(remote_file, local_path, session, href=None):
        calls.append((remote_file, Path(local_path), session, href))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_esa_https_session", lambda auth: session)
    monkeypatch.setattr(l1b, "_download_named_file_https", fake_https)
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("FTP fallback should not be used")
        ),
    )

    result = l1b.download_single_file(track_id)

    assert result.endswith(remote_file)
    assert calls == [
        (remote_file, tmp_path / "2020" / "01" / remote_file, session, href)
    ]
    assert session.closed


def test_download_single_file_fails_fast_on_pds_failure(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_base_name = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series({track_time: remote_base_name}),
    )
    monkeypatch.setattr(l1b, "_create_esa_https_session", lambda auth: DummySession())
    monkeypatch.setattr(
        l1b,
        "_download_named_file_https",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("https failure")),
    )
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("FTP fallback must not be used")
        ),
    )
    with pytest.raises(RuntimeError, match="enhancement #76"):
        l1b.download_single_file(track_id)


def test_download_wrapper_returns_0_without_credentials_for_cached_tracks(
    monkeypatch, tmp_path, capsys
):
    track = pd.Timestamp("2020-09-01 00:00:00")
    local_dir = tmp_path / "2020" / "09"
    local_dir.mkdir(parents=True)
    (local_dir / "CS_OFFL_SIR_SIN_1B_20200901T000000_TEST.nc").touch()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_ftp_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("credentials should not be read")),
    )
    monkeypatch.setattr(
        l1b,
        "request_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("workers should not be created")
        ),
    )

    assert l1b.download_wrapper(track_idx=pd.DatetimeIndex([track])) == 0
    assert "already present" in capsys.readouterr().out


def test_download_wrapper_returns_failure_when_credentials_are_unavailable(
    monkeypatch, tmp_path
):
    track = pd.Timestamp("2020-09-01 00:00:00")
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_ftp_credentials",
        lambda: (_ for _ in ()).throw(RuntimeError("no credentials")),
    )
    monkeypatch.setattr(
        l1b,
        "request_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("workers should not be created")
        ),
    )

    with pytest.warns(UserWarning, match="PDS HTTPS credentials"):
        result = l1b.download_wrapper(track_idx=pd.DatetimeIndex([track]))

    assert result == 1
    assert not (tmp_path / "2020" / "09").exists()


def test_download_wrapper_resolves_once_and_dispatches_only_missing_tracks(
    monkeypatch, tmp_path
):
    track_idx = pd.DatetimeIndex(["2020-09-01 00:00:00", "2020-09-02 00:00:00"])
    local_dir = tmp_path / "2020" / "09"
    local_dir.mkdir(parents=True)
    (local_dir / "CS_OFFL_SIR_SIN_1B_20200901T000000_TEST.nc").touch()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    credential_calls = []
    dispatched = []

    def resolve_credentials():
        credential_calls.append(None)
        return "esa-user", "esa-password", "env"

    def record_worker(track_idx, stop_event, https_auth):
        dispatched.append((pd.DatetimeIndex(track_idx), https_auth))

    monkeypatch.setattr(l1b, "_resolve_esa_ftp_credentials", resolve_credentials)
    monkeypatch.setattr(l1b, "_download_files_with_auth", record_worker)

    assert l1b.download_wrapper(track_idx=track_idx, n_threads=1) == 0
    assert credential_calls == [None]
    assert dispatched == [
        (pd.DatetimeIndex([track_idx[1]]), ("esa-user", "esa-password"))
    ]


def test_download_wrapper_returns_failure_when_worker_fails(monkeypatch, tmp_path):
    def failing_download_files(track_idx, stop_event, https_auth):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(l1b, "_download_files_with_auth", failing_download_files)

    with pytest.warns(UserWarning):
        result = l1b.download_wrapper(
            track_idx=pd.DatetimeIndex(["2020-09-01"]),
            n_threads=1,
        )

    assert result == 1


def test_download_files_fails_fast_for_unresolved_tracks(monkeypatch, tmp_path):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    resolved_track = track_idx[0]
    remote_base_name = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series({resolved_track: remote_base_name}),
    )
    https_calls = []
    session = DummySession()

    def fake_https(remote_file, local_path, session, href=None):
        https_calls.append((remote_file, Path(local_path), session))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_esa_https_session", lambda auth: session)
    monkeypatch.setattr(l1b, "_download_named_file_https", fake_https)
    monkeypatch.setattr(
        l1b,
        "_download_files_via_ftp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FTP fallback must not be used")
        ),
    )
    with pytest.raises(RuntimeError, match="20200102T000000"):
        l1b.download_files(track_idx)
    assert len(https_calls) == 1
    assert https_calls[0][0] == remote_base_name + ".nc"
    assert https_calls[0][2] is session
    assert session.closed
    assert session.closed


def test_download_files_fails_fast_when_pds_auth_is_unavailable(monkeypatch):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_ftp_credentials",
        lambda: (_ for _ in ()).throw(RuntimeError("no credentials")),
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: (_ for _ in ()).throw(
            AssertionError("file-name lookup should be skipped")
        ),
    )
    monkeypatch.setattr(
        l1b,
        "_download_files_via_ftp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FTP fallback must not be used")
        ),
    )
    with pytest.raises(RuntimeError, match="PDS HTTPS credentials"):
        l1b.download_files(track_idx)


def test_download_files_reuses_one_https_session_for_batch(monkeypatch, tmp_path):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    remote_base_names = pd.Series(
        {
            track_idx[0]: "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST",
            track_idx[1]: "CS_OFFL_SIR_SIN_1B_20200102T000000_TEST",
        }
    )
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    credential_calls = []

    def resolve_credentials():
        credential_calls.append(None)
        return "esa-user", "esa-password", "env"

    monkeypatch.setattr(l1b, "_resolve_esa_ftp_credentials", resolve_credentials)
    monkeypatch.setattr(
        l1b, "_load_cs_full_file_names_for", lambda idx: remote_base_names
    )
    session = DummySession()
    session_calls = []

    def fake_https(remote_file, local_path, session, href=None):
        session_calls.append((remote_file, session))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_esa_https_session", lambda auth: session)
    monkeypatch.setattr(l1b, "_download_named_file_https", fake_https)
    l1b.download_files(track_idx)

    assert [call[0] for call in session_calls] == [
        "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc",
        "CS_OFFL_SIR_SIN_1B_20200102T000000_TEST.nc",
    ]
    assert all(call[1] is session for call in session_calls)
    assert session.closed
    assert credential_calls == [None]


def test_download_named_file_https_rejects_html_payload(monkeypatch, tmp_path):
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file
    session = DummySession(
        get_responses=[
            DummyResponse(
                content=b"<!DOCTYPE html><html>login page</html>",
                headers={"content-type": "text/html"},
            ),
            DummyResponse(
                content=b"<!DOCTYPE html><html>login page</html>",
                headers={"content-type": "text/html"},
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="HTML/XML"):
        l1b._download_named_file_https(
            remote_file=remote_file,
            local_path=local_path,
            session=session,
        )
    assert not local_path.exists()


def test_download_named_file_https_accepts_netcdf4_magic(monkeypatch, tmp_path):
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file
    session = DummySession(
        get_responses=[
            DummyResponse(
                content=b"\x89HDF\r\n\x1a\n" + b"payload",
                headers={"content-type": "application/x-netcdf"},
            )
        ]
    )
    result = l1b._download_named_file_https(
        remote_file=remote_file,
        local_path=local_path,
        session=session,
    )
    assert Path(result).name == remote_file


def test_l1b_product_name_candidates_prefer_lta_then_offl():
    offl = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    assert l1b._l1b_product_name_candidates(offl)[:2] == [
        "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc",
        offl,
    ]


def test_select_lta_then_offl_for_track_prefers_lta():
    track_id = "20200101T000000"
    remote_files = [
        "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc",
        "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc",
    ]
    assert (
        l1b._select_lta_then_offl_for_track(track_id, remote_files)
        == "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc"
    )


def test_select_lta_then_offl_for_track_raises_when_missing():
    with pytest.raises(FileNotFoundError, match="No LTA_ or OFFL"):
        l1b._select_lta_then_offl_for_track(
            "20200101T000000",
            ["CS_GDR_SIR_SIN_1B_20200101T000000_TEST.nc"],
        )


def test_download_named_file_https_uses_pds_when_catalog_has_maap_href(tmp_path):
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file
    session = DummySession(
        get_responses=[
            DummyResponse(
                content=b"\x89HDF\r\n\x1a\n" + b"payload",
                headers={"content-type": "application/x-netcdf"},
            )
        ]
    )
    result = l1b._download_named_file_https(
        remote_file=remote_file,
        local_path=local_path,
        session=session,
        href="https://catalog.maap.eo.esa.int/data/cryosat/example.nc",
    )
    assert result == str(local_path)
    assert session.get_calls[0][0] == l1b._pds_l1b_download_url(remote_file)


def test_download_remote_file_via_ftp_atomic_success(tmp_path):
    local_path = tmp_path / "file.nc"

    class FakeFtp:
        def retrbinary(self, cmd, callback):
            assert cmd == "RETR remote.nc"
            callback(b"abc123")

    result = l1b._download_remote_file_via_ftp_atomic(
        FakeFtp(), "remote.nc", local_path
    )
    assert result == str(local_path)
    assert local_path.read_bytes() == b"abc123"
    assert [p.name for p in tmp_path.iterdir()] == ["file.nc"]


def test_download_remote_file_via_ftp_atomic_cleans_temp_on_failure(tmp_path):
    local_path = tmp_path / "file.nc"

    class FailingFtp:
        def retrbinary(self, cmd, callback):
            callback(b"partial")
            raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        l1b._download_remote_file_via_ftp_atomic(FailingFtp(), "remote.nc", local_path)
    assert not local_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_live_download_single_file_prefers_https_when_enabled(monkeypatch, tmp_path):
    if os.environ.get("CRYOSWATH_RUN_LIVE_ESA") != "1":
        pytest.skip(
            "Set CRYOSWATH_RUN_LIVE_ESA=1 to run the live ESA HTTPS smoke test."
        )

    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("Live HTTPS smoke test should not fall back to FTP")
        ),
    )

    result = l1b.download_single_file("20230306T025949")

    path = Path(result)
    assert path.is_file()
    with path.open("rb") as handle:
        header = handle.read(16)
    assert header.startswith(b"\x89HDF\r\n\x1a\n") or header.startswith(
        (b"CDF\x01", b"CDF\x02", b"CDF\x05")
    )
