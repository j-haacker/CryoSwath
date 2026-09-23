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
        self.headers = {}

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_responses.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.post_responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_implicit_stac_catalog_refresh(monkeypatch, request):
    if request.node.name.startswith("test_catalog_loader_refreshes_"):
        return
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


def test_create_maap_session_exchanges_offline_token(monkeypatch):
    session = DummySession(
        post_responses=[DummyResponse(json_data={"access_token": "short-lived"})]
    )
    monkeypatch.setattr(l1b.requests, "Session", lambda: session)

    assert l1b._create_maap_session("offline-token") is session
    assert session.post_calls == [
        (
            l1b._ESA_MAAP_TOKEN_URL,
            {
                "data": {
                    "client_id": "offline-token",
                    "client_secret": l1b._ESA_MAAP_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": "offline-token",
                    "scope": "offline_access openid",
                },
                "timeout": 120,
            },
        )
    ]
    assert session.headers["Authorization"] == "Bearer short-lived"


def test_create_maap_session_closes_on_missing_access_token(monkeypatch):
    session = DummySession(post_responses=[DummyResponse(json_data={})])
    monkeypatch.setattr(l1b.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="access token"):
        l1b._create_maap_session("offline-token")
    assert session.closed


def test_create_maap_session_closes_on_http_failure(monkeypatch):
    session = DummySession(post_responses=[DummyResponse(status_code=401)])
    monkeypatch.setattr(l1b.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        l1b._create_maap_session("offline-token")
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


def test_download_single_file_uses_stac_catalog_href(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    catalog_time = pd.Timestamp("2019-12-31T23:59:59")
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001.nc"
    href = "https://science-pds.cryosat.esa.int/?do=download&file=test.nc"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_maap_offline_token", lambda: ("token", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(
            {"filename": [remote_file], "href": [href]}, index=[catalog_time]
        ),
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series(dtype="object"),
    )
    session = DummySession()
    calls = []

    def fake_maap(remote_file, local_path, session, href):
        calls.append((remote_file, Path(local_path), session, href))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_maap_session", lambda token: session)
    monkeypatch.setattr(l1b, "_download_named_file_maap", fake_maap)
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


def test_catalog_loader_refreshes_cached_track_without_maap_href(monkeypatch):
    track = pd.Timestamp("2020-01-01T00:00:00")
    stale = pd.DataFrame(
        {"filename": ["CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"], "href": [None]},
        index=[track],
    )
    refreshed = stale.assign(href="https://catalog.maap.eo.esa.int/data/file.nc")
    catalog_reads = iter([stale, refreshed])
    refresh_calls = []

    monkeypatch.setattr(
        l1b, "load_cs_l1b_track_catalog", lambda update: next(catalog_reads)
    )
    monkeypatch.setattr(
        l1b,
        "load_cs_ground_tracks",
        lambda **kwargs: refresh_calls.append(kwargs),
    )

    result = l1b._load_cs_l1b_track_catalog_for(pd.DatetimeIndex([track]))

    assert refresh_calls
    assert result.loc[track, "href"] == "https://catalog.maap.eo.esa.int/data/file.nc"


def test_catalog_loader_refreshes_cache_without_href_column(monkeypatch):
    track = pd.Timestamp("2020-01-01T00:00:00")
    stale = pd.DataFrame(
        {"filename": ["CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"]}, index=[track]
    )
    refreshed = stale.assign(href="https://catalog.maap.eo.esa.int/data/file.nc")
    catalog_reads = iter([stale, refreshed])
    refresh_calls = []

    monkeypatch.setattr(
        l1b, "load_cs_l1b_track_catalog", lambda update: next(catalog_reads)
    )
    monkeypatch.setattr(
        l1b, "load_cs_ground_tracks", lambda **kwargs: refresh_calls.append(kwargs)
    )

    result = l1b._load_cs_l1b_track_catalog_for(pd.DatetimeIndex([track]))

    assert refresh_calls
    assert result.loc[track, "href"] == "https://catalog.maap.eo.esa.int/data/file.nc"


def test_download_single_file_reports_maap_failure(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_maap_offline_token", lambda: ("token", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(
            {"filename": [remote_file], "href": ["https://example.test/file.nc"]},
            index=[track_time],
        ),
    )
    monkeypatch.setattr(l1b, "_create_maap_session", lambda token: DummySession())
    monkeypatch.setattr(
        l1b,
        "_download_named_file_maap",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("maap failure")),
    )
    with pytest.raises(RuntimeError, match="CryoSat L1b delivery failed"):
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
        "_resolve_esa_maap_offline_token",
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


def test_download_wrapper_defers_credentials_to_workers(monkeypatch, tmp_path):
    track = pd.Timestamp("2020-09-01 00:00:00")
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_maap_offline_token",
        lambda: (_ for _ in ()).throw(RuntimeError("no token")),
    )
    dispatched = []

    def record_worker(track_idx, stop_event):
        dispatched.append(pd.DatetimeIndex(track_idx))

    monkeypatch.setattr(l1b, "_download_files_with_catalog_routes", record_worker)
    assert l1b.download_wrapper(track_idx=pd.DatetimeIndex([track]), n_threads=1) == 0
    assert dispatched == [pd.DatetimeIndex([track])]


def test_download_wrapper_dispatches_only_missing_tracks_without_token_preflight(
    monkeypatch, tmp_path
):
    track_idx = pd.DatetimeIndex(["2020-09-01 00:00:00", "2020-09-02 00:00:00"])
    local_dir = tmp_path / "2020" / "09"
    local_dir.mkdir(parents=True)
    (local_dir / "CS_OFFL_SIR_SIN_1B_20200901T000000_TEST.nc").touch()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    dispatched = []

    def record_worker(track_idx, stop_event):
        dispatched.append(pd.DatetimeIndex(track_idx))

    monkeypatch.setattr(
        l1b,
        "_resolve_esa_maap_offline_token",
        lambda: (_ for _ in ()).throw(
            AssertionError("token should not be preflighted")
        ),
    )
    monkeypatch.setattr(l1b, "_download_files_with_catalog_routes", record_worker)

    assert l1b.download_wrapper(track_idx=track_idx, n_threads=1) == 0
    assert dispatched == [pd.DatetimeIndex([track_idx[1]])]


def test_download_wrapper_returns_failure_when_worker_fails(monkeypatch, tmp_path):
    def failing_download_files(track_idx, stop_event):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_download_files_with_catalog_routes", failing_download_files
    )

    with pytest.warns(UserWarning):
        result = l1b.download_wrapper(
            track_idx=pd.DatetimeIndex(["2020-09-01"]),
            n_threads=1,
        )

    assert result == 1


def test_download_files_fails_fast_for_unresolved_tracks(monkeypatch, tmp_path):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    resolved_track = track_idx[0]
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_maap_offline_token", lambda: ("token", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(
            {"filename": [remote_file], "href": ["https://example.test/file.nc"]},
            index=[resolved_track],
        ),
    )
    maap_calls = []
    session = DummySession()

    def fake_maap(remote_file, local_path, session, href):
        maap_calls.append((remote_file, Path(local_path), session, href))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_maap_session", lambda token: session)
    monkeypatch.setattr(l1b, "_download_named_file_maap", fake_maap)
    with pytest.raises(RuntimeError, match="20200102T000000"):
        l1b.download_files(track_idx)
    assert len(maap_calls) == 1
    assert maap_calls[0][0] == remote_file
    assert maap_calls[0][2] is session
    assert session.closed


def test_download_files_reports_unavailable_maap_token_after_catalog_lookup(
    monkeypatch,
):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_maap_offline_token",
        lambda: (_ for _ in ()).throw(RuntimeError("no token")),
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_l1b_track_catalog_for",
        lambda idx: pd.DataFrame(
            {
                "filename": ["CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"],
                "href": ["https://maap.example/test.nc"],
                "provider": ["maap"],
            },
            index=[track_idx[0]],
        ),
    )
    monkeypatch.setattr(
        l1b,
        "_download_files_via_ftp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FTP fallback must not be used")
        ),
    )
    with pytest.raises(RuntimeError, match="no token"):
        l1b.download_files(track_idx)


def test_download_files_reuses_one_maap_session_for_batch(monkeypatch, tmp_path):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    remote_files = pd.DataFrame(
        {
            "filename": [
                "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc",
                "CS_OFFL_SIR_SIN_1B_20200102T000000_TEST.nc",
            ],
            "href": ["https://example.test/one.nc", "https://example.test/two.nc"],
        },
        index=track_idx,
    )
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    credential_calls = []

    def resolve_token():
        credential_calls.append(None)
        return "token", "env"

    monkeypatch.setattr(l1b, "_resolve_esa_maap_offline_token", resolve_token)
    monkeypatch.setattr(l1b, "_load_cs_l1b_track_catalog_for", lambda idx: remote_files)
    session = DummySession()
    session_calls = []

    def fake_maap(remote_file, local_path, session, href):
        session_calls.append((remote_file, session))
        return str(local_path)

    monkeypatch.setattr(l1b, "_create_maap_session", lambda token: session)
    monkeypatch.setattr(l1b, "_download_named_file_maap", fake_maap)
    l1b.download_files(track_idx)

    assert [call[0] for call in session_calls] == [
        *remote_files["filename"],
    ]
    assert all(call[1] is session for call in session_calls)
    assert session.closed
    assert credential_calls == [None]


def test_download_single_file_uses_ftp_when_maap_href_is_missing(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    expected = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_F001.nc"
    selected = tmp_path / "2020" / "01" / expected.replace("OFFL", "LTA_")
    catalog = pd.DataFrame(
        {"filename": [expected], "href": [None], "provider": ["maap"]},
        index=[pd.Timestamp(track_id)],
    )
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(l1b, "_load_cs_l1b_track_catalog_for", lambda idx: catalog)

    def fake_ftp(_):
        selected.parent.mkdir(parents=True)
        selected.write_bytes(b"\x89HDF\r\n\x1a\nfixture")
        return str(selected)

    monkeypatch.setattr(l1b, "_download_single_file_via_ftp", fake_ftp)
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_maap_offline_token",
        lambda: (_ for _ in ()).throw(AssertionError("MAAP token must not be used")),
    )

    with pytest.warns(UserWarning, match="no usable MAAP asset URL"):
        result = l1b.download_single_file(track_id)

    assert result == str(selected)


def test_download_single_file_uses_ftp_when_maap_has_no_catalog_entry(
    monkeypatch, tmp_path
):
    track_id = "20200101T000000"
    selected = tmp_path / "2020" / "01" / "selected.nc"
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_load_cs_l1b_track_catalog_for", lambda idx: pd.DataFrame()
    )

    def fake_ftp(_):
        selected.parent.mkdir(parents=True)
        selected.write_bytes(b"\x89HDF\r\n\x1a\nfixture")
        return str(selected)

    monkeypatch.setattr(l1b, "_download_single_file_via_ftp", fake_ftp)
    monkeypatch.setattr(
        l1b,
        "_resolve_esa_maap_offline_token",
        lambda: (_ for _ in ()).throw(AssertionError("MAAP token must not be used")),
    )

    with pytest.warns(UserWarning, match="no MAAP catalogue entry"):
        assert l1b.download_single_file(track_id) == str(selected)


def test_maap_failure_does_not_use_ftp_fallback(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001.nc"
    catalog = pd.DataFrame(
        {
            "filename": [remote_file],
            "href": ["https://catalog.maap.eo.esa.int/data/file.nc"],
            "provider": ["maap"],
        },
        index=[pd.Timestamp(track_id)],
    )
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(l1b, "_load_cs_l1b_track_catalog_for", lambda idx: catalog)
    monkeypatch.setattr(
        l1b, "_resolve_esa_maap_offline_token", lambda: ("token", "env")
    )
    monkeypatch.setattr(l1b, "_create_maap_session", lambda _: DummySession())
    monkeypatch.setattr(
        l1b,
        "_download_named_file_maap",
        lambda **_: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda _: (_ for _ in ()).throw(AssertionError("FTP must not be used")),
    )

    with pytest.raises(RuntimeError, match="gone"):
        l1b.download_single_file(track_id)


def test_download_named_file_maap_rejects_html_payload(monkeypatch, tmp_path):
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
        l1b._download_named_file_maap(
            remote_file=remote_file,
            local_path=local_path,
            session=session,
            href="https://catalog.maap.eo.esa.int/data/cryosat/example.nc",
        )
    assert not local_path.exists()


def test_download_named_file_maap_accepts_netcdf4_magic(monkeypatch, tmp_path):
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
    result = l1b._download_named_file_maap(
        remote_file=remote_file,
        local_path=local_path,
        session=session,
        href="https://catalog.maap.eo.esa.int/data/cryosat/example.nc",
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


def test_download_named_file_maap_uses_catalog_href(tmp_path):
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
    href = "https://catalog.maap.eo.esa.int/data/cryosat/example.nc"
    result = l1b._download_named_file_maap(
        remote_file=remote_file,
        local_path=local_path,
        session=session,
        href=href,
    )
    assert result == str(local_path)
    assert session.get_calls[0][0] == href


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


def test_live_download_single_file_uses_maap_when_enabled(monkeypatch, tmp_path):
    if os.environ.get("CRYOSWATH_RUN_LIVE_ESA") != "1" or not os.environ.get(
        "ESA_MAAP_OFFLINE_TOKEN"
    ):
        pytest.skip(
            "Set CRYOSWATH_RUN_LIVE_ESA=1 and ESA_MAAP_OFFLINE_TOKEN to run "
            "the live ESA MAAP smoke test."
        )

    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(
        l1b,
        "_download_single_file_via_ftp",
        lambda track_id: (_ for _ in ()).throw(
            AssertionError("Live MAAP smoke test should not fall back to FTP")
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


def test_download_single_file_via_ftp_uses_baseline_e_when_current_lacks_track(
    monkeypatch, tmp_path
):
    track_id = "20200101T000000"
    current_directory = "/SIR_SIN_L1/2020/01"
    legacy_directory = "/Ice_Baseline_E/SIR_SIN_L1/2020/01"
    legacy_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_LEGACY.nc"

    class FakeFtp:
        def __init__(self):
            self.directory = None
            self.cwd_calls = []

        def cwd(self, directory):
            self.directory = directory
            self.cwd_calls.append(directory)

        def nlst(self):
            return {
                current_directory: ["CS_OFFL_SIR_SIN_1B_20200102T000000_CURRENT.nc"],
                legacy_directory: [legacy_file],
            }[self.directory]

        def retrbinary(self, command, callback):
            assert self.directory == legacy_directory
            assert command == f"RETR {legacy_file}"
            callback(b"\x89HDF\r\n\x1a\nfixture")

    class FakeFtpContext:
        def __init__(self):
            self.ftp = FakeFtp()

        def __enter__(self):
            return self.ftp

        def __exit__(self, *args):
            return False

    context = FakeFtpContext()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    ftp_calls = []
    monkeypatch.setattr(
        l1b,
        "ftp_cs2_server",
        lambda **kwargs: (ftp_calls.append(kwargs), context)[1],
    )

    result = l1b._download_single_file_via_ftp(track_id)

    assert Path(result).name == legacy_file
    assert context.ftp.cwd_calls == [current_directory, legacy_directory]
    assert ftp_calls == [{"timeout": 120}]


def test_download_files_via_ftp_prefers_current_directory(monkeypatch, tmp_path):
    track_id = pd.Timestamp("2020-01-01T00:00:00")
    current_directory = "/SIR_SIN_L1/2020/01"
    legacy_directory = "/Ice_Baseline_E/SIR_SIN_L1/2020/01"
    current_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_CURRENT.nc"
    legacy_file = "CS_LTA__SIR_SIN_1B_20200101T000000_LEGACY.nc"

    class FakeFtp:
        def __init__(self):
            self.directory = None
            self.cwd_calls = []

        def cwd(self, directory):
            self.directory = directory
            self.cwd_calls.append(directory)

        def nlst(self):
            return {
                current_directory: [current_file],
                legacy_directory: [legacy_file],
            }[self.directory]

        def retrbinary(self, command, callback):
            assert self.directory == current_directory
            assert command == f"RETR {current_file}"
            callback(b"\x89HDF\r\n\x1a\nfixture")

    class FakeFtpContext:
        def __init__(self):
            self.ftp = FakeFtp()

        def __enter__(self):
            return self.ftp

        def __exit__(self, *args):
            return False

    context = FakeFtpContext()
    monkeypatch.setattr(l1b, "l1b_path", str(tmp_path))
    monkeypatch.setattr(l1b, "ftp_cs2_server", lambda **kwargs: context)

    l1b._download_files_via_ftp(pd.DatetimeIndex([track_id]))

    assert (tmp_path / "2020" / "01" / current_file).is_file()
    assert context.ftp.cwd_calls == [current_directory]
