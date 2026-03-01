from pathlib import Path

import pandas as pd
import pytest

import cryoswath.l1b as l1b


def test_download_single_file_prefers_https(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_base_name = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"
    monkeypatch.setattr(l1b, "data_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series({track_time: remote_base_name}),
    )
    calls = []

    def fake_https(track_id, remote_file, local_path, auth):
        calls.append((track_id, remote_file, Path(local_path), auth))
        return str(local_path)

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
    assert calls[0][0] == track_time
    assert calls[0][1] == remote_base_name + ".nc"
    assert calls[0][3] == ("esa-user", "esa-password")


def test_download_single_file_falls_back_to_ftp_on_https_failure(monkeypatch, tmp_path):
    track_id = "20200101T000000"
    track_time = pd.to_datetime(track_id)
    remote_base_name = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"
    monkeypatch.setattr(l1b, "data_path", str(tmp_path))
    monkeypatch.setattr(
        l1b, "_resolve_esa_ftp_credentials", lambda: ("esa-user", "esa-password", "env")
    )
    monkeypatch.setattr(
        l1b,
        "_load_cs_full_file_names_for",
        lambda idx: pd.Series({track_time: remote_base_name}),
    )
    monkeypatch.setattr(
        l1b,
        "_download_named_file_https",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("https failure")),
    )
    monkeypatch.setattr(
        l1b, "_download_single_file_via_ftp", lambda track_id: "ftp-path"
    )
    assert l1b.download_single_file(track_id) == "ftp-path"


def test_download_files_uses_https_and_falls_back_for_unresolved_tracks(
    monkeypatch, tmp_path
):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    resolved_track = track_idx[0]
    unresolved_track = track_idx[1]
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
    ftp_calls = []

    def fake_https(track_id, remote_file, local_path, auth):
        https_calls.append((track_id, remote_file, Path(local_path), auth))
        return str(local_path)

    def fake_ftp(track_idx, stop_event=None):
        ftp_calls.append(pd.DatetimeIndex(track_idx))

    monkeypatch.setattr(l1b, "_download_named_file_https", fake_https)
    monkeypatch.setattr(l1b, "_download_files_via_ftp", fake_ftp)
    l1b.download_files(track_idx)
    assert len(https_calls) == 1
    assert https_calls[0][0] == resolved_track
    assert len(ftp_calls) == 1
    assert unresolved_track in ftp_calls[0]
    assert resolved_track not in ftp_calls[0]


def test_download_files_uses_ftp_when_https_auth_is_unavailable(monkeypatch):
    track_idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    ftp_calls = []
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
        lambda track_idx, stop_event=None: ftp_calls.append(
            pd.DatetimeIndex(track_idx)
        ),
    )
    l1b.download_files(track_idx)
    assert len(ftp_calls) == 1
    assert ftp_calls[0].equals(track_idx)


def test_download_named_file_https_rejects_html_payload(monkeypatch, tmp_path):
    track_time = pd.to_datetime("2020-01-01 00:00:00")
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file

    def fake_download(url, dest, auth, timeout):
        Path(dest).write_text("<!DOCTYPE html><html>login page</html>")
        return str(dest)

    monkeypatch.setattr(l1b, "_http_download_file", fake_download)

    with pytest.raises(RuntimeError, match="HTML/XML"):
        l1b._download_named_file_https(
            track_id=track_time,
            remote_file=remote_file,
            local_path=local_path,
            auth=("user", "password"),
        )
    assert not local_path.exists()


def test_download_named_file_https_accepts_netcdf4_magic(monkeypatch, tmp_path):
    track_time = pd.to_datetime("2020-01-01 00:00:00")
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file

    def fake_download(url, dest, auth, timeout):
        Path(dest).write_bytes(b"\x89HDF\r\n\x1a\n" + b"payload")
        return str(dest)

    monkeypatch.setattr(l1b, "_http_download_file", fake_download)
    result = l1b._download_named_file_https(
        track_id=track_time,
        remote_file=remote_file,
        local_path=local_path,
        auth=("user", "password"),
    )
    assert Path(result).name == "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc"


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


def test_download_named_file_https_falls_back_to_offl_when_lta_missing(
    monkeypatch, tmp_path
):
    track_time = pd.to_datetime("2020-01-01 00:00:00")
    remote_file = "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc"
    local_path = tmp_path / remote_file
    calls = []

    def fake_download(url, dest, auth, timeout):
        calls.append(url)
        if "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc" in url:
            raise RuntimeError("missing LTA")
        Path(dest).write_bytes(b"\x89HDF\r\n\x1a\n" + b"payload")
        return str(dest)

    monkeypatch.setattr(l1b, "_http_download_file", fake_download)
    result = l1b._download_named_file_https(
        track_id=track_time,
        remote_file=remote_file,
        local_path=local_path,
        auth=("user", "password"),
    )
    assert result == str(local_path)
    assert "CS_LTA__SIR_SIN_1B_20200101T000000_TEST.nc" in calls[0]
    assert "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.nc" in calls[1]


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
