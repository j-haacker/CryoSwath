from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString


def _load_tool(name):
    path = Path(__file__).parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


track_validator = _load_tool("validate_track_database.py")
archive_builder = _load_tool("build_auxiliary_archive.py")
installed_wheel_tester = _load_tool("test_installed_wheel.py")


def test_installed_wheel_excludes_source_only_maintainer_tools(tmp_path):
    copied = installed_wheel_tester.copy_unit_tests(
        Path(__file__).parents[1], tmp_path / "tests"
    )

    assert "test_maintainer_tools.py" not in {path.name for path in copied}


def _write_caches(tmp_path, timestamps):
    tracks = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)]) for _ in timestamps]},
        index=pd.DatetimeIndex(timestamps, name="index"),
        crs=4326,
    )
    tracks_path = tmp_path / "tracks.feather"
    tracks.to_feather(tracks_path)
    filenames_path = tmp_path / "filenames.pkl"
    pd.Series(["file"] * len(timestamps), index=tracks.index).to_pickle(filenames_path)
    return tracks_path, filenames_path


def test_track_validator_accepts_startup_and_partial_tail(tmp_path):
    timestamps = pd.date_range("2010-07-16", "2026-08-25", freq="7D")
    tracks_path, filenames_path = _write_caches(tmp_path, timestamps)

    report = track_validator.validate_track_database(
        tracks_path, filenames_path, now="2026-10-02"
    )

    assert report.first_timestamp == timestamps[0]
    assert not report.warnings


def test_track_validator_warns_for_late_gap_low_month_and_stale_tail(tmp_path):
    timestamps = pd.date_range("2010-07-16", "2025-12-20", freq="7D")
    timestamps = timestamps[~((timestamps.month == 6) & (timestamps.year == 2024))]
    february_2025 = (timestamps.month == 2) & (timestamps.year == 2025)
    timestamps = timestamps[~february_2025 | (timestamps == pd.Timestamp("2025-02-07"))]
    tracks_path, filenames_path = _write_caches(tmp_path, timestamps)

    report = track_validator.validate_track_database(
        tracks_path, filenames_path, now="2026-10-02"
    )

    assert any("missing monthly coverage" in warning for warning in report.warnings)
    assert any("low track count" in warning for warning in report.warnings)
    assert any("largest post-startup gap" in warning for warning in report.warnings)
    assert any("latest track month" in warning for warning in report.warnings)


def test_track_validator_rejects_filename_outside_tracks(tmp_path):
    tracks_path, filenames_path = _write_caches(
        tmp_path, pd.date_range("2010-07-16", periods=2, freq="D")
    )
    pd.Series(["unexpected"], index=pd.DatetimeIndex(["2010-07-20"])).to_pickle(
        filenames_path
    )

    with pytest.raises(track_validator.ValidationError, match="absent"):
        track_validator.validate_track_database(tracks_path, filenames_path)


def test_track_validator_rejects_invalid_track_structure(tmp_path):
    tracks = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1)])]},
        index=pd.DatetimeIndex(["2010-07-16", "2010-07-16"], name="index"),
        crs=4326,
    )
    tracks_path = tmp_path / "tracks.feather"
    tracks.to_feather(tracks_path)
    filenames_path = tmp_path / "filenames.pkl"
    pd.Series(["file"], index=pd.DatetimeIndex(["2010-07-16"])).to_pickle(
        filenames_path
    )

    with pytest.raises(track_validator.ValidationError, match="duplicate"):
        track_validator.validate_track_database(tracks_path, filenames_path)


def _data_checkout(tmp_path):
    data_dir = tmp_path / "data"
    auxiliary = data_dir / "auxiliary"
    rgi = auxiliary / "RGI"
    rgi.mkdir(parents=True)
    (auxiliary / "CryoSat-2_SARIn_ground_tracks.feather").write_bytes(b"tracks")
    (auxiliary / "CryoSat-2_SARIn_file_names.pkl").write_bytes(b"filenames")
    (rgi / "metadata.txt").write_text("rgi")
    (auxiliary / "CryoSwath-aux-data-old.zip").write_bytes(b"old")
    (auxiliary / ".CryoSat-2_SARIn_ground_tracks.resume.pkl").write_bytes(b"resume")
    subprocess.run(["git", "init", "-q", str(data_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(data_dir), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(data_dir), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(data_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(data_dir), "commit", "-qm", "data"], check=True)
    subprocess.run(["git", "-C", str(data_dir), "branch", "-M", "data"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(data_dir),
            "update-ref",
            "refs/remotes/origin/data",
            "HEAD",
        ],
        check=True,
    )
    return data_dir


def test_archive_builder_creates_valid_archive_and_excludes_transient_files(tmp_path):
    data_dir = _data_checkout(tmp_path)

    archive, commit, members = archive_builder.build_archive(
        data_dir, output_dir=tmp_path
    )

    assert archive.name.startswith("CryoSwath-aux-data-")
    assert archive.suffix == ".zip"
    assert len(commit) == 40
    assert "CryoSat-2_SARIn_ground_tracks.feather" in members
    assert all("CryoSwath-aux-data-old.zip" not in member for member in members)
    assert all("resume.pkl" not in member for member in members)


def test_archive_builder_rejects_dirty_checkout(tmp_path):
    data_dir = _data_checkout(tmp_path)
    (data_dir / "auxiliary" / "uncommitted.txt").write_text("dirty")

    with pytest.raises(archive_builder.ArchiveError, match="clean"):
        archive_builder.build_archive(data_dir, output=tmp_path / "release.zip")
