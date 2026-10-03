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
    timestamps = pd.DatetimeIndex(timestamps, name="index")
    tracks = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)]) for _ in timestamps]},
        index=timestamps,
        crs=4326,
    )
    tracks_path = tmp_path / "tracks.feather"
    tracks.to_feather(tracks_path)
    filenames_path = tmp_path / "filenames.pkl"
    pd.Series(
        [_filename(timestamp) for timestamp in timestamps], index=tracks.index
    ).to_pickle(filenames_path)
    return tracks_path, filenames_path


def _filename(timestamp):
    timestamp = pd.Timestamp(timestamp)
    end = timestamp + pd.Timedelta(minutes=1)
    return f"CS_OFFL_SIR_SIN_1B_{timestamp:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_E001"


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
    assert any("largest post-startup gap" in warning for warning in report.warnings)
    assert any("latest track month" in warning for warning in report.warnings)


def test_track_validator_warns_for_robust_monthly_count_anomaly(tmp_path):
    timestamps = []
    for month in pd.period_range("2021-01", "2025-03", freq="M"):
        count = 21 if month == pd.Period("2025-02", freq="M") else 30
        timestamps.extend(pd.date_range(month.start_time, periods=count, freq="12h"))
    tracks_path, filenames_path = _write_caches(tmp_path, timestamps)

    report = track_validator.validate_track_database(tracks_path, filenames_path)

    assert any("low track count in 2025-02" in warning for warning in report.warnings)


def test_track_validator_warns_for_small_filename_superset(tmp_path):
    tracks_path, filenames_path = _write_caches(
        tmp_path, pd.date_range("2010-07-16", periods=25, freq="D")
    )
    filenames = pd.read_pickle(filenames_path)
    extra_time = pd.Timestamp("2010-08-20")
    filenames.loc[extra_time] = _filename(extra_time)
    filenames.to_pickle(filenames_path)

    report = track_validator.validate_track_database(tracks_path, filenames_path)

    assert any(
        "1 entries without ground-track geometry" in warning
        for warning in report.warnings
    )


def test_track_validator_rejects_track_without_filename(tmp_path):
    tracks_path, filenames_path = _write_caches(
        tmp_path, pd.date_range("2010-07-16", periods=2, freq="D")
    )
    filenames = pd.read_pickle(filenames_path).iloc[:1]
    filenames.to_pickle(filenames_path)

    with pytest.raises(track_validator.ValidationError, match="tracks.*filename"):
        track_validator.validate_track_database(tracks_path, filenames_path)


def test_track_validator_rejects_large_monthly_geometry_gap(tmp_path):
    tracks_path, filenames_path = _write_caches(
        tmp_path, pd.date_range("2010-07-16", periods=10, freq="D")
    )
    filenames = pd.read_pickle(filenames_path)
    for extra_time in pd.to_datetime(["2010-07-26", "2010-07-27"]):
        filenames.loc[extra_time] = _filename(extra_time)
    filenames.to_pickle(filenames_path)

    with pytest.raises(track_validator.ValidationError, match="geometry.*2010-07"):
        track_validator.validate_track_database(tracks_path, filenames_path)


def test_track_validator_warns_and_infers_missing_geographic_crs(tmp_path):
    timestamp = pd.Timestamp("2010-07-16")
    tracks = gpd.GeoDataFrame(
        {"geometry": [LineString([(-10, 70), (10, 71)])]},
        index=pd.DatetimeIndex([timestamp], name="index"),
    )
    tracks_path = tmp_path / "tracks.feather"
    tracks.to_feather(tracks_path)
    filenames_path = tmp_path / "filenames.pkl"
    pd.Series([_filename(timestamp)], index=tracks.index).to_pickle(filenames_path)

    report = track_validator.validate_track_database(tracks_path, filenames_path)

    assert any("missing CRS" in warning for warning in report.warnings)


def test_track_validator_rejects_malformed_filename(tmp_path):
    tracks_path, filenames_path = _write_caches(tmp_path, ["2010-07-16"])
    filenames = pd.read_pickle(filenames_path)
    filenames.iloc[0] = "not-a-cryosat-filename"
    filenames.to_pickle(filenames_path)

    with pytest.raises(track_validator.ValidationError, match="malformed"):
        track_validator.validate_track_database(tracks_path, filenames_path)


def test_track_validator_rejects_filename_timestamp_mismatch(tmp_path):
    tracks_path, filenames_path = _write_caches(tmp_path, ["2010-07-16"])
    filenames = pd.read_pickle(filenames_path)
    filenames.iloc[0] = _filename("2010-07-17")
    filenames.to_pickle(filenames_path)

    with pytest.raises(track_validator.ValidationError, match="timestamp.*index"):
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
    timestamp = pd.Timestamp("2010-07-16")
    tracks = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]},
        index=pd.DatetimeIndex([timestamp], name="index"),
        crs=4326,
    )
    tracks.to_feather(auxiliary / "CryoSat-2_SARIn_ground_tracks.feather")
    pd.Series([_filename(timestamp)], index=tracks.index).to_pickle(
        auxiliary / "CryoSat-2_SARIn_file_names.pkl"
    )
    (rgi / "metadata.txt").write_text("rgi")
    (auxiliary / "CryoSwath-aux-data-old.zip").write_bytes(b"old")
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
    (auxiliary / archive_builder.misc._TRACK_UPDATE_CHECKPOINT_NAME).write_bytes(
        b"resume"
    )
    (auxiliary / archive_builder.misc._TRACK_UPDATE_LOCK_NAME).write_bytes(b"lock")
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
    assert archive_builder.misc._TRACK_UPDATE_CHECKPOINT_NAME not in members
    assert archive_builder.misc._TRACK_UPDATE_LOCK_NAME not in members


def test_archive_builder_rejects_dirty_checkout(tmp_path):
    data_dir = _data_checkout(tmp_path)
    (data_dir / "auxiliary" / "uncommitted.txt").write_text("dirty")

    with pytest.raises(archive_builder.ArchiveError, match="clean"):
        archive_builder.build_archive(data_dir, output=tmp_path / "release.zip")


def test_archive_builder_rejects_semantically_invalid_track_database(tmp_path):
    data_dir = _data_checkout(tmp_path)
    filenames_path = data_dir / "auxiliary" / "CryoSat-2_SARIn_file_names.pkl"
    pd.Series(dtype="object").to_pickle(filenames_path)
    subprocess.run(["git", "-C", str(data_dir), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(data_dir), "commit", "-qm", "invalid database"],
        check=True,
    )
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

    with pytest.raises(archive_builder.ArchiveError, match="track database"):
        archive_builder.build_archive(data_dir, output=tmp_path / "release.zip")
