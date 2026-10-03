"""Validate local CryoSat-2 ground-track and filename caches.

This is a maintainer check: it reads only local files and reports plausible
coverage irregularities without attempting to discover expected remote tracks.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryoswath import misc

_START_MONTH = pd.Period("2010-07", freq="M")
_STARTUP_END_MONTH = pd.Period("2010-12", freq="M")
_MAX_GAP = pd.Timedelta(days=14)
_TAIL_GRACE_MONTHS = 3
_MIN_SEASONAL_COUNT = 20
_FILENAME_ONLY_WARNING_FRACTION = 0.05
_FILENAME_ONLY_ERROR_FRACTION = 0.10
_FILENAME_PATTERN = (
    r"^CS_(?:OFFL|LTA_)_SIR_SIN_1B_"
    r"(?P<start>\d{8}T\d{6})_\d{8}T\d{6}_[A-Z]\d{3}$"
)


class ValidationError(RuntimeError):
    """Raised when a cache is structurally invalid."""


@dataclass
class ValidationReport:
    track_count: int
    filename_count: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    warnings: list[str]


def _datetime_index(index: pd.Index, label: str) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(index, errors="coerce")
    if timestamps.isna().any():
        raise ValidationError(f"{label} contains invalid or null timestamps.")
    timestamps = pd.DatetimeIndex(timestamps)
    if timestamps.has_duplicates:
        raise ValidationError(f"{label} contains duplicate timestamps.")
    return timestamps


def _read_tracks(
    path: Path,
) -> tuple[gpd.GeoDataFrame, pd.DatetimeIndex, list[str]]:
    try:
        tracks = gpd.read_feather(path)
    except Exception as err:
        raise ValidationError(f"Could not read ground tracks at {path}: {err}") from err
    if "geometry" not in tracks:
        raise ValidationError("Ground tracks have no geometry column.")
    timestamps = _datetime_index(tracks.index, "Ground tracks")
    if not len(tracks):
        raise ValidationError("Ground tracks are empty.")
    geometry = tracks.geometry
    if geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all():
        raise ValidationError(
            "Ground tracks contain missing, empty, or invalid geometry."
        )
    if not geometry.geom_type.eq("LineString").all():
        raise ValidationError("Ground tracks must contain LineString geometry.")
    bounds = tracks.total_bounds
    if (
        not pd.notna(bounds).all()
        or bounds[0] < -180
        or bounds[2] > 180
        or bounds[1] < -90
        or bounds[3] > 90
    ):
        raise ValidationError("Ground-track coordinates exceed geographic bounds.")
    warnings = []
    if tracks.crs is None:
        tracks = tracks.set_crs(4326)
        warnings.append(
            "ground tracks have missing CRS metadata; inferred WGS84 (EPSG:4326) "
            "from valid geographic coordinate bounds."
        )
    elif tracks.crs.to_epsg() != 4326:
        raise ValidationError("Ground tracks must use WGS84 (EPSG:4326).")
    return tracks, timestamps, warnings


def _read_filenames(path: Path) -> tuple[pd.Series, pd.DatetimeIndex]:
    try:
        filenames = pd.read_pickle(path)
    except Exception as err:
        raise ValidationError(
            f"Could not read filename catalog at {path}: {err}"
        ) from err
    if not isinstance(filenames, pd.Series):
        raise ValidationError("Filename catalog must be a pandas Series.")
    timestamps = _datetime_index(filenames.index, "Filename catalog")
    if filenames.isna().any():
        raise ValidationError("Filename catalog contains null values.")
    filename_parts = filenames.astype("string").str.extract(_FILENAME_PATTERN)
    malformed = filename_parts["start"].isna()
    if malformed.any():
        raise ValidationError(
            f"Filename catalog contains {malformed.sum()} malformed product names."
        )
    embedded_times = pd.DatetimeIndex(
        pd.to_datetime(filename_parts["start"], format="%Y%m%dT%H%M%S")
    )
    mismatched = embedded_times != timestamps
    if mismatched.any():
        raise ValidationError(
            f"Filename catalog contains {mismatched.sum()} product timestamps "
            "that do not match their index."
        )
    return filenames, timestamps


def _coverage_warnings(timestamps: pd.DatetimeIndex, now: pd.Timestamp) -> list[str]:
    ordered = timestamps.sort_values()
    first_month = ordered[0].to_period("M")
    last_month = ordered[-1].to_period("M")
    warnings: list[str] = []
    if first_month != _START_MONTH:
        warnings.append(
            f"coverage begins in {first_month}, expected {_START_MONTH} (mid-2010)."
        )

    observed_months = pd.PeriodIndex(ordered.to_period("M").unique(), freq="M")
    expected_months = pd.period_range(first_month, last_month, freq="M")
    missing = expected_months.difference(observed_months)
    if len(missing):
        warnings.append(
            "missing monthly coverage: " + ", ".join(map(str, missing.tolist())) + "."
        )

    complete_end = last_month - 1
    baseline_months = pd.period_range(_STARTUP_END_MONTH + 1, complete_end, freq="M")
    counts = pd.Series(1, index=ordered.to_period("M")).groupby(level=0).sum()
    for month in baseline_months:
        history = counts[(counts.index.month == month.month) & (counts.index < month)]
        if len(history) < 3 or month not in counts:
            continue
        median = history.median()
        mad = (history - median).abs().median()
        if median < _MIN_SEASONAL_COUNT:
            continue
        threshold = max(median * 0.5, median - 6 * mad)
        if counts[month] < threshold:
            warnings.append(
                f"low track count in {month}: {counts[month]} "
                f"(seasonal median {median:.0f}, MAD {mad:.0f})."
            )

    post_startup = ordered[ordered.to_period("M") > _STARTUP_END_MONTH]
    gaps = pd.Series(post_startup).diff()
    if gaps.max() > _MAX_GAP:
        gap_at = gaps.idxmax()
        warnings.append(
            f"largest post-startup gap is {gaps.loc[gap_at]} "
            f"before {post_startup[gap_at]}."
        )

    if now.to_period("M").ordinal - last_month.ordinal > _TAIL_GRACE_MONTHS:
        warnings.append(
            f"latest track month is {last_month}, more than {_TAIL_GRACE_MONTHS} "
            "completed calendar months behind."
        )
    return warnings


def validate_track_database(
    tracks_path: str | Path,
    filenames_path: str | Path,
    *,
    now: str | pd.Timestamp | None = None,
) -> ValidationReport:
    """Validate local caches and return coverage warnings without remote access."""
    _, track_times, warnings = _read_tracks(Path(tracks_path))
    _, filename_times = _read_filenames(Path(filenames_path))
    tracks_without_filenames = track_times[~track_times.isin(filename_times)]
    if len(tracks_without_filenames):
        raise ValidationError(
            f"Ground tracks contain {len(tracks_without_filenames)} timestamps "
            "without filename entries."
        )
    current_time = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    warnings.extend(_coverage_warnings(track_times, current_time))
    filename_only = filename_times[~filename_times.isin(track_times)]
    if len(filename_only):
        filename_months = filename_times.to_period("M")
        filename_counts = pd.Series(1, index=filename_months).groupby(level=0).sum()
        filename_only_counts = (
            pd.Series(1, index=filename_only.to_period("M")).groupby(level=0).sum()
        )
        monthly_fractions = (
            filename_only_counts / filename_counts[filename_only_counts.index]
        )
        worst_month = monthly_fractions.idxmax()
        worst_fraction = monthly_fractions.loc[worst_month]
        if worst_fraction > _FILENAME_ONLY_ERROR_FRACTION:
            raise ValidationError(
                "Filename entries without ground-track geometry account for "
                f"{worst_fraction:.1%} of {worst_month}, exceeding the "
                f"{_FILENAME_ONLY_ERROR_FRACTION:.0%} publication limit."
            )
        qualifier = (
            " elevated" if worst_fraction > _FILENAME_ONLY_WARNING_FRACTION else ""
        )
        warnings.append(
            f"filename catalog has {len(filename_only)} entries without "
            f"ground-track geometry ({len(filename_only) / len(filename_times):.3%} "
            f"overall;{qualifier} maximum {worst_fraction:.1%} in {worst_month})."
        )
    return ValidationReport(
        track_count=len(track_times),
        filename_count=len(filename_times),
        first_timestamp=track_times.min(),
        last_timestamp=track_times.max(),
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Project base directory.")
    args = parser.parse_args()
    _, _, paths = misc._resolve_path_configuration(cwd=Path(args.base_dir))
    filenames_path = Path(paths["aux"]) / "CryoSat-2_SARIn_file_names.pkl"
    try:
        report = validate_track_database(paths["cs_ground_tracks"], filenames_path)
    except ValidationError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        raise SystemExit(1) from err
    print(f"Tracks: {report.track_count}")
    print(f"Filename entries: {report.filename_count}")
    print(f"Coverage: {report.first_timestamp} to {report.last_timestamp}")
    print(f"Startup exemption: {_START_MONTH} through {_STARTUP_END_MONTH}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
