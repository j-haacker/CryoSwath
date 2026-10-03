"""Build a reviewable CryoSwath auxiliary-data archive for manual Zenodo upload."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_track_database import ValidationError, validate_track_database

from cryoswath import misc

_ARCHIVE_PREFIX = "CryoSwath-aux-data"
_REQUIRED_MEMBERS = {
    "CryoSat-2_SARIn_ground_tracks.feather",
    "CryoSat-2_SARIn_file_names.pkl",
}


class ArchiveError(RuntimeError):
    """Raised when an archive cannot safely be built."""


def _git(data_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(data_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ArchiveError(result.stderr.strip() or "Git command failed.")
    return result.stdout.strip()


def _source_commit(data_dir: Path) -> str:
    transient_files = {
        f"auxiliary/{misc._TRACK_UPDATE_CHECKPOINT_NAME}",
        f"auxiliary/{misc._TRACK_UPDATE_LOCK_NAME}",
    }
    dirty_files = [
        entry
        for entry in _git(data_dir, "status", "--porcelain").splitlines()
        if entry[3:] not in transient_files
    ]
    if dirty_files:
        raise ArchiveError("Data checkout must be clean.")
    head = _git(data_dir, "rev-parse", "HEAD")
    remote_data = _git(data_dir, "rev-parse", "origin/data")
    if head != remote_data:
        raise ArchiveError("Data checkout HEAD must match origin/data.")
    return head


def _include(path: Path) -> bool:
    return not (
        path.name.startswith(f"{_ARCHIVE_PREFIX}") and path.suffix == ".zip"
    ) and path.name not in {
        misc._TRACK_UPDATE_CHECKPOINT_NAME,
        misc._TRACK_UPDATE_LOCK_NAME,
    }


def _default_output(output_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{_ARCHIVE_PREFIX}-{timestamp}.zip"


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(path: Path) -> list[str]:
    misc._validate_zip_members(path)
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ArchiveError(f"Archive CRC check failed for {corrupt_member}.")
        members = [
            member.filename for member in archive.infolist() if not member.is_dir()
        ]
    missing = _REQUIRED_MEMBERS.difference(members)
    if missing:
        raise ArchiveError(
            "Archive lacks required members: " + ", ".join(sorted(missing))
        )
    if not any(member.startswith("RGI/") for member in members):
        raise ArchiveError("Archive lacks RGI metadata.")
    return members


def build_archive(
    data_dir: str | Path,
    *,
    output: str | Path | None = None,
    output_dir: str | Path = ".",
) -> tuple[Path, str, list[str]]:
    """Build an auxiliary archive from a clean checkout at ``origin/data``."""
    data_dir = Path(data_dir).resolve()
    commit = _source_commit(data_dir)
    auxiliary = data_dir / "auxiliary"
    if not auxiliary.is_dir():
        raise ArchiveError(f"No auxiliary directory in {data_dir}.")
    try:
        validate_track_database(
            auxiliary / "CryoSat-2_SARIn_ground_tracks.feather",
            auxiliary / "CryoSat-2_SARIn_file_names.pkl",
        )
    except ValidationError as err:
        raise ArchiveError(f"Invalid track database: {err}") from err
    target = Path(output) if output is not None else _default_output(Path(output_dir))
    target = target.resolve()
    if target.exists():
        raise ArchiveError(f"Refusing to overwrite existing archive {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(auxiliary.rglob("*")):
                if source.is_file() and _include(source):
                    archive.write(source, source.relative_to(auxiliary).as_posix())
        members = _validate_archive(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, commit, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Clean origin/data checkout.")
    parser.add_argument("--output", help="Explicit output archive path.")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for timestamped output."
    )
    args = parser.parse_args()
    try:
        archive, commit, members = build_archive(
            args.data_dir, output=args.output, output_dir=args.output_dir
        )
    except ArchiveError as err:
        parser.error(str(err))
    print(f"Archive: {archive}")
    print(f"Source commit: {commit}")
    print(f"Created: {datetime.now(UTC).isoformat()}")
    print(f"Members ({len(members)}):")
    print("\n".join(members))
    print(f"Size: {archive.stat().st_size}")
    print(f"SHA-256: {_checksum(archive)}")


if __name__ == "__main__":
    main()
