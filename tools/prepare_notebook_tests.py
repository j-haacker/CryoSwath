"""Prepare isolated CryoSwath projects for notebook test workflows."""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser
from collections.abc import Iterable
from configparser import ConfigParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cryoswath import misc

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PROJECT = REPO_ROOT / "tests" / "reports" / "artifacts" / "project"
TUTORIAL_PROJECT = REPO_ROOT / "tests" / "tutorials" / "artifacts" / "project"

AUXILIARY_SENTINELS = (
    Path("data/auxiliary/CryoSat-2_SARIn_file_names.pkl"),
    Path("data/auxiliary/CryoSat-2_SARIn_ground_tracks.feather"),
    Path("data/auxiliary/RGI/RGI2000-v7.0-o1regions.feather"),
)

TUTORIAL_SUPPORT_FILES = {
    "arcticdem_mosaic_100m_v4.1_dem__excerpt_barnes-ice-cap.tif": (
        Path(
            "data/tutorials/arcticdem_mosaic_100m_v4.1_dem__excerpt_barnes-ice-cap.tif"
        ),
        Path(
            "data/auxiliary/DEM/"
            "arcticdem_mosaic_100m_v4.1_dem__excerpt_barnes-ice-cap.tif"
        ),
    ),
    "barnes_ice_cap.feather": (
        Path("data/tutorials/barnes_ice_cap.feather"),
        Path("data/auxiliary/RGI/barnes_ice_cap.feather"),
    ),
}


@dataclass(frozen=True)
class PreparedNotebookProject:
    project_dir: Path
    config_path: Path
    tutorial_dir: Path | None = None


PATH_OVERRIDE_ENV_VARS = (
    "CRYOSWATH_DATA",
    "CRYOSWATH_L1B",
    "CRYOSWATH_L2_SWATH",
    "CRYOSWATH_L2_POCA",
    "CRYOSWATH_L3",
    "CRYOSWATH_L4",
    "CRYOSWATH_TMP",
    "CRYOSWATH_AUX",
    "CRYOSWATH_DEM",
    "CRYOSWATH_RGI",
    "CRYOSWATH_CS_GROUND_TRACKS",
)


@contextmanager
def _using_cryoswath_config(config_path: Path):
    managed_env_vars = ("CRYOSWATH_CONFIG", *PATH_OVERRIDE_ENV_VARS)
    old_values = {name: os.environ.get(name) for name in managed_env_vars}
    os.environ["CRYOSWATH_CONFIG"] = str(config_path)
    for name in PATH_OVERRIDE_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _write_project_config(project_dir: Path) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "cryoswath.cfg"
    config = ConfigParser()
    if config_path.is_file():
        config.read(config_path)
    if "path" not in config:
        config["path"] = {}
    config["path"]["data"] = "data"
    with config_path.open("w") as file_obj:
        config.write(file_obj)
    (project_dir / "data").mkdir(parents=True, exist_ok=True)
    return config_path


def _missing_auxiliary_files(project_dir: Path) -> list[Path]:
    return [
        project_dir / relative_path
        for relative_path in AUXILIARY_SENTINELS
        if not (project_dir / relative_path).is_file()
    ]


def _format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _ensure_auxiliary_data(
    project_dir: Path,
    config_path: Path,
    *,
    timeout: int | float,
    skip_download: bool = False,
) -> None:
    missing = _missing_auxiliary_files(project_dir)
    if not missing:
        return
    if skip_download:
        raise RuntimeError(
            "Missing notebook auxiliary file(s): "
            f"{_format_paths(missing)}. Re-run without --skip-aux-download."
        )

    with _using_cryoswath_config(config_path):
        misc.download_auxiliary_data(base_dir=project_dir, timeout=timeout)

    missing_after_download = _missing_auxiliary_files(project_dir)
    if missing_after_download:
        raise RuntimeError(
            "Auxiliary-data setup finished but required notebook file(s) are "
            f"still missing: {_format_paths(missing_after_download)}"
        )


def _tutorial_support_source(repo_root: Path, filename: str) -> Path | None:
    candidates = (
        repo_root / "data" / "tutorials" / filename,
        repo_root / "tests" / "tutorials" / "resources" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _copy_tutorial_support_files(project_dir: Path, repo_root: Path) -> list[Path]:
    copied: list[Path] = []
    missing_sources: list[str] = []
    for filename, relative_destinations in TUTORIAL_SUPPORT_FILES.items():
        destinations = [
            project_dir / destination for destination in relative_destinations
        ]
        if all(destination.is_file() for destination in destinations):
            continue

        source = _tutorial_support_source(repo_root, filename)
        if source is None:
            missing_sources.append(filename)
            continue

        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)

    if missing_sources:
        raise FileNotFoundError(
            "Missing tutorial support file(s): "
            f"{', '.join(missing_sources)}. Expected them under "
            f"{repo_root / 'data' / 'tutorials'} or "
            f"{repo_root / 'tests' / 'tutorials' / 'resources'}."
        )
    return copied


def prepare_report_project(
    project_dir: str | Path = REPORT_PROJECT,
    *,
    timeout: int | float = 120,
    skip_aux_download: bool = False,
) -> PreparedNotebookProject:
    project_path = Path(project_dir).expanduser().resolve()
    config_path = _write_project_config(project_path)
    _ensure_auxiliary_data(
        project_path,
        config_path,
        timeout=timeout,
        skip_download=skip_aux_download,
    )
    return PreparedNotebookProject(project_path, config_path)


def prepare_tutorial_project(
    project_dir: str | Path = TUTORIAL_PROJECT,
    *,
    repo_root: str | Path = REPO_ROOT,
    timeout: int | float = 120,
    skip_aux_download: bool = False,
) -> PreparedNotebookProject:
    project_path = Path(project_dir).expanduser().resolve()
    repo_path = Path(repo_root).expanduser().resolve()
    config_path = _write_project_config(project_path)
    _ensure_auxiliary_data(
        project_path,
        config_path,
        timeout=timeout,
        skip_download=skip_aux_download,
    )
    _copy_tutorial_support_files(project_path, repo_path)
    tutorial_dir = Path(misc.copy_tutorials(base_dir=project_path, force=True))
    return PreparedNotebookProject(project_path, config_path, tutorial_dir)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Prepare isolated project directories for notebook tests."
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=("all", "reports", "tutorials"),
        default="all",
        help="Notebook workflow to prepare (default: all).",
    )
    parser.add_argument(
        "--report-project",
        default=REPORT_PROJECT,
        type=Path,
        help="Generated project directory for report notebooks.",
    )
    parser.add_argument(
        "--tutorial-project",
        default=TUTORIAL_PROJECT,
        type=Path,
        help="Generated project directory for tutorial notebooks.",
    )
    parser.add_argument(
        "--timeout",
        default=120,
        type=float,
        help="Timeout in seconds for auxiliary-data downloads.",
    )
    parser.add_argument(
        "--skip-aux-download",
        action="store_true",
        help="Fail if auxiliary data is missing instead of downloading it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepared: list[PreparedNotebookProject] = []

    if args.target in {"all", "reports"}:
        prepared.append(
            prepare_report_project(
                args.report_project,
                timeout=args.timeout,
                skip_aux_download=args.skip_aux_download,
            )
        )
    if args.target in {"all", "tutorials"}:
        prepared.append(
            prepare_tutorial_project(
                args.tutorial_project,
                timeout=args.timeout,
                skip_aux_download=args.skip_aux_download,
            )
        )

    for project in prepared:
        message = f"Prepared {project.project_dir} with {project.config_path}"
        if project.tutorial_dir is not None:
            message += f" and tutorials in {project.tutorial_dir}"
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
