#!/usr/bin/env python3
"""Run a Pixi test task from a clean checkout and fresh home directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

DEFAULT_ENVIRONMENT = "test"
DEFAULT_TASK = "test-all"
DEFAULT_REF = "HEAD"
DEFAULT_SOURCE = "worktree"

DEFAULT_PASSTHROUGH_ENV = (
    "EOIAM_USER",
    "EOIAM_PASSWORD",
    "EARTHDATA_USERNAME",
    "EARTHDATA_PASSWORD",
    "EARTHDATA_TOKEN",
)

BASE_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "GIT_SSL_CAINFO",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "PIXI_CACHE_DIR",
    "CONDA_PKGS_DIRS",
    "MAMBA_ROOT_PREFIX",
    "PIP_CACHE_DIR",
    "TMPDIR",
)

LOCAL_STATE_ENV_PREFIXES = ("CRYOSWATH_",)
LOCAL_STATE_ENV = ("PYTHONPATH",)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def tracked_worktree_dirty(repo_root: Path) -> bool:
    status = git_output(repo_root, "status", "--porcelain", "--untracked-files=no")
    return bool(status.strip())


def export_committed_source(repo_root: Path, destination: Path, ref: str) -> None:
    archive_path = destination.parent / "source.tar"
    run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output",
            str(archive_path),
            ref,
        ],
        cwd=repo_root,
    )
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination, filter="data")


def copy_tracked_worktree(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        relative_path = Path(raw_name.decode())
        source = repo_root / relative_path
        target = destination / relative_path
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)


def _keep_from_current_env(name: str) -> bool:
    return name in BASE_ENV or name in DEFAULT_PASSTHROUGH_ENV


def fresh_environment(work_dir: Path, pass_env: list[str]) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if _keep_from_current_env(name)
    }
    for name in pass_env:
        if name in os.environ:
            env[name] = os.environ[name]

    for name in list(env):
        if name in LOCAL_STATE_ENV or any(
            name.startswith(prefix) for prefix in LOCAL_STATE_ENV_PREFIXES
        ):
            if name not in pass_env:
                env.pop(name)

    home = work_dir / "home"
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    return env


def copy_netrc(source_home: Path, target_home: Path) -> None:
    source = source_home / ".netrc"
    if not source.is_file():
        raise FileNotFoundError(f"No .netrc file found at {source}.")
    target_home.mkdir(parents=True, exist_ok=True)
    target = target_home / ".netrc"
    shutil.copy2(source, target)
    target.chmod(0o600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        "-e",
        default=DEFAULT_ENVIRONMENT,
        help="Pixi environment to install and run (default: test).",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Pixi task to run inside the fresh checkout (default: test-all).",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git ref to export when --source committed is used (default: HEAD).",
    )
    parser.add_argument(
        "--source",
        choices=("committed", "worktree"),
        default=DEFAULT_SOURCE,
        help=(
            "Use committed git archive or a copy of tracked worktree files "
            "(default: worktree)."
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow --source committed even when tracked worktree files are dirty.",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Pass an additional environment variable through to the fresh run. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--copy-netrc",
        action="store_true",
        help="Copy ~/.netrc into the fresh HOME for credentialed downloads.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pixi install and run the selected task directly.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary checkout/home directory for debugging.",
    )
    parser.add_argument(
        "--pixi",
        default=shutil.which("pixi") or "pixi",
        help="Pixi executable to use (default: first pixi on PATH).",
    )
    parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the Pixi task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    if args.source == "committed" and not args.allow_dirty:
        if tracked_worktree_dirty(repo_root):
            raise SystemExit(
                "Tracked working tree files are dirty. Commit/stash them, use "
                "--source worktree, or pass --allow-dirty to test committed HEAD."
            )

    temp_context = tempfile.TemporaryDirectory(prefix="cryoswath-fresh-test-")
    work_dir = Path(temp_context.name)
    checkout = work_dir / "checkout"

    try:
        print(f"Using temporary work directory: {work_dir}", flush=True)
        if args.source == "committed":
            export_committed_source(repo_root, checkout, args.ref)
        else:
            copy_tracked_worktree(repo_root, checkout)

        env = fresh_environment(work_dir, args.pass_env)
        if args.copy_netrc:
            copy_netrc(Path.home(), Path(env["HOME"]))

        if not args.skip_install:
            run(
                [args.pixi, "install", "--locked", "-e", args.environment],
                cwd=checkout,
                env=env,
            )
        task_args = (
            args.task_args[1:] if args.task_args[:1] == ["--"] else args.task_args
        )
        run(
            [args.pixi, "run", "-e", args.environment, args.task, *task_args],
            cwd=checkout,
            env=env,
        )

        if args.keep_temp:
            temp_context.cleanup = lambda: None  # type: ignore[method-assign]
            print(f"Kept temporary work directory: {work_dir}", flush=True)
    finally:
        temp_context.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
