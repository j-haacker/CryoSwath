#!/usr/bin/env python3
"""Build CryoSwath and run unit tests against the installed wheel."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def build_and_check(repo_root: Path, work_dir: Path) -> Path:
    dist_dir = work_dir / "dist"
    dist_dir.mkdir()
    run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=repo_root)

    artifacts = sorted(dist_dir.iterdir())
    if not artifacts:
        raise RuntimeError("Build did not produce any distribution artifacts.")
    run([sys.executable, "-m", "twine", "check", *map(str, artifacts)], cwd=repo_root)

    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one wheel in {dist_dir}, found {len(wheels)}."
        )
    return wheels[0]


def copy_unit_tests(repo_root: Path, target: Path) -> list[Path]:
    source = repo_root / "tests"
    target.mkdir()
    copied: list[Path] = []
    for test_file in sorted(source.glob("test_*.py")):
        destination = target / test_file.name
        shutil.copy2(test_file, destination)
        copied.append(destination)
    if not copied:
        raise RuntimeError(f"No unit tests matching test_*.py found in {source}.")
    return copied


def write_import_guard(repo_root: Path, target: Path) -> Path:
    guard = target / "test_installed_package_guard.py"
    guard.write_text(
        "from pathlib import Path\n\n"
        "\n"
        "def test_cryoswath_imports_from_installed_wheel():\n"
        "    import cryoswath\n\n"
        f"    repo_root = Path({str(repo_root)!r}).resolve()\n"
        "    package_file = Path(cryoswath.__file__).resolve()\n"
        "    assert repo_root not in package_file.parents, (\n"
        "        f'cryoswath imported from source checkout: {package_file}'\n"
        "    )\n"
    )
    return guard


def run_installed_tests(repo_root: Path, wheel: Path, work_dir: Path) -> None:
    venv_dir = work_dir / "venv"
    tests_dir = work_dir / "tests"
    run_dir = work_dir / "run"
    run_dir.mkdir()

    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    python = venv_python(venv_dir)

    run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "pytest"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(wheel),
        ]
    )

    copied_tests = copy_unit_tests(repo_root, tests_dir)
    guard_test = write_import_guard(repo_root, tests_dir)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CRYOSWATH_DATA"] = str(work_dir / "data")

    run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            str(guard_test),
            *map(str, copied_tests),
        ],
        cwd=run_dir,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-check-only",
        action="store_true",
        help=(
            "Build distributions and run twine check without creating the "
            "installed-test venv."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary build/test directory for debugging.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    temp_context = tempfile.TemporaryDirectory(prefix="cryoswath-installed-test-")
    work_dir = Path(temp_context.name)

    try:
        print(f"Using temporary work directory: {work_dir}", flush=True)
        wheel = build_and_check(repo_root, work_dir)
        if not args.package_check_only:
            run_installed_tests(repo_root, wheel, work_dir)
        if args.keep_temp:
            temp_context.cleanup = lambda: None  # type: ignore[method-assign]
            print(f"Kept temporary work directory: {work_dir}", flush=True)
    finally:
        temp_context.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
