#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from packaging.requirements import Requirement

NAME_NORMALIZER = re.compile(r"[-_.]+")
DEPENDENCIES_BLOCK_PATTERNS = (
    re.compile(
        r"(?ms)^(?P<header>\[feature\.runtime\.dependencies\]\n)(?P<body>.*?)(?=^\[)"
    ),
    re.compile(r"(?ms)^(?P<header>\[dependencies\]\n)(?P<body>.*?)(?=^\[)"),
)
REQUIREMENTS_RUNTIME_HEADER = (
    "## CryoSwath dependencies #############################################"
)
REQUIREMENTS_OPTIONAL_HEADER = (
    "## CryoSwath optional #################################################"
)


@dataclass(frozen=True)
class RuntimeDependency:
    pypi_name: str
    spec: str
    source: str


def normalize_name(name: str) -> str:
    return NAME_NORMALIZER.sub("-", name).lower()


def clean_spec(spec: str) -> str:
    return spec.replace(" ", "").strip()


def parse_pyproject_dependency(raw_requirement: str) -> RuntimeDependency:
    requirement = Requirement(raw_requirement)
    requirement_text = raw_requirement.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*(.*)$", requirement_text)
    if match is None:
        raise ValueError(f"Could not parse dependency: {raw_requirement}")
    return RuntimeDependency(
        pypi_name=normalize_name(requirement.name),
        spec=clean_spec(match.group(2)),
        source=raw_requirement,
    )


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def load_pyproject_runtime_dependencies(
    pyproject_path: Path,
) -> list[RuntimeDependency]:
    data = load_toml(pyproject_path)
    deps: list[RuntimeDependency] = []
    seen: set[str] = set()
    for raw_requirement in data["project"].get("dependencies", []):
        dep = parse_pyproject_dependency(raw_requirement)
        if dep.pypi_name in seen:
            raise ValueError(
                f"Duplicate runtime dependency in {pyproject_path}: {dep.pypi_name}"
            )
        seen.add(dep.pypi_name)
        deps.append(dep)
    return deps


def load_conda_pypi_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    return {name: normalize_name(pypi_name) for name, pypi_name in data.items()}


def reverse_conda_pypi_map(conda_to_pypi: dict[str, str]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for conda_name, pypi_name in conda_to_pypi.items():
        reverse.setdefault(pypi_name, []).append(conda_name)
    return reverse


def load_current_pixi_runtime_dependencies(
    pixi_path: Path, conda_to_pypi: dict[str, str]
) -> tuple[str, dict[str, tuple[str, str]]]:
    data = load_toml(pixi_path)
    runtime_feature = data.get("feature", {}).get("runtime", {})
    pixi_deps = runtime_feature.get("dependencies", data.get("dependencies"))
    if pixi_deps is None:
        raise ValueError(
            "Could not find runtime dependency block in pixi.toml. Expected either "
            "[feature.runtime.dependencies] or top-level [dependencies]."
        )
    python_spec = str(pixi_deps["python"])

    current: dict[str, tuple[str, str]] = {}
    for conda_name, raw_spec in pixi_deps.items():
        if conda_name == "python":
            continue
        pypi_name = conda_to_pypi.get(conda_name, normalize_name(conda_name))
        spec = "" if str(raw_spec) == "*" else clean_spec(str(raw_spec))
        current[pypi_name] = (conda_name, spec)
    return python_spec, current


def choose_conda_name(
    dependency: RuntimeDependency,
    current_pixi_deps: dict[str, tuple[str, str]],
    pypi_to_conda: dict[str, list[str]],
) -> str:
    if dependency.pypi_name in current_pixi_deps:
        return current_pixi_deps[dependency.pypi_name][0]

    candidates = pypi_to_conda.get(dependency.pypi_name, [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "No conda package mapping known for runtime dependency "
            f"{dependency.source!r}. Update conda-pypi-map.json or pixi.toml."
        )
    raise ValueError(
        "Ambiguous conda package mapping for "
        f"{dependency.source!r}: {', '.join(candidates)}"
    )


def desired_pixi_dependencies(
    pyproject_deps: list[RuntimeDependency],
    python_spec: str,
    current_pixi_deps: dict[str, tuple[str, str]],
    pypi_to_conda: dict[str, list[str]],
) -> list[tuple[str, str]]:
    desired: list[tuple[str, str]] = [("python", python_spec)]
    used_names: set[str] = {"python"}

    for dep in pyproject_deps:
        conda_name = choose_conda_name(dep, current_pixi_deps, pypi_to_conda)
        if conda_name in used_names:
            raise ValueError(
                f"Duplicate conda dependency target selected: {conda_name}"
            )
        used_names.add(conda_name)
        desired.append((conda_name, dep.spec or "*"))

    return desired


def build_dependency_block(entries: list[tuple[str, str]]) -> str:
    lines = ["[dependencies]"]
    for name, spec in entries:
        escaped_spec = spec.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{name} = "{escaped_spec}"')
    return "\n".join(lines) + "\n\n"


def replace_dependencies_block(pixi_text: str, new_block: str) -> str:
    for pattern in DEPENDENCIES_BLOCK_PATTERNS:
        match = pattern.search(pixi_text)
        if match is not None:
            return pixi_text[: match.start()] + new_block + pixi_text[match.end() :]
    raise ValueError(
        "Could not locate runtime dependency block in pixi.toml. Expected either "
        "[feature.runtime.dependencies] or top-level [dependencies]."
    )


def build_requirements_text(
    requirements_text: str, entries: list[tuple[str, str]]
) -> str:
    runtime_start = requirements_text.find(REQUIREMENTS_RUNTIME_HEADER)
    if runtime_start == -1:
        raise ValueError(
            "Could not locate runtime dependency block in requirements.txt. "
            f"Expected header: {REQUIREMENTS_RUNTIME_HEADER}"
        )

    optional_start = requirements_text.find(REQUIREMENTS_OPTIONAL_HEADER, runtime_start)
    if optional_start == -1:
        optional_block = ""
    else:
        optional_block = requirements_text[optional_start:].lstrip("\n")

    prefix = requirements_text[:runtime_start]
    requirement_lines = []
    for name, spec in entries:
        if name == "python":
            continue
        requirement_lines.append(f"{name}{'' if spec == '*' else spec}")

    new_text = (
        prefix
        + REQUIREMENTS_RUNTIME_HEADER
        + "\n"
        + "\n".join(requirement_lines)
        + "\n"
    )
    if optional_block:
        new_text += "\n" + optional_block
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text


def compare_runtime_dependencies(
    pyproject_deps: list[RuntimeDependency],
    current_pixi_deps: dict[str, tuple[str, str]],
) -> list[str]:
    issues: list[str] = []
    pyproject_by_name = {dep.pypi_name: dep for dep in pyproject_deps}

    missing_in_pixi = sorted(set(pyproject_by_name) - set(current_pixi_deps))
    if missing_in_pixi:
        joined = ", ".join(missing_in_pixi)
        issues.append(f"Missing from pixi default [dependencies]: {joined}")

    extra_in_pixi = sorted(set(current_pixi_deps) - set(pyproject_by_name))
    if extra_in_pixi:
        joined = ", ".join(extra_in_pixi)
        issues.append(f"Present only in pixi default [dependencies]: {joined}")

    for name in sorted(set(pyproject_by_name) & set(current_pixi_deps)):
        pyproject_spec = pyproject_by_name[name].spec
        pixi_spec = current_pixi_deps[name][1]
        if pyproject_spec != pixi_spec:
            display_pyproject = pyproject_spec or "*"
            display_pixi = pixi_spec or "*"
            issues.append(
                f"Constraint mismatch for {name}: pyproject={display_pyproject}, "
                f"pixi={display_pixi}"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize Pixi and requirements runtime dependencies from "
            "pyproject.toml. pyproject.toml is treated as the source of truth "
            "for package runtime metadata."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify pixi.toml is already synchronized instead of editing it.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    pixi_path = repo_root / "pixi.toml"
    requirements_path = repo_root / "requirements.txt"
    conda_pypi_map_path = repo_root / "conda-pypi-map.json"

    try:
        pyproject_deps = load_pyproject_runtime_dependencies(pyproject_path)
        conda_to_pypi = load_conda_pypi_map(conda_pypi_map_path)
        pypi_to_conda = reverse_conda_pypi_map(conda_to_pypi)
        python_spec, current_pixi_deps = load_current_pixi_runtime_dependencies(
            pixi_path, conda_to_pypi
        )
        desired_entries = desired_pixi_dependencies(
            pyproject_deps, python_spec, current_pixi_deps, pypi_to_conda
        )
        requirements_text = requirements_path.read_text()
        desired_requirements_text = build_requirements_text(
            requirements_text, desired_entries
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    issues = compare_runtime_dependencies(pyproject_deps, current_pixi_deps)
    if requirements_text != desired_requirements_text:
        issues.append("requirements.txt is out of sync with pyproject.toml")

    if args.check:
        if not issues:
            print(
                "pixi.toml and requirements.txt runtime dependencies match "
                "pyproject.toml."
            )
            return 0
        print(
            "Runtime dependency definitions drift from pyproject.toml:", file=sys.stderr
        )
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        print(
            "Run: python tools/sync_pixi_runtime_deps_from_pyproject.py",
            file=sys.stderr,
        )
        return 1

    pixi_text = pixi_path.read_text()
    new_pixi_text = replace_dependencies_block(
        pixi_text, build_dependency_block(desired_entries)
    )

    changed = False

    if pixi_text != new_pixi_text:
        pixi_path.write_text(new_pixi_text)
        print("Synchronized pixi.toml runtime dependencies from pyproject.toml.")
        changed = True

    if requirements_text != desired_requirements_text:
        requirements_path.write_text(desired_requirements_text)
        print("Synchronized requirements.txt runtime dependencies from pyproject.toml.")
        changed = True

    if not changed:
        print(
            "pixi.toml and requirements.txt runtime dependencies already match "
            "pyproject.toml."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
