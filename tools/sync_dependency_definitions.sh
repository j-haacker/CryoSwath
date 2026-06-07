#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: sync_dependency_definitions.sh [--check]

Synchronize dependency definitions in this order:
1. pixi.toml runtime dependencies and requirements.txt from pyproject.toml
2. environment.yml and docs/environment.yml from Pixi environments

Use --check to verify both are already synchronized.
EOF
}

mode="sync"
if [[ "${1:-}" == "--check" ]]; then
  mode="check"
  shift
fi

if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pixi_exe="${PIXI_EXE:-$(command -v pixi || true)}"
python_exe="${PYTHON:-$(command -v python3.12 || command -v python3 || command -v python || true)}"

if [[ -z "$pixi_exe" ]]; then
  echo "Could not find pixi executable. Set PIXI_EXE or install pixi." >&2
  exit 1
fi

if [[ -z "$python_exe" ]]; then
  echo "Could not find a Python interpreter." >&2
  exit 1
fi

if ! "$python_exe" -c '
import importlib.util
import sys

missing = []
if importlib.util.find_spec("packaging") is None:
    missing.append("packaging")
if importlib.util.find_spec("tomllib") is None and importlib.util.find_spec("tomli") is None:
    missing.append("tomllib-or-tomli")
if missing:
    raise SystemExit(
        "Selected Python interpreter is missing required modules: " + ", ".join(missing)
    )
'; then
  echo "Unable to run dependency sync with $python_exe." >&2
  echo "Install the missing Python module(s) or set PYTHON to a suitable interpreter." >&2
  exit 1
fi

if [[ "$mode" == "check" ]]; then
  "$python_exe" "$repo_root/tools/sync_pixi_runtime_deps_from_pyproject.py" --check
  PIXI_EXE="$pixi_exe" bash "$repo_root/tools/sync_environment_yml_from_pixi.sh" --check
  exit 0
fi

"$python_exe" "$repo_root/tools/sync_pixi_runtime_deps_from_pyproject.py"
PIXI_EXE="$pixi_exe" bash "$repo_root/tools/sync_environment_yml_from_pixi.sh"
