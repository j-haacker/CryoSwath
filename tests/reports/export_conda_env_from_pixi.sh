#!/usr/bin/env bash
set -euo pipefail

pixi_exe="${PIXI_EXE:-$(command -v pixi || true)}"
if [[ -z "$pixi_exe" ]]; then
  echo "Could not find pixi executable. Set PIXI_EXE or run via pixi." >&2
  exit 1
fi

repo_root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
out_dir="$repo_root/tests/reports"
env_yaml="$out_dir/cryoswath-test.yaml"
pin_file="$out_dir/cryoswath-test.linux-64.pin.txt"
tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$out_dir"

if "$pixi_exe" workspace export conda-environment -e test -p linux-64 "$env_yaml" 2>/dev/null; then
  :
else
  echo "Failed to export conda environment YAML from pixi." >&2
  exit 1
fi

if "$pixi_exe" workspace export conda-explicit-spec -e test -p linux-64 --ignore-pypi-errors --ignore-source-errors "$tmp_dir" 2>/dev/null; then
  :
elif "$pixi_exe" workspace export conda-explicit-spec -e test -p linux-64 --ignore-pypi-errors "$tmp_dir" 2>/dev/null; then
  :
else
  echo "Failed to export conda explicit spec from pixi." >&2
  exit 1
fi

generated_pin="$(find "$tmp_dir" -maxdepth 1 -type f | sort | head -n 1)"
if [[ -z "$generated_pin" ]]; then
  echo "Pixi did not emit any explicit spec file in $tmp_dir." >&2
  exit 1
fi

cp "$generated_pin" "$pin_file"
echo "Wrote $env_yaml"
echo "Wrote $pin_file"
