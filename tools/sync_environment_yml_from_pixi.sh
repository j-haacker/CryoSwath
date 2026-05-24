#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: sync_environment_yml_from_pixi.sh [--check]

Synchronize environment.yml from the Pixi default environment.
Use --check to verify the file is already in sync.
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

pixi_exe="${PIXI_EXE:-$(command -v pixi || true)}"
if [[ -z "$pixi_exe" ]]; then
  echo "Could not find pixi executable. Set PIXI_EXE or install pixi." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_yaml="$repo_root/environment.yml"
tmp_yaml="$(mktemp)"
docs_yaml="$(mktemp)"

cleanup() {
  rm -f "$tmp_yaml" "$docs_yaml"
}
trap cleanup EXIT

"$pixi_exe" workspace export conda-environment \
  --manifest-path "$repo_root" \
  -e default \
  -p linux-64 \
  "$tmp_yaml" >/dev/null
"$pixi_exe" workspace export conda-environment \
  --manifest-path "$repo_root" \
  -e docs \
  -p linux-64 \
  "$docs_yaml" >/dev/null

if [[ "$mode" == "check" ]]; then
  if [[ -f "$env_yaml" ]] && cmp -s "$env_yaml" "$tmp_yaml"; then
    echo "environment.yml matches the Pixi default environment."
    exit 0
  fi

  echo "environment.yml is out of sync with the Pixi default environment." >&2
  echo "Run: bash tools/sync_environment_yml_from_pixi.sh" >&2
  if [[ -f "$env_yaml" ]]; then
    diff -u "$env_yaml" "$tmp_yaml" || true
  else
    echo "environment.yml does not exist yet." >&2
  fi
  exit 1
fi

if [[ -f "$env_yaml" ]] && cmp -s "$env_yaml" "$tmp_yaml"; then
  echo "environment.yml already matches the Pixi default environment."
  exit 0
fi

mv "$tmp_yaml" "$env_yaml"
mv "$docs_yaml" "$repo_root/docs/environment.yml"
echo "Updated $env_yaml from the Pixi default environment."
