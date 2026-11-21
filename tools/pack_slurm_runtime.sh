#!/usr/bin/env bash
# Pack the locked CryoSwath Linux runtime and a wheel for offline Slurm use.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'Usage: %s CRYOSWATH_WHEEL OUTPUT_TAR\n' "$0" >&2
    exit 2
fi

wheel=$1
if [[ ! -f "$wheel" || "${wheel##*/}" != cryoswath-*.whl ]]; then
    printf 'Expected a CryoSwath wheel: %s\n' "$wheel" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec pixi-pack "$repo_root/pixi.toml" \
    --environment default --platform linux-64 --output-file "$2" \
    --ignore-pypi-non-wheel --inject "$wheel"
