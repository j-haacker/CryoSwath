"""Helpers for guarding Zarr stores with optional xzarrguard support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import xarray as xr

__all__ = [
    "guard_existing_store",
    "guarded_to_zarr",
    "is_xzarrguard_available",
]

_XZARRGUARD_MODULE = "xzarrguard"


def is_xzarrguard_available() -> bool:
    """Return whether xzarrguard can be imported in this interpreter."""
    return _load_xzarrguard_module() is not None


def guarded_to_zarr(
    dataset: xr.Dataset,
    store_path: str | Path,
    *,
    to_zarr_kwargs: Mapping[str, Any] | None = None,
    guard_kwargs: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    python_executable: str | None = None,
) -> Any:
    """Write a Zarr store and immediately guard it with xzarrguard.

    This helper keeps the write step explicit and only supports materialized
    writes. If ``compute=False`` is supplied, the caller must finish the write
    before calling :func:`guard_existing_store`.
    """

    write_kwargs = dict(to_zarr_kwargs or {})
    if write_kwargs.get("compute") is False:
        raise ValueError(
            "guarded_to_zarr requires a materialized write; call to_zarr with "
            "compute=True or guard the store afterwards."
        )
    write_kwargs.setdefault("compute", True)
    _maybe_default_write_empty_chunks(dataset, write_kwargs)

    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    result = dataset.to_zarr(store_path, **write_kwargs)
    guard_call_kwargs = dict(guard_kwargs or {})
    if command is not None:
        guard_call_kwargs["command"] = command
    if python_executable is not None:
        guard_call_kwargs["python_executable"] = python_executable
    guard_existing_store(
        store_path,
        **guard_call_kwargs,
    )
    return result


def guard_existing_store(
    store_path: str | Path,
    *,
    no_data_chunks: Mapping[str, Sequence[Sequence[int]]] | None = None,
    infer_no_data_from_store: bool = True,
    command: Sequence[str] | None = None,
    python_executable: str | None = None,
) -> None:
    """Mark a completed Zarr store as guarded with xzarrguard.

    The direct Python API is preferred when available. Otherwise, the helper
    falls back to invoking the xzarrguard CLI in a separate process.
    """

    store = Path(store_path)
    module = _load_xzarrguard_module()
    if module is not None:
        _guard_with_import(
            module,
            store,
            no_data_chunks=no_data_chunks,
            infer_no_data_from_store=infer_no_data_from_store,
        )
        return

    _guard_with_subprocess(
        store,
        no_data_chunks=no_data_chunks,
        infer_no_data_from_store=infer_no_data_from_store,
        command=command,
        python_executable=python_executable,
    )


def _maybe_default_write_empty_chunks(
    dataset: xr.Dataset,
    write_kwargs: dict[str, Any],
) -> None:
    """Default `write_empty_chunks=True` when the xarray API supports it."""

    if "write_empty_chunks" in write_kwargs:
        return
    params = inspect.signature(dataset.to_zarr).parameters
    if "write_empty_chunks" in params:
        write_kwargs["write_empty_chunks"] = True


def _load_xzarrguard_module() -> Any | None:
    """Import xzarrguard lazily so CryoSwath still imports on Python 3.11."""

    try:
        return importlib.import_module(_XZARRGUARD_MODULE)
    except Exception:
        return None


def _guard_with_import(
    module: Any,
    store: Path,
    *,
    no_data_chunks: Mapping[str, Sequence[Sequence[int]]] | None,
    infer_no_data_from_store: bool,
) -> None:
    """Use the xzarrguard Python API when it is importable."""

    kwargs: dict[str, Any] = {
        "in_place_metadata_only": True,
    }
    if no_data_chunks is not None:
        kwargs["no_data_chunks"] = {
            str(variable): [tuple(int(value) for value in coord) for coord in coords]
            for variable, coords in no_data_chunks.items()
        }
        kwargs["infer_no_data_from_store"] = False
    else:
        kwargs["infer_no_data_from_store"] = infer_no_data_from_store
    module.create_store(None, store, **kwargs)


def _guard_with_subprocess(
    store: Path,
    *,
    no_data_chunks: Mapping[str, Sequence[Sequence[int]]] | None,
    infer_no_data_from_store: bool,
    command: Sequence[str] | None,
    python_executable: str | None,
) -> None:
    """Fallback to a subprocess when xzarrguard cannot be imported."""

    if command is None:
        if python_executable is not None:
            command = [python_executable, "-m", _XZARRGUARD_MODULE]
        else:
            executable = shutil.which(_XZARRGUARD_MODULE)
            if executable is not None:
                command = [executable]
            else:
                raise RuntimeError(
                    "xzarrguard is not importable in this environment. Install it in "
                    "a Python >=3.12 environment, or pass command=/python_executable "
                    "to guard_existing_store()."
                )

    cli_args = [*command, "create", str(store), "--in-place-metadata-only"]
    tmp_path: Path | None = None
    if no_data_chunks is not None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="xzarrguard-no-data-",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(
                {
                    str(variable): [list(map(int, coord)) for coord in coords]
                    for variable, coords in no_data_chunks.items()
                },
                tmp_file,
                indent=2,
                sort_keys=True,
            )
            tmp_file.write("\n")
        cli_args.extend(["--no-data", str(tmp_path)])
    elif infer_no_data_from_store:
        cli_args.append("--infer-no-data-from-store")

    try:
        subprocess.run(cli_args, check=True)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
