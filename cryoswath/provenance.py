"""Provenance and history helpers for CryoSwath outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version as package_version
from inspect import signature
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

try:  # optional import for richer serialization support
    import numpy as np
except Exception:  # pragma: no cover - numpy is a runtime dependency
    np = None

try:  # optional import for richer serialization support
    import pandas as pd
except Exception:  # pragma: no cover - pandas is a runtime dependency
    pd = None

__all__ = [
    "InputReference",
    "ProvenanceStep",
    "ProvenanceRecord",
    "append_history",
    "build_provenance_record",
    "build_provenance_step",
    "capture_call_arguments",
    "coerce_input_reference",
    "format_history_line",
    "load_provenance_sidecar",
    "package_revision",
    "provenance_path",
    "serialize_provenance",
    "write_provenance_sidecar",
]

PROVENANCE_DIRNAME = ".cryoswath"
PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 1
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _package_version() -> str:
    try:
        return package_version("cryoswath")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if pyproject.is_file():
            try:
                import tomllib

                with pyproject.open("rb") as handle:
                    return str(tomllib.load(handle)["project"]["version"])
            except Exception:
                pass
    except Exception:
        pass
    return "unknown"


def _normalize_git_sha(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not _GIT_SHA_RE.fullmatch(value):
        return None
    return value.lower()


def _resolve_git_commit() -> str | None:
    for key in ("READTHEDOCS_GIT_COMMIT_HASH", "CRYOSWATH_GIT_COMMIT", "GIT_COMMIT"):
        commit = _normalize_git_sha(os.environ.get(key))
        if commit:
            return commit

    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return _normalize_git_sha(result.stdout)


def package_revision() -> dict[str, str | None]:
    """Return the package version and git revision if available."""

    return {
        "version": _package_version(),
        "commit_hash": _resolve_git_commit(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _callable_name(func: Callable[..., Any] | str) -> str:
    if isinstance(func, str):
        return func
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else qualname


def _normalize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
    if isinstance(value, date):
        return value.isoformat()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if pd is not None:
        if isinstance(value, pd.Timestamp):
            return _normalize_scalar(value.to_pydatetime())
        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, pd.Index):
            return [_normalize_value(item) for item in value.tolist()]
    return value


def _normalize_value(value: Any) -> Any:
    value = _normalize_scalar(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {field.name: _normalize_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize_value(item) for item in value), key=repr)
    if np is not None and isinstance(value, np.ndarray):
        return [_normalize_value(item) for item in value.tolist()]
    if pd is not None and isinstance(value, pd.Series):
        return [_normalize_value(item) for item in value.tolist()]
    if hasattr(value, "to_dict") and not isinstance(value, (Path,)):
        try:
            return _normalize_value(value.to_dict())
        except Exception:
            pass
    return repr(value)


def _compact_json(value: Any, *, max_length: int = 220) -> str:
    payload = json.dumps(_normalize_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(payload) <= max_length:
        return payload
    return payload[: max_length - 3] + "..."


@dataclass(slots=True)
class InputReference:
    """Reference to one input used during processing."""

    path: str
    role: str = "input"
    version: str | None = None
    commit_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "version": self.version,
            "commit_hash": self.commit_hash,
            "metadata": _normalize_value(self.metadata),
        }


@dataclass(slots=True)
class ProvenanceStep:
    """Machine-readable provenance for one processing step."""

    step: str
    timestamp_utc: str
    function: str
    cryoswath_version: str
    cryoswath_commit: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    inputs: list[InputReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_utc": self.timestamp_utc,
            "function": self.function,
            "cryoswath_version": self.cryoswath_version,
            "cryoswath_commit": self.cryoswath_commit,
            "arguments": _normalize_value(self.arguments),
            "inputs": [item.to_dict() for item in self.inputs],
            "metadata": _normalize_value(self.metadata),
        }


@dataclass(slots=True)
class ProvenanceRecord:
    """Collection of provenance steps stored alongside an output."""

    created_utc: str
    schema_version: int = PROVENANCE_SCHEMA_VERSION
    package: str = "cryoswath"
    steps: list[ProvenanceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package": self.package,
            "created_utc": self.created_utc,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": _normalize_value(self.metadata),
        }


def coerce_input_reference(value: InputReference | str | Path | dict[str, Any]) -> InputReference:
    """Normalize a dynamic or static input reference."""

    if isinstance(value, InputReference):
        return value
    if isinstance(value, (str, Path)):
        return InputReference(path=str(value))
    if isinstance(value, dict):
        payload = dict(value)
        if "path" not in payload:
            raise KeyError("Input reference mappings must define a 'path'")
        metadata = payload.pop("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"value": _normalize_value(metadata)}
        return InputReference(
            path=str(payload.pop("path")),
            role=str(payload.pop("role", "input")),
            version=payload.pop("version", None),
            commit_hash=payload.pop("commit_hash", None),
            metadata={
                **metadata,
                **{str(key): _normalize_value(item) for key, item in payload.items()},
            },
        )
    raise TypeError(f"Unsupported input reference type: {type(value)!r}")


def capture_call_arguments(
    func: Callable[..., Any],
    /,
    *args: Any,
    include_defaults: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bind positional and keyword arguments to a callable's signature."""

    bound = signature(func).bind_partial(*args, **kwargs)
    if include_defaults:
        bound.apply_defaults()
    return _normalize_value(bound.arguments)


def build_provenance_step(
    step: str,
    func: Callable[..., Any] | str,
    /,
    *args: Any,
    inputs: Iterable[InputReference | str | Path | dict[str, Any]] | None = None,
    include_defaults: bool = False,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProvenanceStep:
    """Build a provenance step with normalized arguments and inputs."""

    revision = package_revision()
    if callable(func):
        arguments = capture_call_arguments(
            func,
            *args,
            include_defaults=include_defaults,
            **kwargs,
        )
    else:
        arguments = {
            "args": [_normalize_value(item) for item in args],
            "kwargs": _normalize_value(kwargs),
        }
    return ProvenanceStep(
        step=step,
        timestamp_utc=_utc_now(),
        function=_callable_name(func),
        cryoswath_version=revision["version"] or "unknown",
        cryoswath_commit=revision["commit_hash"],
        arguments=arguments,
        inputs=[
            coerce_input_reference(item)
            for item in (inputs or [])
        ],
        metadata=_normalize_value(metadata or {}),
    )


def build_provenance_record(
    steps: Iterable[ProvenanceStep],
    *,
    metadata: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> ProvenanceRecord:
    """Build a provenance record from one or more provenance steps."""

    return ProvenanceRecord(
        created_utc=created_utc or _utc_now(),
        steps=list(steps),
        metadata=dict(metadata or {}),
    )


def format_history_line(
    step: ProvenanceStep,
    *,
    max_argument_length: int = 220,
) -> str:
    """Format one CF-style history line."""

    revision = f"cryoswath {step.cryoswath_version}"
    if step.cryoswath_commit:
        revision += f" (git {step.cryoswath_commit[:8]})"
    return (
        f"{step.timestamp_utc} {revision}: {step.step} "
        f"{_compact_json(step.arguments, max_length=max_argument_length)}"
    )


def append_history(history: str | None, step: ProvenanceStep) -> str:
    """Append one history line to an existing CF history attribute."""

    new_line = format_history_line(step)
    if not history:
        return new_line
    return f"{history.rstrip()}\n{new_line}"


def provenance_path(store_path: str | Path, filename: str = PROVENANCE_FILENAME) -> Path:
    """Return the companion metadata path for a Zarr store."""

    return Path(store_path) / PROVENANCE_DIRNAME / filename


def serialize_provenance(
    steps: Iterable[ProvenanceStep] | ProvenanceRecord,
    *,
    metadata: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Serialize provenance data to a JSON-compatible mapping."""

    if isinstance(steps, ProvenanceRecord):
        record = steps
        if metadata is not None or created_utc is not None:
            record = ProvenanceRecord(
                created_utc=created_utc or steps.created_utc,
                steps=list(steps.steps),
                metadata=dict(metadata or steps.metadata),
                package=steps.package,
                schema_version=steps.schema_version,
            )
    else:
        record = build_provenance_record(steps, metadata=metadata, created_utc=created_utc)
    return record.to_dict()


def write_provenance_sidecar(
    store_path: str | Path,
    steps: Iterable[ProvenanceStep],
    *,
    metadata: Mapping[str, Any] | None = None,
    filename: str = PROVENANCE_FILENAME,
) -> Path:
    """Write provenance metadata as a Zarr-sidecar JSON file."""

    path = provenance_path(store_path, filename=filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_provenance(steps, metadata=metadata)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_provenance_sidecar(
    store_path: str | Path,
    *,
    filename: str = PROVENANCE_FILENAME,
) -> dict[str, Any]:
    """Load provenance metadata from a Zarr-sidecar JSON file."""

    path = provenance_path(store_path, filename=filename)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported provenance schema in {path}")
    return payload
