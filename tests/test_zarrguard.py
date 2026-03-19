from __future__ import annotations

import io
import json
from pathlib import Path

import cryoswath.zarrguard as zarrguard


class _FakeDataset:
    def __init__(self):
        self.calls = []

    def to_zarr(self, store_path, *, compute=True, mode="w", write_empty_chunks=False):
        kwargs = {
            "compute": compute,
            "mode": mode,
            "write_empty_chunks": write_empty_chunks,
        }
        self.calls.append((Path(store_path), kwargs))
        return "write-result"


def test_guarded_to_zarr_defaults_write_empty_chunks(monkeypatch, tmp_path: Path):
    dataset = _FakeDataset()
    guard_calls = []

    monkeypatch.setattr(
        zarrguard,
        "guard_existing_store",
        lambda *args, **kwargs: guard_calls.append((args, kwargs)),
    )

    result = zarrguard.guarded_to_zarr(
        dataset,
        tmp_path / "store.zarr",
        to_zarr_kwargs={"mode": "w"},
    )

    assert result == "write-result"
    assert dataset.calls == [
        (tmp_path / "store.zarr", {"mode": "w", "compute": True, "write_empty_chunks": True})
    ]
    assert guard_calls == [((tmp_path / "store.zarr",), {})]


def test_guard_existing_store_prefers_import_backend(monkeypatch, tmp_path: Path):
    calls = []

    class FakeXzarrguard:
        def create_store(self, dataset, store, **kwargs):
            calls.append((dataset, Path(store), kwargs))

    monkeypatch.setattr(zarrguard, "_load_xzarrguard_module", lambda: FakeXzarrguard())

    zarrguard.guard_existing_store(
        tmp_path / "store.zarr",
        no_data_chunks={"a": [(0, 1), (2, 3)]},
    )

    assert calls == [
        (
            None,
            tmp_path / "store.zarr",
            {
                "in_place_metadata_only": True,
                "no_data_chunks": {"a": [(0, 1), (2, 3)]},
                "infer_no_data_from_store": False,
            },
        )
    ]


def test_guard_existing_store_cli_fallback_serializes_no_data(monkeypatch, tmp_path: Path):
    captured = {}
    fake_file = io.StringIO()

    class FakeTempFile:
        name = str(tmp_path / "xzarrguard-no-data.json")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, text):
            return fake_file.write(text)

    monkeypatch.setattr(zarrguard, "_load_xzarrguard_module", lambda: None)
    monkeypatch.setattr(zarrguard.tempfile, "NamedTemporaryFile", lambda **_: FakeTempFile())
    monkeypatch.setattr(zarrguard.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        zarrguard.subprocess,
        "run",
        lambda cmd, check: captured.update({"cmd": cmd, "check": check}),
    )

    zarrguard.guard_existing_store(
        tmp_path / "store.zarr",
        no_data_chunks={"a": [(4, 5)]},
        command=["xzarrguard"],
    )

    assert captured == {
        "cmd": [
            "xzarrguard",
            "create",
            str(tmp_path / "store.zarr"),
            "--in-place-metadata-only",
            "--no-data",
            str(tmp_path / "xzarrguard-no-data.json"),
        ],
        "check": True,
    }
    assert json.loads(fake_file.getvalue()) == {"a": [[4, 5]]}
