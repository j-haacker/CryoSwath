from pathlib import Path

import cryoswath.provenance as provenance


def sample_step(a, b=2, *args, c=3, **kwargs):
    return a, b, args, c, kwargs


def test_capture_call_arguments_normalizes_positional_keyword_and_defaults():
    captured = provenance.capture_call_arguments(
        sample_step,
        1,
        4,
        5,
        c=7,
        d=Path("/tmp/source.zarr"),
        include_defaults=True,
    )

    assert captured == {
        "a": 1,
        "b": 4,
        "args": [5],
        "c": 7,
        "kwargs": {"d": "/tmp/source.zarr"},
    }


def test_history_line_mentions_revision_and_arguments(monkeypatch):
    monkeypatch.setattr(provenance, "_package_version", lambda: "0.2.5")
    monkeypatch.setattr(provenance, "_resolve_git_commit", lambda: "abcdef1234567890")

    step = provenance.build_provenance_step(
        "l3.build_dataset",
        sample_step,
        1,
        2,
        c=9,
        inputs=[{"path": "l2/swath.zarr", "role": "dynamic", "metadata": {"query": "month=1"}}],
    )

    line = provenance.format_history_line(step)

    assert "cryoswath 0.2.5 (git abcdef12)" in line
    assert "l3.build_dataset" in line
    assert '"a":1' in line
    assert '"c":9' in line
    assert line.startswith(step.timestamp_utc)


def test_history_appends_new_lines():
    step = provenance.ProvenanceStep(
        step="l4.fill_voids",
        timestamp_utc="2026-03-19T12:00:00Z",
        function="cryoswath.l4.fill_voids",
        cryoswath_version="0.2.5",
        cryoswath_commit="abcdef12",
        arguments={"main_var": "_median"},
    )

    out = provenance.append_history("first line", step)

    assert out == (
        "first line\n"
        "2026-03-19T12:00:00Z cryoswath 0.2.5 (git abcdef12): l4.fill_voids "
        '{"main_var":"_median"}'
    )


def test_provenance_sidecar_round_trip(tmp_path: Path):
    store = tmp_path / "output.zarr"
    store.mkdir()
    step = provenance.ProvenanceStep(
        step="l3.build_dataset",
        timestamp_utc="2026-03-19T12:00:00Z",
        function="cryoswath.l3.build_dataset",
        cryoswath_version="0.2.5",
        cryoswath_commit=None,
        arguments={"region": "05-01"},
        inputs=[provenance.coerce_input_reference({"path": "l2/cache.zarr", "role": "dynamic"})],
        metadata={"note": "test"},
    )

    written = provenance.write_provenance_sidecar(store, [step], metadata={"output": "L3"})
    assert written == store / ".cryoswath" / "provenance.json"

    payload = provenance.load_provenance_sidecar(store)
    assert payload["schema_version"] == 1
    assert payload["package"] == "cryoswath"
    assert payload["metadata"] == {"output": "L3"}
    assert payload["steps"][0]["inputs"][0]["path"] == "l2/cache.zarr"
    assert payload["steps"][0]["metadata"] == {"note": "test"}
