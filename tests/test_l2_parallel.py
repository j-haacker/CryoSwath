from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

import cryoswath.l2 as l2


def test_get_parallel_pool_uses_spawn_context(monkeypatch):
    expected_pool = object()
    called = {}

    class DummyContext:
        Pool = expected_pool

    def fake_get_context(method):
        called["method"] = method
        return DummyContext

    monkeypatch.setattr(l2.mp, "get_context", fake_get_context)
    assert l2._get_parallel_pool() is expected_pool
    assert called["method"] == "spawn"


def test_process_track_creates_configured_l2_parents(monkeypatch, tmp_path):
    idx = pd.Timestamp("2020-01-01 00:00:00")
    swath_root = tmp_path / "custom-swath"
    poca_root = tmp_path / "custom-poca"
    file_names = pd.Series({idx: "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST"})

    class DummyL1b:
        def close(self):
            pass

    def fake_to_l2(*args, **kwargs):
        data = gpd.GeoDataFrame(
            {"h_diff": [1.0]}, geometry=[shapely.Point(0, 0)], crs=4326
        )
        return data, data.copy()

    monkeypatch.setattr(l2, "l2_swath_path", str(swath_root))
    monkeypatch.setattr(l2, "l2_poca_path", str(poca_root))
    monkeypatch.setattr(l2.l1b, "from_id", lambda *args, **kwargs: DummyL1b())
    monkeypatch.setattr(l2.l1b, "to_l2", fake_to_l2)

    l2.process_track(
        idx,
        reprocess=False,
        l2_paths=pd.DataFrame(columns=["swath", "poca"]),
        save_or_return="save",
        current_subdir=str(Path("2020") / "01"),
        kwargs={"cs_full_file_names": file_names},
    )

    assert (
        swath_root / "2020" / "01" / "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.feather"
    ).is_file()
    assert (
        poca_root / "2020" / "01" / "CS_OFFL_SIR_SIN_1B_20200101T000000_TEST.feather"
    ).is_file()


def test_from_id_reuses_one_pool_across_batches_and_months(monkeypatch, tmp_path):
    tracks = pd.date_range("2020-01-01", periods=5).append(
        pd.date_range("2020-02-01", periods=2)
    )
    events = []

    class DummyPool:
        def __init__(self, processes):
            events.append(("start", processes))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append(("stop",))

        def starmap(self, function, arguments, chunksize):
            events.append(("batch", [args[0] for args in arguments]))
            return [function(*args) for args in arguments]

    def process(idx, *args):
        data = gpd.GeoDataFrame(
            {"track": [idx]}, geometry=[shapely.Point(0, 0)], crs=4326
        )
        return data, data.copy()

    monkeypatch.setattr(l2, "_get_parallel_pool", lambda: DummyPool)
    monkeypatch.setattr(l2, "load_cs_full_file_names", lambda **kwargs: None)
    monkeypatch.setattr(l2, "process_track", process)
    monkeypatch.setattr(l2, "l2_swath_path", str(tmp_path / "swath"))
    monkeypatch.setattr(l2, "l2_poca_path", str(tmp_path / "poca"))

    swath, poca = l2.from_id(tracks, cores=2, save_or_return="return")

    assert events == [
        ("start", 2),
        ("batch", list(tracks[:2])),
        ("batch", list(tracks[2:4])),
        ("batch", list(tracks[5:])),
        ("stop",),
    ]
    assert list(swath.track) == list(poca.track) == list(tracks)
