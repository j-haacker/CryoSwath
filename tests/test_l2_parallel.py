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
