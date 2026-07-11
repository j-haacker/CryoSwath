import geopandas as gpd
import pandas as pd
import pytest
import shapely

import cryoswath.misc as misc


class DummyResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def _item(
    item_id,
    *,
    start="2020-01-01T00:00:00Z",
    end="2020-01-01T00:02:00Z",
    version="E001",
    href=None,
):
    href = href or (
        "https://science-pds.cryosat.esa.int/?do=download&file="
        f"Cry0Sat2_data%2FSIR_SIN_L1%2F2020%2F01%2F{item_id}.nc"
    )
    return {
        "id": item_id,
        "properties": {
            "product:type": "SIR_SIN_1B",
            "sar:instrument_mode": "SARIN",
            "version": version,
            "start_datetime": start,
            "end_datetime": end,
            "processing:datetime": "2020-02-01T00:00:00Z",
            "published": "2020-02-01T01:00:00Z",
        },
        "assets": {"enclosure": {"href": href}},
        "geometry": {"type": "LineString", "coordinates": [[0, 70], [1, 71]]},
    }


def _catalog_path(monkeypatch, tmp_path):
    path = tmp_path / "CryoSat-2_SARIn_L1B_track_catalog.feather"
    monkeypatch.setattr(misc, "cs_l1b_track_catalog_path", str(path))
    return path


def test_stac_catalog_selects_highest_supported_baseline_before_lta():
    start = "2016-04-04T16:21:31Z"
    items = [
        _item(
            "CS_LTA__SIR_SIN_1B_20160404T162131_20160404T162445_D001",
            start=start,
            version="D001",
        ),
        _item(
            "CS_OFFL_SIR_SIN_1B_20160404T162131_20160404T162445_E001",
            start=start,
            version="E001",
        ),
    ]

    catalog = misc._stac_items_to_l1b_track_catalog(items, "eocat")

    assert len(catalog) == 1
    assert catalog.iloc[0]["filename"] == (
        "CS_OFFL_SIR_SIN_1B_20160404T162131_20160404T162445_E001.nc"
    )


def test_stac_catalog_prefers_lta_for_same_baseline_and_version():
    start = "2020-01-01T00:00:00Z"
    items = [
        _item(
            "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001",
            start=start,
        ),
        _item(
            "CS_LTA__SIR_SIN_1B_20200101T000000_20200101T000200_E001",
            start=start,
        ),
    ]

    catalog = misc._stac_items_to_l1b_track_catalog(items, "eocat")

    assert catalog.iloc[0]["stage"] == "LTA_"


def test_stac_catalog_warns_and_excludes_unsupported_baselines():
    item = _item(
        "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_F001",
        version="F001",
    )

    with pytest.warns(UserWarning, match="unsupported baseline"):
        catalog = misc._stac_items_to_l1b_track_catalog([item], "eocat")

    assert catalog.empty


def test_stac_query_uses_maap_pagination(monkeypatch):
    calls = []
    first_item = _item("CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001")
    second_item = _item(
        "CS_OFFL_SIR_SIN_1B_20200102T000000_20200102T000200_E001",
        start="2020-01-02T00:00:00Z",
    )

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if params is not None:
            assert "maap" in url
            assert params["collections"] == "CryoSatIceL110"
            assert params["productType"] == "SIR_SIN_1B"
            assert params["sensorMode"] == "SARIN"
            return DummyResponse(
                {
                    "features": [first_item],
                    "links": [{"rel": "next", "href": "https://next.example"}],
                }
            )
        return DummyResponse({"features": [second_item], "links": []})

    monkeypatch.setattr(misc.requests, "get", fake_get)

    catalog = misc._query_stac_l1b_track_catalog(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")
    )

    assert len(catalog) == 2
    assert list(catalog["provider"].unique()) == ["maap"]
    assert len(calls) == 2


def test_stac_query_prefers_maap_and_uses_its_collection(monkeypatch):
    calls = []
    item = _item(
        "CS_OFFL_SIR_SIN_1B_20220917T082319_20220917T082404_E001",
        start="2022-09-17T08:23:19Z",
        end="2022-09-17T08:24:05Z",
        href=(
            "https://catalog.maap.eo.esa.int/data/cryosat-pdgs-01/CRYOSAT/"
            "SIR_SIN_1B/E001/2022/09/17/example.nc"
        ),
    )
    item["assets"] = {"enclosure_nc": item["assets"].pop("enclosure")}

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if "maap" not in url:
            return DummyResponse({"features": []})
        assert params["collections"] == "CryoSatIceL110"
        return DummyResponse({"features": [item], "links": []})

    monkeypatch.setattr(misc.requests, "get", fake_get)

    catalog = misc._query_stac_l1b_track_catalog(
        pd.Timestamp("2022-09-17T08:23:19"),
        pd.Timestamp("2022-09-17T08:24:05"),
    )

    assert list(catalog["provider"].unique()) == ["maap"]
    assert catalog.iloc[0]["href"].endswith("example.nc")
    assert len(calls) == 1


def test_load_cs_full_file_names_overlays_stac_catalog(monkeypatch, tmp_path):
    _catalog_path(monkeypatch, tmp_path)
    monkeypatch.setattr(misc, "aux_path", tmp_path)
    legacy_path = tmp_path / "CryoSat-2_SARIn_file_names.pkl"
    track_time = pd.Timestamp("2020-01-01T00:00:00")
    pd.Series(
        {track_time: "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000100_D001"}
    ).to_pickle(legacy_path)
    catalog = misc._stac_items_to_l1b_track_catalog(
        [
            _item(
                "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001",
                start="2020-01-01T00:00:00Z",
            )
        ],
        "eocat",
    )
    misc._save_cs_l1b_track_catalog(catalog)

    file_names = misc.load_cs_full_file_names(update="no")

    assert file_names.loc[track_time].endswith("_E001")


def test_load_cs_ground_tracks_auto_uses_local_when_covered(monkeypatch, tmp_path):
    _catalog_path(monkeypatch, tmp_path)
    legacy_path = tmp_path / "tracks.feather"
    monkeypatch.setattr(misc, "cs_ground_tracks_path", str(legacy_path))
    legacy = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(0, 70), (1, 71)])],
        index=pd.DatetimeIndex(["2020-01-02"], name="index"),
        crs=4326,
    )
    legacy.to_feather(legacy_path)
    monkeypatch.setattr(
        misc,
        "_refresh_cs_l1b_track_catalog",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("STAC should not be queried")
        ),
    )

    tracks = misc.load_cs_ground_tracks(
        start_datetime="2020-01-01",
        end_datetime="2020-01-02",
        source="auto",
    )

    assert len(tracks) == 1


def test_load_cs_ground_tracks_auto_refreshes_missing_tail(monkeypatch, tmp_path):
    _catalog_path(monkeypatch, tmp_path)
    legacy_path = tmp_path / "tracks.feather"
    monkeypatch.setattr(misc, "cs_ground_tracks_path", str(legacy_path))
    legacy = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(0, 70), (1, 71)])],
        index=pd.DatetimeIndex(["2020-01-01"], name="index"),
        crs=4326,
    )
    legacy.to_feather(legacy_path)
    refreshed = misc._stac_items_to_l1b_track_catalog(
        [
            _item(
                "CS_OFFL_SIR_SIN_1B_20200102T000000_20200102T000200_E001",
                start="2020-01-02T00:00:00Z",
            )
        ],
        "eocat",
    )
    calls = []

    def fake_refresh(start_datetime, end_datetime, *, replace=False):
        calls.append((start_datetime, end_datetime, replace))
        misc._save_cs_l1b_track_catalog(refreshed)
        return refreshed

    monkeypatch.setattr(misc, "_refresh_cs_l1b_track_catalog", fake_refresh)

    tracks = misc.load_cs_ground_tracks(
        start_datetime="2020-01-01",
        end_datetime="2020-01-03",
        source="auto",
    )

    assert calls
    assert pd.Timestamp("2020-01-02") in tracks.index
