import geopandas as gpd
import pandas as pd
import pytest
import shapely

import cryoswath.misc as misc


class DummyResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

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

    catalog = misc._stac_items_to_l1b_track_catalog(items, "maap")

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

    catalog = misc._stac_items_to_l1b_track_catalog(items, "maap")

    assert catalog.iloc[0]["stage"] == "LTA_"


def test_stac_catalog_warns_and_excludes_unsupported_baselines():
    item = _item(
        "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_G001",
        version="G001",
    )

    with pytest.warns(UserWarning, match="unsupported baseline"):
        catalog = misc._stac_items_to_l1b_track_catalog([item], "maap")

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
        "maap",
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
        "maap",
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


class DummyFtp:
    def __init__(self, listings, payloads=None):
        self.listings = listings
        self.payloads = payloads or {}
        self.directory = None
        self.retrieved = []

    def cwd(self, directory):
        if directory not in self.listings:
            raise misc.ftplib.error_perm("missing")
        self.directory = directory

    def nlst(self):
        return self.listings[self.directory]

    def retrbinary(self, command, callback):
        name = command.removeprefix("RETR ")
        self.retrieved.append((self.directory, name))
        callback(self.payloads[(self.directory, name)])


def _hdr_payload(start_lat):
    return f"""<Earth_Explorer_File><Variable_Header><SPH><Product_Location>
<Start_Long>0</Start_Long><Start_Lat>{start_lat}</Start_Lat>
<Stop_Long>1000000</Stop_Long><Stop_Lat>{start_lat}</Stop_Lat>
</Product_Location></SPH></Variable_Header></Earth_Explorer_File>""".encode()


def test_ftp_month_listings_prefer_current_root_when_available():
    current = "/SIR_SIN_L1/2020/01"
    legacy = "/Ice_Baseline_E/SIR_SIN_L1/2020/01"
    ftp = DummyFtp({current: ["current.nc"], legacy: ["legacy.nc"]})

    listings = list(misc._ftp_l1b_month_listings(ftp, "2020/01"))

    assert listings == [(current, ["current.nc"]), (legacy, ["legacy.nc"])]


def test_ftp_ground_track_discovery_falls_back_and_keeps_current_duplicate(
    monkeypatch, tmp_path
):
    current = "/SIR_SIN_L1/2020/01"
    legacy = "/Ice_Baseline_E/SIR_SIN_L1/2020/01"
    name = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000100_E001.HDR"
    ftp = DummyFtp(
        {current: [name], legacy: [name]},
        {
            (current, name): _hdr_payload(70000000),
            (legacy, name): _hdr_payload(71000000),
        },
    )

    class DummyFtpServer:
        def __enter__(self):
            return ftp

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(misc, "ftp_cs2_server", lambda: DummyFtpServer())
    monkeypatch.setattr(misc, "aux_path", tmp_path)

    tracks = misc._ftp_cs_ground_tracks(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), gpd.GeoDataFrame()
    )

    assert tracks.iloc[0].geometry.coords[0][1] == 70
    assert ftp.retrieved == [(current, name)]


def test_ftp_filename_discovery_prefers_highest_version_in_current_root(
    monkeypatch, tmp_path
):
    root = "/SIR_SIN_L1/2020/01"
    low = "CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000100_E001.nc"
    high = "CS_LTA__SIR_SIN_1B_20200101T000000_20200101T000100_E002.nc"
    ftp = DummyFtp({root: [low, high]}, {})

    class DummyFtpServer:
        def __enter__(self):
            return ftp

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(misc, "ftp_cs2_server", lambda: DummyFtpServer())
    monkeypatch.setattr(misc, "aux_path", tmp_path)

    misc._ftp_cs_ground_tracks(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), gpd.GeoDataFrame()
    )

    cached = pd.read_pickle(tmp_path / "CryoSat-2_SARIn_file_names.pkl")
    assert cached.iloc[0].endswith("E002")


@pytest.mark.parametrize("stac_error", [None, RuntimeError("MAAP unavailable")])
def test_load_cs_ground_tracks_auto_uses_bounded_ftp_fallback(
    monkeypatch, tmp_path, stac_error
):
    _catalog_path(monkeypatch, tmp_path)
    legacy_path = tmp_path / "tracks.feather"
    monkeypatch.setattr(misc, "cs_ground_tracks_path", str(legacy_path))
    calls = []

    def fake_refresh(start_datetime, end_datetime, *, replace=False):
        if stac_error:
            raise stac_error
        return misc._empty_cs_l1b_track_catalog()

    def fake_ftp(start_datetime, end_datetime, present_tracks):
        calls.append((start_datetime, end_datetime, present_tracks.copy()))
        return gpd.GeoDataFrame(
            geometry=[shapely.LineString([(0, 70), (1, 71)])],
            index=pd.DatetimeIndex(["2020-01-01"], name="index"),
            crs=4326,
        )

    monkeypatch.setattr(misc, "_refresh_cs_l1b_track_catalog", fake_refresh)
    monkeypatch.setattr(misc, "_ftp_cs_ground_tracks", fake_ftp)

    tracks = misc.load_cs_ground_tracks(
        start_datetime="2020-01-01", end_datetime="2020-01-02", source="auto"
    )

    assert calls[0][:2] == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    assert legacy_path.is_file()
    assert len(tracks) == 1


def test_load_cs_ground_tracks_auto_does_not_use_ftp_after_stac_tracks(
    monkeypatch, tmp_path
):
    _catalog_path(monkeypatch, tmp_path)
    monkeypatch.setattr(misc, "cs_ground_tracks_path", str(tmp_path / "tracks.feather"))
    refreshed = misc._stac_items_to_l1b_track_catalog(
        [_item("CS_OFFL_SIR_SIN_1B_20200101T000000_20200101T000200_E001")], "maap"
    )
    def fake_refresh(*args, **kwargs):
        misc._save_cs_l1b_track_catalog(refreshed)
        return refreshed

    monkeypatch.setattr(misc, "_refresh_cs_l1b_track_catalog", fake_refresh)
    monkeypatch.setattr(
        misc,
        "_ftp_cs_ground_tracks",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("FTP should not be queried")
        ),
    )

    tracks = misc.load_cs_ground_tracks(
        start_datetime="2020-01-01", end_datetime="2020-01-02", source="auto"
    )

    assert len(tracks) == 1
