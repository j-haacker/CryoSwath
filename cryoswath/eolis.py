"""Compare CryoSwath swath tracks with ESA EOLIS point products."""

__all__ = ["compare_track"]

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import CRS, Transformer
from scipy.stats import median_abs_deviation

from cryoswath import l1b, l2, misc

# RGI geographic regions can include several overlapping EOLIS products.
_REGIONS = {
    "01": ("ALASKA",),
    "02": ("WESTCANUS",),
    "03": ("ARCCANNOR",),
    "04": ("ARCCANSOU",),
    "05": ("GREENLAND", "GREENPERI"),
    "06": ("ICELAND",),
    "07": ("SVALBARD",),
    "08": ("SCANDINAV",),
    "09": ("RUSSIANAR",),
    "11": ("CENTRALEU",),
    "13": ("CENTRASIA",),
    "14": ("SOUASIAWE",),
    "15": ("SOUASIAEA",),
    "16": ("LOWLAT",),
    "17": ("SOUTHANDE",),
    "18": ("NEWZEALND",),
    "19": ("ANTARCTIC", "ANTARCPER", "ANTICESHF"),
    "20": ("ANTARCTIC", "ANTARCPER", "ANTICESHF"),
}
_TO_ECEF = Transformer.from_crs(4979, 4978, always_xy=True)
_METRICS = [
    "height_difference_m",
    "along_difference_m",
    "across_difference_m",
    "horizontal_distance_m",
    "radial_difference_m",
    "geometry_range_bias_m",
]


def _ecef(points):
    geographic = points.to_crs(4326)
    return np.column_stack(
        _TO_ECEF.transform(
            geographic.geometry.x.to_numpy(),
            geographic.geometry.y.to_numpy(),
            points.elevation.to_numpy(),
        )
    )


def _download(ftp, remote, cache):
    target = Path(cache) / remote.lstrip("/")
    if not target.is_file():
        l1b._download_remote_file_via_ftp_atomic(ftp, remote, target)
    return target


def _tile_names(tiles, ground, along_track):
    """Coarse chunk selection; across extent comes from the entire tile index."""
    centre = ground.geometry.iloc[len(ground) // 2]
    planar = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre.y} +lon_0={centre.x} +datum=WGS84 +units=m"
    )
    projected = tiles.to_crs(planar)
    positions = ground.to_crs(planar).get_coordinates().to_numpy()
    if len(positions) < 2:
        # One waveform has no positional tangent; retain all regional chunks.
        return [Path(value).name for value in tiles.pdgs_path]
    tangents = np.gradient(positions, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    left, bottom, right, top = projected.total_bounds
    corners = np.array([[left, bottom], [left, top], [right, bottom], [right, top]])
    # Tile-scale along padding absorbs projection/edge curvature in this coarse
    # query. The exact 170 m velocity-plane gate is applied only in _associate.
    bounds = projected.bounds
    padding = np.hypot(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny).max()
    strips = []
    for point, tangent in zip(positions, tangents):
        normal = np.array([-tangent[1], tangent[0]])
        extent = np.linalg.norm(corners - point, axis=1).max()
        strips.append(
            shapely.Polygon(
                [
                    point + a * tangent * (padding + along_track) + b * normal * extent
                    for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                ]
            )
        )
    selected = projected.intersects(shapely.union_all(strips))
    return [Path(value).name for value in tiles.loc[selected, "pdgs_path"]]


def _retrieve(filename, regions, ground, cache, version, along_track):
    name = Path(filename).name
    months = pd.period_range(
        misc.cs_id_to_time(name[19:34]), misc.cs_id_to_time(name[35:50]), freq="M"
    )
    paths = []
    with misc.ftp_cs2_server() as ftp:
        for month in months:
            for region in regions:
                folder = f"/TEMPO_SWATH_POINT/{month.year:04}/{month.month:02}/{region}"
                names = [Path(n).name for n in ftp.nlst(folder)]
                indices = [n for n in names if n.endswith(".tgz")]
                if len(indices) != 1:
                    raise ValueError(f"Expected one EOLIS tile index in {folder}")
                archive = _download(ftp, f"{folder}/{indices[0]}", cache)
                tiles = gpd.read_file(f"/vsitar/{archive.resolve()}")
                identities = {
                    n.rsplit("_", 1)[0] for n in _tile_names(tiles, ground, along_track)
                }
                candidates = [
                    n
                    for n in names
                    if n.endswith(".nc")
                    and n.rsplit("_", 1)[0] in identities
                    and (version is None or n.endswith(f"_{version}.nc"))
                ]
                if {n.rsplit("_", 1)[0] for n in candidates} != identities:
                    raise FileNotFoundError(
                        f"Selected EOLIS tiles/version absent in {folder}"
                    )
                if len(candidates) != len(identities):
                    raise ValueError(
                        f"Multiple EOLIS versions in {folder}; specify version."
                    )
                paths.extend(
                    _download(ftp, f"{folder}/{n}", cache) for n in sorted(candidates)
                )
    return paths


def _read_points(paths, track):
    frames = []
    for path in paths:
        with xr.open_dataset(path, decode_coords="all") as ds:
            fileids = ds.attrs.get("fileids", "")
            text = fileids if isinstance(fileids, str) else "\n".join(fileids)
            sources = {
                int(k): v for k, v in re.findall(r"(-?\d+)\s*:\s*([^,;\s]+)", text)
            }
            ids = [key for key, value in sources.items() if track in value]
            keep = np.isin(ds.input_file_id.values, ids) & (ds.is_swath.values == 1)
            frame = ds.isel(row=np.flatnonzero(keep)).to_dataframe().reset_index()
            if frame.empty:
                continue
            if np.issubdtype(ds.time.dtype, np.number):
                if (
                    ds.time.attrs.get("units")
                    != "Seconds from 1970 in the UTC timezone."
                ):
                    raise ValueError("Unrecognised numeric EOLIS time units.")
                frame["time"] = pd.to_datetime(frame.time, unit="s", utc=True)
            frame["source_file"] = frame.input_file_id.map(sources)
            frame["eolis_file"] = str(path)
            if {"lon", "lat"} <= set(frame):
                geometry = gpd.points_from_xy(frame.lon, frame.lat)
                crs = 4326
            else:
                geometry = gpd.points_from_xy(frame.x, frame.y)
                crs = ds.attrs["geospatial_projection"]
            frames.append(
                gpd.GeoDataFrame(frame, geometry=geometry, crs=crs).to_crs(4326)
            )
    if not frames:
        return gpd.GeoDataFrame(
            columns=["row", "time", "source_file", "eolis_file", "elevation"],
            geometry=[],
            crs=4326,
        ), 0
    points = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    unique = points.drop_duplicates(["source_file", "time", "geometry", "elevation"])
    return unique.reset_index(drop=True), len(points) - len(unique)


def _associate(raw, swath, reference, along_track=170):
    """Native-bin association and diagnostic differences, all in metres."""
    own = swath.loc[np.isfinite(swath.height)].copy().reset_index()
    if own.empty or not swath.index.is_unique:
        raise ValueError(
            "Comparison requires nonempty swath data with unique time/sample IDs."
        )
    times = pd.DatetimeIndex(raw.time_20_ku.values)
    own_times = pd.DatetimeIndex(own.time).tz_localize(None)
    indices = times.get_indexer(own_times)
    if (indices < 0).any():
        raise ValueError(
            "Cached swath waveform times do not match the raw L1b product."
        )
    sat = np.column_stack(
        _TO_ECEF.transform(raw.lon_20_ku, raw.lat_20_ku, raw.alt_20_ku)
    )
    lon, lat = np.deg2rad(raw.lon_20_ku.values), np.deg2rad(raw.lat_20_ku.values)
    up = np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    velocity = raw.sat_vel_vec_20_ku.values
    along = velocity - np.einsum("ij,ij->i", velocity, up)[:, None] * up
    along /= np.linalg.norm(along, axis=1)[:, None]
    basis = np.stack([along, np.cross(along, up), -up], axis=1)
    native = l1b.ref_range(raw).values
    own_xyz = _ecef(own.rename(columns={"height": "elevation"}))
    ref_xyz = _ecef(reference)
    own_ranges = np.linalg.norm(own_xyz - sat[indices], axis=1)
    parts = []
    for k in np.unique(indices):
        displacement = ref_xyz - sat[k]
        along_offset = displacement @ along[k]
        ri = np.flatnonzero(np.abs(along_offset) <= along_track)
        if not len(ri):
            continue
        ranges = np.linalg.norm(displacement[ri], axis=1)
        fractional_sample = (
            misc.reference_bin + (ranges - native[k]) / misc.sample_width
        )
        samples = np.rint(fractional_sample).astype(int)
        native_ranges = native[k] + (samples - misc.reference_bin) * misc.sample_width
        group = own.loc[indices == k]
        lookup = pd.Index(group["sample"]).get_indexer(samples)
        present = lookup >= 0
        oi = group.index.to_numpy()[lookup[present]]
        part = reference.iloc[ri].copy()
        part["reference_position"] = ri
        part["waveform_time"] = times[k]  # Native TAI clock, shared with cached L2.
        part["sample"] = samples
        part["radial_difference_m"] = native_ranges - ranges
        part["matched"] = present & (
            np.abs(native_ranges - ranges) <= misc.sample_width / 2
        )
        part["status"] = np.where(
            part.matched,
            "matched",
            np.where(present, "outside radial tolerance", "missing cached sample"),
        )
        for metric in _METRICS[0:4] + ["geometry_range_bias_m"]:
            part[metric] = np.nan
        delta = (own_xyz[oi] - ref_xyz[ri[present]]) @ basis[k].T
        part.loc[present, "height_difference_m"] = (
            own.height.to_numpy()[oi] - reference.elevation.to_numpy()[ri[present]]
        )
        part.loc[present, "along_difference_m"] = delta[:, 0]
        part.loc[present, "across_difference_m"] = delta[:, 1]
        part.loc[present, "horizontal_distance_m"] = np.hypot(delta[:, 0], delta[:, 1])
        part.loc[present, "geometry_range_bias_m"] = (
            own_ranges[oi] - native_ranges[present]
        )
        parts.append(part)
    points = (
        gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
        if parts
        else reference.iloc[:0].copy()
    )
    if not parts:
        for name in [
            "reference_position",
            "waveform_time",
            "sample",
            "status",
            "matched",
            *_METRICS,
        ]:
            points[name] = pd.Series(dtype=bool if name == "matched" else float)
    matched = points.loc[points.matched]
    duplicates = matched.duplicated("reference_position", keep=False)
    if duplicates.any():
        details = matched.loc[
            duplicates, ["eolis_file", "row", "waveform_time", "sample"]
        ]
        raise ValueError(
            f"Multiple waveform matches for {raw.attrs.get('product_name', 'track')}. "
            "Please file a GitHub issue at "
            "https://github.com/j-haacker/cryoswath/issues "
            f"with these track/waveform/sample details:\n"
            f"{details.to_string(index=False)}"
        )
    values = matched[_METRICS]
    statistics = values.agg(["count", "mean", "median", "std"]).T
    statistics["rmse"] = np.sqrt(values.pow(2).mean())
    statistics["nmad"] = values.apply(
        lambda col: (
            median_abs_deviation(col.dropna(), scale="normal")
            if col.notna().any()
            else np.nan
        )
    )
    statistics[["p2.5", "p97.5"]] = values.quantile([0.025, 0.975]).T
    coverage = pd.Series(
        {
            "cryoswath_points": len(own),
            "eolis_observations": len(reference),
            "selected_observations": points.reference_position.nunique(),
            "matched_observations": len(matched),
            "missing_cached_samples": int(
                (points.status == "missing cached sample").sum()
            ),
            "outside_radial_tolerance": int(
                (points.status == "outside radial tolerance").sum()
            ),
        }
    )
    points.attrs.update(
        waveform_time_scale="TAI", difference_sign="CryoSwath minus EOLIS"
    )
    return {"points": points, "coverage": coverage, "statistics": statistics}


def compare_track(
    track_id,
    *,
    reprocess=False,
    along_track=170,
    cache_dir=None,
    version=None,
    **processing_kwargs,
):
    """Retrieve and compare an entire CryoSwath swath track with EOLIS.

    Existing project paths, ESA credentials and cached L2 are reused. Missing L2
    is processed through l2.from_id; processing_kwargs apply when processing is
    needed. Set reprocess=True explicitly to replace processing results.

    Returns a dictionary of points (GeoDataFrame, including unmatched cases),
    coverage (Series) and statistics (DataFrame). Point differences are metres.

    Timestamp fields in points use different clocks: waveform_time retains native
    L1b TAI, while EOLIS time is UTC. Cached L2 waveform indices also retain TAI
    despite their UTC timezone labels. Convert clocks before comparing timestamps;
    a timezone change alone does not perform that conversion.

    Association uses the ±along_track velocity-plane window and nearest native
    range bin with half-bin tolerance.
    Height, across-track and approximate EOLIS timestamps do not gate matches.
    Statistics describe matched observations, not independent accuracy estimates.
    """
    if not np.isfinite(along_track) or along_track <= 0:
        raise ValueError("along_track must be a positive finite distance in metres.")
    filename = l1b._resolve_l1b_file(track_id)
    track = Path(filename).name[19:34]
    processing_kwargs.setdefault("cores", 1)
    swath, _ = l2.from_id(
        misc.cs_id_to_time(track),
        reprocess=reprocess,
        save_or_return="both",
        **processing_kwargs,
    )
    if swath.empty:
        raise ValueError(f"No CryoSwath swath points available for track {track}.")
    with xr.open_dataset(filename, decode_timedelta=False) as raw:
        time_index = pd.DatetimeIndex(raw.time_20_ku.values)
        selected = time_index.isin(
            pd.DatetimeIndex(swath.index.get_level_values("time")).tz_localize(None)
        )
        ground = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                raw.lon_20_ku.values[selected], raw.lat_20_ku.values[selected]
            ),
            crs=4326,
        )
        if ground.empty:
            raise ValueError("Cached swath waveform times do not match raw L1b.")
        codes = misc.find_region_id(
            shapely.MultiPoint(ground.geometry.to_list()), scope="o1", all_matches=True
        )
        regions = sorted(
            {region for code in codes for region in _REGIONS.get(code, ())}
        )
        if not regions:
            raise ValueError(f"No documented EOLIS coverage for RGI regions {codes}.")
        cache = (
            Path(cache_dir) if cache_dir is not None else Path(misc.aux_path) / "EOLIS"
        )
        paths = _retrieve(filename, regions, ground, cache, version, along_track)
        reference, removed = _read_points(paths, track)
        result = _associate(raw, swath, reference, along_track)
    result["coverage"]["exact_duplicates_removed"] = removed
    result["points"].attrs.update(
        track_id=track,
        l1b_file=str(filename),
        eolis_files=[str(path) for path in paths],
        regions=regions,
        unsupported_rgi_regions=[code for code in codes if code not in _REGIONS],
    )
    return result
