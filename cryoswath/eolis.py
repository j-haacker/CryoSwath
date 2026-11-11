"""Compare CryoSwath swath tracks with ESA EOLIS point products."""

__all__ = [
    "sample_ranges",
    "associate_points",
    "evaluate_associations",
    "compare_track",
]

import hashlib
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
    "range_difference_m",
]
_DIAGNOSTICS = ["sample_assignment_residual_m", "range_reconstruction_error_m"]


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
    # query. The exact velocity-plane gate is applied only in _associate.
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


def sample_ranges(raw):
    """Native sample centres and half-open range intervals, in metres.

    Dimensions are ``time_20_ku`` and ``sample``. Native sample IDs are zero-based
    positions, regardless of labels stored on the raw ``ns_20_ku`` coordinate.
    Waveform timestamps retain the product's native TAI clock.
    """
    count = raw.sizes["ns_20_ku"]
    if count <= 0:
        raise ValueError("The raw waveform must contain native samples.")
    native = l1b.ref_range(raw)
    sample = xr.DataArray(np.arange(count), dims="sample", name="sample")
    # Derive shared edges first so neighbouring intervals have identical bounds.
    lower = native + (sample - misc.reference_bin - 0.5) * misc.sample_width
    upper = native + (sample - misc.reference_bin + 0.5) * misc.sample_width
    return xr.Dataset(
        {
            "centre_m": native + (sample - misc.reference_bin) * misc.sample_width,
            "lower_m": lower,
            "upper_m": upper,
        },
        attrs={
            "waveform_time_scale": "TAI",
            "range_interval": "[lower, upper)",
            "reference_bin": misc.reference_bin,
            "sample_width_m": misc.sample_width,
        },
    )


def _raw_identity(raw):
    """Bind portable associations to the raw geometry and, when present, file."""
    digest = hashlib.sha256()
    for name in (
        "time_20_ku",
        "lon_20_ku",
        "lat_20_ku",
        "alt_20_ku",
        "sat_vel_vec_20_ku",
    ):
        digest.update(np.ascontiguousarray(raw[name].values).tobytes())
    digest.update(np.ascontiguousarray(l1b.ref_range(raw).values).tobytes())
    digest.update(
        str((raw.sizes["ns_20_ku"], misc.reference_bin, misc.sample_width)).encode()
    )
    identity = {
        "raw_product": str(raw.attrs.get("product_name", "")),
        "raw_geometry_sha256": digest.hexdigest(),
    }
    source = raw.encoding.get("source")
    if source and Path(source).is_file():
        with Path(source).open("rb") as stream:
            identity["raw_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return identity


def _sensor_frame(raw):
    times = pd.DatetimeIndex(raw.time_20_ku.values)
    if not times.is_unique:
        raise ValueError("Raw waveform times must be unique.")
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
    return times, sat, np.stack([along, np.cross(along, up), -up], axis=1)


def associate_points(raw, reference, along_track=140):
    """Associate EOLIS records with all raw waveforms, independently of L2.

    Retain out-of-range and ambiguous candidates for diagnostics. Only in-range
    records assigned to one waveform have ``candidate_matched=True``. Distinct
    records assigned to the same sample remain separate observations.
    """
    if not np.isfinite(along_track) or along_track <= 0:
        raise ValueError("along_track must be a positive finite distance in metres.")
    ranges = sample_ranges(raw)
    times, sat, basis = _sensor_frame(raw)
    reference = reference.reset_index(drop=True)
    ref_xyz = _ecef(reference)
    native_reference = l1b.ref_range(raw).values
    parts = []
    for k in range(len(times)):
        displacement = ref_xyz - sat[k]
        along_offset = displacement @ basis[k, 0]
        ri = np.flatnonzero(np.abs(along_offset) <= along_track)
        if not len(ri):
            continue
        distances = np.linalg.norm(displacement[ri], axis=1)
        edges = np.r_[ranges.lower_m.values[k], ranges.upper_m.values[k, -1]]
        if not np.isfinite(edges).all():
            raise ValueError(f"Non-finite raw range intervals at waveform {times[k]}")
        samples = np.searchsorted(edges, distances, side="right") - 1
        in_range = (samples >= 0) & (samples < raw.sizes["ns_20_ku"])
        native = (
            native_reference[k] + (samples - misc.reference_bin) * misc.sample_width
        )
        part = reference.iloc[ri].copy()
        part["reference_position"] = ri
        part["waveform_time"] = times[k]
        part["sample"] = samples
        part["native_range_m"] = native
        part["sample_assignment_residual_m"] = native - distances
        part["in_range"] = in_range
        parts.append(part)
    points = (
        gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=reference.crs)
        if parts
        else reference.iloc[:0].copy()
    )
    if not parts:
        for name, dtype in {
            "reference_position": "int64",
            "waveform_time": "datetime64[ns]",
            "sample": "int64",
            "native_range_m": "float64",
            "sample_assignment_residual_m": "float64",
            "in_range": "bool",
        }.items():
            points[name] = pd.Series(dtype=dtype)
    counts = points.loc[points.in_range].groupby("reference_position").size()
    points["ambiguous"] = points.reference_position.map(counts).fillna(0).gt(1)
    points["candidate_matched"] = points.in_range & ~points.ambiguous
    points["along_track_m"] = along_track
    points["eolis_observation_count"] = len(reference)
    # Columns survive GeoParquet/Feather; dataframe attrs generally do not.
    identity = _raw_identity(raw)
    for name, value in identity.items():
        points[name] = value
    points["range_interval"] = "[lower, upper)"
    points.attrs.update(
        identity,
        waveform_time_scale="TAI",
        along_track_m=along_track,
        range_interval="[lower, upper)",
        eolis_observation_count=len(reference),
    )
    return points


def evaluate_associations(raw, swath, associations):
    """Evaluate saved raw associations against one version's L2 points.

    Index labels on L2 ``time`` retain native TAI despite legacy UTC timezone
    labels. Strip those labels for identity joins; do not shift the clock.
    Persist the returned associations' provenance manifest alongside empty tables.
    """
    identity = _raw_identity(raw)
    for name, expected in identity.items():
        actual = (
            associations[name].unique().tolist()
            if len(associations) and name in associations
            else [associations.attrs.get(name)]
        )
        if actual != [expected]:
            raise ValueError(f"Association raw source mismatch: {name}")
    if len(associations) and not associations.range_interval.eq("[lower, upper)").all():
        raise ValueError("Unsupported association range interval convention.")
    points = associations.copy().reset_index(drop=True)
    times, sat, basis = _sensor_frame(raw)
    if not swath.index.is_unique:
        raise ValueError("Swath data require unique time/sample IDs.")
    if swath.empty:
        own = gpd.GeoDataFrame(
            {
                "height": pd.Series(dtype=float),
                "time": pd.Series(dtype="datetime64[ns]"),
                "sample": pd.Series(dtype="int64"),
            },
            geometry=[],
            crs=4326,
        )
    else:
        own = swath.loc[np.isfinite(swath.height)].copy().reset_index()
    own_times = pd.DatetimeIndex(own.time).tz_localize(None)
    indices = times.get_indexer(own_times)
    if (indices < 0).any():
        raise ValueError(
            "Cached swath waveform times do not match the raw L1b product."
        )
    own_index = pd.MultiIndex.from_arrays([own_times, own["sample"]])
    target = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(points.waveform_time), points["sample"]]
    )
    lookup = own_index.get_indexer(target)
    present = lookup >= 0
    points["l2_available"] = present
    points["matched"] = points.candidate_matched & present
    points["status"] = np.select(
        [~points.in_range, points.ambiguous, ~present],
        ["outside raw sample intervals", "ambiguous waveform", "missing cached sample"],
        default="matched",
    )
    for metric in _METRICS + [
        "range_reconstruction_error_m",
        "own_height_m",
        "own_longitude",
        "own_latitude",
    ]:
        points[metric] = np.nan
    if present.any():
        oi = lookup[present]
        k = times.get_indexer(pd.DatetimeIndex(points.loc[present, "waveform_time"]))
        geographic = own.to_crs(4326)
        points.loc[present, "own_height_m"] = own.height.to_numpy()[oi]
        points.loc[present, "own_longitude"] = geographic.geometry.x.to_numpy()[oi]
        points.loc[present, "own_latitude"] = geographic.geometry.y.to_numpy()[oi]
        own_xyz = _ecef(own.rename(columns={"height": "elevation"}))
        ref_xyz = _ecef(points.loc[present])
        delta = np.einsum("ij,ikj->ik", own_xyz[oi] - ref_xyz, basis[k])
        points.loc[present, "height_difference_m"] = (
            own.height.to_numpy()[oi] - points.loc[present, "elevation"].to_numpy()
        )
        points.loc[present, "along_difference_m"] = delta[:, 0]
        points.loc[present, "across_difference_m"] = delta[:, 1]
        points.loc[present, "horizontal_distance_m"] = np.hypot(
            delta[:, 0], delta[:, 1]
        )
        points.loc[present, "range_reconstruction_error_m"] = (
            np.linalg.norm(own_xyz[oi] - sat[k], axis=1)
            - points.loc[present, "native_range_m"].to_numpy()
        )
        points.loc[present, "range_difference_m"] = np.linalg.norm(
            own_xyz[oi] - sat[k], axis=1
        ) - np.linalg.norm(ref_xyz - sat[k], axis=1)
    matched = points.loc[points.matched]
    values = matched[_METRICS + _DIAGNOSTICS]
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
    samples = matched[["waveform_time", "sample"]].drop_duplicates()
    coverage = pd.Series(
        {
            "cryoswath_points": len(own),
            "eolis_observations": int(associations.eolis_observation_count.iloc[0])
            if len(associations)
            else int(associations.attrs.get("eolis_observation_count", 0)),
            "selected_observations": points.reference_position.nunique(),
            "matched_observations": len(matched),
            "matched_samples": len(samples),
            "records_sharing_samples": int(
                matched.duplicated(["waveform_time", "sample"], keep=False).sum()
            ),
            "ambiguous_observations": points.loc[
                points.ambiguous, "reference_position"
            ].nunique(),
            "missing_cached_samples": int(
                (points.status == "missing cached sample").sum()
            ),
            "outside_radial_tolerance": int((~points.in_range).sum()),
        }
    )
    points.attrs.update(
        associations.attrs,
        waveform_time_scale="TAI",
        difference_sign="CryoSwath minus EOLIS",
    )
    return {
        "points": points,
        "coverage": coverage,
        "statistics": statistics.loc[_METRICS],
        "diagnostics": statistics.loc[_DIAGNOSTICS],
    }


def _associate(raw, swath, reference, along_track=140):
    return evaluate_associations(
        raw, swath, associate_points(raw, reference, along_track)
    )


def compare_track(
    track_id,
    *,
    reprocess=False,
    along_track=140,
    cache_dir=None,
    version=None,
    **processing_kwargs,
):
    """Retrieve and compare an entire CryoSwath swath track with EOLIS.

    Existing project paths, ESA credentials and cached L2 are reused. Missing L2
    is processed through l2.from_id; processing_kwargs apply when processing is
    needed. Set reprocess=True explicitly to replace processing results.

    Returns associations and points (GeoDataFrames, including unmatched cases),
    coverage (Series) and statistics (DataFrame). Point differences are metres.

    Timestamp fields in points use different clocks: waveform_time retains native
    L1b TAI, while EOLIS time is UTC. Cached L2 waveform indices also retain TAI
    despite their UTC timezone labels. Convert clocks before comparing timestamps;
    a timezone change alone does not perform that conversion.

    Association uses the ±along_track velocity-plane window and
    half-open native range intervals. Ambiguous waveform assignments are retained
    but excluded from statistics.
    Height, across-track and approximate EOLIS timestamps do not gate matches.
    Statistics describe matched observations, not independent accuracy estimates.
    """
    if not np.isfinite(along_track) or along_track <= 0:
        raise ValueError("along_track must be a positive finite distance in metres.")
    filename = l1b._resolve_l1b_file(track_id)
    track = Path(filename).name[19:34]
    with xr.open_dataset(filename, decode_timedelta=False) as raw:
        ground = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(raw.lon_20_ku.values, raw.lat_20_ku.values),
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
        associations = associate_points(raw, reference, along_track)
        processing_kwargs.setdefault("cores", 1)
        swath, _ = l2.from_id(
            misc.cs_id_to_time(track),
            reprocess=reprocess,
            save_or_return="both",
            **processing_kwargs,
        )
        result = evaluate_associations(raw, swath, associations)
        result["associations"] = associations
    result["coverage"]["exact_duplicates_removed"] = removed
    result["points"].attrs.update(
        track_id=track,
        l1b_file=str(filename),
        eolis_files=[str(path) for path in paths],
        regions=regions,
        unsupported_rgi_regions=[code for code in codes if code not in _REGIONS],
    )
    return result
