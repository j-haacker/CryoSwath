"""Shared utilities for paths, I/O, interpolation, and compatibility patches."""

__all__ = [
    # classes
    "binary_chache",
    # functions
    "chunk_idx",
    "convert_all_esri_to_feather",
    "copy_tutorials",
    "cs_id_to_time",
    "cs_time_to_id",
    "create_config",
    "cryoswath_cli",
    "define_elev_band_edges",
    "discard_frontal_retreat_zone",
    "download_file",
    "download_auxiliary_data",
    "download_auxiliary_data_cli",
    "download_rgi_o1region",
    "download_rgi_cli",
    "effective_sample_size",
    "extend_filename",
    "fill_missing_coords",
    "filter_kwargs",
    "find_region_id",
    "flag_outliers",
    "flag_translator",
    "ftp_cs2_server",
    "gauss_filter_DataArray",
    "get_dem_reader",
    "interpolate_hypsometrically",
    "load_cs_full_file_names",
    "load_cs_ground_tracks",
    "load_glacier_outlines",
    "merge_l2_cache",
    "nan_unique",
    "request_workers",
    "repair_l2_cache",
    "rgi_code_translator",
    "rgi_o1region_translator",
    "rgi_o2region_translator",
    "sandbox_write_to",
    "sel_chunk_idx_range",
    "sel_chunk_range",
    "update_email",
    "get_tutorials_cli",
    "init_project_cli",
    "update_keyring",
    "update_keyring_cli",
    "update_netrc",
    "update_netrc_cli",
    "update_track_database",
    "update_track_database_cli",
    "warn_with_traceback",
    "weighted_mean_excl_outliers",
    "xycut",
    # variables
    "antenna_baseline",
    "cryosat_id_pattern",
    "empty_GeoDataFrame",
    "Ku_band_freq",
    "nanoseconds_per_year",
    "sample_width",
    "speed_of_light",
    "WGS84_ellpsoid",
    # patches
    "monkeypatch",
    "patched_xr_decode_tDel",
    "patched_xr_decode_scaling",
]  # path variables are currently defined below

import fnmatch
import ftplib
import getpass
import glob
import hashlib
import inspect
import netrc
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import warnings
import zipfile
from collections.abc import Callable, Iterable, Mapping
from configparser import ConfigParser
from contextlib import contextmanager
from importlib import resources as importlib_resources
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Union

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
import scipy.stats
import shapely
import stackstac
import xarray as xr
from dateutil.relativedelta import relativedelta
from defusedxml.ElementTree import fromstring as ET_from_str
from packaging.version import Version
from pyproj import CRS, Geod
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from rasterio.warp import Resampling
from scipy.constants import speed_of_light
from scipy.stats import median_abs_deviation, norm
from scipy.stats import t as student_t
from sklearn import linear_model, preprocessing
from tables import NaturalNameWarning

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:
    keyring = None

    class KeyringError(Exception):
        """Fallback keyring error if the keyring package is unavailable."""


from cryoswath import gis

_PGC_STAC_API_URL = "https://stac.pgc.umn.edu/api/v1/"
_PGC_STAC_TIMEOUT = (10, 60)


def _add_create_config_arguments(parser) -> None:
    """Add shared project-configuration arguments to an argparse parser."""
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project base directory (default: current directory).",
    )
    parser.add_argument(
        "--config",
        default="cryoswath.cfg",
        help="Configuration file to create, relative to --base-dir by default.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Data path to write to the configuration file (default: data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing configuration file.",
    )


def _create_config_from_args(args) -> None:
    """Create a CryoSwath config from parsed CLI arguments."""
    try:
        create_config(
            config_file=args.config,
            data=args.data,
            base_dir=args.base_dir,
            force=args.force,
        )
    except FileExistsError as err:
        raise SystemExit(str(err)) from err


def init_project_cli() -> None:
    """Compatibility CLI wrapper around :func:`create_config`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath-init",
        description="Create a CryoSwath project path configuration.",
    )
    _add_create_config_arguments(parser)
    args = parser.parse_args()
    try:
        create_config(
            config_file=args.config,
            data=args.data,
            base_dir=args.base_dir,
            force=args.force,
        )
    except FileExistsError as err:
        parser.error(str(err))


def create_config(
    config_file: str | Path = "cryoswath.cfg",
    data: str | Path = "data",
    *,
    base_dir: str | Path = ".",
    force: bool = False,
) -> str:
    """Create a CryoSwath path configuration without cloning data branches."""
    base_path = Path(base_dir).expanduser().resolve()
    config_path = Path(config_file).expanduser()
    if not config_path.is_absolute():
        config_path = base_path / config_path
    if config_path.exists() and not force:
        raise FileExistsError(
            f"{config_path} already exists. Use --force to overwrite it."
        )

    config = ConfigParser()
    if config_path.is_file():
        config.read(config_path)
    if "path" not in config:
        config["path"] = {}
    config["path"]["data"] = str(data)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        config.write(f)
    print(f"Wrote CryoSwath path configuration to {config_path}.")
    return str(config_path)


def init_project(
    config_file: str | Path = "cryoswath.cfg",
    data: str | Path = "data",
    *,
    force: bool = False,
    base_dir: str | Path = ".",
) -> str:
    """Compatibility wrapper around :func:`create_config`."""
    return create_config(
        config_file=config_file,
        data=data,
        base_dir=base_dir,
        force=force,
    )


# Paths ##############################################################

_CRYOSWATH_CONFIG_FILE = "cryoswath.cfg"
_LEGACY_CONFIG_FILE = "config.ini"
_CRYOSWATH_CONFIG_ENV = "CRYOSWATH_CONFIG"

_PATH_ENV_VARS = {
    "data": "CRYOSWATH_DATA",
    "l1b": "CRYOSWATH_L1B",
    "l2_swath": "CRYOSWATH_L2_SWATH",
    "l2_poca": "CRYOSWATH_L2_POCA",
    "l3": "CRYOSWATH_L3",
    "l4": "CRYOSWATH_L4",
    "tmp": "CRYOSWATH_TMP",
    "aux": "CRYOSWATH_AUX",
    "dem": "CRYOSWATH_DEM",
    "rgi": "CRYOSWATH_RGI",
    "cs_ground_tracks": "CRYOSWATH_CS_GROUND_TRACKS",
}


def _resolve_path_value(value: str | Path, base: Path) -> Path:
    """Resolve a configured path value against a base path."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _parent_dirs(start: Path) -> Iterable[Path]:
    """Yield a directory and all of its parents."""
    start = Path(start).expanduser().resolve()
    yield start
    yield from start.parents


def _discover_legacy_config_file(cwd: str | Path = None) -> Path | None:
    """Find a legacy config.ini in or above the current project layout."""
    start = Path.cwd() if cwd is None else Path(cwd)
    for directory in _parent_dirs(start):
        candidates = (
            directory / _LEGACY_CONFIG_FILE,
            directory / "scripts" / _LEGACY_CONFIG_FILE,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _user_config_file(environ: Mapping[str, str], cwd: Path) -> Path:
    """Return the per-user CryoSwath config path."""
    if "XDG_CONFIG_HOME" in environ:
        base = _resolve_path_value(environ["XDG_CONFIG_HOME"], cwd)
    else:
        base = Path.home() / ".config"
    return base / "cryoswath" / _CRYOSWATH_CONFIG_FILE


def _discover_path_config_file(
    cwd: str | Path = None, environ: Mapping[str, str] = None
) -> Path | None:
    """Find the CryoSwath path config file to use."""
    start = Path.cwd() if cwd is None else Path(cwd)
    start = start.expanduser().resolve()
    environ = os.environ if environ is None else environ

    explicit_config = environ.get(_CRYOSWATH_CONFIG_ENV)
    if explicit_config:
        return _resolve_path_value(explicit_config, start)

    for directory in _parent_dirs(start):
        candidate = directory / _CRYOSWATH_CONFIG_FILE
        if candidate.is_file():
            return candidate

    legacy_config = _discover_legacy_config_file(start)
    if legacy_config is not None:
        return legacy_config

    user_config = _user_config_file(environ, start)
    if user_config.is_file():
        return user_config
    return None


def _config_base_dir(config_file: Path | None, cwd: Path) -> Path:
    """Return the directory against which relative config paths resolve."""
    if config_file is None:
        return cwd
    if config_file.name == _LEGACY_CONFIG_FILE and config_file.parent.name == "scripts":
        return config_file.parent.parent
    return config_file.parent


def _read_path_config(config_file: Path | None) -> ConfigParser:
    """Read a path config file, returning an empty path section if absent."""
    path_config = ConfigParser()
    if config_file is not None:
        read_files = path_config.read(config_file)
        if not read_files:
            warnings.warn(
                f"Could not read CryoSwath config file {config_file}. "
                "Falling back to default paths.",
                category=UserWarning,
                stacklevel=2,
            )
    if "path" not in path_config:
        path_config["path"] = {}
    return path_config


def _resolve_config_path(
    key: str,
    default: str,
    base: Path,
    path_config: ConfigParser,
    environ: Mapping[str, str],
) -> Path:
    """Resolve one path key from env, config, or a default child path."""
    env_value = environ.get(_PATH_ENV_VARS[key])
    if env_value is not None:
        return _resolve_path_value(env_value, base)
    if key in path_config["path"]:
        return _resolve_path_value(path_config["path"][key], base)
    return base / default


def _resolve_path_configuration(
    cwd: str | Path = None, environ: Mapping[str, str] = None
) -> tuple[ConfigParser, Path | None, dict[str, Path]]:
    """Resolve CryoSwath path configuration and derived path values."""
    start = Path.cwd() if cwd is None else Path(cwd)
    start = start.expanduser().resolve()
    environ = os.environ if environ is None else environ
    config_file = _discover_path_config_file(start, environ)
    loaded_config_file = (
        config_file if config_file is not None and config_file.is_file() else None
    )
    path_config = _read_path_config(config_file)
    base_dir = _config_base_dir(loaded_config_file, start)
    path_section = path_config["path"]

    data_env = environ.get(_PATH_ENV_VARS["data"])
    if data_env is not None:
        resolved_data_path = _resolve_path_value(data_env, start)
    elif "data" in path_section:
        resolved_data_path = _resolve_path_value(path_section["data"], base_dir)
    elif "base" in path_section:
        resolved_data_path = (
            _resolve_path_value(path_section["base"], base_dir) / "data"
        )
    else:
        resolved_data_path = (
            start / "data" if loaded_config_file is None else base_dir / "data"
        )

    resolved_paths = {"data": resolved_data_path}
    for key, default in {
        "l1b": "L1b",
        "l2_swath": "L2_swath",
        "l2_poca": "L2_poca",
        "l3": "L3",
        "l4": "L4",
        "tmp": "tmp",
        "aux": "auxiliary",
    }.items():
        resolved_paths[key] = _resolve_config_path(
            key, default, resolved_data_path, path_config, environ
        )

    aux_path = resolved_paths["aux"]
    for key, default in {
        "dem": "DEM",
        "rgi": "RGI",
        "cs_ground_tracks": "CryoSat-2_SARIn_ground_tracks.feather",
    }.items():
        resolved_paths[key] = _resolve_config_path(
            key, default, aux_path, path_config, environ
        )
    return path_config, loaded_config_file, resolved_paths


def _get_path(name: str, base: Path, alternative: str = None) -> str:
    """Resolve configured project path with fallback to default."""
    key = name.lower()
    if key in config["path"]:
        return str(_resolve_path_value(config["path"][key], Path(base)))
    return str(Path(base) / (name if alternative is None else alternative))


config, _path_config_file, _resolved_paths = _resolve_path_configuration()

data_path = str(_resolved_paths["data"])
l1b_path = str(_resolved_paths["l1b"])
l2_swath_path = str(_resolved_paths["l2_swath"])
l2_poca_path = str(_resolved_paths["l2_poca"])
l3_path = str(_resolved_paths["l3"])
l4_path = str(_resolved_paths["l4"])
tmp_path = str(_resolved_paths["tmp"])
aux_path = _resolved_paths["aux"]
dem_path = _resolved_paths["dem"]
rgi_path = str(_resolved_paths["rgi"])
cs_ground_tracks_path = str(_resolved_paths["cs_ground_tracks"])

_ZENODO_AUX_CONCEPT_RECORD_API_URL = "https://zenodo.org/api/records/20241526"
_AUX_DATA_ARCHIVE_KEY = "CryoSwath-aux-data.zip"
_TUTORIAL_PACKAGE = "cryoswath.tutorials"
_TUTORIAL_PATTERN = "tutorial__*.ipynb"

__all__.extend(
    [  # pathes
        "aux_path",
        "cs_ground_tracks_path",
        "data_path",
        "dem_path",
        "l1b_path",
        "l2_swath_path",
        "l2_poca_path",
        "l3_path",
        "l4_path",
        "rgi_path",
        "tmp_path",
    ]
)


# Config #############################################################
WGS84_ellpsoid = Geod(ellps="WGS84")
# The following is advised to set for pandas<v3 (default for later versions)
pd.options.mode.copy_on_write = True

# Constants ##########################################################
antenna_baseline = 1.1676
Ku_band_freq = 13.575e9
sample_width = speed_of_light / (320e6 * 2) / 2
cryosat_id_pattern = re.compile(
    "20[12][0-9][01][0-9][0-3][0-9]T[0-2][0-9]([0-5][0-9]){2}"
)
nanoseconds_per_year = 365.25 * 24 * 60 * 60 * 1e9
_norm_isf_025 = norm.isf(0.025)
_norm_isf_25 = norm.isf(0.25)
_norm_sf_1 = norm.sf(1)
empty_GeoDataFrame = gpd.GeoDataFrame(
    columns=["dummy", "geometry"], geometry="geometry"
)
_ESA_AUTH_IDP_HOST = "eoiam-idp.eo.esa.int"
_ESA_CS2_HOST = "science-pds.cryosat.esa.int"
_ESA_CRYOSWATH_KEYRING_SERVICE = "cryoswath.esa"  # legacy keyring service name
_ESA_KEYRING_SERVICE_CANDIDATES = (
    _ESA_AUTH_IDP_HOST,
    _ESA_CS2_HOST,
    _ESA_CRYOSWATH_KEYRING_SERVICE,
)
_ESA_KEYRING_DEFAULT_USER_KEY = "__default_user__"
_ESA_KEYRING_DEFAULT_USER_KEYS = (
    _ESA_KEYRING_DEFAULT_USER_KEY,
    "default_user",
    "username",
)
_ESA_ENV_USER = "EOIAM_USER"
_ESA_ENV_PASSWORD = "EOIAM_PASSWORD"
_RGI_DOWNLOAD_BASE_URL = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files"
)
_ARCTICDEM_100M_V41_ARCHIVE_URL = (
    "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v4.1/100m/"
    "arcticdem_mosaic_100m_v4.1.tar.gz"
)
_REMA_100M_V20_FILLED_COP30_ARCHIVE_URL = (
    "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/100m/"
    "rema_mosaic_100m_v2.0_filled_cop30.tar.gz"
)

# Functions ##########################################################


# security issue?
class binary_chache:
    """Helper class to download via ftp."""

    def __init__(self):
        self._cache = bytearray()

    @property
    def cache(self):
        return self._cache.decode()

    @cache.deleter
    def cache(self):
        del self._cache[:]

    def add(self, new_part):
        """Appends to cache.

        Args:
            new_part (binary): New part.
        """
        self._cache.extend(new_part)


def chunk_idx(ds, dim, values):
    """Map coordinate value(s) to chunk index along ``dim``."""

    def _inner(val):
        if val < ds[dim][0] or val > ds[dim][-1]:
            return None
        return (
            (ds[dim].isel({dim: np.cumsum(ds.chunks[dim]) - 1}) <= val).argmin().item(0)
        )

    if isinstance(values, Iterable):
        return [_inner(val) for val in values]
    return _inner(values)


def cs_id_to_time(cs_id: str) -> pd.Timestamp:
    """Formats CryoSat-2 file time tag as timestamp.

    Args:
        cs_id (str): CryoSat-2 file time tag.

    Returns:
        pd.Timestamp: Timestamp.
    """
    return pd.to_datetime(cs_id, format="%Y%m%dT%H%M%S")


def cs_time_to_id(time: pd.Timestamp) -> str:
    """Converts timestamp to CryoSat-2 file time tag.

    Args:
        time (pd.Timestamp): Timestamp.

    Returns:
        str: CryoSat-2 file time tag.
    """
    return time.strftime("%Y%m%dT%H%M%S")


def convert_all_esri_to_feather(dir_path: str = None) -> None:
    """Converts ESRI/ArcGIS formatted files to feathers

    Finds all .shp in given directory. Not recursive.

    Args:
        dir_path (str, optional): Root directory. Defaults to None.
    """
    if dir_path is None:
        dir_path = "."
    for shp_file in glob.glob("*.shp", root_dir=dir_path):
        try:
            gis.esri_to_feather(os.path.join(dir_path, shp_file))
        except Exception as err:
            print("Error occured while translating", shp_file, " ... skipped.")
            print("Error message:", str(err))
        else:
            print("Converted", shp_file)
            basename = os.path.extsep.join(shp_file.split(os.path.extsep)[:-1])
            for associated_file in glob.glob(basename + ".*", root_dir=dir_path):
                if associated_file.split(os.path.extsep)[-1] != "feather":
                    try:
                        os.remove(os.path.join(dir_path, associated_file))
                    except Exception as err:
                        print("Couldn't clean up", associated_file, " ... skipped.")
                        print("Error message:", str(err))
                    else:
                        print("Removed", associated_file)


def dataframe_to_rioxr(df, crs):
    """Convert tabular gridded data to CRS-aware xarray dataset."""
    return fill_missing_coords(df.to_xarray()).rio.write_crs(crs)


def define_elev_band_edges(elevations: xr.DataArray) -> np.ndarray:
    """Derive elevation-band edges from robust elevation spread."""
    elev_range_80pctl = float(
        elevations.quantile([0.1, 0.9]).diff(dim="quantile").values.item(0)
    )
    if elev_range_80pctl >= 500:
        elev_bin_width = 50
    else:
        elev_bin_width = elev_range_80pctl / 10
    return np.arange(
        elevations.min(), elevations.max() + elev_bin_width, elev_bin_width
    )


def discard_frontal_retreat_zone(
    ds,
    replace_vars: list,
    main_var: str = "_median",
    elev: str = "ref_elev",
    mode: str = None,
    threshold: float = None,
    diagnostic_hook: Callable[[str, dict[str, Any]], Any] | None = None,
) -> xr.Dataset:
    """Unsets values in zone of frontal retreat

    Areas that are not continuesly glacierized distort the fitted
    polynomial that is used to fill voids, biasing aggregates of later
    products.

    This function compares the change rates in lower elevation bands. If
    the lowest bands show smaller changes than those immediately above
    them, this is interpreted as indication of a temporarily
    glacier-free surface.

    Args:
        ds (xr.Dataset): Input data with ``main_var`` and ``elev``.
        replace_vars (list[str]): Variables to set to ``NaN`` in
            detected retreat zones.
        main_var (str, optional): Variable used to detect anomalous
            low-elevation behavior. Defaults to ``"_median"``.
        elev (str, optional): Elevation reference variable used for
            banding. Defaults to ``"ref_elev"``.
        mode (str, optional): ``"temporal"`` (analyze per time step) or
            ``"trend"`` (analyze long-term trend). If ``None``, mode is
            inferred from the presence of a ``time`` dimension.
        threshold (float, optional): Detection threshold. If ``None``,
            defaults depend on ``mode``.
        diagnostic_hook (Callable, optional): Opt-in hook called with
            diagnostic event names and payloads.

    Returns:
        xr.Dataset: Dataset with flagged retreat-zone values masked.
    """

    if mode is None:
        if "time" in ds:
            mode = "temporal"
        else:
            mode = "trend"

    if threshold is None:
        if mode == "temporal":
            threshold = 10
        elif mode == "trend":
            threshold = 1
        else:
            ValueError("Value for 'mode' not allowed.")

    def custom_count(data, **kwargs):
        return ((~np.isnan(data)).sum(0) > 5).sum() > 4

    def median_mad(data, **kwargs):
        return np.nanmedian(median_abs_deviation(data, 0, **kwargs))

    try:
        bands = ds[main_var].groupby_bins(
            ds[elev], define_elev_band_edges(ds[elev])[:5], include_lowest=True
        )
    except ValueError as err:
        if str(err) == "arange: cannot compute length":
            return ds
        raise

    if mode == "temporal":
        if (ds[main_var].count("time") > 5).sum() < 5 or not bands.reduce(
            custom_count, ...
        ).all():
            return ds
        tmp = bands.reduce(median_mad, ..., nan_policy="omit")
    else:
        if ds[main_var].count() < 5 or not (bands.count() > 4).all():
            return ds
        tmp = np.abs(bands.mean())

    if not (tmp > threshold).any():
        return ds

    # Temporary downstream workaround for the xarray IntervalIndex idxmax
    # regression. Remove once upstream idxmax() works again for IntervalIndex
    # coordinates.
    front_mask = (tmp > tmp.max() / 2).cumsum() != 0
    front_positions = np.flatnonzero(front_mask.to_numpy())
    if front_positions.size == 0:
        return ds
    bin_dim = front_mask.dims[0]
    front_bin = front_mask[bin_dim].to_numpy()[front_positions[0]]

    if diagnostic_hook is not None:
        diagnostic_hook(
            "frontal_retreat_zone.threshold",
            {
                "ds": ds,
                "replace_vars": replace_vars,
                "main_var": main_var,
                "elev": elev,
                "mode": mode,
                "threshold": threshold,
                "band_values": tmp,
                "front_mask": front_mask,
                "front_bin": front_bin,
                "diagnostic_context": {},
            },
        )

    if isinstance(replace_vars, str):
        replace_vars = [replace_vars]
    for var_ in replace_vars:
        ds[var_] = xr.where(
            ~(ds[elev] < front_bin.left),
            ds[var_],
            (
                np.nan
                if "_FillValue" not in ds[var_].attrs
                else ds[var_].attrs["_FillValue"]
            ),
            keep_attrs=True,
        )

    return ds


def _read_stac(item):
    """Private helper to read ArcticDEM or REMA tiles"""
    if "proj:code" in item.properties:
        code = item.properties["proj:code"]
        if code.lower().startswith("epsg:"):
            code = int(code.split(":")[-1])
        else:
            raise Exception(f"Implement parsing proj:code format {code}")
    else:
        raise Exception(f"Implement getting crs from properties {item.properties}")

    tmp = stackstac.stack(item, epsg=code)
    # nodata and data_type are coordinates along dimension band.
    # they are drop on conversion to dataset and must be stored
    # before.
    coords_band_dim = {
        k: v
        for k, v in tmp.coords.items()
        if "band" in v.dims and k in ["nodata", "data_type"]
    }
    tmp = tmp.to_dataset("band", promote_attrs=True)
    tmp = xr.Dataset(
        {
            da.name: da.assign_attrs(
                {k: v.sel(band=da.name).item(0) for k, v in coords_band_dim.items()}
            )
            .drop_vars([k for k, v in tmp.coords.items() if len(v.dims) == 0])
            .squeeze()
            for da in tmp.data_vars.values()
        }
    )
    return (
        xr.Dataset(
            {
                da.name: da.drop_attrs()
                .astype(da.attrs["data_type"])
                .assign_attrs(encoding={"_FillValue": da.attrs["nodata"]})
                for da in tmp.data_vars.values()
            }
        )
        .drop_vars(["time", "id"])
        .rio.write_crs(code)
    )


def _is_pgc_stac_connectivity_error(exc: Exception) -> bool:
    """Return whether an exception looks like a PGC STAC network failure."""
    if isinstance(exc, (TimeoutError, requests.exceptions.RequestException)):
        return True
    if not isinstance(exc, APIError):
        return False

    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "connection",
            "connect timeout",
            "max retries exceeded",
            "network",
            "read timed out",
            "timed out",
            "timeout",
        )
    )


def _raise_pgc_stac_unavailable(exc: Exception) -> None:
    raise RuntimeError(
        f"The PGC STAC API at {_PGC_STAC_API_URL} did not respond within the "
        "configured timeout or could not be reached. This is usually an upstream "
        "service availability issue. Retry later or check the endpoint manually "
        "with curl."
    ) from exc


def _open_pgc_stac_catalog():
    try:
        return Client.open(
            _PGC_STAC_API_URL,
            stac_io=StacApiIO(max_retries=0),
            timeout=_PGC_STAC_TIMEOUT,
        )
    except (APIError, TimeoutError, requests.exceptions.RequestException) as exc:
        if _is_pgc_stac_connectivity_error(exc):
            _raise_pgc_stac_unavailable(exc)
        raise


def _pgc_stac_items(catalog, gpd_obj):
    try:
        collections = catalog.collection_search(
            q="((arcticdem AND v4+1) OR (rema AND v2)) AND 32m"
        ).collections()
        return list(
            catalog.search(
                collections=[coll.id for coll in collections],
                # not sure how this behaves if it covers the poles
                bbox=gpd_obj.to_crs(4326).total_bounds,
            ).items()
        )
    except (APIError, TimeoutError, requests.exceptions.RequestException) as exc:
        if _is_pgc_stac_connectivity_error(exc):
            _raise_pgc_stac_unavailable(exc)
        raise


def download_dem(
    gpd_obj: Union[gpd.GeoSeries, gpd.GeoDataFrame, gpd.array.GeometryArray],
    provider: Literal["PGC"] = "PGC",
):
    """
    Download DEM tiles that intersect the provided geometries

    Parameters
    ----------
    gpd_obj : geopandas.GeoSeries | geopandas.GeoDataFrame | sequence[Geometry]
        Geometries defining the area of interest. Can be a geopandas
        geometry object (needs `.to_crs()` and `.total_bounds`). The
        input will be reprojected to EPSG:4326 if necessary; its
        total_bounds are used as the STAC bbox for item discovery.
    provider : Literal['PGC'], optional
        Data provider to query. Only "PGC" is implemented (queries the PGC STAC API at
        https://stac.pgc.umn.edu/api/v1/). Default is "PGC".

    Behavior
    --------
    - Searches the PGC STAC catalog for arcticdem (v4.1) and rema (v2)
      32 m collections covering the provided bbox.
    - Creates a Zarr store at Path(dem_path) / '<collection_id>.zarr'.
      Note: the function expects a caller-defined variable `dem_path` to
      exist and be a valid filesystem path.
    - Initializes the store on a fixed regular grid
      (x,y in [-3_500_000, 3_500_000]) with 100 m spacing and chunking
      tuned for large tile writes.
    - For each discovered STAC item:
      - Skips writing if the existing store already contains sufficient
        data for the item's bbox.
      - Reads the item into an xarray.Dataset, reprojects/resamples it to
        match the store grid (using rioxarray and rasterio Resampling),
        fills nodata values from the existing store, and writes the result
        back into the Zarr store using region writes.
    - Uses external libraries (stac client, rioxarray, xarray, shapely,
      numpy); network I/O and heavy disk operations are performed.
    """
    if provider == "PGC":
        catalog = _open_pgc_stac_catalog()
        # transforming collection extent is difficult, maybe the code behind
        # rioxr transform_bounds helps
        limits = {"x": (-3_500_000, 3_500_000), "y": (-3_500_000, 3_500_000)}
        items = _pgc_stac_items(catalog, gpd_obj)

    this_dem_path = Path(dem_path) / (
        items[0].get_collection().id + "_100m-mean.zarr"  # pyright: ignore[reportOptionalMemberAccess]
    )  # don't .with_suffix; . in name!
    this_dem_path.parent.mkdir(parents=True, exist_ok=True)

    for item in items:
        if not this_dem_path.exists():  # init dem store
            (
                xr.full_like(_read_stac(item), np.nan)
                .reindex(
                    {
                        xy: np.arange(limits[xy][0], limits[xy][1] + 1, 100)
                        for xy in ["x", "y"]
                    }
                )
                .chunk(x=1000, y=1000)
            ).to_zarr(this_dem_path, mode="w", compute=False)

        parent = xr.open_zarr(this_dem_path, decode_coords="all", mask_and_scale=True)
        if (
            parent["count"].rio.clip_box(*item.properties["proj:bbox"]).mean().compute()
            > 0.1
        ):
            continue
        ds = _read_stac(item)
        # # the general case:
        # x0, y0, x1, y1 = ds.rio.bounds()
        # excerpt = parent.pipe(sel_chunk_range, x=[x0, x1], y=[y0, y1]).load()
        # however, if chunks tuned to tiles:
        c = shapely.box(*item.properties["proj:bbox"]).centroid
        excerpt = parent.pipe(
            sel_chunk_range, **{xy: [getattr(c, xy)] * 2 for xy in ["x", "y"]}
        ).load()
        add = ds.map(
            lambda da: da.rio.reproject_match(
                excerpt, resampling=Resampling.average
            ).astype(da.dtype)
        )
        add = add.map(lambda da: da.where(da != da.attrs["encoding"]["_FillValue"]))
        add = add.map(lambda da: da.fillna(excerpt[da.name]))
        add.drop_attrs().drop_vars(["spatial_ref"]).to_zarr(
            this_dem_path, region="auto"
        )


def _stream_download_response(response, tmp_file) -> None:
    """Write streamed HTTP response content to an open binary file."""
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=8192):
        tmp_file.write(chunk)


def download_file(
    url: str,
    dest: str | Path,
    auth: tuple[str, str] | None = None,
    timeout: int | float = 120,
) -> str:
    """Download ``url`` to ``dest`` using streamed HTTP requests."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dest_path.name}.",
            suffix=".part",
            dir=dest_path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            with requests.get(url, stream=True, auth=auth, timeout=timeout) as r:
                _stream_download_response(r, tmp_file)
        os.replace(temp_path, dest_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return str(dest_path)


def _download_earthdata_file(
    url: str,
    dest: str | Path,
    timeout: int | float = 120,
) -> str:
    """Download an Earthdata-protected URL to ``dest`` using earthaccess."""
    try:
        import earthaccess
    except ImportError as err:
        raise RuntimeError(
            "earthaccess is required for NASA Earthdata downloads. Install the "
            "project dependencies or add `earthaccess` to your environment."
        ) from err

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dest_path.name}.",
            suffix=".part",
            dir=dest_path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            earthaccess.login(strategy="environment", persist=False)
            session = earthaccess.get_requests_https_session()
            with session.get(url, stream=True, timeout=timeout) as response:
                _stream_download_response(response, tmp_file)
        os.replace(temp_path, dest_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return str(dest_path)


def _fetch_zenodo_record_metadata(timeout: int | float = 120) -> dict:
    """Fetch metadata for the latest CryoSwath auxiliary-data Zenodo record."""
    response = requests.get(_ZENODO_AUX_CONCEPT_RECORD_API_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _zenodo_auxiliary_archive(record_metadata: dict) -> dict:
    """Return metadata for the single auxiliary-data archive in a Zenodo record."""
    files = record_metadata.get("files", [])
    matches = [
        file_info
        for file_info in files
        if file_info.get("key") == _AUX_DATA_ARCHIVE_KEY
    ]
    if not matches and len(files) == 1:
        matches = files
    if len(matches) != 1:
        raise RuntimeError(
            "Zenodo auxiliary-data record should contain exactly one "
            f"{_AUX_DATA_ARCHIVE_KEY!r} file."
        )
    file_info = matches[0]
    if "checksum" not in file_info:
        raise RuntimeError("Zenodo auxiliary-data archive metadata has no checksum.")
    try:
        file_info["links"]["self"]
    except KeyError as err:
        raise RuntimeError(
            "Zenodo auxiliary-data archive metadata has no download link."
        ) from err
    return file_info


def _verify_checksum(path: str | Path, checksum: str) -> None:
    """Verify a checksum string in the form ``algorithm:hex``."""
    try:
        algorithm, expected = checksum.split(":", 1)
    except ValueError as err:
        raise RuntimeError(f"Unsupported checksum format: {checksum!r}") from err
    try:
        digest = hashlib.new(algorithm)
    except ValueError as err:
        raise RuntimeError(f"Unsupported checksum algorithm: {algorithm!r}") from err
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {checksum}, "
            f"got {algorithm}:{actual}."
        )


def _validate_zip_members(archive_path: str | Path) -> None:
    """Reject invalid zip files and paths that would escape extraction root."""
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Downloaded archive is not a zip file: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(member_name)
            if (
                member_name.startswith("/")
                or re.match(r"^[A-Za-z]:", member_name)
                or ".." in member_path.parts
            ):
                raise RuntimeError(
                    f"Unsafe path {member.filename!r} in archive {archive_path}."
                )


def _merge_extracted_tree(source_dir: Path, target_dir: Path, *, force: bool) -> None:
    """Move extracted files into a target tree, preserving existing files by default."""
    for source in sorted(source_dir.rglob("*")):
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        if source.is_dir():
            if target.exists() and not target.is_dir():
                if not force:
                    raise RuntimeError(
                        f"Cannot create directory {target}; a file already exists. "
                        "Use force=True to replace it."
                    )
                target.unlink()
            target.mkdir(parents=True, exist_ok=True)
            continue

        if target.exists():
            if not force:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


def _extract_auxiliary_archive(
    archive_path: str | Path,
    target_dir: str | Path,
    *,
    force: bool,
) -> None:
    """Safely extract an auxiliary-data zip archive into ``target_dir``."""
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    _validate_zip_members(archive_path)
    extract_root = Path(
        tempfile.mkdtemp(prefix=f".{archive_path.stem}.", dir=target_dir)
    )
    try:
        shutil.unpack_archive(archive_path, extract_root, format="zip")
        _merge_extracted_tree(extract_root, target_dir, force=force)
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)


def download_auxiliary_data(
    base_dir: str | Path = ".",
    *,
    force: bool = False,
    timeout: int | float = 120,
) -> str:
    """Download and install the Zenodo CryoSwath auxiliary-data snapshot."""
    base_path = Path(base_dir).expanduser().resolve()
    _, _, paths = _resolve_path_configuration(cwd=base_path)
    target_dir = Path(paths["aux"])
    target_dir.mkdir(parents=True, exist_ok=True)

    record_metadata = _fetch_zenodo_record_metadata(timeout=timeout)
    archive_metadata = _zenodo_auxiliary_archive(record_metadata)
    archive_url = archive_metadata["links"]["self"]
    checksum = archive_metadata["checksum"]

    temp_archive = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{archive_metadata.get('key', _AUX_DATA_ARCHIVE_KEY)}.",
            suffix=".zip",
            dir=target_dir,
            delete=False,
        ) as archive_file:
            temp_archive = Path(archive_file.name)
        download_file(archive_url, temp_archive, timeout=timeout)
        _verify_checksum(temp_archive, checksum)
        _extract_auxiliary_archive(temp_archive, target_dir, force=force)
    finally:
        if temp_archive is not None and temp_archive.exists():
            temp_archive.unlink()
    return str(target_dir)


def _tutorial_resources():
    """Yield packaged tutorial notebook resources."""
    tutorial_root = importlib_resources.files(_TUTORIAL_PACKAGE)
    tutorials = [
        item
        for item in tutorial_root.iterdir()
        if item.is_file() and fnmatch.fnmatch(item.name, _TUTORIAL_PATTERN)
    ]
    yield from sorted(tutorials, key=lambda item: item.name)


def copy_tutorials(
    destination: str | Path = None,
    *,
    base_dir: str | Path = ".",
    force: bool = False,
) -> str:
    """Copy packaged tutorial notebooks into a project directory."""
    base_path = Path(base_dir).expanduser().resolve()
    destination_path = (
        Path(destination) if destination is not None else Path("tutorials")
    )
    if not destination_path.is_absolute():
        destination_path = base_path / destination_path

    resources = list(_tutorial_resources())
    conflicts = [
        destination_path / resource.name
        for resource in resources
        if (destination_path / resource.name).exists()
    ]
    if conflicts and not force:
        conflict_list = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Tutorial file(s) already exist: {conflict_list}. "
            "Use --force to overwrite."
        )

    destination_path.mkdir(parents=True, exist_ok=True)
    for resource in resources:
        target = destination_path / resource.name
        if target.exists() and force:
            target.unlink()
        target.write_bytes(resource.read_bytes())
    return str(destination_path)


def drop_small_glaciers(
    df: pd.DataFrame,
    area_threshold: float,  # in km²
) -> pd.DataFrame:
    """Remove glaciers smaller than threshold

    Designed for use with RGI data. Requires column "area_km2".

    Args:
        gdf (pd.DataFrame): RGI glacier/complex (Geo)DataFrame.
        area_threshold (float): Minimum glacier size in km².

    Returns:
        pd.DataFrame: As input without rows that do not pass threshold.
    """
    small_glacier_mask = df.area_km2 < area_threshold
    if sum(small_glacier_mask) != 0:
        warnings.warn(
            f"Dropping {sum(small_glacier_mask)} glaciers < {area_threshold} km² "
            "from RGI o1 region."
        )
    return df[~small_glacier_mask]


def effective_sample_size(weights: np.ndarray | xr.DataArray):
    """Calculates the effective sample size based on sample weights

    Args:
        weights (np.ndarray | xr.DataArray): Weights

    Returns:
        float | xr.DataArray: Effective sample size.
    """
    return weights.sum() ** 2 / (weights**2).sum()


def extend_filename(file_name: str, extension: str) -> str:
    """Adds string at end of file name, before last "."

    Args:
        file_name (str): File name or path.
        extension (str): String to insert at end.

    Returns:
        str: As input, including extension.
    """
    fn_parts = file_name.split(os.path.extsep)
    return (
        os.path.extsep.join(fn_parts[:-1]) + extension + os.path.extsep + fn_parts[-1]
    )


def fill_missing_coords(
    l3_data, minx: int = 9e7, miny: int = 9e7, maxx: int = -9e7, maxy: int = -9e7
) -> xr.Dataset:
    """Reindex x/y coordinates to fill missing grid cells."""
    # previous version inspired by user9413641
    # https://stackoverflow.com/questions/68207994/fill-in-missing-index-positions-in-xarray-dataarray
    # ! resx, resy = [int(r) for r in l3_data.rio.resolution()] don't
    # use `rio.resolution()`: this assumes no holes which renders this
    # function obsolete
    l3_data = l3_data.sortby("x").sortby("y")  # ensure monotonix x and y
    for dim, _min, _max in [("x", minx, maxx), ("y", miny, maxy)]:
        if len(l3_data[dim]) == 1:
            continue
        res = l3_data[dim].diff(dim).min().values.astype("int")
        _min = int(_min + res / 2)
        _max = int(_max - res / 2)
        if l3_data[dim].min().values < minx:
            _min = l3_data[dim].min().values.astype("int")
        else:
            _min = int(_min + (l3_data[dim].min().values - _min) % res - res)
        if l3_data[dim].max().values > _max:
            _max = l3_data[dim].max().values.astype("int")
        else:
            _max = int(_max - (_max - l3_data[dim].max().values) % res + res)
        if hasattr(l3_data, "data_vars"):
            fill_value = {
                _var: (
                    l3_data[_var].attrs["_FillValue"]
                    if "_FillValue" in l3_data[_var].attrs
                    else np.nan
                )
                for _var in [*l3_data.data_vars, "x", "y"]
            }
        else:
            fill_value = getattr(l3_data.attrs, "_FillValue", np.nan)
        l3_data = l3_data.reindex(
            {dim: range(_min, _max + 1, res)},
            method=None,
            copy=False,
            fill_value=fill_value,
        )
    return l3_data


# ! make recursive
def filter_kwargs(
    func: callable,
    kwargs: dict,
    *,
    blacklist: list[str] = None,
    whitelist: list[str] = None,
) -> dict:
    """Automatically reduces dict to accepted inputs

    Detects expected key-word arguments of a function and only passes
    those. Use black- and whitelists to refine.

    Args:
        func (callable): Target function.
        kwargs (dict): KW-args to be filtered.
        blacklist (list[str], optional): Blacklist undesired arguments.
            Defaults to None.
        whitelist (list[str], optional): Include extra arguments, that are
            not part of the functions signature. Defaults to None.

    Returns:
        dict: Filtered kw-args.
    """

    def ensure_list(tmp_list):
        if tmp_list is None:
            return []
        elif isinstance(tmp_list, str):
            return [tmp_list]
        else:
            return tmp_list

    blacklist = ensure_list(blacklist)
    whitelist = ensure_list(whitelist)
    params = inspect.signature(func).parameters
    return {
        k: v
        for k, v in kwargs.items()
        if (k in params and k not in blacklist) or k in whitelist
    }


def find_region_id(location: any, scope: str = "o2") -> str:
    """Returns RGI id for multitude of inputs

    Special behavior in Greenland! If o2 region is requested, return id of
    "custom" subregion: 05-11--05-15 for N, W, SW, SE, E. See geo-feathers
    in `data/auxiliary/RGI/05-1*.feather`.

    Args:
        location (any): Can be a geo-referenced xarray.DataArray, a
            geopandas.GeoDataFrame or Series, or a shapely.Geometry.
        scope (str, optional): One of "o1", "o2", or "basin". Defaults to
            "o2".

    Raises:
        Exception: `scope` is "o2" and `location` is in Greenland but
            - not in one of the custom subregions or
            - in more than one custom subregion.

    Returns:
        str: RGI id.
    """
    if isinstance(location, xr.DataArray) or isinstance(location, xr.Dataset):
        left, lower, right, upper = location.rio.transform_bounds(4326)
        location = shapely.Point(left + (right - left) / 2, lower + (upper - lower) / 2)
    if isinstance(location, gpd.GeoDataFrame):
        location = location.geometry
    if isinstance(location, gpd.GeoSeries):
        location = location.to_crs(4326).union_all("coverage")
    if not isinstance(location, shapely.Geometry):
        if isinstance(location, tuple) or (
            isinstance(location, list) and len(location) < 3
        ):
            location = shapely.Point(location[1], location[0])
        else:
            location = shapely.Polygon([(coord[1], coord[0]) for coord in location])
    rgi_o2_gpdf = gpd.read_feather(
        os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather")
    )
    rgi_region = rgi_o2_gpdf[rgi_o2_gpdf.contains(location.centroid)]
    if rgi_region.empty:
        raise ValueError(f"Location {location} is not in any RGI o2 region.")
    if scope == "o1":
        return rgi_region["o1region"].values[0]
    elif scope == "o2":
        out = rgi_region["o2region"].values[0]
        if out == "05-01":
            sub_o2 = gpd.GeoSeries(
                [
                    _load_o2region(f"05-1{i + 1}").union_all("coverage").envelope
                    for i in range(5)
                ],
                crs=4326,
            )
            contains_location = sub_o2.contains(location)
            if not any(contains_location):
                raise Exception(
                    f"Location {location} not in any of Greenlands subregions "
                    "(N,W,SW,SE,E)."
                )
            elif sum(contains_location) > 1:
                raise Exception(
                    f"Location {location} is in multiple subregions (N,W,SW,SE,E)."
                )
            out = f"05-1{contains_location.argmax() + 1}"
        return out
    elif scope == "basin":
        rgi_glacier_gpdf = _load_o2region(rgi_region["o2region"].values[0], "glaciers")
        return rgi_glacier_gpdf[rgi_glacier_gpdf.contains(location.centroid)][
            "rgi_id"
        ].values[0]
    raise Exception('`scope` can be one of "o1", "o2", or "basin".')

    # ! tbi: if only small region/one glacier, make get its
    # to_planar = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3413))
    # if shapely.ops.transform(to_planar.transform, region_outlines).area >


def flag_outliers(
    data,
    *,
    weights=None,
    stat: callable = np.median,
    deviation_factor: float = 3,
    scaling_factor: float = 2 * 2**0.5 * scipy.special.erfinv(0.5),
):
    """Flags data that is considered outlier given a set of assumptions

    Data too far from a reference point is marked. Works analogous comparing
    data to its mean in terms of standard deviations.

    Function was meant to be versatile. However, I'm not sure it makes
    sense using it with other than the "usual" statistics: mean and median.

    It defaults to marking data further from the median than 3 scaled MADs.

    Args:
        data (ArrayLike): If data is an array, outliers will be flagged
            along first dimension (given `stat` works like most numpy
            functions).
        weights (ArrayLike): If weights are provided, they are passed as
            the keyword argument to `stat`.
        stat (callable, optional): Function to return first and second
            reference points. Defaults to np.median.
        deviation_factor (float, optional): Allowed number of reference
            point distances between data and first reference point.
            Defaults to 3.
        scaling_factor (float, optional): Reference distance scaling.
            Defaults to 2*2**.5*scipy.special.erfinv(.5)).

    Returns:
        bool, shaped like input: Mask that is positive for outliers.
    """
    if weights is None:
        first_moment = stat(data)
    else:
        first_moment = stat(data, weights=weights)
    # print(first_moment)
    deviation = np.abs(data - first_moment)
    # print(deviation)
    if weights is None:
        deviation_limit = stat(deviation) * deviation_factor * scaling_factor
    else:
        deviation_limit = (
            stat(deviation, weights=weights) * deviation_factor * scaling_factor
        )
    # print(deviation_limit)
    return deviation > deviation_limit


def flag_translator(cs_l1b_flag):
    """Retrieves the meaning of a flag from the attributes.

    If attributes contain "flag_masks", it converts the value to a
    binary mask and returns a list of flags. Else it expects
    "flag_values" and interprets and returns the flag as one of a set of
    options.

    This works for CryoSat-2 L1b netCDF data. It depends on the
    attribute structure and names.

    Args:
        cs_l1b_flag (0-dim xarray.DataArray): Flag variable of waveform.

    Returns:
        list or string: List of flags or single option, depending on
        flag.
    """
    if "flag_masks" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(
            data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
            index=np.log2(
                np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64"))
            ).astype("int"),
        ).sort_index()
        bin_str = bin(int(cs_l1b_flag.values))[2:]
        flag_list = []
        for i, b in enumerate(reversed(bin_str)):
            if b == "1":
                try:
                    flag_list.append(flag_dictionary.loc[i])
                except KeyError:
                    raise (
                        f"Unkown flag: {2**i}! This points to a bug either in the code "
                        "or in the data!"
                    )
        return flag_list
    else:
        flag_dictionary = pd.Series(
            data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
            index=cs_l1b_flag.attrs["flag_values"],
        )
        return flag_dictionary.loc[int(cs_l1b_flag.values)]


@contextmanager
def ftp_cs2_server(**kwargs):
    """Yield authenticated FTP connection to ESA CryoSat server."""
    user, password, source = _resolve_esa_ftp_credentials()
    with ftplib.FTP_TLS(_ESA_CS2_HOST, **kwargs) as ftp:
        try:
            ftp.login(user=user, passwd=password)
        except ftplib.error_perm as err:
            raise RuntimeError(
                "ESA FTP authentication failed using credentials from "
                f"{source}. Configure keyring via cryoswath update-keyring, set "
                f"{_ESA_ENV_USER}/{_ESA_ENV_PASSWORD}, or use "
                "~/.netrc (plaintext fallback)."
            ) from err
        yield ftp


def _resolve_esa_env_credentials() -> tuple[str, str, str] | None:
    """Resolve ESA credentials from environment variables."""
    env_user = os.environ.get(_ESA_ENV_USER)
    env_password = os.environ.get(_ESA_ENV_PASSWORD)
    if env_user and env_password:
        return env_user, env_password, "environment variables"
    if env_user or env_password:
        warnings.warn(
            f"Both {_ESA_ENV_USER} and {_ESA_ENV_PASSWORD} are required to use "
            "environment-variable credentials. Falling back to keyring/netrc.",
            category=UserWarning,
            stacklevel=2,
        )
    return None


def _resolve_esa_keyring_credentials() -> tuple[str, str, str] | None:
    """Resolve ESA credentials from keyring if available."""
    if keyring is None:
        return None
    try:
        env_users = []
        user = os.environ.get(_ESA_ENV_USER)
        if user:
            env_users.append(user)
        for service in _ESA_KEYRING_SERVICE_CANDIDATES:
            for user in env_users:
                password = keyring.get_password(service, user)
                if password:
                    return user, password, f"keyring service {service}"
            for username_key in _ESA_KEYRING_DEFAULT_USER_KEYS:
                keyring_user = keyring.get_password(service, username_key)
                if keyring_user:
                    keyring_password = keyring.get_password(service, keyring_user)
                    if keyring_password:
                        return (
                            keyring_user,
                            keyring_password,
                            f"keyring service {service}",
                        )
    except KeyringError as err:
        warnings.warn(
            f"Could not read ESA credentials from keyring: {err}",
            category=UserWarning,
            stacklevel=2,
        )
    return None


def _resolve_esa_ftp_credentials() -> tuple[str, str, str]:
    """Resolve ESA credentials from env, keyring, netrc, and legacy config."""

    env_auth = _resolve_esa_env_credentials()
    if env_auth is not None:
        return env_auth

    keyring_auth = _resolve_esa_keyring_credentials()
    if keyring_auth is not None:
        return keyring_auth

    try:
        netrc_auth = netrc.netrc().authenticators(_ESA_CS2_HOST)
    except (FileNotFoundError, netrc.NetrcParseError):
        netrc_auth = None
    if netrc_auth is not None:
        login, _, password = netrc_auth
        if login and password:
            return login, password, "~/.netrc"
        if password and not login:
            raise RuntimeError(
                f"~/.netrc entry for {_ESA_CS2_HOST} is missing login. "
                "Anonymous FTP login is no longer supported."
            )

    legacy_config_file = _discover_legacy_config_file()
    config = ConfigParser()
    if legacy_config_file is not None:
        config.read(legacy_config_file)
    if "user" in config:
        section = config["user"]
        if "name" in section and "password" in section:
            warnings.warn(
                "Using [user] name/password from config.ini is deprecated. "
                "Prefer environment variables, keyring, or ~/.netrc.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return (
                section["name"],
                section["password"],
                "config.ini [user] name/password",
            )

    raise RuntimeError(
        f"No ESA credentials found. Configure {_ESA_ENV_USER} and "
        f"{_ESA_ENV_PASSWORD}, keyring via cryoswath update-keyring, or "
        "~/.netrc (plaintext fallback), or use legacy config.ini [user] "
        "name/password. Anonymous login is no longer supported."
    )


def gauss_filter_DataArray(
    da: xr.DataArray, dim: str, window_extent: int, std: int
) -> xr.DataArray:
    """Low-pass filters input array.

    Convolves each vector of an array along the specified dimension with a
    normalized gauss-function having the specified standard deviation.

    Args:
        da (xr.DataArray): Data to be filtered.
        dim (str): Dimension to apply filter along.
        window_extent (int): Window width. If not uneven, it is increased.
        std (int): Standard deviation of gauss-filter.

    Returns:
        xr.DataArray: Filtered array, preserving input dimensions.
    """
    # force window_extent to be uneven to ensure center to be where expected
    half_window_extent = window_extent // 2
    window_extent = 2 * half_window_extent + 1
    gauss_weights = scipy.stats.norm.pdf(
        np.arange(-half_window_extent, half_window_extent + 1), scale=std
    )
    gauss_weights = xr.DataArray(
        gauss_weights / np.sum(gauss_weights), dims=["window_dim"]
    )
    if np.iscomplexobj(da):
        helper = (
            da.rolling({dim: window_extent}, center=True, min_periods=1)
            .construct("window_dim")
            .dot(gauss_weights)
        )
        return helper / np.abs(helper)
    else:
        return (
            da.rolling({dim: window_extent}, center=True, min_periods=1)
            .construct("window_dim")
            .dot(gauss_weights)
        )


def get_dem_reader(data: any = None) -> rasterio.DatasetReader:
    """Determines which DEM to use

    Attempts to determine location of `data` and returns appropriate
    `rasterio.io.DatasetReader` or `xarray.DataArray`. Only implemented
    for ArcticDEM and REMA.

    Args:
        data (any): Defaults to None.

    Raises:
        Exception: If region can't be inferred or path doesn't exist.

    Returns:
        rasterio.DatasetReader: Reader pointing to the file.
    """

    raster_extensions = ["tif", "nc", "zarr"]

    def reader_or_store(path: Path):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".tif":
            return rasterio.open(path)
        elif path.suffix == ".nc":
            return xr.open_dataset(path, decode_coords="all", engine="h5netcdf").dem
        elif path.suffix == ".zarr":
            return xr.open_dataset(path, decode_coords="all", engine="zarr").dem
        else:
            raise Exception(str(path) + " cant be read.")

    if isinstance(data, shapely.Geometry):
        lat = np.mean(data.bounds[1::2])
    elif (
        isinstance(data, float)
        or isinstance(data, int)
        or (isinstance(data, np.ndarray) and data.size == 1)
    ):
        lat = data
    elif "lat_20_ku" in data:
        lat = data.lat_20_ku.values[0]
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        lat = np.mean(data.rio.transform_bounds("EPSG:4326")[1::2])
    elif isinstance(data, gpd.GeoSeries) or isinstance(data, gpd.GeoDataFrame):
        lat = data.to_crs(4326).union_all("coverage").centroid.y
    elif isinstance(data, str):
        if data.lower() in ["arctic", "arcticdem"]:
            lat = 90
        elif data.lower() in ["antarctic", "rema"]:
            lat = -90
        elif os.path.sep in data:
            return reader_or_store(data)
        elif any([data.split(".")[-1] in raster_extensions]):
            return reader_or_store(dem_path / data)
    if "lat" not in locals():
        raise Exception(
            f"`get_dem_reader` could not handle the input of type {data.__class__}. "
            "See doc for further info."
        )
    if lat > 0:
        preferred_dem_filename = "arcticdem_mosaic_100m_v4.1_dem.tif"
        fallback_dem_filename = "arcticdem-mosaics-v4.1-32m_100m-mean.zarr"
    else:
        preferred_dem_filename = "rema_mosaic_100m_v2.0_filled_cop30_dem.tif"
        fallback_dem_filename = "rema-mosaics-v2.0-32m_100m-mean.zarr"
    dem_filename = preferred_dem_filename
    if not (dem_path / dem_filename).is_file():
        dem_filename = fallback_dem_filename

    def default_dem_archive_url(filename: str) -> str | None:
        if filename.startswith("arcticdem_mosaic_100m_v4.1_"):
            return _ARCTICDEM_100M_V41_ARCHIVE_URL
        if filename.startswith("rema_mosaic_100m_v2.0_filled_cop30_"):
            return _REMA_100M_V20_FILLED_COP30_ARCHIVE_URL
        return None

    def download_default_dem(filename: str, timeout: int | float = 120) -> Path:
        archive_url = default_dem_archive_url(filename)
        if archive_url is None:
            raise FileNotFoundError(
                f"No automatic download source configured for DEM file {filename}."
            )
        dem_dir = Path(dem_path)
        dem_dir.mkdir(parents=True, exist_ok=True)
        archive_name = archive_url.rsplit("/", maxsplit=1)[-1]
        archive_path = dem_dir / archive_name
        extract_root = None
        download_file(
            url=archive_url,
            dest=archive_path,
            auth=None,
            timeout=timeout,
        )
        try:
            extract_root = Path(
                tempfile.mkdtemp(prefix=f".{archive_name}.", dir=str(dem_dir))
            )
            shutil.unpack_archive(archive_path, extract_root)
            extract_entries = list(extract_root.iterdir())
            source_dir = (
                extract_entries[0]
                if len(extract_entries) == 1 and extract_entries[0].is_dir()
                else extract_root
            )
            for entry in list(source_dir.iterdir()):
                target = dem_dir / entry.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(entry), str(target))
        finally:
            if archive_path.exists():
                archive_path.unlink()
            if extract_root is not None and extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)

        output_file = dem_dir / filename
        if not output_file.is_file():
            raise FileNotFoundError(
                "Downloaded DEM archive did not contain expected file "
                f"{output_file.name}."
            )
        return output_file

    if not (dem_path / dem_filename).exists():
        archive_url = default_dem_archive_url(preferred_dem_filename)
        if archive_url is not None and not (dem_path / preferred_dem_filename).exists():
            warnings.warn(
                f"DEM file {preferred_dem_filename} is missing. "
                "Attempting automatic download now.",
                category=UserWarning,
                stacklevel=2,
            )
            try:
                download_default_dem(preferred_dem_filename)
            except Exception as err:
                warnings.warn(
                    "Automatic DEM download failed for "
                    f"{preferred_dem_filename}: {err}",
                    category=UserWarning,
                    stacklevel=2,
                )
        if (dem_path / preferred_dem_filename).exists():
            return reader_or_store(dem_path / preferred_dem_filename)
        if (dem_path / dem_filename).exists():
            return reader_or_store(dem_path / dem_filename)

        raster_file_list = []
        for ext in raster_extensions:
            raster_file_list.extend(glob.glob("*." + ext, root_dir=dem_path))
        # raster_file_list = [file.name for file in dem_path.glob("*.tif")]
        if sys.stdin.isatty() and len(raster_file_list) > 0:
            print(
                "DEM not found with default filename. "
                "Please select from the following:\n",
                ", ".join(raster_file_list),
                flush=True,
            )
            dem_filename = input("Enter filename:")
        else:
            raise FileNotFoundError(
                f"DEM file {dem_filename} is missing in {dem_path}. "
                "Automatic download was unsuccessful or unavailable."
            )
    return reader_or_store(dem_path / dem_filename)


def interpolate_hypsometrically(
    ds: xr.Dataset,
    main_var: str,
    error: str,
    elev: str = "ref_elev",
    weights: str = "weights",
    outlier_replace: bool = False,
    outlier_limit: float = 2,
    return_coeffs: bool = False,
    fit_sanity_check: dict = None,
    fill_flag: tuple[str, int] = None,
    diagnostic_hook: Callable[[str, dict[str, Any]], Any] | None = None,
) -> xr.Dataset:
    """Fills data gaps by hypsometrical interpolation

    If sufficient data is provided, this routine sorts and bins the data
    by elevation bands and fits a third-order polynomial to the weighted
    averages.

    Sufficient data requires 4 or more bands, with an effective sample
    size of 6 or larger, that span at least 2/3 of the total elevation
    range. The weights used to calculate the weighted average are the
    reciprocal squared errors if no weights are provided.

    If dimension "time" exists, recurse into time steps and interpolate
    per time step.

    Args:
        ds (xr.Dataset): Input with voids. The input has to be along
            dimension "stacked_x_y".
        main_var (str): Name of variable to interpolate. error (str):
            Name of errors. Where interpolated, errors will be filled by
            the scaled RMSE of the fit. The scaling factor will be
            inferred from `error`! Include one of "std", "iqr", "mad",
            "95" in the `error` variable name. If non can be found, it
            is assumed to be the standard deviation ("std"). The error
            data are only used if weights are not provided.
        elev (str, optional): Name of variable that contains the
            reference elevation used for binning. If the variable does
            not exist, it is attempted to read the reference elevations
            from disk. Defaults to "ref_elev".
        weights (str, optional): Provide name of variable that contains
            the weights. The weights will be passed to `numpy.average`
            and should be 1/variance or similar. Defaults to "weights".
        outlier_replace (bool, optional): If enabled, also interpolates
            outliers. Defaults to False.
        outlier_limit (float, optional): Factor of outlier scale (e.g.
            standard deviation). Defaults to 2.
        return_coeffs (bool, optional): If enabled, also returns 3rd
            order polynomial parameters in `numpy.polyfit
            <https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html>`_
            order (highest to lowest). Defaults to False.
        fit_sanity_check (dict, optional): Defaults to None. If None or
            False, it will not be used. If you want to test the
            polynomial gradients, either set to True or pass a `dict`.
            If True, default values will be used; that are 0.1 if used
            with elevation difference to a ref. DEM and 0.05 if used
            with elevation change trends. If you pass a `dict`, the key
            "max_allowed_gradient" will be used. If the gradient is
            steeper than the threshold the model is rejected.
        fill_flag (tuple[str, int], optional): Defaults to None. If provided,
            assigns `fill_flag[1]` to `ds[fill_flag[0]]` where filled.
        diagnostic_hook (Callable, optional): Opt-in hook called with
            diagnostic event names and payloads.
    Returns:
        xr.Dataset: Filled dataset.
    """

    def select_returns(return_coeffs, ds, coeffs):
        if return_coeffs:
            return ds, coeffs
        else:
            return ds

    def design_matrix(x_vals):
        return np.hstack([x_vals, x_vals**2, x_vals**3])

    def invert_3rd_order_coeff_scaling(scaler, coeffs):
        mu, sig = scaler.mean_[0], scaler.scale_[0]
        p0, p1, p2, p3 = coeffs / np.hstack([1, design_matrix(sig)])[::-1]
        return np.array(
            [
                p0,
                p1 - 3 * mu * p0,
                p2 - 2 * mu * p1 + 3 * mu**2 * p0,
                p3 - mu * p2 + mu**2 * p1 - mu**3 * p0,
            ]
        )

    def emit_diagnostic(name, payload):
        if diagnostic_hook is None:
            return
        payload = {"diagnostic_context": {}, **payload}
        diagnostic_hook(name, payload)

    def fit_curve_payload(fit, scaler):
        if diagnostic_hook is None:
            return {}
        fit_x_vals = np.linspace(float(ds[elev].min()), float(ds[elev].max()), 50)[
            :, None
        ]
        return {
            "fit_x_vals": fit_x_vals,
            "fit_y_vals": fit.predict(design_matrix(scaler.transform(fit_x_vals))),
        }

    if "time" in ds.dims and len(ds.time) > 1:
        # note: `groupby("time")` creates time depencies for all data_vars. this
        #       requires taking note of those data_vars that do not depend on
        #       time and reset those after the operation
        no_time_dep = [
            data_var for data_var in ds.data_vars if "time" not in ds[data_var].dims
        ]
        if fit_sanity_check is True:
            # set default sanity check for elevation differences wrt. ref. DEM
            fit_sanity_check = {
                "max_allowed_gradient": 10 / 100
            }  # [10 m elev.diff. per 100 m of elevation]
        ds = ds.groupby("time", squeeze=False).map(
            interpolate_hypsometrically,
            main_var=main_var,
            elev=elev,
            error=error,
            outlier_replace=outlier_replace,
            return_coeffs=return_coeffs,
            fit_sanity_check=fit_sanity_check,
            fill_flag=fill_flag,
            diagnostic_hook=diagnostic_hook,
        )
        for var_name in no_time_dep:
            ds[var_name] = ds[var_name].isel(time=0)
        return select_returns(return_coeffs, ds, np.array([np.nan] * 4))
    else:
        if fit_sanity_check is True:
            # set default sanity check for elevation change rate
            fit_sanity_check = {
                "max_allowed_gradient": 5 / 100
            }  # [10 m/yr elev.trend per 100 m of elevation]

    # this function uses boolean indexing which is not possible with dask
    # arrays. so if ds contains dask arrays, compute them.
    if ds.chunks is not None:
        ds = ds.compute()

    # tbi: currently the data needs to have the single dimension "stacked_x_y".
    #      implement automatic stacking and remove the hard coded dim name.

    # abort if too little data (checking elevation and data validity).
    # necessary to prevent errors but also introduces data gaps
    if ds[elev].where(ds[error] > 0).count() < 24:
        return select_returns(return_coeffs, ds, np.array([np.nan] * 4))
    # also, abort if there isn't anything to do
    if not ds[error].isnull().any() and not outlier_replace:
        return select_returns(return_coeffs, ds, np.array([np.nan] * 4))

    # below might need fill_missing_coords. naively: should not be important
    neighbours = (
        ds[main_var]
        .unstack()
        .sortby("x")
        .sortby("y")
        .rolling(x=5, y=5, min_periods=3, center=True)
    )

    def local_stats(groups):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Degrees of freedom <= 0 for slice.",
                RuntimeWarning,
                r"numpy\.lib\..*nanfunctions.*",
            )
            _cnt = groups.count()
            _mean = (
                groups.mean()
                .where(_cnt >= 6)
                .stack(stacked_x_y=["x", "y"])
                .reindex_like(ds[main_var])
            )
            _std = (
                groups.std(ddof=1)
                .where(_cnt >= 6)
                .stack(stacked_x_y=["x", "y"])
                .reindex_like(ds[main_var])
            )
            _cnt = _cnt.stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
        return _mean, _std, _cnt

    neighbour_mean, neighbour_std, neighbour_count = local_stats(neighbours)
    if outlier_replace:
        neighbour_elev = (
            ds[elev]
            .unstack()
            .sortby("x")
            .sortby("y")
            .rolling(x=5, y=5, min_periods=3, center=True)
        )
        _cnt = neighbour_elev.count()
        neighbour_elev_mean = (
            neighbour_elev.mean()
            .where(_cnt >= 6)
            .stack(stacked_x_y=["x", "y"])
            .reindex_like(ds[main_var])
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Degrees of freedom <= 0 for slice.",
                RuntimeWarning,
                r"numpy\.lib\..*nanfunctions.*",
            )
            neighbour_elev_std = (
                neighbour_elev.std(ddof=1)
                .where(_cnt >= 6)
                .stack(stacked_x_y=["x", "y"])
                .reindex_like(ds[main_var])
            )
        noise = (
            np.abs(ds[main_var] - neighbour_mean) / neighbour_std
            - np.abs(ds[elev] - neighbour_elev_mean) / neighbour_elev_std
        ) > outlier_limit
        emit_diagnostic(
            "hypsometry.outlier_neighbour_check",
            {
                "ds": ds,
                "main_var": main_var,
                "elev": elev,
                "outlier_limit": outlier_limit,
                "neighbour_mean": neighbour_mean,
                "neighbour_std": neighbour_std,
                "neighbour_count": neighbour_count,
                "neighbour_elev_mean": neighbour_elev_mean,
                "neighbour_elev_std": neighbour_elev_std,
                "noise": noise,
            },
        )
        ds[main_var] = xr.where(
            ~np.logical_and(neighbour_count >= 6, noise),
            ds[main_var],
            np.nan,
            keep_attrs=True,
        )
        neighbours = (
            ds[main_var]
            .unstack()
            .sortby("x")
            .sortby("y")
            .rolling(x=5, y=5, min_periods=3, center=True)
        )
        neighbour_mean, neighbour_std, neighbour_count = local_stats(neighbours)
    # # if the reference elevations contain nan values, this leads to errors
    # index_with_nan_in_elev = ds[ds[elev].dims[0]]
    # ds = ds.where(~ds[elev].isnull()).dropna(ds[elev].dims[0])
    # assign weights if not present. use previously assigned weights to
    # prevent using previously filled cells from inform a new average.
    if weights not in ds:
        ds[weights] = 1 / ds[error] ** 2
    group_obj = ds.groupby_bins(
        elev, define_elev_band_edges(ds[elev]), include_lowest=True
    )
    elev_bin_means = pd.Series(index=group_obj.groups)
    elev_bin_errs = pd.Series(index=group_obj.groups)
    fill_mask = -1 * xr.ones_like(ds[main_var])
    for label, group in group_obj:
        if (group[weights] > 0).sum() < 6:
            continue
        vals = (
            group[main_var].fillna(-9999).squeeze()
        )  # make it obvious if anything goes wrong
        w = group[weights].fillna(0).squeeze()
        avg, _var, _ess, to_be_filled_mask = weighted_mean_excl_outliers(
            values=vals, weights=w, deviation_factor=outlier_limit, return_mask=True
        )
        err = student_t.isf(_norm_sf_1, _ess) * (_var / _ess) ** 0.5
        if outlier_replace:
            to_be_filled_mask = np.logical_or(
                group[main_var].isnull().squeeze(), to_be_filled_mask
            )
        else:
            to_be_filled_mask = group[main_var].isnull().squeeze()
        to_be_filled_mask = xr.align(
            fill_mask, to_be_filled_mask, join="left", fill_value=-1
        )[1]
        fill_mask = xr.where(to_be_filled_mask != -1, to_be_filled_mask, fill_mask)
        if np.isnan(avg):
            continue
        elev_bin_means.loc[label] = avg
        elev_bin_errs.loc[label] = err
    elev_bin_means.dropna(inplace=True)
    elev_bin_errs.dropna(inplace=True)
    if elev_bin_means.empty or len(elev_bin_means.index) < 5:
        return select_returns(return_coeffs, ds, np.array([np.nan] * 4))
    # fit polynomial
    try:
        x_vals = np.array([[idx.mid for idx in elev_bin_means.index]]).T
        scaler = preprocessing.StandardScaler().fit(
            x_vals, sample_weight=1 / elev_bin_errs.values
        )
        # cov = covariance.EmpiricalCovariance().fit(design_matrix(scaler.transform(
        #       x_vals
        # ))).covariance_
        fit = linear_model.Ridge(1, solver="svd").fit(
            design_matrix(scaler.transform(x_vals)),
            elev_bin_means.values,
            1 / elev_bin_errs.values,
        )
        coeffs = np.hstack((fit.intercept_, fit.coef_))[::-1]
    except np.linalg.LinAlgError:  # not sure what error sklearn raises
        print(elev_bin_means)
        print(elev_bin_errs)
        return select_returns(return_coeffs, ds, np.array([np.nan] * 4))
    if fit_sanity_check is not None:
        if (
            np.abs(
                np.polyval(
                    np.polyder(coeffs),
                    scaler.transform(
                        np.array(
                            [elev_bin_means.index[0].mid, elev_bin_means.index[-1].mid]
                        )[:, None]
                    ),
                )
            ).max()
            > fit_sanity_check["max_allowed_gradient"] * scaler.var_**0.5
        ):
            # warnings.warn("discarding fit because unrealistic - !note: this is "
            #               "usually not the desired behavior const./linear "
            #               "extrapolation is used instead of the fit which renders "
            #               "this check obsolete!")
            return select_returns(return_coeffs, ds, np.array([np.nan] * 4))

    def scale(x):
        if isinstance(x, xr.DataArray):
            x = x.values
        elif isinstance(x, (int, float, list)):
            x = np.array(x)
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        return scaler.transform(x)

    def const_extrapol(data, pivot):
        return np.polyval(coeffs, scale(pivot).flatten()) + xr.zeros_like(data)

    def linear_extrapol(data, pivot):
        return (
            np.polyval(np.polyder(coeffs), scale(pivot)) * (scale(data) - scale(pivot))
        ).flatten() + const_extrapol(data, pivot)

    extrap_below = ds[elev] < elev_bin_means.index[0].mid
    extrap_above = ds[elev] > elev_bin_means.index[-1].mid
    modelled_list = [
        xr.DataArray(
            fit.predict(design_matrix(scale(ds[elev]))),
            coords={"stacked_x_y": ds.stacked_x_y},
            dims="stacked_x_y",
        )[~np.logical_or(extrap_below, extrap_above)]
    ]
    if extrap_below.any():
        modelled_list.append(
            linear_extrapol(ds[elev][extrap_below], elev_bin_means.index[0].mid)
        )
    if extrap_above.any():
        modelled_list.append(
            linear_extrapol(ds[elev][extrap_above], elev_bin_means.index[-1].mid)
        )
    modelled = (
        xr.concat(modelled_list, "stacked_x_y")
        .reindex_like(ds[main_var])
        .astype(ds[main_var].dtype)
    )
    fit_x_range = elev_bin_means.index[-1].mid - elev_bin_means.index[0].mid
    modelled = xr.where(
        ds[elev] < elev_bin_means.index[0].mid - fit_x_range / 3,
        xr.zeros_like(ds[elev])
        + linear_extrapol(
            xr.zeros_like(ds[elev][:1]) + elev_bin_means.index[0].mid - fit_x_range / 3,
            elev_bin_means.index[0].mid,
        ).values[0],
        modelled,
    )
    modelled = xr.where(
        ds[elev] > elev_bin_means.index[-1].mid + fit_x_range / 3,
        xr.zeros_like(ds[elev])
        + linear_extrapol(
            xr.zeros_like(ds[elev][:1])
            + elev_bin_means.index[-1].mid
            + fit_x_range / 3,
            elev_bin_means.index[-1].mid,
        ).values[0],
        modelled,
    )
    elev_bin_min = (
        elev_bin_means.min() - 2 * elev_bin_errs.iloc[elev_bin_means.argmin()]
    )
    elev_bin_max = (
        elev_bin_means.max() + 2 * elev_bin_errs.iloc[elev_bin_means.argmax()]
    )
    modelled = xr.where(modelled > elev_bin_max, elev_bin_max, modelled)
    modelled = xr.where(modelled < elev_bin_min, elev_bin_min, modelled)
    emit_diagnostic(
        "hypsometry.model_preview",
        {
            "ds": ds,
            "main_var": main_var,
            "elev": elev,
            "modelled": modelled,
            "x_vals": x_vals,
            "elev_bin_means": elev_bin_means,
            "elev_bin_errs": elev_bin_errs,
            "coeffs": coeffs,
            "scaler": scaler,
            **fit_curve_payload(fit, scaler),
        },
    )
    residuals = ds[main_var] - modelled
    local_deviation_metric = (
        np.abs(neighbour_mean - modelled) - outlier_limit * neighbour_std.mean()
    )
    local_deviation = np.logical_and(
        neighbour_count >= 6,
        np.abs(neighbour_mean - modelled) > outlier_limit * neighbour_std.mean(),
    )
    emit_diagnostic(
        "hypsometry.local_deviation",
        {
            "ds": ds,
            "main_var": main_var,
            "elev": elev,
            "modelled": modelled,
            "neighbour_mean": neighbour_mean,
            "neighbour_std": neighbour_std,
            "neighbour_count": neighbour_count,
            "local_deviation": local_deviation,
            "local_deviation_metric": local_deviation_metric,
            "outlier_limit": outlier_limit,
        },
    )
    modelled = xr.where(local_deviation, neighbour_mean, modelled)
    if outlier_replace:
        fill_mask = xr.where(local_deviation, 0, fill_mask)
        # std is used as a deviation measure because the fit relies on
        # normal distributed errors anyway. however, maybe it would be
        # better to use the MAD (and maybe go for some max.likelihood
        # optimizer)
        fill_mask = np.logical_or(
            ds[main_var].isnull(),
            np.logical_and(
                fill_mask != 0,
                np.abs(residuals) > outlier_limit * residuals.std(ddof=4),
            ),
        )
    else:
        fill_mask = ds[main_var].isnull()
    emit_diagnostic(
        "hypsometry.fit_fill_mask",
        {
            "ds": ds,
            "main_var": main_var,
            "elev": elev,
            "fill_mask": fill_mask,
            "modelled": modelled,
            "x_vals": x_vals,
            "elev_bin_means": elev_bin_means,
            "elev_bin_errs": elev_bin_errs,
            "neighbour_std": neighbour_std,
            "residuals": residuals,
            "coeffs": coeffs,
            "scaler": scaler,
            **fit_curve_payload(fit, scaler),
        },
    )
    ds[main_var] = xr.where(~fill_mask, ds[main_var], modelled, keep_attrs=True)
    if fill_flag is not None:
        ds[fill_flag[0]] = xr.where(
            ~fill_mask, ds[fill_flag[0]], fill_flag[1], keep_attrs=True
        )
    RMSE = (residuals.where(~fill_mask) ** 2).mean() ** 0.5
    if "std" in error.lower():
        pass
    elif "iqr" in error.lower():
        RMSE *= 2 * _norm_isf_25
    elif "mad" in error.lower():
        RMSE *= _norm_isf_25
    elif "95" in error.lower():
        RMSE *= _norm_isf_025
    ds[error] = xr.where(~fill_mask, ds[error], RMSE, keep_attrs=True)
    ds[weights] = xr.where(~fill_mask, ds[weights], 0, keep_attrs=True)
    # # restore data gaps
    # ds = ds.reindex_like(index_with_nan_in_elev)
    # tbi: if initially stacked, unstack here
    return select_returns(
        return_coeffs, ds, invert_3rd_order_coeff_scaling(scaler, coeffs)
    )


def load_cs_full_file_names(
    update: Literal["no", "quick", "regular", "full"] = "no",
) -> pd.Series:
    """Loads a pandas.Series of the original CryoSat-2 L1b file names.

    Having the file names available can be handy to organize your local
    data.

    This function can be used to update your local list by setting `update`.

    Args:
        update (str, optional): One of "no", "quick", "regular, or "full".
            "quick" continues from the last locally known file name,
            "regular" checks for changes between the stages OFFL and LTA,
            and "full" replaces the local data base with a new one. Defaults
            to "no".

    Returns:
        pd.Series: Full L1b file names without path or extension.
    """
    file_names_path = aux_path / "CryoSat-2_SARIn_file_names.pkl"
    if os.path.isfile(file_names_path):
        file_names = pd.read_pickle(file_names_path).sort_index()
    else:
        file_names = pd.Series(dtype="object")
    if update == "no":
        return file_names
    if update != "full" and file_names.empty:
        warnings.warn(
            f"No local file-name catalog found at {file_names_path}. "
            "Switching to full update.",
            category=UserWarning,
        )
        update = "full"
    elif update == "quick":
        last_lta_idx = file_names.index[-1]
        print(last_lta_idx + pd.offsets.MonthBegin(-1, normalize=True))
    elif update == "version":
        # implement, also, to actually update the files or remove outdated - or
        # think of something to prevent that old data receives a new name
        raise Exception(
            "Functionality to update L1b file version (e.g. ...E001.nc"
            + "vs ...E003.nc) is not yet implemented."
        )
    if update in ["regular", "version"]:
        # ! "regular" should also be baseline and version aware
        lta_file_names = file_names[file_names.str[3:7] == "LTA_"]
        if lta_file_names.empty:
            last_lta_idx = file_names.index[-1]
        else:
            last_lta_idx = lta_file_names.index[-1]
        print(last_lta_idx + pd.offsets.MonthBegin(-1, normalize=True))

    with ftp_cs2_server() as ftp:
        ftp.cwd("/SIR_SIN_L1")
        year_entries = sorted(
            name
            for name, facts in ftp.mlsd()
            if facts.get("type") == "dir" and re.fullmatch(r"\d{4}", name)
        )
        for year in year_entries:
            if update != "full" and year < str(last_lta_idx.year):
                print("skip", year)
                continue
            month = None
            try:
                ftp.cwd(f"/SIR_SIN_L1/{year}")
                print(f"entered /SIR_SIN_L1/{year}")
                month_entries = sorted(
                    name
                    for name, facts in ftp.mlsd()
                    if facts.get("type") == "dir" and re.fullmatch(r"\d{2}", name)
                )
                for month in month_entries:
                    if update != "full" and pd.to_datetime(
                        f"{year}-{month}"
                    ) < last_lta_idx + pd.offsets.MonthBegin(-1, normalize=True):
                        print("skip", month)
                        continue
                    print(f"cwd /SIR_SIN_L1/{year}/{month}")
                    ftp.cwd(f"/SIR_SIN_L1/{year}/{month}")
                    print(f"scanning /SIR_SIN_L1/{year}/{month}")
                    remote_files = sorted(
                        name
                        for name, facts in ftp.mlsd()
                        if facts.get("type") == "file" and name.endswith(".nc")
                    )
                    for remote_file in remote_files:
                        remote_idx = pd.to_datetime(remote_file[19:34])
                        if (
                            update == "regular"
                            and remote_idx in file_names.index
                            and (
                                file_names.loc[remote_idx][3:7] == "LTA_"
                                or remote_file[3:7] == "OFFL"
                            )
                        ):
                            continue
                        file_names.loc[remote_idx] = remote_file[:-3]
            except Exception:
                if month is None:
                    location = f"/SIR_SIN_L1/{year}"
                else:
                    location = f"/SIR_SIN_L1/{year}/{month}"
                warnings.warn(f"Error occurred in remote directory {location}.")

    file_names.to_pickle(file_names_path)
    print("updated track name list")
    return file_names


def load_cs_ground_tracks(
    region_of_interest: str | shapely.Polygon = None,
    start_datetime: str | pd.Timestamp = "2010",
    end_datetime: str | pd.Timestamp = "2030",
    *,
    buffer_period_by: relativedelta = None,
    buffer_region_by: float = None,
    update: Literal["no", "regular", "full"] = "no",
    n_threads: int = 8,
) -> gpd.GeoDataFrame:
    """Read the GeoDataFrame of CryoSat-2 tracks from disk.

    If desired, you can query certain extents or periods by specifying
    arguments.

    Further, you can update the database by setting `update` to "regular" or
    "full". Mind that this typically takes some time (regular on the order
    of minutes, full rather hours).

    Args:
        region_of_interest (str | shapely.Polygon, optional): Can be any RGI
            code or a polygon in lat/lon (CRS EPSG:4326). If requesting o1
            regions, provide the long code, e.g., "01_alaska". Defaults to None.
        start_datetime (str | pd.Timestamp, optional): Defaults to "2010".
        end_datetime (str | pd.Timestamp, optional): Defaults to "2030".
        buffer_period_by (relativedelta, optional): Extends the period to
            both sides. Handy if you use this function to query tracks for an
            aggregated product. Defaults to None.
        buffer_region_by (float, optional): Handy to also query tracks in the
            proximity that may return elevation estimates for your region of
            interest. Unit are meters here. CryoSat's footprint is +- 7.5 km to
            both sides, anything above 30_000 does not make much sense. Defaults
            to None.
        update (str, optional): If you are interested in the latest tracks,
            update frequently with `update="regular"`. If you believe tracks are
            missing for some reason, choose `update="full"` (be aware this takes
            a while). Defaults to "no".
        n_threads (int, optional): Number of parallel ftp connections. If you
            choose too many, ESA will refuse the connection. Defaults to 8.

    Raises:
        ValueError: For invalid `update` arguments.

    Returns:
        gpd.GeoDataFrame: CryoSat-2 tracks.
    """
    advance_end = isinstance(end_datetime, str) and re.match(
        r"^20[0-9]{2}.?[01][0-9]$", end_datetime
    )
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    if advance_end:
        end_datetime = end_datetime + pd.DateOffset(months=1)
    if os.path.isfile(cs_ground_tracks_path):
        cs_tracks = gpd.read_feather(cs_ground_tracks_path)
        if "index" in cs_tracks.columns:
            cs_tracks.set_index("index", inplace=True)
        cs_tracks.index = pd.to_datetime(cs_tracks.index)
        cs_tracks.sort_index(inplace=True)
    else:
        cs_tracks = gpd.GeoSeries()
        update = "full"
    if update == "full":
        last_idx = pd.Timestamp("2010-07-01")
    # ! should be consistent with load names -> rather call it "quick"?
    elif update == "regular":
        last_idx = pd.to_datetime(cs_tracks.index[-1])
    elif update != "no":
        raise ValueError(
            'Allowed values for `update` are "full". "regular". or "no". '
            + f'You set it to "{update}".'
        )
    if update != "no":
        # the next two function have only a local purpose.
        def save_current_track_list(new_track_series: gpd.GeoSeries):
            """saves the tracklist; backing up the old if older than 5 days."""
            track_path = Path(cs_ground_tracks_path)
            backup_path = Path(extend_filename(cs_ground_tracks_path, "__backup"))
            track_path.parent.mkdir(parents=True, exist_ok=True)
            if track_path.is_file() and (
                not backup_path.is_file()
                or time.time() - track_path.stat().st_mtime > 5 * 24 * 60 * 60
            ):
                print('backing up "old" track file')
                shutil.copyfile(track_path, backup_path)
            print("saving current track list to file")
            new_track_series.to_feather(track_path)

        def collect_missing_tracks(
            remote_files: list[str], present_tracks: gpd.GeoSeries
        ) -> gpd.GeoSeries:
            """Gets track if not in list already.

            Args:
                files (list[str]): HDR file names. All of the same month.
                present_tracks (gpd.GeoSeries): Known tracks.

            Returns:
                gpd.GeoSeries: Missing tracks to be added to the collection.
            """
            with ftp_cs2_server() as ftp:
                ftp.cwd(
                    "/SIR_SIN_L1/"
                    + pd.to_datetime(remote_files[0][19:34]).strftime("%Y/%m")
                )
                tracks_to_be_added = gpd.GeoDataFrame(columns=["geometry"]).rename_axis(
                    "index"
                )
                for rf_name in remote_files:
                    if fnmatch.fnmatch(rf_name, "CS_????_SIR_SIN_1B_*.HDR"):
                        if pd.to_datetime(rf_name[19:34]) in present_tracks.index:
                            continue
                        cache = binary_chache()
                        ftp.retrbinary("RETR " + rf_name, cache.add)
                        et = ET_from_str(cache.cache)
                        root = et.find("Variable_Header/SPH/Product_Location")
                        coordinates = {
                            coord: int(root.find(coord).text) / 1e6
                            for coord in [
                                "Start_Long",
                                "Start_Lat",
                                "Stop_Long",
                                "Stop_Lat",
                            ]
                        }
                        tracks_to_be_added.loc[pd.to_datetime(rf_name[19:34])] = (
                            shapely.LineString(
                                (
                                    [
                                        coordinates["Start_Long"],
                                        coordinates["Start_Lat"],
                                    ],
                                    [coordinates["Stop_Long"], coordinates["Stop_Lat"]],
                                )
                            )
                        )
                        if not all(
                            [(v > -180) and (v < 360) for k, v in coordinates.items()]
                        ):
                            warnings.warn(f"whats with {rf_name} giving {coordinates}?")
                            print("track is:", tracks_to_be_added.loc[rf_name[19:34]])
                    elif rf_name[-3:].lower() != ".nc":
                        warnings.warn(
                            "Encountered unexpected file:"
                            + rf_name
                            + "\n\tShould this appear more often, adapt this function."
                        )
            return tracks_to_be_added

        result_queue = queue.SimpleQueue()
        task_queue = request_workers(collect_missing_tracks, n_threads, result_queue)
        # for each month after last_idx, list all HDR-files and check whether
        # they are in the local collection.
        while True:
            with ftp_cs2_server() as ftp:
                try:
                    ftp.cwd("/SIR_SIN_L1/" + last_idx.strftime("%Y/%m"))
                except ftplib.error_perm:
                    print(
                        "couldn't switch to month(?)",
                        last_idx.strftime("%Y/%m"),
                        "This should only concern you, if you do expect tracks there.",
                    )
                    break
                remote_files = [
                    x[0] for x in ftp.mlsd() if x[0].lower().endswith(".hdr")
                ]
            # cut the file list into chunks and dispatch to workers
            batch_size = len(remote_files) // (n_threads * 3) + 1
            while remote_files:
                try:
                    task_queue.put((remote_files[:batch_size], cs_tracks))
                    remote_files[:batch_size] = []
                except IndexError:
                    task_queue.put((remote_files[:], cs_tracks))
                    remote_files[:] = []
            # wait for and collect new tracks
            new_tracks_collection = []
            while not task_queue.empty() or not result_queue.empty():
                try:
                    tmp = result_queue.get(block=True, timeout=10 * 60)
                    if not tmp.empty:
                        new_tracks_collection.append(tmp)
                except queue.Empty:
                    print("waiting for task queue")
                    time.sleep(10)
            # append to local collection and save the result, if any
            if new_tracks_collection:
                cs_tracks = pd.concat(
                    [cs_tracks, pd.concat(new_tracks_collection)]
                ).sort_index()
                duplicate = cs_tracks.index.duplicated(keep="last")
                if duplicate.sum() > 0:
                    warnings.warn(f"{duplicate.sum()} duplicates found; dropping them.")
                    cs_tracks = cs_tracks[~duplicate]
                    cs_tracks.sort_index(inplace=True)
                save_current_track_list(cs_tracks)
            print("scanned all files in", last_idx.strftime("%Y/%m"))
            last_idx = last_idx + pd.DateOffset(months=1)
            print("switching to", last_idx.strftime("%Y/%m"))

    # the local collection has been updated. now, return the tracks
    if buffer_period_by is not None:
        start_datetime = start_datetime - buffer_period_by
        end_datetime = end_datetime + buffer_period_by
    cs_tracks = cs_tracks.loc[start_datetime:end_datetime]
    if region_of_interest is not None:
        if isinstance(region_of_interest, str):
            # union=False neccessary for Greenland and large regions
            region_of_interest = load_glacier_outlines(
                region_of_interest, "glaciers", union=False
            ).geometry.values
        if buffer_region_by is not None:
            region_of_interest = gis.buffer_4326_shp(
                region_of_interest, buffer_region_by
            )
        else:
            region_of_interest = gis.simplify_4326_shp(
                shapely.ops.unary_union(region_of_interest)
            )
        # find all tracks that intersect the buffered region of interest.
        # mind that this are calculations on a sphere. currently, the
        # polygon is transformed to ellipsoidal coordinates. not a 100 %
        # sure that this doesn't raise issues close to the poles.
        cs_tracks = cs_tracks[cs_tracks.intersects(region_of_interest)]
    return cs_tracks.set_crs(4326)


def _normalize_rgi_product(product: str) -> str:
    """Normalize RGI product aliases to `C` or `G`."""
    product_lower = product.lower()
    product_upper = product.upper()
    if product_upper == "C" or product_lower == "complexes":
        return "C"
    if product_upper == "G" or product_lower in {"glaciers", "basins"}:
        return "G"
    raise ValueError(
        f'Argument product should be either "glaciers" or "complexes", not {product!r}.'
    )


def _normalize_rgi_o1code(o1code: str | int) -> str:
    """Normalize an o1 region code to zero-padded two-digit form."""
    o1code = str(o1code).strip()
    if o1code.isdigit() and len(o1code) <= 2:
        return f"{int(o1code):02d}"
    match = re.match(r"^([0-9]{2})", o1code)
    if match is None:
        raise ValueError(f'o1code should start with "01".."20", not {o1code!r}.')
    return match.group(1)


def _rgi_remote_product_url(product: str) -> str:
    """Return RGI product directory URL for product code `C` or `G`."""
    product = _normalize_rgi_product(product)
    return f"{_RGI_DOWNLOAD_BASE_URL}/RGI2000-v7.0-{product}/"


def _rgi_product_cli_name(product: str) -> str:
    """Return CLI product name for normalized RGI product code."""
    product = _normalize_rgi_product(product)
    return "complexes" if product == "C" else "glaciers"


def _rgi_region_pattern(product: str, o1code: str | int) -> str:
    """Return abbreviated RGI o1 region pattern for messages."""
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)
    return f"RGI2000-v7.0-{product}-{o1code}_..."


def _rgi_auto_download_message(product: str, o1code: str | int) -> str:
    """Return warning message for the automatic RGI download attempt."""
    return (
        f"RGI region {_rgi_region_pattern(product, o1code)} is missing; "
        "attempting automatic NSIDC download with NASA Earthdata credentials."
    )


def _rgi_missing_message(product: str, o1code: str | int) -> str:
    """Return final user-facing message for missing RGI o1 product."""
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)
    cli_product = _rgi_product_cli_name(product)
    return (
        f"RGI region {_rgi_region_pattern(product, o1code)} could not be found "
        "or downloaded. See docs/prerequisites.rst; try "
        f"`cryoswath download-rgi --o1 {o1code} --product {cli_product}`; "
        f"source: {_rgi_remote_product_url(product)}."
    )


def _rgi_o1_archive_stem(o1code: str | int, product: str) -> str:
    """Build deterministic archive stem from o1 metadata table."""
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)
    long_code = rgi_code_translator(o1code, out_type="long_code")
    return f"RGI2000-v7.0-{product}-{long_code}"


def _find_rgi_o1region_source(o1code: str | int, product: str) -> Path | None:
    """Return local path for an o1 region product, if available."""
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)

    try:
        candidates = sorted(Path(rgi_path).iterdir())
    except FileNotFoundError:
        return None

    pattern = re.compile(rf"RGI2000-v7\.0-{product}-{o1code}_.*")
    for path in candidates:
        if pattern.match(path.name) and (
            path.is_dir() or path.suffix in {".shp", ".feather"}
        ):
            return path
    return None


def _read_rgi_o1region_source(file_path: str | Path) -> gpd.GeoDataFrame:
    """Load one supported RGI o1 source file."""
    file_path = Path(file_path)
    if file_path.suffix == ".feather":
        return gpd.read_feather(file_path)
    if file_path.suffix == ".shp" or file_path.is_dir():
        return gpd.read_file(file_path)
    raise ValueError(f"Unsupported RGI source format: {file_path}")


def download_rgi_o1region(
    o1code: str | int,
    product: str = "complexes",
    force: bool = False,
    timeout: int | float = 120,
) -> str:
    """Download and extract one RGI o1 region product."""
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)

    if not force:
        existing_source = _find_rgi_o1region_source(o1code, product)
        if existing_source is not None:
            return str(existing_source)

    archive_stem = _rgi_o1_archive_stem(o1code, product)
    remote_url = f"{_rgi_remote_product_url(product)}{archive_stem}.zip"
    rgi_dir = Path(rgi_path)
    rgi_dir.mkdir(parents=True, exist_ok=True)
    archive_path = rgi_dir / f"{archive_stem}.zip"
    target_dir = rgi_dir / archive_stem

    _download_earthdata_file(
        url=remote_url,
        dest=archive_path,
        timeout=timeout,
    )

    try:
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                "Downloaded RGI payload is not a zip archive. This usually means "
                "NASA Earthdata returned a login or error page; check credentials "
                f"and source URL: {remote_url}"
            )

        extract_root = Path(tempfile.mkdtemp(prefix=f".{archive_stem}.", dir=rgi_dir))
        try:
            shutil.unpack_archive(archive_path, extract_root, format="zip")
            nested_dir = extract_root / archive_stem
            if target_dir.exists():
                if target_dir.is_dir():
                    shutil.rmtree(target_dir)
                else:
                    target_dir.unlink()
            if nested_dir.is_dir() and len(list(extract_root.iterdir())) == 1:
                shutil.move(str(nested_dir), str(target_dir))
            else:
                shutil.move(str(extract_root), str(target_dir))
                extract_root = None
        finally:
            if extract_root is not None and extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    archive_path.unlink(missing_ok=True)
    return str(target_dir)


def _load_o1region(
    o1code: str,
    product: str = "complexes",
    area_threshold: float = 1,  # in km²
) -> gpd.GeoDataFrame:
    """Loads RGI v7 basin or complex outlines and meta data

    Use :py:func:`load_glacier_outlines` instead.

    Args:
        o1code (str): starting with "01".."20"
        product (str, optional): Either "glaciers" or "complexes".
            Defaults to "complexes".
        area_threshold (float, optional): Glaciers smaller than the
            threshold will not be returned. Defaults to 1 km².

    Raises:
        ValueError: If o1code can't be recognized.
        FileNotFoundError: If RGI data is missing.

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
    """
    product = _normalize_rgi_product(product)
    o1code = _normalize_rgi_o1code(o1code)
    source = _find_rgi_o1region_source(o1code, product)
    if source is None:
        warnings.warn(
            _rgi_auto_download_message(product, o1code),
            category=UserWarning,
            stacklevel=2,
        )
        try:
            download_rgi_o1region(o1code=o1code, product=product)
        except Exception as err:
            missing_message = _rgi_missing_message(product, o1code)
            warnings.warn(
                f"Automatic RGI download failed: {err}. {missing_message}",
                category=UserWarning,
                stacklevel=2,
            )
            raise FileNotFoundError(missing_message) from err
        source = _find_rgi_o1region_source(o1code, product)
        if source is None:
            missing_message = _rgi_missing_message(product, o1code)
            warnings.warn(missing_message, category=UserWarning, stacklevel=2)
            raise FileNotFoundError(missing_message)
    return _read_rgi_o1region_source(source)


def _load_o2region(o2code: str, product: str = "complexes") -> gpd.GeoDataFrame:
    """Loads RGI v7 basin or complex outlines and meta data

    Use :py:func:`load_glacier_outlines` instead.

    Args:
        o2code (str): RGI o2 code.
        product (str, optional): Either "glaciers" or "complexes". Defaults
            to "complexes".

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
    """
    o1region = _load_o1region(o2code[:2], product)
    # special handling for greenland periphery
    if o2code.startswith("05") and not o2code.endswith("01"):
        lut = {
            "11": "North",
            "12": "West",
            "13": "Southwest",
            "14": "Southeast",
            "15": "East",
        }
        if o2code[-2:] in lut:
            subregion = lut[o2code[-2:]]
        else:
            subregion = re.split("[^A-Za-z]", o2code)[-1].capitalize()
        if product in ["basins", "glaciers"]:
            product = "glacier"
        elif product == "complexes":
            product = "complex"
        with open(
            Path(
                rgi_path,
                f"RGI_{product}_ID_list__Greenland_Periphery__{subregion}.txt",
            ),
            "r",
        ) as f:
            rgi_ids = f.read().splitlines()
        return _load_basins(rgi_ids)
    return o1region[o1region["o2region"] == o2code[:5]]


def _load_basins(rgi_ids: list[str]) -> gpd.GeoDataFrame:
    """Loads RGI v7 basin ~or complex~ outlines and meta data

    Use :py:func:`load_glacier_outlines` instead.

    Args:
        rgi_ids (list[str]): RGI basin ids, all within the same RGI o1 region.

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
    """
    if len(rgi_ids) > 1:
        assert all([id[:17] == rgi_ids[0][:17]] for id in rgi_ids)
    product_code, o1_code = rgi_ids[0].split("-")[2:4]
    rgi_o1_gpdf = _load_o1region(
        o1_code,
        product="glaciers" if product_code == "G" else "complexes",
        area_threshold=0,
    )
    id_to_index_series = pd.Series(data=rgi_o1_gpdf.index, index=rgi_o1_gpdf.rgi_id)
    return rgi_o1_gpdf.loc[id_to_index_series.loc[rgi_ids].values]


def load_glacier_outlines(
    identifier: str | list[str],
    product: str = "complexes",
    union: bool = True,
    crs: int | CRS = None,
    area_threshold: float = None,  # in km²
) -> shapely.MultiPolygon:
    """Loads RGI v7 basin or complex outlines and meta data

    Args:
        identifier (str | list[str]): RGI id: either o1, o2, or
            basin/complex id.
        product (str, optional): Either "glaciers" or "complexes". Defaults
            to "complexes".
        union (bool, optional): For backward compatibility, if enabled (by
            default) only return union of all shapes. If disabled, return
            full GeoDataFrame. Defaults to True.
        crs (int | CRS, optional): Convenience option to reproject shape(s)
            to crs. Defaults to None.
        area_threshold (float, optional): Glaciers smaller than the
            threshold will not be returned. Defaults to 1 km² when used
            with complexes and to 0 otherwise.

    Raises:
        ValueError: If identifier was not understood.

    Returns:
        shapely.MultiPolygon: Union of basin shapes. If `union` is disabled,
        instead return geopandas.GeoDataFrame including the full data.
    """
    if isinstance(identifier, list):
        out = _load_basins(identifier)
    elif (
        len(identifier) == (7 + 4 + 1 + 2 + 5 + 4)
        and identifier.split("-")[:2]
        == [
            "RGI2000",
            "v4.1",
        ]
        and identifier.split("-")[2] in ["C", "G"]
    ):
        out = _load_basins([identifier])
    # the pattern is rather allowing, set it to
    # "^(-?[012][0-9]){2}(_[a-z]+){1,5}(_[0-9][a-z][0-9]?)?$" to make it tight
    elif len(identifier) >= 5 and re.match("^(-?[0-3][0-9]){2}$", identifier[:5]):
        out = _load_o2region(identifier[:5], product=product)
    elif re.match("[012][0-9](_[a-z]+)?", identifier):
        out = _load_o1region(identifier[:2], product=product)
    else:
        raise ValueError(
            f'Provided o1, o2, or RGI identifiers. "{identifier}" not understood.'
        )
    if product == "complexes":
        if area_threshold is None:
            # ! work-around: drop small glaciers
            # issue: takes long to do computations or kernel crashes if
            # (assumption) region contains too many small glaciers. this is
            # equally true for o2 regions, which is why I drop them here
            # already. observed for the Alps.
            area_threshold = 1
    if area_threshold is not None:
        out = drop_small_glaciers(out, area_threshold)
    if crs is not None:
        out = out.to_crs(crs)
    if union:  # former default
        try:
            out = out.make_valid().union_all(method="coverage")
        except:  # TODO specify exception # noqa: E722
            out = out.make_valid().union_all(method="unary")
    return out


def merge_l2_cache(
    source_glob: str,
    destination_file_name: str,
    exclude_endswith: list[str] = ["backup", "collection"],
) -> None:
    """Append cached l2 data from various hdf files into one.

    Tests whether data is present in destination; if not, copies the data.

    This function is very specifically for cached l2 data as created by
    `l3.build_dataset`.

    Args:
        source_glob (str): Unix-like glob pattern to match source files
            in `misc.tmp_path` (default: data/tmp/).
        destination_file_name (str): ... in `misc.tmp_path`.
        exclude_endswith (list[str], optional): Do not include files with
            the specified ending. Useful to exclude backups. Defaults to
            ["backup", "collection"].
    """
    # this snippet turned out useful: one can split the caching process,
    # e.g., into years and combine the cache files using this function
    # afterward.
    # not tested after migrating here from notebook
    destination_path = Path(tmp_path) / destination_file_name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination_path, "a") as h5_dest:
        for source_path in sorted(glob.glob(os.path.join(tmp_path, source_glob))):
            print("\n", source_path)
            if any([source_path.endswith(ending) for ending in exclude_endswith]):
                continue
            with h5py.File(source_path, "r") as h5_src:

                def collect_groups(name, node):
                    if name.split("/")[-1].startswith("t_"):
                        if name not in h5_dest:
                            print(name, "will be copied ...")
                            h5_src.copy(h5_src["/" + name], h5_dest, "/" + name)
                        else:
                            print(name, "exists in collection")
                    else:
                        pass
                        # print(name, "is not an end node")

                h5_src.visititems(collect_groups)


def patch_gatekeeper(module_version: str, rules: list[dict]):
    """Checks whether a patch should be applied

    Use with a list of dict like

    [{  "version":      "2.3",
        "comperator":   operator.lt,
        "action":       "skip"},
     {  "version":      "3",
        "comperator":   operator.ge,
        "action":       "warn" }]

    Args:
        module_version (str): current version of the patched module
        rules (dict): Requires keys "comparator", "version", and
            "action".

    Returns:
        str: rules["action"] if condition is met, else None
    """
    for rule in rules:
        if rule["comperator"](Version(module_version), Version(rule["version"])):
            return rule["action"]


@contextmanager
def monkeypatch(dictlist: list[dict]):
    """contructs a patched context

    Patching the backend of foreign funktions quickly leads to
    inconsitencies. Using the patch only within a chosen context limits
    side effects.

    Optionally, have :func:`patch_gatekeeper` manage for which version
    to apply the patch, to warn about compatibility issues, or to raise
    an error.

    Use like:

    .. code-block:: python

        patchdicts = [{
            "module":       mod1,
            "target":       "obj1",
            "replacement":  patch1,
            "version":      base_mod1.__version__,  # optional
            "rules":        rules1  # optional
        },
        {   "module":       mod2,
            "target":       "obj2",
            "replacement":  patch2
        }]

        with monkeypatch(patchdicts):
            <your code>

    Args:
        dictlist (list[dict]): Requires keys "module", "target", and
            "replacement".
    """
    for d in dictlist:
        if "rules" in d:
            verdict = patch_gatekeeper(d["version"], d["rules"])
            if verdict == "skip":
                continue
            elif verdict == "raise":
                raise
            elif verdict == "warn":
                warnings.warn(
                    f"Patch not meant for {d['module']} version {d['version']}."
                )
        d.update({"original": getattr(d["module"], d["target"])})
        setattr(d["module"], d["target"], d["replacement"])
    try:
        yield
    finally:
        for d in dictlist:
            if "original" in d:
                setattr(d["module"], d["target"], d["original"])


def patched_xr_decode_scaling(
    data, scale_factor, add_offset, dtype: np.typing.DTypeLike
):
    """Compatibility patch for xarray scale/offset decoding."""
    data = data.astype(dtype=dtype, copy=True)
    if scale_factor is not None:
        data = data * scale_factor
    if add_offset is not None:
        data += add_offset
    return data


def patched_xr_decode_tDel(num_timedeltas, units: str, time_unit="ns") -> np.ndarray:
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64 ["s", "ms", "us", "ns"] array.
    """
    from xarray.coding.times import (
        _check_timedelta_range,
        _netcdf_to_numpy_timeunit,
        _numbers_to_timedelta,
        ravel,
        reshape,
    )

    num_timedeltas = np.asarray(num_timedeltas)
    unit = _netcdf_to_numpy_timeunit(units)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        _check_timedelta_range(np.nanmin(num_timedeltas), unit, time_unit)
        _check_timedelta_range(np.nanmax(num_timedeltas), unit, time_unit)

    timedeltas = _numbers_to_timedelta(num_timedeltas, unit, "ns", "timedelta")
    pd_timedeltas = pd.to_timedelta(ravel(timedeltas), unit="ns")

    if np.isnat(timedeltas).all():
        empirical_unit = time_unit
    else:
        empirical_unit = pd_timedeltas.unit

    if np.timedelta64(1, time_unit) > np.timedelta64(1, empirical_unit):
        time_unit = empirical_unit

    if time_unit not in {"s", "ms", "us", "ns"}:
        raise ValueError(
            f"time_unit must be one of 's', 'ms', 'us', or 'ns'. Got: {time_unit}"
        )

    result = pd_timedeltas.as_unit(time_unit).to_numpy()
    return reshape(result, num_timedeltas.shape)


def nan_unique(data: np.typing.ArrayLike) -> list:
    """Returns unique values that are not nan.

    Args:
        data (np.typing.ArrayLike): Input data.

    Returns:
        list: List of unique values.
    """
    return [element for element in np.unique(data) if not np.isnan(element)]


def request_workers(
    task_func: callable, n_workers: int, result_queue: queue.Queue = None
) -> queue.Queue:
    """Creates workers and provides queue to assign work

    Args:
        task_func (callable): Task.
        n_workers (int): Number of requested workers.
        result_queue (queue.Queue, optional): Queue in which to drop
            results. Defaults to None.

    Returns:
        queue.Queue: Task queue.
    """
    task_queue = queue.Queue()
    task_queue.worker_errors = []

    def worker():
        while True:
            next_task = task_queue.get()
            try:
                if next_task is None:
                    return
                result = task_func(*next_task)
                if result_queue is not None:
                    result_queue.put(result)
            except BaseException as err:
                task_queue.worker_errors.append((err, traceback.format_exc()))
                warnings.warn(
                    "Worker task failed; continuing with remaining queued work. "
                    f"Original error: {err!r}",
                    category=UserWarning,
                )
            finally:
                task_queue.task_done()

    for i in range(n_workers):
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
    return task_queue


def repair_l2_cache(
    filepath: str,
    *,
    region_of_interest: shapely.MultiPolygon = None,
    force: bool = False,
) -> None:
    """Attempts to repair corrupted l2 cache files.

    The caching logic is not 100% safe. To repair a cache, this function
    removes duplicates and sorts the data index.
    If the note names for some reason

    Args:
        filepath (str): Path to l2 cache file.
        region_of_interest (shapely.Geometry, optional): EPSG:4326
            outline of considered region. If provided, removes chunks
            with no points inside projected bounding box of outline.
        force (bool): Disregard file size safety, e.g., if you
            expect less than 2/3 of the data to remain.
    """
    if region_of_interest is not None:
        crs = gis.find_planar_crs(shp=region_of_interest)
        bbox = shapely.box(
            *gpd.GeoSeries(region_of_interest, crs=4326).to_crs(crs).bounds.values[0]
        )

    def move_node(name, node):
        if isinstance(node, h5py.Dataset):
            pass
        elif "_i_table" in node:
            tmp = pd.read_hdf(tmp_h5, key=node.name)
            if "bbox" not in locals() or any(
                shapely.within(shapely.points(tmp.x, tmp.y), bbox)
            ):
                tmp.drop_duplicates(keep="first").sort_index().to_hdf(
                    filepath, key=node.name, format="table"
                )

    tmp_h5 = os.path.join(tmp_path, "tmp")
    Path(tmp_h5).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(tmp_h5):
        if os.path.isfile(tmp_h5):
            os.remove(tmp_h5)
        elif os.path.isdir(tmp_h5):
            shutil.rmtree(tmp_h5)
        else:
            raise Exception(f"Can't remove {tmp_h5}; neither file nor directory!?")
    # I expect `shutil.move` to be safe and believe: either it succeeds
    # or nothing happens
    shutil.move(filepath, tmp_h5)
    try:
        print(
            "Starting to repair file. This may take several minutes. You can",
            f"monitor the progress by viewing the file sizes of {filepath} and",
            f"{tmp_h5}. They will be similar at the end of the process. It should",
            "be reasonably safe to abort.",
        )
        # below hides warnings about a minus sign in node names. this
        # can safely be ignored.
        warnings.filterwarnings("ignore", category=NaturalNameWarning)
        with h5py.File(tmp_h5, "r") as h5:
            h5.visititems(move_node)
        warnings.filterwarnings("default", category=NaturalNameWarning)
        try:
            clean_data_fraction = os.path.getsize(filepath) / os.path.getsize(tmp_h5)
        except FileNotFoundError:
            print(
                "No data remain - this will show as FileNotFoundError. The initial",
                "file will be restored. Delete it, if you're sure about it.",
            )
            raise
        if not force and clean_data_fraction < 0.67:
            raise Exception(
                f"Only {clean_data_fraction:%} of the original file size remain. If "
                "this seems plausible to you, rerun setting `force=True`."
            )
    except Exception:
        print("Restoring original (potentially corrupt) file because error occurred.")
        if os.path.isfile(filepath):
            os.remove(filepath)
        shutil.move(tmp_h5, filepath)
        print("Successfully restored initial state. Reraising error:")
        raise
    else:
        print("Reperation was successful: removed duplicates and sorted index.")
    finally:
        if os.path.isfile(tmp_h5):
            os.remove(tmp_h5)


def rgi_code_translator(
    input: str | int | tuple[int, int] | list,
    out_type: str = "full_name",
) -> Union[str, list[str]]:
    """Translate o1 or o2 codes to region names

    Args:
        input (str): RGI o1 or o2 codes.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Raises:
        ValueError: If input is not understood.

    Returns:
        str | list[str]: Either full name or RGI "long_code".
    """
    if isinstance(input, list):
        return [rgi_code_translator(element, out_type) for element in input]
    elif isinstance(input, int) or (
        isinstance(input, str) and len(input) <= 2 and int(input) <= 20
    ):
        return rgi_o1region_translator(int(input), out_type)
    elif (
        isinstance(input, tuple)
        and len(input) == 2
        and all([isinstance(x, int) for x in input])
    ):
        return rgi_o2region_translator(*input, out_type=out_type)
    elif isinstance(input, str) and re.match(r"\d\d-\d\d", input):
        return rgi_o2region_translator(
            *[int(x) for x in input.split("-")], out_type=out_type
        )
    raise ValueError(f"Input {input} not understood. Pass RGI o1- or o2region codes.")


def rgi_o1region_translator(input: int, out_type: str = "full_name") -> str:
    """Finds region name for given RGI o1 number.

    Args:
        input (int): RGI o1 number.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Returns:
        str: Either full name or RGI "long_code".
    """
    if isinstance(input, list):
        return [rgi_o1region_translator(element, out_type) for element in input]
    lut = pd.read_feather(
        os.path.join(rgi_path, "RGI2000-v7.0-o1regions.feather"),
        columns=["o1region", "full_name", "long_code"],
    ).set_index("o1region")
    return lut.loc[f"{input:02d}", out_type]


def rgi_o2region_translator(o1: int, o2: int, out_type: str = "full_name") -> str:
    """Finds subregion name for given RGI o1 and o2 number.

    Args:
        o1 (int): RGI o1 number.
        o2 (int): RGI o2 number.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Returns:
        str: Either full name or RGI "long_code".
    """
    if isinstance(o1, list):
        return [rgi_o2region_translator(o1_, o2_, out_type) for o1_, o2_ in zip(o1, o2)]
    if isinstance(o2, list):
        return [rgi_o2region_translator(o1, o2_, out_type) for o2_ in o2]
    if o1 == 5 and o2 in range(11, 16):
        if out_type != "full_name":
            raise NotImplementedError()
        return dict(
            [
                (11, "North Greenland"),
                (12, "West Greenland"),
                (13, "South West Greenland"),
                (14, "South East Greenland"),
                (15, "East Greenland"),
            ]
        )[o2]
    lut = pd.read_feather(
        os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"),
        columns=["o2region", "full_name", "long_code"],
    ).set_index("o2region")
    return lut.loc[f"{o1:02d}-{o2:02d}", out_type]


@contextmanager
def sandbox_write_to(target: str):
    """Guard writes with a lock and recover from stale backup sidecars."""
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    backup = target + "__backup"
    lock = target + "__lock"
    lock_fd = None

    try:
        # Claim this target atomically to avoid concurrent writers.
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
    except FileExistsError:
        raise Exception(
            f"Write lock exists unexpectedly at {lock}. This may point to a running "
            "process. If this is a relict, remove it manually."
        )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)

    try:
        # ! other functions depend on the "__backup" extension
        if os.path.isfile(backup):
            if os.path.isfile(target):
                warnings.warn(
                    f"Removing stale backup file at {backup} before writing.",
                    RuntimeWarning,
                )
                os.remove(backup)
            else:
                warnings.warn(
                    f"Recovering missing target from backup file at {backup}.",
                    RuntimeWarning,
                )
                shutil.move(backup, target)
        yield target
    finally:
        if os.path.isfile(backup):
            os.remove(backup)
        if os.path.isfile(lock):
            os.remove(lock)


def sel_chunk_idx_range(ds, dim, start, stop):
    """Select a range of dask chunks by index along one dimension."""
    chunk_borders = np.cumsum([0] + list(ds.chunks[dim]))
    return ds.isel({dim: slice(*chunk_borders[[start, stop + 1]])})


def sel_chunk_range(ds, **dim_intervals):
    """Select chunk range by coordinate intervals per dimension."""
    for dim, interval in dim_intervals.items():
        ds = sel_chunk_idx_range(ds, dim, *chunk_idx(ds, dim, interval))
    return ds


def update_keyring(
    user: str = None,
    password: str = None,
    *,
    service: str = _ESA_AUTH_IDP_HOST,
    username_key: str = _ESA_KEYRING_DEFAULT_USER_KEY,
) -> str:
    """Create or update keyring credentials for ESA data access."""
    if keyring is None:
        raise RuntimeError(
            "The keyring package is not installed. Install `keyring` or use "
            "~/.netrc (plaintext fallback)."
        )
    env_user = os.environ.get(_ESA_ENV_USER)
    env_password = os.environ.get(_ESA_ENV_PASSWORD)
    user = user or env_user
    password = password or env_password
    if user is None:
        user = input("Enter ESA username: ").strip()
    if password is None:
        password = getpass.getpass("Enter ESA password: ")
    if not user or not password:
        raise ValueError("Both ESA user and password are required.")
    try:
        keyring.set_password(service, user, password)
        keyring.set_password(service, username_key, user)
        verify_user = keyring.get_password(service, username_key)
        verify_password = keyring.get_password(service, user)
    except KeyringError as err:
        raise RuntimeError(
            f"Could not store ESA credentials in keyring: {err}"
        ) from err
    if verify_user != user or verify_password != password:
        raise RuntimeError(
            "Could not verify keyring credentials after writing. "
            "Your keyring backend may be locked or unsupported."
        )
    return user


def update_keyring_cli() -> None:
    """Compatibility CLI wrapper around :func:`update_keyring`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath-update-keyring",
        description="Create or update keyring credentials for ESA access.",
    )
    _add_update_keyring_arguments(parser)
    args = parser.parse_args()
    _update_keyring_from_args(args)


def update_netrc(
    user: str = None,
    password: str = None,
    *,
    machine: str = _ESA_CS2_HOST,
    netrc_file: str | Path = None,
) -> str:
    """Create or update a plaintext ``.netrc`` entry for ESA credentials.

    Missing ``user`` or ``password`` values are read from
    ``EOIAM_USER`` and ``EOIAM_PASSWORD`` first, then prompted interactively.

    Args:
        user (str, optional): ESA username.
        password (str, optional): ESA password.
        machine (str, optional): Netrc machine host key.
        netrc_file (str | Path, optional): Override for target file path.

    Returns:
        str: Absolute path to the written netrc file.
    """
    user = user or os.environ.get(_ESA_ENV_USER)
    password = password or os.environ.get(_ESA_ENV_PASSWORD)
    if user is None:
        user = input("Enter ESA username: ").strip()
    if password is None:
        password = getpass.getpass("Enter ESA password: ")
    if not user or not password:
        raise ValueError("Both ESA user and password are required.")

    netrc_path = (
        Path(netrc_file).expanduser().resolve()
        if netrc_file is not None
        else (Path.home() / ".netrc").resolve()
    )
    netrc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = netrc_path.read_text() if netrc_path.exists() else ""
    entry = f"machine {machine}\n  login {user}\n  password {password}\n"
    entry_pattern = re.compile(
        rf"(?ms)^machine\s+{re.escape(machine)}\b.*?(?=^machine\s+\S+|\Z)"
    )
    if entry_pattern.search(existing):
        updated = entry_pattern.sub(entry, existing, count=1)
    else:
        updated = entry if not existing.strip() else f"{existing.rstrip()}\n\n{entry}"

    netrc_path.write_text(updated)
    try:
        netrc_path.chmod(0o600)
    except OSError as err:
        warnings.warn(
            f"Could not set permissions 600 on {netrc_path}: {err}",
            category=UserWarning,
            stacklevel=2,
        )
    return str(netrc_path)


def update_netrc_cli() -> None:
    """Compatibility CLI wrapper around :func:`update_netrc`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath-update-netrc",
        description=(
            "Create or update ~/.netrc credentials for ESA access (plaintext fallback)."
        ),
    )
    _add_update_netrc_arguments(parser)
    args = parser.parse_args()
    _update_netrc_from_args(args)


def download_rgi_cli() -> None:
    """Compatibility CLI wrapper around :func:`download_rgi_o1region`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath-download-rgi",
        description=(
            "Download and extract one RGI o1 region file bundle to data/auxiliary/RGI."
        ),
    )
    _add_download_rgi_arguments(parser)
    args = parser.parse_args()
    _download_rgi_from_args(args)


def _add_download_auxiliary_data_arguments(parser) -> None:
    """Add shared auxiliary-data download arguments to an argparse parser."""
    parser.add_argument(
        "--base-dir",
        default=".",
        help=(
            "Project base directory used for config discovery "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files contained in the auxiliary-data archive.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="HTTP timeout in seconds (default: 120).",
    )


def _download_auxiliary_data_from_args(args) -> None:
    """Download auxiliary data from parsed CLI arguments."""
    out_path = download_auxiliary_data(
        base_dir=args.base_dir,
        force=args.force,
        timeout=args.timeout,
    )
    print(out_path)


def download_auxiliary_data_cli() -> None:
    """CLI wrapper around :func:`download_auxiliary_data`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath download-aux-data",
        description="Download the CryoSwath auxiliary-data snapshot from Zenodo.",
    )
    _add_download_auxiliary_data_arguments(parser)
    args = parser.parse_args()
    _download_auxiliary_data_from_args(args)


def _add_get_tutorials_arguments(parser) -> None:
    """Add shared tutorial-copy arguments to an argparse parser."""
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project base directory (default: current directory).",
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Directory for tutorials (default: <base-dir>/tutorials).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tutorial notebooks.",
    )


def _get_tutorials_from_args(args) -> None:
    """Copy packaged tutorials from parsed CLI arguments."""
    out_path = copy_tutorials(
        destination=args.destination,
        base_dir=args.base_dir,
        force=args.force,
    )
    print(out_path)


def get_tutorials_cli() -> None:
    """CLI wrapper around :func:`copy_tutorials`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath get-tutorials",
        description="Copy packaged CryoSwath tutorial notebooks into a project.",
    )
    _add_get_tutorials_arguments(parser)
    args = parser.parse_args()
    _get_tutorials_from_args(args)


def _add_download_rgi_arguments(parser) -> None:
    """Add shared RGI download arguments to an argparse parser."""
    parser.add_argument(
        "--o1",
        required=True,
        help='RGI o1 region code (e.g. "09").',
    )
    parser.add_argument(
        "--product",
        default="complexes",
        choices=["complexes", "glaciers", "C", "G"],
        help="RGI product type (default: complexes).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and replace extracted directory even if local match exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="HTTP timeout in seconds (default: 120).",
    )


def _download_rgi_from_args(args) -> None:
    """Download RGI data from parsed CLI arguments."""
    out_path = download_rgi_o1region(
        o1code=args.o1,
        product=args.product,
        force=args.force,
        timeout=args.timeout,
    )
    print(out_path)


def _add_update_keyring_arguments(parser) -> None:
    """Add shared keyring arguments to an argparse parser."""
    parser.add_argument("--user", default=None, help="ESA username.")
    parser.add_argument("--password", default=None, help="ESA password.")
    parser.add_argument(
        "--service",
        default=_ESA_AUTH_IDP_HOST,
        help="Keyring service name.",
    )
    parser.add_argument(
        "--username-key",
        default=_ESA_KEYRING_DEFAULT_USER_KEY,
        help="Keyring username key for storing the default user.",
    )


def _update_keyring_from_args(args) -> None:
    """Update keyring credentials from parsed CLI arguments."""
    user = update_keyring(
        user=args.user,
        password=args.password,
        service=args.service,
        username_key=args.username_key,
    )
    print(f"Stored credentials for {user} in keyring service {args.service}.")


def _add_update_netrc_arguments(parser) -> None:
    """Add shared netrc arguments to an argparse parser."""
    parser.add_argument("--user", default=None, help="ESA username.")
    parser.add_argument("--password", default=None, help="ESA password.")
    parser.add_argument(
        "--machine",
        default=_ESA_CS2_HOST,
        help="Netrc machine host key.",
    )
    parser.add_argument(
        "--netrc-file",
        default=None,
        help="Override path to netrc file (default: ~/.netrc).",
    )


def _update_netrc_from_args(args) -> None:
    """Update netrc credentials from parsed CLI arguments."""
    netrc_path = update_netrc(
        user=args.user,
        password=args.password,
        machine=args.machine,
        netrc_file=args.netrc_file,
    )
    print(
        f"Wrote plaintext credentials for {args.machine} to {netrc_path}. "
        "Prefer keyring for interactive setups."
    )


def cryoswath_cli(argv: list[str] | None = None) -> None:
    """Top-level CryoSwath command group."""
    from argparse import ArgumentParser

    parser = ArgumentParser("cryoswath", description="CryoSwath command line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_config_parser = subparsers.add_parser(
        "create-config",
        help="Create a CryoSwath project path configuration.",
    )
    _add_create_config_arguments(create_config_parser)
    create_config_parser.set_defaults(func=_create_config_from_args)

    aux_parser = subparsers.add_parser(
        "download-aux-data",
        help="Download the CryoSwath auxiliary-data snapshot from Zenodo.",
    )
    _add_download_auxiliary_data_arguments(aux_parser)
    aux_parser.set_defaults(func=_download_auxiliary_data_from_args)

    tutorials_parser = subparsers.add_parser(
        "get-tutorials",
        help="Copy packaged CryoSwath tutorial notebooks into a project.",
    )
    _add_get_tutorials_arguments(tutorials_parser)
    tutorials_parser.set_defaults(func=_get_tutorials_from_args)

    rgi_parser = subparsers.add_parser(
        "download-rgi",
        help="Download and extract one RGI o1 region bundle.",
    )
    _add_download_rgi_arguments(rgi_parser)
    rgi_parser.set_defaults(func=_download_rgi_from_args)

    update_tracks_parser = subparsers.add_parser(
        "update-tracks",
        help="Refresh cached ground-track and filename lookup tables.",
    )
    update_tracks_parser.set_defaults(func=lambda args: update_track_database())

    keyring_parser = subparsers.add_parser(
        "update-keyring",
        help="Create or update keyring credentials for ESA access.",
    )
    _add_update_keyring_arguments(keyring_parser)
    keyring_parser.set_defaults(func=_update_keyring_from_args)

    netrc_parser = subparsers.add_parser(
        "update-netrc",
        help="Create or update ~/.netrc credentials for ESA access.",
    )
    _add_update_netrc_arguments(netrc_parser)
    netrc_parser.set_defaults(func=_update_netrc_from_args)

    args = parser.parse_args(argv)
    args.func(args)


def update_email(email: str = None):
    """Deprecated helper for pre-2026 email-based FTP auth."""
    warnings.warn(
        "update_email() is deprecated. Anonymous/email FTP login is no longer "
        "supported. Use environment variables, keyring, ~/.netrc (plaintext "
        "fallback), or legacy config.ini [user] name/password.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    if email is None:
        email = input("Enter your email")
    if re.fullmatch(r"[^@]+@[^@]+\.[a-z]{2,9}", email.strip().lower()):
        config = ConfigParser()
        config.read("config.ini")
        if "user" not in config:
            config["user"] = dict()
        config["user"].update({"email": email})
        with open("config.ini", "w") as f:  # update/overwrite
            config.write(f)
    else:
        print(
            'Didn\'t match required pattern "[^@]+@[^@]+\\.[a-z]{2,9}". If',
            "your email indeed doesn't match, file an issue. Your input was:",
            email,
        )


def update_track_database() -> None:
    """Refresh cached ground-track and filename lookup tables."""
    load_cs_ground_tracks(update="regular")
    load_cs_full_file_names(update="regular")


def update_track_database_cli() -> None:
    """CLI wrapper around :func:`update_track_database`."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "cryoswath-update-tracks",
        description="Updates the track database. Run this once in a while and always "
        "if you wish to include the latest tracks.",
    )
    parser.parse_args()
    update_track_database()


# CREDIT: mgab https://stackoverflow.com/a/22376126
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Warning hook that appends stack trace to warning output."""
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def weighted_mean_excl_outliers(
    df: pd.DataFrame | xr.Dataset = None,
    weights: np.ndarray | str = "weights",
    *,
    values: np.ndarray | str = None,
    deviation_factor: int = 5,
    return_mask: bool = False,
) -> float:
    """Calculates the weighted average after excluding outliers.

    Note: This function uses `np.average` which expects weights similar
          to 1/variance - incontrast to `np.lstsq` and derivates, that
          expect 1/std and square the weights internally.

    Args:
        df (DataFrame): DataFrame containing values and weights.
        values (1d-numpy array): Values to average or name of dataframe
            column to average.
        weights (1d-numpy array): Weights to apply to values or name
            of dataframe column to use.
        deviation_factor (int, optional): Factor to apply to standard
            deviation. Values further appart from average are excluded.
            Defaults to 5.

    Returns:
        float: Weighted average excluding outliers. if `return_mask`,
        returns a boolean mask that is true where outliers were detected.
        The mask is same as input type.
    """
    # todo: write a test: mainly confirm math works
    if isinstance(df, pd.DataFrame) or isinstance(df, xr.Dataset):
        values = df[values].values
        if isinstance(weights, str):
            weights = df[weights].values
    outlier_mask = flag_outliers(
        values,
        weights=weights,
        stat=np.average,
        deviation_factor=deviation_factor,
        scaling_factor=1,
    )
    _ess = float(effective_sample_size(weights[~outlier_mask]))
    # print(outlier_mask)
    if _ess > 6:
        avg = float(np.average(values[~outlier_mask], weights=weights[~outlier_mask]))
        _var = float(
            np.average(
                (values[~outlier_mask] - avg) ** 2, weights=weights[~outlier_mask]
            )
        )
        if return_mask:
            return avg, _var, _ess, outlier_mask
    else:
        avg = np.nan
        _var = np.nan
        if return_mask:
            return avg, _var, _ess, outlier_mask == -1  # all False
    return avg, _var, _ess


def xycut(
    data: gpd.GeoDataFrame,
    x_chunk_meter=3 * 4 * 5 * 1_000,
    y_chunk_meter=3 * 4 * 5 * 1_000,
) -> list[dict[str, Union[float, gpd.GeoDataFrame]]]:
    """Chunk point data in planar reference system

    This mainly is a helper function for `l3.build_dataset()` that takes
    many data points and chunks them based on their location.
    However, it may be helpful in other contexts.

    Returns:
        list: List of dicts of which each contains the x and y extents of
        the current chunk and the GeoDataFrame or Series of the point data.
    """
    # 3*4*5=60 [km] fits many grid cell sizes and makes reasonable chunks
    # ! only implemented for l2 data. however, easily convertible for l1b and l3 data

    def lower_x(x):
        return (x // x_chunk_meter) * x_chunk_meter

    def lower_y(y):
        return (y // y_chunk_meter) * y_chunk_meter

    minx, miny, maxx, maxy = data.total_bounds
    chunks = []
    for x in np.arange(lower_x(minx), lower_x(maxx) + 1, x_chunk_meter):
        for y in np.arange(lower_y(miny), lower_y(maxy) + 1, y_chunk_meter):
            tmp = data.cx[x : x + x_chunk_meter, y : y + y_chunk_meter]
            if tmp.empty:
                continue
            chunks.append(
                dict(
                    x_interval_start=x,
                    x_interval_stop=x + x_chunk_meter,
                    y_interval_start=y,
                    y_interval_stop=y + y_chunk_meter,
                    data=tmp,
                )
            )
    return chunks
