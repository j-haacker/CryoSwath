"""Split/remerge experiment for the finalized Svalbard L3 NetCDF product."""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

l3 = importlib.import_module("cryoswath.l3")


TRUTH_PATH = Path(
    "data/L3/"
    "Glacier_surface_elevation__Svalbard_and_Jan_Mayen__Svalbard__monthly_500x500m.nc"
)
OUTPUT_DIR = Path("data/tmp/l3_extension_experiment")
BASE_PATH = OUTPUT_DIR / "svalbard_base_to_202310.nc"
EXTENSION_PATH = OUTPUT_DIR / "svalbard_extension_from_202211.nc"
MERGED_PATH = OUTPUT_DIR / "svalbard_merged.nc"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _values_match(left: xr.DataArray, right: xr.DataArray) -> bool:
    return l3._array_values_match(left.values, right.values)


def _attrs_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(l3._attribute_values_match(left[key], right[key]) for key in left)


def _schema_matches(left: xr.Dataset, right: xr.Dataset) -> bool:
    if set(left.dims) != set(right.dims):
        return False
    if any(left.sizes[dim] != right.sizes[dim] for dim in left.dims):
        return False
    if set(left.coords) != set(right.coords):
        return False
    if set(left.data_vars) != set(right.data_vars):
        return False
    for name in left.variables:
        if left[name].dims != right[name].dims:
            return False
        if left[name].dtype != right[name].dtype:
            return False
        if not _attrs_match(left[name].attrs, right[name].attrs):
            return False
    return True


def _write_split_products(truth: xr.Dataset) -> None:
    encoding = l3._finalized_netcdf_encoding_from(truth)
    base = truth.sel(time=slice("2010-06-01", "2023-10-01"))
    extension = truth.sel(time=slice("2022-11-01", "2024-11-01"))
    base.to_netcdf(BASE_PATH, engine="h5netcdf", encoding=encoding)
    extension.to_netcdf(EXTENSION_PATH, engine="h5netcdf", encoding=encoding)


def _validation_report(truth: xr.Dataset, merged: xr.Dataset) -> dict[str, Any]:
    truth_history = str(truth.attrs.get("history", "")).splitlines()
    merged_history = str(merged.attrs.get("history", "")).splitlines()
    data_var_matches = {
        name: _values_match(merged[name], truth[name]) for name in truth.data_vars
    }
    coord_matches = {
        name: _values_match(merged.coords[name], truth.coords[name])
        for name in truth.coords
    }
    non_history_truth_attrs = {
        key: value for key, value in truth.attrs.items() if key != "history"
    }
    non_history_merged_attrs = {
        key: value for key, value in merged.attrs.items() if key != "history"
    }
    report = {
        "truth_path": str(TRUTH_PATH),
        "base_path": str(BASE_PATH),
        "extension_path": str(EXTENSION_PATH),
        "merged_path": str(MERGED_PATH),
        "time_axis": {
            "start": str(merged.indexes["time"][0].date()),
            "end": str(merged.indexes["time"][-1].date()),
            "steps": int(merged.sizes["time"]),
            "expected_start": "2010-06-01",
            "expected_end": "2024-11-01",
            "expected_steps": 174,
            "matches_expected": (
                str(merged.indexes["time"][0].date()) == "2010-06-01"
                and str(merged.indexes["time"][-1].date()) == "2024-11-01"
                and int(merged.sizes["time"]) == 174
            ),
        },
        "data_variables_match": data_var_matches,
        "all_data_variables_match": all(data_var_matches.values()),
        "coordinates_match": coord_matches,
        "all_coordinates_match": all(coord_matches.values()),
        "schema_matches": _schema_matches(merged, truth),
        "non_history_global_attrs_match": _attrs_match(
            non_history_merged_attrs,
            non_history_truth_attrs,
        ),
        "history": {
            "original_line_count": len(truth_history),
            "merged_line_count": len(merged_history),
            "contains_new_merge_line": bool(
                merged_history
                and "merge finalized L3 dataset extension" in merged_history[-1]
            ),
            "base_history_preserved": merged_history[: len(truth_history)]
            == truth_history,
        },
    }
    report["passed"] = bool(
        report["time_axis"]["matches_expected"]
        and report["all_data_variables_match"]
        and report["all_coordinates_match"]
        and report["schema_matches"]
        and report["non_history_global_attrs_match"]
        and report["history"]["contains_new_merge_line"]
        and report["history"]["base_history_preserved"]
        and report["history"]["merged_line_count"]
        == report["history"]["original_line_count"] + 1
    )
    return _jsonable(report)


def main() -> None:
    if not TRUTH_PATH.is_file():
        raise FileNotFoundError(f"Missing finalized Svalbard NetCDF: {TRUTH_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(TRUTH_PATH, decode_coords="all") as truth:
        _write_split_products(truth)
        l3.merge_finalized_dataset_extension(
            BASE_PATH,
            EXTENSION_PATH,
            output_path=MERGED_PATH,
            overlap_time_steps=12,
            overlap_policy="abort",
            overlap_rtol=0.0,
            overlap_atol=0.0,
        )
        with xr.open_dataset(MERGED_PATH, decode_coords="all") as merged:
            report = _validation_report(truth, merged)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
