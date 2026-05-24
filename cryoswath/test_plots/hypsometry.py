"""Quick-look plots for hypsometry and retreat-zone diagnostic payloads."""

__all__ = [
    "fit_fill_mask",
    "frontal_retreat_threshold",
    "local_deviation",
    "model_preview",
    "outlier_neighbour_check",
    "plot_event",
]

import numpy as np
from matplotlib import pyplot as plt


def _values(data):
    if hasattr(data, "values"):
        data = data.values
    return np.asarray(data)


def _unstacked_plot(data, ax, **kwargs):
    try:
        data.unstack().sortby("x").sortby("y").T.plot(ax=ax, **kwargs)
    except (AttributeError, KeyError, ValueError):
        data.plot(ax=ax, **kwargs)


def outlier_neighbour_check(payload: dict, ax: plt.Axes = None) -> plt.Axes:
    """Plot neighbour counts used by the outlier-replacement pre-check."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 5))[1]
    _unstacked_plot(payload["neighbour_count"], ax, cmap="cool")
    ax.set_title("Neighbour count")
    return ax


def model_preview(payload: dict, axes=None) -> plt.Figure:
    """Plot input data next to the hypsometric model before local correction."""
    if axes is None:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    else:
        fig = axes[0].figure
    ds = payload["ds"]
    main_var = payload["main_var"]
    _unstacked_plot(ds[main_var], axes[0], robust=True, cmap="RdYlBu")
    _unstacked_plot(payload["modelled"], axes[1], robust=True, cmap="RdYlBu")
    axes[0].set_title(main_var)
    axes[1].set_title("Modelled")
    fig.tight_layout()
    return fig


def local_deviation(payload: dict, ax: plt.Axes = None) -> plt.Axes:
    """Plot local deviation from the hypsometric model."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 5))[1]
    _unstacked_plot(payload["local_deviation_metric"], ax, robust=True, cmap="RdYlBu")
    ax.set_title("Local deviation")
    return ax


def fit_fill_mask(payload: dict, ax: plt.Axes = None) -> plt.Axes:
    """Plot elevation/value scatter, fitted curve, uncertainty band, and fill mask."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 5))[1]
    ds = payload["ds"]
    main_var = payload["main_var"]
    elev = payload["elev"]
    elev_vals = _values(ds[elev]).reshape(-1)
    main_vals = _values(ds[main_var]).reshape(-1)
    modelled_vals = _values(payload["modelled"]).reshape(-1)
    fill_mask = _values(payload["fill_mask"]).astype(bool).reshape(-1)
    initially_missing = np.isnan(main_vals)

    ax.scatter(elev_vals[~fill_mask], main_vals[~fill_mask], s=8, label="kept")
    ax.scatter(
        elev_vals[fill_mask],
        main_vals[fill_mask],
        s=20,
        ec="tab:purple",
        fc="none",
        label="filled/outlier",
    )
    ax.scatter(
        elev_vals[initially_missing],
        modelled_vals[initially_missing],
        s=20,
        ec="tab:orange",
        fc="none",
        label="modelled missing",
    )

    if "fit_x_vals" in payload and "fit_y_vals" in payload:
        fit_x_vals = _values(payload["fit_x_vals"]).reshape(-1)
        fit_y_vals = _values(payload["fit_y_vals"]).reshape(-1)
        ax.plot(fit_x_vals, fit_y_vals, c="tab:orange", label="fit")
        if "neighbour_std" in payload:
            spread = 2 * float(payload["neighbour_std"].mean())
            ax.plot(fit_x_vals, fit_y_vals + spread, c="tab:gray", ls="dashed")
            ax.plot(fit_x_vals, fit_y_vals - spread, c="tab:gray", ls="dashed")

    if "elev_bin_means" in payload and "elev_bin_errs" in payload:
        means = payload["elev_bin_means"]
        errs = payload["elev_bin_errs"]
        bin_x = [idx.mid for idx in means.index]
        ax.errorbar(bin_x, means, errs, ls="none", c="tab:red", label="bin means")

    ax.set_xlabel(elev)
    ax.set_ylabel(main_var)
    ax.legend()
    return ax


def frontal_retreat_threshold(payload: dict, ax: plt.Axes = None) -> plt.Axes:
    """Plot elevation-band values used to detect a frontal retreat zone."""
    if ax is None:
        ax = plt.subplots(figsize=(7, 5))[1]
    band_values = payload["band_values"]
    band_values.plot(ax=ax, marker="o")
    ax.axhline(payload["threshold"], color="tab:red", ls="dashed", label="threshold")
    ax.axvline(payload["front_bin"].left, color="tab:purple", ls="dotted")
    ax.set_title("Frontal retreat threshold")
    ax.legend()
    return ax


def plot_event(name: str, payload: dict):
    """Dispatch a diagnostic event payload to the matching quick-look plot."""
    dispatch = {
        "hypsometry.outlier_neighbour_check": outlier_neighbour_check,
        "hypsometry.model_preview": model_preview,
        "hypsometry.local_deviation": local_deviation,
        "hypsometry.fit_fill_mask": fit_fill_mask,
        "frontal_retreat_zone.threshold": frontal_retreat_threshold,
    }
    return dispatch[name](payload)
