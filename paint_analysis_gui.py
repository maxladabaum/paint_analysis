from __future__ import annotations

import math
import os
import queue
import sys
import threading
import traceback
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from origami_analysis import (
    OrigamiAnalysisResult,
    OrigamiPickResult,
    align_picked_origamis,
    density_map_for_origami_picking,
    identify_origami_regions,
    integrate_rendered_density_at_sites,
    render_aligned_origami_density,
    render_localization_preview,
)
from drift_analysis import undrift_rcc_with_lattice_suppression

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "DNA PAINT Picasso-Style ROI Analyzer"
DEFAULT_DATA_DIR = Path.home() / "Desktop" / "LBNL_PAINT"


def user_state_dir() -> Path:
    """Return an OS-appropriate directory for machine-specific app state."""
    override = os.environ.get("PAINT_ANALYSIS_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "PaintAnalysis"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PaintAnalysis"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home) if xdg_state_home else Path.home() / ".local" / "state"
    return base / "paint-analysis"


APP_STATE_DIR = user_state_dir()
RECENT_DIR_FILE = APP_STATE_DIR / "recent-data-directory.txt"
MAX_RENDER_PIXELS = 30_000_000
MAP_AXES_RECT = (0.10, 0.12, 0.74, 0.78)
MAP_COLORBAR_RECT = (0.87, 0.18, 0.025, 0.66)
ORIGAMI_SOURCE_AXES_RECT = (0.14, 0.10, 0.72, 0.72)
ORIGAMI_SOURCE_COLORBAR_RECT = (0.90, 0.16, 0.02, 0.60)
ORIGAMI_RESULT_AXES_RECT = (0.14, 0.12, 0.72, 0.68)
ORIGAMI_RESULT_COLORBAR_RECT = (0.90, 0.18, 0.02, 0.56)
RAW_MAP_TAB = 0
CORRECTED_MAP_TAB = 1
LINKED_MAP_TAB = 2
FILTERED_MAP_TAB = 3
HISTOGRAM_TAB = 4
TEMPORAL_TAB = 5
ORIGAMI_TAB = 6


@dataclass
class LoadedData:
    path: Path
    locs: pd.DataFrame
    info: list[dict[str, Any]]
    metadata: dict[str, Any]


def read_yaml_metadata(path: Path) -> dict[str, Any]:
    yaml_path = path.with_suffix(".yaml")
    if not yaml_path.exists():
        return {}

    merged: dict[str, Any] = {}
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if isinstance(doc, dict):
                    merged.update(doc)
    except Exception:
        return {}
    return merged


def find_locs_dataset(h5: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visit(_name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.dtype.names:
            names = set(obj.dtype.names)
            if {"frame", "x", "y"}.issubset(names):
                candidates.append(obj)

    h5.visititems(visit)
    if not candidates:
        raise ValueError("No compound localization dataset with frame, x, and y fields was found.")
    candidates.sort(key=lambda dataset: dataset.size, reverse=True)
    return candidates[0]


def picasso_info_from_metadata(metadata: dict[str, Any], locs: pd.DataFrame) -> list[dict[str, Any]]:
    frames = int(metadata.get("Frames") or (np.nanmax(locs["frame"]) + 1))
    width = int(metadata.get("Width") or math.ceil(float(np.nanmax(locs["x"]) + 1)))
    height = int(metadata.get("Height") or math.ceil(float(np.nanmax(locs["y"]) + 1)))
    pixelsize = float(metadata.get("Pixelsize") or 130.0)
    return [{"Frames": frames, "Width": width, "Height": height, "Pixelsize": pixelsize}]


def read_locs_hdf5(path: Path) -> LoadedData:
    with h5py.File(path, "r") as h5:
        dataset = find_locs_dataset(h5)
        raw = dataset[()]

    data: dict[str, np.ndarray] = {}
    for name in raw.dtype.names or ():
        values = np.asarray(raw[name])
        if np.issubdtype(values.dtype, np.number):
            data[name] = values.astype(float, copy=False)

    locs = pd.DataFrame(data)
    locs = locs[np.isfinite(locs["frame"]) & np.isfinite(locs["x"]) & np.isfinite(locs["y"])].copy()
    locs["frame"] = locs["frame"].astype(np.uint32)

    metadata = read_yaml_metadata(path)
    metadata["Localization count"] = int(len(locs))
    metadata["Fields"] = ", ".join(locs.columns)
    info = picasso_info_from_metadata(metadata, locs)
    return LoadedData(path=path, locs=locs, info=info, metadata=metadata)


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def precision_qc_values(values_nm: np.ndarray) -> np.ndarray:
    values = finite_values(values_nm)
    values = values[values > 0]
    if values.size < 4:
        return values
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return values
    return values[values <= q75 + 12.0 * iqr]


def automatic_histogram_bins(values: np.ndarray) -> int:
    values = finite_values(values)
    count = values.size
    if count < 2:
        return 1
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    value_range = float(np.max(values) - np.min(values))
    if value_range <= 0:
        return 1
    if iqr > 0:
        bin_width = 2.0 * iqr / np.cbrt(count)
        bins = int(math.ceil(value_range / bin_width)) if bin_width > 0 else 0
    else:
        bins = int(math.ceil(math.sqrt(count)))
    return int(np.clip(bins, 10, 150))


def fixed_width_histogram_bins(values: np.ndarray, bin_size: float) -> np.ndarray | None:
    values = finite_values(values)
    if values.size == 0 or not np.isfinite(bin_size) or bin_size <= 0:
        return None
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return None
    if data_max <= data_min:
        left = math.floor(data_min / bin_size) * bin_size
        return np.asarray([left, left + bin_size], dtype=float)
    left = math.floor(data_min / bin_size) * bin_size
    right = math.ceil(data_max / bin_size) * bin_size
    if right <= data_max:
        right += bin_size
    bins = np.arange(left, right + bin_size * 0.5, bin_size, dtype=float)
    if bins.size < 2:
        return np.asarray([left, left + bin_size], dtype=float)
    return bins


def histogram_axis_limits(counts: np.ndarray, bin_edges: np.ndarray) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    nonzero = np.flatnonzero(counts > 0)
    if nonzero.size == 0:
        return None, None

    first = int(nonzero[0])
    last = int(nonzero[-1])
    x_min = float(bin_edges[first])
    x_max = float(bin_edges[last + 1])
    xlim = None
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
        padding = 0.02 * (x_max - x_min)
        xlim = (x_min - padding, x_max + padding)

    y_max = float(np.max(counts))
    ylim = (0.0, y_max * 1.08 if y_max > 0 else 1.0)
    return xlim, ylim


def histogram_axis_limits_for_values(values: np.ndarray, counts: np.ndarray, bin_edges: np.ndarray, mode: str | None = None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if mode not in {"precision_radial_nm", "lpx_nm", "lpy_nm"}:
        return histogram_axis_limits(counts, bin_edges)
    precision_range = robust_precision_display_range(values)
    if precision_range is None:
        return histogram_axis_limits(counts, bin_edges)
    x_min, x_max = precision_range
    visible = (bin_edges[:-1] >= x_min) & (bin_edges[1:] <= x_max) & (counts > 0)
    if not np.any(visible):
        visible = (bin_edges[:-1] < x_max) & (bin_edges[1:] > x_min) & (counts > 0)
    if not np.any(visible):
        return histogram_axis_limits(counts, bin_edges)
    y_max = float(np.max(np.asarray(counts, dtype=float)[visible]))
    x_padding = 0.02 * (x_max - x_min)
    return (x_min - x_padding, x_max + x_padding), (0.0, y_max * 1.08 if y_max > 0 else 1.0)


def robust_precision_display_range(values: np.ndarray) -> tuple[float, float] | None:
    finite = finite_values(values)
    finite = finite[finite > 0]
    if finite.size < 10:
        return None
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        x_min = max(0.0, float(q25 - 3.0 * iqr))
        x_max = float(q75 + 3.0 * iqr)
    else:
        x_min = max(0.0, float(np.nanmin(finite)))
        x_max = float(np.nanpercentile(finite, 99.0))
    if not np.isfinite(x_max) or x_max <= x_min:
        return None
    return x_min, x_max


def robust_precision_axis_limits(values: np.ndarray) -> tuple[float, float] | None:
    display_range = robust_precision_display_range(values)
    if display_range is None:
        return None
    low, high = display_range
    padding = 0.04 * (high - low)
    return max(0.0, low - padding), high + padding


def df_to_arrays(locs: pd.DataFrame) -> dict[str, np.ndarray]:
    return {column: locs[column].to_numpy(dtype=float, copy=False) for column in locs.columns}


class PicassoAimStatusProgress:
    def __init__(self, callback: Any, description: str = "Undrifting by AIM (1/2)") -> None:
        self.callback = callback
        self.description = description
        self.phase_index = 0
        self.phase_count = 2
        self.start = 0
        self.end = 1

    def get_iterator(self, start: int = 0, end: int = 100, unit: str = "segment") -> range:
        self.start = int(start)
        self.end = max(int(end), self.start + 1)
        self._emit(self.start)
        return range(self.start, self.end)

    def set_value(self, value: int, *args: Any, **kwargs: Any) -> None:
        self._emit(int(value))

    def zero_progress(self, description: str | None = None, *args: Any, **kwargs: Any) -> None:
        if description:
            self.description = description
        self.phase_index += 1
        self.start = 0
        self.end = 1
        self._emit(0)

    def setMaximum(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setLabelText(self, text: str, *args: Any, **kwargs: Any) -> None:
        self.description = text

    def close(self, *args: Any, **kwargs: Any) -> None:
        self.callback("AIM drift correction: 100% complete.")

    def closeEvent(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def play_sound_notification(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _emit(self, value: int) -> None:
        total = max(1, self.end - self.start)
        done = min(max(value - self.start + 1, 0), total)
        phase_percent = 100.0 * done / total
        overall_percent = 100.0 * (self.phase_index + done / total) / self.phase_count
        self.callback(
            f"AIM drift correction: {overall_percent:5.1f}% overall "
            f"({self.description}, {phase_percent:5.1f}% through pass)."
        )


class SyncedMapToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas: FigureCanvasTkAgg, window: tk.Widget, app: Any, axis: Any) -> None:
        self.app = app
        self.axis = axis
        super().__init__(canvas, window)

    def home(self, *args: Any) -> None:
        if getattr(self.axis, "images", None):
            extent = self.axis.images[0].get_extent()
            self.axis.set_xlim(float(extent[0]), float(extent[1]))
            self.axis.set_ylim(float(extent[2]), float(extent[3]))
            self.canvas.draw_idle()
            self.app.after_idle(lambda: self.app._sync_map_limits_from(self.axis))
            return
        super().home(*args)
        self.app.after_idle(lambda: self.app._sync_map_limits_from(self.axis))

    def back(self, *args: Any) -> None:
        super().back(*args)
        self.app.after_idle(lambda: self.app._sync_map_limits_from(self.axis))

    def forward(self, *args: Any) -> None:
        super().forward(*args)
        self.app.after_idle(lambda: self.app._sync_map_limits_from(self.axis))


class OrigamiToolbar(NavigationToolbar2Tk):
    """Navigation toolbar with a stable Home view for rebuilt gallery plots."""

    GALLERY_OPTIONS = {"Individual origami gallery", "Individual Picasso G5M sites"}

    def __init__(self, canvas: FigureCanvasTkAgg, window: tk.Widget, app: Any) -> None:
        self.app = app
        super().__init__(canvas, window)

    def home(self, *args: Any) -> None:
        limits = self.app.origami_gallery_home_limits
        if (
            self.app.origami_last_rendered_plot_option in self.GALLERY_OPTIONS
            and limits is not None
            and self.canvas.figure.axes
        ):
            axis = self.canvas.figure.axes[0]
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            # Home becomes the view carried across the two synchronized
            # galleries until the user zooms or pans again.
            self.app.origami_gallery_view_limits = limits
            self.canvas.draw_idle()
            return
        super().home(*args)


def apply_drift_correction(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    method: str,
    segmentation: int,
    aim_intersect_nm: float,
    aim_roi_nm: float,
    progress_callback: Any | None = None,
    rcc_lattice_pitch_nm: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if method == "none":
        if progress_callback is not None:
            progress_callback("Using loaded coordinates without drift correction.")
        return locs.copy(), None, "No drift correction"

    from picasso import aim, postprocess

    frames = int(info[0]["Frames"])
    pixelsize = float(info[0]["Pixelsize"])
    if segmentation <= 0:
        raise ValueError("Drift segmentation must be greater than 0 frames.")
    if frames / segmentation < 4:
        raise ValueError("RCC/AIM needs at least four time segments. Use a smaller segmentation value.")

    locs_for_picasso = locs.copy()
    locs_for_picasso["frame"] = locs_for_picasso["frame"].astype(np.uint32)

    if method == "rcc":
        if not np.isfinite(rcc_lattice_pitch_nm) or rcc_lattice_pitch_nm < 0:
            raise ValueError("RCC lattice pitch must be a finite, non-negative value in nm.")
        segment_count = int(round(frames / int(segmentation)))
        pair_count = max(1, segment_count * (segment_count - 1) // 2)

        def segmentation_progress(index: int) -> None:
            if progress_callback is not None:
                current = min(index + 1, segment_count)
                overall = 100.0 * current / (segment_count + pair_count)
                phase = 100.0 * current / segment_count
                progress_callback(f"RCC drift correction: {overall:5.1f}% overall (generating segments, {phase:5.1f}%).")

        def rcc_progress(index: int) -> None:
            if progress_callback is not None:
                current = min(index + 1, pair_count)
                overall = 100.0 * (segment_count + current) / (segment_count + pair_count)
                phase = 100.0 * current / pair_count
                progress_callback(f"RCC drift correction: {overall:5.1f}% overall (correlating image pairs, {phase:5.1f}%).")

        if progress_callback is not None:
            progress_callback(f"RCC drift correction started ({segment_count} segments).")
        if rcc_lattice_pitch_nm > 0:
            drift, corrected_locs = undrift_rcc_with_lattice_suppression(
                locs_for_picasso,
                info,
                int(segmentation),
                float(rcc_lattice_pitch_nm),
                segmentation_callback=segmentation_progress,
                rcc_callback=rcc_progress,
            )
            label = (
                f"Picasso RCC, segmentation={segmentation} frames, "
                f"Fourier lattice notch pitch={rcc_lattice_pitch_nm:g} nm"
            )
        else:
            drift, corrected_locs = postprocess.undrift(
                locs_for_picasso,
                info,
                int(segmentation),
                display=False,
                segmentation_callback=segmentation_progress,
                rcc_callback=rcc_progress,
            )
            label = f"Picasso RCC, segmentation={segmentation} frames"
        return corrected_locs, drift, label

    if method == "aim":
        aim_progress = PicassoAimStatusProgress(progress_callback) if progress_callback is not None else None
        if progress_callback is not None:
            progress_callback("AIM drift correction started.")
        original_progress_dialog = aim.lib.ProgressDialog
        try:
            if aim_progress is not None:
                aim.lib.ProgressDialog = PicassoAimStatusProgress
            corrected_locs, _new_info, drift = aim.aim(
                locs_for_picasso,
                info,
                segmentation=int(segmentation),
                intersect_d=float(aim_intersect_nm) / pixelsize,
                roi_r=float(aim_roi_nm) / pixelsize,
                progress=aim_progress,
            )
        finally:
            aim.lib.ProgressDialog = original_progress_dialog
        return corrected_locs, drift, f"Picasso AIM, segmentation={segmentation} frames, intersect={aim_intersect_nm:g} nm, ROI={aim_roi_nm:g} nm"

    raise ValueError(f"Unknown drift correction method: {method}")


def render_picasso_map(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    disp_px_size_nm: float,
    blur_method: str,
    min_blur_width: float,
    viewport_nm: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    from picasso import render

    blur = None if blur_method == "none" else blur_method
    pixelsize = float(info[0]["Pixelsize"])
    if viewport_nm is None:
        viewport = None
        extent = (0.0, float(info[0]["Width"]) * pixelsize, 0.0, float(info[0]["Height"]) * pixelsize)
    else:
        x0_nm, x1_nm, y0_nm, y1_nm = viewport_nm
        x_min_nm = max(0.0, min(x0_nm, x1_nm))
        x_max_nm = min(float(info[0]["Width"]) * pixelsize, max(x0_nm, x1_nm))
        y_min_nm = max(0.0, min(y0_nm, y1_nm))
        y_max_nm = min(float(info[0]["Height"]) * pixelsize, max(y0_nm, y1_nm))
        if x_max_nm <= x_min_nm or y_max_nm <= y_min_nm:
            viewport = None
            extent = (0.0, float(info[0]["Width"]) * pixelsize, 0.0, float(info[0]["Height"]) * pixelsize)
        else:
            viewport = ((y_min_nm / pixelsize, x_min_nm / pixelsize), (y_max_nm / pixelsize, x_max_nm / pixelsize))
            extent = (x_min_nm, x_max_nm, y_min_nm, y_max_nm)

    n_rendered, image = render.render(
        locs,
        info,
        viewport=viewport,
        blur_method=blur,
        min_blur_width=float(min_blur_width),
        disp_px_size=float(disp_px_size_nm),
    )
    return {
        "result_type": "map",
        "image": np.asarray(image, dtype=float),
        "extent": extent,
        "viewport_nm": viewport_nm,
        "n_rendered": int(n_rendered),
        "disp_px_size_nm": float(disp_px_size_nm),
        "blur_method": blur_method,
    }


def render_fast_density_map(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    disp_px_size_nm: float,
    viewport_nm: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    pixelsize = float(info[0]["Pixelsize"])
    full_extent = (0.0, float(info[0]["Width"]) * pixelsize, 0.0, float(info[0]["Height"]) * pixelsize)
    if viewport_nm is None:
        x_min_nm, x_max_nm, y_min_nm, y_max_nm = full_extent
    else:
        x0_nm, x1_nm, y0_nm, y1_nm = viewport_nm
        x_min_nm = max(full_extent[0], min(x0_nm, x1_nm))
        x_max_nm = min(full_extent[1], max(x0_nm, x1_nm))
        y_min_nm = max(full_extent[2], min(y0_nm, y1_nm))
        y_max_nm = min(full_extent[3], max(y0_nm, y1_nm))
        if x_max_nm <= x_min_nm or y_max_nm <= y_min_nm:
            x_min_nm, x_max_nm, y_min_nm, y_max_nm = full_extent
            viewport_nm = None

    width_nm = max(float(disp_px_size_nm), x_max_nm - x_min_nm)
    height_nm = max(float(disp_px_size_nm), y_max_nm - y_min_nm)
    width_px = max(1, int(math.ceil(width_nm / float(disp_px_size_nm))))
    height_px = max(1, int(math.ceil(height_nm / float(disp_px_size_nm))))
    effective_disp_px = float(disp_px_size_nm)
    total_px = width_px * height_px
    if total_px > MAX_RENDER_PIXELS:
        effective_disp_px *= math.sqrt(total_px / MAX_RENDER_PIXELS) * 1.05
        width_px = max(1, int(math.ceil(width_nm / effective_disp_px)))
        height_px = max(1, int(math.ceil(height_nm / effective_disp_px)))

    if locs.empty:
        image = np.zeros((height_px, width_px), dtype=float)
        n_rendered = 0
    else:
        x_nm = locs["x"].to_numpy(dtype=float) * pixelsize
        y_nm = locs["y"].to_numpy(dtype=float) * pixelsize
        finite = np.isfinite(x_nm) & np.isfinite(y_nm)
        in_view = finite & (x_nm >= x_min_nm) & (x_nm <= x_max_nm) & (y_nm >= y_min_nm) & (y_nm <= y_max_nm)
        image, _y_edges, _x_edges = np.histogram2d(
            y_nm[in_view],
            x_nm[in_view],
            bins=(height_px, width_px),
            range=((y_min_nm, y_max_nm), (x_min_nm, x_max_nm)),
        )
        n_rendered = int(np.count_nonzero(in_view))

    return {
        "result_type": "fast_density_map",
        "image": np.asarray(image, dtype=float),
        "extent": (x_min_nm, x_max_nm, y_min_nm, y_max_nm),
        "viewport_nm": viewport_nm,
        "n_rendered": n_rendered,
        "disp_px_size_nm": effective_disp_px,
        "blur_method": "fast binning",
    }


def render_filtered_map_with_settings(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    disp_px_size_nm: float,
    blur_method: str,
    min_blur_width: float,
    viewport_nm: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    if blur_method == "none":
        return render_fast_density_map(locs, info, disp_px_size_nm, viewport_nm)
    return render_picasso_map(locs, info, disp_px_size_nm, blur_method, min_blur_width, viewport_nm)


def scale_density_like_picasso(image: np.ndarray, min_density: float, max_density: float) -> tuple[np.ndarray, tuple[float, float]]:
    image = np.asarray(image, dtype=float)
    if max_density <= min_density:
        max_density = 0.5 * float(np.nanmax(image)) if image.size else 1.0
        min_density = 0.0
    if min_density == max_density:
        max_density = min_density + 1e-6
    scaled = (image - min_density) / (max_density - min_density)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.clip(scaled, 0.0, 1.0), (float(min_density), float(max_density))


def roi_locs(locs: pd.DataFrame, roi_nm: tuple[float, float, float, float] | None, pixelsize_nm: float) -> pd.DataFrame:
    if roi_nm is None:
        return locs
    x0_nm, x1_nm, y0_nm, y1_nm = roi_nm
    x0 = min(x0_nm, x1_nm) / pixelsize_nm
    x1 = max(x0_nm, x1_nm) / pixelsize_nm
    y0 = min(y0_nm, y1_nm) / pixelsize_nm
    y1 = max(y0_nm, y1_nm) / pixelsize_nm
    mask = (locs["x"] >= x0) & (locs["x"] <= x1) & (locs["y"] >= y0) & (locs["y"] <= y1)
    return locs[mask].copy()


def radial_precision_nm(arrays: dict[str, np.ndarray], pixel_size_nm: float) -> np.ndarray:
    if "lpx" not in arrays or "lpy" not in arrays:
        raise ValueError("This file does not contain lpx and lpy localization precision fields.")
    return np.sqrt(arrays["lpx"] ** 2 + arrays["lpy"] ** 2) * pixel_size_nm


def nearest_neighbor_distance_nm(arrays: dict[str, np.ndarray], pixel_size_nm: float, max_points: int = 10000) -> np.ndarray:
    if "x" not in arrays or "y" not in arrays:
        raise ValueError("This file does not contain x and y coordinates.")
    x = arrays["x"]
    y = arrays["y"]
    mask = np.isfinite(x) & np.isfinite(y)
    points = np.column_stack([x[mask], y[mask]])
    if len(points) < 2:
        return np.asarray([])
    if len(points) > max_points:
        rng = np.random.default_rng(12345)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1] * pixel_size_nm


def nearest_neighbor_series_nm(locs: pd.DataFrame, pixel_size_nm: float) -> pd.Series:
    result = pd.Series(np.nan, index=locs.index, dtype=float)
    if not {"x", "y"}.issubset(locs.columns):
        raise ValueError("This file does not contain x and y coordinates.")
    valid = np.isfinite(locs["x"]) & np.isfinite(locs["y"])
    if int(valid.sum()) < 2:
        return result
    from scipy.spatial import cKDTree

    points = locs.loc[valid, ["x", "y"]].to_numpy(dtype=float)
    distances, _ = cKDTree(points).query(points, k=2)
    result.loc[valid] = distances[:, 1] * pixel_size_nm
    return result


def precision_qc_series(values_nm: np.ndarray, index: pd.Index) -> pd.Series:
    values = np.asarray(values_nm, dtype=float)
    series = pd.Series(values, index=index, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if int(valid.sum()) >= 20:
        valid = valid & (values <= float(np.nanquantile(values[valid], 0.995)))
    series.loc[~valid] = np.nan
    return series


def link_binding_events(
    arrays: dict[str, np.ndarray],
    exposure_ms: float,
    pixel_size_nm: float,
    radius_nm: float,
    max_gap_frames: int,
    progress_callback: Any | None = None,
) -> dict[str, np.ndarray]:
    for field in ("frame", "x", "y"):
        if field not in arrays:
            raise ValueError(f"This file does not contain the {field!r} field needed for event linking.")

    frame = arrays["frame"]
    x = arrays["x"]
    y = arrays["y"]
    valid = np.isfinite(frame) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        if progress_callback is not None:
            progress_callback(1, 1)
        empty = np.asarray([])
        return {"event_length_frames": empty, "event_length_ms": empty, "event_locs": empty, "event_photons": empty}

    frame = frame[valid].astype(int)
    x = x[valid]
    y = y[valid]
    photons = arrays.get("photons")
    photons = photons[valid] if photons is not None else np.zeros_like(x)

    order = np.lexsort((x, y, frame))
    frame = frame[order]
    x = x[order]
    y = y[order]
    photons = photons[order]

    radius_px = radius_nm / pixel_size_nm
    active: list[dict[str, float]] = []
    finished: list[dict[str, float]] = []
    total = int(order.size)
    progress_every = max(1, total // 100)

    for processed, (fr, px, py, ph) in enumerate(zip(frame, x, y, photons), start=1):
        still_active: list[dict[str, float]] = []
        for event in active:
            if fr - int(event["last_frame"]) <= max_gap_frames:
                still_active.append(event)
            else:
                finished.append(event)
        active = still_active

        best_index = None
        best_distance = float("inf")
        for idx, event in enumerate(active):
            if fr <= int(event["last_frame"]):
                continue
            distance = math.hypot(px - event["x"], py - event["y"])
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is None:
            active.append({"start_frame": float(fr), "last_frame": float(fr), "x": float(px), "y": float(py), "count": 1.0, "photons": float(ph) if np.isfinite(ph) else 0.0})
        else:
            event = active[best_index]
            count = event["count"] + 1.0
            event["x"] = (event["x"] * event["count"] + px) / count
            event["y"] = (event["y"] * event["count"] + py) / count
            event["count"] = count
            event["last_frame"] = float(fr)
            if np.isfinite(ph):
                event["photons"] += float(ph)
        if progress_callback is not None and (processed % progress_every == 0 or processed == total):
            progress_callback(processed, total)

    finished.extend(active)
    length_frames = np.asarray([event["last_frame"] - event["start_frame"] + 1.0 for event in finished])
    return {
        "event_length_frames": length_frames,
        "event_length_ms": length_frames * exposure_ms,
        "event_locs": np.asarray([event["count"] for event in finished]),
        "event_photons": np.asarray([event["photons"] for event in finished]),
    }


def event_metric_series_for_localizations(
    locs: pd.DataFrame,
    exposure_ms: float,
    pixel_size_nm: float,
    radius_nm: float,
    max_gap_frames: int,
    mode: str,
) -> pd.Series:
    for field in ("frame", "x", "y"):
        if field not in locs.columns:
            raise ValueError(f"This file does not contain the {field!r} field needed for event linking.")
    result = pd.Series(np.nan, index=locs.index, dtype=float)
    valid = np.isfinite(locs["frame"]) & np.isfinite(locs["x"]) & np.isfinite(locs["y"])
    if not np.any(valid):
        return result

    valid_locs = locs.loc[valid]
    frame = valid_locs["frame"].to_numpy(dtype=int)
    x = valid_locs["x"].to_numpy(dtype=float)
    y = valid_locs["y"].to_numpy(dtype=float)
    photons = valid_locs["photons"].to_numpy(dtype=float) if "photons" in valid_locs.columns else np.zeros_like(x)
    source_index = np.asarray(valid_locs.index)
    order = np.lexsort((x, y, frame))

    radius_px = radius_nm / pixel_size_nm
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []

    for ordered_index in order:
        fr = int(frame[ordered_index])
        px = float(x[ordered_index])
        py = float(y[ordered_index])
        ph = float(photons[ordered_index]) if np.isfinite(photons[ordered_index]) else 0.0
        loc_index = source_index[ordered_index]

        still_active: list[dict[str, Any]] = []
        for event in active:
            if fr - int(event["last_frame"]) <= max_gap_frames:
                still_active.append(event)
            else:
                finished.append(event)
        active = still_active

        best_index = None
        best_distance = float("inf")
        for idx, event in enumerate(active):
            if fr <= int(event["last_frame"]):
                continue
            distance = math.hypot(px - float(event["x"]), py - float(event["y"]))
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is None:
            active.append({"start_frame": float(fr), "last_frame": float(fr), "x": px, "y": py, "count": 1.0, "photons": ph, "indices": [loc_index]})
        else:
            event = active[best_index]
            count = float(event["count"]) + 1.0
            event["x"] = (float(event["x"]) * float(event["count"]) + px) / count
            event["y"] = (float(event["y"]) * float(event["count"]) + py) / count
            event["count"] = count
            event["last_frame"] = float(fr)
            event["photons"] = float(event["photons"]) + ph
            event["indices"].append(loc_index)

    finished.extend(active)
    for event in finished:
        length_frames = float(event["last_frame"]) - float(event["start_frame"]) + 1.0
        if mode == "event_length_frames":
            value = length_frames
        elif mode == "event_length_ms":
            value = length_frames * exposure_ms
        elif mode == "event_locs":
            value = float(event["count"])
        elif mode == "event_photons":
            value = float(event["photons"])
        else:
            raise ValueError(f"Unsupported event metric: {mode}")
        result.loc[event["indices"]] = value
    return result


def linked_events_dataframe(
    locs: pd.DataFrame,
    exposure_ms: float,
    pixel_size_nm: float,
    radius_nm: float,
    max_gap_frames: int,
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    for field in ("frame", "x", "y"):
        if field not in locs.columns:
            raise ValueError(f"This file does not contain the {field!r} field needed for event linking.")
    if locs.empty:
        if progress_callback is not None:
            progress_callback(1, 1)
        return pd.DataFrame(columns=["frame", "x", "y", "photons", "event_length_frames", "event_length_ms", "event_locs", "event_photons"])

    valid = np.isfinite(locs["frame"]) & np.isfinite(locs["x"]) & np.isfinite(locs["y"])
    valid_locs = locs.loc[valid].copy()
    if valid_locs.empty:
        if progress_callback is not None:
            progress_callback(1, 1)
        return pd.DataFrame(columns=["frame", "x", "y", "photons", "event_length_frames", "event_length_ms", "event_locs", "event_photons"])

    frame = valid_locs["frame"].to_numpy(dtype=int)
    x = valid_locs["x"].to_numpy(dtype=float)
    y = valid_locs["y"].to_numpy(dtype=float)
    source_index = np.asarray(valid_locs.index)
    order = np.lexsort((x, y, frame))

    radius_px = radius_nm / pixel_size_nm
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    total = int(order.size)
    progress_total = max(1, total * 2)
    progress_every = max(1, progress_total // 100)

    for processed, ordered_index in enumerate(order, start=1):
        fr = int(frame[ordered_index])
        px = float(x[ordered_index])
        py = float(y[ordered_index])
        loc_index = source_index[ordered_index]

        still_active: list[dict[str, Any]] = []
        for event in active:
            if fr - int(event["last_frame"]) <= max_gap_frames:
                still_active.append(event)
            else:
                finished.append(event)
        active = still_active

        best_index = None
        best_distance = float("inf")
        for idx, event in enumerate(active):
            if fr <= int(event["last_frame"]):
                continue
            distance = math.hypot(px - float(event["x"]), py - float(event["y"]))
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is None:
            active.append({"start_frame": float(fr), "last_frame": float(fr), "x": px, "y": py, "count": 1.0, "indices": [loc_index]})
        else:
            event = active[best_index]
            count = float(event["count"]) + 1.0
            event["x"] = (float(event["x"]) * float(event["count"]) + px) / count
            event["y"] = (float(event["y"]) * float(event["count"]) + py) / count
            event["count"] = count
            event["last_frame"] = float(fr)
            event["indices"].append(loc_index)
        if progress_callback is not None and (processed % progress_every == 0 or processed == total):
            progress_callback(processed, progress_total)

    finished.extend(active)
    rows: list[dict[str, float]] = []
    row_total = max(1, len(finished))
    row_progress_every = max(1, row_total // 100)
    for event_id, event in enumerate(finished, start=1):
        event_locs = valid_locs.loc[event["indices"]]
        length_frames = float(event["last_frame"]) - float(event["start_frame"]) + 1.0
        row: dict[str, float] = {
            "frame": float(event["start_frame"]),
            "x": float(event["x"]),
            "y": float(event["y"]),
            "event_length_frames": length_frames,
            "event_length_ms": length_frames * exposure_ms,
            "event_locs": float(event["count"]),
            "event_photons": float(event_locs["photons"].sum()) if "photons" in event_locs.columns else 0.0,
        }
        row["photons"] = row["event_photons"]
        for column in ("lpx", "lpy", "sx", "sy", "bg"):
            if column in event_locs.columns:
                row[column] = float(event_locs[column].mean())
        rows.append(row)
        if progress_callback is not None and (event_id % row_progress_every == 0 or event_id == row_total):
            processed_work = total + int(round(total * event_id / row_total))
            progress_callback(min(processed_work, progress_total - 1), progress_total)
    linked = pd.DataFrame(rows, index=pd.Index([f"event_{idx}" for idx in range(len(rows))], name="linked_event_id"))
    if progress_callback is not None:
        progress_callback(progress_total, progress_total)
    return linked


def linked_localization_plot_data(
    locs: pd.DataFrame,
    pixel_size_nm: float,
    radius_nm: float,
    max_gap_frames: int,
    max_points: int = 75000,
) -> dict[str, Any]:
    arrays = df_to_arrays(locs)
    for field in ("frame", "x", "y"):
        if field not in arrays:
            raise ValueError(f"This file does not contain the {field!r} field needed for event linking.")

    frame = arrays["frame"]
    x = arrays["x"]
    y = arrays["y"]
    valid = np.isfinite(frame) & np.isfinite(x) & np.isfinite(y)
    frame = frame[valid].astype(int)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        raise ValueError("No finite x/y localizations were found.")

    order = np.lexsort((x, y, frame))
    frame = frame[order]
    x = x[order]
    y = y[order]
    radius_px = radius_nm / pixel_size_nm
    active: list[dict[str, float]] = []
    event_ids = np.full(x.size, -1, dtype=int)
    event_counts: dict[int, int] = {}
    next_event_id = 0

    for point_index, (fr, px, py) in enumerate(zip(frame, x, y)):
        active = [event for event in active if fr - int(event["last_frame"]) <= max_gap_frames]
        best_index = None
        best_distance = float("inf")
        for idx, event in enumerate(active):
            if fr <= int(event["last_frame"]):
                continue
            distance = math.hypot(px - event["x"], py - event["y"])
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is None:
            event_id = next_event_id
            next_event_id += 1
            active.append({"id": float(event_id), "last_frame": float(fr), "x": float(px), "y": float(py), "count": 1.0})
        else:
            event = active[best_index]
            event_id = int(event["id"])
            count = event["count"] + 1.0
            event["x"] = (event["x"] * event["count"] + px) / count
            event["y"] = (event["y"] * event["count"] + py) / count
            event["count"] = count
            event["last_frame"] = float(fr)

        event_ids[point_index] = event_id
        event_counts[event_id] = event_counts.get(event_id, 0) + 1

    linked_mask = np.asarray([event_counts[event_id] > 1 for event_id in event_ids])
    if x.size > max_points:
        rng = np.random.default_rng(12345)
        linked_indices = np.flatnonzero(linked_mask)
        singleton_indices = np.flatnonzero(~linked_mask)
        linked_keep = min(linked_indices.size, int(max_points * 0.75))
        singleton_keep = max_points - linked_keep
        selected_parts = []
        if linked_keep > 0:
            selected_parts.append(rng.choice(linked_indices, size=linked_keep, replace=False))
        if singleton_keep > 0 and singleton_indices.size > 0:
            selected_parts.append(rng.choice(singleton_indices, size=min(singleton_keep, singleton_indices.size), replace=False))
        selected = np.concatenate(selected_parts) if selected_parts else np.arange(min(max_points, x.size))
    else:
        selected = np.arange(x.size)

    return {
        "x_nm": x[selected] * pixel_size_nm,
        "y_nm": y[selected] * pixel_size_nm,
        "event_ids": event_ids[selected],
        "linked": linked_mask[selected],
        "plotted_count": int(selected.size),
        "total_count": int(x.size),
        "linked_event_count": int(sum(count > 1 for count in event_counts.values())),
    }


def histogram_values_for_mode(
    locs: pd.DataFrame,
    mode: str,
    pixel_size_nm: float,
    exposure_ms: float,
    link_radius_nm: float,
    max_gap_frames: int,
) -> tuple[np.ndarray, str]:
    arrays = df_to_arrays(locs)
    if mode == "photons":
        return arrays["photons"], "Photons per localization"
    if mode == "precision_radial_nm":
        return precision_qc_values(radial_precision_nm(arrays, pixel_size_nm)), "Radial localization precision (nm, QC filtered)"
    if mode == "lpx_nm":
        return precision_qc_values(arrays["lpx"] * pixel_size_nm), "Localization precision x (nm, QC filtered)"
    if mode == "lpy_nm":
        return precision_qc_values(arrays["lpy"] * pixel_size_nm), "Localization precision y (nm, QC filtered)"
    if mode == "frame":
        return arrays["frame"], "Acquisition frame number"
    if mode == "frame_gap":
        return frame_gap_values(locs), "Frames between occupied localization frames"
    if mode == "localizations_per_frame":
        return localizations_per_frame_values(locs), "Localizations per frame"
    if mode == "sx":
        return arrays["sx"], "Fitted PSF sigma x (pixels)"
    if mode == "sy":
        return arrays["sy"], "Fitted PSF sigma y (pixels)"
    if mode == "bg":
        return arrays["bg"], "Fitted local background (camera counts per pixel)"
    if mode == "nearest_neighbor_nm":
        return nearest_neighbor_distance_nm(arrays, pixel_size_nm), "Nearest-neighbor distance (nm)"
    if mode in {"event_length_frames", "event_length_ms", "event_locs", "event_photons"}:
        event_arrays = link_binding_events(arrays, exposure_ms, pixel_size_nm, link_radius_nm, max_gap_frames)
        labels = {
            "event_length_frames": "Binding-event length (frames)",
            "event_length_ms": "Binding-event length (ms)",
            "event_locs": "Localizations per linked event",
            "event_photons": "Photons per linked event",
        }
        return event_arrays[mode], labels[mode]
    if mode in arrays:
        return arrays[mode], mode.replace("_", " ")
    raise ValueError(f"Unsupported analysis mode: {mode}")


def localization_series_for_mode(
    locs: pd.DataFrame,
    mode: str,
    pixel_size_nm: float,
    exposure_ms: float,
    link_radius_nm: float,
    max_gap_frames: int,
) -> tuple[pd.Series, str]:
    arrays = df_to_arrays(locs)
    index = locs.index
    if mode == "photons":
        return pd.Series(arrays["photons"], index=index, dtype=float), "Photons per localization"
    if mode == "precision_radial_nm":
        return precision_qc_series(radial_precision_nm(arrays, pixel_size_nm), index), "Radial localization precision (nm, QC filtered)"
    if mode == "lpx_nm":
        return precision_qc_series(arrays["lpx"] * pixel_size_nm, index), "Localization precision x (nm, QC filtered)"
    if mode == "lpy_nm":
        return precision_qc_series(arrays["lpy"] * pixel_size_nm, index), "Localization precision y (nm, QC filtered)"
    if mode == "frame":
        return pd.Series(arrays["frame"], index=index, dtype=float), "Acquisition frame number"
    if mode == "localizations_per_frame":
        counts = locs.groupby(locs["frame"].astype(int), sort=True).transform("size")
        return pd.Series(counts.to_numpy(dtype=float), index=index, dtype=float), "Localizations per frame"
    if mode == "sx":
        return pd.Series(arrays["sx"], index=index, dtype=float), "Fitted PSF sigma x (pixels)"
    if mode == "sy":
        return pd.Series(arrays["sy"], index=index, dtype=float), "Fitted PSF sigma y (pixels)"
    if mode == "bg":
        return pd.Series(arrays["bg"], index=index, dtype=float), "Fitted local background (camera counts per pixel)"
    if mode == "nearest_neighbor_nm":
        return nearest_neighbor_series_nm(locs, pixel_size_nm), "Nearest-neighbor distance (nm)"
    if mode in {"event_length_frames", "event_length_ms", "event_locs", "event_photons"}:
        labels = {
            "event_length_frames": "Binding-event length (frames)",
            "event_length_ms": "Binding-event length (ms)",
            "event_locs": "Localizations per linked event",
            "event_photons": "Photons per linked event",
        }
        if mode in locs.columns:
            return pd.Series(locs[mode].to_numpy(dtype=float), index=index, dtype=float), labels[mode]
        return event_metric_series_for_localizations(locs, exposure_ms, pixel_size_nm, link_radius_nm, max_gap_frames, mode), labels[mode]
    if mode in arrays:
        return pd.Series(arrays[mode], index=index, dtype=float), mode.replace("_", " ")
    raise ValueError(f"Unsupported analysis mode: {mode}")


def localizations_per_frame_values(locs: pd.DataFrame) -> np.ndarray:
    if locs.empty:
        return np.asarray([], dtype=float)
    counts = locs.groupby(locs["frame"].astype(int), sort=True).size()
    return counts.to_numpy(dtype=float)


def frame_gap_values(locs: pd.DataFrame) -> np.ndarray:
    if locs.empty or "frame" not in locs.columns:
        return np.asarray([], dtype=float)
    frames = locs["frame"].to_numpy(dtype=float)
    frames = frames[np.isfinite(frames)]
    if frames.size < 2:
        return np.asarray([], dtype=float)
    unique_frames = np.unique(frames.astype(int))
    if unique_frames.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(unique_frames).astype(float)


class PaintAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x780")
        self.minsize(1040, 680)

        self.loaded: LoadedData | None = None
        self.corrected_locs: pd.DataFrame | None = None
        self.linked_locs: pd.DataFrame | None = None
        self.linked_source_count = 0
        self.linked_roi_nm: tuple[float, float, float, float] | None = None
        self.linked_params: tuple[float, float, int, str, str] | None = None
        self.linked_source_name = "Corrected map"
        self.linked_scope_name = "Selected ROI"
        self.drift: pd.DataFrame | None = None
        self.correction_label = "No drift correction"
        self.roi_nm: tuple[float, float, float, float] | None = None
        self.roi_patch = None
        self.linked_roi_patch = None
        self.filtered_roi_patch = None
        self.raw_roi_highlight = None
        self.selector: RectangleSelector | None = None
        self.linked_selector: RectangleSelector | None = None
        self.raw_map_colorbar = None
        self.map_colorbar = None
        self.linked_map_colorbar = None
        self.filtered_map_colorbar = None
        self.render_viewport_nm: tuple[float, float, float, float] | None = None
        self.raw_render_viewport_nm: tuple[float, float, float, float] | None = None
        self.shared_map_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.syncing_map_limits = False
        self.suspend_map_limit_sync = False
        self.active_notebook_tab = RAW_MAP_TAB
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.exposure_ms = tk.DoubleVar(value=100.0)
        self.pixel_size_nm = tk.DoubleVar(value=130.0)
        self.link_radius_nm = tk.DoubleVar(value=75.0)
        self.max_gap_frames = tk.IntVar(value=1)
        self.linking_source = tk.StringVar(value="Corrected map")
        self.linking_scope = tk.StringVar(value="Selected ROI")
        self.drift_method = tk.StringVar(value="none")
        self.drift_segmentation = tk.IntVar(value=1000)
        self.rcc_lattice_pitch_nm = tk.DoubleVar(value=700.0)
        self.aim_intersect_nm = tk.DoubleVar(value=20.0)
        self.aim_roi_nm = tk.DoubleVar(value=60.0)
        self.render_disp_px_nm = tk.DoubleVar(value=10.0)
        self.render_blur_method = tk.StringVar(value="smooth")
        self.min_blur_width = tk.DoubleVar(value=1.0)
        self.render_min_density = tk.DoubleVar(value=0.0)
        self.render_max_density = tk.DoubleVar(value=0.0)
        self.hist_mode = tk.StringVar(value="photons")
        self.hist_filter_scope = tk.StringVar(value="ROI localizations")
        self.filtered_map_source = tk.StringVar(value="Corrected map")
        self.hist_bin_size = tk.StringVar(value="")
        self.hist_frame_start = tk.StringVar(value="")
        self.hist_frame_end = tk.StringVar(value="")
        self.temporal_mode = tk.StringVar(value="precision_radial_nm")
        self.temporal_use_roi = tk.BooleanVar(value=True)
        self.temporal_use_linked = tk.BooleanVar(value=False)
        self.temporal_frame_start = tk.StringVar(value="")
        self.temporal_frame_end = tk.StringVar(value="")
        self.temporal_window_frames = tk.IntVar(value=100)
        self.temporal_step_frames = tk.IntVar(value=100)
        self.temporal_stat = tk.StringVar(value="mean")
        self.filter_scope_label = tk.StringVar(value="Filter scope: selected ROI")
        self.filter_bounds_label = tk.StringVar(value="No active histogram filter")
        self.status = tk.StringVar(value="Load a Picasso *_locs.hdf5 file.")
        self.file_label = tk.StringVar(value="No file loaded")
        self.roi_label = tk.StringVar(value="ROI: full corrected map")
        self.last_error_message = ""
        self.last_error_details = ""
        self.hist_filter_bounds: dict[str, tuple[float, float]] = {}
        self.current_hist_indices: pd.Index | None = None
        self.current_hist_mode: str | None = None
        self.hist_filter_enabled: dict[str, tk.BooleanVar] = {}
        self.hist_filter_rows: dict[str, ttk.Frame] = {}
        self.hist_filter_lines: list[Any] = []
        self.dragging_filter_line: int | None = None
        self.hist_motion_cid: int | None = None
        self.hist_press_cid: int | None = None
        self.hist_release_cid: int | None = None
        self.temporal_request_id = 0
        self.origami_result: OrigamiAnalysisResult | None = None
        self.origami_source_points_nm: np.ndarray | None = None
        self.origami_source_render_result: dict[str, Any] | None = None
        self.origami_pick_result: OrigamiPickResult | None = None
        self.origami_loaded_source_label = ""
        self.origami_loaded_source_path: Path | None = None
        self.origami_result_source = ""
        self.origami_result_source_count = 0
        self.origami_result_render_settings: dict[str, Any] | None = None
        self.origami_result_occupancy_threshold = 1
        self.origami_plot_option = tk.StringVar(value="Individual origami gallery")
        self.origami_last_rendered_plot_option = ""
        self.origami_gallery_view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.origami_gallery_home_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.origami_identification_progress = tk.DoubleVar(value=0.0)
        self.origami_identification_progress_text = tk.StringVar(value="Ready to identify origami")
        self.origami_identification_running = False
        self.origami_source = tk.StringVar(value="Corrected localizations")
        self.origami_use_roi = tk.BooleanVar(value=True)
        self.origami_pick_bin_nm = tk.DoubleVar(value=10.0)
        self.origami_connect_distance_nm = tk.DoubleVar(value=35.0)
        self.origami_min_density_contrast = tk.DoubleVar(value=0.30)
        self.origami_min_points = tk.IntVar(value=500)
        self.origami_max_points = tk.IntVar(value=5000)
        self.origami_rows = tk.IntVar(value=3)
        self.origami_columns = tk.IntVar(value=4)
        self.origami_spacing_x_nm = tk.DoubleVar(value=20.0)
        self.origami_spacing_y_nm = tk.DoubleVar(value=20.0)
        self.origami_rectangle_margin_nm = tk.DoubleVar(value=20.0)
        self.origami_min_rectangle_confidence = tk.DoubleVar(value=0.80)
        self.origami_preview_pixel_nm = tk.DoubleVar(value=1.0)
        self.origami_alignment_iterations = tk.IntVar(value=2)
        self.origami_g5m_sigma_min_nm = tk.DoubleVar(value=1.0)
        self.origami_g5m_sigma_max_nm = tk.DoubleVar(value=8.0)
        self.origami_g5m_min_locs = tk.IntVar(value=20)
        self.origami_g5m_bic_patience = tk.IntVar(value=3)
        self.origami_site_radius_nm = tk.DoubleVar(value=7.5)
        self.origami_occupancy_threshold = tk.IntVar(value=1)
        self.origami_overlay_pixel_nm = tk.DoubleVar(value=0.5)
        self.origami_overlay_padding_nm = tk.DoubleVar(value=20.0)
        self.origami_overlay_blur_nm = tk.DoubleVar(value=1.0)
        self.origami_allow_mirror = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar_outer = ttk.Frame(self)
        sidebar_outer.grid(row=0, column=0, sticky="nsew")
        sidebar_outer.columnconfigure(0, weight=1)
        sidebar_outer.rowconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_outer, width=340, highlightthickness=0)
        sidebar_scrollbar = ttk.Scrollbar(sidebar_outer, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")

        sidebar = ttk.Frame(sidebar_canvas, padding=12)
        sidebar.columnconfigure(0, weight=1)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _update_sidebar_scrollregion(_event: tk.Event) -> None:
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _resize_sidebar_window(event: tk.Event) -> None:
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        def _bind_sidebar_mousewheel(_event: tk.Event) -> None:
            sidebar_canvas.bind_all("<MouseWheel>", _sidebar_mousewheel)

        def _unbind_sidebar_mousewheel(_event: tk.Event) -> None:
            sidebar_canvas.unbind_all("<MouseWheel>")

        def _sidebar_mousewheel(event: tk.Event) -> None:
            sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        sidebar.bind("<Configure>", _update_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", _resize_sidebar_window)
        sidebar_canvas.bind("<Enter>", _bind_sidebar_mousewheel)
        sidebar_canvas.bind("<Leave>", _unbind_sidebar_mousewheel)

        ttk.Label(sidebar, text=APP_TITLE, font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Button(sidebar, text="Load Locs File", command=self.load_file).grid(row=1, column=0, sticky="ew")
        ttk.Label(sidebar, textvariable=self.file_label, wraplength=290).grid(row=2, column=0, sticky="ew", pady=(8, 12))

        roi_box = ttk.LabelFrame(sidebar, text="ROI", padding=10)
        roi_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        roi_box.columnconfigure(0, weight=1)
        ttk.Label(roi_box, textvariable=self.roi_label, wraplength=290).grid(row=0, column=0, sticky="ew")
        ttk.Button(roi_box, text="Clear ROI", command=self.clear_roi).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        drift_box = ttk.LabelFrame(sidebar, text="Drift Correction", padding=10)
        drift_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        drift_box.columnconfigure(1, weight=1)
        self._number_row(drift_box, 0, "Pixel size (nm)", self.pixel_size_nm)
        ttk.Label(drift_box, text="Drift method").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(drift_box, textvariable=self.drift_method, state="readonly", values=("none", "rcc", "aim")).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        self._number_row(drift_box, 2, "Segmentation", self.drift_segmentation)
        self._number_row(drift_box, 3, "RCC lattice pitch (nm)", self.rcc_lattice_pitch_nm)
        self._number_row(drift_box, 4, "AIM intersect (nm)", self.aim_intersect_nm)
        self._number_row(drift_box, 5, "AIM ROI (nm)", self.aim_roi_nm)
        ttk.Button(drift_box, text="Apply Drift Correction", command=self.apply_correction).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        render_box = ttk.LabelFrame(sidebar, text="Render Settings", padding=10)
        render_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        render_box.columnconfigure(1, weight=1)
        self._number_row(render_box, 0, "Render pixel (nm)", self.render_disp_px_nm)
        ttk.Label(render_box, text="Render blur").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(render_box, textvariable=self.render_blur_method, state="readonly", values=("smooth", "none", "gaussian", "gaussian_iso", "convolve")).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        self._number_row(render_box, 2, "Min blur (px)", self.min_blur_width)
        self._number_row(render_box, 3, "Min density", self.render_min_density)
        self._number_row(render_box, 4, "Max density", self.render_max_density)
        ttk.Button(render_box, text="Render Raw Map", command=self.show_raw_map).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(render_box, text="Render Corrected Map", command=self.show_current_map).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(render_box, text="Render Linked Map", command=self.color_by_links).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        linking_box = ttk.LabelFrame(sidebar, text="Linking Settings", padding=10)
        linking_box.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        linking_box.columnconfigure(1, weight=1)
        self._number_row(linking_box, 0, "Exposure (ms)", self.exposure_ms)
        self._number_row(linking_box, 1, "Link radius (nm)", self.link_radius_nm)
        self._number_row(linking_box, 2, "Max gap (frames)", self.max_gap_frames)
        ttk.Label(linking_box, text="Link on").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            linking_box,
            textvariable=self.linking_source,
            state="readonly",
            values=("Corrected map", "Raw map"),
        ).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(linking_box, text="Link scope").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(
            linking_box,
            textvariable=self.linking_scope,
            state="readonly",
            values=("Selected ROI", "Whole image"),
        ).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(linking_box, text="Run Linking Analysis", command=self.run_linking_analysis).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        meta_box = ttk.LabelFrame(sidebar, text="File Metadata", padding=10)
        meta_box.grid(row=7, column=0, sticky="nsew", pady=(0, 0))
        sidebar.rowconfigure(7, weight=1)
        self.meta_text = tk.Text(meta_box, width=38, height=10, wrap="word", state="disabled")
        self.meta_text.grid(row=0, column=0, sticky="nsew")
        meta_box.columnconfigure(0, weight=1)
        meta_box.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(main)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        top_bar.columnconfigure(0, weight=1)
        self.error_indicator = tk.Label(
            top_bar,
            text="X",
            bg="#dc2626",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            width=2,
            cursor="hand2",
        )
        self.error_indicator.grid(row=0, column=1, sticky="e")
        self.error_indicator.grid_remove()
        self.error_indicator.bind("<Button-1>", lambda _event: self._show_last_error_details())

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        raw_map_tab = ttk.Frame(self.notebook)
        raw_map_tab.columnconfigure(0, weight=1)
        raw_map_tab.rowconfigure(0, weight=1)
        self.notebook.add(raw_map_tab, text="Raw Map")

        self.raw_map_figure = Figure(figsize=(7, 5), dpi=100)
        self.raw_map_axis = self.raw_map_figure.add_subplot(111)
        self.raw_map_axis.set_title("No raw localization map loaded")
        self.raw_map_axis.set_xlabel("x position (nm)")
        self.raw_map_axis.set_ylabel("y position (nm)")
        self.raw_map_axis.grid(False)
        self.raw_map_axis.set_position(MAP_AXES_RECT)
        self.raw_map_canvas = FigureCanvasTkAgg(self.raw_map_figure, master=raw_map_tab)
        self.raw_map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        raw_map_toolbar_frame = ttk.Frame(raw_map_tab)
        raw_map_toolbar_frame.grid(row=1, column=0, sticky="ew")
        SyncedMapToolbar(self.raw_map_canvas, raw_map_toolbar_frame, self, self.raw_map_axis)

        map_tab = ttk.Frame(self.notebook)
        map_tab.columnconfigure(0, weight=1)
        map_tab.rowconfigure(0, weight=1)
        self.notebook.add(map_tab, text="Corrected Map")

        self.map_figure = Figure(figsize=(7, 5), dpi=100)
        self.map_axis = self.map_figure.add_subplot(111)
        self.map_axis.set_title("No localization map loaded")
        self.map_axis.set_xlabel("x position (nm)")
        self.map_axis.set_ylabel("y position (nm)")
        self.map_axis.grid(False)
        self.map_axis.set_position(MAP_AXES_RECT)
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=map_tab)
        self.map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        map_toolbar_frame = ttk.Frame(map_tab)
        map_toolbar_frame.grid(row=1, column=0, sticky="ew")
        SyncedMapToolbar(self.map_canvas, map_toolbar_frame, self, self.map_axis)

        linked_map_tab = ttk.Frame(self.notebook)
        linked_map_tab.columnconfigure(0, weight=1)
        linked_map_tab.rowconfigure(0, weight=1)
        self.notebook.add(linked_map_tab, text="Linked Map")

        self.linked_map_figure = Figure(figsize=(7, 5), dpi=100)
        self.linked_map_axis = self.linked_map_figure.add_subplot(111)
        self.linked_map_axis.set_title("No linked map rendered")
        self.linked_map_axis.set_xlabel("x position (nm)")
        self.linked_map_axis.set_ylabel("y position (nm)")
        self.linked_map_axis.grid(False)
        self.linked_map_axis.set_position(MAP_AXES_RECT)
        self.linked_map_canvas = FigureCanvasTkAgg(self.linked_map_figure, master=linked_map_tab)
        self.linked_map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        linked_map_toolbar_frame = ttk.Frame(linked_map_tab)
        linked_map_toolbar_frame.grid(row=1, column=0, sticky="ew")
        SyncedMapToolbar(self.linked_map_canvas, linked_map_toolbar_frame, self, self.linked_map_axis)

        filtered_map_tab = ttk.Frame(self.notebook)
        filtered_map_tab.columnconfigure(0, weight=1)
        filtered_map_tab.rowconfigure(0, weight=1)
        self.notebook.add(filtered_map_tab, text="Filtered Map")

        self.filtered_map_figure = Figure(figsize=(7, 5), dpi=100)
        self.filtered_map_axis = self.filtered_map_figure.add_subplot(111)
        self.filtered_map_axis.set_title("No filtered map rendered")
        self.filtered_map_axis.set_xlabel("x position (nm)")
        self.filtered_map_axis.set_ylabel("y position (nm)")
        self.filtered_map_axis.grid(False)
        self.filtered_map_axis.set_position(MAP_AXES_RECT)
        self.filtered_map_canvas = FigureCanvasTkAgg(self.filtered_map_figure, master=filtered_map_tab)
        self.filtered_map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        filtered_map_toolbar_frame = ttk.Frame(filtered_map_tab)
        filtered_map_toolbar_frame.grid(row=1, column=0, sticky="ew")
        SyncedMapToolbar(self.filtered_map_canvas, filtered_map_toolbar_frame, self, self.filtered_map_axis)

        self._connect_map_zoom_sync()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        hist_tab = ttk.Frame(self.notebook)
        hist_tab.columnconfigure(0, weight=1)
        hist_tab.rowconfigure(1, weight=1)
        self.notebook.add(hist_tab, text="Histogram")

        hist_controls = ttk.LabelFrame(hist_tab, text="Histogram Filters", padding=10)
        hist_controls.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 6))
        hist_controls.columnconfigure(1, weight=1)
        hist_controls.columnconfigure(3, weight=1)
        ttk.Label(hist_controls, textvariable=self.roi_label, wraplength=520).grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Button(hist_controls, text="Show Corrected Map", command=self.show_current_map).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(hist_controls, text="Clear ROI", command=self.clear_roi).grid(row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Label(hist_controls, text="Histogram").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.hist_combo = ttk.Combobox(hist_controls, textvariable=self.hist_mode, state="readonly", values=())
        self.hist_combo.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))
        ttk.Label(hist_controls, text="Filter maps").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            hist_controls,
            textvariable=self.hist_filter_scope,
            state="readonly",
            values=("ROI localizations", "Entire image", "Linked events in ROI", "Linked events entire image"),
            width=10,
        ).grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))
        ttk.Label(hist_controls, text="Frame start").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(hist_controls, textvariable=self.hist_frame_start, width=12).grid(row=3, column=1, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(hist_controls, text="Frame end").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(hist_controls, textvariable=self.hist_frame_end, width=12).grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(hist_controls, text="Filtered map source").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            hist_controls,
            textvariable=self.filtered_map_source,
            state="readonly",
            values=("Corrected map", "Raw map", "Linked map"),
        ).grid(row=4, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))
        ttk.Label(hist_controls, text="Bin size").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(hist_controls, textvariable=self.hist_bin_size, width=12).grid(row=4, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Button(hist_controls, text="Plot Histogram", command=self.plot_roi_histogram).grid(row=5, column=0, sticky="ew", pady=(8, 0), padx=(0, 6))
        ttk.Button(hist_controls, text="Apply Histogram Filters", command=self.apply_histogram_filters_to_maps).grid(row=5, column=1, sticky="ew", pady=(8, 0), padx=(0, 12))
        ttk.Button(hist_controls, text="Export Current CSV", command=self.export_csv).grid(row=5, column=2, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(hist_controls, textvariable=self.filter_bounds_label).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.active_filters_frame = ttk.LabelFrame(hist_controls, text="Active Histogram Filters", padding=8)
        self.active_filters_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.active_filters_frame.columnconfigure(0, weight=1)
        self._refresh_filter_list()

        self.hist_figure = Figure(figsize=(7, 5), dpi=100)
        self.hist_axis = self.hist_figure.add_subplot(111)
        self.hist_axis.set_title("No ROI histogram plotted")
        self.hist_axis.set_xlabel("Value")
        self.hist_axis.set_ylabel("Count")
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, master=hist_tab)
        self.hist_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        hist_toolbar_frame = ttk.Frame(hist_tab)
        hist_toolbar_frame.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.hist_canvas, hist_toolbar_frame)

        temporal_tab = ttk.Frame(self.notebook)
        temporal_tab.columnconfigure(0, weight=1)
        temporal_tab.rowconfigure(1, weight=1)
        self.notebook.add(temporal_tab, text="Temporal Metrics")

        temporal_controls = ttk.LabelFrame(temporal_tab, text="Temporal Metric Plot", padding=10)
        temporal_controls.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 6))
        temporal_controls.columnconfigure(1, weight=1)
        temporal_controls.columnconfigure(3, weight=1)
        ttk.Label(temporal_controls, text="Metric").grid(row=0, column=0, sticky="w")
        self.temporal_combo = ttk.Combobox(temporal_controls, textvariable=self.temporal_mode, state="readonly", values=())
        self.temporal_combo.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Checkbutton(temporal_controls, text="Use selected ROI", variable=self.temporal_use_roi).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(temporal_controls, text="Use linked events", variable=self.temporal_use_linked).grid(row=0, column=3, sticky="w")
        ttk.Label(temporal_controls, text="Frame start").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(temporal_controls, textvariable=self.temporal_frame_start, width=12).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(temporal_controls, text="Frame end").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(temporal_controls, textvariable=self.temporal_frame_end, width=12).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self._number_row(temporal_controls, 2, "Window (frames)", self.temporal_window_frames)
        self._number_row(temporal_controls, 3, "Step (frames)", self.temporal_step_frames)
        ttk.Label(temporal_controls, text="Statistic").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(temporal_controls, textvariable=self.temporal_stat, state="readonly", values=("mean", "median", "IQR mean")).grid(row=4, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))
        ttk.Button(temporal_controls, text="Plot Temporal Metric", command=self.plot_temporal_metric).grid(row=4, column=2, columnspan=2, sticky="ew", pady=(8, 0))

        self.temporal_figure = Figure(figsize=(7, 5), dpi=100)
        self.temporal_axis = self.temporal_figure.add_subplot(111)
        self.temporal_axis.set_title("No temporal metric plotted")
        self.temporal_axis.set_xlabel("Frame")
        self.temporal_axis.set_ylabel("Metric")
        self.temporal_canvas = FigureCanvasTkAgg(self.temporal_figure, master=temporal_tab)
        self.temporal_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        temporal_toolbar_frame = ttk.Frame(temporal_tab)
        temporal_toolbar_frame.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.temporal_canvas, temporal_toolbar_frame)

        origami_tab = ttk.Frame(self.notebook)
        origami_tab.columnconfigure(0, weight=1)
        origami_tab.rowconfigure(1, weight=1)
        self.notebook.add(origami_tab, text="Origami Overlay")

        origami_controls = ttk.Frame(origami_tab, padding=4)
        origami_controls.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 6))
        origami_controls.configure(height=330)
        origami_controls.grid_propagate(False)
        origami_controls.rowconfigure(0, weight=1)
        for column in range(4):
            origami_controls.columnconfigure(column, weight=1, uniform="origami_sections")

        def compact_number_row(parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(parent, textvariable=variable, width=8).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)

        scrollable_origami_sections: list[tuple[tk.Widget, Callable[[tk.Event], None]]] = []

        def scrollable_settings_section(
            title: str,
            column: int,
            padx: tuple[int, int] | int,
        ) -> tuple[ttk.LabelFrame, ttk.Frame]:
            section = ttk.LabelFrame(origami_controls, text=title, padding=2)
            section.grid(row=0, column=column, sticky="nsew", padx=padx)
            section.columnconfigure(0, weight=1)
            section.rowconfigure(0, weight=1)
            canvas = tk.Canvas(
                section,
                highlightthickness=0,
                borderwidth=0,
                background=ttk.Style().lookup("TFrame", "background") or self.cget("background"),
            )
            scrollbar = ttk.Scrollbar(section, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            fields = ttk.Frame(canvas, padding=(4, 2, 4, 6))
            fields.columnconfigure(1, weight=1)
            window = canvas.create_window((0, 0), window=fields, anchor="nw")
            fields.bind(
                "<Configure>",
                lambda _event, current=canvas: current.configure(scrollregion=current.bbox("all")),
            )
            canvas.bind(
                "<Configure>",
                lambda event, current=canvas, item=window: current.itemconfigure(item, width=event.width),
            )

            def scroll(event: tk.Event, current: tk.Canvas = canvas) -> None:
                if getattr(event, "num", None) == 4:
                    units = -1
                elif getattr(event, "num", None) == 5:
                    units = 1
                else:
                    delta = int(getattr(event, "delta", 0))
                    units = -1 if delta > 0 else 1 if delta < 0 else 0
                if units:
                    current.yview_scroll(units, "units")

            scrollable_origami_sections.append((section, scroll))
            return section, fields

        source_section, source_fields = scrollable_settings_section("1. Source Data", 0, (0, 3))
        ttk.Label(source_fields, text="Source").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(
            source_fields,
            textvariable=self.origami_source,
            state="readonly",
            values=("Filtered linked events", "Linked events", "Filtered corrected localizations", "Corrected localizations"),
            width=20,
        ).grid(row=0, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Checkbutton(source_fields, text="Use selected ROI", variable=self.origami_use_roi).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(source_fields, text="Reload after changing source, ROI, or filters.", wraplength=230).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 5))
        ttk.Button(source_fields, text="Load Source Data", command=self.load_origami_source_data).grid(row=3, column=0, columnspan=2, sticky="ew")

        identify_section, identify_fields = scrollable_settings_section("2. Identify Origami", 1, 3)

        compact_number_row(identify_fields, 0, "Pick bin (nm)", self.origami_pick_bin_nm)
        compact_number_row(identify_fields, 1, "Connect (nm)", self.origami_connect_distance_nm)
        compact_number_row(identify_fields, 2, "Min density", self.origami_min_density_contrast)
        ttk.Label(identify_fields, text="Point limits").grid(row=3, column=0, sticky="w", pady=2)
        point_limits = ttk.Frame(identify_fields)
        point_limits.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(point_limits, textvariable=self.origami_min_points, width=6).pack(side="left")
        ttk.Label(point_limits, text=" to ").pack(side="left")
        ttk.Entry(point_limits, textvariable=self.origami_max_points, width=6).pack(side="left")
        ttk.Label(identify_fields, text="Grid rows × cols").grid(row=4, column=0, sticky="w", pady=2)
        grid_shape = ttk.Frame(identify_fields)
        grid_shape.grid(row=4, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(grid_shape, textvariable=self.origami_rows, width=5).pack(side="left")
        ttk.Label(grid_shape, text=" x ").pack(side="left")
        ttk.Entry(grid_shape, textvariable=self.origami_columns, width=5).pack(side="left")
        ttk.Label(identify_fields, text="Spacing x/y (nm)").grid(row=5, column=0, sticky="w", pady=2)
        grid_spacing = ttk.Frame(identify_fields)
        grid_spacing.grid(row=5, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(grid_spacing, textvariable=self.origami_spacing_x_nm, width=5).pack(side="left")
        ttk.Label(grid_spacing, text=" / ").pack(side="left")
        ttk.Entry(grid_spacing, textvariable=self.origami_spacing_y_nm, width=5).pack(side="left")
        compact_number_row(identify_fields, 6, "Image margin (nm)", self.origami_rectangle_margin_nm)
        compact_number_row(identify_fields, 7, "Alignment pixel (nm)", self.origami_preview_pixel_nm)
        compact_number_row(identify_fields, 8, "Template passes", self.origami_alignment_iterations)
        compact_number_row(identify_fields, 9, "Min image correlation", self.origami_min_rectangle_confidence)
        self.origami_identify_button = ttk.Button(identify_fields, text="Identify Origami", command=self.identify_origamis)
        self.origami_identify_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Progressbar(
            identify_fields,
            variable=self.origami_identification_progress,
            maximum=100.0,
            mode="determinate",
        ).grid(row=11, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(
            identify_fields,
            textvariable=self.origami_identification_progress_text,
            wraplength=230,
        ).grid(row=12, column=0, columnspan=2, sticky="w")

        overlay_section, overlay_fields = scrollable_settings_section("3. Overlay and Statistics", 2, 3)
        ttk.Checkbutton(overlay_fields, text="Allow mirrors", variable=self.origami_allow_mirror).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(overlay_fields, text="G5M σ min/max").grid(row=1, column=0, sticky="w", pady=2)
        g5m_sigma = ttk.Frame(overlay_fields)
        g5m_sigma.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(g5m_sigma, textvariable=self.origami_g5m_sigma_min_nm, width=5).pack(side="left")
        ttk.Label(g5m_sigma, text=" / ").pack(side="left")
        ttk.Entry(g5m_sigma, textvariable=self.origami_g5m_sigma_max_nm, width=5).pack(side="left")
        compact_number_row(overlay_fields, 2, "G5M min locs", self.origami_g5m_min_locs)
        compact_number_row(overlay_fields, 3, "G5M BIC patience", self.origami_g5m_bic_patience)
        compact_number_row(overlay_fields, 4, "Site radius (nm)", self.origami_site_radius_nm)
        compact_number_row(overlay_fields, 5, "Pixel (nm)", self.origami_overlay_pixel_nm)
        compact_number_row(overlay_fields, 6, "Padding (nm)", self.origami_overlay_padding_nm)
        compact_number_row(overlay_fields, 7, "Blur σ (nm)", self.origami_overlay_blur_nm)
        ttk.Button(overlay_fields, text="Overlay Origami", command=self.overlay_origamis).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(overlay_fields, text="Plot").grid(row=9, column=0, sticky="w", pady=(4, 2))
        ttk.Combobox(
            overlay_fields,
            textvariable=self.origami_plot_option,
            state="readonly",
            values=(
                "Identified origami image matches",
                "Individual origami gallery",
                "Individual Picasso G5M sites",
                "Aligned density",
                "Integrated density per site",
                "Mean site counts",
                "Site occupancy",
                "Occupied-site completeness",
            ),
            width=24,
        ).grid(row=9, column=1, sticky="ew", padx=(6, 0), pady=(4, 2))
        ttk.Button(overlay_fields, text="Render Plot", command=self.render_origami_plot).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )

        export_section, export_fields = scrollable_settings_section("4. Export Results", 3, (3, 0))
        ttk.Label(
            export_fields,
            text="Exports per-origami counts and site-level occupancy statistics as two CSV files.",
            wraplength=210,
        ).grid(row=0, column=0, sticky="nw", pady=(2, 6))
        ttk.Button(export_fields, text="Export Site CSVs", command=self.export_origami_csvs).grid(row=1, column=0, sticky="ew")

        def bind_panel_scroll(widget: tk.Widget, scroll: Callable[[tk.Event], None]) -> None:
            widget.bind("<MouseWheel>", scroll)
            widget.bind("<Button-4>", scroll)
            widget.bind("<Button-5>", scroll)
            for child in widget.winfo_children():
                bind_panel_scroll(child, scroll)

        for scroll_section, scroll_handler in scrollable_origami_sections:
            bind_panel_scroll(scroll_section, scroll_handler)

        self.origami_figure = Figure(figsize=(8, 6), dpi=100)
        self.origami_figure.suptitle("1. Load source   2. Identify   3. Overlay   4. Export")
        self.origami_canvas = FigureCanvasTkAgg(self.origami_figure, master=origami_tab)
        self.origami_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        origami_toolbar_frame = ttk.Frame(origami_tab)
        origami_toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.origami_toolbar = OrigamiToolbar(self.origami_canvas, origami_toolbar_frame, self)

        ttk.Label(main, textvariable=self.status, anchor="w").grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _show_error_indicator(self, message: str, details: str) -> None:
        self.last_error_message = str(message)
        self.last_error_details = str(details)
        if hasattr(self, "error_indicator"):
            self.error_indicator.grid()

    def _hide_error_indicator(self) -> None:
        self.last_error_message = ""
        self.last_error_details = ""
        if hasattr(self, "error_indicator"):
            self.error_indicator.grid_remove()

    def _show_last_error_details(self) -> None:
        if not self.last_error_details:
            return
        messagebox.showerror("Last analysis error", f"{self.last_error_message}\n\n{self.last_error_details}")

    def report_callback_exception(self, exc_type: Any, exc: BaseException, tb: Any) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        self.status.set("Error")
        self._show_error_indicator(str(exc), details)
        messagebox.showerror("Analysis error", f"{exc}\n\n{details}")

    def _number_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)

    def _connect_map_zoom_sync(self) -> None:
        for axis in (self.raw_map_axis, self.map_axis, self.linked_map_axis, self.filtered_map_axis):
            axis.callbacks.connect("xlim_changed", self._sync_map_limits_from)
            axis.callbacks.connect("ylim_changed", self._sync_map_limits_from)
        for axis, canvas in self._map_axis_canvas_pairs():
            canvas.mpl_connect("button_release_event", lambda event, source_axis=axis: self._defer_map_limit_sync(event, source_axis))
            canvas.mpl_connect("scroll_event", lambda event, source_axis=axis: self._defer_map_limit_sync(event, source_axis))

    def _map_axis_canvas_pairs(self) -> tuple[tuple[Any, Any], ...]:
        return (
            (self.raw_map_axis, self.raw_map_canvas),
            (self.map_axis, self.map_canvas),
            (self.linked_map_axis, self.linked_map_canvas),
            (self.filtered_map_axis, self.filtered_map_canvas),
        )

    def _axis_canvas_for_tab(self, tab_index: int) -> tuple[Any, Any] | None:
        if tab_index == RAW_MAP_TAB:
            return self.raw_map_axis, self.raw_map_canvas
        if tab_index == CORRECTED_MAP_TAB:
            return self.map_axis, self.map_canvas
        if tab_index == LINKED_MAP_TAB:
            return self.linked_map_axis, self.linked_map_canvas
        if tab_index == FILTERED_MAP_TAB:
            return self.filtered_map_axis, self.filtered_map_canvas
        return None

    def _current_notebook_tab_index(self) -> int | None:
        try:
            return int(self.notebook.index(self.notebook.select()))
        except Exception:
            return None

    def _current_linked_params(self) -> tuple[float, float, int, str, str]:
        return (
            float(self.exposure_ms.get()),
            float(self.link_radius_nm.get()),
            int(self.max_gap_frames.get()),
            self.linking_source.get(),
            self.linking_scope.get(),
        )

    def _link_source_locs(self) -> pd.DataFrame:
        assert self.loaded is not None
        if self.linking_source.get() == "Raw map":
            return self.loaded.locs.copy()
        assert self.corrected_locs is not None
        return self.corrected_locs.copy()

    def _link_source_label(self) -> str:
        return "raw" if self.linking_source.get() == "Raw map" else "corrected"

    def _linking_uses_roi(self) -> bool:
        return self.linking_scope.get() == "Selected ROI" and self.roi_nm is not None

    def _same_roi(self, left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(left, right))

    def _cached_linked_events_for_scope(self, use_roi: bool, pixelsize: float) -> tuple[pd.DataFrame | None, str | None]:
        if self.linked_locs is None or self.linked_params != self._current_linked_params():
            return None, None
        if use_roi and self.roi_nm is not None:
            if self.linked_roi_nm is None:
                return roi_locs(self.linked_locs, self.roi_nm, pixelsize), "cached full-image linked events filtered to selected ROI"
            if self._same_roi(self.linked_roi_nm, self.roi_nm):
                return self.linked_locs.copy(), "cached selected-ROI linked events"
            return None, None
        if self.linked_roi_nm is None:
            return self.linked_locs.copy(), "cached full-image linked events"
        return None, None

    def _defer_map_limit_sync(self, event: Any, source_axis: Any) -> None:
        if self.syncing_map_limits or self.suspend_map_limit_sync:
            return
        if event.inaxes is not source_axis:
            return
        self.after_idle(lambda axis=source_axis: self._sync_map_limits_from(axis))

    def _on_notebook_tab_changed(self, _event: Any) -> None:
        previous_pair = self._axis_canvas_for_tab(self.active_notebook_tab)
        if previous_pair is not None:
            previous_axis, _previous_canvas = previous_pair
            self._sync_map_limits_from(previous_axis)

        current_tab = self._current_notebook_tab_index()
        if current_tab is None:
            return
        self.active_notebook_tab = current_tab
        current_pair = self._axis_canvas_for_tab(current_tab)
        if current_pair is not None:
            current_axis, current_canvas = current_pair
            self._apply_shared_map_limits(current_axis, current_canvas)

    def _sync_map_limits_from(self, source_axis: Any) -> None:
        if self.syncing_map_limits or self.suspend_map_limit_sync:
            return
        if not getattr(source_axis, "images", None):
            return
        try:
            xlim = tuple(float(value) for value in source_axis.get_xlim())
            ylim = tuple(float(value) for value in source_axis.get_ylim())
        except Exception:
            return
        if not all(np.isfinite(value) for value in (*xlim, *ylim)):
            return

        self.shared_map_limits = (xlim, ylim)
        self.syncing_map_limits = True
        try:
            axis_canvas_pairs = (
                (self.raw_map_axis, self.raw_map_canvas),
                (self.map_axis, self.map_canvas),
                (self.linked_map_axis, self.linked_map_canvas),
                (self.filtered_map_axis, self.filtered_map_canvas),
            )
            for axis, canvas in axis_canvas_pairs:
                if axis is source_axis:
                    continue
                axis.set_xlim(xlim[0], xlim[1], emit=False)
                axis.set_ylim(ylim[0], ylim[1], emit=False)
                canvas.draw_idle()
        finally:
            self.syncing_map_limits = False

    def _apply_shared_map_limits(self, axis: Any, canvas: Any) -> None:
        if self.shared_map_limits is None:
            if not getattr(axis, "images", None):
                return
            try:
                xlim = tuple(float(value) for value in axis.get_xlim())
                ylim = tuple(float(value) for value in axis.get_ylim())
            except Exception:
                return
            if all(np.isfinite(value) for value in (*xlim, *ylim)):
                self.shared_map_limits = (xlim, ylim)
            return
        xlim, ylim = self.shared_map_limits
        if not all(np.isfinite(value) for value in (*xlim, *ylim)):
            return
        self.syncing_map_limits = True
        try:
            axis.set_xlim(xlim[0], xlim[1], emit=False)
            axis.set_ylim(ylim[0], ylim[1], emit=False)
            canvas.draw_idle()
        finally:
            self.syncing_map_limits = False

    def _clear_shared_map_limits(self) -> None:
        self.shared_map_limits = None

    def _file_dialog_initial_dir(self) -> Path:
        candidates: list[Path] = []
        if self.loaded is not None:
            candidates.append(Path(self.loaded.path).parent)
        try:
            recent_text = RECENT_DIR_FILE.read_text(encoding="utf-8").strip()
            if recent_text:
                candidates.append(Path(recent_text))
        except OSError:
            pass
        candidates.extend([DEFAULT_DATA_DIR, Path.home() / "Desktop", Path.home()])
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return Path.home()

    def _remember_file_dialog_dir(self, path: Path) -> None:
        try:
            RECENT_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
            RECENT_DIR_FILE.write_text(str(path.parent), encoding="utf-8")
        except OSError:
            pass

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Picasso localization file",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[("Localization files", "*.hdf5 *.h5"), ("All files", "*.*")],
        )
        if not path:
            return
        self._remember_file_dialog_dir(Path(path))
        self.status.set("Loading localization file...")
        self._run_worker(lambda: ("loaded", read_locs_hdf5(Path(path))))

    def apply_correction(self) -> None:
        if self.loaded is None:
            messagebox.showinfo("No file", "Load a Picasso *_locs.hdf5 file first.")
            return
        self.roi_nm = None
        self._remove_roi_patch()
        self._remove_raw_roi_highlight()
        self._update_roi_label()
        self.status.set("Applying drift correction...")
        self._run_worker(self._correction_worker)

    def show_current_map(self) -> None:
        if self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Apply drift correction first.")
            return
        self.render_viewport_nm = self._shared_map_viewport_nm() or self._current_map_viewport_nm()
        estimate = self._estimate_render_shape(self.render_viewport_nm, float(self.render_disp_px_nm.get()))
        if estimate is not None:
            width_px, height_px, total_px = estimate
            if total_px > MAX_RENDER_PIXELS:
                self.status.set("Render skipped: requested image is too large.")
                messagebox.showwarning(
                    "Render too large",
                    (
                        f"Requested render is {width_px:,} x {height_px:,} pixels "
                        f"({total_px / 1_000_000:.1f} MP).\n\n"
                        "Zoom into a smaller region or increase Render pixel (nm)."
                    ),
                )
                return
        self._clear_map_before_render()
        self.status.set("Rendering corrected map...")
        self._run_worker(self._render_current_corrected_worker)

    def show_raw_map(self, auto_fit: bool = False) -> None:
        if self.loaded is None:
            messagebox.showinfo("No file", "Load a Picasso *_locs.hdf5 file first.")
            return
        render_px_nm = float(self.render_disp_px_nm.get())
        self.raw_render_viewport_nm = self._shared_map_viewport_nm() or self._current_raw_map_viewport_nm()
        estimate = self._estimate_render_shape(self.raw_render_viewport_nm, render_px_nm)
        if estimate is not None:
            width_px, height_px, total_px = estimate
            if total_px > MAX_RENDER_PIXELS:
                if auto_fit:
                    render_px_nm = render_px_nm * math.sqrt(total_px / MAX_RENDER_PIXELS) * 1.05
                    estimate = self._estimate_render_shape(self.raw_render_viewport_nm, render_px_nm)
                    if estimate is not None:
                        width_px, height_px, total_px = estimate
                if total_px <= MAX_RENDER_PIXELS:
                    self.status.set(f"Rendering raw map with auto-fit render pixel {render_px_nm:.3g} nm...")
                else:
                    self.status.set("Raw render skipped: requested image is too large.")
                    messagebox.showwarning(
                        "Render too large",
                        (
                            f"Requested raw render is {width_px:,} x {height_px:,} pixels "
                            f"({total_px / 1_000_000:.1f} MP).\n\n"
                            "Increase Render pixel (nm) before rendering the full raw map."
                        ),
                    )
                    return
            else:
                self.status.set("Rendering raw uncorrected map...")
        else:
            self.status.set("Rendering raw uncorrected map...")
        self._clear_raw_map_before_render()
        self._run_worker(lambda: self._render_raw_worker(render_px_nm))

    def clear_roi(self) -> None:
        self.roi_nm = None
        self._remove_roi_patch()
        self._remove_raw_roi_highlight()
        self._update_roi_label()
        self.map_canvas.draw_idle()
        self.linked_map_canvas.draw_idle()
        self.raw_map_canvas.draw_idle()
        self.status.set("ROI cleared.")

    def _current_map_viewport_nm(self) -> tuple[float, float, float, float] | None:
        if self.loaded is None:
            return None
        if not self.map_axis.images:
            return None
        xlim = self.map_axis.get_xlim()
        ylim = self.map_axis.get_ylim()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        full_width_nm = float(self.loaded.info[0]["Width"]) * pixelsize
        full_height_nm = float(self.loaded.info[0]["Height"]) * pixelsize
        x0 = max(0.0, min(float(xlim[0]), float(xlim[1])))
        x1 = min(full_width_nm, max(float(xlim[0]), float(xlim[1])))
        y0 = max(0.0, min(float(ylim[0]), float(ylim[1])))
        y1 = min(full_height_nm, max(float(ylim[0]), float(ylim[1])))
        if x1 <= x0 or y1 <= y0:
            return None
        if abs(x1 - x0 - full_width_nm) < 1e-6 and abs(y1 - y0 - full_height_nm) < 1e-6:
            return None
        return (x0, x1, y0, y1)

    def _current_raw_map_viewport_nm(self) -> tuple[float, float, float, float] | None:
        if self.loaded is None:
            return None
        if not self.raw_map_axis.images:
            return None
        xlim = self.raw_map_axis.get_xlim()
        ylim = self.raw_map_axis.get_ylim()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        full_width_nm = float(self.loaded.info[0]["Width"]) * pixelsize
        full_height_nm = float(self.loaded.info[0]["Height"]) * pixelsize
        x0 = max(0.0, min(float(xlim[0]), float(xlim[1])))
        x1 = min(full_width_nm, max(float(xlim[0]), float(xlim[1])))
        y0 = max(0.0, min(float(ylim[0]), float(ylim[1])))
        y1 = min(full_height_nm, max(float(ylim[0]), float(ylim[1])))
        if x1 <= x0 or y1 <= y0:
            return None
        if abs(x1 - x0 - full_width_nm) < 1e-6 and abs(y1 - y0 - full_height_nm) < 1e-6:
            return None
        return (x0, x1, y0, y1)

    def _shared_map_viewport_nm(self) -> tuple[float, float, float, float] | None:
        if self.loaded is None or self.shared_map_limits is None:
            return None
        xlim, ylim = self.shared_map_limits
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        full_width_nm = float(self.loaded.info[0]["Width"]) * pixelsize
        full_height_nm = float(self.loaded.info[0]["Height"]) * pixelsize
        x0 = max(0.0, min(float(xlim[0]), float(xlim[1])))
        x1 = min(full_width_nm, max(float(xlim[0]), float(xlim[1])))
        y0 = max(0.0, min(float(ylim[0]), float(ylim[1])))
        y1 = min(full_height_nm, max(float(ylim[0]), float(ylim[1])))
        if x1 <= x0 or y1 <= y0:
            return None
        if abs(x1 - x0 - full_width_nm) < 1e-6 and abs(y1 - y0 - full_height_nm) < 1e-6:
            return None
        return (x0, x1, y0, y1)

    def _estimate_render_shape(self, viewport_nm: tuple[float, float, float, float] | None, disp_px_size_nm: float) -> tuple[int, int, int] | None:
        if self.loaded is None or disp_px_size_nm <= 0:
            return None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        if viewport_nm is None:
            width_nm = float(self.loaded.info[0]["Width"]) * pixelsize
            height_nm = float(self.loaded.info[0]["Height"]) * pixelsize
        else:
            x0, x1, y0, y1 = viewport_nm
            width_nm = abs(x1 - x0)
            height_nm = abs(y1 - y0)
        width_px = max(1, int(math.ceil(width_nm / disp_px_size_nm)))
        height_px = max(1, int(math.ceil(height_nm / disp_px_size_nm)))
        return width_px, height_px, width_px * height_px

    def _clear_map_before_render(self) -> None:
        self.suspend_map_limit_sync = True
        try:
            if self.selector is not None:
                self.selector.set_active(False)
                self.selector = None
            self._remove_map_colorbar()
            self._remove_roi_patch()
            self.map_axis.clear()
            self.map_axis.set_title("Rendering...")
            self.map_axis.set_xlabel("x position (nm)")
            self.map_axis.set_ylabel("y position (nm)")
            self.map_axis.grid(False)
            self._center_map_axis(self.map_axis)
            self.map_canvas.draw_idle()
        finally:
            self.suspend_map_limit_sync = False
        gc.collect()

    def _clear_raw_map_before_render(self) -> None:
        self.suspend_map_limit_sync = True
        try:
            self._remove_raw_map_colorbar()
            self._remove_raw_roi_highlight()
            self.raw_map_axis.clear()
            self.raw_map_axis.set_title("Rendering raw uncorrected map...")
            self.raw_map_axis.set_xlabel("x position (nm)")
            self.raw_map_axis.set_ylabel("y position (nm)")
            self.raw_map_axis.grid(False)
            self._center_map_axis(self.raw_map_axis)
            self.raw_map_canvas.draw_idle()
        finally:
            self.suspend_map_limit_sync = False
        gc.collect()

    def plot_roi_histogram(self) -> None:
        if self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Render a corrected localization map first.")
            return
        try:
            self._custom_hist_bin_size()
        except ValueError as exc:
            messagebox.showerror("Invalid bin size", str(exc))
            return
        self.status.set("Generating ROI histogram from corrected localizations...")
        self._run_worker(self._histogram_worker)

    def apply_histogram_filters_to_maps(self) -> None:
        if self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Apply drift correction first.")
            return
        shared_viewport = self._shared_map_viewport_nm()
        self.render_viewport_nm = shared_viewport or self._current_map_viewport_nm()
        self.raw_render_viewport_nm = shared_viewport or self._current_raw_map_viewport_nm()
        self.status.set("Filtered map render: 0.0% overall (starting).")
        self._run_worker(self._render_filtered_maps_worker)

    def run_linking_analysis(self) -> None:
        if self.loaded is None:
            messagebox.showinfo("No file", "Load a Picasso *_locs.hdf5 file first.")
            return
        if self.linking_source.get() == "Corrected map" and self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Apply drift correction first, or set Link on to Raw map.")
            return
        self.status.set("Linking analysis: 0.0% overall (starting).")
        self._run_worker(self._linking_summary_worker)

    def color_by_links(self) -> None:
        if self.linked_locs is None:
            messagebox.showinfo("No linked localizations", "Run Linking Analysis first, then render the linked map.")
            return
        if self.linked_params != self._current_linked_params():
            messagebox.showinfo(
                "Linked localizations are stale",
                "The cached linked localizations were generated with different linking settings or a different linking source. Run Linking Analysis again before rendering the linked map.",
            )
            return
        self.render_viewport_nm = self._shared_map_viewport_nm() or self._current_map_viewport_nm()
        self.status.set("Rendering linked map from cached collapsed linked events...")
        self._run_worker(self._link_color_worker)

    def plot_temporal_metric(self) -> None:
        if self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Apply drift correction first.")
            return
        self.temporal_request_id += 1
        params = {
            "request_id": self.temporal_request_id,
            "mode": self.temporal_mode.get(),
            "use_roi": bool(self.temporal_use_roi.get()),
            "use_linked": bool(self.temporal_use_linked.get()),
            "frame_start": self.temporal_frame_start.get(),
            "frame_end": self.temporal_frame_end.get(),
            "window": int(self.temporal_window_frames.get()),
            "step": int(self.temporal_step_frames.get()),
            "stat": self.temporal_stat.get(),
        }
        self.status.set(f"Generating temporal metric plot for {params['mode']}...")
        self._run_worker(lambda: self._temporal_metric_worker(params))

    def load_origami_source_data(self) -> None:
        if self.loaded is None or self.corrected_locs is None:
            messagebox.showinfo("No corrected data", "Load a localization file and apply drift correction first.")
            return
        source = self.origami_source.get()
        if "linked" in source.lower() and self.linked_locs is None:
            messagebox.showinfo("No linked events", "Run Linking Analysis before using a linked-event origami source.")
            return
        if "linked" in source.lower() and self.linked_params != self._current_linked_params():
            messagebox.showinfo(
                "Linked events are stale",
                "The linking settings or scope changed. Run Linking Analysis again before picking origamis.",
            )
            return
        params = {
            "source": source,
            "use_roi": bool(self.origami_use_roi.get()),
            "active_filters": list(self._active_map_filter_items()),
            "exposure_ms": float(self.exposure_ms.get()),
            "link_radius_nm": float(self.link_radius_nm.get()),
            "max_gap_frames": int(self.max_gap_frames.get()),
            "source_path": self.loaded.path,
            "render_pixel_nm": float(self.render_disp_px_nm.get()),
            "render_blur_method": self.render_blur_method.get(),
            "render_min_blur_width": float(self.min_blur_width.get()),
            "render_min_density": float(self.render_min_density.get()),
            "render_max_density": float(self.render_max_density.get()),
            "render_viewport_nm": (
                self.roi_nm
                if bool(self.origami_use_roi.get()) and self.roi_nm is not None
                else (self._shared_map_viewport_nm() or self._current_map_viewport_nm())
            ),
        }
        self.status.set("Loading selected source points for origami inspection...")
        self._run_worker(lambda: self._load_origami_source_worker(params))

    def _load_origami_source_worker(self, params: dict[str, Any]) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        source = str(params["source"])
        if "linked" in source.lower():
            assert self.linked_locs is not None
            selected = self.linked_locs.copy()
            source_note = self.linked_scope_name.lower()
        else:
            selected = self.corrected_locs.copy()
            source_note = "whole corrected image"

        if bool(params["use_roi"]) and self.roi_nm is not None:
            selected = roi_locs(selected, self.roi_nm, pixelsize)
            source_note = "selected ROI"

        if source.lower().startswith("filtered") and params["active_filters"]:
            keep = pd.Series(True, index=selected.index, dtype=bool)
            for mode, (left, right) in params["active_filters"]:
                series, _xlabel = localization_series_for_mode(
                    selected,
                    mode,
                    pixelsize,
                    float(params["exposure_ms"]),
                    float(params["link_radius_nm"]),
                    int(params["max_gap_frames"]),
                )
                values = series.to_numpy(dtype=float)
                keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
            selected = selected.loc[keep].copy()

        if selected.empty:
            raise ValueError("The selected source contains no points after ROI and histogram filtering.")
        points_nm = selected[["x", "y"]].to_numpy(dtype=float) * pixelsize
        render_result = render_picasso_map(
            selected,
            self.loaded.info,
            float(params["render_pixel_nm"]),
            str(params["render_blur_method"]),
            float(params["render_min_blur_width"]),
            params["render_viewport_nm"],
        )
        render_result["min_density"] = float(params["render_min_density"])
        render_result["max_density"] = float(params["render_max_density"])
        return "origami_source", {
            "points_nm": points_nm,
            "render_result": render_result,
            "source_label": f"{source} ({source_note})",
            "source_path": params["source_path"],
        }

    def identify_origamis(self) -> None:
        if self.origami_source_points_nm is None or self.origami_loaded_source_path is None:
            messagebox.showinfo("Source not loaded", "Click Load Source Data before identifying origami.")
            return
        try:
            params = {
                "pick_bin_size_nm": float(self.origami_pick_bin_nm.get()),
                "connect_distance_nm": float(self.origami_connect_distance_nm.get()),
                "density_threshold": float(self.origami_min_density_contrast.get()),
                "min_candidate_points": int(self.origami_min_points.get()),
                "max_candidate_points": int(self.origami_max_points.get()),
                "rows": int(self.origami_rows.get()),
                "columns": int(self.origami_columns.get()),
                "spacing_x_nm": float(self.origami_spacing_x_nm.get()),
                "spacing_y_nm": float(self.origami_spacing_y_nm.get()),
                "rectangle_margin_nm": float(self.origami_rectangle_margin_nm.get()),
                "min_rectangle_confidence": float(self.origami_min_rectangle_confidence.get()),
                "alignment_pixel_nm": float(self.origami_preview_pixel_nm.get()),
                "alignment_iterations": int(self.origami_alignment_iterations.get()),
                "source_path": self.origami_loaded_source_path,
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid identification settings", str(exc))
            return
        if not 0.0 <= params["min_rectangle_confidence"] <= 1.0:
            messagebox.showerror("Invalid identification settings", "Minimum image correlation must be between 0 and 1.")
            return
        if params["alignment_pixel_nm"] <= 0 or params["alignment_iterations"] < 1:
            messagebox.showerror("Invalid identification settings", "Alignment pixel must be positive and template passes must be at least 1.")
            return
        points_nm = self.origami_source_points_nm.copy()
        self.origami_identification_running = True
        self.origami_identification_progress.set(0.0)
        self.origami_identification_progress_text.set(f"Starting with {len(points_nm):,} source points...")
        self.origami_identify_button.state(["disabled"])
        self.status.set(f"Identifying whole origami regions in {len(points_nm):,} loaded source points...")
        self._run_worker(lambda: self._identify_origami_worker(points_nm, params))

    def _identify_origami_worker(self, points_nm: np.ndarray, params: dict[str, Any]) -> tuple[str, Any]:
        picks = identify_origami_regions(
            points_nm,
            pick_bin_size_nm=float(params["pick_bin_size_nm"]),
            connect_distance_nm=float(params["connect_distance_nm"]),
            density_threshold=float(params["density_threshold"]),
            min_candidate_points=int(params["min_candidate_points"]),
            max_candidate_points=int(params["max_candidate_points"]),
            rows=int(params["rows"]),
            columns=int(params["columns"]),
            spacing_x_nm=float(params["spacing_x_nm"]),
            spacing_y_nm=float(params["spacing_y_nm"]),
            rectangle_margin_nm=float(params["rectangle_margin_nm"]),
            min_rectangle_confidence=float(params["min_rectangle_confidence"]),
            alignment_pixel_nm=float(params["alignment_pixel_nm"]),
            alignment_iterations=int(params["alignment_iterations"]),
            progress_callback=self._origami_identification_worker_progress,
        )
        return "origami_picks", {"picks": picks, "source_path": params["source_path"]}

    def _origami_identification_worker_progress(self, percent: float, message: str) -> None:
        self.worker_queue.put(("origami_identification_progress", (float(percent), str(message))))

    def _finish_origami_identification_progress(self, message: str | None = None) -> None:
        self.origami_identification_running = False
        self.origami_identify_button.state(["!disabled"])
        if message is not None:
            self.origami_identification_progress_text.set(message)

    def overlay_origamis(self) -> None:
        picks = self.origami_pick_result
        if picks is None:
            messagebox.showinfo("Origami not identified", "Click Identify Origami and inspect the colored boxes first.")
            return
        if picks.accepted_count == 0:
            messagebox.showinfo(
                "No accepted origami",
                "No candidates pass both the point and image-correlation limits. Adjust the settings and run Identify Origami again.",
            )
            return
        try:
            params = {
                "rows": int(self.origami_rows.get()),
                "columns": int(self.origami_columns.get()),
                "spacing_x_nm": float(self.origami_spacing_x_nm.get()),
                "spacing_y_nm": float(self.origami_spacing_y_nm.get()),
                "g5m_sigma_min_nm": float(self.origami_g5m_sigma_min_nm.get()),
                "g5m_sigma_max_nm": float(self.origami_g5m_sigma_max_nm.get()),
                "g5m_min_locs": int(self.origami_g5m_min_locs.get()),
                "g5m_bic_patience": int(self.origami_g5m_bic_patience.get()),
                "site_radius_nm": float(self.origami_site_radius_nm.get()),
                "occupancy_threshold": 1,
                "allow_mirror": bool(self.origami_allow_mirror.get()),
                "overlay_pixel_nm": float(self.origami_overlay_pixel_nm.get()),
                "overlay_padding_nm": float(self.origami_overlay_padding_nm.get()),
                "overlay_blur_nm": float(self.origami_overlay_blur_nm.get()),
                "source_path": self.origami_loaded_source_path,
                "source_label": self.origami_loaded_source_label,
                "source_count": len(self.origami_source_points_nm) if self.origami_source_points_nm is not None else 0,
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid overlay settings", str(exc))
            return
        if (
            params["g5m_sigma_min_nm"] <= 0
            or params["g5m_sigma_max_nm"] < params["g5m_sigma_min_nm"]
            or params["g5m_min_locs"] < 1
            or params["g5m_bic_patience"] < 1
            or params["site_radius_nm"] <= 0
        ):
            messagebox.showerror(
                "Invalid overlay settings",
                "G5M sigma bounds must be positive and ordered; minimum localizations, BIC patience, and site-match radius must be positive.",
            )
            return
        if params["overlay_pixel_nm"] <= 0 or params["overlay_padding_nm"] < 0 or params["overlay_blur_nm"] < 0:
            messagebox.showerror(
                "Invalid overlay settings",
                "Overlay pixel size must be greater than zero; padding and blur cannot be negative.",
            )
            return
        accepted_regions = [region.copy() for region in picks.accepted_aligned_regions]
        accepted_centers = np.asarray(
            [np.median(region, axis=0) for region, accepted in zip(picks.regions, picks.accepted_mask) if bool(accepted)],
            dtype=float,
        )
        rejected_count = len(picks.regions) - picks.accepted_count
        self.status.set(f"Clustering docking sites for {len(accepted_regions)} image-aligned origamis...")
        self._run_worker(lambda: self._overlay_origami_worker(accepted_regions, accepted_centers, rejected_count, params))

    def _overlay_origami_worker(
        self,
        accepted_regions: list[np.ndarray],
        accepted_centers: np.ndarray,
        rejected_count: int,
        params: dict[str, Any],
    ) -> tuple[str, Any]:
        result = align_picked_origamis(
            accepted_regions,
            rows=int(params["rows"]),
            columns=int(params["columns"]),
            spacing_x_nm=float(params["spacing_x_nm"]),
            spacing_y_nm=float(params["spacing_y_nm"]),
            site_radius_nm=float(params["site_radius_nm"]),
            g5m_sigma_min_nm=float(params["g5m_sigma_min_nm"]),
            g5m_sigma_max_nm=float(params["g5m_sigma_max_nm"]),
            g5m_min_locs=int(params["g5m_min_locs"]),
            g5m_max_rounds_without_best_bic=int(params["g5m_bic_patience"]),
            prealigned=True,
            source_centers_nm=accepted_centers,
            allow_mirror=bool(params["allow_mirror"]),
            initially_rejected_count=rejected_count,
            progress_callback=self._worker_status,
        )
        return "origami", {
            "result": result,
            "source": params["source_label"],
            "source_note": "identified preview",
            "source_count": int(params["source_count"]),
            "occupancy_threshold": int(params["occupancy_threshold"]),
            "render_settings": {
                "rows": int(params["rows"]),
                "columns": int(params["columns"]),
                "spacing_x_nm": float(params["spacing_x_nm"]),
                "spacing_y_nm": float(params["spacing_y_nm"]),
                "pixel_size_nm": float(params["overlay_pixel_nm"]),
                "padding_nm": float(params["overlay_padding_nm"]),
                "blur_nm": float(params["overlay_blur_nm"]),
            },
            "source_path": params["source_path"],
        }

    def _correction_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        corrected_locs, drift, label = apply_drift_correction(
            self.loaded.locs,
            self.loaded.info,
            self.drift_method.get(),
            int(self.drift_segmentation.get()),
            float(self.aim_intersect_nm.get()),
            float(self.aim_roi_nm.get()),
            self._worker_status,
            float(self.rcc_lattice_pitch_nm.get()),
        )
        return "correction", {"locs": corrected_locs, "drift": drift, "label": label, "source_path": self.loaded.path}

    def _render_current_corrected_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        map_result = render_picasso_map(
            self.corrected_locs,
            self.loaded.info,
            float(self.render_disp_px_nm.get()),
            self.render_blur_method.get(),
            float(self.min_blur_width.get()),
            self.render_viewport_nm,
        )
        map_result["source_path"] = self.loaded.path
        return "map_only", map_result

    def _render_raw_worker(self, disp_px_size_nm: float | None = None) -> tuple[str, Any]:
        assert self.loaded is not None
        map_result = render_picasso_map(
            self.loaded.locs,
            self.loaded.info,
            float(disp_px_size_nm if disp_px_size_nm is not None else self.render_disp_px_nm.get()),
            self.render_blur_method.get(),
            float(self.min_blur_width.get()),
            self.raw_render_viewport_nm,
        )
        map_result["source_path"] = self.loaded.path
        return "raw_map", map_result

    def _histogram_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        mode = self.hist_mode.get()
        if mode in {"event_length_frames", "event_length_ms", "event_locs", "event_photons"}:
            selected = self._histogram_linked_event_locs()
        else:
            selected = self._histogram_scope_locs()
        if mode == "frame_gap":
            values = frame_gap_values(selected)
            indices = []
            xlabel = "Frames between occupied localization frames"
            occupied_count = int(np.unique(selected["frame"].to_numpy(dtype=int)).size) if not selected.empty and "frame" in selected.columns else 0
        else:
            series, xlabel = localization_series_for_mode(
                selected,
                mode,
                pixelsize,
                float(self.exposure_ms.get()),
                float(self.link_radius_nm.get()),
                int(self.max_gap_frames.get()),
            )
            finite_mask = np.isfinite(series.to_numpy(dtype=float))
            values = series.to_numpy(dtype=float)[finite_mask]
            indices = list(series.index[finite_mask])
            occupied_count = 0
        roi_text = self._histogram_scope_text()
        if mode == "frame_gap":
            title = f"{xlabel}\n{roi_text}, {occupied_count:,} occupied frames, {values.size:,} frame gaps"
        else:
            title = f"{xlabel}\n{roi_text}, {len(selected):,} corrected localizations"
        return "hist", {
            "values": values,
            "indices": indices,
            "mode": mode,
            "xlabel": xlabel,
            "title": title,
            "selected_count": len(selected),
            "scope_text": roi_text,
        }

    def _temporal_metric_worker(self, params: dict[str, Any]) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        mode = str(params["mode"])
        stat = str(params["stat"])
        window = max(1, int(params["window"]))
        step = max(1, int(params["step"]))
        frame_start = str(params["frame_start"])
        frame_end = str(params["frame_end"])
        use_linked = bool(params.get("use_linked", False))
        use_roi = bool(params["use_roi"]) and self.roi_nm is not None
        base_locs = self._link_source_locs() if use_linked else self.corrected_locs.copy()
        if use_roi:
            selected = roi_locs(base_locs, self.roi_nm, pixelsize)
            scope = "selected ROI"
        else:
            selected = base_locs.copy()
            scope = f"full {self._link_source_label()} image" if use_linked else "full corrected image"
        unlinked_selected = selected.copy()
        if use_linked:
            cached, cache_scope = self._cached_linked_events_for_scope(use_roi, pixelsize)
            if cached is not None:
                selected = cached
                self._worker_status(f"Temporal metric plot: using {cache_scope}.")
            else:
                selected = self._filter_locs_by_frame_range(selected, frame_start, frame_end)
                self._worker_status(f"Temporal metric plot: linked-event cache unavailable; rebuilding linked events from {self._link_source_label()} localizations for this plot.")
                selected = linked_events_dataframe(
                    selected,
                    float(self.exposure_ms.get()),
                    pixelsize,
                    float(self.link_radius_nm.get()),
                    int(self.max_gap_frames.get()),
                    progress_callback=lambda processed, total: self._worker_status(
                        f"Temporal metric plot: {min(45.0, 45.0 * processed / max(1, total)):.1f}% overall (building linked events)."
                    ),
                )
            scope = f"linked events, {scope}"
        selected = self._filter_locs_by_frame_range(selected, frame_start, frame_end)
        unlinked_selected = self._filter_locs_by_frame_range(unlinked_selected, frame_start, frame_end)
        frame_text = self._frame_range_text(frame_start, frame_end)
        if frame_text:
            scope = f"{scope}, {frame_text}"
        if selected.empty:
            return "temporal", {"request_id": params["request_id"], "mode": mode, "frames": np.asarray([]), "values": np.asarray([]), "counts": np.asarray([]), "xlabel": "Frame", "ylabel": mode, "title": f"No localizations in {scope}"}

        primary = self._temporal_series_for_locs(
            selected,
            mode,
            stat,
            window,
            step,
            pixelsize,
            progress_start=45.0 if use_linked else 0.0,
            progress_span=45.0 if use_linked else 100.0,
            progress_label="linked events" if use_linked else "windows",
        )
        comparison = None
        if use_linked and not unlinked_selected.empty:
            self._worker_status("Temporal metric plot: computing unlinked comparison trace.")
            comparison = self._temporal_series_for_locs(
                unlinked_selected,
                mode,
                stat,
                window,
                step,
                pixelsize,
                start_frame=int(primary["start_frame"]),
                end_frame=int(primary["end_frame"]),
                progress_start=90.0,
                progress_span=10.0,
                progress_label="unlinked comparison",
            )
        centers = list(primary["frames"])
        values = list(primary["values"])
        q1_values = list(primary["q1"])
        q3_values = list(primary["q3"])
        counts = list(primary["counts"])
        ylabel = str(primary["ylabel"])
        stat_label = "IQR mean" if stat == "IQR mean" else stat.title()
        title = f"{stat_label} {ylabel} vs frame\n{scope}, window={window} frames, step={step} frames"
        self._worker_status("Temporal metric plot: 100.0% overall (done).")
        return "temporal", {
            "request_id": params["request_id"],
            "mode": mode,
            "stat": stat,
            "frames": np.asarray(centers),
            "values": np.asarray(values),
            "q1": np.asarray(q1_values),
            "q3": np.asarray(q3_values),
            "counts": np.asarray(counts),
            "xlabel": "Frame",
            "ylabel": ylabel,
            "title": title,
            "comparison": comparison,
        }

    def _temporal_series_for_locs(
        self,
        selected: pd.DataFrame,
        mode: str,
        stat: str,
        window: int,
        step: int,
        pixelsize: float,
        start_frame: int | None = None,
        end_frame: int | None = None,
        progress_start: float = 0.0,
        progress_span: float = 100.0,
        progress_label: str = "windows",
    ) -> dict[str, Any]:
        frames = selected["frame"].to_numpy(dtype=int)
        if start_frame is None:
            start_frame = int(np.nanmin(frames))
        if end_frame is None:
            end_frame = int(np.nanmax(frames))
        order = np.argsort(frames, kind="mergesort")
        sorted_frames = frames[order]
        sorted_selected = selected.iloc[order]
        centers: list[float] = []
        values: list[float] = []
        q1_values: list[float] = []
        q3_values: list[float] = []
        counts: list[int] = []
        total_windows = max(1, int(math.floor((end_frame - start_frame) / step)) + 1)
        ylabel = f"Localizations per {window}-frame window"
        for window_index, left in enumerate(range(start_frame, end_frame + 1, step), start=1):
            right = left + window - 1
            left_index = int(np.searchsorted(sorted_frames, left, side="left"))
            right_index = int(np.searchsorted(sorted_frames, right, side="right"))
            window_locs = sorted_selected.iloc[left_index:right_index]
            centers.append((left + right) / 2.0)
            if window_locs.empty:
                values.append(np.nan)
                q1_values.append(np.nan)
                q3_values.append(np.nan)
                counts.append(0)
            elif mode == "localizations_per_frame":
                count_value = int(len(window_locs))
                values.append(float(count_value))
                q1_values.append(np.nan)
                q3_values.append(np.nan)
                counts.append(count_value)
            else:
                series, ylabel = localization_series_for_mode(
                    window_locs,
                    mode,
                    pixelsize,
                    float(self.exposure_ms.get()),
                    float(self.link_radius_nm.get()),
                    int(self.max_gap_frames.get()),
                )
                finite = finite_values(series.to_numpy(dtype=float))
                counts.append(int(finite.size))
                if finite.size == 0:
                    values.append(np.nan)
                    q1_values.append(np.nan)
                    q3_values.append(np.nan)
                elif stat == "median":
                    q1_values.append(float(np.quantile(finite, 0.25)))
                    q3_values.append(float(np.quantile(finite, 0.75)))
                    values.append(float(np.median(finite)))
                elif stat == "IQR mean":
                    q1 = float(np.quantile(finite, 0.25))
                    q3 = float(np.quantile(finite, 0.75))
                    central = finite[(finite >= q1) & (finite <= q3)]
                    q1_values.append(q1)
                    q3_values.append(q3)
                    values.append(float(np.mean(central)) if central.size else np.nan)
                else:
                    q1_values.append(float(np.quantile(finite, 0.25)))
                    q3_values.append(float(np.quantile(finite, 0.75)))
                    values.append(float(np.mean(finite)))
            if window_index % 10 == 0 or window_index == total_windows:
                overall = progress_start + progress_span * window_index / total_windows
                self._worker_status(f"Temporal metric plot: {overall:5.1f}% overall ({progress_label}, {window_index}/{total_windows} windows).")
        if mode != "localizations_per_frame" and selected.empty is False:
            ylabel = localization_series_for_mode(
                selected.head(1),
                mode,
                pixelsize,
                float(self.exposure_ms.get()),
                float(self.link_radius_nm.get()),
                int(self.max_gap_frames.get()),
            )[1]
        return {
            "frames": np.asarray(centers),
            "values": np.asarray(values),
            "q1": np.asarray(q1_values),
            "q3": np.asarray(q3_values),
            "counts": np.asarray(counts),
            "ylabel": ylabel,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    def _histogram_scope_locs(self) -> pd.DataFrame:
        assert self.corrected_locs is not None
        assert self.loaded is not None
        scope = self.hist_filter_scope.get()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        use_roi = scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None
        selected = roi_locs(self.corrected_locs, self.roi_nm, pixelsize) if use_roi else self.corrected_locs.copy()
        selected = self._filter_locs_by_frame_range(selected, self.hist_frame_start.get(), self.hist_frame_end.get())
        if scope in {"Linked events in ROI", "Linked events entire image"}:
            return self._histogram_linked_event_locs()
        return selected

    def _histogram_linked_event_locs(self) -> pd.DataFrame:
        assert self.corrected_locs is not None
        assert self.loaded is not None
        scope = self.hist_filter_scope.get()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        use_roi = scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None
        cached, cache_scope = self._cached_linked_events_for_scope(use_roi, pixelsize)
        if cached is not None:
            self._worker_status(f"Histogram: using {cache_scope}.")
            return self._filter_locs_by_frame_range(cached, self.hist_frame_start.get(), self.hist_frame_end.get())

        link_source = self._link_source_locs()
        selected = roi_locs(link_source, self.roi_nm, pixelsize) if use_roi else link_source.copy()
        selected = self._filter_locs_by_frame_range(selected, self.hist_frame_start.get(), self.hist_frame_end.get())
        self._worker_status(f"Histogram: linked-event cache unavailable; rebuilding linked events from {self._link_source_label()} localizations for this plot.")
        return linked_events_dataframe(
                selected,
                float(self.exposure_ms.get()),
                pixelsize,
                float(self.link_radius_nm.get()),
                int(self.max_gap_frames.get()),
                progress_callback=lambda processed, total: self._worker_status(
                    f"Histogram: {100.0 * processed / max(1, total):5.1f}% overall (building linked events)."
                ),
            )

    def _histogram_base_locs_for_raw_mapping(self) -> pd.DataFrame:
        assert self.corrected_locs is not None
        assert self.loaded is not None
        scope = self.hist_filter_scope.get()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        use_roi = scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None
        if use_roi:
            selected = roi_locs(self.corrected_locs, self.roi_nm, pixelsize)
        else:
            selected = self.corrected_locs.copy()
        return self._filter_locs_by_frame_range(selected, self.hist_frame_start.get(), self.hist_frame_end.get())

    def _histogram_scope_text(self) -> str:
        parts = []
        scope = self.hist_filter_scope.get()
        if scope in {"Linked events in ROI", "Linked events entire image"} or self.hist_mode.get() in {"event_length_frames", "event_length_ms", "event_locs", "event_photons"}:
            parts.append("linked events")
        if scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None:
            parts.append("selected ROI")
        else:
            parts.append("full corrected image")
        frame_text = self._frame_range_text(self.hist_frame_start.get(), self.hist_frame_end.get())
        if frame_text:
            parts.append(frame_text)
        return ", ".join(parts)

    def _parse_frame_range(self, start_text: str, end_text: str) -> tuple[int | None, int | None]:
        start = int(start_text.strip()) if start_text.strip() else None
        end = int(end_text.strip()) if end_text.strip() else None
        if start is not None and start < 0:
            raise ValueError("Frame start must be >= 0.")
        if end is not None and end < 0:
            raise ValueError("Frame end must be >= 0.")
        if start is not None and end is not None and end < start:
            raise ValueError("Frame end must be greater than or equal to frame start.")
        return start, end

    def _custom_hist_bin_size(self) -> float | None:
        text = self.hist_bin_size.get().strip()
        if not text:
            return None
        value = float(text)
        if value <= 0:
            raise ValueError("Histogram bin size must be greater than 0, or blank for automatic bins.")
        return value

    def _filter_locs_by_frame_range(self, locs: pd.DataFrame, start_text: str, end_text: str) -> pd.DataFrame:
        start, end = self._parse_frame_range(start_text, end_text)
        if locs.empty:
            return locs.copy()
        frames = locs["frame"].astype(float)
        if start is None:
            start = int(np.nanmin(frames))
        if end is None:
            end = int(np.nanmax(frames))
        mask = pd.Series(True, index=locs.index, dtype=bool)
        mask &= frames >= float(start)
        mask &= frames <= float(end)
        return locs.loc[mask].copy()

    def _frame_range_text(self, start_text: str, end_text: str) -> str:
        start, end = self._parse_frame_range(start_text, end_text)
        if start is None and end is None:
            return "all frames"
        if start is None:
            return f"frames <= {end}"
        if end is None:
            return f"frames >= {start}"
        return f"frames {start}-{end}"

    def _filtered_corrected_locs(self, progress_callback: Any | None = None) -> pd.DataFrame:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        scope = self.hist_filter_scope.get()
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        use_roi = scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None
        selected = roi_locs(self.corrected_locs, self.roi_nm, pixelsize) if use_roi else self.corrected_locs.copy()
        selected = self._filter_locs_by_frame_range(selected, self.hist_frame_start.get(), self.hist_frame_end.get())
        if progress_callback is not None:
            progress_callback(f"Filtered map render: 10.0% overall (selected {len(selected):,} source localizations).")
        active_filters = self._active_map_filter_items()
        if selected.empty or not active_filters:
            if progress_callback is not None:
                progress_callback("Filtered map render: 45.0% overall (no active histogram gates to evaluate).")
            return selected
        keep = pd.Series(True, index=selected.index, dtype=bool)
        for filter_index, (mode, (left, right)) in enumerate(active_filters, start=1):
            series, _xlabel = localization_series_for_mode(
                selected,
                mode,
                pixelsize,
                float(self.exposure_ms.get()),
                float(self.link_radius_nm.get()),
                int(self.max_gap_frames.get()),
            )
            values = series.to_numpy(dtype=float)
            keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
            if progress_callback is not None:
                percent = 10.0 + 35.0 * filter_index / max(1, len(active_filters))
                progress_callback(
                    f"Filtered map render: {percent:5.1f}% overall "
                    f"(evaluated {filter_index}/{len(active_filters)} enabled histogram gates; {int(keep.sum()):,} remain)."
                )
        return selected.loc[keep].copy()

    def _filtered_linked_locs(self, progress_callback: Any | None = None) -> pd.DataFrame:
        assert self.loaded is not None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        scope = self.hist_filter_scope.get()
        use_roi = scope in {"ROI localizations", "Linked events in ROI"} and self.roi_nm is not None
        selected = self._histogram_linked_event_locs()
        if progress_callback is not None:
            progress_callback(f"Filtered map render: 10.0% overall (selected {len(selected):,} linked events).")
        active_filters = self._active_map_filter_items()
        if selected.empty or not active_filters:
            if progress_callback is not None:
                progress_callback("Filtered map render: 45.0% overall (no active histogram gates to evaluate).")
            return selected
        if use_roi:
            selected = roi_locs(selected, self.roi_nm, pixelsize)
        keep = pd.Series(True, index=selected.index, dtype=bool)
        for filter_index, (mode, (left, right)) in enumerate(active_filters, start=1):
            series, _xlabel = localization_series_for_mode(
                selected,
                mode,
                pixelsize,
                float(self.exposure_ms.get()),
                float(self.link_radius_nm.get()),
                int(self.max_gap_frames.get()),
            )
            values = series.to_numpy(dtype=float)
            keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
            if progress_callback is not None:
                percent = 10.0 + 35.0 * filter_index / max(1, len(active_filters))
                progress_callback(
                    f"Filtered map render: {percent:5.1f}% overall "
                    f"(evaluated {filter_index}/{len(active_filters)} enabled histogram gates; {int(keep.sum()):,} linked events remain)."
                )
        return selected.loc[keep].copy()

    def _render_filtered_maps_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        render_px_nm = float(self.render_disp_px_nm.get())
        blur_method = self.render_blur_method.get()
        min_blur_width = float(self.min_blur_width.get())
        min_density = float(self.render_min_density.get())
        max_density = float(self.render_max_density.get())
        source = self.filtered_map_source.get()
        self._worker_status("Filtered map render: 0.0% overall (building localization filter).")
        if source == "Linked map":
            filtered_locs = self._filtered_linked_locs(self._worker_status)
            viewport = self.render_viewport_nm
            source_label = "linked"
        else:
            filtered_corrected = self._filtered_corrected_locs(self._worker_status)
            if source == "Raw map":
                raw_index = filtered_corrected.index.intersection(self.loaded.locs.index)
                filtered_locs = self.loaded.locs.loc[raw_index].copy()
                viewport = self.raw_render_viewport_nm
                source_label = "raw"
            else:
                filtered_locs = filtered_corrected
                viewport = self.render_viewport_nm
                source_label = "corrected"
        common: dict[str, Any] = {
            "source_path": self.loaded.path,
            "filtered_count": int(len(filtered_locs)),
            "map_source": source,
            "source_label": source_label,
            "scope_text": self._histogram_scope_text(),
            "filter_text": self._active_map_filter_text(),
            "min_density": min_density,
            "max_density": max_density,
            "blur_method": blur_method,
            "min_blur_width": min_blur_width,
            "render_px_nm": render_px_nm,
        }
        if filtered_locs.empty:
            self._worker_status("Filtered map render: 100.0% overall (no localizations passed filters).")
            return "filtered_maps", {**common, "map": None}
        self._worker_status(f"Filtered map render: 65.0% overall (rendering {source_label} filtered map from {len(filtered_locs):,} localizations).")
        filtered_map = render_filtered_map_with_settings(
            filtered_locs,
            self.loaded.info,
            render_px_nm,
            blur_method,
            min_blur_width,
            viewport,
        )
        self._worker_status("Filtered map render: 100.0% overall (done).")
        return "filtered_maps", {**common, "map": filtered_map}

    def _linking_summary_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        link_source = self._link_source_locs()
        use_roi = self._linking_uses_roi()
        selected = roi_locs(link_source, self.roi_nm, pixelsize) if use_roi else link_source.copy()
        arrays = df_to_arrays(selected)
        exposure_ms = float(self.exposure_ms.get())
        radius_nm = float(self.link_radius_nm.get())
        max_gap_frames = int(self.max_gap_frames.get())
        source_label = self._link_source_label()
        self._worker_status(f"Linking analysis: 0.0% overall (preparing {source_label} localizations).")

        def summary_progress(processed: int, total: int) -> None:
            total = max(1, int(total))
            phase_percent = min(100.0, 100.0 * float(processed) / float(total))
            overall_percent = 0.5 * phase_percent
            self._worker_status(
                f"Linking analysis: {overall_percent:.1f}% overall "
                f"(event summary pass, {phase_percent:.1f}%)."
            )

        def linked_table_progress(processed: int, total: int) -> None:
            total = max(1, int(total))
            phase_percent = min(100.0, 100.0 * float(processed) / float(total))
            overall_percent = 50.0 + 0.5 * phase_percent
            substep = "assigning localizations" if phase_percent < 50.0 else "building collapsed events"
            self._worker_status(
                f"Linking analysis: {overall_percent:.1f}% overall "
                f"(linked-event table pass, {substep}, {phase_percent:.1f}%)."
            )

        event_arrays = link_binding_events(
            arrays,
            exposure_ms=exposure_ms,
            pixel_size_nm=pixelsize,
            radius_nm=radius_nm,
            max_gap_frames=max_gap_frames,
            progress_callback=summary_progress,
        )
        linked_locs = linked_events_dataframe(
            selected,
            exposure_ms,
            pixelsize,
            radius_nm,
            max_gap_frames,
            progress_callback=linked_table_progress,
        )
        roi_text = f"selected ROI on {source_label} map" if use_roi else f"full {source_label} map"
        return "link_summary", {
            "events": event_arrays,
            "linked_locs": linked_locs,
            "selected_count": len(selected),
            "roi_text": roi_text,
            "roi_nm": self.roi_nm if use_roi else None,
            "linked_params": (exposure_ms, radius_nm, max_gap_frames, self.linking_source.get(), self.linking_scope.get()),
            "link_source": self.linking_source.get(),
            "source_label": source_label,
            "link_scope": self.linking_scope.get(),
        }

    def _link_color_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.linked_locs is not None
        linked_locs = self.linked_locs
        self._worker_status(f"Rendering linked map from cached {len(linked_locs):,} collapsed events...")
        map_result = render_picasso_map(
            linked_locs,
            self.loaded.info,
            float(self.render_disp_px_nm.get()),
            self.render_blur_method.get(),
            float(self.min_blur_width.get()),
            self.render_viewport_nm,
        )
        map_result["linked_count"] = int(len(linked_locs))
        map_result["source_count"] = int(self.linked_source_count)
        map_result["source_path"] = self.loaded.path
        map_result["link_source"] = self.linked_source_name
        map_result["source_label"] = "raw" if self.linked_source_name == "Raw map" else "corrected"
        map_result["roi_text"] = f"full {map_result['source_label']} map" if self.linked_roi_nm is None else f"selected ROI on {map_result['source_label']} map"
        return "link_map", map_result

    def _run_worker(self, func: Any) -> None:
        self._hide_error_indicator()

        def target() -> None:
            try:
                self.worker_queue.put(("result", func()))
            except Exception as exc:
                self.worker_queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=target, daemon=True).start()

    def _worker_status(self, message: str) -> None:
        self.worker_queue.put(("status", message))

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "origami_identification_progress":
                    percent, message = payload
                    self.origami_identification_progress.set(max(0.0, min(100.0, float(percent))))
                    self.origami_identification_progress_text.set(str(message))
                    self.status.set(f"Identify Origami: {float(percent):.0f}% — {message}")
                elif kind == "error":
                    exc, details = payload
                    if self.origami_identification_running:
                        self._finish_origami_identification_progress("Identification stopped because of an error.")
                    self.status.set("Error")
                    self._show_error_indicator(str(exc), str(details))
                    messagebox.showerror("Analysis error", f"{exc}\n\n{details}")
                elif kind == "result":
                    try:
                        result_kind, result_payload = payload
                        if result_kind == "loaded":
                            self._after_load(result_payload)
                        elif result_kind == "correction":
                            if self.loaded is None or result_payload.get("source_path") != self.loaded.path:
                                continue
                            self.corrected_locs = result_payload["locs"]
                            self.linked_locs = None
                            self.linked_source_count = 0
                            self.linked_roi_nm = None
                            self.linked_params = None
                            self.linked_source_name = self.linking_source.get()
                            self.linked_scope_name = self.linking_scope.get()
                            self.origami_result = None
                            self.origami_result_source = ""
                            self.origami_source_points_nm = None
                            self.origami_source_render_result = None
                            self.origami_pick_result = None
                            self.origami_loaded_source_label = ""
                            self.origami_loaded_source_path = None
                            self.drift = result_payload["drift"]
                            self.correction_label = result_payload["label"]
                            self.status.set(f"{self.correction_label} ready. Rendering map with current render settings...")
                            self.show_current_map()
                        elif result_kind == "raw_map":
                            self._plot_raw_map(result_payload)
                        elif result_kind == "map_only":
                            self._plot_map(result_payload)
                        elif result_kind == "filtered_maps":
                            self._plot_filtered_maps(result_payload)
                        elif result_kind == "hist":
                            self._plot_histogram(result_payload)
                        elif result_kind == "link_summary":
                            self.linked_locs = result_payload.get("linked_locs")
                            self.linked_source_count = int(result_payload.get("selected_count", 0))
                            self.linked_roi_nm = result_payload.get("roi_nm")
                            self.linked_params = result_payload.get("linked_params")
                            self.linked_source_name = str(result_payload.get("link_source", self.linking_source.get()))
                            self.linked_scope_name = str(result_payload.get("link_scope", self.linking_scope.get()))
                            self.origami_result = None
                            self.origami_result_source = ""
                            self.origami_source_points_nm = None
                            self.origami_source_render_result = None
                            self.origami_pick_result = None
                            self.origami_loaded_source_label = ""
                            self.origami_loaded_source_path = None
                            self.status.set("Linking analysis complete. Drawing summary plots...")
                            self.update_idletasks()
                            self._plot_linking_summary(result_payload)
                        elif result_kind == "link_map":
                            self._plot_link_map(result_payload)
                        elif result_kind == "temporal":
                            if result_payload.get("request_id") == self.temporal_request_id:
                                self._plot_temporal_metric(result_payload)
                        elif result_kind == "origami_source":
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self.origami_source_points_nm = result_payload["points_nm"]
                                self.origami_source_render_result = dict(result_payload["render_result"])
                                self.origami_loaded_source_label = str(result_payload["source_label"])
                                self.origami_loaded_source_path = result_payload["source_path"]
                                self.origami_pick_result = None
                                self.origami_result = None
                                self._plot_origami_source_data()
                        elif result_kind == "origami_picks":
                            completed_picks: OrigamiPickResult = result_payload["picks"]
                            self.origami_identification_progress.set(100.0)
                            self._finish_origami_identification_progress(
                                f"Complete: {completed_picks.accepted_count}/{len(completed_picks.regions)} image-matched origamis accepted."
                            )
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self.origami_pick_result = completed_picks
                                self.origami_result = None
                                self._plot_identified_origamis()
                        elif result_kind == "origami":
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self._plot_origami_analysis(result_payload)
                    except Exception as exc:
                        details = traceback.format_exc()
                        self.status.set("Error")
                        self._show_error_indicator(str(exc), details)
                        messagebox.showerror("Analysis error", f"{exc}\n\n{details}")
        except queue.Empty:
            pass
        self.after(100, self._poll_worker)

    def _after_load(self, loaded: LoadedData) -> None:
        self.loaded = loaded
        self.corrected_locs = None
        self.linked_locs = None
        self.linked_source_count = 0
        self.linked_roi_nm = None
        self.linked_params = None
        self.linked_source_name = self.linking_source.get()
        self.linked_scope_name = self.linking_scope.get()
        self.drift = None
        self.roi_nm = None
        self.render_viewport_nm = None
        self.raw_render_viewport_nm = None
        self._clear_shared_map_limits()
        self.current_values = None
        self.current_xlabel = "value"
        self.current_hist_indices = None
        self.current_hist_mode = None
        self.origami_result = None
        self.origami_result_source = ""
        self.origami_source_points_nm = None
        self.origami_source_render_result = None
        self.origami_pick_result = None
        self.origami_loaded_source_label = ""
        self.origami_loaded_source_path = None
        self.hist_filter_bounds.clear()
        self.hist_filter_enabled.clear()
        self._refresh_filter_list()
        self._update_filter_bounds_label()
        self.correction_label = "No drift correction"
        self.file_label.set(os.path.basename(loaded.path))
        try:
            self.pixel_size_nm.set(float(loaded.info[0]["Pixelsize"]))
        except Exception:
            pass
        self._update_metadata()
        self._update_hist_options()
        self._update_roi_label()
        self._clear_outputs_for_new_file()
        self.status.set(f"Loaded {loaded.path}. Rendering raw uncorrected map...")
        self.show_raw_map(auto_fit=True)

    def _clear_outputs_for_new_file(self) -> None:
        self.suspend_map_limit_sync = True
        try:
            if self.selector is not None:
                try:
                    self.selector.set_active(False)
                except Exception:
                    pass
                self.selector = None
            if self.linked_selector is not None:
                try:
                    self.linked_selector.set_active(False)
                except Exception:
                    pass
                self.linked_selector = None
            self._remove_raw_map_colorbar()
            self._remove_map_colorbar()
            self._remove_linked_map_colorbar()
            self._remove_filtered_map_colorbar()
            self._remove_raw_roi_highlight()
            self._remove_roi_patch()

            self.raw_map_axis.clear()
            self.raw_map_axis.set_title("Rendering raw uncorrected map...")
            self.raw_map_axis.set_xlabel("x position (nm)")
            self.raw_map_axis.set_ylabel("y position (nm)")
            self.raw_map_axis.grid(False)
            self._center_map_axis(self.raw_map_axis)
            self.raw_map_canvas.draw_idle()

            self.map_axis.clear()
            self.map_axis.set_title("No corrected map\nApply drift correction to render corrected map")
            self.map_axis.set_xlabel("x position (nm)")
            self.map_axis.set_ylabel("y position (nm)")
            self.map_axis.grid(False)
            self._center_map_axis(self.map_axis)
            self.map_canvas.draw_idle()

            self.linked_map_axis.clear()
            self.linked_map_axis.set_title("No linked map rendered")
            self.linked_map_axis.set_xlabel("x position (nm)")
            self.linked_map_axis.set_ylabel("y position (nm)")
            self.linked_map_axis.grid(False)
            self._center_map_axis(self.linked_map_axis)
            self.linked_map_canvas.draw_idle()

            self.filtered_map_axis.clear()
            self.filtered_map_axis.set_title("No filtered map rendered")
            self.filtered_map_axis.set_xlabel("x position (nm)")
            self.filtered_map_axis.set_ylabel("y position (nm)")
            self.filtered_map_axis.grid(False)
            self._center_map_axis(self.filtered_map_axis)
            self.filtered_map_canvas.draw_idle()
        finally:
            self.suspend_map_limit_sync = False

        self.hist_figure.clear()
        self.hist_filter_lines = []
        self.dragging_filter_line = None
        self.hist_axis = self.hist_figure.add_subplot(111)
        self.hist_axis.set_title("No ROI histogram plotted")
        self.hist_axis.set_xlabel("Value")
        self.hist_axis.set_ylabel("Count")
        self.hist_figure.tight_layout()
        self.hist_canvas.draw_idle()
        self.temporal_figure.clear()
        self.temporal_axis = self.temporal_figure.add_subplot(111)
        self.temporal_axis.set_title("No temporal metric plotted")
        self.temporal_axis.set_xlabel("Frame")
        self.temporal_axis.set_ylabel("Metric")
        self.temporal_figure.tight_layout()
        self.temporal_canvas.draw_idle()
        self.origami_figure.clear()
        self.origami_figure.suptitle("1. Load source   2. Identify   3. Overlay   4. Export")
        self.origami_canvas.draw_idle()
        self.notebook.select(RAW_MAP_TAB)

    def _update_metadata(self) -> None:
        assert self.loaded is not None
        lines = [f"{key}: {value}" for key, value in sorted(self.loaded.metadata.items())]
        self.meta_text.configure(state="normal")
        self.meta_text.delete("1.0", "end")
        self.meta_text.insert("1.0", "\n".join(lines))
        self.meta_text.configure(state="disabled")

    def _update_hist_options(self) -> None:
        assert self.loaded is not None
        columns = set(self.loaded.locs.columns)
        options = ["photons", "precision_radial_nm", "lpx_nm", "lpy_nm", "frame", "frame_gap", "localizations_per_frame", "sx", "sy", "bg", "nearest_neighbor_nm", "event_length_frames", "event_length_ms", "event_locs", "event_photons"]
        available: list[str] = []
        for option in options:
            if option in {"precision_radial_nm"} and {"lpx", "lpy"}.issubset(columns):
                available.append(option)
            elif option in {"frame_gap", "localizations_per_frame", "nearest_neighbor_nm", "event_length_frames", "event_length_ms", "event_locs", "event_photons"}:
                available.append(option)
            elif option == "lpx_nm" and "lpx" in columns:
                available.append(option)
            elif option == "lpy_nm" and "lpy" in columns:
                available.append(option)
            elif option in columns:
                available.append(option)
        self.hist_combo.configure(values=available)
        self.hist_mode.set(available[0] if available else "")
        if hasattr(self, "temporal_combo"):
            self.temporal_combo.configure(values=available)
            self.temporal_mode.set("precision_radial_nm" if "precision_radial_nm" in available else (available[0] if available else ""))

    def _plot_raw_map(self, result: dict[str, Any]) -> None:
        if self.loaded is None or result.get("source_path") != self.loaded.path:
            return
        self.suspend_map_limit_sync = True
        try:
            self._remove_raw_map_colorbar()
            self._remove_raw_roi_highlight()
            self.raw_map_axis.clear()
            image = np.asarray(result["image"], dtype=float)
            display_image, density_limits = scale_density_like_picasso(
                image,
                float(self.render_min_density.get()),
                float(self.render_max_density.get()),
            )
            im = self.raw_map_axis.imshow(display_image, extent=result["extent"], origin="lower", cmap="magma", interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
            self.raw_map_colorbar = self._add_fixed_colorbar(
                self.raw_map_figure,
                im,
                f"density contrast ({density_limits[0]:.3g}-{density_limits[1]:.3g} locs/render px)",
            )
            self.raw_map_axis.set_title("Raw uncorrected Picasso render map")
            self.raw_map_axis.set_xlabel("x position (nm)")
            self.raw_map_axis.set_ylabel("y position (nm)")
            self.raw_map_axis.grid(False)
            self._highlight_raw_roi_locs()
            self._center_map_axis(self.raw_map_axis)
        finally:
            self.suspend_map_limit_sync = False
        self._apply_shared_map_limits(self.raw_map_axis, self.raw_map_canvas)
        self.raw_map_canvas.draw_idle()
        self.notebook.select(RAW_MAP_TAB)
        self.status.set(
            f"Rendered raw map with {result['n_rendered']:,} uncorrected localizations. "
            "Choose drift settings and click Apply Drift Correction for the corrected map."
        )

    def _plot_map(self, result: dict[str, Any]) -> None:
        if self.loaded is None or result.get("source_path") != self.loaded.path:
            return
        self.current_values = None
        self.suspend_map_limit_sync = True
        try:
            self._remove_map_colorbar()
            self.map_axis.clear()
            image = np.asarray(result["image"], dtype=float)
            display_image, density_limits = scale_density_like_picasso(
                image,
                float(self.render_min_density.get()),
                float(self.render_max_density.get()),
            )
            im = self.map_axis.imshow(display_image, extent=result["extent"], origin="lower", cmap="magma", interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
            self.map_colorbar = self._add_fixed_colorbar(
                self.map_figure,
                im,
                f"density contrast ({density_limits[0]:.3g}-{density_limits[1]:.3g} locs/render px)",
            )
            self.map_axis.set_title(f"Picasso render map\n{self.correction_label}")
            self.map_axis.set_xlabel("x position (nm)")
            self.map_axis.set_ylabel("y position (nm)")
            self.map_axis.grid(False)
            if self.roi_nm is not None:
                self._draw_roi_patch()
            self._enable_roi_selector()
            self._highlight_raw_roi_locs()
            self._center_map_axis(self.map_axis)
        finally:
            self.suspend_map_limit_sync = False
        self._apply_shared_map_limits(self.map_axis, self.map_canvas)
        self.map_canvas.draw_idle()
        self.notebook.select(CORRECTED_MAP_TAB)
        self.status.set(
            f"Rendered {result['n_rendered']:,} corrected localizations with Picasso render. "
            f"Density limits {density_limits[0]:.4g}-{density_limits[1]:.4g}. "
            f"{'Rendered current zoomed viewport. ' if result.get('viewport_nm') is not None else ''}"
            "Drag on the map to select an ROI."
        )

    def _enable_roi_selector(self) -> None:
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector = None
        self.selector = RectangleSelector(
            self.map_axis,
            lambda eclick, erelease: self._on_roi_select(eclick, erelease, "corrected map"),
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="data",
            interactive=False,
            props={"facecolor": "none", "edgecolor": "#00e5ff", "linewidth": 1.2, "linestyle": "-"},
        )

    def _enable_linked_roi_selector(self) -> None:
        if self.linked_selector is not None:
            self.linked_selector.set_active(False)
            self.linked_selector = None
        self.linked_selector = RectangleSelector(
            self.linked_map_axis,
            lambda eclick, erelease: self._on_roi_select(eclick, erelease, "linked map"),
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="data",
            interactive=False,
            props={"facecolor": "none", "edgecolor": "#00e5ff", "linewidth": 1.2, "linestyle": "-"},
        )

    def _on_roi_select(self, eclick: Any, erelease: Any, source: str = "map") -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        self.roi_nm = (float(eclick.xdata), float(erelease.xdata), float(eclick.ydata), float(erelease.ydata))
        self._update_roi_label()
        self._draw_roi_patch()
        highlighted_count = self._highlight_raw_roi_locs()
        self.map_canvas.draw_idle()
        self.linked_map_canvas.draw_idle()
        self.status.set(
            f"ROI selected on {source}. Highlighted {highlighted_count:,} matching raw localizations. "
            "Plot an ROI histogram to analyze corrected localizations inside it."
        )

    def _update_roi_label(self) -> None:
        if self.roi_nm is None:
            self.roi_label.set("ROI: full corrected map")
            return
        x0, x1, y0, y1 = self.roi_nm
        self.roi_label.set(f"ROI: x {min(x0, x1):.1f}-{max(x0, x1):.1f} nm, y {min(y0, y1):.1f}-{max(y0, y1):.1f} nm")

    def _plot_histogram(self, result: dict[str, Any]) -> None:
        self.current_values = result["values"]
        self.current_xlabel = result["xlabel"]
        self.current_hist_indices = pd.Index(result.get("indices", []))
        self.current_hist_mode = result.get("mode")
        self.hist_figure.clear()
        self.hist_filter_lines = []
        self.hist_axis = self.hist_figure.add_subplot(111)
        values = result["values"]
        if values.size == 0:
            self.hist_axis.set_title("No finite values to plot")
            self.hist_axis.set_xlabel(result["xlabel"])
            self.hist_axis.set_ylabel("Count")
            self.status.set("No finite values in the selected corrected ROI.")
        else:
            plot_values = values
            precision_range = None
            if self.current_hist_mode in {"precision_radial_nm", "lpx_nm", "lpy_nm"}:
                precision_range = robust_precision_display_range(values)
                if precision_range is not None:
                    left, right = precision_range
                    plot_values = values[(values >= left) & (values <= right)]
                    if plot_values.size == 0:
                        plot_values = values
                        precision_range = None
            custom_bin_size = self._custom_hist_bin_size()
            custom_bins = fixed_width_histogram_bins(plot_values, custom_bin_size) if custom_bin_size is not None else None
            bins: int | np.ndarray = custom_bins if custom_bins is not None else automatic_histogram_bins(plot_values)
            if precision_range is not None and custom_bins is None and int(bins) > 1:
                counts, bin_edges, _patches = self.hist_axis.hist(
                    plot_values,
                    bins=bins,
                    range=precision_range,
                    color="#2563eb",
                    edgecolor="white",
                    linewidth=0.4,
                )
            else:
                counts, bin_edges, _patches = self.hist_axis.hist(plot_values, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.4)
            xlim, ylim = histogram_axis_limits_for_values(plot_values, counts, bin_edges, self.current_hist_mode)
            if xlim is not None:
                self.hist_axis.set_xlim(*xlim)
            if ylim is not None:
                self.hist_axis.set_ylim(*ylim)
            stat_values = plot_values if precision_range is not None else values
            median = float(np.median(stat_values))
            mean = float(np.mean(stat_values))
            self.hist_axis.axvline(median, color="#dc2626", linestyle="--", linewidth=1.5, label=f"median {median:.3g}")
            self.hist_axis.axvline(mean, color="#16a34a", linestyle=":", linewidth=1.8, label=f"mean {mean:.3g}")
            self.hist_axis.legend()
            self.hist_axis.set_title(result["title"])
            self.hist_axis.set_xlabel(result["xlabel"])
            self.hist_axis.set_ylabel("Count")
            self._draw_hist_filter_lines()
            self.status.set(
                f"Plotted {values.size:,} values from {result['selected_count']:,} corrected localizations "
                f"({result['scope_text']}). Median={median:.4g}, mean={mean:.4g}"
                f"{f', bin size={custom_bin_size:g}' if custom_bin_size is not None else ''}"
            )
        self.hist_figure.tight_layout()
        self._connect_hist_filter_events()
        self._update_filter_bounds_label()
        self.hist_canvas.draw_idle()
        self.notebook.select(HISTOGRAM_TAB)

    def _draw_hist_filter_lines(self) -> None:
        for line in self.hist_filter_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self.hist_filter_lines = []
        values = getattr(self, "current_values", None)
        mode = self.current_hist_mode
        if self.hist_axis is None or values is None or len(values) == 0 or not mode:
            self.hist_canvas.draw_idle()
            return
        finite = finite_values(np.asarray(values, dtype=float))
        if finite.size == 0:
            self.hist_canvas.draw_idle()
            return
        if mode in self.hist_filter_bounds:
            left, right = self.hist_filter_bounds[mode]
        else:
            if mode in {"precision_radial_nm", "lpx_nm", "lpy_nm"} and finite.size >= 10:
                q25, q75 = np.percentile(finite[finite > 0], [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    data_min = max(0.0, float(q25 - 3.0 * iqr))
                    data_max = float(q75 + 3.0 * iqr)
                else:
                    data_min = float(np.nanmin(finite))
                    data_max = float(np.nanpercentile(finite, 99.0))
            else:
                data_min = float(np.nanmin(finite))
                data_max = float(np.nanmax(finite))
            if data_max > data_min:
                pad = 0.02 * (data_max - data_min)
                left, right = data_min + pad, data_max - pad
            else:
                left, right = data_min, data_max
        left, right = min(left, right), max(left, right)
        self.hist_filter_lines = [
            self.hist_axis.axvline(left, color="#f59e0b", linewidth=2.0, linestyle="-", label="filter lower"),
            self.hist_axis.axvline(right, color="#7c3aed", linewidth=2.0, linestyle="-", label="filter upper"),
        ]
        legend = self.hist_axis.legend()
        if legend is not None:
            legend.set_in_layout(False)
        self.hist_canvas.draw_idle()

    def _connect_hist_filter_events(self) -> None:
        if self.hist_press_cid is None:
            self.hist_press_cid = self.hist_canvas.mpl_connect("button_press_event", self._on_hist_filter_press)
            self.hist_motion_cid = self.hist_canvas.mpl_connect("motion_notify_event", self._on_hist_filter_motion)
            self.hist_release_cid = self.hist_canvas.mpl_connect("button_release_event", self._on_hist_filter_release)

    def _on_hist_filter_press(self, event: Any) -> None:
        if event.inaxes != self.hist_axis or event.xdata is None or not self.hist_filter_lines:
            return
        xlim = self.hist_axis.get_xlim()
        tolerance = abs(float(xlim[1]) - float(xlim[0])) * 0.02
        distances = [abs(float(line.get_xdata()[0]) - float(event.xdata)) for line in self.hist_filter_lines]
        closest = int(np.argmin(distances))
        if distances[closest] <= tolerance:
            self.dragging_filter_line = closest

    def _on_hist_filter_motion(self, event: Any) -> None:
        if self.dragging_filter_line is None or event.inaxes != self.hist_axis or event.xdata is None:
            return
        line = self.hist_filter_lines[self.dragging_filter_line]
        line.set_xdata([float(event.xdata), float(event.xdata)])
        self._update_filter_bounds_label(preview=True)
        self.hist_canvas.draw_idle()

    def _on_hist_filter_release(self, event: Any) -> None:
        if self.dragging_filter_line is None:
            return
        self.dragging_filter_line = None
        mode = self.current_hist_mode
        if mode and len(self.hist_filter_lines) == 2:
            bounds = sorted(float(line.get_xdata()[0]) for line in self.hist_filter_lines)
            self.hist_filter_bounds[mode] = (bounds[0], bounds[1])
            self._ensure_filter_enabled_var(mode).set(True)
            self._refresh_filter_list()
            self._update_filter_bounds_label()
            self.status.set(f"Updated {mode} filter to {bounds[0]:.4g}-{bounds[1]:.4g}. Click Apply Histogram Filters to update maps.")

    def _active_filter_text(self) -> str:
        active = self._active_filter_items()
        if not active:
            return "no active filters"
        return "; ".join(f"{mode}: {left:.4g}-{right:.4g}" for mode, (left, right) in active)

    def _active_map_filter_text(self) -> str:
        active = self._active_map_filter_items()
        skipped = [mode for mode, _bounds in self._active_filter_items() if mode == "frame_gap"]
        if not active and not skipped:
            return "no active filters"
        parts = [f"{mode}: {left:.4g}-{right:.4g}" for mode, (left, right) in active]
        if skipped:
            parts.append("frame_gap ignored for map filtering")
        return "; ".join(parts)

    def _all_filter_text(self) -> str:
        if not self.hist_filter_bounds:
            return "no saved filters"
        parts = []
        for mode, (left, right) in sorted(self.hist_filter_bounds.items()):
            enabled_var = self.hist_filter_enabled.get(mode)
            state = "on" if enabled_var is None or enabled_var.get() else "off"
            parts.append(f"{mode}: {left:.4g}-{right:.4g} ({state})")
        return "; ".join(parts)

    def _active_filter_items(self) -> list[tuple[str, tuple[float, float]]]:
        return [
            (mode, bounds)
            for mode, bounds in sorted(self.hist_filter_bounds.items())
            if self.hist_filter_enabled.get(mode) is None or self.hist_filter_enabled[mode].get()
        ]

    def _active_map_filter_items(self) -> list[tuple[str, tuple[float, float]]]:
        return [(mode, bounds) for mode, bounds in self._active_filter_items() if mode != "frame_gap"]

    def _ensure_filter_enabled_var(self, mode: str) -> tk.BooleanVar:
        if mode not in self.hist_filter_enabled:
            self.hist_filter_enabled[mode] = tk.BooleanVar(value=True)
        return self.hist_filter_enabled[mode]

    def _on_filter_toggle(self, mode: str) -> None:
        self._update_filter_bounds_label()
        state = "enabled" if self._ensure_filter_enabled_var(mode).get() else "disabled"
        self.status.set(f"{mode} filter {state}. Click Apply Histogram Filters to update maps.")

    def _remove_histogram_filter(self, mode: str) -> None:
        self.hist_filter_bounds.pop(mode, None)
        self.hist_filter_enabled.pop(mode, None)
        self._refresh_filter_list()
        self._update_filter_bounds_label()
        self.status.set(f"Removed saved {mode} filter. Re-rendering maps with updated filters...")
        self.apply_histogram_filters_to_maps()

    def _refresh_filter_list(self) -> None:
        frame = getattr(self, "active_filters_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self.hist_filter_rows.clear()
        if not self.hist_filter_bounds:
            ttk.Label(frame, text="No saved histogram filters").grid(row=0, column=0, sticky="w")
            return
        for row, (mode, (left, right)) in enumerate(sorted(self.hist_filter_bounds.items())):
            var = self._ensure_filter_enabled_var(mode)
            ttk.Checkbutton(frame, variable=var, command=lambda m=mode: self._on_filter_toggle(m)).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(frame, text=mode, width=24).grid(row=row, column=1, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(frame, text=f"{left:.4g} to {right:.4g}").grid(row=row, column=2, sticky="w", padx=(0, 8), pady=2)
            ttk.Button(frame, text="Remove", command=lambda m=mode: self._remove_histogram_filter(m)).grid(row=row, column=3, sticky="e", pady=2)
        frame.columnconfigure(2, weight=1)

    def _update_filter_bounds_label(self, preview: bool = False) -> None:
        mode = self.current_hist_mode or self.hist_mode.get()
        if preview and len(self.hist_filter_lines) == 2:
            bounds = sorted(float(line.get_xdata()[0]) for line in self.hist_filter_lines)
            self.filter_bounds_label.set(f"{mode} filter preview: {bounds[0]:.4g}-{bounds[1]:.4g}")
            return
        self.filter_bounds_label.set(f"Filters: {self._all_filter_text()}")

    def _plot_linking_summary(self, result: dict[str, Any]) -> None:
        self.current_values = None
        self.current_xlabel = "value"
        self.current_hist_indices = None
        self.current_hist_mode = None
        self.hist_figure.clear()
        self.hist_filter_lines = []
        event_arrays = result["events"]
        plot_items = [
            ("event_length_frames", "Binding-event length (frames)"),
            ("event_length_ms", "Binding-event length (ms)"),
            ("event_locs", "Localizations per linked event"),
            ("event_photons", "Photons per linked event"),
        ]
        axes = self.hist_figure.subplots(2, 2)
        for axis, (key, label) in zip(axes.ravel(), plot_items):
            values = finite_values(event_arrays[key])
            if values.size == 0:
                axis.set_title(f"{label}\nno events")
                axis.set_xlabel(label)
                axis.set_ylabel("Count")
                continue
            bins = automatic_histogram_bins(values)
            counts, bin_edges, _patches = axis.hist(values, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.4)
            xlim, ylim = histogram_axis_limits(counts, bin_edges)
            if xlim is not None:
                axis.set_xlim(*xlim)
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.axvline(float(np.median(values)), color="#dc2626", linestyle="--", linewidth=1.2)
            axis.set_title(label)
            axis.set_xlabel(label)
            axis.set_ylabel("Count")
        self.hist_figure.suptitle(
            f"Linking analysis: {result['roi_text']}, {result['selected_count']:,} corrected localizations",
            fontsize=11,
        )
        self.hist_figure.tight_layout()
        self.hist_canvas.draw_idle()
        n_events = len(event_arrays["event_length_frames"])
        self.status.set(f"Linked {result['selected_count']:,} corrected localizations into {n_events:,} events.")

    def _plot_temporal_metric(self, result: dict[str, Any]) -> None:
        self.temporal_figure.clear()
        self.temporal_axis = self.temporal_figure.add_subplot(111)
        frames = np.asarray(result["frames"], dtype=float)
        values = np.asarray(result["values"], dtype=float)
        q1 = np.asarray(result.get("q1", np.asarray([])), dtype=float)
        q3 = np.asarray(result.get("q3", np.asarray([])), dtype=float)
        comparison = result.get("comparison")
        finite = np.isfinite(frames) & np.isfinite(values)
        if frames.size == 0 or not np.any(finite):
            self.temporal_axis.set_title(result["title"])
            self.temporal_axis.set_xlabel(result.get("xlabel", "Frame"))
            self.temporal_axis.set_ylabel(result.get("ylabel", "Metric"))
            self.status.set("No finite temporal metric values to plot.")
        else:
            if q1.size == frames.size and q3.size == frames.size:
                band_finite = np.isfinite(frames) & np.isfinite(q1) & np.isfinite(q3)
                if np.any(band_finite):
                    self.temporal_axis.fill_between(frames[band_finite], q1[band_finite], q3[band_finite], color="#60a5fa", alpha=0.22, linewidth=0, label="Q1-Q3")
            line_label = "IQR mean" if result.get("stat") == "IQR mean" else str(result.get("stat", "metric"))
            primary_label = f"{line_label} linked events" if comparison is not None else line_label
            self.temporal_axis.plot(frames[finite], values[finite], color="#2563eb", linewidth=1.5, marker="o", markersize=2.5, label=primary_label)
            comparison_values_for_limits: list[np.ndarray] = [values[finite]]
            if comparison is not None:
                comparison_frames = np.asarray(comparison.get("frames", np.asarray([])), dtype=float)
                comparison_values = np.asarray(comparison.get("values", np.asarray([])), dtype=float)
                comparison_finite = np.isfinite(comparison_frames) & np.isfinite(comparison_values)
                if comparison_frames.size and np.any(comparison_finite):
                    comparison_values_for_limits.append(comparison_values[comparison_finite])
                    self.temporal_axis.plot(
                        comparison_frames[comparison_finite],
                        comparison_values[comparison_finite],
                        color="#f97316",
                        linewidth=1.4,
                        marker="s",
                        markersize=2.2,
                        alpha=0.9,
                        label=f"{line_label} unlinked localizations",
                    )
            self.temporal_axis.set_title(result["title"])
            self.temporal_axis.set_xlabel(result.get("xlabel", "Frame"))
            self.temporal_axis.set_ylabel(result.get("ylabel", "Metric"))
            if result.get("mode") in {"precision_radial_nm", "lpx_nm", "lpy_nm"}:
                precision_ylim = robust_precision_axis_limits(np.concatenate(comparison_values_for_limits))
                if precision_ylim is not None:
                    self.temporal_axis.set_ylim(*precision_ylim)
            self.temporal_axis.grid(True, alpha=0.25)
            legend = self.temporal_axis.legend(loc="best")
            if legend is not None:
                legend.set_in_layout(False)
            self.status.set(f"Plotted {int(np.count_nonzero(finite)):,} temporal windows for {result.get('mode', result.get('ylabel', 'metric'))}.")
        self.temporal_figure.tight_layout()
        self.temporal_canvas.draw_idle()
        self.notebook.select(TEMPORAL_TAB)

    def _draw_origami_source_density(
        self,
        axis: Any,
        points_nm: np.ndarray,
        picks: OrigamiPickResult | None = None,
    ) -> dict[str, object]:
        cached_render = self.origami_source_render_result
        if cached_render is not None:
            raw_image = np.asarray(cached_render["image"], dtype=float)
            contrast, density_limits = scale_density_like_picasso(
                raw_image,
                float(cached_render["min_density"]),
                float(cached_render["max_density"]),
            )
            extent = tuple(float(value) for value in cached_render["extent"])
            pixel_x = (extent[1] - extent[0]) / max(1, raw_image.shape[1])
            pixel_y = (extent[3] - extent[2]) / max(1, raw_image.shape[0])
            preview: dict[str, object] = {
                "contrast": contrast,
                "extent": extent,
                "effective_pixel_x_nm": float(pixel_x),
                "effective_pixel_y_nm": float(pixel_y),
                "blur_method": str(cached_render["blur_method"]),
                "density_limits": density_limits,
            }
            colorbar_label = (
                f"density contrast ({density_limits[0]:.3g}-{density_limits[1]:.3g} locs/render px)"
            )
        else:
            preview = render_localization_preview(
                points_nm,
                pixel_size_nm=max(float(self.origami_preview_pixel_nm.get()), 0.1),
                blur_nm=1.0,
            )
            contrast = np.asarray(preview["contrast"], dtype=float)
            extent = tuple(float(value) for value in preview["extent"])
            colorbar_label = "density contrast (0–1)"
        image = axis.imshow(
            contrast,
            extent=extent,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_position(ORIGAMI_SOURCE_AXES_RECT)
        axis.set_anchor("C")
        colorbar_axis = self.origami_figure.add_axes(ORIGAMI_SOURCE_COLORBAR_RECT)
        colorbar = self.origami_figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label)
        picker_contrast = picks.density_contrast if picks is not None else None
        if (
            picks is not None
            and picker_contrast is not None
            and picker_contrast.size
            and np.nanmin(picker_contrast) <= picks.density_threshold <= np.nanmax(picker_contrast)
        ):
            x_min, x_max, y_min, y_max = picks.density_extent_nm
            x_centers = np.linspace(x_min, x_max, picker_contrast.shape[0], endpoint=False) + (x_max - x_min) / (2.0 * picker_contrast.shape[0])
            y_centers = np.linspace(y_min, y_max, picker_contrast.shape[1], endpoint=False) + (y_max - y_min) / (2.0 * picker_contrast.shape[1])
            axis.contour(
                x_centers,
                y_centers,
                picker_contrast.T,
                levels=[picks.density_threshold],
                colors=["#22d3ee"],
                linewidths=0.7,
                alpha=0.75,
            )
        axis.set_xlabel("x position (nm)")
        axis.set_ylabel("y position (nm)")
        axis.grid(False)
        return preview

    def _plot_origami_source_data(self) -> None:
        points = self.origami_source_points_nm
        if points is None or len(points) == 0:
            return
        self.origami_gallery_view_limits = None
        self.origami_gallery_home_limits = None
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        preview = self._draw_origami_source_density(axis, points)
        pixel_x = float(preview["effective_pixel_x_nm"])
        pixel_y = float(preview["effective_pixel_y_nm"])
        axis.set_title(
            f"Loaded source data: {self.origami_loaded_source_label}\n"
            f"{len(points):,} source points; Picasso render pixel {pixel_x:.3g} × {pixel_y:.3g} nm; "
            f"blur={preview.get('blur_method', 'smooth')}"
        )
        self.origami_canvas.draw_idle()
        self.origami_last_rendered_plot_option = "Loaded source data"
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(f"Loaded and displayed {len(points):,} source points. Tune identification settings, then click Identify Origami.")

    def _plot_identified_origamis(self) -> None:
        points = self.origami_source_points_nm
        picks = self.origami_pick_result
        if points is None or picks is None:
            return
        if self.origami_result is None:
            self.origami_gallery_view_limits = None
            self.origami_gallery_home_limits = None
        self.origami_plot_option.set("Identified origami image matches")
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        axis.set_anchor("C")
        preview = self._draw_origami_source_density(axis, points, picks)
        colors = matplotlib.colormaps.get_cmap("tab20")
        accepted_indices = np.flatnonzero(picks.accepted_mask)
        accepted_display_index = 0
        for region_index in range(len(picks.regions)):
            accepted = bool(picks.accepted_mask[region_index])
            if accepted:
                accepted_display_index += 1
            corners = picks.rectangle_corners_nm[region_index]
            color = colors((accepted_display_index - 1) % 20 / 19.0) if accepted else "#9ca3af"
            rectangle = matplotlib.patches.Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.8 if accepted else 1.2,
                linestyle="-" if accepted else "--",
                alpha=1.0 if accepted else 0.8,
            )
            axis.add_patch(rectangle)
            if len(picks.regions) <= 150:
                label_corner = corners[int(np.argmax(corners[:, 1]))]
                prefix = f"A{accepted_display_index}" if accepted else f"R{region_index + 1}"
                annotation = axis.text(
                    label_corner[0],
                    label_corner[1],
                    f"{prefix}: n={picks.point_counts[region_index]:,}; {picks.rectangle_angles_deg[region_index]:.1f}°; "
                    f"corr={picks.rectangle_confidence[region_index]:.2f}",
                    color=color,
                    fontsize=7,
                    va="bottom",
                    ha="left",
                    clip_on=True,
                )
                annotation.set_in_layout(False)
        if len(picks.point_counts):
            count_summary = (
                f"region points min/median/max = {int(np.min(picks.point_counts)):,}/"
                f"{int(np.median(picks.point_counts)):,}/{int(np.max(picks.point_counts)):,}"
            )
        else:
            count_summary = "no connected regions"
        if len(picks.rectangle_confidence):
            confidence_summary = (
                f"image correlation min/median/max = {np.min(picks.rectangle_confidence):.2f}/"
                f"{np.median(picks.rectangle_confidence):.2f}/{np.max(picks.rectangle_confidence):.2f}"
            )
        else:
            confidence_summary = "no image matches"
        axis.set_title(
            f"Identified origami: {picks.accepted_count}/{len(picks.regions)} accepted "
            f"(image correlation ≥ {float(self.origami_min_rectangle_confidence.get()):g}; solid accepted, dashed rejected)\n"
            f"{count_summary}; {confidence_summary}\n"
            f"footprint {picks.rectangle_width_nm:g} × {picks.rectangle_height_nm:g} nm; alignment pixel "
            f"{picks.alignment_pixel_nm:.3g} nm; display pixel "
            f"{float(preview['effective_pixel_x_nm']):.3g} × {float(preview['effective_pixel_y_nm']):.3g} nm; "
            f"pick bin {float(self.origami_pick_bin_nm.get()):g} nm; density ≥ {picks.density_threshold:.3g}"
        )
        self.origami_canvas.draw_idle()
        self.origami_last_rendered_plot_option = "Identified origami image matches"
        self.notebook.select(ORIGAMI_TAB)
        if picks.accepted_count:
            self.status.set(
                f"Outlined {picks.accepted_count} accepted origamis from {len(picks.regions)} connected regions. "
                "Solid footprints are accepted; gray dashed footprints are rejected. Tune the settings and rerun Identify Origami, or click Overlay Origami."
            )
        else:
            self.status.set(
                f"Found {len(picks.regions)} connected regions, but none pass both point and image-correlation limits; "
                f"{count_summary}; {confidence_summary}. Adjust point limits, minimum confidence, or identification settings and rerun."
            )

    def _plot_origami_analysis(self, payload: dict[str, Any]) -> None:
        result: OrigamiAnalysisResult = payload["result"]
        threshold = int(payload["occupancy_threshold"])
        self.origami_result = result
        self.origami_result_source = str(payload["source"])
        self.origami_result_source_count = int(payload["source_count"])
        self.origami_result_render_settings = dict(payload["render_settings"])
        self.origami_result_occupancy_threshold = threshold
        self.origami_gallery_view_limits = None
        self.origami_gallery_home_limits = None
        self.origami_last_rendered_plot_option = ""
        if self.origami_plot_option.get() == "Identified origami image matches":
            self.origami_plot_option.set("Individual origami gallery")

        self.render_origami_plot()
        self.status.set(
            f"Overlaid all {result.origami_count} identified origamis and assigned "
            f"{sum(len(sites) for sites in result.cluster_site_indices)} docking-site clusters; median alignment RMS "
            f"{np.median(result.alignment_rms_nm):.2f} nm and median grid match "
            f"{100.0 * np.median(result.grid_match_fraction):.1f}%."
        )

    def render_origami_plot(self) -> None:
        option = self.origami_plot_option.get()
        if option == "Identified origami image matches":
            if self.origami_pick_result is None or self.origami_source_points_nm is None:
                messagebox.showinfo("No identified origami", "Run Identify Origami before rendering the rectangle preview.")
                return
            self._plot_identified_origamis()
            return

        result = self.origami_result
        render_settings = self.origami_result_render_settings
        if result is None or render_settings is None:
            messagebox.showinfo("No origami overlay", "Run Overlay Origami before rendering a result plot.")
            return

        gallery_options = {"Individual origami gallery", "Individual Picasso G5M sites"}
        if self.origami_last_rendered_plot_option in gallery_options and self.origami_figure.axes:
            previous_axis = self.origami_figure.axes[0]
            previous_xlim = tuple(float(value) for value in previous_axis.get_xlim())
            previous_ylim = tuple(float(value) for value in previous_axis.get_ylim())
            if all(np.isfinite(previous_xlim)) and all(np.isfinite(previous_ylim)):
                self.origami_gallery_view_limits = (previous_xlim, previous_ylim)

        self.origami_figure.clear()
        fixed_colorbar_options = {
            "Aligned density",
            "Integrated density per site",
            "Mean site counts",
            "Site occupancy",
        }
        if option in fixed_colorbar_options:
            self.origami_figure.set_layout_engine("none")
        else:
            self.origami_figure.set_layout_engine("constrained", w_pad=6 / 72, h_pad=6 / 72)
        axis = self.origami_figure.subplots(1, 1)
        grid = result.grid_points_nm

        if option == "Individual origami gallery":
            self._plot_individual_origami_gallery(axis, result, render_settings)
        elif option == "Individual Picasso G5M sites":
            self._plot_individual_origami_clusters(axis, result, render_settings)
        elif option == "Aligned density":
            density_orientations = [
                orientation
                for points in result.aligned_points
                for orientation in (points, -points)
            ] if result.symmetrized_180 else result.aligned_points
            overlay_render = render_aligned_origami_density(density_orientations, **render_settings)
            overlay_image = np.asarray(overlay_render["image"], dtype=float)
            overlay_extent = tuple(float(value) for value in overlay_render["extent"])
            image = axis.imshow(
                overlay_image,
                extent=overlay_extent,
                origin="lower",
                cmap="magma",
                aspect="equal",
                interpolation="none",
            )
            self._add_centered_origami_result_colorbar(axis, image, "Mean source points / origami / bin")
            axis.scatter(grid[:, 0], grid[:, 1], s=55, facecolors="none", edgecolors="#22d3ee", linewidths=1.2)
            effective_x = float(overlay_render["effective_pixel_x_nm"])
            effective_y = float(overlay_render["effective_pixel_y_nm"])
            pixel_text = f"{effective_x:.3g} nm/px" if math.isclose(effective_x, effective_y) else f"{effective_x:.3g} × {effective_y:.3g} nm/px"
            orientation_count = 2 if result.symmetrized_180 else 1
            symmetry_text = "; 0°/180° equal-weight average" if result.symmetrized_180 else ""
            axis.set_title(
                f"Aligned density ({result.origami_count} origamis)\n"
                f"{pixel_text}; Gaussian σ={float(overlay_render['blur_nm']):.3g} nm; "
                f"{int(overlay_render['rendered_point_count'] / orientation_count):,}/"
                f"{int(overlay_render['total_point_count'] / orientation_count):,} physical source points in view"
                f"{symmetry_text}"
            )
            axis.set_xlabel("aligned x (nm)")
            axis.set_ylabel("aligned y (nm)")
        elif option == "Integrated density per site":
            density_orientations = [
                orientation
                for points in result.aligned_points
                for orientation in (points, -points)
            ] if result.symmetrized_180 else result.aligned_points
            overlay_render = render_aligned_origami_density(density_orientations, **render_settings)
            integrated_counts = integrate_rendered_density_at_sites(
                overlay_render,
                grid,
                result.site_match_radius_nm,
            )
            values = integrated_counts.reshape(result.rows, result.columns)
            self._plot_origami_site_heatmap(
                axis,
                values,
                result,
                "magma",
                "Mean rendered source points inside site radius",
                (
                    f"Integrated aligned density within {result.site_match_radius_nm:g} nm of each site\n"
                    f"pixel {float(overlay_render['effective_pixel_x_nm']):.3g} × "
                    f"{float(overlay_render['effective_pixel_y_nm']):.3g} nm; "
                    f"Gaussian σ={float(overlay_render['blur_nm']):.3g} nm; includes G5M-unmatched points"
                ),
            )
        elif option == "Mean site counts":
            values = np.mean(result.site_counts, axis=0).reshape(result.rows, result.columns)
            self._plot_origami_site_heatmap(
                axis,
                values,
                result,
                "viridis",
                "Mean source-point count",
                "Mean count at each expected docking site (0°/180° average)",
            )
        elif option == "Site occupancy":
            values = (100.0 * np.mean(result.site_occupancy, axis=0)).reshape(result.rows, result.columns)
            self._plot_origami_site_heatmap(
                axis,
                values,
                result,
                "YlGn",
                "Occupied origamis (%)",
                "Site occupancy (0°/180° equal-weight cluster presence)",
                suffix="%",
                vmin=0.0,
                vmax=100.0,
            )
        elif option == "Occupied-site completeness":
            occupied_per_origami = np.sum(result.site_occupancy, axis=1)
            bin_step = 0.5 if result.symmetrized_180 else 1.0
            edges = np.arange(-bin_step / 2.0, result.rows * result.columns + bin_step, bin_step)
            axis.hist(occupied_per_origami, bins=edges, color="#2563eb", edgecolor="white")
            axis.set_xticks(np.arange(0, result.rows * result.columns + bin_step, bin_step))
            axis.set_xlabel("occupied sites per origami (0°/180° average)")
            axis.set_ylabel("origami count")
            axis.set_title(
                "Per-origami completeness (accepted docking-site clusters)\n"
                f"median RMS {np.median(result.alignment_rms_nm):.2f} nm; median grid match "
                f"{100.0 * np.median(result.grid_match_fraction):.1f}%; {result.rejected_candidate_count} rejected"
            )
            axis.grid(True, axis="y", alpha=0.25)
        else:
            axis.text(0.5, 0.5, f"Unknown plot option: {option}", ha="center", va="center", transform=axis.transAxes)

        if option in gallery_options:
            # Capture the complete data limits before restoring a shared zoom.
            # Matplotlib's toolbar history is reset when this figure is rebuilt,
            # so OrigamiToolbar.home uses this explicit, stable target.
            self.origami_gallery_home_limits = (
                tuple(float(value) for value in axis.get_xlim()),
                tuple(float(value) for value in axis.get_ylim()),
            )
            if self.origami_gallery_view_limits is not None:
                axis.set_xlim(*self.origami_gallery_view_limits[0])
                axis.set_ylim(*self.origami_gallery_view_limits[1])

        self.origami_figure.suptitle(
            f"Origami overlay — {self.origami_result_source}; {self.origami_result_source_count:,} source points",
            fontsize=12,
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = option
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(f"Rendered {option.lower()} for {result.origami_count} origamis.")

    def _plot_origami_site_heatmap(
        self,
        axis: Any,
        values: np.ndarray,
        result: OrigamiAnalysisResult,
        cmap_name: str,
        colorbar_label: str,
        title: str,
        *,
        suffix: str = "",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        image = axis.imshow(values, origin="upper", cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="equal")
        self._add_centered_origami_result_colorbar(axis, image, colorbar_label)
        axis.set_title(title)
        axis.set_xlabel("column")
        axis.set_ylabel("row")
        axis.set_xticks(np.arange(result.columns), labels=np.arange(1, result.columns + 1))
        axis.set_yticks(np.arange(result.rows), labels=np.arange(1, result.rows + 1))
        colormap = matplotlib.colormaps[cmap_name]
        for row in range(result.rows):
            for column in range(result.columns):
                value = float(values[row, column])
                red, green, blue, _alpha = colormap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column,
                    row,
                    f"{value:.1f}{suffix}",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.55 else "white",
                )

    def _add_centered_origami_result_colorbar(self, axis: Any, image: Any, label: str) -> Any:
        """Keep a result plot centered while placing its colorbar independently."""
        axis.set_position(ORIGAMI_RESULT_AXES_RECT)
        axis.set_anchor("C")
        colorbar_axis = self.origami_figure.add_axes(ORIGAMI_RESULT_COLORBAR_RECT)
        colorbar = self.origami_figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(label)
        return colorbar

    def _plot_individual_origami_gallery(
        self,
        axis: Any,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> None:
        settings = dict(render_settings)
        field_width_nm, field_height_nm, preview_pixel_nm, tile_width, tile_height, columns, rows, gap = (
            self._origami_gallery_geometry(result, settings)
        )
        settings["pixel_size_nm"] = preview_pixel_nm
        rendered = [
            render_aligned_origami_density(
                [points],
                **settings,
            )
            for points in result.aligned_points
        ]
        gallery = np.full((rows * tile_height + (rows - 1) * gap, columns * tile_width + (columns - 1) * gap), np.nan)
        tile_origins: list[tuple[int, int]] = []
        for index, tile_render in enumerate(rendered):
            tile = np.asarray(tile_render["image"], dtype=float)
            # render_aligned_origami_density stores increasing physical y from
            # the first row to the last (the convention used with
            # imshow(origin="lower")).  Gallery rows, including the G5M
            # gallery below, use screen coordinates with row zero at the top.
            # Convert the density tile once here so both individual views show
            # the exact same cached pose instead of opposite y conventions.
            tile = np.flipud(tile)
            peak = float(np.nanmax(tile)) if tile.size else 0.0
            if peak > 0:
                tile = tile / peak
            row, column = divmod(index, columns)
            y0 = row * (tile_height + gap)
            x0 = column * (tile_width + gap)
            gallery[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
            tile_origins.append((x0, y0))

        colormap = matplotlib.colormaps["magma"].copy()
        colormap.set_bad("white")
        axis.imshow(gallery, origin="upper", cmap=colormap, vmin=0.0, vmax=1.0, interpolation="nearest")
        x_min, x_max, y_min, y_max = (float(value) for value in rendered[0]["extent"])
        grid_x = (result.grid_points_nm[:, 0] - x_min) * tile_width / (x_max - x_min)
        grid_y = (y_max - result.grid_points_nm[:, 1]) * tile_height / (y_max - y_min)
        expected_x: list[float] = []
        expected_y: list[float] = []
        for index, ((x0, y0), points) in enumerate(zip(tile_origins, result.aligned_points), start=1):
            expected_x.extend((x0 + grid_x).tolist())
            expected_y.extend((y0 + grid_y).tolist())
            label = axis.text(
                x0 + 2,
                y0 + 2,
                f"#{index}  n={result.source_point_counts[index - 1]:,}",
                color="white",
                fontsize=6,
                va="top",
                ha="left",
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.5},
            )
            label.set_clip_path(axis.patch)
            label.set_in_layout(False)
        axis.scatter(expected_x, expected_y, s=9, facecolors="none", edgecolors="#22d3ee", linewidths=0.5)
        axis.set_title(
            f"Every individual aligned origami ({result.origami_count})\n"
            "One cached orientation per physical origami—the exact coordinates fitted by G5M; "
            "tiles are independently brightness-normalized; cyan circles show expected sites"
        )
        axis.set_axis_off()

    def _origami_gallery_geometry(
        self,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> tuple[float, float, float, int, int, int, int, int]:
        """Return the single shared tile geometry used by both individual views."""
        width_nm = max(
            float(render_settings["spacing_x_nm"]),
            (int(render_settings["columns"]) - 1) * float(render_settings["spacing_x_nm"]),
        ) + 2.0 * float(render_settings["padding_nm"])
        height_nm = max(
            float(render_settings["spacing_y_nm"]),
            (int(render_settings["rows"]) - 1) * float(render_settings["spacing_y_nm"]),
        ) + 2.0 * float(render_settings["padding_nm"])
        preview_pixel_nm = max(
            float(render_settings["pixel_size_nm"]),
            width_nm / 100.0,
            height_nm / 100.0,
        )
        tile_width = max(1, int(np.ceil(width_nm / preview_pixel_nm)))
        tile_height = max(1, int(np.ceil(height_nm / preview_pixel_nm)))
        columns = max(1, int(np.ceil(np.sqrt(result.origami_count * tile_height / tile_width))))
        rows = int(np.ceil(result.origami_count / columns))
        return width_nm, height_nm, preview_pixel_nm, tile_width, tile_height, columns, rows, 3

    def _plot_individual_origami_clusters(
        self,
        axis: Any,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> None:
        settings = dict(render_settings)
        width_nm, height_nm, _preview_pixel_nm, tile_width, tile_height, columns, rows, gap = (
            self._origami_gallery_geometry(result, settings)
        )
        gallery = np.ones(
            (rows * tile_height + (rows - 1) * gap, columns * tile_width + (columns - 1) * gap, 3),
            dtype=float,
        )
        x_min, x_max = -width_nm / 2.0, width_nm / 2.0
        y_min, y_max = -height_nm / 2.0, height_nm / 2.0
        site_count = result.rows * result.columns
        site_colors = matplotlib.colormaps["tab20"](np.linspace(0.0, 1.0, max(site_count, 2)))[:, :3]
        expected_x: list[float] = []
        expected_y: list[float] = []
        center_x: list[float] = []
        center_y: list[float] = []
        center_colors: list[np.ndarray] = []

        def to_tile_pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            inside = (
                (points[:, 0] >= x_min)
                & (points[:, 0] < x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] < y_max)
            )
            x_pixels = np.floor((points[:, 0] - x_min) * tile_width / width_nm).astype(int)
            y_pixels = np.floor((y_max - points[:, 1]) * tile_height / height_nm).astype(int)
            return x_pixels, y_pixels, inside

        for index, (points, labels, centers, sites) in enumerate(
            zip(result.aligned_points, result.cluster_labels, result.cluster_centers_nm, result.cluster_site_indices)
        ):
            row, column = divmod(index, columns)
            y0 = row * (tile_height + gap)
            x0 = column * (tile_width + gap)
            tile = np.zeros((tile_height, tile_width, 3), dtype=float)
            orientations = [(points, centers, sites)]
            for oriented_points, oriented_centers, oriented_sites in orientations:
                x_pixels, y_pixels, inside = to_tile_pixels(oriented_points)

                noise = inside & (labels < 0)
                if np.any(noise):
                    flat = y_pixels[noise] * tile_width + x_pixels[noise]
                    density = np.bincount(flat, minlength=tile_height * tile_width).reshape(tile_height, tile_width)
                    peak = float(np.max(density))
                    if peak > 0:
                        tile += (0.09 * np.sqrt(density / peak))[..., None]

                for cluster_label, site_index in enumerate(oriented_sites):
                    members = inside & (labels == cluster_label)
                    if not np.any(members):
                        continue
                    flat = y_pixels[members] * tile_width + x_pixels[members]
                    density = np.bincount(flat, minlength=tile_height * tile_width).reshape(tile_height, tile_width)
                    peak = float(np.max(density))
                    if peak > 0:
                        intensity = np.sqrt(density / peak)[..., None]
                        tile = np.maximum(tile, intensity * site_colors[int(site_index)])

                if len(oriented_centers):
                    cluster_x, cluster_y, cluster_inside = to_tile_pixels(oriented_centers)
                    center_x.extend((x0 + cluster_x[cluster_inside]).tolist())
                    center_y.extend((y0 + cluster_y[cluster_inside]).tolist())
                    center_colors.extend(site_colors[oriented_sites[cluster_inside]])

            gallery[y0 : y0 + tile_height, x0 : x0 + tile_width] = np.clip(tile, 0.0, 1.0)
            grid_x, grid_y, _grid_inside = to_tile_pixels(result.grid_points_nm)
            expected_x.extend((x0 + grid_x).tolist())
            expected_y.extend((y0 + grid_y).tolist())
            label = axis.text(
                x0 + 2,
                y0 + 2,
                (
                    f"#{index + 1}  n={result.source_point_counts[index]:,}; {len(sites)} matched clusters"
                ),
                color="white",
                fontsize=6,
                va="top",
                ha="left",
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 0.5},
            )
            label.set_clip_path(axis.patch)
            label.set_in_layout(False)

        axis.imshow(gallery, origin="upper", interpolation="nearest")
        axis.scatter(expected_x, expected_y, s=8, facecolors="none", edgecolors="#67e8f9", linewidths=0.45)
        if center_x:
            axis.scatter(center_x, center_y, s=9, marker="x", c=np.asarray(center_colors), linewidths=0.7)
        axis.set_title(
            f"Picasso G5M docking-site components for every individual origami ({result.origami_count})\n"
            f"G5M σ {result.g5m_sigma_min_nm:g}–{result.g5m_sigma_max_nm:g} nm, "
            f"minimum {result.g5m_min_locs} locs, BIC patience {result.g5m_max_rounds_without_best_bic}, "
            f"site match {result.site_match_radius_nm:g} nm\n"
            "Single cached orientation used for the G5M fit; color = assigned site; "
            "cyan circle = expected site; × = matched G5M component center"
        )
        axis.set_axis_off()

    def _plot_filtered_maps(self, result: dict[str, Any]) -> None:
        if self.loaded is None or result.get("source_path") != self.loaded.path:
            return
        filtered = result.get("map")
        if filtered is None:
            self.suspend_map_limit_sync = True
            try:
                self._remove_filtered_map_colorbar()
                self.filtered_map_axis.clear()
                self.filtered_map_axis.set_title(f"No {result.get('source_label', 'selected')} localizations pass active histogram filters")
                self.filtered_map_axis.set_xlabel("x position (nm)")
                self.filtered_map_axis.set_ylabel("y position (nm)")
                self.filtered_map_axis.grid(False)
                self._center_map_axis(self.filtered_map_axis)
            finally:
                self.suspend_map_limit_sync = False
            self._apply_shared_map_limits(self.filtered_map_axis, self.filtered_map_canvas)
            self.filtered_map_canvas.draw_idle()
            self.notebook.select(FILTERED_MAP_TAB)
            self.status.set("No localizations pass the active histogram filters.")
            return

        min_density = float(result.get("min_density", self.render_min_density.get()))
        max_density = float(result.get("max_density", self.render_max_density.get()))
        self.suspend_map_limit_sync = True
        try:
            self._remove_filtered_map_colorbar()
            self.filtered_map_axis.clear()
            filtered_image = np.asarray(filtered["image"], dtype=float)
            filtered_display, filtered_limits = scale_density_like_picasso(
                filtered_image,
                min_density,
                max_density,
            )
            im = self.filtered_map_axis.imshow(filtered_display, extent=filtered["extent"], origin="lower", cmap="magma", interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
            self.filtered_map_colorbar = self._add_fixed_colorbar(self.filtered_map_figure, im, f"filtered density ({filtered_limits[0]:.3g}-{filtered_limits[1]:.3g})")
            self.filtered_map_axis.set_title(f"Filtered {result.get('source_label', 'map')} map\n{result['filter_text']}")
            self.filtered_map_axis.set_xlabel("x position (nm)")
            self.filtered_map_axis.set_ylabel("y position (nm)")
            self.filtered_map_axis.grid(False)
            if self.roi_nm is not None:
                self._draw_roi_patch()
            self._center_map_axis(self.filtered_map_axis)
        finally:
            self.suspend_map_limit_sync = False
        self._apply_shared_map_limits(self.filtered_map_axis, self.filtered_map_canvas)
        self.filtered_map_canvas.draw_idle()
        self.notebook.select(FILTERED_MAP_TAB)
        self.status.set(
            f"Filtered {result.get('source_label', 'map')} map rendered: {result['filtered_count']:,} localizations "
            f"from {result['scope_text']} using source={result.get('map_source', 'unknown')}, blur={result.get('blur_method', 'unknown')}, "
            f"render pixel={float(result.get('render_px_nm', 0.0)):.4g} nm."
        )

    def _plot_link_map(self, result: dict[str, Any]) -> None:
        if self.loaded is None or result.get("source_path") != self.loaded.path:
            return
        self.current_values = None
        self.suspend_map_limit_sync = True
        try:
            self._remove_linked_map_colorbar()
            self.linked_map_axis.clear()
            image = np.asarray(result["image"], dtype=float)
            display_image, density_limits = scale_density_like_picasso(
                image,
                float(self.render_min_density.get()),
                float(self.render_max_density.get()),
            )
            im = self.linked_map_axis.imshow(display_image, extent=result["extent"], origin="lower", cmap="magma", interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
            self.linked_map_colorbar = self._add_fixed_colorbar(
                self.linked_map_figure,
                im,
                f"density contrast ({density_limits[0]:.3g}-{density_limits[1]:.3g} events/render px)",
            )
            source_label = str(result.get("source_label", "corrected"))
            subtitle = self.correction_label if source_label == "corrected" else "Raw localization coordinates"
            self.linked_map_axis.set_title(f"Linked-event render map ({source_label})\n{subtitle}")
            self.linked_map_axis.set_xlabel("x position (nm)")
            self.linked_map_axis.set_ylabel("y position (nm)")
            self.linked_map_axis.grid(False)
            if self.roi_nm is not None:
                self._draw_roi_patch()
            self._enable_linked_roi_selector()
            self._center_map_axis(self.linked_map_axis)
        finally:
            self.suspend_map_limit_sync = False
        self._apply_shared_map_limits(self.linked_map_axis, self.linked_map_canvas)
        self.linked_map_canvas.draw_idle()
        self.notebook.select(LINKED_MAP_TAB)
        roi_text = str(result.get("roi_text", "linked-source"))
        source_label = str(result.get("source_label", "corrected"))
        self.status.set(
            f"Rendered {result['linked_count']:,} collapsed linked events from {result['source_count']:,} {source_label} "
            f"{roi_text} localizations. Density limits {density_limits[0]:.4g}-{density_limits[1]:.4g}."
        )

    def _highlight_raw_roi_locs(self) -> int:
        self._remove_raw_roi_highlight()
        if self.loaded is None or self.corrected_locs is None or self.roi_nm is None:
            self.raw_map_canvas.draw_idle()
            return 0
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        selected = roi_locs(self.corrected_locs, self.roi_nm, pixelsize)
        if selected.empty:
            self.raw_map_canvas.draw_idle()
            return 0
        selected_index = selected.index.intersection(self.loaded.locs.index)
        if selected_index.empty:
            self.raw_map_canvas.draw_idle()
            return 0
        raw_selected = self.loaded.locs.loc[selected_index]
        x_nm = np.asarray(raw_selected["x"], dtype=float) * pixelsize
        y_nm = np.asarray(raw_selected["y"], dtype=float) * pixelsize
        self.raw_roi_highlight = self.raw_map_axis.scatter(
            x_nm,
            y_nm,
            s=8,
            c="#00e5ff",
            alpha=0.75,
            linewidths=0,
            label="corrected ROI localizations",
            zorder=8,
        )
        legend = self.raw_map_axis.legend(loc="upper right")
        if legend is not None:
            legend.set_in_layout(False)
        self.raw_map_canvas.draw_idle()
        return int(len(raw_selected))

    def _remove_map_colorbar(self) -> None:
        if self.map_colorbar is not None:
            self.map_colorbar.remove()
            self.map_colorbar = None
        self._center_map_axis(self.map_axis)

    def _remove_raw_map_colorbar(self) -> None:
        if self.raw_map_colorbar is not None:
            self.raw_map_colorbar.remove()
            self.raw_map_colorbar = None
        self._center_map_axis(self.raw_map_axis)

    def _remove_linked_map_colorbar(self) -> None:
        if self.linked_map_colorbar is not None:
            self.linked_map_colorbar.remove()
            self.linked_map_colorbar = None
        self._center_map_axis(self.linked_map_axis)

    def _remove_filtered_map_colorbar(self) -> None:
        if self.filtered_map_colorbar is not None:
            self.filtered_map_colorbar.remove()
            self.filtered_map_colorbar = None
        self._center_map_axis(self.filtered_map_axis)

    def _center_map_axis(self, axis: Any) -> None:
        axis.set_position(MAP_AXES_RECT)
        axis.set_anchor("C")

    def _add_fixed_colorbar(self, figure: Figure, mappable: Any, label: str) -> Any:
        cax = figure.add_axes(MAP_COLORBAR_RECT)
        colorbar = figure.colorbar(mappable, cax=cax)
        colorbar.set_label(label)
        return colorbar

    def _remove_raw_roi_highlight(self) -> None:
        if self.raw_roi_highlight is not None:
            try:
                self.raw_roi_highlight.remove()
            except (ValueError, NotImplementedError):
                pass
            self.raw_roi_highlight = None
        legend = self.raw_map_axis.get_legend()
        if legend is not None:
            try:
                legend.remove()
            except (ValueError, NotImplementedError):
                pass

    def _draw_roi_patch(self) -> None:
        self._remove_roi_patch()
        if self.roi_nm is None:
            return
        x0, x1, y0, y1 = self.roi_nm
        patch_args = {
            "xy": (min(x0, x1), min(y0, y1)),
            "width": abs(x1 - x0),
            "height": abs(y1 - y0),
            "fill": False,
            "edgecolor": "#00e5ff",
            "linewidth": 2.0,
            "linestyle": "-",
            "zorder": 10,
        }
        if self.map_axis.images:
            self.roi_patch = matplotlib.patches.Rectangle(**patch_args)
            self.map_axis.add_patch(self.roi_patch)
        if self.linked_map_axis.images:
            self.linked_roi_patch = matplotlib.patches.Rectangle(**patch_args)
            self.linked_map_axis.add_patch(self.linked_roi_patch)
        if self.filtered_map_axis.images:
            self.filtered_roi_patch = matplotlib.patches.Rectangle(**patch_args)
            self.filtered_map_axis.add_patch(self.filtered_roi_patch)

    def _remove_roi_patch(self) -> None:
        if self.roi_patch is not None:
            try:
                self.roi_patch.remove()
            except (ValueError, NotImplementedError):
                pass
            self.roi_patch = None
        if self.linked_roi_patch is not None:
            try:
                self.linked_roi_patch.remove()
            except (ValueError, NotImplementedError):
                pass
            self.linked_roi_patch = None
        if self.filtered_roi_patch is not None:
            try:
                self.filtered_roi_patch.remove()
            except (ValueError, NotImplementedError):
                pass
            self.filtered_roi_patch = None

    def export_csv(self) -> None:
        values = getattr(self, "current_values", None)
        xlabel = getattr(self, "current_xlabel", "value")
        if values is None or len(values) == 0:
            messagebox.showinfo("No data", "Plot an ROI histogram before exporting.")
            return
        path = filedialog.asksaveasfilename(title="Export current values", defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile=f"{self.hist_mode.get()}_roi_values.csv")
        if not path:
            return
        np.savetxt(path, values, delimiter=",", header=xlabel.replace(",", " "), comments="")
        self.status.set(f"Exported {len(values):,} values to {path}")

    def export_origami_csvs(self) -> None:
        result = self.origami_result
        if result is None or result.origami_count == 0:
            messagebox.showinfo("No origami overlay", "Run Overlay Origami before exporting.")
            return
        path_text = filedialog.asksaveasfilename(
            title="Export per-origami site counts",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="origami_site_counts.csv",
        )
        if not path_text:
            return
        path = Path(path_text)
        per_origami: dict[str, Any] = {
            "origami_id": np.arange(1, result.origami_count + 1),
            "center_x_nm": result.centers_nm[:, 0],
            "center_y_nm": result.centers_nm[:, 1],
            "source_point_count": result.source_point_counts,
            "alignment_rms_nm": result.alignment_rms_nm,
            "grid_match_fraction": result.grid_match_fraction,
            "accepted_cluster_count": np.asarray([len(sites) for sites in result.cluster_site_indices], dtype=int),
            "occupied_site_count": np.sum(result.site_occupancy, axis=1),
            "orientation_averaging": "0_and_180_degrees_equal_weight",
        }
        summary_rows: list[dict[str, Any]] = []
        for site_index, point in enumerate(result.grid_points_nm):
            row = site_index // result.columns + 1
            column = site_index % result.columns + 1
            label = f"site_r{row}_c{column}"
            counts = result.site_counts[:, site_index]
            occupancy = result.site_occupancy[:, site_index]
            per_origami[f"{label}_count"] = counts
            per_origami[f"{label}_occupancy_weight"] = occupancy
            summary_rows.append(
                {
                    "site": label,
                    "row": row,
                    "column": column,
                    "aligned_x_nm": point[0],
                    "aligned_y_nm": point[1],
                    "mean_count": float(np.mean(counts)),
                    "median_count": float(np.median(counts)),
                    "standard_deviation": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
                    "occupancy_definition": "mean of cluster presence at the 0-degree and 180-degree site counterparts",
                    "orientation_averaging": "0_and_180_degrees_equal_weight",
                    "clustering_method": "Picasso G5M",
                    "g5m_sigma_min_nm": result.g5m_sigma_min_nm,
                    "g5m_sigma_max_nm": result.g5m_sigma_max_nm,
                    "g5m_minimum_localizations": result.g5m_min_locs,
                    "g5m_bic_patience": result.g5m_max_rounds_without_best_bic,
                    "site_match_radius_nm": result.site_match_radius_nm,
                    "occupied_origami_count": float(np.sum(occupancy)),
                    "occupancy_fraction": float(np.mean(occupancy)),
                }
            )
        per_origami_path = path
        summary_path = path.with_name(f"{path.stem}_site_summary.csv")
        pd.DataFrame(per_origami).to_csv(per_origami_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        self._remember_file_dialog_dir(path)
        self.status.set(f"Exported per-origami counts to {per_origami_path.name} and site statistics to {summary_path.name}.")


def main() -> None:
    app = PaintAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
