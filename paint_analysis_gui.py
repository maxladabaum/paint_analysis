from __future__ import annotations

import math
import json
import hashlib
import os
import queue
import re
import sys
import threading
import traceback
import gc
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import h5py
import matplotlib
import contourpy
from matplotlib import image as matplotlib_image
import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image as PillowImage
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path as MatplotlibPath
from matplotlib.widgets import RectangleSelector

from origami_analysis import (
    OrigamiAnalysisResult,
    OrigamiPickResult,
    align_picked_origamis,
    bidirectional_template_classification_scores,
    classify_template_candidates,
    concatenate_origami_pick_results,
    custom_template_physical_axes,
    custom_template_site_points,
    density_map_for_origami_picking,
    digital_group_template_evidence,
    digital_group_localization_responsibilities,
    direct_digital_group_localization_evidence,
    identify_origami_regions,
    integrate_rendered_density_at_sites,
    ideal_grid_points,
    lattice_count_template_probability_agreement,
    origami_gallery_indices,
    origami_gallery_page,
    render_aligned_origami_density,
    render_localization_preview,
    pick_origami_candidates,
    sparse_site_evidence,
    sparse_site_evidence_diagnostics,
    site_gap_contrast_for_regions,
    supported_grid_site_mask,
    supported_site_centroids,
    supported_site_spacing_errors,
)
from drift_analysis import undrift_rcc_with_lattice_suppression

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "DNA PAINT Picasso-Style ROI Analyzer"
DEFAULT_DATA_DIR = Path.home() / "Desktop" / "LBNL_PAINT"
DEFAULT_PIXEL_SIZE_NM = 130.0
DEFAULT_ORIGAMI_PICK_BIN_NM = 10.0
DEFAULT_ORIGAMI_MIN_DENSITY = 0.80
DEFAULT_ORIGAMI_MIN_POINTS = 200
DEFAULT_ORIGAMI_MAX_POINTS = 3000
DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM = 8.0
DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS = 128
DEFAULT_ORIGAMI_ALIGNMENT_PASSES = 3
DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM = 20.0
DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE = 0.10
DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM = 5.0
DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS = 5
DEFAULT_ORIGAMI_CORRELATION_THRESHOLD = 0.30
DEFAULT_ORIGAMI_USE_CORRELATION_GATE = True
DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION = 0.30
DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE = False
DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR = 0.25
DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY = True
DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY = False
DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY = False
DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS = False
DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS = False
DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY = False
DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS = False
DEFAULT_ORIGAMI_ROWS = 8
DEFAULT_ORIGAMI_COLUMNS = 12
DEFAULT_ORIGAMI_GRID_WIDTH_NM = 120.0
DEFAULT_ORIGAMI_GRID_HEIGHT_NM = 35.0
DEFAULT_ORIGAMI_SPACING_X_NM = DEFAULT_ORIGAMI_GRID_WIDTH_NM / (DEFAULT_ORIGAMI_COLUMNS - 1)
DEFAULT_ORIGAMI_SPACING_Y_NM = DEFAULT_ORIGAMI_GRID_HEIGHT_NM / (DEFAULT_ORIGAMI_ROWS - 1)
ORIGAMI_ALIGNMENT_CACHE_KEYS = (
    "_alignment_template_signature",
    "alignment_pixel_nm",
    "alignment_max_patch_pixels",
    "alignment_iterations",
    "site_mask_radius_nm",
    "min_site_localizations",
    "min_supported_sites",
)
ORIGAMI_SITE_CACHE_KEYS = (
    "min_site_evidence",
    "rows",
    "columns",
    "spacing_x_nm",
    "spacing_y_nm",
    "digital_pixel_model",
)
ORIGAMI_TRANSIENT_PIPELINE_KEYS = (
    "_precomputed_candidates",
    "_prealigned_picks",
    "_premeasured_picks",
    "_candidate_core_bounds_nm",
    "_display_stage",
    "_alignment_accepted_mask",
)


def independent_origami_pipeline_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove ROI-local caches before running the complete workflow on another tile."""
    fresh = {
        key: value
        for key, value in params.items()
        if key not in ORIGAMI_TRANSIENT_PIPELINE_KEYS
    }
    fresh["_inspection_stage"] = 5
    return fresh


def summarize_origami_rejection_reasons(
    details: list[dict[str, Any]],
) -> dict[str, int]:
    """Count rejected candidates once in each broad acceptance-gate category."""
    categories = {
        "point limits": 0,
        "alignment correlation": 0,
        "digital-group evidence": 0,
        "cell-pattern correlation": 0,
        "other": 0,
    }
    for detail in details:
        matched: set[str] = set()
        for reason in detail.get("failure_reasons", ()):
            normalized = str(reason).lower()
            if "points " in normalized:
                matched.add("point limits")
            elif "digital-group" in normalized or "template probability" in normalized:
                matched.add("digital-group evidence")
            elif "cell-pattern" in normalized:
                matched.add("cell-pattern correlation")
            elif "corr" in normalized or "alignment pose" in normalized:
                matched.add("alignment correlation")
            else:
                matched.add("other")
        if not matched:
            matched.add("other")
        for category in matched:
            categories[category] += 1
    return {name: count for name, count in categories.items() if count}


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
DEVELOPMENT_CACHE_DIR = APP_STATE_DIR / "development-session-cache"
DEVELOPMENT_CACHE_VERSION = 1
LATEST_DEVELOPMENT_SESSION_FILE = DEVELOPMENT_CACHE_DIR / "latest-session.json"
MAX_RENDER_PIXELS = 30_000_000
MIN_DYNAMIC_RENDER_PIXEL_NM = 1.0
DYNAMIC_RENDER_DEBOUNCE_MS = 350
AUTO_DENSITY_HISTOGRAM_BINS = 512
AUTO_DENSITY_SATURATION_FRACTION = 0.001
AUTO_DENSITY_HIGHLIGHT_LEVEL = 0.98
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


def optimal_dynamic_render_pixel_nm(
    viewport_nm: tuple[float, float, float, float],
    display_width_px: float,
    display_height_px: float,
    minimum_pixel_nm: float = MIN_DYNAMIC_RENDER_PIXEL_NM,
) -> float:
    """Choose one render pixel per displayed plot pixel, bounded by a physical floor."""
    x0, x1, y0, y1 = viewport_nm
    width_nm = abs(float(x1) - float(x0))
    height_nm = abs(float(y1) - float(y0))
    if width_nm <= 0 or height_nm <= 0:
        raise ValueError("Dynamic render viewport must have positive width and height.")
    if display_width_px <= 0 or display_height_px <= 0:
        raise ValueError("Dynamic render display dimensions must be positive.")
    if minimum_pixel_nm <= 0:
        raise ValueError("Minimum dynamic render pixel size must be positive.")
    return max(
        float(minimum_pixel_nm),
        width_nm / float(display_width_px),
        height_nm / float(display_height_px),
    )


def responsive_column_count(
    width_px: int,
    minimum_cell_width_px: int,
    maximum_columns: int,
) -> int:
    """Return a stable 1/2/4-style column count for a resizable control area."""
    if minimum_cell_width_px <= 0 or maximum_columns <= 0:
        raise ValueError("Responsive grid dimensions must be positive.")
    fitting = max(1, min(int(maximum_columns), int(width_px) // int(minimum_cell_width_px)))
    if maximum_columns >= 4 and fitting == 3:
        return 2
    return fitting


def origami_stage_cache_matches(
    previous_params: dict[str, Any],
    current_params: dict[str, Any],
    keys: tuple[str, ...],
) -> bool:
    """Return whether the inputs that actually affect a cached stage are unchanged."""
    return all(previous_params.get(key) == current_params.get(key) for key in keys)


def origami_source_fingerprint(points_nm: np.ndarray) -> str:
    """Return a stable identity for the exact point cloud used by Step 1."""
    points = np.ascontiguousarray(np.asarray(points_nm, dtype=np.float64))
    digest = hashlib.sha1()
    digest.update(np.asarray(points.shape, dtype=np.int64).tobytes())
    digest.update(points.view(np.uint8))
    return digest.hexdigest()


def theoretical_grid_in_footprint(grid_points_nm: np.ndarray, corners_nm: np.ndarray) -> np.ndarray:
    """Transform centered theoretical grid points into a fitted world-space footprint."""
    grid = np.asarray(grid_points_nm, dtype=float)
    corners = np.asarray(corners_nm, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("The theoretical grid must be an N x 2 coordinate array.")
    if corners.shape != (4, 2):
        raise ValueError("A fitted footprint must contain four x/y corners.")
    x_edge = corners[1] - corners[0]
    y_edge = corners[3] - corners[0]
    x_length = float(np.linalg.norm(x_edge))
    y_length = float(np.linalg.norm(y_edge))
    if x_length <= 0.0 or y_length <= 0.0:
        raise ValueError("A fitted footprint must have positive width and height.")
    center = np.mean(corners, axis=0)
    return center + grid[:, :1] * (x_edge / x_length) + grid[:, 1:] * (y_edge / y_length)


def classified_template_overlay_indices(
    grid_points_nm: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Select the physical sites belonging to a classified logical template.

    Digital-pixel detection intentionally retains the complete physical lattice
    for bright/dark measurements. Once a template has been classified, the
    theoretical overlay should instead show its common alignment sites plus the
    physical sites belonging to its active logical bits.
    """
    grid = np.asarray(grid_points_nm, dtype=float)
    all_indices = np.arange(len(grid), dtype=int)
    model = params.get("logical_model")
    if not isinstance(model, dict):
        return all_indices
    rows, columns = model.get("physical_shape", (0, 0))
    if int(rows) * int(columns) != len(grid):
        # Legacy/sparse alignment results are already the desired overlay.
        return all_indices
    selected = {
        int(index)
        for index in model.get("alignment_cells", ())
        if 0 <= int(index) < len(grid)
    }
    active_bits = tuple(bool(value) for value in model.get("active_bits", ()))
    bit_cells = tuple(model.get("bit_physical_cells", ()))
    if len(active_bits) != len(bit_cells):
        return all_indices
    for active, cells in zip(active_bits, bit_cells):
        if active:
            selected.update(
                int(index) for index in cells if 0 <= int(index) < len(grid)
            )
    return np.asarray(sorted(selected), dtype=int)


def classified_template_overlay_points(
    alignment_points_nm: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Return classified-template sites, reconstructing a full logical lattice if needed."""
    alignment_points = np.asarray(alignment_points_nm, dtype=float)
    model = params.get("logical_model")
    if not isinstance(model, dict):
        return alignment_points
    try:
        rows, columns = (int(value) for value in model["physical_shape"])
        full_grid = ideal_grid_points(
            rows,
            columns,
            float(params["spacing_x_nm"]),
            float(params["spacing_y_nm"]),
        )
    except (KeyError, TypeError, ValueError):
        return alignment_points
    return full_grid[classified_template_overlay_indices(full_grid, params)]


def alignment_template_overlay_points(
    fallback_points_nm: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Return the bright-site geometry extracted from the Step 2 template."""
    recorded = np.asarray(
        params.get("alignment_template_overlay_points_nm", ()), dtype=float
    )
    if recorded.size and recorded.ndim == 2 and recorded.shape[1] == 2:
        return recorded
    alignment = params.get("shared_alignment_template")
    if not isinstance(alignment, dict):
        return np.asarray(fallback_points_nm, dtype=float).reshape(-1, 2)
    loaded_points = np.asarray(alignment.get("overlay_points_nm", ()), dtype=float)
    if loaded_points.size and loaded_points.ndim == 2 and loaded_points.shape[1] == 2:
        return loaded_points
    if alignment.get("image") is None:
        # A shared Step 2 template exists, so the full Step 3 lattice is never
        # a valid substitute for its missing sparse geometry.
        return np.empty((0, 2), dtype=float)
    try:
        rows = int(alignment["rows"])
        columns = int(alignment["columns"])
        spacing_x_nm = float(alignment["spacing_x_nm"])
        spacing_y_nm = float(alignment["spacing_y_nm"])
        margin_nm = float(alignment["rectangle_margin_nm"])
        active_width_nm = max(spacing_x_nm, (columns - 1) * spacing_x_nm)
        active_height_nm = max(spacing_y_nm, (rows - 1) * spacing_y_nm)
        return custom_template_site_points(
            np.asarray(alignment["image"], dtype=float),
            rectangle_width_nm=active_width_nm + 2.0 * margin_nm,
            rectangle_height_nm=active_height_nm + 2.0 * margin_nm,
            spacing_x_nm=spacing_x_nm,
            spacing_y_nm=spacing_y_nm,
            active_width_nm=active_width_nm,
            active_height_nm=active_height_nm,
            pixel_size_x_nm=float(alignment["template_pixel_size_x_nm"]),
            pixel_size_y_nm=float(alignment["template_pixel_size_y_nm"]),
        )
    except (KeyError, TypeError, ValueError):
        # Do not mislabel the complete Step 3 measurement grid as the Step 2
        # alignment shape when template extraction fails.
        return np.empty((0, 2), dtype=float)


def digital_group_display_color(group_index: int) -> str:
    """Return the one stable identity color used everywhere for a group."""
    return matplotlib.colors.to_hex(
        matplotlib.colormaps["tab10"](int(group_index) % 10),
        keep_alpha=False,
    )


def digital_group_decision_annotations(
    grid_points_nm: np.ndarray,
    params: dict[str, Any],
    candidate_index: int,
) -> list[tuple[np.ndarray, str, str]] | None:
    """Return concise group labels, expanding only failed ON decisions."""
    model = params.get("digital_pixel_model")
    if not isinstance(model, dict):
        model = params.get("logical_model")
    if not isinstance(model, dict):
        return None
    bit_ids = tuple(str(value) for value in model.get("bit_ids", ()))
    bit_cells = tuple(model.get("bit_physical_cells", model.get("bit_cells", ())))
    if not bit_ids or len(bit_ids) != len(bit_cells):
        return None
    probability_rows = params.get("digital_pixel_probabilities", ())
    if not len(probability_rows):
        probability_rows = params.get("classification_logical_bit_probabilities", ())
    if not 0 <= int(candidate_index) < len(probability_rows):
        return None
    probabilities = np.asarray(probability_rows[int(candidate_index)], dtype=float)
    if len(probabilities) != len(bit_ids):
        return None
    group_evidence_rows = params.get("digital_group_localization_evidence", ())
    group_prominence_rows = params.get("digital_group_prominences", ())
    group_evidence: np.ndarray | None = None
    group_prominence: np.ndarray | None = None
    if 0 <= int(candidate_index) < len(group_evidence_rows):
        candidate_evidence = np.asarray(
            group_evidence_rows[int(candidate_index)], dtype=float
        )
        if candidate_evidence.shape == (len(bit_ids),):
            group_evidence = candidate_evidence
    if 0 <= int(candidate_index) < len(group_prominence_rows):
        candidate_prominence = np.asarray(
            group_prominence_rows[int(candidate_index)], dtype=float
        )
        if candidate_prominence.shape == (len(bit_ids),):
            group_prominence = candidate_prominence
    grid = np.asarray(grid_points_nm, dtype=float)
    try:
        rows, columns = (int(value) for value in model.get("physical_shape", (0, 0)))
        if rows * columns != len(grid):
            grid = ideal_grid_points(
                rows,
                columns,
                float(params["spacing_x_nm"]),
                float(params["spacing_y_nm"]),
            )
    except (KeyError, TypeError, ValueError):
        return None
    annotations: list[tuple[np.ndarray, str, str]] = []
    for bit_index, (bit_id, cells, probability) in enumerate(
        zip(bit_ids, bit_cells, probabilities)
    ):
        indices = np.asarray(
            sorted({int(value) for value in cells if 0 <= int(value) < len(grid)}),
            dtype=int,
        )
        if not len(indices) or not np.isfinite(probability):
            continue
        center = np.mean(grid[indices], axis=0)
        failure_reasons: list[str] = []
        if group_evidence is not None and np.isfinite(group_evidence[bit_index]):
            support_per_position = float(group_evidence[bit_index])
            digital_prominence = (
                float(group_prominence[bit_index])
                if group_prominence is not None
                and np.isfinite(group_prominence[bit_index])
                else max(0.0, 2.0 * float(probability) - 1.0)
            )
            minimum_support = float(params.get("min_site_localizations", 0.0))
            minimum_prominence = float(params.get("min_site_evidence", 0.0))
            if support_per_position < minimum_support:
                failure_reasons.append(
                    f"locs/position {support_per_position:.1f} < {minimum_support:g}"
                )
            if digital_prominence < minimum_prominence:
                failure_reasons.append(
                    f"prominence {digital_prominence:.2f} < {minimum_prominence:g}"
                )
        if float(probability) < 0.5 and not failure_reasons:
            failure_reasons.append(
                f"ON probability {float(probability):.2f} < 0.50"
            )
        passed = float(probability) >= 0.5 and not failure_reasons
        text = bit_id if passed else f"{bit_id} — FAIL: {'; '.join(failure_reasons)}"
        color = "#22c55e" if passed else "#ff3030"
        annotations.append((
            center,
            text,
            color,
        ))
    return annotations


def digital_group_decision_overlay_points(
    grid_points_nm: np.ndarray,
    params: dict[str, Any],
    candidate_index: int,
) -> np.ndarray | None:
    """Return analog sites belonging to inferred-ON groups plus alignment sites."""
    model = params.get("digital_pixel_model")
    if not isinstance(model, dict):
        model = params.get("logical_model")
    if not isinstance(model, dict):
        return None
    bit_cells = tuple(model.get("bit_physical_cells", model.get("bit_cells", ())))
    probability_rows = params.get("digital_pixel_probabilities", ())
    if not len(probability_rows):
        probability_rows = params.get("classification_logical_bit_probabilities", ())
    if not 0 <= int(candidate_index) < len(probability_rows):
        return None
    probabilities = np.asarray(probability_rows[int(candidate_index)], dtype=float)
    if len(probabilities) != len(bit_cells):
        return None
    grid = np.asarray(grid_points_nm, dtype=float)
    try:
        rows, columns = (int(value) for value in model.get("physical_shape", (0, 0)))
        if rows * columns != len(grid):
            grid = ideal_grid_points(
                rows,
                columns,
                float(params["spacing_x_nm"]),
                float(params["spacing_y_nm"]),
            )
    except (KeyError, TypeError, ValueError):
        return None
    selected = {
        int(value)
        for value in model.get("alignment_cells", ())
        if 0 <= int(value) < len(grid)
    }
    for probability, cells in zip(probabilities, bit_cells):
        if np.isfinite(probability) and float(probability) >= 0.5:
            selected.update(
                int(value) for value in cells if 0 <= int(value) < len(grid)
            )
    return grid[np.asarray(sorted(selected), dtype=int)] if selected else np.empty((0, 2))


def digital_group_decision_blobs(
    grid_points_nm: np.ndarray,
    params: dict[str, Any],
    candidate_index: int,
) -> list[tuple[str, list[np.ndarray], str, float, str]]:
    """Return identity-colored group envelopes with state encoded by styling."""
    model = params.get("digital_pixel_model")
    if not isinstance(model, dict):
        model = params.get("logical_model")
    if not isinstance(model, dict):
        return []
    group_ids = tuple(str(value) for value in model.get("bit_ids", ()))
    probability_rows = params.get("digital_pixel_probabilities", ())
    if not len(probability_rows):
        probability_rows = params.get("classification_logical_bit_probabilities", ())
    if not 0 <= int(candidate_index) < len(probability_rows):
        return []
    probabilities = np.asarray(probability_rows[int(candidate_index)], dtype=float)
    if probabilities.shape != (len(group_ids),):
        return []
    decisions: dict[str, tuple[str, float, str]] = {}
    for group_id, probability in zip(group_ids, probabilities):
        if not np.isfinite(probability):
            continue
        is_on = float(probability) >= 0.5
        alpha, linestyle = (0.18, "solid") if is_on else (0.14, "dashed")
        decisions[group_id] = (
            "#22c55e" if is_on else "#ff3030",
            alpha,
            linestyle,
        )
    styled: list[tuple[str, list[np.ndarray], str, float, str]] = []
    for group_id, contours in digital_group_blob_contours(
        grid_points_nm,
        model,
        radius_nm=float(params.get("site_mask_radius_nm", 0.0)) or None,
    ):
        decision = decisions.get(group_id)
        if decision is None:
            continue
        color, alpha, linestyle = decision
        styled.append((group_id, contours, color, alpha, linestyle))
    return styled


def digital_group_blob_contours(
    grid_points_nm: np.ndarray,
    model: dict[str, Any] | None,
    *,
    active_only: bool = False,
    radius_nm: float | None = None,
) -> list[tuple[str, list[np.ndarray]]]:
    """Build group outlines whose per-position radius matches the assignment mask."""
    if not isinstance(model, dict):
        return []
    grid = np.asarray(grid_points_nm, dtype=float)
    if grid.ndim != 2 or grid.shape[1:] != (2,) or not len(grid):
        return []
    bit_ids = tuple(str(value) for value in model.get("bit_ids", ()))
    bit_cells = tuple(model.get("bit_physical_cells", model.get("bit_cells", ())))
    if not bit_ids or len(bit_ids) != len(bit_cells):
        return []
    active_bits = tuple(bool(value) for value in model.get("active_bits", ()))
    if active_only and len(active_bits) == len(bit_ids):
        selected = [
            (bit_id, cells)
            for bit_id, cells, active in zip(bit_ids, bit_cells, active_bits)
            if active
        ]
        bit_ids = tuple(item[0] for item in selected)
        bit_cells = tuple(item[1] for item in selected)
    if len(grid) > 1:
        differences = grid[:, None, :] - grid[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        positive = distances[distances > 1e-9]
        spacing = float(np.min(positive)) if len(positive) else 1.0
    else:
        spacing = 1.0
    contour_level = 0.30
    display_radius = (
        float(radius_nm)
        if radius_nm is not None and np.isfinite(radius_nm) and radius_nm > 0.0
        else 0.589 * spacing
    )
    # For a Gaussian centered on each member position, choose sigma so the
    # requested site-mask radius lies exactly on the rendered contour level.
    sigma = max(
        display_radius / math.sqrt(-2.0 * math.log(contour_level)),
        1e-6,
    )
    padding = 1.15 * display_radius
    outlines: list[tuple[str, list[np.ndarray]]] = []
    for bit_id, cells in zip(bit_ids, bit_cells):
        indices = np.asarray(
            sorted({int(value) for value in cells if 0 <= int(value) < len(grid)}),
            dtype=int,
        )
        if not len(indices):
            continue
        points = grid[indices]
        x_values = np.linspace(
            float(np.min(points[:, 0]) - padding),
            float(np.max(points[:, 0]) + padding),
            max(32, min(160, 24 + 12 * len(indices))),
        )
        y_values = np.linspace(
            float(np.min(points[:, 1]) - padding),
            float(np.max(points[:, 1]) + padding),
            max(32, min(160, 24 + 12 * len(indices))),
        )
        sample_x, sample_y = np.meshgrid(x_values, y_values)
        field = np.zeros_like(sample_x)
        for point in points:
            squared_distance = (sample_x - point[0]) ** 2 + (sample_y - point[1]) ** 2
            field = np.maximum(field, np.exp(-0.5 * squared_distance / sigma**2))
        generator = contourpy.contour_generator(x=x_values, y=y_values, z=field)
        contours = [
            np.asarray(line, dtype=float)
            for line in generator.lines(contour_level)
            if len(line) >= 3
        ]
        if contours:
            outlines.append((bit_id, contours))
    return outlines


def digital_group_site_memberships(
    model: dict[str, Any] | None,
    site_count: int,
    *,
    active_only: bool = False,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Map each analog lattice site to every custom digital group containing it."""
    if not isinstance(model, dict) or site_count < 1:
        return (), tuple(() for _ in range(max(0, int(site_count))))
    bit_ids = tuple(str(value) for value in model.get("bit_ids", ()))
    bit_cells = tuple(model.get("bit_physical_cells", model.get("bit_cells", ())))
    if not bit_ids or len(bit_ids) != len(bit_cells):
        return (), tuple(() for _ in range(int(site_count)))
    active_bits = tuple(bool(value) for value in model.get("active_bits", ()))
    if active_only and len(active_bits) == len(bit_ids):
        selected = [
            (bit_id, cells)
            for bit_id, cells, active in zip(bit_ids, bit_cells, active_bits)
            if active
        ]
        bit_ids = tuple(item[0] for item in selected)
        bit_cells = tuple(item[1] for item in selected)
    memberships: list[list[int]] = [[] for _ in range(int(site_count))]
    for group_index, cells in enumerate(bit_cells):
        for value in cells:
            site_index = int(value)
            if 0 <= site_index < site_count and group_index not in memberships[site_index]:
                memberships[site_index].append(group_index)
    return bit_ids, tuple(tuple(values) for values in memberships)


def custom_template_contours_nm(
    template_image: np.ndarray,
    footprint_width_nm: float,
    footprint_height_nm: float,
    *,
    spacing_x_nm: float | None = None,
    spacing_y_nm: float | None = None,
    active_width_nm: float | None = None,
    active_height_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
    relative_level: float = 0.35,
    maximum_pixels: int = 256,
) -> list[np.ndarray]:
    """Return bright-signal contours from a custom template in local physical coordinates."""
    image = np.asarray(template_image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 2 or not np.all(np.isfinite(image)):
        return []
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum or footprint_width_nm <= 0.0 or footprint_height_nm <= 0.0:
        return []
    if max(image.shape) > maximum_pixels:
        row_stride = max(1, int(math.ceil(image.shape[0] / maximum_pixels)))
        column_stride = max(1, int(math.ceil(image.shape[1] / maximum_pixels)))
        padded_rows = int(math.ceil(image.shape[0] / row_stride) * row_stride)
        padded_columns = int(math.ceil(image.shape[1] / column_stride) * column_stride)
        padded = np.full((padded_rows, padded_columns), minimum, dtype=float)
        padded[: image.shape[0], : image.shape[1]] = image
        image = padded.reshape(
            padded_rows // row_stride,
            row_stride,
            padded_columns // column_stride,
            column_stride,
        ).max(axis=(1, 3))
    x_nm, y_nm = custom_template_physical_axes(
        image,
        rectangle_width_nm=footprint_width_nm,
        rectangle_height_nm=footprint_height_nm,
        spacing_x_nm=spacing_x_nm,
        spacing_y_nm=spacing_y_nm,
        active_width_nm=active_width_nm,
        active_height_nm=active_height_nm,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    generator = contourpy.contour_generator(x=x_nm, y=y_nm, z=image)
    level = minimum + float(np.clip(relative_level, 0.0, 1.0)) * (maximum - minimum)
    return [np.asarray(line, dtype=float) for line in generator.lines(level) if len(line) >= 2]


def load_custom_template_image(path: str | Path) -> np.ndarray:
    """Load a bright-on-dark custom template and return physical-y-up grayscale data."""
    template_path = Path(path)
    if template_path.suffix.lower() in {".tif", ".tiff"}:
        raw = np.asarray(tifffile.imread(template_path))
    else:
        raw = np.asarray(matplotlib_image.imread(template_path))
    image = np.asarray(raw, dtype=float)
    if raw.ndim == 3 and raw.shape[2] in {3, 4}:
        rgb = np.asarray(raw[..., :3], dtype=float)
        metadata = load_custom_template_metadata(template_path)
        if metadata is not None and isinstance(metadata.get("logical_model"), dict):
            # Generator group colors are annotations, not intensity weights.
            # Every exported group color has a full-strength channel, so max
            # recovers one equal-brightness physical alignment image.
            image = np.max(rgb, axis=2)
        else:
            image = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        if raw.shape[2] == 4:
            image *= np.asarray(raw[..., 3], dtype=float)
    if image.ndim != 2:
        raise ValueError("Custom templates must be a single 2D grayscale image or one RGB/RGBA image.")
    if min(image.shape) < 2 or not np.all(np.isfinite(image)):
        raise ValueError("Custom templates must be at least 2 x 2 pixels and contain only finite values.")
    if float(np.max(image)) <= float(np.min(image)):
        raise ValueError("Custom templates must contain both dark background and bright signal.")
    # Raster rows run downward; alignment coordinates use positive y upward.
    return np.flipud(np.asarray(image, dtype=float)).copy()


def custom_template_display_name(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Return a readable label for symbolic template filenames.

    Finder represents a slash in a filename as a colon at the POSIX layer,
    and symbolic templates can also acquire a repeated image suffix (for
    example ``:.png.png``). Keep the filesystem-safe path untouched while
    restoring the intended label in selectors and plots.
    """
    if metadata is not None:
        embedded_name = metadata.get("display_name", metadata.get("name"))
        if isinstance(embedded_name, str) and embedded_name.strip():
            return embedded_name.strip()
    name = Path(path).stem
    image_suffixes = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    while any(name.lower().endswith(suffix) for suffix in image_suffixes):
        name = Path(name).stem
    if name == ":":
        return "/"
    if name in {":\\", "\\"}:
        return "\\"
    return name


def load_custom_template_metadata(path: str | Path) -> dict[str, Any] | None:
    """Read Picklist Generator geometry from the PNG or its JSON sidecar."""
    template_path = Path(path)
    candidates: list[dict[str, Any]] = []
    try:
        with PillowImage.open(template_path) as opened:
            embedded = opened.info.get("paint_analysis_template")
        if isinstance(embedded, str):
            parsed = json.loads(embedded)
            if isinstance(parsed, dict):
                candidates.append(parsed)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    sidecar = template_path.with_suffix(".json")
    try:
        parsed = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    for metadata in candidates:
        if metadata.get("format") != "paint-analysis-origami-template-v1":
            continue
        required = ("rows", "columns", "spacing_x_nm", "spacing_y_nm", "margin_nm", "width_px", "height_px")
        if not all(key in metadata for key in required):
            continue
        metadata = dict(metadata)
        # Early Picklist Generator exports used a 120 x 40 nm active lattice.
        # Correct that recognizable 8 x 12 calibration to the measured
        # 120 x 35 nm geometry while leaving arbitrary/custom geometries alone.
        legacy_x = 120.0 / 11.0
        legacy_y = 40.0 / 7.0
        if (
            int(metadata["rows"]) == DEFAULT_ORIGAMI_ROWS
            and int(metadata["columns"]) == DEFAULT_ORIGAMI_COLUMNS
            and math.isclose(float(metadata["spacing_x_nm"]), legacy_x, rel_tol=0.0, abs_tol=1e-3)
            and math.isclose(float(metadata["spacing_y_nm"]), legacy_y, rel_tol=0.0, abs_tol=1e-3)
        ):
            margin_nm = float(metadata["margin_nm"])
            metadata["spacing_y_nm"] = DEFAULT_ORIGAMI_SPACING_Y_NM
            metadata["height_nm"] = DEFAULT_ORIGAMI_GRID_HEIGHT_NM + 2.0 * margin_nm
            metadata.pop("pixel_size_y_nm", None)
            metadata["geometry_migrated_from_120x40"] = True
        width_px = int(metadata["width_px"])
        height_px = int(metadata["height_px"])
        width_nm = float(metadata.get("width_nm", 0.0))
        height_nm = float(metadata.get("height_nm", 0.0))
        pixel_x_nm = float(metadata.get("pixel_size_x_nm", width_nm / max(width_px - 1, 1)))
        pixel_y_nm = float(metadata.get("pixel_size_y_nm", height_nm / max(height_px - 1, 1)))
        if width_px > 1 and height_px > 1 and pixel_x_nm > 0.0 and pixel_y_nm > 0.0:
            result = dict(metadata)
            result["pixel_size_x_nm"] = pixel_x_nm
            result["pixel_size_y_nm"] = pixel_y_nm
            return result
    return None


def logical_bit_model_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Validate arbitrary logical-bit definitions embedded by the generator."""
    model = metadata.get("logical_model")
    if not isinstance(model, dict):
        return None
    if model.get("format") not in {
        "paint-analysis-logical-bits-v1",
        "paint-analysis-logical-strokes-v1",
    }:
        raise ValueError("Unsupported logical-bit template metadata format.")
    rows = int(metadata["rows"])
    columns = int(metadata["columns"])
    if int(model.get("physical_rows", -1)) != rows or int(model.get("physical_columns", -1)) != columns:
        raise ValueError("Logical-stroke metadata does not match the physical template lattice.")
    raw_bits = model.get("logical_bits")
    raw_alignment_groups = model.get("alignment_groups", model.get("alignment_strokes", []))
    active_ids = model.get("active_logical_bits")
    if (
        not isinstance(raw_bits, list)
        or not raw_bits
        or not isinstance(raw_alignment_groups, list)
        or not isinstance(active_ids, list)
    ):
        raise ValueError("Logical-stroke metadata must define logical bits and their active states.")

    def cells_from_sites(raw_sites: object, owner: str) -> tuple[int, ...]:
        if not isinstance(raw_sites, list):
            raise ValueError(f"{owner} has invalid physical sites.")
        cells: set[int] = set()
        for site in raw_sites:
            if not isinstance(site, (list, tuple)) or len(site) != 2:
                raise ValueError(f"{owner} contains an invalid physical site.")
            source_row, column = int(site[0]) - 1, int(site[1]) - 1
            if not (0 <= source_row < rows and 0 <= column < columns):
                raise ValueError(f"{owner} contains a site outside the lattice.")
            # Generator rows are top-to-bottom. Alignment coordinates and the
            # full physical lattice use y-up ordering, so reverse the row here.
            physical_row = rows - 1 - source_row
            cells.add(physical_row * columns + column)
        if not cells:
            raise ValueError(f"{owner} contains no physical sites.")
        return tuple(sorted(cells))

    alignment_cells: set[int] = set()
    for group in raw_alignment_groups:
        if not isinstance(group, dict):
            raise ValueError("Each alignment group must be an object.")
        alignment_cells.update(
            cells_from_sites(group.get("physical_sites"), f"Alignment group {group.get('id', '')}")
        )
    bit_ids: list[str] = []
    bit_cells: list[tuple[int, ...]] = []
    bit_physical_cells: list[tuple[int, ...]] = []
    for bit in raw_bits:
        if not isinstance(bit, dict) or not isinstance(bit.get("physical_sites"), list):
            raise ValueError("Each logical bit must list its physical lattice sites.")
        bit_id = str(bit.get("id", "")).strip()
        if not bit_id or bit_id in bit_ids:
            raise ValueError("Logical bit identifiers must be nonempty and unique.")
        evidence_sites = bit.get("evidence_sites", bit["physical_sites"])
        bit_ids.append(bit_id)
        bit_cells.append(cells_from_sites(evidence_sites, f"Logical bit {bit_id}"))
        bit_physical_cells.append(
            cells_from_sites(bit["physical_sites"], f"Logical bit {bit_id}")
        )
    unknown = sorted(set(str(value) for value in active_ids) - set(bit_ids))
    if unknown:
        raise ValueError(f"Active logical bit {unknown[0]} is not defined by the template schema.")
    ordered = sorted(zip(bit_ids, bit_cells, bit_physical_cells), key=lambda item: item[0])
    bit_ids = [item[0] for item in ordered]
    bit_cells = [item[1] for item in ordered]
    bit_physical_cells = [item[2] for item in ordered]
    active_set = set(str(value) for value in active_ids)
    return {
        "format": "paint-analysis-logical-bits-v1",
        "physical_shape": (rows, columns),
        "bit_ids": tuple(bit_ids),
        "bit_cells": tuple(bit_cells),
        "bit_physical_cells": tuple(bit_physical_cells),
        "alignment_cells": tuple(sorted(alignment_cells)),
        "active_bits": tuple(bit_id in active_set for bit_id in bit_ids),
    }


# Compatibility name retained for callers and older tests that used the first
# fixed-stroke implementation.
logical_stroke_model_from_metadata = logical_bit_model_from_metadata


def origami_candidate_failure_reasons(
    *,
    point_count: int,
    correlation: float,
    supported_sites: int,
    supported_rows: int,
    supported_columns: int,
    spacing_error_nm: float,
    params: dict[str, Any],
) -> list[str]:
    """Describe every identification gate failed by one candidate."""
    reasons: list[str] = []
    minimum_points = int(params.get("min_candidate_points", 0))
    maximum_points = int(params.get("max_candidate_points", np.iinfo(np.int64).max))
    if point_count < minimum_points:
        reasons.append(f"points {point_count} < {minimum_points}")
    if point_count > maximum_points:
        reasons.append(f"points {point_count} > {maximum_points}")
    minimum_correlation = float(params.get("min_rectangle_confidence", 0.0))
    if bool(params.get("use_correlation_gate", True)) and not correlation >= minimum_correlation:
        reasons.append(f"corr {correlation:.2f} < {minimum_correlation:g}")
    # Digital templates are scored from direct localization-to-group evidence.
    # The physical-grid statistics below remain useful QC, but are not gates.
    if isinstance(params.get("digital_pixel_model"), dict):
        return reasons
    minimum_sites = int(params.get("min_supported_sites", 0))
    if supported_sites < minimum_sites:
        reasons.append(f"sites {supported_sites} < {minimum_sites}")
    minimum_rows = int(params.get("min_supported_rows", 0))
    if supported_rows < minimum_rows:
        reasons.append(f"rows {supported_rows} < {minimum_rows}")
    minimum_columns = int(params.get("min_supported_columns", 0))
    if supported_columns < minimum_columns:
        reasons.append(f"columns {supported_columns} < {minimum_columns}")
    maximum_spacing = float(params.get("max_site_spacing_error_nm", float("inf")))
    if not spacing_error_nm <= maximum_spacing:
        reasons.append(f"spacing {spacing_error_nm:.1f} > {maximum_spacing:g} nm")
    return reasons


CLASSIFICATION_DIAGNOSTIC_GATES = (
    "Template evidence",
    "Pattern match",
    "Correlation",
    "Point limits",
    "Supported sites",
    "Row coverage",
    "Column coverage",
    "Spacing",
    "Other",
)


def classification_bias_diagnostics(
    template_names: list[str],
    assigned_counts: list[int] | np.ndarray,
    unclassified_details: list[dict[str, Any]],
    unclassified_count: int,
) -> dict[str, np.ndarray | int]:
    """Summarize winner preferences and gate failures without relabeling data."""
    names = [str(name) for name in template_names]
    assigned = np.asarray(assigned_counts, dtype=int)
    if assigned.shape != (len(names),):
        raise ValueError("Assigned counts must contain one value per template.")
    name_to_index = {name: index for index, name in enumerate(names)}
    rejected_best = np.zeros(len(names), dtype=int)
    failure_counts = np.zeros((len(names), len(CLASSIFICATION_DIAGNOSTIC_GATES)), dtype=int)

    def gate_index(reason: str) -> int:
        lowered = reason.lower()
        if (
            "monte carlo" in lowered
            or "logical-stroke template probability" in lowered
            or "logical-bit template probability" in lowered
        ):
            return 0
        if "cell-pattern" in lowered or "pattern match" in lowered:
            return 1
        if lowered.startswith("corr") or "correlation" in lowered:
            return 2
        if lowered.startswith("points"):
            return 3
        if lowered.startswith("sites"):
            return 4
        if lowered.startswith("rows"):
            return 5
        if lowered.startswith("columns"):
            return 6
        if lowered.startswith("spacing"):
            return 7
        return 8

    described = 0
    for detail in unclassified_details:
        template_index = name_to_index.get(str(detail.get("template_name", "")))
        if template_index is None:
            continue
        described += 1
        rejected_best[template_index] += 1
        reasons = tuple(str(reason) for reason in detail.get("failure_reasons", ()))
        if not reasons:
            failure_counts[template_index, -1] += 1
        else:
            # A candidate may fail several gates. Count each gate once so the
            # heatmap reveals co-occurring causes without duplicate messages.
            for index in sorted({gate_index(reason) for reason in reasons}):
                failure_counts[template_index, index] += 1
    preference_counts = assigned + rejected_best
    assignment_rates = np.divide(
        assigned,
        preference_counts,
        out=np.zeros(len(names), dtype=float),
        where=preference_counts > 0,
    )
    return {
        "assigned": assigned,
        "rejected_best": rejected_best,
        "preference_counts": preference_counts,
        "assignment_rates": assignment_rates,
        "failure_counts": failure_counts,
        "unresolved_unclassified": max(0, int(unclassified_count) - described),
    }


def origami_site_decision_label(
    count: int,
    prominence: float,
    minimum_count: int,
    minimum_prominence: float,
) -> tuple[bool, str, str]:
    """Return support state, diagnostic color, and exact per-site decision text."""
    count_passes = int(count) >= int(minimum_count)
    prominence_passes = float(prominence) >= float(minimum_prominence)
    if count_passes and prominence_passes:
        return True, "#84cc16", f"{int(count)} loc; p={float(prominence):.2f} ✓"
    failures: list[str] = []
    if not count_passes:
        failures.append(f"loc {int(count)}<{int(minimum_count)}")
    if not prominence_passes:
        failures.append(f"p {float(prominence):.2f}<{float(minimum_prominence):g}")
    if not count_passes and not prominence_passes:
        color = "#ef4444"
    elif not count_passes:
        color = "#f59e0b"
    else:
        color = "#e879f9"
    return False, color, "; ".join(failures)


@dataclass
class LoadedData:
    path: Path
    locs: pd.DataFrame
    info: list[dict[str, Any]]
    metadata: dict[str, Any]


def _cache_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot cache value of type {type(value).__name__}.")


def file_cache_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def drift_cache_settings(
    method: str,
    segmentation: int,
    aim_intersect_nm: float,
    aim_roi_nm: float,
    rcc_lattice_pitch_nm: float,
    drift_file_path: Path | None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "method": str(method),
        "segmentation": int(segmentation),
        "aim_intersect_nm": float(aim_intersect_nm),
        "aim_roi_nm": float(aim_roi_nm),
        "rcc_lattice_pitch_nm": float(rcc_lattice_pitch_nm),
        "drift_file": None,
    }
    if str(method) == "file" and drift_file_path is not None:
        settings["drift_file"] = file_cache_fingerprint(drift_file_path)
    return settings


def _development_cache_path(
    source_path: Path,
    settings: dict[str, Any],
    cache_dir: Path = DEVELOPMENT_CACHE_DIR,
) -> Path:
    identity = {
        "source_path": str(source_path.expanduser().resolve()),
        "settings": settings,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=_cache_json_default).encode("utf-8")
    ).hexdigest()[:24]
    return cache_dir / f"session-{digest}.h5"


def _write_cached_dataframe(handle: h5py.File, name: str, frame: pd.DataFrame) -> None:
    records = frame.to_records(index=False)
    if records.dtype.hasobject:
        raise TypeError("Development caching requires numeric localization columns.")
    handle.create_dataset(name, data=records)


def save_development_session_cache(
    loaded: LoadedData,
    corrected_locs: pd.DataFrame,
    drift: pd.DataFrame | None,
    correction_label: str,
    settings: dict[str, Any],
    cache_dir: Path = DEVELOPMENT_CACHE_DIR,
) -> Path | None:
    """Persist one source/correction pair for fast development restarts."""
    try:
        source_fingerprint = file_cache_fingerprint(loaded.path)
        cache_path = _development_cache_path(loaded.path, settings, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        with h5py.File(temporary_path, "w") as handle:
            handle.attrs["cache_version"] = DEVELOPMENT_CACHE_VERSION
            handle.attrs["source_fingerprint"] = json.dumps(source_fingerprint, sort_keys=True)
            handle.attrs["settings"] = json.dumps(settings, sort_keys=True, default=_cache_json_default)
            handle.attrs["info"] = json.dumps(loaded.info, default=_cache_json_default)
            handle.attrs["metadata"] = json.dumps(loaded.metadata, default=_cache_json_default)
            handle.attrs["correction_label"] = str(correction_label)
            _write_cached_dataframe(handle, "raw_locs", loaded.locs)
            _write_cached_dataframe(handle, "corrected_locs", corrected_locs)
            if drift is not None:
                _write_cached_dataframe(handle, "drift", drift)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, cache_path)
        latest_path = cache_dir / LATEST_DEVELOPMENT_SESSION_FILE.name
        latest_temporary = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
        latest_temporary.write_text(
            json.dumps({"source_path": str(loaded.path.expanduser().resolve()), "settings": settings}, default=_cache_json_default),
            encoding="utf-8",
        )
        os.chmod(latest_temporary, 0o600)
        os.replace(latest_temporary, latest_path)
        return cache_path
    except Exception:
        try:
            if "temporary_path" in locals() and temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass
        return None


def load_development_session_cache(
    source_path: Path,
    settings: dict[str, Any],
    cache_dir: Path = DEVELOPMENT_CACHE_DIR,
) -> dict[str, Any] | None:
    """Load a valid app-generated session cache, or return None when stale."""
    try:
        source_path = source_path.expanduser().resolve()
        drift_file = settings.get("drift_file")
        if drift_file is not None and file_cache_fingerprint(Path(drift_file["path"])) != drift_file:
            return None
        cache_path = _development_cache_path(source_path, settings, cache_dir)
        if not cache_path.exists():
            return None
        with h5py.File(cache_path, "r") as handle:
            if int(handle.attrs["cache_version"]) != DEVELOPMENT_CACHE_VERSION:
                return None
            if json.loads(str(handle.attrs["source_fingerprint"])) != file_cache_fingerprint(source_path):
                return None
            if json.loads(str(handle.attrs["settings"])) != settings:
                return None
            loaded = LoadedData(
                path=source_path,
                locs=pd.DataFrame.from_records(np.asarray(handle["raw_locs"])),
                info=json.loads(str(handle.attrs["info"])),
                metadata=json.loads(str(handle.attrs["metadata"])),
            )
            drift = pd.DataFrame.from_records(np.asarray(handle["drift"])) if "drift" in handle else None
            cached_map = None
            if "corrected_map" in handle:
                map_group = handle["corrected_map"]
                cached_map = json.loads(str(map_group.attrs["metadata"]))
                cached_map["extent"] = tuple(float(value) for value in cached_map["extent"])
                cached_map["viewport_nm"] = None
                cached_map["image"] = np.asarray(map_group["image"], dtype=float)
                cached_map["source_path"] = source_path
                cached_map["restored_from_cache"] = True
            return {
                "loaded": loaded,
                "locs": pd.DataFrame.from_records(np.asarray(handle["corrected_locs"])),
                "drift": drift,
                "label": str(handle.attrs["correction_label"]),
                "source_path": source_path,
                "restored_from_cache": True,
                "cache_settings": settings,
                "cached_map": cached_map,
                "cache_path": cache_path,
            }
    except Exception:
        return None


def save_development_map_cache(cache_path: Path, map_result: dict[str, Any]) -> bool:
    """Add one full-field corrected render to an existing session cache."""
    if map_result.get("viewport_nm") is not None:
        return False
    try:
        metadata = {
            "result_type": str(map_result.get("result_type", "map")),
            "extent": [float(value) for value in map_result["extent"]],
            "n_rendered": int(map_result["n_rendered"]),
            "disp_px_size_nm": float(map_result["disp_px_size_nm"]),
            "blur_method": str(map_result["blur_method"]),
            "min_blur_width": float(map_result.get("min_blur_width", 1.0)),
        }
        with h5py.File(cache_path, "a") as handle:
            if "corrected_map" in handle:
                del handle["corrected_map"]
            group = handle.create_group("corrected_map")
            group.attrs["metadata"] = json.dumps(metadata)
            # Favor fast development restarts over disk compression. Compressing
            # a large full-field raster can itself monopolize CPU after render.
            group.create_dataset("image", data=np.asarray(map_result["image"], dtype=float))
        return True
    except Exception:
        return False


def latest_development_session_request(
    cache_dir: Path = DEVELOPMENT_CACHE_DIR,
) -> tuple[Path, dict[str, Any]] | None:
    try:
        payload = json.loads((cache_dir / LATEST_DEVELOPMENT_SESSION_FILE.name).read_text(encoding="utf-8"))
        source_path = Path(payload["source_path"])
        settings = dict(payload["settings"])
        if not source_path.is_file():
            return None
        return source_path, settings
    except Exception:
        return None


@dataclass(frozen=True, order=True)
class TemporalVLineAnnotation:
    frame: int
    label: str


def parse_temporal_vline_annotation(frame_value: object, label_value: object) -> TemporalVLineAnnotation:
    """Validate one user-entered vertical frame annotation."""
    try:
        numeric_frame = float(str(frame_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Annotation frame must be a non-negative whole number.") from exc
    if not np.isfinite(numeric_frame) or numeric_frame < 0 or not numeric_frame.is_integer():
        raise ValueError("Annotation frame must be a non-negative whole number.")
    label = str(label_value).strip()
    if not label:
        raise ValueError("Annotation text cannot be empty.")
    return TemporalVLineAnnotation(int(numeric_frame), label)


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
    pixelsize = float(metadata.get("Pixelsize") or DEFAULT_PIXEL_SIZE_NM)
    return [{"Frames": frames, "Width": width, "Height": height, "Pixelsize": pixelsize}]


def finalize_loaded_locs(path: Path, locs: pd.DataFrame, metadata: dict[str, Any]) -> LoadedData:
    missing = {"frame", "x", "y"}.difference(locs.columns)
    if missing:
        raise ValueError(f"Localization file is missing required field(s): {', '.join(sorted(missing))}.")

    for column in locs.columns:
        locs[column] = pd.to_numeric(locs[column], errors="coerce")
    finite_rows = np.isfinite(locs["frame"]) & np.isfinite(locs["x"]) & np.isfinite(locs["y"])
    if not bool(finite_rows.all()):
        locs = locs.loc[finite_rows].copy()
    if locs.empty:
        raise ValueError("No finite localizations with frame, x, and y values were found.")
    if (locs["frame"] < 0).any():
        raise ValueError("Localization frame numbers must be non-negative.")
    locs["frame"] = locs["frame"].astype(np.uint32)

    metadata["Localization count"] = int(len(locs))
    metadata["Fields"] = ", ".join(locs.columns)
    info = picasso_info_from_metadata(metadata, locs)
    return LoadedData(path=path, locs=locs, info=info, metadata=metadata)


def emit_load_progress(
    callback: Callable[[float, str], None] | None,
    percent: float,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0.0, min(100.0, float(percent))), str(message))


def read_locs_hdf5(
    path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LoadedData:
    with h5py.File(path, "r") as h5:
        dataset = find_locs_dataset(h5)
        if dataset.ndim != 1:
            raise ValueError("Localization HDF5 dataset must be one-dimensional.")
        numeric_names = [
            name
            for name in dataset.dtype.names or ()
            if np.issubdtype(dataset.dtype.fields[name][0], np.number)
        ]
        row_count = int(dataset.shape[0])
        data = {name: np.empty(row_count, dtype=float) for name in numeric_names}
        chunk_size = 500_000
        emit_load_progress(progress_callback, 2.0, f"Reading {row_count:,} HDF5 localizations...")
        for start in range(0, row_count, chunk_size):
            end = min(row_count, start + chunk_size)
            raw = dataset[start:end]
            for name in numeric_names:
                data[name][start:end] = np.asarray(raw[name], dtype=float)
            emit_load_progress(
                progress_callback,
                5.0 + 88.0 * end / max(1, row_count),
                f"Reading HDF5 localizations: {end:,}/{row_count:,}",
            )

    metadata = read_yaml_metadata(path)
    metadata["Source format"] = "HDF5"
    return finalize_loaded_locs(path, pd.DataFrame(data), metadata)


def normalized_csv_column(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


CSV_COLUMN_ALIASES: dict[str, tuple[str, bool]] = {
    "frame": ("frame", False),
    "frameindex": ("frame", False),
    "x": ("x", False),
    "xnm": ("x", True),
    "y": ("y", False),
    "ynm": ("y", True),
    "z": ("z", False),
    "znm": ("z", True),
    "photons": ("photons", False),
    "intensityphotons": ("photons", False),
    "sx": ("sx", False),
    "sigmaxnm": ("sx", True),
    "sy": ("sy", False),
    "sigmaynm": ("sy", True),
    "bg": ("bg", False),
    "backgroundphotonsnm2": ("bg", False),
    "lpx": ("lpx", False),
    "lpxnm": ("lpx", True),
    "lpy": ("lpy", False),
    "lpynm": ("lpy", True),
    "localizationprecisionnm": ("precision_nm", False),
    "channelindex": ("channel", False),
}


def count_csv_data_rows(
    path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> int:
    file_size = max(1, path.stat().st_size)
    bytes_read = 0
    line_count = 0
    final_byte = b""
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            bytes_read += len(block)
            line_count += block.count(b"\n")
            final_byte = block[-1:]
            emit_load_progress(
                progress_callback,
                10.0 * bytes_read / file_size,
                f"Scanning CSV: {100.0 * bytes_read / file_size:.0f}%",
            )
    if bytes_read and final_byte != b"\n":
        line_count += 1
    return max(0, line_count - 1)


def read_locs_csv(
    path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LoadedData:
    header = pd.read_csv(path, nrows=0)
    selected: dict[str, tuple[str, bool]] = {}
    mapped_targets: set[str] = set()
    for source_name in header.columns:
        mapped = CSV_COLUMN_ALIASES.get(normalized_csv_column(str(source_name)))
        if mapped is not None and mapped[0] not in mapped_targets:
            selected[str(source_name)] = mapped
            mapped_targets.add(mapped[0])

    required = {"frame", "x", "y"}
    missing = required.difference(mapped_targets)
    if missing:
        raise ValueError(
            "CSV localization file is missing required column(s): "
            f"{', '.join(sorted(missing))}. Expected frame/frameIndex, x/x (nm), and y/y (nm)."
        )

    metadata = read_yaml_metadata(path)
    pixelsize = float(metadata.get("Pixelsize") or DEFAULT_PIXEL_SIZE_NM)
    if not np.isfinite(pixelsize) or pixelsize <= 0:
        raise ValueError("Pixelsize metadata must be a positive finite number.")
    metadata["Pixelsize"] = pixelsize

    row_count = count_csv_data_rows(path, progress_callback)
    # Preallocation keeps loading memory close to the final table size while
    # allowing determinate progress for multi-million-row CSV exports.
    arrays = {mapped[0]: np.empty(row_count, dtype=np.float32) for mapped in selected.values()}
    converted: list[str] = []
    for source_name, (_target_name, is_nm) in selected.items():
        if is_nm:
            converted.append(source_name)

    loaded_rows = 0
    reader = pd.read_csv(
        path,
        usecols=list(selected),
        dtype={name: np.float32 for name in selected},
        chunksize=500_000,
    )
    for chunk in reader:
        end = loaded_rows + len(chunk)
        if end > row_count:
            grow_by = max(end - row_count, max(1, row_count // 10))
            for target_name, values in arrays.items():
                arrays[target_name] = np.resize(values, row_count + grow_by)
            row_count += grow_by
        for source_name, (target_name, is_nm) in selected.items():
            values = chunk[source_name].to_numpy(dtype=np.float32, copy=False)
            if is_nm:
                values = values / np.float32(pixelsize)
            arrays[target_name][loaded_rows:end] = values
        loaded_rows = end
        emit_load_progress(
            progress_callback,
            10.0 + 83.0 * loaded_rows / max(1, row_count),
            f"Loading CSV localizations: {loaded_rows:,}/{row_count:,}",
        )

    locs = pd.DataFrame(
        {target_name: values[:loaded_rows] for target_name, values in arrays.items()},
        copy=False,
    )

    metadata["Source format"] = "CSV"
    metadata["CSV columns"] = ", ".join(str(column) for column in header.columns)
    if converted:
        metadata["CSV nm-to-pixel conversion"] = f"{pixelsize:g} nm/pixel ({', '.join(converted)})"
    return finalize_loaded_locs(path, locs, metadata)


def read_locs(
    path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LoadedData:
    emit_load_progress(progress_callback, 0.0, f"Opening {path.name}...")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        loaded = read_locs_csv(path, progress_callback)
    elif suffix in {".h5", ".hdf5"}:
        loaded = read_locs_hdf5(path, progress_callback)
    else:
        raise ValueError("Unsupported localization file type. Choose a .csv, .h5, or .hdf5 file.")
    emit_load_progress(progress_callback, 100.0, f"Loaded {len(loaded.locs):,} localizations.")
    return loaded


DRIFT_COLUMN_ALIASES: dict[str, tuple[str, bool]] = {
    "frame": ("frame", False),
    "frameindex": ("frame", False),
    "x": ("x", False),
    "driftx": ("x", False),
    "xdrift": ("x", False),
    "xnm": ("x", True),
    "driftxnm": ("x", True),
    "xdriftpixel": ("x", False),
    "xdriftpixels": ("x", False),
    "xdriftnm": ("x", True),
    "y": ("y", False),
    "drifty": ("y", False),
    "ydrift": ("y", False),
    "ynm": ("y", True),
    "driftynm": ("y", True),
    "ydriftpixel": ("y", False),
    "ydriftpixels": ("y", False),
    "ydriftnm": ("y", True),
    "z": ("z", False),
    "driftz": ("z", False),
    "zdrift": ("z", False),
    "znm": ("z", True),
    "driftznm": ("z", True),
    "zdriftpixel": ("z", False),
    "zdriftpixels": ("z", False),
    "zdriftnm": ("z", True),
}


def read_drift_csv(path: Path, frame_count: int, pixel_size_nm: float) -> pd.DataFrame:
    if frame_count < 1:
        raise ValueError("Localization metadata must contain at least one frame.")
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
        raise ValueError("Pixel size must be a positive finite number.")

    header = pd.read_csv(path, nrows=0)
    selected: dict[str, tuple[str, bool]] = {}
    mapped_targets: set[str] = set()
    for source_name in header.columns:
        mapped = DRIFT_COLUMN_ALIASES.get(normalized_csv_column(str(source_name)))
        if mapped is not None and mapped[0] not in mapped_targets:
            selected[str(source_name)] = mapped
            mapped_targets.add(mapped[0])

    missing_columns = {"frame", "x", "y"}.difference(mapped_targets)
    if missing_columns:
        raise ValueError(
            "Drift CSV is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}. Expected Frame and x/y drift columns."
        )

    drift = pd.read_csv(path, usecols=list(selected)).rename(
        columns={name: mapped[0] for name, mapped in selected.items()}
    )
    for column in drift.columns:
        drift[column] = pd.to_numeric(drift[column], errors="coerce")
    if drift.empty:
        raise ValueError("Drift CSV does not contain any rows.")
    if not np.isfinite(drift.to_numpy(dtype=float)).all():
        raise ValueError("Drift CSV contains blank, non-numeric, or non-finite values.")

    frame_values = drift["frame"].to_numpy(dtype=float)
    if np.any(frame_values < 0) or not np.equal(frame_values, np.floor(frame_values)).all():
        raise ValueError("Drift CSV frame numbers must be non-negative integers.")
    drift["frame"] = frame_values.astype(np.int64)
    if drift["frame"].duplicated().any():
        duplicates = drift.loc[drift["frame"].duplicated(), "frame"].head(5).tolist()
        raise ValueError(f"Drift CSV contains duplicate frame numbers: {duplicates}.")

    drift = drift.set_index("frame").sort_index()
    required_frames = pd.RangeIndex(frame_count)
    missing_frames = required_frames.difference(drift.index)
    if len(missing_frames):
        preview = ", ".join(str(frame) for frame in missing_frames[:5])
        suffix = "..." if len(missing_frames) > 5 else ""
        raise ValueError(
            f"Drift CSV does not cover all {frame_count} localization frames; "
            f"missing {len(missing_frames)} frame(s), starting with {preview}{suffix}."
        )

    drift = drift.loc[required_frames, [column for column in ("x", "y", "z") if column in drift.columns]].copy()
    for _source_name, (target_name, is_nm) in selected.items():
        if is_nm and target_name in drift.columns:
            drift[target_name] = drift[target_name].to_numpy(dtype=float) / float(pixel_size_nm)
    drift.index.name = "frame"
    return drift


def apply_drift_file(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_count = int(info[0]["Frames"])
    pixel_size_nm = float(info[0]["Pixelsize"])
    drift = read_drift_csv(path, frame_count, pixel_size_nm)
    localization_frames = locs["frame"].to_numpy(dtype=np.int64, copy=False)
    if localization_frames.size and int(localization_frames.max()) >= frame_count:
        raise ValueError("Localization frame numbers exceed the frame count in the loaded metadata.")

    corrected = locs.copy()
    x_dtype = corrected["x"].dtype if np.issubdtype(corrected["x"].dtype, np.floating) else float
    y_dtype = corrected["y"].dtype if np.issubdtype(corrected["y"].dtype, np.floating) else float
    corrected["x"] = corrected["x"].to_numpy(dtype=x_dtype) - drift["x"].to_numpy(dtype=x_dtype)[localization_frames]
    corrected["y"] = corrected["y"].to_numpy(dtype=y_dtype) - drift["y"].to_numpy(dtype=y_dtype)[localization_frames]
    if "z" in corrected.columns and "z" in drift.columns:
        z_dtype = corrected["z"].dtype if np.issubdtype(corrected["z"].dtype, np.floating) else float
        corrected["z"] = corrected["z"].to_numpy(dtype=z_dtype) - drift["z"].to_numpy(dtype=z_dtype)[localization_frames]
    return corrected, drift


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
    def __init__(
        self,
        callback: Any,
        description: str = "Undrifting by AIM (1/2)",
        phase_count: int = 2,
    ) -> None:
        self.callback = callback
        self.description = description
        self.phase_index = 0
        self.phase_count = max(1, int(phase_count))
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
        self.phase_index = min(self.phase_index + 1, self.phase_count - 1)
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
        overall_percent = min(
            100.0,
            100.0 * (self.phase_index + done / total) / self.phase_count,
        )
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
            extent = self.app._full_map_viewport_nm()
            if extent is None:
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

    GALLERY_OPTIONS = {"Individual origami gallery", "Individual site assignments"}
    DYNAMIC_OPTIONS = {
        "Loaded source data",
        "Identified origami template matches",
        "Random ROI inspection",
    }

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
        if (
            self.app.origami_last_rendered_plot_option in self.DYNAMIC_OPTIONS
            and self.canvas.figure.axes
        ):
            extent = self.app._origami_home_viewport_nm()
            if extent is not None:
                axis = self.canvas.figure.axes[0]
                axis.set_xlim(float(extent[0]), float(extent[1]), emit=False)
                axis.set_ylim(float(extent[2]), float(extent[3]), emit=False)
                self.canvas.draw_idle()
                self.app._schedule_origami_footprint_refresh()
                self.app._schedule_origami_zoom_render(delay_ms=0)
                return
        super().home(*args)
        self._refresh_dynamic_origami_view()

    def back(self, *args: Any) -> None:
        super().back(*args)
        self._refresh_dynamic_origami_view()

    def forward(self, *args: Any) -> None:
        super().forward(*args)
        self._refresh_dynamic_origami_view()

    def _refresh_dynamic_origami_view(self) -> None:
        if self.app.origami_last_rendered_plot_option in self.DYNAMIC_OPTIONS:
            self.app.after_idle(self.app._on_origami_view_limits_changed)


class WidgetTooltip:
    """Small delayed tooltip used by compact Origami controls."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = str(text)
        self.delay_ms = int(delay_ms)
        self.after_id: str | None = None
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._hide()
        self.after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        self.after_id = None
        if not self.widget.winfo_exists():
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window.wm_geometry(f"+{x}+{y}")
        ttk.Label(self.window, text=self.text, padding=(7, 4), wraplength=320, relief="solid").pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None


def apply_drift_correction(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    method: str,
    segmentation: int,
    aim_intersect_nm: float,
    aim_roi_nm: float,
    progress_callback: Any | None = None,
    rcc_lattice_pitch_nm: float = 0.0,
    drift_file_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if method == "none":
        if progress_callback is not None:
            progress_callback("Using loaded coordinates without drift correction.")
        return locs.copy(), None, "No drift correction"

    if method == "file":
        if drift_file_path is None:
            raise ValueError("Choose a drift correction CSV before applying file-based drift correction.")
        if progress_callback is not None:
            progress_callback(f"Loading and applying frame-by-frame drift from {drift_file_path.name}...")
        corrected_locs, drift = apply_drift_file(locs, info, drift_file_path)
        if progress_callback is not None:
            progress_callback("File-based drift correction: 100% complete.")
        return corrected_locs, drift, f"Drift file: {drift_file_path.name}"

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
        # Picasso runs two x/y AIM passes, followed by two z passes for 3D
        # localizations. Account for all four passes so progress reaches 100%
        # only when the entire correction is complete.
        aim_phase_count = 4 if "z" in locs_for_picasso.columns else 2
        aim_progress = (
            PicassoAimStatusProgress(progress_callback, phase_count=aim_phase_count)
            if progress_callback is not None
            else None
        )
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
        "min_blur_width": float(min_blur_width),
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


def histogram_density_limits(
    image: np.ndarray,
    saturation_fraction: float = AUTO_DENSITY_SATURATION_FRACTION,
    histogram_bins: int = AUTO_DENSITY_HISTOGRAM_BINS,
) -> tuple[float, float]:
    """Choose contrast limits from the brightness distribution of populated pixels.

    The upper edge contains all but a small, configurable outlier fraction and is
    given a little display headroom so the retained upper histogram bin does not
    map to the saturated endpoint of the colormap.
    """
    image = np.asarray(image, dtype=float)
    populated = image[np.isfinite(image) & (image > 0)]
    if populated.size == 0:
        return 0.0, 1.0

    populated_min = float(np.min(populated))
    populated_max = float(np.max(populated))
    if populated_max <= populated_min:
        return 0.0, populated_max if populated_max > 0 else 1.0

    saturation_fraction = float(np.clip(saturation_fraction, 0.0, 0.5))
    bin_count = max(2, int(histogram_bins))
    # Logarithmic bins retain useful resolution near the bulk of a long-tailed
    # brightness distribution instead of allowing a few outliers to dominate.
    if populated_min > 0 and populated_max / populated_min >= 100.0:
        edges = np.geomspace(populated_min, populated_max, bin_count + 1)
    else:
        edges = np.linspace(populated_min, populated_max, bin_count + 1)
    counts, edges = np.histogram(populated, bins=edges)
    target_count = max(1, int(math.ceil((1.0 - saturation_fraction) * populated.size)))
    upper_bin = min(int(np.searchsorted(np.cumsum(counts), target_count, side="left")), counts.size - 1)
    retained_upper_edge = float(edges[upper_bin + 1])
    max_density = retained_upper_edge / AUTO_DENSITY_HIGHLIGHT_LEVEL
    return 0.0, max_density if max_density > 0 else 1.0


def iqr_density_limits(image: np.ndarray) -> tuple[float, float]:
    """Backward-compatible alias for the histogram-based automatic limits."""
    return histogram_density_limits(image)


def scale_density_like_picasso(image: np.ndarray, min_density: float, max_density: float) -> tuple[np.ndarray, tuple[float, float]]:
    image = np.asarray(image, dtype=float)
    if max_density <= min_density:
        min_density, max_density = histogram_density_limits(image)
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


def fully_fitting_roi_tiles(
    full_width_nm: float,
    full_height_nm: float,
    validation_roi_nm: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Tile an image on the validation ROI lattice, retaining only complete tiles."""
    x0, x1, y0, y1 = validation_roi_nm
    anchor_x = min(float(x0), float(x1))
    anchor_y = min(float(y0), float(y1))
    tile_width = abs(float(x1) - float(x0))
    tile_height = abs(float(y1) - float(y0))
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("The validation ROI must have positive width and height.")
    if full_width_nm <= 0 or full_height_nm <= 0:
        raise ValueError("The full image must have positive width and height.")

    tolerance = 1e-9
    first_x_step = math.ceil((-anchor_x - tolerance) / tile_width)
    last_x_step = math.floor((full_width_nm - tile_width - anchor_x + tolerance) / tile_width)
    first_y_step = math.ceil((-anchor_y - tolerance) / tile_height)
    last_y_step = math.floor((full_height_nm - tile_height - anchor_y + tolerance) / tile_height)
    x_starts = [anchor_x + step * tile_width for step in range(first_x_step, last_x_step + 1)]
    y_starts = [anchor_y + step * tile_height for step in range(first_y_step, last_y_step + 1)]
    return [
        (start_x, start_x + tile_width, start_y, start_y + tile_height)
        for start_y in y_starts
        for start_x in x_starts
    ]


def evenly_distributed_tile_indices(
    tiles: list[tuple[float, float, float, float]],
    requested_count: int,
) -> np.ndarray:
    """Choose a deterministic spatially distributed subset of a tile lattice."""
    if requested_count < 1:
        raise ValueError("Tile count must be at least 1.")
    tile_count = len(tiles)
    if tile_count == 0:
        return np.empty(0, dtype=int)
    selected_count = min(int(requested_count), tile_count)
    if selected_count == tile_count:
        return np.arange(tile_count, dtype=int)

    centers = np.asarray(
        [[(tile[0] + tile[1]) / 2.0, (tile[2] + tile[3]) / 2.0] for tile in tiles],
        dtype=float,
    )
    lower = np.min(centers, axis=0)
    span = np.maximum(np.max(centers, axis=0) - lower, 1.0)
    normalized = (centers - lower) / span
    field_center = np.asarray([0.5, 0.5])
    first = int(np.argmin(np.sum(np.square(normalized - field_center), axis=1)))
    selected = [first]
    minimum_distance = np.sum(np.square(normalized - normalized[first]), axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < selected_count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum(np.square(normalized - normalized[next_index]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=int)] = -1.0
    return np.asarray(sorted(selected), dtype=int)


def randomized_tile_order(
    tile_count: int,
    excluded_indices: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return every available tile once in random order."""
    if tile_count < 0:
        raise ValueError("Tile count cannot be negative.")
    available = np.asarray(
        [index for index in range(int(tile_count)) if index not in excluded_indices],
        dtype=int,
    )
    return rng.permutation(available) if len(available) else available


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
    if mode == "precision_nm":
        return precision_qc_values(arrays["precision_nm"]), "Localization precision (nm, QC filtered)"
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
    if mode == "precision_nm":
        return precision_qc_series(arrays["precision_nm"], index), "Localization precision (nm, QC filtered)"
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
        # Keep the application usable on laptop-sized and split-screen windows.
        # The control groups below reflow before reaching this lower bound.
        self.minsize(760, 560)

        self.loaded: LoadedData | None = None
        self.corrected_locs: pd.DataFrame | None = None
        self.linked_locs: pd.DataFrame | None = None
        self.filtered_map_locs: pd.DataFrame | None = None
        self.filtered_map_render_context: dict[str, Any] = {}
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
        self.dynamic_render_after_id: str | None = None
        self.load_progress_hide_id: str | None = None
        self.density_refresh_after_id: str | None = None
        self.map_density_images: dict[int, np.ndarray] = {}
        self.dynamic_render_request_id = 0
        self.dynamic_render_running = False
        self.dynamic_render_pending = False
        self.active_notebook_tab = RAW_MAP_TAB
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.session_load_in_progress = False
        self.development_session_cache_path: Path | None = None

        self.exposure_ms = tk.DoubleVar(value=100.0)
        self.pixel_size_nm = tk.DoubleVar(value=DEFAULT_PIXEL_SIZE_NM)
        self.link_radius_nm = tk.DoubleVar(value=75.0)
        self.max_gap_frames = tk.IntVar(value=1)
        self.linking_source = tk.StringVar(value="Corrected map")
        self.linking_scope = tk.StringVar(value="Selected ROI")
        self.drift_method = tk.StringVar(value="none")
        self.drift_file_path: Path | None = None
        self.drift_file_label = tk.StringVar(value="No drift file selected")
        self.drift_segmentation = tk.IntVar(value=1000)
        self.rcc_lattice_pitch_nm = tk.DoubleVar(value=700.0)
        self.aim_intersect_nm = tk.DoubleVar(value=20.0)
        self.aim_roi_nm = tk.DoubleVar(value=60.0)
        self.render_disp_px_nm = tk.DoubleVar(value=10.0)
        self.dynamic_zoom_render = tk.BooleanVar(value=True)
        self.render_blur_method = tk.StringVar(value="smooth")
        self.min_blur_width = tk.DoubleVar(value=1.0)
        self.auto_density_contrast = tk.BooleanVar(value=True)
        self.auto_density_multiplier = tk.DoubleVar(value=1.0)
        self.auto_density_multiplier_label = tk.StringVar(value="Density multiplier: 1×")
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
        self.temporal_annotation_frame = tk.StringVar(value="")
        self.temporal_annotation_label = tk.StringVar(value="")
        self.temporal_annotations: list[TemporalVLineAnnotation] = []
        self.temporal_annotation_artists: list[Any] = []
        self.filter_scope_label = tk.StringVar(value="Filter scope: selected ROI")
        self.filter_bounds_label = tk.StringVar(value="No active histogram filter")
        self.status = tk.StringVar(value="Load a localization CSV or Picasso HDF5 file.")
        self.file_label = tk.StringVar(value="No file loaded")
        self.load_progress_value = tk.DoubleVar(value=0.0)
        self.load_progress_text = tk.StringVar(value="")
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
        self.origami_source_locs: pd.DataFrame | None = None
        self.origami_source_render_result: dict[str, Any] | None = None
        self.origami_pick_result: OrigamiPickResult | None = None
        self.origami_loaded_source_label = ""
        self.origami_loaded_source_path: Path | None = None
        self.origami_loaded_roi_nm: tuple[float, float, float, float] | None = None
        self.origami_loaded_source_params: dict[str, Any] | None = None
        self.origami_identification_params: dict[str, Any] | None = None
        self.origami_result_source = ""
        self.origami_result_source_count = 0
        self.origami_result_render_settings: dict[str, Any] | None = None
        self.origami_result_occupancy_threshold = 1
        self.origami_zoom_render_after_id: str | None = None
        self.origami_zoom_render_request_id = 0
        self.origami_zoom_render_running = False
        self.origami_zoom_render_pending = False
        self.origami_zoom_render_applying = False
        self.origami_source_density_artist: Any | None = None
        self.origami_source_colorbar: Any | None = None
        self.origami_plot_option = tk.StringVar(value="Individual origami gallery")
        self.origami_workflow_stage = tk.StringVar(value="Source")
        self.origami_sidebar_visible = tk.BooleanVar(value=True)
        self.origami_fullscreen_plot = tk.BooleanVar(value=False)
        self.origami_identify_advanced_visible = tk.BooleanVar(value=True)
        self.origami_show_theoretical_overlay = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY
        )
        self.origami_show_alignment_overlay = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY
        )
        self.origami_show_detected_sites_overlay = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY
        )
        self.origami_show_site_diagnostics = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS
        )
        self.origami_show_localization_group_assignments = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS
        )
        self.origami_show_prominence_geometry = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY
        )
        self.origami_show_text_statistics = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS
        )
        self.origami_overlay_advanced_visible = tk.BooleanVar(value=False)
        self.origami_match_panel = tk.StringVar(value="All panels")
        self.origami_settings_state = tk.StringVar(value="Identification settings not yet validated")
        self.origami_identification_baseline: tuple[Any, ...] | None = None
        self.origami_pending_identification_snapshot: tuple[Any, ...] | None = None
        self.origami_popout_window: tk.Toplevel | None = None
        self.origami_last_rendered_plot_option = ""
        self.origami_gallery_view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.origami_gallery_home_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.origami_gallery_page = tk.IntVar(value=1)
        self.origami_gallery_page_size = tk.StringVar(value="64")
        self.origami_gallery_sort = tk.StringVar(value="Origami ID")
        self.origami_gallery_min_match = tk.StringVar(value="")
        self.origami_gallery_max_rms = tk.StringVar(value="")
        self.origami_gallery_page_label = tk.StringVar(value="Page 1/1")
        self.origami_gallery_page_count = 1
        self.origami_gallery_current_indices = np.empty(0, dtype=int)
        self.origami_gallery_tile_hitboxes: list[tuple[float, float, float, float, int]] = []
        self.origami_selected_index: int | None = None
        self.origami_detail_number = tk.IntVar(value=1)
        self.origami_detail_label = tk.StringVar(value="Origami –/–")
        self.origami_navigation_idle_label = tk.StringVar(value="No pages in this view")
        self.origami_density_cache_key: tuple[Any, ...] | None = None
        self.origami_density_cache: dict[str, Any] | None = None
        self.origami_footprint_artists: list[Any] = []
        self.origami_footprint_refresh_after_id: str | None = None
        self.origami_footprint_release_cid: int | None = None
        self.origami_footprint_scroll_cid: int | None = None
        self.origami_identification_progress = tk.DoubleVar(value=0.0)
        self.origami_identification_progress_text = tk.StringVar(value="Ready to identify origami")
        self.origami_step_progress = {
            step: tk.DoubleVar(value=0.0) for step in range(1, 6)
        }
        self.origami_step_progress_text = {
            step: tk.StringVar(value="Not run") for step in range(1, 6)
        }
        self.origami_active_step = 0
        self.origami_match_roi = tk.IntVar(value=1)
        self.origami_match_roi_label = tk.StringVar(value="Candidate –/–")
        self.origami_navigation_kind = tk.StringVar(value="Candidate")
        self.origami_identification_running = False
        self.origami_staged_candidates: tuple[Any, ...] | None = None
        self.origami_staged_candidate_signature: tuple[Any, ...] | None = None
        self.origami_source_candidate_fingerprint = ""
        self.origami_staged_status = tk.StringVar(
            value="Run Step 1, then continue through alignment, digital pixels, classification, and optional tiling."
        )
        self.origami_identification_generation = 0
        self.origami_random_generator = np.random.default_rng()
        self.origami_inspected_tile_indices: set[int] = set()
        self.origami_roi_history: list[dict[str, Any]] = []
        self.origami_roi_history_position = -1
        self.origami_pending_roi_view: str | None = None
        self.origami_selected_match_index: int | None = None
        self.origami_random_inspection_payload: dict[str, Any] | None = None
        self.origami_source = tk.StringVar(value="Corrected localizations")
        self.origami_use_roi = tk.BooleanVar(value=True)
        self.origami_pick_bin_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_PICK_BIN_NM)
        self.origami_connect_distance_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM)
        self.origami_min_density_contrast = tk.DoubleVar(value=DEFAULT_ORIGAMI_MIN_DENSITY)
        self.origami_min_points = tk.IntVar(value=DEFAULT_ORIGAMI_MIN_POINTS)
        self.origami_max_points = tk.IntVar(value=DEFAULT_ORIGAMI_MAX_POINTS)
        self.origami_rows = tk.IntVar(value=DEFAULT_ORIGAMI_ROWS)
        self.origami_columns = tk.IntVar(value=DEFAULT_ORIGAMI_COLUMNS)
        self.origami_template_mode = tk.StringVar(value="Simulated grid")
        self.origami_custom_template_name = tk.StringVar(value="No custom template loaded")
        self.origami_custom_template_path: Path | None = None
        self.origami_custom_template_image: np.ndarray | None = None
        self.origami_custom_templates: list[dict[str, Any]] = []
        self.origami_shared_alignment_template: dict[str, Any] | None = None
        self.origami_shared_alignment_template_name = tk.StringVar(
            value="No alignment template loaded"
        )
        self.origami_digital_pixel_model: dict[str, Any] | None = None
        self.origami_digital_pixel_schema_name = tk.StringVar(
            value="No digital-pixel JSON loaded"
        )
        self.origami_multi_template_results: dict[str, dict[str, Any]] = {}
        self.origami_multi_template_counts: dict[str, int] = {}
        self.origami_multi_template_overlay_results: dict[str, dict[str, Any]] = {}
        self.origami_multi_template_overlays_building: set[str] = set()
        self.origami_single_overlay_building = False
        self.origami_multi_template_unclassified_count = 0
        self.origami_multi_template_unclassified_centers_nm = np.empty((0, 2), dtype=float)
        self.origami_multi_template_unclassified_details: list[dict[str, Any]] = []
        self.origami_template_result_view = tk.StringVar(value="All templates")
        self.origami_template_pixel_x_nm = tk.DoubleVar(value=1.0)
        self.origami_template_pixel_y_nm = tk.DoubleVar(value=1.0)
        self.origami_spacing_x_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_SPACING_X_NM)
        self.origami_spacing_y_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_SPACING_Y_NM)
        self.origami_rectangle_margin_nm = tk.DoubleVar(value=20.0)
        self.origami_min_rectangle_confidence = tk.DoubleVar(value=DEFAULT_ORIGAMI_CORRELATION_THRESHOLD)
        self.origami_use_correlation_gate = tk.BooleanVar(value=DEFAULT_ORIGAMI_USE_CORRELATION_GATE)
        self.origami_min_cell_pattern_correlation = tk.DoubleVar(
            value=DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION
        )
        self.origami_use_cell_pattern_gate = tk.BooleanVar(
            value=DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE
        )
        self.origami_min_monte_carlo_probability = tk.DoubleVar(
            value=DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR
        )
        self.origami_site_mask_radius_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)
        self.origami_min_supported_sites = tk.IntVar(value=5)
        self.origami_min_site_evidence = tk.DoubleVar(value=DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)
        self.origami_min_site_localizations = tk.IntVar(value=DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)
        self.origami_min_supported_rows = tk.IntVar(value=2)
        self.origami_min_supported_columns = tk.IntVar(value=2)
        self.origami_max_site_spacing_error_nm = tk.DoubleVar(value=DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM)
        self.origami_preview_pixel_nm = tk.DoubleVar(value=1.0)
        self.origami_alignment_max_pixels = tk.IntVar(value=DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS)
        self.origami_alignment_iterations = tk.IntVar(value=DEFAULT_ORIGAMI_ALIGNMENT_PASSES)
        self.origami_tile_count = tk.IntVar(value=100)
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
        self.after(150, self._restore_latest_development_session)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar_outer = ttk.Frame(self)
        sidebar_outer.grid(row=0, column=0, sticky="nsew")
        self.global_sidebar_outer = sidebar_outer
        sidebar_outer.columnconfigure(0, weight=1)
        sidebar_outer.rowconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_outer, width=300, highlightthickness=0)
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

        ttk.Label(sidebar, text="DNA PAINT ROI Analyzer", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Button(sidebar, text="Load Locs File", command=self.load_file).grid(row=1, column=0, sticky="ew")
        file_status = ttk.Frame(sidebar)
        file_status.grid(row=2, column=0, sticky="ew", pady=(8, 12))
        file_status.columnconfigure(0, weight=1)
        ttk.Label(file_status, textvariable=self.file_label, wraplength=250).grid(row=0, column=0, sticky="ew")
        self.load_progress_bar = ttk.Progressbar(
            file_status,
            variable=self.load_progress_value,
            maximum=100.0,
            mode="determinate",
        )
        self.load_progress_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.load_progress_label = ttk.Label(file_status, textvariable=self.load_progress_text, wraplength=250)
        self.load_progress_label.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.load_progress_bar.grid_remove()
        self.load_progress_label.grid_remove()

        roi_box = ttk.LabelFrame(sidebar, text="ROI", padding=10)
        roi_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        roi_box.columnconfigure(0, weight=1)
        ttk.Label(roi_box, textvariable=self.roi_label, wraplength=250).grid(row=0, column=0, sticky="ew")
        ttk.Button(roi_box, text="Clear ROI", command=self.clear_roi).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        drift_box = ttk.LabelFrame(sidebar, text="Drift Correction", padding=10)
        drift_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        drift_box.columnconfigure(1, weight=1)
        self._number_row(drift_box, 0, "Pixel size (nm)", self.pixel_size_nm)
        ttk.Label(drift_box, text="Drift method").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(drift_box, textvariable=self.drift_method, state="readonly", values=("none", "rcc", "aim", "file")).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        self._number_row(drift_box, 2, "Segmentation", self.drift_segmentation)
        self._number_row(drift_box, 3, "RCC lattice pitch (nm)", self.rcc_lattice_pitch_nm)
        self._number_row(drift_box, 4, "AIM intersect (nm)", self.aim_intersect_nm)
        self._number_row(drift_box, 5, "AIM ROI (nm)", self.aim_roi_nm)
        ttk.Button(drift_box, text="Load Drift CSV", command=self.load_drift_file).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(drift_box, textvariable=self.drift_file_label, wraplength=250).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(drift_box, text="Apply Drift Correction", command=self.apply_correction).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        render_box = ttk.LabelFrame(sidebar, text="Render Settings", padding=10)
        render_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        render_box.columnconfigure(1, weight=1)
        self._number_row(render_box, 0, "Render pixel (nm)", self.render_disp_px_nm)
        ttk.Label(render_box, text="Render blur").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(render_box, textvariable=self.render_blur_method, state="readonly", values=("smooth", "none", "gaussian", "gaussian_iso", "convolve")).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        self._number_row(render_box, 2, "Min blur (px)", self.min_blur_width)
        ttk.Checkbutton(
            render_box,
            text="Auto density (histogram)",
            variable=self.auto_density_contrast,
            command=self._on_auto_density_toggled,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(render_box, textvariable=self.auto_density_multiplier_label).grid(row=4, column=0, sticky="w", pady=3)
        ttk.Scale(
            render_box,
            from_=0.1,
            to=10.0,
            variable=self.auto_density_multiplier,
            command=self._on_auto_density_multiplier_changed,
        ).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=3)
        self._number_row(render_box, 5, "Min density", self.render_min_density)
        self._number_row(render_box, 6, "Max density", self.render_max_density)
        ttk.Checkbutton(render_box, text="Dynamic zoom rendering (minimum 1 nm/pixel)", variable=self.dynamic_zoom_render).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(render_box, text="Render Raw Map", command=self.show_raw_map).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(render_box, text="Render Corrected Map", command=self.show_current_map).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(render_box, text="Render Linked Map", command=self.color_by_links).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 0))

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
        self.meta_text = tk.Text(meta_box, width=30, height=10, wrap="word", state="disabled")
        self.meta_text.grid(row=0, column=0, sticky="nsew")
        meta_box.columnconfigure(0, weight=1)
        meta_box.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky="nsew")
        self.main_content = main
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(main)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.main_top_bar = top_bar
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
        self.notebook.add(raw_map_tab, text="Raw")

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
        self.notebook.add(map_tab, text="Corrected")

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
        self.notebook.add(linked_map_tab, text="Linked")

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
        self.notebook.add(filtered_map_tab, text="Filtered")

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
        hist_controls.columnconfigure(0, weight=1)
        hist_roi_label = ttk.Label(hist_controls, textvariable=self.roi_label, wraplength=520)
        hist_roi_label.grid(row=0, column=0, sticky="ew", pady=(0, 3))

        hist_actions = ttk.Frame(hist_controls)
        hist_actions.grid(row=1, column=0, sticky="ew")
        show_map_button = ttk.Button(hist_actions, text="Show Corrected Map", command=self.show_current_map)
        clear_roi_button = ttk.Button(hist_actions, text="Clear ROI", command=self.clear_roi)
        self._bind_responsive_grid(hist_actions, [show_map_button, clear_roi_button], 170, 2)

        hist_fields = ttk.Frame(hist_controls)
        hist_fields.grid(row=2, column=0, sticky="ew")

        def histogram_field(
            label: str,
            control_factory: Callable[[ttk.Frame], tk.Widget],
        ) -> tuple[ttk.Frame, tk.Widget]:
            field = ttk.Frame(hist_fields)
            field.columnconfigure(0, weight=1)
            ttk.Label(field, text=label).grid(row=0, column=0, sticky="w")
            control = control_factory(field)
            control.grid(row=1, column=0, sticky="ew", pady=(2, 0))
            return field, control

        hist_mode_field, self.hist_combo = histogram_field(
            "Histogram",
            lambda parent: ttk.Combobox(parent, textvariable=self.hist_mode, state="readonly", values=()),
        )
        hist_scope_field, _ = histogram_field(
            "Filter maps",
            lambda parent: ttk.Combobox(
                parent,
                textvariable=self.hist_filter_scope,
                state="readonly",
                values=("ROI localizations", "Entire image", "Linked events in ROI", "Linked events entire image"),
            ),
        )
        hist_frame_start_field, _ = histogram_field(
            "Frame start", lambda parent: ttk.Entry(parent, textvariable=self.hist_frame_start, width=12)
        )
        hist_frame_end_field, _ = histogram_field(
            "Frame end", lambda parent: ttk.Entry(parent, textvariable=self.hist_frame_end, width=12)
        )
        filtered_source_field, _ = histogram_field(
            "Filtered map source",
            lambda parent: ttk.Combobox(
                parent,
                textvariable=self.filtered_map_source,
                state="readonly",
                values=("Corrected map", "Raw map", "Linked map"),
            ),
        )
        hist_bin_field, _ = histogram_field(
            "Bin size", lambda parent: ttk.Entry(parent, textvariable=self.hist_bin_size, width=12)
        )
        histogram_fields = [
            hist_mode_field,
            hist_scope_field,
            hist_frame_start_field,
            hist_frame_end_field,
            filtered_source_field,
            hist_bin_field,
        ]
        self._bind_responsive_grid(hist_fields, histogram_fields, 160, 2)

        histogram_buttons = ttk.Frame(hist_controls)
        histogram_buttons.grid(row=3, column=0, sticky="ew")
        plot_hist_button = ttk.Button(histogram_buttons, text="Plot Histogram", command=self.plot_roi_histogram)
        apply_filters_button = ttk.Button(histogram_buttons, text="Apply Histogram Filters", command=self.apply_histogram_filters_to_maps)
        export_hist_button = ttk.Button(histogram_buttons, text="Export Current CSV", command=self.export_csv)
        self._bind_responsive_grid(
            histogram_buttons,
            [plot_hist_button, apply_filters_button, export_hist_button],
            170,
            3,
        )
        ttk.Label(hist_controls, textvariable=self.filter_bounds_label).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.active_filters_frame = ttk.LabelFrame(hist_controls, text="Active Histogram Filters", padding=8)
        self.active_filters_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.active_filters_frame.columnconfigure(0, weight=1)
        self._refresh_filter_list()

        hist_controls.bind(
            "<Configure>",
            lambda event: hist_roi_label.configure(wraplength=max(180, event.width - 30)),
            add="+",
        )

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
        self.notebook.add(temporal_tab, text="Temporal")

        temporal_controls = ttk.LabelFrame(temporal_tab, text="Temporal Metric Plot", padding=10)
        temporal_controls.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 6))
        temporal_controls.columnconfigure(0, weight=1)
        temporal_fields = ttk.Frame(temporal_controls)
        temporal_fields.grid(row=0, column=0, sticky="ew")

        def temporal_field(
            label: str,
            control_factory: Callable[[ttk.Frame], tk.Widget],
        ) -> tuple[ttk.Frame, tk.Widget]:
            field = ttk.Frame(temporal_fields)
            field.columnconfigure(0, weight=1)
            ttk.Label(field, text=label).grid(row=0, column=0, sticky="w")
            control = control_factory(field)
            control.grid(row=1, column=0, sticky="ew", pady=(2, 0))
            return field, control

        temporal_metric_field, self.temporal_combo = temporal_field(
            "Metric",
            lambda parent: ttk.Combobox(parent, textvariable=self.temporal_mode, state="readonly", values=()),
        )
        temporal_start_field, _ = temporal_field(
            "Frame start", lambda parent: ttk.Entry(parent, textvariable=self.temporal_frame_start, width=12)
        )
        temporal_end_field, _ = temporal_field(
            "Frame end", lambda parent: ttk.Entry(parent, textvariable=self.temporal_frame_end, width=12)
        )
        temporal_window_field, _ = temporal_field(
            "Window (frames)", lambda parent: ttk.Entry(parent, textvariable=self.temporal_window_frames, width=12)
        )
        temporal_step_field, _ = temporal_field(
            "Step (frames)", lambda parent: ttk.Entry(parent, textvariable=self.temporal_step_frames, width=12)
        )
        temporal_stat_field, _ = temporal_field(
            "Statistic",
            lambda parent: ttk.Combobox(
                parent,
                textvariable=self.temporal_stat,
                state="readonly",
                values=("mean", "median", "IQR mean"),
            ),
        )
        self._bind_responsive_grid(
            temporal_fields,
            [
                temporal_metric_field,
                temporal_start_field,
                temporal_end_field,
                temporal_window_field,
                temporal_step_field,
                temporal_stat_field,
            ],
            180,
            2,
        )

        temporal_actions = ttk.Frame(temporal_controls)
        temporal_actions.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        use_roi_check = ttk.Checkbutton(temporal_actions, text="Use selected ROI", variable=self.temporal_use_roi)
        use_linked_check = ttk.Checkbutton(temporal_actions, text="Use linked events", variable=self.temporal_use_linked)
        temporal_plot_button = ttk.Button(temporal_actions, text="Plot Temporal Metric", command=self.plot_temporal_metric)
        self._bind_responsive_grid(
            temporal_actions,
            [use_roi_check, use_linked_check, temporal_plot_button],
            170,
            3,
        )

        annotation_box = ttk.LabelFrame(temporal_controls, text="Vertical frame annotations", padding=8)
        annotation_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        annotation_box.columnconfigure(3, weight=1)
        ttk.Label(annotation_box, text="Frame").grid(row=0, column=0, sticky="w")
        annotation_frame_entry = ttk.Entry(
            annotation_box, textvariable=self.temporal_annotation_frame, width=11
        )
        annotation_frame_entry.grid(row=0, column=1, sticky="ew", padx=(5, 12))
        ttk.Label(annotation_box, text="Label").grid(row=0, column=2, sticky="w")
        annotation_label_entry = ttk.Entry(
            annotation_box, textvariable=self.temporal_annotation_label
        )
        annotation_label_entry.grid(row=0, column=3, sticky="ew", padx=(5, 8))
        ttk.Button(
            annotation_box,
            text="Add / Update",
            command=self.add_temporal_annotation,
        ).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(
            annotation_box,
            text="Remove selected",
            command=self.remove_selected_temporal_annotations,
        ).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(
            annotation_box,
            text="Clear all",
            command=self.clear_temporal_annotations,
        ).grid(row=0, column=6)
        annotation_frame_entry.bind("<Return>", lambda _event: self.add_temporal_annotation())
        annotation_label_entry.bind("<Return>", lambda _event: self.add_temporal_annotation())
        self.temporal_annotation_tree = ttk.Treeview(
            annotation_box,
            columns=("Frame", "Label"),
            show="headings",
            height=3,
            selectmode="extended",
        )
        self.temporal_annotation_tree.heading("Frame", text="Frame")
        self.temporal_annotation_tree.heading("Label", text="Annotation text")
        self.temporal_annotation_tree.column("Frame", width=85, stretch=False, anchor="e")
        self.temporal_annotation_tree.column("Label", width=500, stretch=True)
        self.temporal_annotation_tree.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(7, 0))
        self.temporal_annotation_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._load_selected_temporal_annotation()
        )

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
        origami_tab.rowconfigure(0, weight=1)
        self.notebook.add(origami_tab, text="Origami")
        self.origami_tab = origami_tab

        self.origami_workspace = ttk.Frame(origami_tab)
        self.origami_workspace.grid(row=0, column=0, sticky="nsew")
        self.origami_workspace.columnconfigure(1, weight=1)
        self.origami_workspace.rowconfigure(0, weight=1)

        # The staged workflow contains paired numeric fields and descriptive
        # controls that cannot be read reliably in the old 310 px column,
        # especially with macOS display scaling. Keep a useful plot area while
        # giving the controls enough width even before fullscreen is enabled.
        self.origami_sidebar = ttk.Frame(self.origami_workspace, width=370, padding=(6, 6, 4, 6))
        self.origami_sidebar.grid(row=0, column=0, sticky="nsew")
        self.origami_sidebar.grid_propagate(False)
        self.origami_sidebar.columnconfigure(0, weight=1)
        self.origami_sidebar.rowconfigure(1, weight=1)

        stage_bar = ttk.Frame(self.origami_sidebar)
        stage_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        for column, (stage, label) in enumerate(
            (("Source", "1 Source"), ("Identify", "2 Identify"), ("Overlay", "3 Overlay"))
        ):
            stage_bar.columnconfigure(column, weight=1)
            button = ttk.Radiobutton(
                stage_bar,
                text=label,
                variable=self.origami_workflow_stage,
                value=stage,
                command=lambda selected=stage: self._show_origami_stage(selected),
                style="Toolbutton",
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 2, 0))
            WidgetTooltip(button, f"Show the {stage.lower()} stage settings.")

        sidebar_body = ttk.Frame(self.origami_sidebar)
        sidebar_body.grid(row=1, column=0, sticky="nsew")
        sidebar_body.columnconfigure(0, weight=1)
        sidebar_body.rowconfigure(0, weight=1)
        self.origami_settings_canvas = tk.Canvas(sidebar_body, highlightthickness=0, borderwidth=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_body, orient="vertical", command=self.origami_settings_canvas.yview)
        self.origami_settings_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.origami_settings_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        self.origami_settings_content = ttk.Frame(self.origami_settings_canvas, padding=(5, 3, 7, 8))
        self.origami_settings_content.columnconfigure(0, weight=1)
        settings_window = self.origami_settings_canvas.create_window(
            (0, 0), window=self.origami_settings_content, anchor="nw"
        )
        self.origami_settings_content.bind(
            "<Configure>",
            lambda _event: self.origami_settings_canvas.configure(scrollregion=self.origami_settings_canvas.bbox("all")),
        )
        self.origami_settings_canvas.bind(
            "<Configure>",
            lambda event: self.origami_settings_canvas.itemconfigure(settings_window, width=event.width),
        )

        def scroll_origami_settings(event: tk.Event) -> None:
            if getattr(event, "num", None) == 4:
                units = -1
            elif getattr(event, "num", None) == 5:
                units = 1
            else:
                delta = int(getattr(event, "delta", 0))
                units = -1 if delta > 0 else 1 if delta < 0 else 0
            if units:
                self.origami_settings_canvas.yview_scroll(units, "units")

        def bind_settings_scroll(widget: tk.Widget) -> None:
            widget.bind("<MouseWheel>", scroll_origami_settings, add="+")
            widget.bind("<Button-4>", scroll_origami_settings, add="+")
            widget.bind("<Button-5>", scroll_origami_settings, add="+")
            for child in widget.winfo_children():
                bind_settings_scroll(child)

        self.origami_stage_frames: dict[str, ttk.Frame] = {}

        def stage_frame(name: str, title: str) -> ttk.Frame:
            frame = ttk.Frame(self.origami_settings_content)
            frame.columnconfigure(0, weight=1)
            ttk.Label(frame, text=title, font=("TkDefaultFont", 12, "bold")).grid(
                row=0, column=0, sticky="w", pady=(0, 8)
            )
            self.origami_stage_frames[name] = frame
            return frame

        def setting_row(
            parent: ttk.Frame,
            row: int,
            label: str,
            variable: tk.Variable,
            help_text: str,
        ) -> ttk.Entry:
            # Keep the value column usable in the narrow, scrollable sidebar.
            # Without a wrap length, long labels claim their full requested
            # width and Tk clips the controls at the right edge.
            label_widget = ttk.Label(parent, text=label, wraplength=125, justify="left")
            label_widget.grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(parent, textvariable=variable, width=10)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
            WidgetTooltip(label_widget, help_text)
            WidgetTooltip(entry, help_text)
            return entry

        def paired_setting_row(
            parent: ttk.Frame,
            row: int,
            label: str,
            left_variable: tk.Variable,
            separator: str,
            right_variable: tk.Variable,
            help_text: str,
        ) -> tuple[ttk.Entry, ttk.Entry]:
            """Lay out two equally sized values without letting either get clipped."""
            label_widget = ttk.Label(parent, text=label, wraplength=125, justify="left")
            label_widget.grid(row=row, column=0, sticky="w", pady=3)
            values = ttk.Frame(parent)
            values.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
            values.columnconfigure(0, weight=1, uniform=f"origami_pair_{row}")
            values.columnconfigure(2, weight=1, uniform=f"origami_pair_{row}")
            left_entry = ttk.Entry(values, textvariable=left_variable, width=1)
            left_entry.grid(row=0, column=0, sticky="ew")
            ttk.Label(values, text=separator).grid(row=0, column=1)
            right_entry = ttk.Entry(values, textvariable=right_variable, width=1)
            right_entry.grid(row=0, column=2, sticky="ew")
            for widget in (label_widget, values, left_entry, right_entry):
                WidgetTooltip(widget, help_text)
            return left_entry, right_entry

        def workflow_group(parent: ttk.Frame, row: int, title: str) -> ttk.LabelFrame:
            group = ttk.LabelFrame(parent, text=title, padding=7)
            group.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            group.columnconfigure(1, weight=1)
            return group

        def step_progress(parent: ttk.Frame, row: int, step: int) -> None:
            ttk.Progressbar(
                parent,
                variable=self.origami_step_progress[step],
                maximum=100.0,
                mode="determinate",
            ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 1))
            ttk.Label(
                parent,
                textvariable=self.origami_step_progress_text[step],
                wraplength=245,
            ).grid(row=row + 1, column=0, columnspan=2, sticky="ew")

        source_fields = stage_frame("Source", "Source Data")
        source_form = ttk.Frame(source_fields)
        source_form.grid(row=1, column=0, sticky="ew")
        source_form.columnconfigure(1, weight=1)
        source_label = ttk.Label(source_form, text="Source")
        source_label.grid(row=0, column=0, sticky="w", pady=3)
        source_combo = ttk.Combobox(
            source_form,
            textvariable=self.origami_source,
            state="readonly",
            width=1,
            values=("Filtered linked events", "Linked events", "Filtered corrected localizations", "Corrected localizations"),
        )
        source_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        WidgetTooltip(source_combo, "Choose corrected localizations or linked events, optionally after active histogram filters.")
        source_roi = ttk.Checkbutton(source_form, text="Use selected ROI", variable=self.origami_use_roi)
        source_roi.grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        WidgetTooltip(source_roi, "Restrict source loading to the rectangle selected on a localization map.")
        ttk.Label(
            source_fields,
            text="Reload after changing the source, selected ROI, or active histogram filters.",
            wraplength=270,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 4))
        ttk.Button(source_fields, text="Reset Source Defaults", command=lambda: self._reset_origami_stage("Source")).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

        identify_fields = stage_frame("Identify", "Identify Origami")
        identify_form = ttk.Frame(identify_fields)
        identify_form.grid(row=1, column=0, sticky="ew")
        identify_form.columnconfigure(0, weight=1)

        coarse_group = workflow_group(identify_form, 0, "1 · Coarse candidate detection")
        setting_row(coarse_group, 0, "Pick bin (nm)", self.origami_pick_bin_nm, "Coarse density-map bin size in nanometres.")
        setting_row(coarse_group, 1, "Minimum density", self.origami_min_density_contrast, "Normalized coarse-density threshold used to seed candidate regions.")
        setting_row(
            coarse_group,
            2,
            "Connect distance (nm)",
            self.origami_connect_distance_nm,
            "Gap used to recover points around supported bins. Custom templates also join bins across one configured site pitch so hollow shapes stay intact.",
        )
        setting_row(
            coarse_group,
            3,
            "Min candidate points",
            self.origami_min_points,
            "Components below this localization count are discarded during Step 1. This is the same minimum shown in the Step 4 point limits.",
        )
        ttk.Label(
            coarse_group,
            text="These settings determine which objects become candidates at all. The minimum-point control is shared with Step 4.",
            wraplength=245,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.origami_run_candidates_button = ttk.Button(
            coarse_group,
            text="Run Step 1 · Inspect Candidates",
            command=self._run_origami_candidate_stage,
        )
        self.origami_run_candidates_button.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        step_progress(coarse_group, 6, 1)

        template_group = workflow_group(identify_form, 1, "2 · Fit alignment template to candidates")
        ttk.Label(template_group, text="Alignment template", wraplength=125, justify="left").grid(
            row=0, column=0, sticky="w", pady=3
        )
        template_mode = ttk.Combobox(
            template_group,
            textvariable=self.origami_template_mode,
            values=("Simulated grid", "Custom image"),
            state="readonly",
            width=1,
        )
        template_mode.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        WidgetTooltip(template_mode, "Use the generated full grid or a bright-on-dark uploaded image for alignment and correlation.")
        paired_setting_row(
            template_group,
            1,
            "Grid rows × columns",
            self.origami_rows,
            " × ",
            self.origami_columns,
            "Physical row and column count in the theoretical docking-site design.",
        )
        paired_setting_row(
            template_group,
            2,
            "Spacing x / y (nm)",
            self.origami_spacing_x_nm,
            " / ",
            self.origami_spacing_y_nm,
            "Theoretical x/y docking-site spacing in nanometres.",
        )
        paired_setting_row(
            template_group,
            3,
            "Template pixel x / y (nm)",
            self.origami_template_pixel_x_nm,
            " / ",
            self.origami_template_pixel_y_nm,
            "Nanometres between adjacent template pixel centers. Loaded automatically from Picklist Generator metadata; editable for older images.",
        )
        setting_row(template_group, 4, "Image margin (nm)", self.origami_rectangle_margin_nm, "Extra nanometres around the theoretical grid retained in each candidate footprint.")
        ttk.Button(
            template_group,
            text="Load Alignment Template…",
            command=self._load_origami_shared_alignment_template,
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(
            template_group,
            text="Clear Alignment Template",
            command=self._clear_origami_shared_alignment_template,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Label(
            template_group,
            textvariable=self.origami_shared_alignment_template_name,
            wraplength=245,
            foreground="#3d626f",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 0))
        setting_row(template_group, 8, "Alignment pixel (nm)", self.origami_preview_pixel_nm, "Requested nanometres per pixel for candidate/template alignment.")
        setting_row(template_group, 9, "Max alignment pixels", self.origami_alignment_max_pixels, "Maximum pixels across the diagonal of each candidate/template alignment image. Increase this to honor finer alignment-pixel requests.")
        setting_row(template_group, 10, "Alignment passes", self.origami_alignment_iterations, "Number of residual rotational/template-alignment passes.")
        setting_row(
            template_group,
            11,
            "Pose site radius (nm)",
            self.origami_site_mask_radius_nm,
            "The sparse-pose selector uses this radius during alignment. This is the same radius shown in Step 3.",
        )
        setting_row(
            template_group,
            12,
            "Pose min locs / site",
            self.origami_min_site_localizations,
            "The sparse-pose selector uses this localization floor during alignment. This is the same value shown in Step 3.",
        )
        setting_row(
            template_group,
            13,
            "Pose min supported sites",
            self.origami_min_supported_sites,
            "Alignment uses this many supported sites when choosing among candidate poses. This is the same value shown in Step 4.",
        )
        ttk.Label(
            template_group,
            text="Fits the uploaded alignment image, then locks rotation and translation for digital-pixel detection.",
            wraplength=245,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.origami_run_alignment_button = ttk.Button(
            template_group,
            text="Run Step 2 · Fit and Inspect",
            command=lambda: self.identify_origamis(target_stage=3, display_stage=2),
        )
        self.origami_run_alignment_button.grid(
            row=15, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        step_progress(template_group, 16, 2)

        ttk.Label(identify_fields, text="Digital detection, classification, and QC").grid(row=2, column=0, sticky="w", pady=(2, 4))
        self.origami_identify_advanced_frame = ttk.Frame(identify_fields)
        self.origami_identify_advanced_frame.columnconfigure(0, weight=1)

        site_group = workflow_group(self.origami_identify_advanced_frame, 0, "3 · Detect digital-pixel outcomes")
        ttk.Button(
            site_group,
            text="Load Digital Pixel Groups JSON…",
            command=self._load_origami_digital_pixel_schema,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(
            site_group,
            textvariable=self.origami_digital_pixel_schema_name,
            wraplength=245,
            foreground="#3d626f",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        setting_row(site_group, 2, "Site mask radius (nm)", self.origami_site_mask_radius_nm, "Radius around each theoretical docking site used for the localization-count floor and measured centroid.")
        setting_row(site_group, 3, "Min group prominence", self.origami_min_site_evidence, "Minimum 0–1 ON-separation strength required for a digital group. This exact threshold is shown in the group-decision labels and used by Step 4.")
        setting_row(site_group, 4, "Min locs / group position", self.origami_min_site_localizations, "Minimum directly assigned localization support per position in a digital group. This exact threshold is shown in the group-decision labels and used by Step 4.")
        ttk.Label(
            site_group,
            text="A digital group is ON only when both its support per position and group prominence pass these thresholds.",
            wraplength=245,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.origami_run_sites_button = ttk.Button(
            site_group,
            text="Run Step 3 · Inspect Digital Pixels",
            command=lambda: self.identify_origamis(target_stage=4, display_stage=3),
        )
        self.origami_run_sites_button.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        step_progress(site_group, 7, 3)

        acceptance_group = workflow_group(self.origami_identify_advanced_frame, 1, "4 · Fit templates to digital-pixel outcomes")
        ttk.Button(
            acceptance_group,
            text="Load Classification Templates…",
            command=self._load_multiple_origami_custom_templates,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        ttk.Label(
            acceptance_group,
            textvariable=self.origami_custom_template_name,
            wraplength=245,
            foreground="#3d626f",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        point_limits = ttk.Frame(acceptance_group)
        point_limits.grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
        point_limits.columnconfigure(1, weight=1)
        point_limits.columnconfigure(3, weight=1)
        ttk.Label(point_limits, text="Point limits").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(point_limits, textvariable=self.origami_min_points, width=6).grid(row=0, column=1, sticky="ew")
        ttk.Label(point_limits, text="to").grid(row=0, column=2, padx=7)
        ttk.Entry(point_limits, textvariable=self.origami_max_points, width=6).grid(row=0, column=3, sticky="ew")
        WidgetTooltip(point_limits, "Accept candidates only when their cropped localization count lies in this inclusive range.")
        setting_row(acceptance_group, 3, "Min supported sites", self.origami_min_supported_sites, "Minimum independently supported docking sites. This value is also reused while selecting the best sparse alignment pose.")
        setting_row(acceptance_group, 4, "Min supported rows", self.origami_min_supported_rows, "Supported sites must span at least this many template rows.")
        setting_row(acceptance_group, 5, "Min supported columns", self.origami_min_supported_columns, "Supported sites must span at least this many template columns.")
        setting_row(acceptance_group, 6, "Max spacing error (nm)", self.origami_max_site_spacing_error_nm, "Largest permitted error between any pair of supported-site centroid spacings and the corresponding theoretical grid spacing.")
        setting_row(acceptance_group, 7, "Correlation threshold", self.origami_min_rectangle_confidence, "Optional normalized full-template correlation gate from 0 to 1; retained as a diagnostic when its gate is disabled.")
        setting_row(
            acceptance_group,
            8,
            "Min winning-template probability",
            self.origami_min_monte_carlo_probability,
            "For digital classification, require the winner's equal-prior probability after all loaded templates compete. With six templates, 0.167 is chance; the default 0.25 requires evidence above chance without demanding an absolute majority.",
        )
        setting_row(
            acceptance_group,
            9,
            "Min cell-pattern correlation",
            self.origami_min_cell_pattern_correlation,
            "Optional multiple-template gate comparing the complete detected bright/dark lattice with each template. It remains part of the soft classification score even when this hard gate is disabled.",
        )
        correlation_gate_toggle = ttk.Checkbutton(
            acceptance_group,
            text="Use correlation acceptance gate",
            variable=self.origami_use_correlation_gate,
        )
        correlation_gate_toggle.grid(row=10, column=0, columnspan=2, sticky="w", pady=(3, 0))
        WidgetTooltip(correlation_gate_toggle, "Enabled by default at a 0.30 threshold; disable it to retain correlation for QC only.")
        pattern_gate_toggle = ttk.Checkbutton(
            acceptance_group,
            text="Use cell-pattern acceptance gate",
            variable=self.origami_use_cell_pattern_gate,
        )
        pattern_gate_toggle.grid(row=11, column=0, columnspan=2, sticky="w", pady=(3, 0))
        WidgetTooltip(
            pattern_gate_toggle,
            "Disabled by default because thresholded site detection can be noisy. Enable it to reject template fits below the adjustable cell-pattern threshold.",
        )
        self.origami_run_acceptance_button = ttk.Button(
            acceptance_group,
            text="Run Step 4 · Classify Templates",
            command=lambda: self.identify_origamis(target_stage=5, display_stage=4),
        )
        self.origami_run_acceptance_button.grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        step_progress(acceptance_group, 13, 4)

        ttk.Button(identify_fields, text="View Coarse Density Map", command=self._show_origami_coarse_density).grid(
            row=4, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Progressbar(
            identify_fields,
            variable=self.origami_identification_progress,
            maximum=100.0,
            mode="determinate",
        ).grid(row=5, column=0, sticky="ew", pady=(10, 2))
        ttk.Label(identify_fields, textvariable=self.origami_identification_progress_text, wraplength=270).grid(
            row=6, column=0, sticky="ew"
        )
        ttk.Label(identify_fields, textvariable=self.origami_settings_state, wraplength=270).grid(
            row=7, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Label(
            identify_fields,
            textvariable=self.origami_staged_status,
            wraplength=270,
            foreground="#3d626f",
        ).grid(row=8, column=0, sticky="ew", pady=(5, 0))
        tile_box = ttk.LabelFrame(identify_fields, text="5 · Optional tile or whole-image analysis", padding=6)
        tile_box.grid(row=9, column=0, sticky="ew", pady=(10, 0))
        tile_box.columnconfigure(1, weight=1)
        setting_row(tile_box, 0, "Tiles to analyze", self.origami_tile_count, "Number of spatially distributed complete ROI-sized tiles to analyze.")
        self.origami_n_tiles_button = ttk.Button(
            tile_box,
            text="Analyze N Tiles",
            command=lambda: self.analyze_tiled_origamis(use_tile_limit=True),
        )
        self.origami_n_tiles_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.origami_n_tiles_button.state(["disabled"])
        WidgetTooltip(
            self.origami_n_tiles_button,
            "Analyze the requested number of spatially distributed tiles. Available after an accepted identification run loaded from a selected ROI.",
        )
        self.origami_random_roi_button = ttk.Button(
            tile_box, text="Inspect Random ROI", command=self.inspect_random_origami_roi
        )
        self.origami_random_roi_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.origami_random_roi_button.state(["disabled"])
        WidgetTooltip(
            self.origami_random_roi_button,
            "Run the validated settings on a random, previously unseen complete ROI-sized tile and display its candidates.",
        )
        self.origami_tiled_button = ttk.Button(
            tile_box, text="Analyze Whole Image as Tiles", command=lambda: self.analyze_tiled_origamis(use_tile_limit=False)
        )
        self.origami_tiled_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.origami_tiled_button.state(["disabled"])
        WidgetTooltip(self.origami_tiled_button, "Available after an accepted identification run loaded from a selected ROI.")
        step_progress(tile_box, 4, 5)
        ttk.Label(
            tile_box,
            text="Validate identification in one ROI first; its dimensions define the tile lattice.",
            wraplength=320,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(identify_fields, text="Reset Identification Defaults", command=lambda: self._reset_origami_stage("Identify")).grid(
            row=10, column=0, sticky="ew", pady=(9, 0)
        )

        overlay_fields = stage_frame("Overlay", "Overlay and Statistics")
        overlay_form = ttk.Frame(overlay_fields)
        overlay_form.grid(row=1, column=0, sticky="ew")
        overlay_form.columnconfigure(1, weight=1)
        mirror_check = ttk.Checkbutton(overlay_form, text="Allow mirrored orientations", variable=self.origami_allow_mirror)
        mirror_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        WidgetTooltip(mirror_check, "Permit reflected template orientations during overlay alignment.")
        setting_row(overlay_form, 1, "Site radius (nm)", self.origami_site_radius_nm, "Maximum nanometre distance for assigning a localization or cluster to an expected site.")
        overlay_advanced_button = ttk.Checkbutton(
            overlay_fields,
            text="Advanced overlay and gallery settings",
            variable=self.origami_overlay_advanced_visible,
            command=self._toggle_origami_advanced_sections,
        )
        overlay_advanced_button.grid(row=2, column=0, sticky="w", pady=(9, 2))
        self.origami_overlay_advanced_frame = ttk.Frame(overlay_fields)
        self.origami_overlay_advanced_frame.columnconfigure(1, weight=1)
        paired_setting_row(
            self.origami_overlay_advanced_frame,
            0,
            "G5M σ min / max (nm)",
            self.origami_g5m_sigma_min_nm,
            " / ",
            self.origami_g5m_sigma_max_nm,
            "Minimum and maximum Gaussian width accepted by Picasso G5M.",
        )
        setting_row(self.origami_overlay_advanced_frame, 1, "G5M min locs", self.origami_g5m_min_locs, "Minimum localizations required by Picasso G5M.")
        setting_row(self.origami_overlay_advanced_frame, 2, "G5M BIC patience", self.origami_g5m_bic_patience, "Model-selection rounds allowed without an improved BIC.")
        setting_row(self.origami_overlay_advanced_frame, 3, "Render pixel (nm)", self.origami_overlay_pixel_nm, "Overlay rendering resolution in nanometres per pixel.")
        setting_row(self.origami_overlay_advanced_frame, 4, "Render padding (nm)", self.origami_overlay_padding_nm, "Extra nanometres around the grid in overlay plots.")
        setting_row(self.origami_overlay_advanced_frame, 5, "Blur σ (nm)", self.origami_overlay_blur_nm, "Gaussian overlay blur in nanometres.")
        ttk.Separator(self.origami_overlay_advanced_frame).grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(self.origami_overlay_advanced_frame, text="Gallery sort", wraplength=125).grid(row=7, column=0, sticky="w", pady=3)
        ttk.Combobox(
            self.origami_overlay_advanced_frame,
            textvariable=self.origami_gallery_sort,
            state="readonly",
            width=1,
            values=("Origami ID", "Source position", "Most localizations", "Highest alignment RMS", "Lowest grid match", "Fewest occupied sites", "QC sample"),
        ).grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=3)
        paired_setting_row(
            self.origami_overlay_advanced_frame,
            8,
            "Min match % / max RMS",
            self.origami_gallery_min_match,
            " / ",
            self.origami_gallery_max_rms,
            "Optional minimum grid match percentage and maximum alignment RMS for the gallery.",
        )
        ttk.Label(self.origami_overlay_advanced_frame, text="Page size").grid(row=9, column=0, sticky="w", pady=3)
        ttk.Combobox(
            self.origami_overlay_advanced_frame,
            textvariable=self.origami_gallery_page_size,
            state="readonly",
            width=1,
            values=("25", "64", "100", "256"),
        ).grid(row=9, column=1, sticky="ew", padx=(8, 0), pady=3)
        gallery_navigation = ttk.Frame(self.origami_overlay_advanced_frame)
        gallery_navigation.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(gallery_navigation, text="◀", width=4, command=lambda: self._change_origami_gallery_page(-1)).pack(side="left")
        ttk.Entry(gallery_navigation, textvariable=self.origami_gallery_page, width=5).pack(side="left", padx=4)
        ttk.Button(gallery_navigation, text="Go", width=4, command=lambda: self._change_origami_gallery_page(0)).pack(side="left")
        ttk.Button(gallery_navigation, text="▶", width=4, command=lambda: self._change_origami_gallery_page(1)).pack(side="right")
        ttk.Label(self.origami_overlay_advanced_frame, textvariable=self.origami_gallery_page_label, wraplength=260).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=(3, 0)
        )
        ttk.Button(self.origami_overlay_advanced_frame, text="Apply Gallery Filters", command=self._apply_origami_gallery_view).grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        self.origami_refine_button = ttk.Button(
            overlay_fields, text="Refine Current Overlay with G5M", command=self.refine_origami_overlay_with_g5m
        )
        self.origami_refine_button.grid(row=4, column=0, sticky="ew", pady=(9, 0))
        WidgetTooltip(self.origami_refine_button, "Available after building a fast overlay; replaces direct assignments with Picasso G5M components.")
        self.origami_back_gallery_button = ttk.Button(overlay_fields, text="Back to Gallery", command=self._show_origami_gallery)
        self.origami_back_gallery_button.grid(row=5, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(overlay_fields, text="Reset Overlay Defaults", command=lambda: self._reset_origami_stage("Overlay")).grid(
            row=6, column=0, sticky="ew", pady=(9, 0)
        )

        export_fields = ttk.LabelFrame(overlay_fields, text="Export Results", padding=6)
        export_fields.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        export_fields.columnconfigure(0, weight=1)
        ttk.Label(
            export_fields,
            text="Save site statistics or the current paged gallery after building an overlay.",
            wraplength=250,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.origami_export_csv_button = ttk.Button(
            export_fields, text="Export Site CSVs", command=self.export_origami_csvs
        )
        self.origami_export_csv_button.grid(row=1, column=0, sticky="ew")
        self.origami_export_pdf_button = ttk.Button(
            export_fields, text="Export Paged Gallery PDF", command=self.export_origami_gallery_pdf
        )
        self.origami_export_pdf_button.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        WidgetTooltip(self.origami_export_pdf_button, "Available after an overlay result exists.")

        sticky_actions = ttk.Frame(self.origami_sidebar)
        sticky_actions.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        sticky_actions.columnconfigure(0, weight=1)
        self.origami_primary_action = ttk.Button(sticky_actions, text="Load Source Data", command=self.load_origami_source_data)
        self.origami_primary_action.grid(row=0, column=0, sticky="ew")
        self.origami_identify_button = self.origami_primary_action
        self.origami_primary_help = ttk.Label(sticky_actions, text="Load or refresh the current source.", wraplength=280)
        self.origami_primary_help.grid(row=1, column=0, sticky="ew", pady=(3, 0))

        self.origami_plot_area = ttk.Frame(self.origami_workspace, padding=(4, 4, 6, 4))
        self.origami_plot_area.grid(row=0, column=1, sticky="nsew")
        self.origami_plot_area.columnconfigure(0, weight=1)
        self.origami_plot_area.rowconfigure(1, weight=1)
        self.origami_plot_header = ttk.Frame(self.origami_plot_area)
        self.origami_plot_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.origami_plot_header.columnconfigure(2, weight=1)
        sidebar_toggle = ttk.Button(self.origami_plot_header, text="Plot Only", command=self._toggle_origami_sidebar)
        sidebar_toggle.grid(row=0, column=0, padx=(0, 6))
        self.origami_sidebar_toggle_button = sidebar_toggle
        self.origami_view_label = ttk.Label(self.origami_plot_header, text="View")
        self.origami_view_label.grid(row=0, column=1, sticky="w")
        plot_options = (
            "Coarse identification density",
            "Identified origami template matches",
            "Origami type counts",
            "Classification diagnostics",
            "Digital-group bias heatmap",
            "Individual origami gallery",
            "Individual site assignments",
            "Selected origami detail",
            "Aligned density",
            "Integrated density per site",
            "Mean site counts",
            "Site occupancy",
            "Occupied-site completeness",
        )
        self.origami_all_plot_options = plot_options
        self.origami_plot_combo = ttk.Combobox(
            self.origami_plot_header, textvariable=self.origami_plot_option, state="readonly", values=plot_options
        )
        self.origami_plot_combo.grid(row=0, column=2, sticky="ew", padx=(6, 8))
        self.origami_plot_combo.bind("<<ComboboxSelected>>", self._on_origami_plot_selection)
        self.origami_template_result_label = ttk.Label(self.origami_plot_header, text="Classification")
        self.origami_template_result_label.grid(row=0, column=3, padx=(0, 4))
        self.origami_template_result_combo = ttk.Combobox(
            self.origami_plot_header,
            textvariable=self.origami_template_result_view,
            values=("All templates",),
            state="disabled",
            width=18,
        )
        self.origami_template_result_combo.grid(row=0, column=4, padx=(0, 8))
        self.origami_template_result_combo.bind(
            "<<ComboboxSelected>>", self._on_origami_template_result_selection
        )
        self.origami_match_label = ttk.Label(self.origami_plot_header, text="Match")
        self.origami_match_label.grid(row=0, column=5, padx=(0, 4))
        self.origami_match_panel_combo = ttk.Combobox(
            self.origami_plot_header,
            textvariable=self.origami_match_panel,
            state="disabled",
            width=20,
            values=(
                "Template competition",
                "All panels",
                "Candidate",
                "Template",
                "Overlay",
                "Contributions",
            ),
        )
        self.origami_match_panel_combo.grid(row=0, column=6, padx=(0, 6))
        self.origami_match_panel_combo.bind("<<ComboboxSelected>>", self._on_origami_match_panel_selection)
        popout_button = ttk.Button(self.origami_plot_header, text="Pop Out", command=self._pop_out_origami_plot)
        popout_button.grid(row=0, column=7, padx=(0, 4))
        self.origami_popout_button = popout_button
        fullscreen_button = ttk.Button(self.origami_plot_header, text="Full Screen", command=self._toggle_origami_fullscreen_plot)
        fullscreen_button.grid(row=0, column=8)
        self.origami_fullscreen_button = fullscreen_button
        WidgetTooltip(sidebar_toggle, "Hide or restore the Origami workflow controls to maximize the plot area.")
        WidgetTooltip(popout_button, "Open a resizable snapshot of the current plot with its own navigation toolbar.")
        WidgetTooltip(fullscreen_button, "Hide application controls and expand the plot; press Escape to restore them.")

        self.origami_qc_display_bar = ttk.LabelFrame(
            self.origami_plot_header,
            text="QC and display — no filtering",
            padding=(6, 3),
        )
        self.origami_qc_display_bar.grid(
            row=2, column=0, columnspan=9, sticky="ew", pady=(4, 0)
        )
        self.origami_qc_display_note = ttk.Label(
            self.origami_qc_display_bar,
            text="Display-only controls; they do not change classification.",
            foreground="#3d626f",
        )
        self.origami_qc_display_note.grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        qc_display_specs = (
            (
                "Show theoretical overlay",
                self.origami_show_theoretical_overlay,
                self._toggle_origami_theoretical_overlay,
                "Overlay the active digital groups from the winning Step 4 classification template.",
            ),
            (
                "Show alignment template overlay",
                self.origami_show_alignment_overlay,
                self._toggle_origami_alignment_overlay,
                "Overlay the bright-site geometry extracted from the Step 2 alignment template at the locked fitted pose.",
            ),
            (
                "Show detected sites overlay",
                self.origami_show_detected_sites_overlay,
                self._toggle_origami_detected_sites_overlay,
                "Overlay filled markers at measured supported-site centroids and connect them to their theoretical targets.",
            ),
            (
                "Show digital-group ON/OFF",
                self.origami_show_site_diagnostics,
                self._toggle_origami_site_diagnostics,
                "Shade inferred-ON digital groups green and inferred-OFF groups red, without recoloring localizations.",
            ),
            (
                "Show localization group assignments",
                self.origami_show_localization_group_assignments,
                self._toggle_origami_localization_group_assignments,
                "Color each contributing localization by digital-group identity; shared assignments use concentric rings.",
            ),
            (
                "Show prominence sampling",
                self.origami_show_prominence_geometry,
                self._toggle_origami_prominence_geometry,
                "Show selected peaks, sampling boundaries, and prominence reference positions.",
            ),
            (
                "Show text statistics",
                self.origami_show_text_statistics,
                self._toggle_origami_text_statistics,
                "Show candidate classification and quality-control statistics.",
            ),
        )
        qc_display_controls: list[ttk.Checkbutton] = []
        for control_index, (text, variable, command, help_text) in enumerate(qc_display_specs):
            control = ttk.Checkbutton(
                self.origami_qc_display_bar,
                text=text,
                variable=variable,
                command=command,
            )
            control.grid(
                row=1 + control_index // 3,
                column=control_index % 3,
                sticky="w",
                padx=(0, 12),
                pady=1,
            )
            WidgetTooltip(control, help_text)
            qc_display_controls.append(control)
        self.origami_qc_display_controls = tuple(qc_display_controls)
        for column in range(3):
            self.origami_qc_display_bar.columnconfigure(column, weight=1)
        self.origami_plot_header.bind("<Configure>", self._layout_origami_plot_header, add="+")

        self.origami_figure = Figure(figsize=(8, 6), dpi=100)
        self.origami_figure.suptitle("1. Load source   2. Identify   3. Overlay")
        self.origami_canvas = FigureCanvasTkAgg(self.origami_figure, master=self.origami_plot_area)
        self.origami_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.origami_canvas.mpl_connect("button_press_event", self._on_origami_canvas_click)

        self.origami_candidate_bar = ttk.Frame(self.origami_plot_area)
        self.origami_candidate_bar.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        self.origami_candidate_bar.columnconfigure(3, weight=1)
        self.origami_prev_candidate_button = ttk.Button(
            self.origami_candidate_bar, text="◀", width=4, command=lambda: self._change_origami_navigation(-1)
        )
        self.origami_prev_candidate_button.grid(row=0, column=0)
        ttk.Label(self.origami_candidate_bar, textvariable=self.origami_navigation_kind).grid(row=0, column=1, padx=(6, 3))
        self.origami_navigation_entry = ttk.Entry(
            self.origami_candidate_bar, textvariable=self.origami_match_roi, width=6
        )
        self.origami_navigation_entry.grid(row=0, column=2)
        self.origami_navigation_status = ttk.Label(
            self.origami_candidate_bar, textvariable=self.origami_match_roi_label
        )
        self.origami_navigation_status.grid(row=0, column=3, sticky="w", padx=8)
        self.origami_candidate_go_button = ttk.Button(
            self.origami_candidate_bar, text="Go", width=4, command=lambda: self._change_origami_navigation(0)
        )
        self.origami_candidate_go_button.grid(row=0, column=4, padx=3)
        self.origami_candidate_overview_button = ttk.Button(
            self.origami_candidate_bar, text="Overview", command=self._show_origami_navigation_overview
        )
        self.origami_candidate_overview_button.grid(row=0, column=5, padx=3)
        self.origami_next_candidate_button = ttk.Button(
            self.origami_candidate_bar, text="▶", width=4, command=lambda: self._change_origami_navigation(1)
        )
        self.origami_next_candidate_button.grid(row=0, column=6)
        self.origami_prev_candidate_button.state(["disabled"])
        self.origami_next_candidate_button.state(["disabled"])
        self.origami_candidate_go_button.state(["disabled"])
        self.origami_candidate_overview_button.state(["disabled"])

        self.origami_toolbar_frame = ttk.Frame(self.origami_plot_area)
        self.origami_toolbar_frame.grid(row=3, column=0, sticky="ew")
        self.origami_toolbar = OrigamiToolbar(self.origami_canvas, self.origami_toolbar_frame, self)

        for frame in self.origami_stage_frames.values():
            bind_settings_scroll(frame)
        self._toggle_origami_advanced_sections()
        self._show_origami_stage("Source")
        self._refresh_origami_action_states()
        self.bind("<Escape>", lambda _event: self._exit_origami_fullscreen_plot(), add="+")
        self.bind("<Left>", lambda event: self._origami_candidate_key(event, -1), add="+")
        self.bind("<Right>", lambda event: self._origami_candidate_key(event, 1), add="+")
        for variable in self._origami_identification_variables():
            variable.trace_add("write", self._on_origami_identification_setting_changed)

        self.status_bar = ttk.Label(main, textvariable=self.status, anchor="w")
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _origami_identification_variables(self) -> tuple[tk.Variable, ...]:
        return (
            self.origami_pick_bin_nm,
            self.origami_connect_distance_nm,
            self.origami_min_density_contrast,
            self.origami_min_points,
            self.origami_max_points,
            self.origami_rows,
            self.origami_columns,
            self.origami_template_mode,
            self.origami_custom_template_name,
            self.origami_shared_alignment_template_name,
            self.origami_digital_pixel_schema_name,
            self.origami_template_pixel_x_nm,
            self.origami_template_pixel_y_nm,
            self.origami_spacing_x_nm,
            self.origami_spacing_y_nm,
            self.origami_rectangle_margin_nm,
            self.origami_use_correlation_gate,
            self.origami_min_cell_pattern_correlation,
            self.origami_use_cell_pattern_gate,
            self.origami_min_monte_carlo_probability,
            self.origami_site_mask_radius_nm,
            self.origami_min_supported_sites,
            self.origami_min_site_evidence,
            self.origami_min_site_localizations,
            self.origami_min_supported_rows,
            self.origami_min_supported_columns,
            self.origami_max_site_spacing_error_nm,
            self.origami_preview_pixel_nm,
            self.origami_alignment_max_pixels,
            self.origami_alignment_iterations,
            self.origami_min_rectangle_confidence,
        )

    def _origami_identification_snapshot(self) -> tuple[Any, ...]:
        values: list[Any] = []
        for variable in self._origami_identification_variables():
            try:
                values.append(variable.get())
            except tk.TclError:
                values.append(None)
        return tuple(values)

    def _on_origami_identification_setting_changed(self, *_args: Any) -> None:
        if self.origami_identification_baseline is None:
            self.origami_settings_state.set("Identification settings not yet validated")
        elif self._origami_identification_snapshot() == self.origami_identification_baseline:
            self.origami_settings_state.set("Identification settings match the last validated run")
        else:
            self.origami_settings_state.set("Settings changed since identification — rerun to validate")

    def _show_origami_stage(self, stage: str) -> None:
        stage = stage if stage in self.origami_stage_frames else "Source"
        self.origami_workflow_stage.set(stage)
        for name, frame in self.origami_stage_frames.items():
            if name == stage:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_remove()
        self.origami_settings_canvas.yview_moveto(0.0)
        actions: dict[str, tuple[str, Callable[[], None], str, bool]] = {
            "Source": (
                "Load Source Data",
                self.load_origami_source_data,
                "Load or refresh the selected source, ROI, and filters.",
                self.corrected_locs is not None,
            ),
            "Identify": (
                "Identify Origami",
                self.identify_origamis,
                "Run coarse candidate detection and theoretical-template matching.",
                self.origami_source_points_nm is not None and not self.origami_identification_running,
            ),
            "Overlay": (
                "Build Fast Overlay",
                self.overlay_origamis,
                "Build the cached aligned-density and site-statistics result.",
                self.origami_pick_result is not None and self.origami_pick_result.accepted_count > 0,
            ),
        }
        text, command, help_text, enabled = actions[stage]
        analysis_running = bool(self.origami_identification_running)
        enabled = bool(enabled and not analysis_running)
        self.origami_primary_action.configure(text=text, command=command)
        unavailable_reasons = {
            "Source": "Apply drift correction before loading an Origami source.",
            "Identify": "Load Source Data before running identification.",
            "Overlay": "Identify and accept at least one origami before building an overlay.",
        }
        self.origami_primary_help.configure(
            text=help_text if enabled else "Origami analysis is currently running." if analysis_running else unavailable_reasons[stage]
        )
        self.origami_primary_action.state(["!disabled"] if enabled else ["disabled"])
        export_state = self.origami_result is not None
        self.origami_export_csv_button.state(["!disabled"] if export_state else ["disabled"])
        self.origami_export_pdf_button.state(["!disabled"] if export_state else ["disabled"])

    def _refresh_origami_action_states(self) -> None:
        PaintAnalysisApp._sync_origami_classification_selector_state(self)
        if hasattr(self, "origami_stage_frames"):
            self._show_origami_stage(self.origami_workflow_stage.get())
        if hasattr(self, "origami_prev_candidate_button"):
            self._configure_origami_navigation_controls()
        overlay_state = ["!disabled"] if self.origami_result is not None else ["disabled"]
        if hasattr(self, "origami_refine_button"):
            self.origami_refine_button.state(overlay_state)
            self.origami_back_gallery_button.state(overlay_state)
        if hasattr(self, "origami_plot_combo"):
            classification_diagnostic_options = (
                "Origami type counts",
                "Classification diagnostics",
                "Digital-group bias heatmap",
            )
            identification_options: tuple[str, ...] = ()
            if self.origami_pick_result is not None:
                identification_options = self.origami_all_plot_options[:2]
            if self.origami_multi_template_results:
                identification_options = (
                    (
                        *self.origami_all_plot_options[:2],
                        *classification_diagnostic_options,
                    )
                    if self.origami_pick_result is not None
                    else (
                        self.origami_all_plot_options[1],
                        *classification_diagnostic_options,
                    )
                )
            identification_names = {
                "Coarse identification density",
                "Identified origami template matches",
                *classification_diagnostic_options,
            }
            overlay_options = (
                tuple(
                    option
                    for option in self.origami_all_plot_options
                    if option not in identification_names
                )
                if self.origami_result is not None
                else ()
            )
            available_options = (*identification_options, *overlay_options)
            self.origami_plot_combo.configure(values=available_options)
            self.origami_plot_combo.state(["!disabled", "readonly"] if available_options else ["disabled"])

    def _sync_origami_classification_selector_state(self) -> None:
        """Derive Classification availability solely from installed results."""
        if not hasattr(self, "origami_template_result_combo"):
            return
        results = getattr(self, "origami_multi_template_results", {})
        if not results:
            self.origami_template_result_combo.state(["disabled"])
            return
        values = ("All templates", *(str(name) for name in results))
        self.origami_template_result_combo.configure(values=values)
        variable = getattr(self, "origami_template_result_view", None)
        if variable is not None and str(variable.get()) not in values:
            variable.set("All templates")
        self.origami_template_result_combo.state(["!disabled", "readonly"])

    def _toggle_origami_advanced_sections(self) -> None:
        self.origami_identify_advanced_frame.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        if self.origami_overlay_advanced_visible.get():
            self.origami_overlay_advanced_frame.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        else:
            self.origami_overlay_advanced_frame.grid_remove()
        self.after_idle(
            lambda: self.origami_settings_canvas.configure(scrollregion=self.origami_settings_canvas.bbox("all"))
        )

    def _set_origami_stage_preview_overlays(self, completed_stage: int) -> None:
        """Select the most useful overlays for the stage that just completed."""
        inspecting_digital_groups = int(completed_stage) == 3
        self.origami_show_theoretical_overlay.set(
            False
            if inspecting_digital_groups
            else DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY
        )
        self.origami_show_alignment_overlay.set(DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY)
        self.origami_show_detected_sites_overlay.set(
            DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY
        )
        self.origami_show_site_diagnostics.set(
            True
            if inspecting_digital_groups
            else DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS
        )
        self.origami_show_localization_group_assignments.set(
            DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS
        )
        self.origami_show_prominence_geometry.set(
            DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY
        )
        self.origami_show_text_statistics.set(DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS)

    def _toggle_origami_theoretical_overlay(self) -> None:
        enabled = bool(self.origami_show_theoretical_overlay.get())
        if self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"The theoretical-site overlay is {state} for the identification overview.")

    def _toggle_origami_alignment_overlay(self) -> None:
        enabled = bool(self.origami_show_alignment_overlay.get())
        if self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"The Step 2 alignment-template overlay is {state}.")

    def _toggle_origami_detected_sites_overlay(self) -> None:
        enabled = bool(self.origami_show_detected_sites_overlay.get())
        if self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"The detected-site overlay is {state} for the identification overview.")

    def _toggle_origami_site_diagnostics(self) -> None:
        enabled = bool(self.origami_show_site_diagnostics.get())
        if (
            self.origami_last_rendered_plot_option == "Identified origami match ROI"
            and self.origami_selected_match_index is not None
        ):
            self._plot_identified_origami_match_roi(self.origami_selected_match_index)
        elif self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"Digital-group ON/OFF shading is {state} for identification views.")

    def _toggle_origami_localization_group_assignments(self) -> None:
        enabled = bool(self.origami_show_localization_group_assignments.get())
        if (
            self.origami_last_rendered_plot_option == "Identified origami match ROI"
            and self.origami_selected_match_index is not None
        ):
            self._plot_identified_origami_match_roi(self.origami_selected_match_index)
        elif self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(
                f"Localization-to-digital-group highlighting is {state} for identification views."
            )

    def _toggle_origami_prominence_geometry(self) -> None:
        enabled = bool(self.origami_show_prominence_geometry.get())
        if (
            self.origami_last_rendered_plot_option == "Identified origami match ROI"
            and self.origami_selected_match_index is not None
        ):
            self._plot_identified_origami_match_roi(self.origami_selected_match_index)
        elif self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"Prominence sampling geometry is {state} for identification views.")

    def _toggle_origami_text_statistics(self) -> None:
        enabled = bool(self.origami_show_text_statistics.get())
        if self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            self._refresh_origami_footprints()
        else:
            state = "enabled" if enabled else "disabled"
            self.status.set(f"Per-origami text statistics are {state} for the identification overview.")

    def _reset_origami_stage(self, stage: str) -> None:
        if stage == "Source":
            self.origami_source.set("Corrected localizations")
            self.origami_use_roi.set(True)
        elif stage == "Identify":
            defaults: tuple[tuple[tk.Variable, Any], ...] = (
                (self.origami_pick_bin_nm, DEFAULT_ORIGAMI_PICK_BIN_NM),
                (self.origami_connect_distance_nm, DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM),
                (self.origami_min_density_contrast, DEFAULT_ORIGAMI_MIN_DENSITY),
                (self.origami_min_points, DEFAULT_ORIGAMI_MIN_POINTS),
                (self.origami_max_points, DEFAULT_ORIGAMI_MAX_POINTS),
                (self.origami_rows, DEFAULT_ORIGAMI_ROWS),
                (self.origami_columns, DEFAULT_ORIGAMI_COLUMNS),
                (self.origami_template_mode, "Simulated grid"),
                (self.origami_template_pixel_x_nm, 1.0),
                (self.origami_template_pixel_y_nm, 1.0),
                (self.origami_spacing_x_nm, DEFAULT_ORIGAMI_SPACING_X_NM),
                (self.origami_spacing_y_nm, DEFAULT_ORIGAMI_SPACING_Y_NM),
                (self.origami_rectangle_margin_nm, 20.0),
                (self.origami_use_correlation_gate, DEFAULT_ORIGAMI_USE_CORRELATION_GATE),
                (
                    self.origami_min_cell_pattern_correlation,
                    DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION,
                ),
                (
                    self.origami_use_cell_pattern_gate,
                    DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE,
                ),
                (
                    self.origami_min_monte_carlo_probability,
                    DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR,
                ),
                (self.origami_site_mask_radius_nm, DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM),
                (self.origami_min_supported_sites, 5),
                (self.origami_min_site_evidence, DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE),
                (self.origami_min_site_localizations, DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS),
                (self.origami_min_supported_rows, 2),
                (self.origami_min_supported_columns, 2),
                (self.origami_max_site_spacing_error_nm, DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM),
                (self.origami_preview_pixel_nm, 1.0),
                (self.origami_alignment_max_pixels, DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS),
                (self.origami_alignment_iterations, DEFAULT_ORIGAMI_ALIGNMENT_PASSES),
                (self.origami_min_rectangle_confidence, DEFAULT_ORIGAMI_CORRELATION_THRESHOLD),
                (self.origami_tile_count, 100),
                (
                    self.origami_show_theoretical_overlay,
                    DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY,
                ),
                (
                    self.origami_show_alignment_overlay,
                    DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY,
                ),
                (
                    self.origami_show_detected_sites_overlay,
                    DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY,
                ),
                (
                    self.origami_show_site_diagnostics,
                    DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS,
                ),
                (
                    self.origami_show_localization_group_assignments,
                    DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS,
                ),
                (
                    self.origami_show_prominence_geometry,
                    DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY,
                ),
                (self.origami_show_text_statistics, DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS),
            )
            for variable, value in defaults:
                variable.set(value)
            self.origami_template_result_view.set("All templates")
        elif stage == "Overlay":
            defaults = (
                (self.origami_allow_mirror, False),
                (self.origami_site_radius_nm, 7.5),
                (self.origami_g5m_sigma_min_nm, 1.0),
                (self.origami_g5m_sigma_max_nm, 8.0),
                (self.origami_g5m_min_locs, 20),
                (self.origami_g5m_bic_patience, 3),
                (self.origami_overlay_pixel_nm, 0.5),
                (self.origami_overlay_padding_nm, 20.0),
                (self.origami_overlay_blur_nm, 1.0),
                (self.origami_gallery_sort, "Origami ID"),
                (self.origami_gallery_min_match, ""),
                (self.origami_gallery_max_rms, ""),
                (self.origami_gallery_page_size, "64"),
            )
            for variable, value in defaults:
                variable.set(value)
        self.status.set(f"Reset {stage.lower()} settings to defaults.")

    def _toggle_origami_sidebar(self) -> None:
        visible = bool(self.origami_sidebar_visible.get())
        self.origami_sidebar_visible.set(not visible)
        if visible:
            self.origami_sidebar.grid_remove()
            self.origami_sidebar_toggle_button.configure(text="Show Controls")
        else:
            self.origami_sidebar.grid(row=0, column=0, sticky="nsew")
            self.origami_sidebar_toggle_button.configure(text="Plot Only")
        self.after_idle(self.origami_canvas.draw_idle)

    def _layout_origami_plot_header(self, event: tk.Event) -> None:
        compact = int(event.width) < 1050
        if compact:
            self.origami_sidebar_toggle_button.grid_configure(row=0, column=0, padx=(0, 6), pady=(0, 3))
            self.origami_view_label.grid_configure(row=0, column=1, padx=(0, 4), pady=(0, 3))
            self.origami_plot_combo.grid_configure(row=0, column=2, columnspan=7, padx=0, pady=(0, 3), sticky="ew")
            self.origami_template_result_label.grid_configure(row=1, column=0, padx=(0, 4), pady=0)
            self.origami_template_result_combo.grid_configure(row=1, column=1, columnspan=2, padx=(0, 8), pady=0, sticky="ew")
            self.origami_match_label.grid_configure(row=1, column=3, padx=(0, 4), pady=0)
            self.origami_match_panel_combo.grid_configure(row=1, column=4, columnspan=2, padx=(0, 6), pady=0, sticky="ew")
            self.origami_popout_button.grid_configure(row=1, column=7, padx=(0, 4), pady=0)
            self.origami_fullscreen_button.grid_configure(row=1, column=8, padx=0, pady=0)
        else:
            self.origami_sidebar_toggle_button.grid_configure(row=0, column=0, columnspan=1, padx=(0, 6), pady=0)
            self.origami_view_label.grid_configure(row=0, column=1, columnspan=1, padx=0, pady=0)
            self.origami_plot_combo.grid_configure(row=0, column=2, columnspan=1, padx=(6, 8), pady=0, sticky="ew")
            self.origami_template_result_label.grid_configure(row=0, column=3, columnspan=1, padx=(0, 4), pady=0)
            self.origami_template_result_combo.grid_configure(row=0, column=4, columnspan=1, padx=(0, 8), pady=0, sticky="ew")
            self.origami_match_label.grid_configure(row=0, column=5, columnspan=1, padx=(0, 4), pady=0)
            self.origami_match_panel_combo.grid_configure(row=0, column=6, columnspan=1, padx=(0, 6), pady=0, sticky="ew")
            self.origami_popout_button.grid_configure(row=0, column=7, padx=(0, 4), pady=0)
            self.origami_fullscreen_button.grid_configure(row=0, column=8, padx=0, pady=0)
        qc_controls = getattr(self, "origami_qc_display_controls", ())
        qc_bar = getattr(self, "origami_qc_display_bar", None)
        qc_note = getattr(self, "origami_qc_display_note", None)
        if qc_controls and qc_bar is not None:
            column_count = 2 if compact else 3
            for column in range(3):
                qc_bar.columnconfigure(
                    column,
                    weight=1 if column < column_count else 0,
                )
            if qc_note is not None:
                qc_note.grid_configure(columnspan=column_count)
            for control_index, control in enumerate(qc_controls):
                control.grid_configure(
                    row=1 + control_index // column_count,
                    column=control_index % column_count,
                    sticky="w",
                    padx=(0, 12),
                    pady=1,
                )

    def _toggle_origami_fullscreen_plot(self) -> None:
        if self.origami_fullscreen_plot.get():
            self._exit_origami_fullscreen_plot()
            return
        self.origami_fullscreen_plot.set(True)
        self.global_sidebar_outer.grid_remove()
        self.main_top_bar.grid_remove()
        self.status_bar.grid_remove()
        self.origami_sidebar.grid_remove()
        self.origami_plot_header.grid_remove()
        self.origami_candidate_bar.grid_remove()
        self.origami_toolbar_frame.grid_remove()
        try:
            self.attributes("-fullscreen", True)
        except tk.TclError:
            pass
        self.after_idle(self.origami_canvas.draw_idle)

    def _exit_origami_fullscreen_plot(self) -> None:
        if not self.origami_fullscreen_plot.get():
            return
        self.origami_fullscreen_plot.set(False)
        try:
            self.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self.main_top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._sync_global_sidebar_visibility()
        if self.origami_sidebar_visible.get():
            self.origami_sidebar.grid(row=0, column=0, sticky="nsew")
        self.origami_plot_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.origami_candidate_bar.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        self.origami_toolbar_frame.grid(row=3, column=0, sticky="ew")
        self.after_idle(self.origami_canvas.draw_idle)

    def _pop_out_origami_plot(self) -> None:
        if self.origami_popout_window is not None and self.origami_popout_window.winfo_exists():
            self.origami_popout_window.lift()
            return
        self.origami_canvas.draw()
        rgba = np.asarray(self.origami_canvas.buffer_rgba()).copy()
        window = tk.Toplevel(self)
        window.title(f"Origami Plot — {self.origami_plot_option.get()}")
        window.geometry("1100x760")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        figure = Figure(figsize=(10, 7), dpi=100)
        axis = figure.add_subplot(111)
        axis.imshow(rgba, origin="upper")
        axis.set_axis_off()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(window)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas, toolbar_frame)
        canvas.draw_idle()
        self.origami_popout_window = window

        def close() -> None:
            self.origami_popout_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close)

    def _origami_candidate_key(self, event: tk.Event, delta: int) -> str | None:
        if (
            self.notebook.index(self.notebook.select()) != ORIGAMI_TAB
        ):
            return None
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox"}:
            return None
        if self._origami_navigation_mode() == "none":
            return None
        self._change_origami_navigation(delta)
        return "break"

    def _origami_navigation_mode(self) -> str:
        if self.origami_last_rendered_plot_option in {
            "Individual origami gallery",
            "Individual site assignments",
        }:
            return "gallery"
        if self.origami_last_rendered_plot_option in {
            "Coarse identification density",
            "Identified origami template matches",
            "Identified origami match ROI",
            "Random ROI inspection",
        }:
            return "roi"
        if self.origami_last_rendered_plot_option == "Selected origami detail":
            return "detail"
        return "none"

    def _configure_origami_navigation_controls(self) -> None:
        # Plot and Match changes rebuild portions of the Origami header. Keep
        # Classification from inheriting a stale disabled ttk state after a
        # prior loading/running transition.
        PaintAnalysisApp._sync_origami_classification_selector_state(self)
        mode = self._origami_navigation_mode()
        if mode == "gallery":
            enabled = self.origami_result is not None
            page = int(self.origami_gallery_page.get())
            self.origami_navigation_kind.set("Page")
            self.origami_navigation_entry.configure(textvariable=self.origami_gallery_page)
            self.origami_navigation_status.configure(textvariable=self.origami_gallery_page_label)
            self.origami_candidate_overview_button.configure(text="Overview")
            self.origami_candidate_overview_button.state(["disabled"])
            self.origami_prev_candidate_button.state(["!disabled"] if enabled and page > 1 else ["disabled"])
            self.origami_next_candidate_button.state(
                ["!disabled"] if enabled and page < self.origami_gallery_page_count else ["disabled"]
            )
            self.origami_candidate_go_button.state(["!disabled"] if enabled else ["disabled"])
            return
        elif mode == "roi":
            enabled = self.origami_pick_result is not None and self.origami_loaded_roi_nm is not None
            position = int(self.origami_roi_history_position)
            self.origami_navigation_kind.set("ROI")
            self.origami_navigation_entry.configure(textvariable=self.origami_match_roi)
            self.origami_navigation_status.configure(textvariable=self.origami_match_roi_label)
            if position < 0:
                self.origami_match_roi.set(1)
                self.origami_match_roi_label.set("Validation ROI")
            elif position < len(self.origami_roi_history):
                payload = self.origami_roi_history[position]
                tile_index = int(payload["tile_index"])
                available = int(payload["available_tile_count"])
                candidate_count = len(payload["picks"].regions)
                self.origami_match_roi.set(position + 2)
                self.origami_match_roi_label.set(
                    f"ROI {position + 2} • random tile {tile_index + 1}/{available} • {candidate_count} candidates"
                )
            self.origami_candidate_overview_button.configure(text="Validation ROI")
            self.origami_candidate_overview_button.state(["!disabled"] if enabled and position >= 0 else ["disabled"])
            self.origami_prev_candidate_button.state(["!disabled"] if enabled and position >= 0 else ["disabled"])
            self.origami_next_candidate_button.state(["!disabled"] if enabled and not self.origami_identification_running else ["disabled"])
            self.origami_candidate_go_button.state(["!disabled"] if enabled else ["disabled"])
            return
        elif mode == "detail":
            count = self.origami_result.origami_count if self.origami_result is not None else 0
            selected = self.origami_selected_index
            enabled = count > 0 and selected is not None
            self.origami_navigation_kind.set("Origami")
            self.origami_navigation_entry.configure(textvariable=self.origami_detail_number)
            self.origami_navigation_status.configure(textvariable=self.origami_detail_label)
            self.origami_candidate_overview_button.configure(text="Gallery")
            self.origami_candidate_overview_button.state(["!disabled"] if enabled else ["disabled"])
            self.origami_prev_candidate_button.state(
                ["!disabled"] if enabled and int(selected) > 0 else ["disabled"]
            )
            self.origami_next_candidate_button.state(
                ["!disabled"] if enabled and int(selected) < count - 1 else ["disabled"]
            )
            self.origami_candidate_go_button.state(["!disabled"] if enabled else ["disabled"])
            return
        else:
            enabled = False
            self.origami_navigation_kind.set("View")
            self.origami_navigation_entry.configure(textvariable=self.origami_match_roi)
            self.origami_navigation_status.configure(textvariable=self.origami_navigation_idle_label)
            self.origami_candidate_overview_button.configure(text="Overview")
            self.origami_candidate_overview_button.state(["disabled"])
        state = ["!disabled"] if enabled else ["disabled"]
        self.origami_prev_candidate_button.state(state)
        self.origami_next_candidate_button.state(state)
        self.origami_candidate_go_button.state(state)

    def _change_origami_navigation(self, delta: int) -> None:
        mode = self._origami_navigation_mode()
        if mode == "gallery":
            self._change_origami_gallery_page(delta)
        elif mode == "roi":
            self._change_origami_roi_window(delta)
        elif mode == "detail":
            self._change_origami_detail(delta)

    def _show_origami_navigation_overview(self) -> None:
        if self._origami_navigation_mode() == "detail":
            self._show_origami_gallery()
        elif self._origami_navigation_mode() == "roi":
            self.origami_roi_history_position = -1
            self._render_current_origami_roi_window()
        else:
            self._show_identified_origami_overview()

    def _change_origami_roi_window(self, delta: int) -> None:
        """Navigate validation/inspection ROI windows without stepping through candidates."""
        if self.origami_pick_result is None or self.origami_loaded_roi_nm is None:
            return
        if delta > 0:
            next_position = self.origami_roi_history_position + 1
            if next_position < len(self.origami_roi_history):
                self.origami_roi_history_position = next_position
                self._render_current_origami_roi_window()
            elif not self.origami_identification_running:
                self.inspect_random_origami_roi(target_view=self.origami_plot_option.get())
            return
        if delta < 0:
            if self.origami_roi_history_position > 0:
                self.origami_roi_history_position -= 1
                self._render_current_origami_roi_window()
            elif self.origami_roi_history_position == 0:
                self.origami_roi_history_position = -1
                self._render_current_origami_roi_window()
            return
        try:
            requested = int(self.origami_match_roi.get())
        except (tk.TclError, ValueError):
            requested = 1
        requested = max(1, min(len(self.origami_roi_history) + 1, requested))
        if requested == 1:
            self.origami_roi_history_position = -1
        else:
            self.origami_roi_history_position = requested - 2
        self._render_current_origami_roi_window()

    def _render_current_origami_roi_window(self) -> None:
        if self.origami_plot_option.get() == "Coarse identification density":
            self._plot_origami_coarse_density()
        elif self.origami_roi_history_position < 0:
            self._plot_identified_origamis()
        else:
            self._plot_random_origami_roi(self.origami_roi_history[self.origami_roi_history_position])

    def _change_origami_detail(self, delta: int) -> None:
        result = self.origami_result
        if result is None or result.origami_count == 0:
            return
        try:
            requested = int(self.origami_detail_number.get())
        except (tk.TclError, ValueError):
            requested = 1
        if delta:
            current = 0 if self.origami_selected_index is None else int(self.origami_selected_index)
            requested = current + 1 + int(delta)
        requested = max(1, min(result.origami_count, requested))
        self.origami_selected_index = requested - 1
        self.origami_detail_number.set(requested)
        self.origami_plot_option.set("Selected origami detail")
        self.render_origami_plot()

    def _on_origami_plot_selection(self, _event: tk.Event | None = None) -> None:
        self.origami_match_panel_combo.state(["disabled"])
        self.render_origami_plot()

    def _on_origami_template_result_selection(self, _event: tk.Event | None = None) -> None:
        name = self.origami_template_result_view.get()
        requested_view = self.origami_plot_option.get()
        if name == "All templates":
            # "All templates" is a display choice, not an invalidation of the
            # validated fit.  Retain the last active template underneath the
            # combined view so ROI/tiled actions and their geometry remain
            # available.
            self.origami_result = None
            if requested_view == "Origami type counts":
                self._plot_origami_type_counts()
            elif requested_view == "Classification diagnostics":
                self._plot_classification_diagnostics()
            elif requested_view == "Digital-group bias heatmap":
                self._plot_digital_group_bias_heatmap()
            else:
                self.origami_plot_option.set("Identified origami template matches")
                self._plot_all_template_classifications()
        else:
            payload = self.origami_multi_template_results.get(name)
            if payload is None:
                return
            self.origami_pick_result = payload["picks"]
            self.origami_identification_params = dict(payload["params"])
            overlay_payload = self.origami_multi_template_overlay_results.get(name)
            if overlay_payload is None:
                self.origami_result = None
                self.origami_result_render_settings = None
            else:
                self.origami_result = overlay_payload["result"]
                self.origami_result_source = str(overlay_payload["source"])
                self.origami_result_source_count = int(overlay_payload["source_count"])
                self.origami_result_render_settings = dict(overlay_payload["render_settings"])
                self.origami_result_occupancy_threshold = int(overlay_payload["occupancy_threshold"])
            if requested_view == "Identified origami template matches":
                self._plot_identified_origamis()
            elif requested_view == "Coarse identification density":
                self._plot_origami_coarse_density()
            elif requested_view == "Origami type counts":
                self._plot_origami_type_counts()
            elif requested_view == "Classification diagnostics":
                self._plot_classification_diagnostics()
            elif requested_view == "Digital-group bias heatmap":
                self._plot_digital_group_bias_heatmap()
            elif overlay_payload is not None:
                self.render_origami_plot()
            else:
                self.status.set(f"Building the {name} overlay; {requested_view} will remain selected.")
        self._refresh_origami_action_states()
        if name != "All templates" and overlay_payload is None:
            self.after_idle(self._auto_build_active_template_overlay)

    def _auto_build_active_template_overlay(self) -> None:
        name = self.origami_template_result_view.get()
        if not self.origami_multi_template_results:
            if (
                self.origami_result is None
                and self.origami_pick_result is not None
                and self.origami_pick_result.accepted_count > 0
                and not getattr(self, "origami_single_overlay_building", False)
            ):
                self.origami_single_overlay_building = True
                self.overlay_origamis()
            return
        if (
            name not in self.origami_multi_template_results
            or name in self.origami_multi_template_overlay_results
            or self.origami_pick_result is None
            or self.origami_pick_result.accepted_count == 0
        ):
            return
        building = getattr(self, "origami_multi_template_overlays_building", set())
        if name in building:
            return
        self.origami_multi_template_overlays_building = building
        building.add(name)
        self.overlay_origamis()

    def _on_origami_match_panel_selection(self, _event: tk.Event | None = None) -> None:
        active = self._active_origami_picks_and_params()
        if active is not None and self.origami_selected_match_index is not None:
            self._plot_identified_origami_match_roi(self.origami_selected_match_index)
        PaintAnalysisApp._sync_origami_classification_selector_state(self)

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

    def _bind_responsive_grid(
        self,
        container: tk.Widget,
        widgets: list[tk.Widget],
        minimum_cell_width: int,
        maximum_columns: int,
        *,
        equal_row_weights: bool = False,
        honor_requested_width: bool = False,
    ) -> None:
        """Reflow a set of controls as their parent gains or loses width."""
        current_columns = 0

        def descendant_requested_width(widget: tk.Widget) -> int:
            requested = int(widget.winfo_reqwidth())
            for child in widget.winfo_children():
                requested = max(requested, descendant_requested_width(child))
            return requested

        def reflow(event: tk.Event | None = None) -> None:
            nonlocal current_columns
            width = int(getattr(event, "width", 0)) or int(container.winfo_width())
            effective_cell_width = int(minimum_cell_width)
            if honor_requested_width:
                effective_cell_width = max(
                    effective_cell_width,
                    max(descendant_requested_width(widget) for widget in widgets) + 15,
                )
            columns = responsive_column_count(width, effective_cell_width, maximum_columns)
            if columns == current_columns:
                return
            old_rows = math.ceil(len(widgets) / current_columns) if current_columns else 0
            new_rows = math.ceil(len(widgets) / columns)
            for column in range(maximum_columns):
                container.columnconfigure(column, weight=1 if column < columns else 0)
            for row in range(max(old_rows, new_rows)):
                container.rowconfigure(row, weight=1 if equal_row_weights and row < new_rows else 0)
            for index, widget in enumerate(widgets):
                widget.grid_forget()
                widget.grid(
                    row=index // columns,
                    column=index % columns,
                    sticky="nsew",
                    padx=3,
                    pady=3,
                )
            current_columns = columns

        container.bind("<Configure>", reflow, add="+")
        container.after_idle(reflow)

    def _scale_map_density(
        self,
        image: np.ndarray,
        min_density: float | None = None,
        max_density: float | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        automatic = bool(self.auto_density_contrast.get())
        if automatic:
            base_min, base_max = histogram_density_limits(image)
            multiplier = max(0.1, min(10.0, float(self.auto_density_multiplier.get())))
            requested_min = base_min * multiplier
            requested_max = base_max * multiplier
        else:
            requested_min = float(self.render_min_density.get()) if min_density is None else float(min_density)
            requested_max = float(self.render_max_density.get()) if max_density is None else float(max_density)
        scaled, limits = scale_density_like_picasso(image, requested_min, requested_max)
        if automatic:
            self.render_min_density.set(limits[0])
            self.render_max_density.set(limits[1])
        return scaled, limits

    def _on_auto_density_multiplier_changed(self, value: str) -> None:
        multiplier = max(0.1, min(10.0, float(value)))
        self.auto_density_multiplier.set(multiplier)
        self.auto_density_multiplier_label.set(f"Density multiplier: {multiplier:.2g}×")
        self._schedule_density_refresh()

    def _on_auto_density_toggled(self) -> None:
        self._schedule_density_refresh(delay_ms=0)

    def _schedule_density_refresh(self, delay_ms: int = 60) -> None:
        if self.density_refresh_after_id is not None:
            try:
                self.after_cancel(self.density_refresh_after_id)
            except Exception:
                pass
        self.density_refresh_after_id = self.after(delay_ms, self._refresh_active_map_density)

    def _refresh_active_map_density(self) -> None:
        self.density_refresh_after_id = None
        tab_index = self.active_notebook_tab
        raw_image = self.map_density_images.get(tab_index)
        pair = self._axis_canvas_for_tab(tab_index)
        if raw_image is None or pair is None:
            return
        axis, canvas = pair
        if not axis.images:
            return
        display_image, limits = self._scale_map_density(raw_image)
        axis.images[0].set_data(display_image)
        if tab_index == RAW_MAP_TAB:
            colorbar = self.raw_map_colorbar
            units = "locs/render px"
        elif tab_index == CORRECTED_MAP_TAB:
            colorbar = self.map_colorbar
            units = "locs/render px"
        elif tab_index == LINKED_MAP_TAB:
            colorbar = self.linked_map_colorbar
            units = "events/render px"
        else:
            colorbar = self.filtered_map_colorbar
            units = "locs/render px"
        if colorbar is not None:
            colorbar.set_label(f"density contrast ({limits[0]:.3g}-{limits[1]:.3g} {units})")
        canvas.draw_idle()
        if bool(self.auto_density_contrast.get()):
            self.status.set(
                f"Auto density {float(self.auto_density_multiplier.get()):.2g}×: "
                f"limits {limits[0]:.4g}-{limits[1]:.4g}."
            )

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
        self._sync_global_sidebar_visibility(current_tab)
        current_pair = self._axis_canvas_for_tab(current_tab)
        if current_pair is not None:
            current_axis, current_canvas = current_pair
            self._apply_shared_map_limits(current_axis, current_canvas)
            self._schedule_density_refresh(delay_ms=0)
        elif current_tab == ORIGAMI_TAB:
            # A hidden Tk canvas can retain its previous geometry while the
            # tab switch is being processed. Recompute the source render only
            # after the Origami canvas has received its visible dimensions.
            self.after_idle(self._rerender_loaded_origami_source_view)

    def _sync_global_sidebar_visibility(self, tab_index: int | None = None) -> None:
        """Keep map-rendering controls out of the Origami workspace."""
        if tab_index is None:
            tab_index = self._current_notebook_tab_index()
        if self.origami_fullscreen_plot.get() or tab_index == ORIGAMI_TAB:
            self.global_sidebar_outer.grid_remove()
        else:
            self.global_sidebar_outer.grid(row=0, column=0, sticky="nsew")

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
        self._schedule_dynamic_map_render()

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

    def _full_map_viewport_nm(self) -> tuple[float, float, float, float] | None:
        if self.loaded is None:
            return None
        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        return (
            0.0,
            float(self.loaded.info[0]["Width"]) * pixelsize,
            0.0,
            float(self.loaded.info[0]["Height"]) * pixelsize,
        )

    def _dynamic_render_pixel_nm(
        self,
        axis: Any,
        viewport_nm: tuple[float, float, float, float],
    ) -> float:
        try:
            bounds = axis.get_window_extent()
            display_width = float(bounds.width)
            display_height = float(bounds.height)
        except Exception:
            display_width = display_height = 1.0
        if display_width <= 1 or display_height <= 1:
            display_width = max(1.0, float(axis.figure.get_figwidth() * axis.figure.dpi))
            display_height = max(1.0, float(axis.figure.get_figheight() * axis.figure.dpi))
        pixel_nm = optimal_dynamic_render_pixel_nm(viewport_nm, display_width, display_height)
        # Keep the control readable and make the latest automatic resolution
        # immediately available as the manual value if dynamic rendering is disabled.
        pixel_nm = float(f"{pixel_nm:.6g}")
        self.render_disp_px_nm.set(pixel_nm)
        return pixel_nm

    def _render_pixel_nm_for_request(
        self,
        axis: Any,
        viewport_nm: tuple[float, float, float, float] | None,
        allow_dynamic: bool,
    ) -> float:
        pixel_nm = float(self.render_disp_px_nm.get())
        if allow_dynamic and bool(self.dynamic_zoom_render.get()):
            pixel_viewport = viewport_nm or self._full_map_viewport_nm()
            if pixel_viewport is not None:
                pixel_nm = self._dynamic_render_pixel_nm(axis, pixel_viewport)
        return pixel_nm

    def _cancel_dynamic_render_for_manual_request(self) -> None:
        """Prevent queued or stale automatic work from replacing a manual render."""
        self.dynamic_render_request_id += 1
        self.dynamic_render_pending = False
        if self.dynamic_render_after_id is not None:
            try:
                self.after_cancel(self.dynamic_render_after_id)
            except Exception:
                pass
            self.dynamic_render_after_id = None

    def _schedule_dynamic_map_render(self) -> None:
        if not bool(self.dynamic_zoom_render.get()) or self.loaded is None:
            return
        if self._axis_canvas_for_tab(self.active_notebook_tab) is None:
            return
        self.dynamic_render_request_id += 1
        if self.dynamic_render_after_id is not None:
            try:
                self.after_cancel(self.dynamic_render_after_id)
            except Exception:
                pass
        self.dynamic_render_after_id = self.after(DYNAMIC_RENDER_DEBOUNCE_MS, self._start_dynamic_map_render)

    def _start_dynamic_map_render(self) -> None:
        self.dynamic_render_after_id = None
        if not bool(self.dynamic_zoom_render.get()) or self.loaded is None:
            return
        if self.dynamic_render_running:
            self.dynamic_render_pending = True
            return

        tab_index = self.active_notebook_tab
        pair = self._axis_canvas_for_tab(tab_index)
        full_viewport = self._full_map_viewport_nm()
        if pair is None or full_viewport is None:
            return
        axis, _canvas = pair
        viewport = self._shared_map_viewport_nm()
        pixel_viewport = viewport or full_viewport

        locs: pd.DataFrame | None = None
        target_kind = ""
        extra: dict[str, Any] = {}
        if tab_index == RAW_MAP_TAB:
            locs = self.loaded.locs
            target_kind = "raw_map"
        elif tab_index == CORRECTED_MAP_TAB and self.corrected_locs is not None:
            locs = self.corrected_locs
            target_kind = "map_only"
        elif tab_index == LINKED_MAP_TAB and self.linked_locs is not None:
            locs = self.linked_locs
            target_kind = "link_map"
            extra = {
                "linked_count": int(len(self.linked_locs)),
                "source_count": int(self.linked_source_count),
                "link_source": self.linked_source_name,
                "source_label": "raw" if self.linked_source_name == "Raw map" else "corrected",
                "roi_text": "full map" if self.linked_roi_nm is None else "selected ROI",
            }
        elif tab_index == FILTERED_MAP_TAB and self.filtered_map_locs is not None:
            locs = self.filtered_map_locs
            target_kind = "filtered_map"
            extra = {
                **self.filtered_map_render_context,
                "filtered_count": int(len(locs)),
                "min_density": float(self.render_min_density.get()),
                "max_density": float(self.render_max_density.get()),
            }
        if locs is None:
            return

        render_pixel_nm = self._dynamic_render_pixel_nm(axis, pixel_viewport)
        request_id = self.dynamic_render_request_id
        source_path = self.loaded.path
        info = self.loaded.info
        blur_method = self.render_blur_method.get()
        min_blur_width = float(self.min_blur_width.get())
        self.dynamic_render_running = True
        self.dynamic_render_pending = False
        self.status.set(
            f"Dynamic zoom render: {render_pixel_nm:.3g} nm/pixel "
            f"(minimum {MIN_DYNAMIC_RENDER_PIXEL_NM:g} nm/pixel)..."
        )

        def worker() -> tuple[str, Any]:
            try:
                renderer = (
                    render_filtered_map_with_settings
                    if target_kind == "filtered_map"
                    else render_picasso_map
                )
                rendered = renderer(
                    locs,
                    info,
                    render_pixel_nm,
                    blur_method,
                    min_blur_width,
                    viewport,
                )
                result = {"map": rendered, **extra} if target_kind == "filtered_map" else rendered
                if target_kind != "filtered_map":
                    result.update(extra)
                result.update(
                    {
                        "source_path": source_path,
                        "dynamic_request_id": request_id,
                        "dynamic_target_kind": target_kind,
                        "dynamic_tab_index": tab_index,
                        "render_px_nm": render_pixel_nm,
                    }
                )
                return "dynamic_map", result
            except Exception as exc:
                return "dynamic_map", {
                    "source_path": source_path,
                    "dynamic_request_id": request_id,
                    "dynamic_tab_index": tab_index,
                    "dynamic_error": str(exc),
                    "dynamic_error_details": traceback.format_exc(),
                }

        self._run_worker(worker)

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

    def _load_origami_custom_template(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Load custom origami alignment template",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[
                ("Template images", "*.png *.tif *.tiff *.jpg *.jpeg"),
                ("PNG images", "*.png"),
                ("TIFF images", "*.tif *.tiff"),
                ("JPEG images", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            image = load_custom_template_image(path)
            metadata = load_custom_template_metadata(path)
        except Exception as exc:
            messagebox.showerror("Invalid custom template", str(exc))
            return
        self.origami_custom_template_path = path
        self.origami_custom_template_image = image
        calibration_note = "manual calibration"
        if (
            metadata is not None
            and int(metadata["width_px"]) == image.shape[1]
            and int(metadata["height_px"]) == image.shape[0]
        ):
            self.origami_rows.set(int(metadata["rows"]))
            self.origami_columns.set(int(metadata["columns"]))
            self.origami_spacing_x_nm.set(float(metadata["spacing_x_nm"]))
            self.origami_spacing_y_nm.set(float(metadata["spacing_y_nm"]))
            self.origami_rectangle_margin_nm.set(float(metadata["margin_nm"]))
            self.origami_template_pixel_x_nm.set(float(metadata["pixel_size_x_nm"]))
            self.origami_template_pixel_y_nm.set(float(metadata["pixel_size_y_nm"]))
            calibration_note = "Picklist Generator calibration"
        else:
            width_nm = (
                max(float(self.origami_spacing_x_nm.get()), (int(self.origami_columns.get()) - 1) * float(self.origami_spacing_x_nm.get()))
                + 2.0 * float(self.origami_rectangle_margin_nm.get())
            )
            height_nm = (
                max(float(self.origami_spacing_y_nm.get()), (int(self.origami_rows.get()) - 1) * float(self.origami_spacing_y_nm.get()))
                + 2.0 * float(self.origami_rectangle_margin_nm.get())
            )
            self.origami_template_pixel_x_nm.set(width_nm / max(image.shape[1] - 1, 1))
            self.origami_template_pixel_y_nm.set(height_nm / max(image.shape[0] - 1, 1))
        display_name = custom_template_display_name(path, metadata)
        self.origami_custom_template_name.set(
            f"{display_name} — {image.shape[1]} × {image.shape[0]} px; {calibration_note}; "
            f"{float(self.origami_template_pixel_x_nm.get()):.4g} × "
            f"{float(self.origami_template_pixel_y_nm.get()):.4g} nm/px"
        )
        self.origami_custom_templates = []
        self.origami_multi_template_results = {}
        self.origami_multi_template_counts = {}
        self.origami_multi_template_overlay_results = {}
        self.origami_template_result_view.set("All templates")
        self.origami_template_result_combo.configure(values=("All templates",))
        self.origami_template_result_combo.state(["disabled"])
        self.origami_template_mode.set("Custom image")
        self._remember_file_dialog_dir(path)
        self.status.set(f"Loaded custom alignment template {display_name} with {calibration_note}.")

    def _load_multiple_origami_custom_templates(self) -> None:
        path_texts = filedialog.askopenfilenames(
            title="Load calibrated origami templates",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[
                ("Template images", "*.png *.tif *.tiff *.jpg *.jpeg"),
                ("PNG images", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path_texts:
            return
        if len(path_texts) < 2:
            messagebox.showinfo(
                "Select multiple templates",
                "Choose at least two calibrated template images, or use Load Custom Template Image for one template.",
            )
            return
        templates: list[dict[str, Any]] = []
        used_names: set[str] = set()
        try:
            for path_text in path_texts:
                path = Path(path_text)
                image = load_custom_template_image(path)
                metadata = load_custom_template_metadata(path)
                if (
                    metadata is None
                    or int(metadata["width_px"]) != image.shape[1]
                    or int(metadata["height_px"]) != image.shape[0]
                ):
                    raise ValueError(
                        f"{path.name} has no matching embedded/sidecar physical calibration. "
                        "Regenerate it in Picklist Generator before multi-template classification."
                    )
                base_name = custom_template_display_name(path, metadata)
                name = base_name
                suffix = 2
                while name in used_names:
                    name = f"{base_name} ({suffix})"
                    suffix += 1
                used_names.add(name)
                logical_model = logical_bit_model_from_metadata(metadata)
                loaded_digital_model = self.__dict__.get("origami_digital_pixel_model")
                if loaded_digital_model is not None:
                    if logical_model is None:
                        raise ValueError(
                            f"{path.name} does not declare which loaded digital pixels are active."
                        )
                    schema = loaded_digital_model
                    if (
                        logical_model["bit_ids"] != schema["bit_ids"]
                        or logical_model["physical_shape"] != schema["physical_shape"]
                    ):
                        raise ValueError(
                            f"{path.name} uses different digital-pixel identifiers or grid dimensions than the Step 3 JSON."
                        )
                    logical_model = {
                        **schema,
                        "active_bits": logical_model["active_bits"],
                    }
                templates.append(
                    {
                        "name": name,
                        "path": path,
                        "image": image,
                        "rows": int(metadata["rows"]),
                        "columns": int(metadata["columns"]),
                        "spacing_x_nm": float(metadata["spacing_x_nm"]),
                        "spacing_y_nm": float(metadata["spacing_y_nm"]),
                        "rectangle_margin_nm": float(metadata["margin_nm"]),
                        "template_pixel_size_x_nm": float(metadata["pixel_size_x_nm"]),
                        "template_pixel_size_y_nm": float(metadata["pixel_size_y_nm"]),
                        "logical_model": logical_model,
                    }
                )
        except Exception as exc:
            messagebox.showerror("Invalid template collection", str(exc))
            return

        self.origami_custom_templates = templates
        logical_models = [template.get("logical_model") for template in templates]
        if any(model is not None for model in logical_models):
            if not all(model is not None for model in logical_models):
                messagebox.showerror(
                    "Incompatible template collection",
                    "Either every template or no template must contain logical-stroke metadata. "
                    "Regenerate the older templates together in Picklist Generator.",
                )
                self.origami_custom_templates = []
                return
            reference = logical_models[0]
            if any(
                model["bit_ids"] != reference["bit_ids"]
                or model["bit_cells"] != reference["bit_cells"]
                or model["bit_physical_cells"] != reference["bit_physical_cells"]
                or model["alignment_cells"] != reference["alignment_cells"]
                or model["physical_shape"] != reference["physical_shape"]
                for model in logical_models[1:]
            ):
                messagebox.showerror(
                    "Incompatible logical templates",
                    "All templates must use the same logical-bit identifiers, physical grid, and site-group definitions.",
                )
                self.origami_custom_templates = []
                return
        first = templates[0]
        self.origami_custom_template_path = Path(first["path"])
        self.origami_custom_template_image = np.asarray(first["image"], dtype=float)
        self.origami_rows.set(int(first["rows"]))
        self.origami_columns.set(int(first["columns"]))
        self.origami_spacing_x_nm.set(float(first["spacing_x_nm"]))
        self.origami_spacing_y_nm.set(float(first["spacing_y_nm"]))
        self.origami_rectangle_margin_nm.set(float(first["rectangle_margin_nm"]))
        self.origami_template_pixel_x_nm.set(float(first["template_pixel_size_x_nm"]))
        self.origami_template_pixel_y_nm.set(float(first["template_pixel_size_y_nm"]))
        self.origami_template_mode.set("Custom image")
        self.origami_custom_template_name.set(
            f"{len(templates)} calibrated templates: " + ", ".join(str(item["name"]) for item in templates)
        )
        self.origami_multi_template_results = {}
        self.origami_multi_template_counts = {}
        self.origami_multi_template_overlay_results = {}
        self.origami_template_result_view.set("All templates")
        self.origami_template_result_combo.configure(
            values=("All templates", *(str(item["name"]) for item in templates))
        )
        self.origami_template_result_combo.state(["disabled"])
        self._remember_file_dialog_dir(Path(templates[-1]["path"]))
        self.status.set(
            f"Loaded {len(templates)} calibrated digital classification templates."
        )

    def _load_origami_shared_alignment_template(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Load one calibrated shared alignment template",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[
                ("Template images", "*.png *.tif *.tiff *.jpg *.jpeg"),
                ("PNG images", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            image = load_custom_template_image(path)
            metadata = load_custom_template_metadata(path)
            if (
                metadata is None
                or int(metadata["width_px"]) != image.shape[1]
                or int(metadata["height_px"]) != image.shape[0]
            ):
                raise ValueError(
                    "The shared alignment template needs matching embedded or sidecar "
                    "Picklist Generator calibration."
                )
            alignment_template = {
                "name": custom_template_display_name(path, metadata),
                "path": path,
                "image": image,
                "rows": int(metadata["rows"]),
                "columns": int(metadata["columns"]),
                "spacing_x_nm": float(metadata["spacing_x_nm"]),
                "spacing_y_nm": float(metadata["spacing_y_nm"]),
                "rectangle_margin_nm": float(metadata["margin_nm"]),
                "template_pixel_size_x_nm": float(metadata["pixel_size_x_nm"]),
                "template_pixel_size_y_nm": float(metadata["pixel_size_y_nm"]),
            }
            alignment_points = alignment_template_overlay_points(
                np.empty((0, 2), dtype=float),
                {"shared_alignment_template": alignment_template},
            )
            if len(alignment_points):
                alignment_template["overlay_points_nm"] = np.asarray(
                    alignment_points, dtype=float
                )
            digital_model = self.__dict__.get("origami_digital_pixel_model")
            if digital_model is not None and digital_model["physical_shape"] != (
                alignment_template["rows"],
                alignment_template["columns"],
            ):
                raise ValueError(
                    "The alignment template grid dimensions do not match the loaded digital-pixel JSON."
                )
        except Exception as exc:
            messagebox.showerror("Invalid shared alignment template", str(exc))
            return
        self.origami_shared_alignment_template = alignment_template
        self.origami_custom_template_path = path
        self.origami_custom_template_image = np.asarray(image, dtype=float)
        alignment_variables = (
            ("origami_rows", int(alignment_template["rows"])),
            ("origami_columns", int(alignment_template["columns"])),
            ("origami_spacing_x_nm", float(alignment_template["spacing_x_nm"])),
            ("origami_spacing_y_nm", float(alignment_template["spacing_y_nm"])),
            ("origami_rectangle_margin_nm", float(alignment_template["rectangle_margin_nm"])),
            ("origami_template_pixel_x_nm", float(alignment_template["template_pixel_size_x_nm"])),
            ("origami_template_pixel_y_nm", float(alignment_template["template_pixel_size_y_nm"])),
            ("origami_template_mode", "Custom image"),
        )
        for variable_name, value in alignment_variables:
            variable = self.__dict__.get(variable_name)
            if variable is not None:
                variable.set(value)
        self.origami_shared_alignment_template_name.set(
            f"Shared alignment: {alignment_template['name']} "
            f"({image.shape[1]} × {image.shape[0]} px)"
        )
        self._remember_file_dialog_dir(path)
        self.status.set(
            f"Loaded {alignment_template['name']} for one-pass locked alignment."
        )

    def _load_origami_digital_pixel_schema(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Load digital-pixel group JSON",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("The JSON must contain an object.")
            raw_model = payload.get("logical_model", payload)
            if not isinstance(raw_model, dict):
                raise ValueError("The JSON does not contain a digital-pixel model.")
            raw_model = dict(raw_model)
            raw_model.setdefault("active_logical_bits", [])
            rows = int(raw_model.get("physical_rows", payload.get("rows", -1)))
            columns = int(raw_model.get("physical_columns", payload.get("columns", -1)))
            wrapper = {
                "rows": rows,
                "columns": columns,
                "logical_model": raw_model,
            }
            model = logical_bit_model_from_metadata(wrapper)
            if model is None:
                raise ValueError("The JSON does not define logical_bits.")
            if self.origami_shared_alignment_template is not None and (
                rows != int(self.origami_shared_alignment_template["rows"])
                or columns != int(self.origami_shared_alignment_template["columns"])
            ):
                raise ValueError(
                    "The digital-pixel grid dimensions do not match the loaded alignment template."
                )
            updated_templates: list[dict[str, Any]] = []
            for template in self.origami_custom_templates:
                template_model = template.get("logical_model")
                if template_model is None or template_model["bit_ids"] != model["bit_ids"]:
                    raise ValueError(
                        f"Classification template {template['name']} does not use these digital-pixel identifiers."
                    )
                updated_templates.append(
                    {
                        **template,
                        "logical_model": {
                            **model,
                            "active_bits": template_model["active_bits"],
                        },
                    }
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Invalid digital-pixel JSON", str(exc))
            return
        self.origami_digital_pixel_model = model
        self.origami_custom_templates = updated_templates
        self.origami_digital_pixel_schema_name.set(
            f"{path.name}: {len(model['bit_ids'])} digital groups on {rows} × {columns} sites"
        )
        self._remember_file_dialog_dir(path)
        self.status.set(
            f"Loaded {len(model['bit_ids'])} digital-pixel groups for Step 3 detection."
        )

    def _clear_origami_shared_alignment_template(self) -> None:
        alignment_path = (
            self.origami_shared_alignment_template.get("path")
            if self.origami_shared_alignment_template is not None
            else None
        )
        self.origami_shared_alignment_template = None
        if self.__dict__.get("origami_custom_template_path") == alignment_path:
            self.origami_custom_template_path = None
            self.origami_custom_template_image = None
        self.origami_shared_alignment_template_name.set(
            "No alignment template loaded"
        )
        self.status.set(
            "Cleared the alignment template. Load one before running Step 2."
        )

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load localization file",
            initialdir=str(self._file_dialog_initial_dir()),
            filetypes=[("Localization files", "*.hdf5 *.h5 *.csv"), ("CSV files", "*.csv"), ("Picasso HDF5 files", "*.hdf5 *.h5"), ("All files", "*.*")],
        )
        if not path:
            return
        self._remember_file_dialog_dir(Path(path))
        self.session_load_in_progress = True
        self._show_load_progress(f"Opening {Path(path).name}...")
        self.status.set("Checking the development cache before loading localization data...")
        source_path = Path(path)
        settings = self._current_drift_cache_settings()
        self._run_worker(lambda: self._load_file_or_cache_worker(source_path, settings))

    def _current_drift_cache_settings(self) -> dict[str, Any]:
        return drift_cache_settings(
            self.drift_method.get(),
            int(self.drift_segmentation.get()),
            float(self.aim_intersect_nm.get()),
            float(self.aim_roi_nm.get()),
            float(self.rcc_lattice_pitch_nm.get()),
            self.drift_file_path,
        )

    def _load_file_or_cache_worker(
        self,
        source_path: Path,
        settings: dict[str, Any],
    ) -> tuple[str, Any]:
        cached = load_development_session_cache(source_path, settings)
        if cached is not None:
            return "cached_session", cached
        return "loaded", read_locs(source_path, self._load_progress_callback)

    def _restore_latest_development_session(self) -> None:
        if (
            self.loaded is not None
            or self.session_load_in_progress
            or os.environ.get("PAINT_ANALYSIS_DISABLE_DEVELOPMENT_CACHE") == "1"
        ):
            return
        request = latest_development_session_request()
        if request is None:
            return
        source_path, settings = request
        self.drift_method.set(str(settings["method"]))
        self.drift_segmentation.set(int(settings["segmentation"]))
        self.aim_intersect_nm.set(float(settings["aim_intersect_nm"]))
        self.aim_roi_nm.set(float(settings["aim_roi_nm"]))
        self.rcc_lattice_pitch_nm.set(float(settings["rcc_lattice_pitch_nm"]))
        drift_file = settings.get("drift_file")
        self.drift_file_path = Path(drift_file["path"]) if drift_file is not None else None
        self.drift_file_label.set(self.drift_file_path.name if self.drift_file_path is not None else "No drift file selected")
        self._show_load_progress(f"Restoring cached session for {source_path.name}...")
        self.status.set(f"Restoring cached localizations and drift correction for {source_path.name}...")
        self.session_load_in_progress = True
        self._run_worker(lambda: self._load_file_or_cache_worker(source_path, settings))

    def _show_load_progress(self, message: str) -> None:
        if self.load_progress_hide_id is not None:
            try:
                self.after_cancel(self.load_progress_hide_id)
            except Exception:
                pass
            self.load_progress_hide_id = None
        self.load_progress_value.set(0.0)
        self.load_progress_text.set(message)
        self.load_progress_bar.grid()
        self.load_progress_label.grid()

    def _hide_load_progress(self) -> None:
        self.load_progress_hide_id = None
        self.load_progress_bar.grid_remove()
        self.load_progress_label.grid_remove()

    def _load_progress_callback(self, percent: float, message: str) -> None:
        self.worker_queue.put(("load_progress", (float(percent), str(message))))

    def load_drift_file(self) -> None:
        if self.loaded is None:
            messagebox.showinfo("No localization file", "Load a localization file before choosing its associated drift CSV.")
            return
        path = filedialog.askopenfilename(
            title="Load frame-by-frame drift correction",
            initialdir=str(Path(self.loaded.path).parent),
            filetypes=[("Drift CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.drift_file_path = Path(path)
        self.drift_file_label.set(self.drift_file_path.name)
        self.drift_method.set("file")
        self.status.set(f"Selected drift correction file {self.drift_file_path.name}. Click Apply Drift Correction.")

    def apply_correction(self) -> None:
        if self.loaded is None:
            messagebox.showinfo("No file", "Load a localization CSV or Picasso HDF5 file first.")
            return
        self.roi_nm = None
        self._remove_roi_patch()
        self._remove_raw_roi_highlight()
        self._update_roi_label()
        self.status.set("Applying drift correction...")
        self._run_worker(self._correction_worker)

    def show_current_map(self, manual_pixel_override: bool = True) -> None:
        if self.corrected_locs is None:
            messagebox.showinfo("No corrected map", "Apply drift correction first.")
            return
        if manual_pixel_override:
            self._cancel_dynamic_render_for_manual_request()
        self.render_viewport_nm = self._shared_map_viewport_nm() or self._current_map_viewport_nm()
        render_px_nm = self._render_pixel_nm_for_request(
            self.map_axis,
            self.render_viewport_nm,
            allow_dynamic=not manual_pixel_override,
        )
        estimate = self._estimate_render_shape(self.render_viewport_nm, render_px_nm)
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
        self.status.set(f"Rendering corrected map at {render_px_nm:.3g} nm/pixel...")
        self._run_worker(lambda: self._render_current_corrected_worker(render_px_nm))

    def show_raw_map(self, auto_fit: bool = False) -> None:
        if self.loaded is None:
            messagebox.showinfo("No file", "Load a localization CSV or Picasso HDF5 file first.")
            return
        manual_pixel_override = not auto_fit
        if manual_pixel_override:
            self._cancel_dynamic_render_for_manual_request()
        self.raw_render_viewport_nm = self._shared_map_viewport_nm() or self._current_raw_map_viewport_nm()
        render_px_nm = self._render_pixel_nm_for_request(
            self.raw_map_axis,
            self.raw_render_viewport_nm,
            allow_dynamic=not manual_pixel_override,
        )
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
            messagebox.showinfo("No file", "Load a localization CSV or Picasso HDF5 file first.")
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
        self._cancel_dynamic_render_for_manual_request()
        self.render_viewport_nm = self._shared_map_viewport_nm() or self._current_map_viewport_nm()
        render_px_nm = self._render_pixel_nm_for_request(
            self.linked_map_axis,
            self.render_viewport_nm,
            allow_dynamic=False,
        )
        self.status.set(f"Rendering linked map at {render_px_nm:.3g} nm/pixel from cached collapsed linked events...")
        self._run_worker(lambda: self._link_color_worker(render_px_nm))

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

    def add_temporal_annotation(self) -> None:
        try:
            annotation = parse_temporal_vline_annotation(
                self.temporal_annotation_frame.get(), self.temporal_annotation_label.get()
            )
            if self.loaded is not None:
                frame_count = int(self.loaded.info[0].get("Frames", 0))
                if frame_count > 0 and annotation.frame >= frame_count:
                    raise ValueError(
                        f"Annotation frame must be between 0 and {frame_count - 1:,} for the loaded acquisition."
                    )
        except ValueError as exc:
            messagebox.showerror("Invalid temporal annotation", str(exc))
            return
        replaced = any(item.frame == annotation.frame for item in self.temporal_annotations)
        self.temporal_annotations = sorted(
            [item for item in self.temporal_annotations if item.frame != annotation.frame] + [annotation]
        )
        self._refresh_temporal_annotation_tree(select_frame=annotation.frame)
        self._draw_temporal_annotations()
        self.temporal_canvas.draw_idle()
        self.status.set(
            f"{'Updated' if replaced else 'Added'} temporal annotation at frame {annotation.frame:,}."
        )

    def remove_selected_temporal_annotations(self) -> None:
        selected = self.temporal_annotation_tree.selection()
        if not selected:
            messagebox.showinfo("No annotation selected", "Select one or more annotations to remove.")
            return
        frames = {int(self.temporal_annotation_tree.item(item, "values")[0]) for item in selected}
        self.temporal_annotations = [item for item in self.temporal_annotations if item.frame not in frames]
        self._refresh_temporal_annotation_tree()
        self._draw_temporal_annotations()
        self.temporal_canvas.draw_idle()
        self.status.set(f"Removed {len(frames)} temporal annotation{'s' if len(frames) != 1 else ''}.")

    def clear_temporal_annotations(self) -> None:
        count = len(self.temporal_annotations)
        self.temporal_annotations.clear()
        self.temporal_annotation_frame.set("")
        self.temporal_annotation_label.set("")
        self._refresh_temporal_annotation_tree()
        self._draw_temporal_annotations()
        self.temporal_canvas.draw_idle()
        self.status.set(f"Cleared {count} temporal annotation{'s' if count != 1 else ''}.")

    def _refresh_temporal_annotation_tree(self, select_frame: int | None = None) -> None:
        tree = self.temporal_annotation_tree
        tree.delete(*tree.get_children())
        selected_id = None
        for annotation in self.temporal_annotations:
            item_id = f"frame-{annotation.frame}"
            tree.insert("", "end", iid=item_id, values=(annotation.frame, annotation.label))
            if annotation.frame == select_frame:
                selected_id = item_id
        if selected_id is not None:
            tree.selection_set(selected_id)
            tree.see(selected_id)

    def _load_selected_temporal_annotation(self) -> None:
        selected = self.temporal_annotation_tree.selection()
        if len(selected) != 1:
            return
        values = self.temporal_annotation_tree.item(selected[0], "values")
        if len(values) >= 2:
            self.temporal_annotation_frame.set(str(values[0]))
            self.temporal_annotation_label.set(str(values[1]))

    def _draw_temporal_annotations(self) -> None:
        for artist in self.temporal_annotation_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.temporal_annotation_artists = []
        for axis in self.temporal_figure.axes:
            for annotation_index, annotation in enumerate(self.temporal_annotations):
                line = axis.axvline(
                    annotation.frame,
                    color="#dc2626",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=5,
                    label="_nolegend_",
                )
                label = axis.text(
                    annotation.frame,
                    0.98 - 0.06 * (annotation_index % 3),
                    annotation.label,
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    rotation_mode="anchor",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#b91c1c",
                    backgroundcolor=(1.0, 1.0, 1.0, 0.72),
                    clip_on=True,
                    zorder=6,
                )
                self.temporal_annotation_artists.extend((line, label))

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
            "source_roi_nm": self.roi_nm if bool(self.origami_use_roi.get()) else None,
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

        if bool(params["use_roi"]) and params["source_roi_nm"] is not None:
            selected = roi_locs(selected, params["source_roi_nm"], pixelsize)
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
            "locs": selected,
            "render_result": render_result,
            "source_label": f"{source} ({source_note})",
            "source_path": params["source_path"],
            "source_roi_nm": params["source_roi_nm"],
            "source_params": dict(params),
        }

    def _origami_candidate_stage_signature(self) -> tuple[Any, ...]:
        fingerprint = str(getattr(self, "origami_source_candidate_fingerprint", ""))
        if not fingerprint and self.origami_source_points_nm is not None:
            fingerprint = origami_source_fingerprint(self.origami_source_points_nm)
            self.origami_source_candidate_fingerprint = fingerprint
        return (
            float(self.origami_pick_bin_nm.get()),
            float(self.origami_connect_distance_nm.get()),
            float(self.origami_min_density_contrast.get()),
            int(self.origami_min_points.get()),
            int(len(self.origami_source_points_nm))
            if self.origami_source_points_nm is not None
            else 0,
            fingerprint,
        )

    def _run_origami_candidate_stage(self) -> None:
        if self.origami_identification_running:
            return
        if self.origami_source_points_nm is None or self.origami_loaded_source_path is None:
            messagebox.showinfo("Source not loaded", "Click Load Source Data before detecting candidates.")
            return
        try:
            signature = self._origami_candidate_stage_signature()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid candidate settings", str(exc))
            return
        (
            bin_size_nm,
            connect_distance_nm,
            density_threshold,
            minimum_points,
            _point_count,
            _source_fingerprint,
        ) = signature
        if (
            bin_size_nm <= 0.0
            or connect_distance_nm <= 0.0
            or not 0.0 <= density_threshold <= 1.0
            or minimum_points < 1
        ):
            messagebox.showerror(
                "Invalid candidate settings",
                "Pick bin and connect distance must be positive, minimum density must be between 0 and 1, and minimum points must be at least one.",
            )
            return
        if (
            self.origami_staged_candidates is not None
            and self.origami_staged_candidate_signature == signature
        ):
            candidate_count = len(self.origami_staged_candidates[0])
            self.origami_active_step = 1
            self.origami_identification_progress.set(100.0)
            self.origami_identification_progress_text.set(
                f"Step 1 reused: {candidate_count:,} cached candidates."
            )
            self.origami_step_progress[1].set(100.0)
            self.origami_step_progress_text[1].set(
                f"Cached: {candidate_count:,} candidates"
            )
            self.origami_staged_status.set(
                "Step 1 cache matches the current source and settings; no detection was rerun."
            )
            self._plot_origami_coarse_density()
            self.status.set(
                f"Reused {candidate_count:,} cached coarse candidates for the unchanged source and settings."
            )
            return
        points_nm = np.asarray(self.origami_source_points_nm, dtype=float).copy()
        source_path = self.origami_loaded_source_path
        self.origami_active_step = 1
        self.origami_step_progress[1].set(0.0)
        self.origami_step_progress_text[1].set("Detecting candidates…")
        self.origami_identification_running = True
        self.origami_identification_progress.set(0.0)
        self.origami_identification_progress_text.set("Step 1: building coarse candidates...")
        self.origami_run_candidates_button.state(["disabled"])
        self.origami_run_alignment_button.state(["disabled"])
        self.origami_run_sites_button.state(["disabled"])
        self.origami_run_acceptance_button.state(["disabled"])
        self._run_worker(
            lambda: (
                "origami_stage_candidates",
                {
                    "candidates": pick_origami_candidates(
                        points_nm,
                        bin_size_nm=bin_size_nm,
                        connect_distance_nm=connect_distance_nm,
                        density_threshold=density_threshold,
                        minimum_points=minimum_points,
                        progress_callback=self._origami_identification_worker_progress,
                    ),
                    "signature": signature,
                    "source_path": source_path,
                },
            )
        )

    def identify_origamis(self, target_stage: int = 5, display_stage: int | None = None) -> None:
        if self.origami_identification_running:
            return
        if self.origami_source_points_nm is None or self.origami_loaded_source_path is None:
            messagebox.showinfo("Source not loaded", "Click Load Source Data before identifying origami.")
            return
        if target_stage >= 3 and self.origami_shared_alignment_template is None:
            messagebox.showinfo(
                "Alignment template not loaded",
                "Load the Step 2 alignment template before fitting candidates.",
            )
            return
        if target_stage >= 4 and self.origami_digital_pixel_model is None:
            messagebox.showinfo(
                "Digital-pixel JSON not loaded",
                "Load the Step 3 digital-pixel group JSON before detecting digital outcomes.",
            )
            return
        if target_stage >= 5 and len(self.origami_custom_templates) < 2:
            messagebox.showinfo(
                "Classification templates not loaded",
                "Load the Step 4 classification template images before fitting digital outcomes.",
            )
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
                "template_pixel_size_x_nm": float(self.origami_template_pixel_x_nm.get()),
                "template_pixel_size_y_nm": float(self.origami_template_pixel_y_nm.get()),
                "rectangle_margin_nm": float(self.origami_rectangle_margin_nm.get()),
                "min_rectangle_confidence": float(self.origami_min_rectangle_confidence.get()),
                "use_correlation_gate": bool(self.origami_use_correlation_gate.get()),
                "min_cell_pattern_correlation": float(
                    self.origami_min_cell_pattern_correlation.get()
                ),
                "use_cell_pattern_gate": bool(self.origami_use_cell_pattern_gate.get()),
                "min_monte_carlo_probability": float(
                    self.origami_min_monte_carlo_probability.get()
                ),
                "site_mask_radius_nm": float(self.origami_site_mask_radius_nm.get()),
                "min_supported_sites": int(self.origami_min_supported_sites.get()),
                "min_site_evidence": float(self.origami_min_site_evidence.get()),
                "min_site_localizations": int(self.origami_min_site_localizations.get()),
                "min_supported_rows": int(self.origami_min_supported_rows.get()),
                "min_supported_columns": int(self.origami_min_supported_columns.get()),
                "max_site_spacing_error_nm": float(self.origami_max_site_spacing_error_nm.get()),
                "alignment_pixel_nm": float(self.origami_preview_pixel_nm.get()),
                "alignment_max_patch_pixels": int(self.origami_alignment_max_pixels.get()),
                "alignment_iterations": int(self.origami_alignment_iterations.get()),
                "template_mode": str(self.origami_template_mode.get()),
                "custom_template_name": (
                    custom_template_display_name(self.origami_custom_template_path)
                    if self.origami_custom_template_path is not None
                    else ""
                ),
                "source_path": self.origami_loaded_source_path,
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid identification settings", str(exc))
            return
        if params["template_mode"] == "Custom image":
            if self.origami_custom_template_image is None:
                messagebox.showerror(
                    "Custom template not loaded",
                    "Choose Load Custom Template Image before running identification with Custom image selected.",
                )
                return
            if params["template_pixel_size_x_nm"] <= 0.0 or params["template_pixel_size_y_nm"] <= 0.0:
                messagebox.showerror(
                    "Invalid template calibration",
                    "Template pixel x/y sizes must both be greater than zero nanometres.",
                )
                return
            if len(self.origami_custom_templates) > 1:
                params["custom_templates"] = [
                    {
                        **template,
                        "image": np.asarray(template["image"], dtype=float).copy(),
                    }
                    for template in self.origami_custom_templates
                ]
                params["alignment_template_image"] = np.asarray(
                    self.origami_custom_templates[0]["image"], dtype=float
                ).copy()
                if self.origami_shared_alignment_template is not None:
                    params["shared_alignment_template"] = {
                        **self.origami_shared_alignment_template,
                        "image": np.asarray(
                            self.origami_shared_alignment_template["image"], dtype=float
                        ).copy(),
                    }
            else:
                params["alignment_template_image"] = self.origami_custom_template_image.copy()
        else:
            params["alignment_template_image"] = None
        if not 0.0 <= params["min_rectangle_confidence"] <= 1.0:
            messagebox.showerror("Invalid identification settings", "Minimum theoretical-template correlation must be between 0 and 1.")
            return
        if not 0.0 <= params["min_monte_carlo_probability"] <= 1.0:
            messagebox.showerror(
                "Invalid identification settings",
                "Minimum winning-template probability must be between 0 and 1.",
            )
            return
        if not -1.0 <= params["min_cell_pattern_correlation"] <= 1.0:
            messagebox.showerror(
                "Invalid identification settings",
                "Minimum cell-pattern correlation must be between -1 and 1.",
            )
            return
        if params["site_mask_radius_nm"] <= 0:
            messagebox.showerror("Invalid identification settings", "Site-mask radius must be positive.")
            return
        if (
            params["min_supported_sites"] < 1
            or params["min_site_localizations"] < 1
            or params["min_supported_rows"] < 1
            or params["min_supported_columns"] < 1
            or not 0.0 <= params["min_site_evidence"] <= 1.0
            or params["min_supported_sites"] > params["rows"] * params["columns"]
            or params["min_supported_rows"] > params["rows"]
            or params["min_supported_columns"] > params["columns"]
            or params["max_site_spacing_error_nm"] <= 0
        ):
            messagebox.showerror(
                "Invalid identification settings",
                "Check sparse-site support: counts must fit the configured grid, site prominence must be between 0 and 1, and maximum spacing error must be positive.",
            )
            return
        if (
            params["alignment_pixel_nm"] <= 0
            or params["alignment_max_patch_pixels"] < 16
            or params["alignment_iterations"] < 1
        ):
            messagebox.showerror(
                "Invalid identification settings",
                "Alignment pixel must be positive, maximum alignment pixels must be at least 16, and alignment passes must be at least 1.",
            )
            return
        points_nm = self.origami_source_points_nm.copy()
        params["_inspection_stage"] = int(target_stage)
        params["_display_stage"] = int(display_stage if display_stage is not None else 4)
        # Classification-template loading changes origami_custom_template_path
        # and its display name. Those are not alignment inputs and must not
        # invalidate the locked Step 2 pose. Identify the actual shared
        # alignment raster and calibration instead.
        shared_alignment = self.origami_shared_alignment_template
        if shared_alignment is not None:
            alignment_image = np.ascontiguousarray(
                np.asarray(shared_alignment["image"], dtype=np.float32)
            )
            params["_alignment_template_signature"] = (
                str(shared_alignment.get("path", "")),
                str(shared_alignment.get("name", "")),
                alignment_image.shape,
                hashlib.sha1(alignment_image.view(np.uint8)).hexdigest(),
                int(shared_alignment["rows"]),
                int(shared_alignment["columns"]),
                float(shared_alignment["spacing_x_nm"]),
                float(shared_alignment["spacing_y_nm"]),
                float(shared_alignment["rectangle_margin_nm"]),
                float(shared_alignment["template_pixel_size_x_nm"]),
                float(shared_alignment["template_pixel_size_y_nm"]),
            )
        if self.origami_digital_pixel_model is not None:
            params["digital_pixel_model"] = dict(self.origami_digital_pixel_model)
        current_candidate_signature = self._origami_candidate_stage_signature()
        if (
            self.origami_staged_candidates is not None
            and self.origami_staged_candidate_signature == current_candidate_signature
        ):
            params["_precomputed_candidates"] = self.origami_staged_candidates
        previous_params = self.origami_identification_params or {}
        if (
            target_stage == 4
            and self.origami_pick_result is not None
            and int(previous_params.get("_inspection_stage", 0)) == 3
            and origami_stage_cache_matches(
                previous_params,
                params,
                ORIGAMI_ALIGNMENT_CACHE_KEYS,
            )
            and self.origami_staged_candidate_signature == current_candidate_signature
        ):
            params["_prealigned_picks"] = self.origami_pick_result
        if (
            target_stage == 5
            and self.origami_pick_result is not None
            and int(previous_params.get("_inspection_stage", 0)) == 4
            and origami_stage_cache_matches(
                previous_params,
                params,
                ORIGAMI_ALIGNMENT_CACHE_KEYS + ORIGAMI_SITE_CACHE_KEYS,
            )
            and self.origami_staged_candidate_signature == current_candidate_signature
            and (
                len(params.get("custom_templates", ())) < 2
                or params.get("shared_alignment_template") is not None
            )
        ):
            params["_premeasured_picks"] = self.origami_pick_result
            # The mask is candidate-local, so it must never be reused on a
            # different tile.  It is, however, part of the cached Step 2/3
            # state for this same validation ROI.  Preserve it when Step 4
            # consumes the staged measurements so cached and fresh complete
            # pipelines enforce the identical alignment gate.
            previous_alignment_mask = previous_params.get("_alignment_accepted_mask")
            if previous_alignment_mask is not None:
                params["_alignment_accepted_mask"] = tuple(
                    bool(value) for value in previous_alignment_mask
                )
        self.origami_identification_generation += 1
        params["identification_generation"] = self.origami_identification_generation
        self.origami_pending_identification_snapshot = self._origami_identification_snapshot()

        # A new identification run invalidates every candidate- and
        # overlay-derived result immediately. Keeping the previous overlay
        # alive until the worker completed made its galleries and site
        # statistics look as though they had been produced by the new run.
        self.origami_pick_result = None
        self.origami_identification_params = None
        self.origami_result = None
        self.origami_multi_template_results = {}
        self.origami_multi_template_counts = {}
        self.origami_multi_template_overlay_results = {}
        self.origami_multi_template_overlays_building = set()
        self.origami_single_overlay_building = False
        self.origami_multi_template_unclassified_count = 0
        self.origami_multi_template_unclassified_centers_nm = np.empty((0, 2), dtype=float)
        self.origami_multi_template_unclassified_details = []
        self.origami_template_result_view.set("All templates")
        self.origami_template_result_combo.state(["disabled"])
        self.origami_result_source = ""
        self.origami_result_source_count = 0
        self.origami_result_render_settings = None
        self.origami_density_cache_key = None
        self.origami_density_cache = None
        self.origami_gallery_current_indices = np.empty(0, dtype=int)
        self.origami_selected_index = None
        self.origami_random_inspection_payload = None
        self.origami_inspected_tile_indices.clear()
        self.origami_roi_history.clear()
        self.origami_roi_history_position = -1
        self.origami_pending_roi_view = None
        self.origami_selected_match_index = None
        self.origami_match_roi.set(1)
        self.origami_match_roi_label.set("Candidate –/–")
        self.origami_tiled_button.state(["disabled"])
        self.origami_n_tiles_button.state(["disabled"])
        self.origami_random_roi_button.state(["disabled"])

        self.origami_identification_running = True
        self.origami_identification_progress.set(0.0)
        self.origami_identification_progress_text.set(f"Starting with {len(points_nm):,} source points...")
        self.origami_identify_button.state(["disabled"])
        self.origami_run_candidates_button.state(["disabled"])
        self.origami_run_alignment_button.state(["disabled"])
        self.origami_run_sites_button.state(["disabled"])
        self.origami_run_acceptance_button.state(["disabled"])
        self.origami_tiled_button.state(["disabled"])
        self.origami_n_tiles_button.state(["disabled"])
        self.origami_random_roi_button.state(["disabled"])
        self._refresh_origami_action_states()
        self._plot_origami_source_data()
        shown_stage = int(params["_display_stage"])
        self.origami_active_step = shown_stage
        self.origami_step_progress[shown_stage].set(0.0)
        self.origami_step_progress_text[shown_stage].set("Running…")
        self.origami_staged_status.set(f"Running Step {shown_stage}...")
        self.status.set(
            f"Running origami workflow through Step {shown_stage} in {len(points_nm):,} loaded source points..."
        )
        self._run_worker(lambda: self._identify_origami_worker(points_nm, params))

    def _identify_origami_with_params(
        self,
        points_nm: np.ndarray,
        params: dict[str, Any],
        progress_callback: Callable[[float, str], None],
        *,
        precomputed_candidates: tuple[Any, ...] | None = None,
        candidate_image_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
    ) -> OrigamiPickResult:
        return identify_origami_regions(
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
            use_correlation_gate=bool(params.get("use_correlation_gate", True)),
            site_mask_radius_nm=float(params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
            min_supported_sites=int(params.get("min_supported_sites", 0)),
            min_site_evidence=float(params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
            min_site_localizations=int(params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
            min_supported_rows=int(params.get("min_supported_rows", 0)),
            min_supported_columns=int(params.get("min_supported_columns", 0)),
            max_site_spacing_error_nm=float(params.get("max_site_spacing_error_nm", float("inf"))),
            alignment_pixel_nm=float(params["alignment_pixel_nm"]),
            alignment_max_patch_pixels=int(params.get("alignment_max_patch_pixels", DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS)),
            alignment_iterations=int(params["alignment_iterations"]),
            alignment_template_image=params.get("alignment_template_image"),
            template_pixel_size_x_nm=float(params["template_pixel_size_x_nm"]),
            template_pixel_size_y_nm=float(params["template_pixel_size_y_nm"]),
            progress_callback=progress_callback,
            compute_grid_blob_bic=bool(params.get("compute_grid_blob_bic", True)),
            measure_sites=int(params.get("_inspection_stage", 5)) not in {3, 4},
            precomputed_candidates=precomputed_candidates,
            candidate_image_cache=candidate_image_cache,
        )

    def _remeasure_origami_sites(
        self,
        picks: OrigamiPickResult,
        params: dict[str, Any],
        progress_callback: Callable[[float, str], None],
    ) -> OrigamiPickResult:
        """Measure Stage 4 on locked Stage 3 poses without repeating alignment."""
        alignment_points = np.asarray(picks.template_points_nm, dtype=float).reshape(-1, 2)
        if len(alignment_points):
            params["alignment_template_overlay_points_nm"] = tuple(
                tuple(float(value) for value in point) for point in alignment_points
            )
        rows = int(params["rows"])
        columns = int(params["columns"])
        spacing_x_nm = float(params["spacing_x_nm"])
        spacing_y_nm = float(params["spacing_y_nm"])
        radius_nm = float(params["site_mask_radius_nm"])
        full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
        # Alignment may use only an L-shaped subset. Digital-pixel detection
        # deliberately samples the complete physical lattice after that pose
        # has been locked.
        template_grid = full_grid
        _distance, template_indices = cKDTree(full_grid).query(template_grid, k=1)
        template_indices = np.asarray(template_indices, dtype=int)
        row_ids = template_indices // columns
        column_ids = template_indices % columns
        candidate_count = len(picks.aligned_regions)
        lattice_count = np.zeros((candidate_count, len(full_grid)), dtype=int)
        lattice_prominence = np.zeros((candidate_count, len(full_grid)), dtype=float)
        lattice_supported = np.zeros((candidate_count, len(full_grid)), dtype=bool)
        site_count = np.zeros((candidate_count, len(template_grid)), dtype=int)
        site_prominence = np.zeros((candidate_count, len(template_grid)), dtype=float)
        peak_positions = np.full((candidate_count, len(template_grid), 2), np.nan)
        boundary_reference_positions = np.full(
            (candidate_count, len(template_grid), 2), np.nan
        )
        boundary_points = np.full(
            (candidate_count, len(template_grid), 32, 2), np.nan
        )
        centroids = np.full((candidate_count, len(template_grid), 2), np.nan)
        supported_counts = np.zeros(candidate_count, dtype=int)
        supported_rows = np.zeros(candidate_count, dtype=int)
        supported_columns = np.zeros(candidate_count, dtype=int)
        spacing_rms = np.full(candidate_count, float("inf"))
        spacing_max = np.full(candidate_count, float("inf"))
        for index, region in enumerate(picks.aligned_regions):
            evidence = sparse_site_evidence_diagnostics(
                region,
                full_grid,
                site_radius_nm=radius_nm,
            )
            lattice_count[index] = evidence.counts
            lattice_prominence[index] = evidence.prominence
            supported_full = (
                (evidence.counts >= int(params["min_site_localizations"]))
                & (evidence.prominence >= float(params["min_site_evidence"]))
            )
            lattice_supported[index] = supported_full
            site_count[index] = evidence.counts[template_indices]
            site_prominence[index] = evidence.prominence[template_indices]
            peak_positions[index] = evidence.peak_positions_nm[template_indices]
            boundary_reference_positions[index] = (
                evidence.boundary_reference_positions_nm[template_indices]
            )
            boundary_points[index] = evidence.boundary_points_nm[template_indices]
            supported = supported_full[template_indices]
            centroids[index] = supported_site_centroids(
                region,
                template_grid,
                supported,
                site_radius_nm=radius_nm,
            )
            supported_indices = np.flatnonzero(supported)
            supported_counts[index] = len(supported_indices)
            if len(supported_indices):
                supported_rows[index] = len(np.unique(row_ids[supported_indices]))
                supported_columns[index] = len(np.unique(column_ids[supported_indices]))
            spacing_rms[index], spacing_max[index] = supported_site_spacing_errors(
                region,
                template_grid,
                supported,
                site_radius_nm=radius_nm,
            )
            if index + 1 == candidate_count or (index + 1) % max(1, candidate_count // 30) == 0:
                progress_callback(
                    100.0 * (index + 1) / max(candidate_count, 1),
                    f"Digital-pixel site detection: {index + 1:,}/{candidate_count:,} candidates...",
                )
        gap_contrast, on_site_fraction, _area = site_gap_contrast_for_regions(
            picks.aligned_regions,
            template_grid,
            rectangle_width_nm=float(picks.rectangle_width_nm),
            rectangle_height_nm=float(picks.rectangle_height_nm),
            site_radius_nm=radius_nm,
        )
        params["_alignment_accepted_mask"] = tuple(
            bool(value) for value in np.asarray(picks.accepted_mask, dtype=bool)
        )
        self._measure_direct_digital_groups(picks, params)
        return replace(
            picks,
            accepted_mask=np.ones(candidate_count, dtype=bool),
            site_mask_radius_nm=radius_nm,
            # Stage 2 may align with only the L_L subset. Stage 3 measures and
            # displays the complete physical lattice, so every site-indexed
            # diagnostic must use this same 8x12 (or custom-sized) grid.
            template_points_nm=template_grid,
            site_gap_contrast=gap_contrast,
            on_site_fraction=on_site_fraction,
            lattice_site_localization_counts=lattice_count,
            lattice_site_prominence=lattice_prominence,
            lattice_supported_sites=lattice_supported,
            site_localization_counts=site_count,
            site_prominence=site_prominence,
            site_peak_positions_nm=peak_positions,
            site_boundary_reference_positions_nm=boundary_reference_positions,
            site_boundary_points_nm=boundary_points,
            site_centroids_nm=centroids,
            supported_site_count=supported_counts,
            supported_row_count=supported_rows,
            supported_column_count=supported_columns,
            site_spacing_rms_nm=spacing_rms,
            site_spacing_max_error_nm=spacing_max,
        )

    @staticmethod
    def _measure_direct_digital_groups(
        picks: OrigamiPickResult,
        params: dict[str, Any],
    ) -> None:
        """Cache one direct localization measurement per user-defined group."""
        digital_model = params.get("digital_pixel_model")
        if not isinstance(digital_model, dict):
            return
        geometry = params.get("shared_alignment_template")
        if not isinstance(geometry, dict):
            geometry = params
        physical_shape = digital_model.get("physical_shape", ())
        rows = int(params.get("rows", geometry.get("rows", physical_shape[0])))
        columns = int(params.get("columns", geometry.get("columns", physical_shape[1])))
        full_grid = ideal_grid_points(
            rows,
            columns,
            float(params.get("spacing_x_nm", geometry["spacing_x_nm"])),
            float(params.get("spacing_y_nm", geometry["spacing_y_nm"])),
        )
        group_cells = digital_model.get(
            "bit_physical_cells", digital_model.get("bit_cells", ())
        )
        template_states = [
            np.asarray(template.get("logical_model", {}).get("active_bits", ()), dtype=float)
            for template in params.get("custom_templates", ())
            if isinstance(template.get("logical_model"), dict)
        ]
        valid_template_states = [
            state for state in template_states
            if state.shape == (len(digital_model["bit_ids"]),)
        ]
        group_on_priors = (
            np.mean(np.stack(valid_template_states, axis=0), axis=0)
            if valid_template_states
            else np.full(len(digital_model["bit_ids"]), 0.5, dtype=float)
        )
        group_evidence = direct_digital_group_localization_evidence(
            picks.aligned_regions,
            full_grid,
            group_cells,
            assignment_radius_nm=float(params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
        )
        _raw_posterior, _raw_log_bayes, raw_group_probability, _raw_correlation = (
            digital_group_template_evidence(
                group_evidence,
                np.zeros(len(digital_model["bit_ids"]), dtype=bool),
                group_cells,
            )
        )
        _posterior, _log_bayes, group_probability, _correlation = (
            digital_group_template_evidence(
                group_evidence,
                np.zeros(len(digital_model["bit_ids"]), dtype=bool),
                group_cells,
                minimum_support_per_position=float(
                    params.get("min_site_localizations", 0)
                ),
                minimum_group_prominence=float(params.get("min_site_evidence", 0.0)),
            )
        )
        params["digital_pixel_ids"] = tuple(digital_model["bit_ids"])
        params["digital_group_expected_on_fractions"] = tuple(
            float(value) for value in group_on_priors
        )
        params["digital_group_localization_evidence"] = tuple(
            tuple(float(value) for value in row) for row in group_evidence
        )
        params["digital_group_prominences"] = tuple(
            tuple(float(max(0.0, 2.0 * value - 1.0)) for value in row)
            for row in raw_group_probability
        )
        params["digital_pixel_probabilities"] = tuple(
            tuple(float(value) for value in row) for row in group_probability
        )

    @staticmethod
    def _apply_origami_acceptance_filters(
        picks: OrigamiPickResult,
        params: dict[str, Any],
    ) -> OrigamiPickResult:
        confidence = np.asarray(picks.rectangle_confidence, dtype=float)
        accepted = (
            (np.asarray(picks.point_counts, dtype=int) >= int(params.get("min_candidate_points", 0)))
            & (np.asarray(picks.point_counts, dtype=int) <= int(params.get("max_candidate_points", np.iinfo(np.int64).max)))
            & (
                (confidence >= float(params.get("min_rectangle_confidence", -float("inf"))))
                if bool(params.get("use_correlation_gate", True))
                else True
            )
        )
        if isinstance(params.get("digital_pixel_model"), dict):
            alignment_accepted = np.asarray(
                params.get("_alignment_accepted_mask", ()), dtype=bool
            )
            if alignment_accepted.shape == accepted.shape:
                accepted &= alignment_accepted
        else:
            accepted &= (
                (np.asarray(picks.supported_site_count, dtype=int) >= int(params.get("min_supported_sites", 0)))
                & (np.asarray(picks.supported_row_count, dtype=int) >= int(params.get("min_supported_rows", 0)))
                & (np.asarray(picks.supported_column_count, dtype=int) >= int(params.get("min_supported_columns", 0)))
                & (
                    np.asarray(picks.site_spacing_max_error_nm, dtype=float)
                    <= float(params.get("max_site_spacing_error_nm", float("inf")))
                )
            )
        return replace(picks, accepted_mask=np.asarray(accepted, dtype=bool))

    def _identify_origami_worker(
        self,
        points_nm: np.ndarray,
        params: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[str, Any]:
        emit_progress = progress_callback or self._origami_identification_worker_progress
        templates = list(params.get("custom_templates", ()))
        inspection_stage = int(params.get("_inspection_stage", 5))
        precomputed_candidates = params.get("_precomputed_candidates")
        prealigned_picks = params.get("_prealigned_picks")
        premeasured_picks = params.get("_premeasured_picks")
        if inspection_stage == 4 and isinstance(prealigned_picks, OrigamiPickResult):
            measured = self._remeasure_origami_sites(
                prealigned_picks,
                params,
                emit_progress,
            )
            return "origami_stage_preview", {
                "stage": int(params.get("_display_stage", 3)),
                "picks": measured,
                "source_path": params["source_path"],
                "params": dict(params),
            }
        if len(templates) < 2:
            if isinstance(premeasured_picks, OrigamiPickResult):
                picks = self._apply_origami_acceptance_filters(premeasured_picks, params)
            else:
                picks = self._identify_origami_with_params(
                    points_nm,
                    params,
                    emit_progress,
                    precomputed_candidates=precomputed_candidates,
                )
            if inspection_stage == 4:
                picks = self._remeasure_origami_sites(picks, params, emit_progress)
            if inspection_stage in {3, 4}:
                return "origami_stage_preview", {
                    "stage": int(params.get("_display_stage", inspection_stage)),
                    "picks": replace(
                        picks,
                        accepted_mask=np.ones(len(picks.regions), dtype=bool),
                    ),
                    "source_path": params["source_path"],
                    "params": dict(params),
                }
            return "origami_picks", {"picks": picks, "source_path": params["source_path"], "params": dict(params)}

        template_results: list[dict[str, Any]] = []
        template_count = len(templates)
        candidate_image_cache: dict[tuple[object, ...], tuple[object, ...]] = {}
        emit_progress(1.0, "Building one shared candidate set...")
        shared_candidates = precomputed_candidates
        if shared_candidates is None:
            shared_candidates = pick_origami_candidates(
                points_nm,
                bin_size_nm=float(params["pick_bin_size_nm"]),
                connect_distance_nm=float(params["connect_distance_nm"]),
                density_threshold=float(params["density_threshold"]),
                minimum_points=int(params["min_candidate_points"]),
            )
        emit_progress(
            2.0,
            f"Retained {len(shared_candidates[0]):,} candidates with at least "
            f"{int(params['min_candidate_points']):,} source points.",
        )
        shared_alignment = params.get("shared_alignment_template")
        locked_picks: OrigamiPickResult | None = (
            premeasured_picks if isinstance(premeasured_picks, OrigamiPickResult) else None
        )
        if locked_picks is not None:
            self._measure_direct_digital_groups(locked_picks, params)
            locked_picks = self._apply_origami_acceptance_filters(locked_picks, params)
        if locked_picks is not None:
            emit_progress(
                5.0,
                "Reusing Step 2 alignment and Step 3 digital-site measurements…",
            )
        if shared_alignment is not None:
            if not all(template.get("logical_model") is not None for template in templates):
                raise ValueError(
                    "A shared alignment template requires logical-bit metadata in every "
                    "classification template. Regenerate the classification templates in "
                    "Picklist Generator with digital groups defined."
                )
            geometry_fields = (
                "rows",
                "columns",
                "spacing_x_nm",
                "spacing_y_nm",
                "rectangle_margin_nm",
            )
            for template in templates:
                for field in geometry_fields:
                    left = float(template[field])
                    right = float(shared_alignment[field])
                    if not np.isclose(left, right, rtol=0.0, atol=1e-9):
                        raise ValueError(
                            f"Shared alignment template {shared_alignment['name']} and "
                            f"classification template {template['name']} disagree on {field}."
                        )
            locked_params = dict(params)
            locked_params.update(
                {
                    "rows": int(shared_alignment["rows"]),
                    "columns": int(shared_alignment["columns"]),
                    "spacing_x_nm": float(shared_alignment["spacing_x_nm"]),
                    "spacing_y_nm": float(shared_alignment["spacing_y_nm"]),
                    "rectangle_margin_nm": float(shared_alignment["rectangle_margin_nm"]),
                    "template_pixel_size_x_nm": float(shared_alignment["template_pixel_size_x_nm"]),
                    "template_pixel_size_y_nm": float(shared_alignment["template_pixel_size_y_nm"]),
                    "alignment_template_image": np.asarray(shared_alignment["image"], dtype=float),
                    "custom_template_name": str(shared_alignment["name"]),
                    "compute_grid_blob_bic": False,
                    # Step 2 does not measure physical or digital sites.  Do
                    # not feed the later-stage site thresholds into
                    # identify_origami_regions here: its unmeasured support
                    # arrays are zero, which would silently reject every pose
                    # before Step 3 gets a chance to measure digital groups.
                    "min_supported_sites": 0,
                    "min_supported_rows": 0,
                    "min_supported_columns": 0,
                    "max_site_spacing_error_nm": float("inf"),
                }
            )
            # Step 2 fits only the shared alignment image. Site measurement is
            # a separate Step 3 operation on the complete physical lattice.
            locked_params["_inspection_stage"] = 3

            def locked_progress(percent: float, message: str) -> None:
                upper = 88.0 if inspection_stage < 5 else 50.0
                emit_progress(
                    2.0 + upper * float(percent) / 100.0,
                    f"Step 2 · shared alignment ({shared_alignment['name']}): {message}",
                )

            if locked_picks is None:
                locked_picks = self._identify_origami_with_params(
                    points_nm,
                    locked_params,
                    locked_progress,
                    precomputed_candidates=shared_candidates,
                    candidate_image_cache=candidate_image_cache,
                )
                if inspection_stage >= 5 and isinstance(
                    params.get("digital_pixel_model"), dict
                ):
                    def digital_progress(percent: float, message: str) -> None:
                        emit_progress(
                            52.0 + 36.0 * float(percent) / 100.0,
                            f"Step 3 · digital-pixel detection: {message}",
                        )

                    locked_picks = self._remeasure_origami_sites(
                        locked_picks,
                        params,
                        digital_progress,
                    )
                    self._measure_direct_digital_groups(locked_picks, params)
                    locked_picks = self._apply_origami_acceptance_filters(
                        locked_picks, params
                    )
            emit_progress(
                90.0,
                f"Step 4 · alignment locked to {shared_alignment['name']}; "
                f"classifying {template_count} digital templates...",
            )
        templates_to_fit = (
            templates[:1]
            if inspection_stage in {3, 4} and locked_picks is None
            else templates
        )
        for template_index, template in enumerate(templates_to_fit):
            template_params = dict(params)
            template_params.update(
                {
                    "rows": int(template["rows"]),
                    "columns": int(template["columns"]),
                    "spacing_x_nm": float(template["spacing_x_nm"]),
                    "spacing_y_nm": float(template["spacing_y_nm"]),
                    "rectangle_margin_nm": float(template["rectangle_margin_nm"]),
                    "template_pixel_size_x_nm": float(template["template_pixel_size_x_nm"]),
                    "template_pixel_size_y_nm": float(template["template_pixel_size_y_nm"]),
                    "alignment_template_image": np.asarray(template["image"], dtype=float),
                    "custom_template_name": str(template["name"]),
                    "logical_model": template.get("logical_model"),
                    "locked_alignment_template_name": (
                        str(shared_alignment["name"]) if shared_alignment is not None else ""
                    ),
                    # ΔBIC is display-only QC and requires several NNLS fits
                    # per candidate. It does not participate in multi-template
                    # acceptance or classification, so defer it here.
                    "compute_grid_blob_bic": False,
                }
            )

            if locked_picks is not None:
                picks = locked_picks
            else:
                def template_progress(percent: float, message: str, *, index: int = template_index) -> None:
                    overall = 2.0 + 98.0 * (index + float(percent) / 100.0) / template_count
                    emit_progress(
                        overall,
                        f"Template {index + 1}/{template_count} ({templates[index]['name']}): {message}",
                    )

                picks = self._identify_origami_with_params(
                    points_nm,
                    template_params,
                    template_progress,
                    precomputed_candidates=shared_candidates,
                    candidate_image_cache=candidate_image_cache,
                )
                if template.get("logical_model") is not None:
                    self._measure_direct_digital_groups(picks, template_params)
                    picks = self._apply_origami_acceptance_filters(
                        picks, template_params
                    )
            template_results.append(
                {"name": str(template["name"]), "picks": picks, "params": template_params}
            )

        if inspection_stage in {3, 4}:
            preview_result = locked_picks or template_results[0]["picks"]
            preview_params = (
                dict(params)
                if locked_picks is not None
                else dict(template_results[0]["params"])
            )
            if inspection_stage == 4:
                preview_result = self._remeasure_origami_sites(
                    preview_result,
                    preview_params,
                    emit_progress,
                )
            return "origami_stage_preview", {
                "stage": int(params.get("_display_stage", inspection_stage)),
                "picks": replace(
                    preview_result,
                    accepted_mask=np.ones(len(preview_result.regions), dtype=bool),
                ),
                "source_path": params["source_path"],
                "params": preview_params,
            }

        centers_by_template = [
            np.asarray([np.median(region, axis=0) for region in result["picks"].regions], dtype=float).reshape(-1, 2)
            for result in template_results
        ]
        classification_scores: list[np.ndarray] = []
        for result in template_results:
            picks = result["picks"]
            result_params = result["params"]
            logical_model = result_params.get("logical_model")
            logical_bit_probability: np.ndarray | None = None
            if logical_model is not None:
                direct_group_evidence = np.asarray(
                    result_params.get("digital_group_localization_evidence", ()),
                    dtype=float,
                )
                expected_shape = (
                    len(picks.aligned_regions),
                    len(logical_model["bit_ids"]),
                )
                if direct_group_evidence.size == 0 and expected_shape[0] == 0:
                    direct_group_evidence = np.empty(expected_shape, dtype=float)
                if direct_group_evidence.shape != expected_shape:
                    raise ValueError(
                        "Direct digital-group measurements are missing or stale. "
                        "Run Step 3 again before fitting templates."
                    )
                (
                    monte_carlo_posterior,
                    monte_carlo_log_bayes_factor,
                    logical_bit_probability,
                    cell_pattern_correlation,
                ) = digital_group_template_evidence(
                    direct_group_evidence,
                    logical_model["active_bits"],
                    logical_model.get(
                        "bit_physical_cells", logical_model["bit_cells"]
                    ),
                    minimum_support_per_position=float(
                        result_params.get("min_site_localizations", 0)
                    ),
                    minimum_group_prominence=float(
                        result_params.get("min_site_evidence", 0.0)
                    ),
                )
                active_bits = np.asarray(logical_model["active_bits"], dtype=bool)
                candidate_count = len(logical_bit_probability)
                bright_cell_probability = (
                    np.mean(logical_bit_probability[:, active_bits], axis=1)
                    if np.any(active_bits)
                    else np.ones(candidate_count, dtype=float)
                )
                empty_cell_fraction = (
                    np.mean(1.0 - logical_bit_probability[:, ~active_bits], axis=1)
                    if np.any(~active_bits)
                    else np.ones(candidate_count, dtype=float)
                )
                lattice_precision = bright_cell_probability
                scores = monte_carlo_log_bayes_factor
                # This compatibility field is used by older diagnostic views.
                # In the digital path it contains group-only evidence and must
                # never reintroduce physical-site support into classification.
                cell_agreement = np.asarray(scores, dtype=float)
            else:
                (
                    bright_cell_probability,
                    empty_cell_fraction,
                    cell_pattern_correlation,
                ) = lattice_count_template_probability_agreement(
                    picks.lattice_site_localization_counts,
                    picks.template_points_nm,
                    rows=int(result_params["rows"]),
                    columns=int(result_params["columns"]),
                    spacing_x_nm=float(result_params["spacing_x_nm"]),
                    spacing_y_nm=float(result_params["spacing_y_nm"]),
                )
                lattice_precision = bright_cell_probability
            if logical_model is None:
                cell_agreement = bidirectional_template_classification_scores(
                    picks.rectangle_confidence,
                    lattice_precision,
                    picks.supported_site_count,
                    len(picks.template_points_nm),
                    off_site_empty_fractions=empty_cell_fraction,
                    bright_site_probabilities=bright_cell_probability,
                    cell_pattern_correlations=cell_pattern_correlation,
                )
                # Legacy PNG-only templates retain the earlier inexpensive
                # cell-agreement classifier. JSON logical-bit templates use
                # the bit evidence above and do not repeat this physical-cell
                # separation as a second classifier.
                scores = cell_agreement
                monte_carlo_posterior = np.ones(len(scores), dtype=float)
                monte_carlo_log_bayes_factor = np.asarray(scores, dtype=float)
            crop_retention = np.clip(picks.crop_retained_fractions, 1e-6, 1.0)
            result["params"]["classification_lattice_precision"] = tuple(
                float(value) for value in lattice_precision
            )
            result["params"]["classification_empty_cell_fraction"] = tuple(
                float(value) for value in empty_cell_fraction
            )
            result["params"]["classification_bright_cell_probability"] = tuple(
                float(value) for value in bright_cell_probability
            )
            result["params"]["classification_cell_pattern_correlation"] = tuple(
                float(value) for value in cell_pattern_correlation
            )
            result["params"]["classification_cell_agreement"] = tuple(
                float(value) for value in cell_agreement
            )
            result["params"]["classification_min_cell_pattern_correlation"] = (
                float(params.get(
                    "min_cell_pattern_correlation",
                    DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION,
                ))
            )
            result["params"]["classification_use_cell_pattern_gate"] = bool(
                params.get(
                    "use_cell_pattern_gate",
                    DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE,
                )
            )
            result["params"]["classification_monte_carlo_posterior"] = tuple(
                float(value) for value in monte_carlo_posterior
            )
            result["params"]["classification_monte_carlo_log_bayes_factor"] = tuple(
                float(value) for value in monte_carlo_log_bayes_factor
            )
            if logical_bit_probability is not None:
                result["params"]["classification_logical_bit_ids"] = tuple(
                    str(value) for value in logical_model["bit_ids"]
                )
                result["params"]["classification_active_logical_bits"] = tuple(
                    bool(value) for value in logical_model["active_bits"]
                )
                result["params"]["classification_logical_bit_probabilities"] = tuple(
                    tuple(float(value) for value in row)
                    for row in logical_bit_probability
                )
                result["params"]["classification_evidence_model"] = "direct digital groups"
            else:
                result["params"]["classification_evidence_model"] = "cell agreement"
            result["params"]["classification_scores"] = tuple(float(value) for value in scores)
            result["params"]["classification_crop_retained_fraction"] = tuple(
                float(value) for value in crop_retention
            )
            classification_scores.append(scores)
        base_match_distance_nm = max(
            float(params["connect_distance_nm"]),
            2.0 * float(params["pick_bin_size_nm"]),
        )
        largest_template_span_nm = max(
            float(np.max(np.ptp(result["picks"].template_points_nm, axis=0)))
            if len(result["picks"].template_points_nm) > 1
            else 0.0
            for result in template_results
        )
        deduplication_distance_nm = max(
            base_match_distance_nm,
            0.4 * largest_template_span_nm,
        )
        digital_classification = all(
            result["params"].get("classification_evidence_model")
            == "direct digital groups"
            for result in template_results
        )
        classification = classify_template_candidates(
            centers_by_template,
            [
                np.asarray(result["picks"].accepted_mask, dtype=bool)
                & (
                    np.ones(len(result["picks"].regions), dtype=bool)
                    if digital_classification
                    else (
                        np.asarray(
                            result["params"]["classification_monte_carlo_posterior"],
                            dtype=float,
                        )
                        >= float(params.get(
                            "min_monte_carlo_probability",
                            DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR,
                        )) - 1e-9
                    )
                )
                & (
                    (
                        np.asarray(
                            result["params"]["classification_cell_pattern_correlation"],
                            dtype=float,
                        )
                        >= float(result["params"].get(
                            "classification_min_cell_pattern_correlation",
                            DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION,
                        ))
                    )
                    if bool(result["params"].get(
                        "classification_use_cell_pattern_gate",
                        DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE,
                    ))
                    else np.ones(len(result["picks"].regions), dtype=bool)
                )
                for result in template_results
            ],
            classification_scores,
            match_distance_nm=deduplication_distance_nm,
            deduplication_distance_nm=deduplication_distance_nm,
            minimum_winner_probability=(
                float(params.get(
                    "min_monte_carlo_probability",
                    DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR,
                ))
                if digital_classification
                else None
            ),
        )
        core_bounds = params.get("_candidate_core_bounds_nm")
        if core_bounds is not None:
            core_x0, core_x1, core_y0, core_y1 = (float(value) for value in core_bounds)
            centers = np.asarray(classification.group_centers_nm, dtype=float)
            in_core = (
                (centers[:, 0] >= core_x0)
                & (centers[:, 0] < core_x1)
                & (centers[:, 1] >= core_y0)
                & (centers[:, 1] < core_y1)
            )
            for template_index, assignment_mask in enumerate(classification.assignment_masks):
                candidate_groups = classification.candidate_group_indices[template_index]
                assignment_mask &= in_core[candidate_groups]
            classification.counts = np.asarray(
                [np.count_nonzero(mask) for mask in classification.assignment_masks],
                dtype=int,
            )
            classification.winning_template_indices[~in_core] = -3
            classification.unclassified_count = int(
                np.count_nonzero(classification.winning_template_indices == -1)
            )
        template_names = [str(result["name"]) for result in template_results]
        for template_index, (result, assignment_mask) in enumerate(
            zip(template_results, classification.assignment_masks)
        ):
            dispositions: list[str] = []
            winner_margins: list[float] = []
            runner_up_templates: list[str] = []
            template_probabilities: list[float] = []
            raw_template_probabilities: list[float] = []
            template_probability_vectors: list[tuple[float, ...]] = []
            raw_template_probability_vectors: list[tuple[float, ...]] = []
            for group_index in classification.candidate_group_indices[template_index]:
                winner = int(classification.winning_template_indices[int(group_index)])
                margin = float(classification.winning_score_margins[int(group_index)])
                runner_up = int(classification.runner_up_template_indices[int(group_index)])
                winner_margins.append(margin)
                runner_up_templates.append(
                    template_names[runner_up] if 0 <= runner_up < len(template_names) else ""
                )
                template_probabilities.append(
                    float(classification.template_probabilities[int(group_index), template_index])
                )
                raw_template_probabilities.append(
                    float(classification.raw_template_probabilities[int(group_index), template_index])
                )
                template_probability_vectors.append(
                    tuple(
                        float(value)
                        for value in classification.template_probabilities[int(group_index)]
                    )
                )
                raw_template_probability_vectors.append(
                    tuple(
                        float(value)
                        for value in classification.raw_template_probabilities[int(group_index)]
                    )
                )
                if winner == -2:
                    dispositions.append("suppressed duplicate of a stronger nearby fit")
                elif winner == -3:
                    dispositions.append("tile halo only")
                elif winner < 0:
                    dispositions.append("unclassified: no template passed")
                elif winner == template_index:
                    dispositions.append(f"classified as {template_names[winner]}")
                else:
                    dispositions.append(f"classified as {template_names[winner]}")
            result["params"]["classification_dispositions"] = tuple(dispositions)
            result["params"]["classification_winner_margins"] = tuple(winner_margins)
            result["params"]["classification_runner_up_templates"] = tuple(runner_up_templates)
            result["params"]["classification_template_probabilities"] = tuple(
                template_probabilities
            )
            result["params"]["classification_raw_template_probabilities"] = tuple(
                raw_template_probabilities
            )
            result["params"]["classification_template_probability_vectors"] = tuple(
                template_probability_vectors
            )
            result["params"]["classification_raw_template_probability_vectors"] = tuple(
                raw_template_probability_vectors
            )
            result["picks"] = replace(result["picks"], accepted_mask=assignment_mask)

        unclassified_details: list[dict[str, Any]] = []
        for group_index in np.flatnonzero(classification.winning_template_indices == -1):
            attempts: list[tuple[float, int, int]] = []
            for template_index, group_indices in enumerate(classification.candidate_group_indices):
                candidate_indices = np.flatnonzero(group_indices == group_index)
                for candidate_index in candidate_indices:
                    score = float(classification_scores[template_index][candidate_index])
                    attempts.append((score if np.isfinite(score) else -float("inf"), template_index, int(candidate_index)))
            if not attempts:
                continue
            score, template_index, candidate_index = max(attempts)
            attempted = template_results[template_index]
            attempted_picks: OrigamiPickResult = attempted["picks"]
            attempted_params = dict(attempted["params"])
            failure_reasons = origami_candidate_failure_reasons(
                point_count=int(attempted_picks.point_counts[candidate_index]),
                correlation=float(attempted_picks.rectangle_confidence[candidate_index]),
                supported_sites=int(attempted_picks.supported_site_count[candidate_index]),
                supported_rows=int(attempted_picks.supported_row_count[candidate_index]),
                supported_columns=int(attempted_picks.supported_column_count[candidate_index]),
                spacing_error_nm=float(attempted_picks.site_spacing_max_error_nm[candidate_index]),
                params=attempted_params,
            )
            alignment_mask = np.asarray(
                attempted_params.get("_alignment_accepted_mask", ()), dtype=bool
            )
            if (
                alignment_mask.shape == (len(attempted_picks.regions),)
                and not bool(alignment_mask[candidate_index])
                and not any("corr" in reason or "points" in reason for reason in failure_reasons)
            ):
                failure_reasons.append("alignment pose rejected at Step 2")
            lattice_precisions = attempted_params.get("classification_lattice_precision", ())
            empty_cell_fractions = attempted_params.get("classification_empty_cell_fraction", ())
            pattern_correlations = attempted_params.get("classification_cell_pattern_correlation", ())
            monte_carlo_posteriors = attempted_params.get("classification_monte_carlo_posterior", ())
            minimum_monte_carlo_probability = float(
                params.get(
                    "min_monte_carlo_probability",
                    DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR,
                )
            )
            direct_digital_evidence = attempted_params.get(
                "classification_evidence_model"
            ) == "direct digital groups"
            reported_template_probability = (
                float(classification.raw_template_probabilities[group_index, template_index])
                if direct_digital_evidence
                else (
                    float(monte_carlo_posteriors[candidate_index])
                    if candidate_index < len(monte_carlo_posteriors)
                    else float("nan")
                )
            )
            if (
                np.isfinite(reported_template_probability)
                and reported_template_probability < minimum_monte_carlo_probability
            ):
                evidence_name = (
                    "digital-group template probability"
                    if direct_digital_evidence
                    else "Monte Carlo template probability"
                )
                failure_reasons.append(
                    f"{evidence_name} "
                    f"{reported_template_probability:.2f} < "
                    f"{minimum_monte_carlo_probability:.2f}"
                )
            minimum_pattern_correlation = float(
                attempted_params.get(
                    "classification_min_cell_pattern_correlation",
                    DEFAULT_ORIGAMI_MIN_CELL_PATTERN_CORRELATION,
                )
            )
            if (
                bool(attempted_params.get(
                    "classification_use_cell_pattern_gate",
                    DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE,
                ))
                and candidate_index < len(pattern_correlations)
                and float(pattern_correlations[candidate_index])
                < minimum_pattern_correlation
            ):
                failure_reasons.append(
                    f"cell-pattern correlation "
                    f"{float(pattern_correlations[candidate_index]):.2f} < "
                    f"{minimum_pattern_correlation:.2f}"
                )
            unclassified_details.append(
                {
                    "center_nm": np.asarray(classification.group_centers_nm[group_index], dtype=float),
                    "template_name": template_names[template_index],
                    "point_count": int(attempted_picks.point_counts[candidate_index]),
                    "correlation": float(attempted_picks.rectangle_confidence[candidate_index]),
                    "supported_sites": int(attempted_picks.supported_site_count[candidate_index]),
                    "spacing_error_nm": float(attempted_picks.site_spacing_max_error_nm[candidate_index]),
                    "site_gap_contrast": float(attempted_picks.site_gap_contrast[candidate_index]),
                    "cell_match": (
                        float(lattice_precisions[candidate_index])
                        if candidate_index < len(lattice_precisions)
                        else float("nan")
                    ),
                    "empty_cell_fraction": (
                        float(empty_cell_fractions[candidate_index])
                        if candidate_index < len(empty_cell_fractions)
                        else float("nan")
                    ),
                    "cell_pattern_correlation": (
                        float(pattern_correlations[candidate_index])
                        if candidate_index < len(pattern_correlations)
                        else float("nan")
                    ),
                    "monte_carlo_posterior": (
                        float(monte_carlo_posteriors[candidate_index])
                        if candidate_index < len(monte_carlo_posteriors)
                        else float("nan")
                    ),
                    "evidence_model": str(
                        attempted_params.get("classification_evidence_model", "Monte Carlo")
                    ),
                    "classification_score": float(score),
                    "crop_retained_fraction": float(
                        attempted_picks.crop_retained_fractions[candidate_index]
                    ),
                    "template_probabilities": {
                        name: float(probability)
                        for name, probability in zip(
                            template_names,
                            classification.template_probabilities[int(group_index)],
                        )
                    },
                    "raw_template_probabilities": {
                        name: float(probability)
                        for name, probability in zip(
                            template_names,
                            classification.raw_template_probabilities[int(group_index)],
                        )
                    },
                    "failure_reasons": tuple(failure_reasons),
                }
            )

        emit_progress(100.0, "Template classification complete.")
        return "origami_multi_picks", {
            "templates": template_results,
            "counts": classification.counts,
            "unclassified_count": classification.unclassified_count,
            "suppressed_duplicate_count": classification.suppressed_duplicate_count,
            "unclassified_centers_nm": classification.group_centers_nm[
                classification.winning_template_indices == -1
            ],
            "unclassified_details": unclassified_details,
            "source_path": params["source_path"],
            "params": dict(params),
        }

    def _origami_identification_worker_progress(self, percent: float, message: str) -> None:
        self.worker_queue.put(("origami_identification_progress", (float(percent), str(message))))

    def _set_origami_step_progress(
        self,
        step: int,
        percent: float,
        message: str,
    ) -> None:
        progress = self.__dict__.get("origami_step_progress", {})
        labels = self.__dict__.get("origami_step_progress_text", {})
        if step in progress:
            progress[step].set(max(0.0, min(100.0, float(percent))))
        if step in labels:
            labels[step].set(str(message))

    def _finish_origami_identification_progress(self, message: str | None = None) -> None:
        self.origami_identification_running = False
        self.origami_pending_identification_snapshot = None
        self.origami_identify_button.state(["!disabled"])
        for button_name in (
            "origami_run_candidates_button",
            "origami_run_alignment_button",
            "origami_run_sites_button",
            "origami_run_acceptance_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.state(["!disabled"])
        has_accepted_fit = (
            self.origami_pick_result is not None
            and self.origami_pick_result.accepted_count > 0
        ) or sum(getattr(self, "origami_multi_template_counts", {}).values()) > 0
        has_identification_params = self.origami_identification_params is not None or any(
            payload.get("params") is not None
            for payload in getattr(self, "origami_multi_template_results", {}).values()
        )
        if (
            has_accepted_fit
            and self.origami_loaded_roi_nm is not None
            and has_identification_params
        ):
            self.origami_tiled_button.state(["!disabled"])
            self.origami_n_tiles_button.state(["!disabled"])
            self.origami_random_roi_button.state(["!disabled"])
        else:
            self.origami_tiled_button.state(["disabled"])
            self.origami_n_tiles_button.state(["disabled"])
            self.origami_random_roi_button.state(["disabled"])
        if message is not None:
            self.origami_identification_progress_text.set(message)
            active_step = int(self.__dict__.get("origami_active_step", 0))
            step_labels = self.__dict__.get("origami_step_progress_text", {})
            if active_step in step_labels:
                step_labels[active_step].set(message)
                if not any(word in message.lower() for word in ("error", "stopped", "failed")):
                    self.__dict__["origami_step_progress"][active_step].set(100.0)
        self._refresh_origami_action_states()

    def _validated_origami_tile_context(
        self,
    ) -> tuple[
        pd.DataFrame,
        float,
        list[tuple[float, float, float, float]],
        dict[str, Any],
        dict[str, Any],
    ] | None:
        if self.loaded is None or self.corrected_locs is None:
            messagebox.showinfo("No corrected data", "Load and drift-correct localization data first.")
            return None
        validated_picks = self.origami_pick_result
        validated_params = self.origami_identification_params
        if (validated_picks is None or validated_params is None) and self.origami_multi_template_results:
            for payload in self.origami_multi_template_results.values():
                candidate_picks = payload.get("picks")
                candidate_params = payload.get("params")
                if (
                    candidate_picks is not None
                    and candidate_picks.accepted_count > 0
                    and candidate_params is not None
                ):
                    validated_picks = candidate_picks
                    validated_params = dict(candidate_params)
                    break
        if (
            validated_picks is None
            or validated_picks.accepted_count == 0
            or validated_params is None
            or self.origami_loaded_source_params is None
        ):
            messagebox.showinfo(
                "Validate one ROI first",
                "Load source data from a selected ROI, run Identify Origami, and inspect the accepted footprints first.",
            )
            return None
        if self.origami_loaded_roi_nm is None:
            messagebox.showinfo(
                "ROI source required",
                "The validated source was not loaded from an ROI. Select an ROI, enable Use selected ROI, reload source data, and validate identification.",
            )
            return None

        source_params = dict(self.origami_loaded_source_params)
        source = str(source_params["source"])
        if "linked" in source.lower():
            if self.linked_locs is None:
                messagebox.showinfo("No linked events", "Run whole-image linking before inspecting random origami ROIs.")
                return None
            if self.linked_roi_nm is not None:
                messagebox.showinfo(
                    "Whole-image linking required",
                    "The linked-event cache only covers an ROI. Run Linking Analysis with Whole image, then reload and validate the origami ROI.",
                )
                return None
            source_locs = self.linked_locs
        else:
            source_locs = self.corrected_locs

        pixelsize = float(self.loaded.info[0]["Pixelsize"])
        full_width_nm = float(self.loaded.info[0]["Width"]) * pixelsize
        full_height_nm = float(self.loaded.info[0]["Height"]) * pixelsize
        try:
            tile_lattice = fully_fitting_roi_tiles(full_width_nm, full_height_nm, self.origami_loaded_roi_nm)
        except ValueError as exc:
            messagebox.showerror("Invalid validation ROI", str(exc))
            return None
        if not tile_lattice:
            messagebox.showinfo("No complete tiles", "No ROI-sized tile fits completely inside the image.")
            return None
        return source_locs, pixelsize, tile_lattice, dict(validated_params), source_params

    def inspect_random_origami_roi(self, target_view: str | None = None) -> None:
        context = self._validated_origami_tile_context()
        if context is None:
            return
        source_locs, pixelsize, tile_lattice, identification_params, source_params = context
        validation_roi = tuple(float(value) for value in self.origami_loaded_roi_nm or ())
        validation_indices = {
            index
            for index, tile in enumerate(tile_lattice)
            if len(validation_roi) == 4 and np.allclose(tile, validation_roi, rtol=0.0, atol=1e-9)
        }
        excluded = validation_indices | self.origami_inspected_tile_indices
        random_order = randomized_tile_order(len(tile_lattice), excluded, self.origami_random_generator)
        if not len(random_order):
            self.origami_inspected_tile_indices.clear()
            random_order = randomized_tile_order(len(tile_lattice), validation_indices, self.origami_random_generator)
        if not len(random_order):
            messagebox.showinfo("No alternate ROI", "The image contains no complete ROI-sized tile besides the validation ROI.")
            return

        self.origami_identification_running = True
        self.origami_active_step = 5
        self._set_origami_step_progress(5, 0.0, "Inspecting a random tile…")
        self.origami_pending_roi_view = target_view or "Identified origami template matches"
        self.origami_identification_progress.set(0.0)
        self.origami_identification_progress_text.set("Selecting a random unseen ROI-sized tile...")
        self.origami_identify_button.state(["disabled"])
        self.origami_tiled_button.state(["disabled"])
        self.origami_n_tiles_button.state(["disabled"])
        self.origami_random_roi_button.state(["disabled"])
        self._refresh_origami_action_states()
        self.status.set("Inspect Random ROI: applying the validated source filters and identification settings...")
        self._run_worker(
            lambda: self._random_origami_roi_worker(
                source_locs,
                pixelsize,
                tile_lattice,
                random_order,
                identification_params,
                source_params,
            )
        )

    def _random_origami_roi_worker(
        self,
        source_locs: pd.DataFrame,
        pixelsize: float,
        tile_lattice: list[tuple[float, float, float, float]],
        random_order: np.ndarray,
        identification_params: dict[str, Any],
        source_params: dict[str, Any],
    ) -> tuple[str, Any]:
        selected = source_locs
        source = str(source_params["source"])
        if source.lower().startswith("filtered") and source_params["active_filters"]:
            keep = pd.Series(True, index=selected.index, dtype=bool)
            for mode, (left, right) in source_params["active_filters"]:
                series, _label = localization_series_for_mode(
                    selected,
                    mode,
                    pixelsize,
                    float(source_params["exposure_ms"]),
                    float(source_params["link_radius_nm"]),
                    int(source_params["max_gap_frames"]),
                )
                values = series.to_numpy(dtype=float)
                keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
            selected = selected.loc[keep]
        if selected.empty:
            raise ValueError("The whole-image source contains no points after applying the validated source filters.")

        tile_index = -1
        tile_locs = selected.iloc[0:0]
        for candidate_index in random_order:
            candidate_locs = roi_locs(selected, tile_lattice[int(candidate_index)], pixelsize)
            if not candidate_locs.empty:
                tile_index = int(candidate_index)
                tile_locs = candidate_locs
                break
        if tile_index < 0:
            raise ValueError("None of the remaining random ROI-sized tiles contains source points.")

        roi_nm = tile_lattice[tile_index]
        points_nm = tile_locs[["x", "y"]].to_numpy(dtype=float) * pixelsize
        render_result = render_picasso_map(
            tile_locs,
            self.loaded.info,
            float(source_params["render_pixel_nm"]),
            str(source_params["render_blur_method"]),
            float(source_params["render_min_blur_width"]),
            roi_nm,
        )
        render_result["min_density"] = float(source_params["render_min_density"])
        render_result["max_density"] = float(source_params["render_max_density"])
        picks = identify_origami_regions(
            points_nm,
            pick_bin_size_nm=float(identification_params["pick_bin_size_nm"]),
            connect_distance_nm=float(identification_params["connect_distance_nm"]),
            density_threshold=float(identification_params["density_threshold"]),
            min_candidate_points=int(identification_params["min_candidate_points"]),
            max_candidate_points=int(identification_params["max_candidate_points"]),
            rows=int(identification_params["rows"]),
            columns=int(identification_params["columns"]),
            spacing_x_nm=float(identification_params["spacing_x_nm"]),
            spacing_y_nm=float(identification_params["spacing_y_nm"]),
            rectangle_margin_nm=float(identification_params["rectangle_margin_nm"]),
            min_rectangle_confidence=float(identification_params["min_rectangle_confidence"]),
            use_correlation_gate=bool(identification_params.get("use_correlation_gate", True)),
            site_mask_radius_nm=float(identification_params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
            min_supported_sites=int(identification_params.get("min_supported_sites", 0)),
            min_site_evidence=float(identification_params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
            min_site_localizations=int(identification_params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
            min_supported_rows=int(identification_params.get("min_supported_rows", 0)),
            min_supported_columns=int(identification_params.get("min_supported_columns", 0)),
            max_site_spacing_error_nm=float(identification_params.get("max_site_spacing_error_nm", float("inf"))),
            alignment_pixel_nm=float(identification_params["alignment_pixel_nm"]),
            alignment_max_patch_pixels=int(identification_params.get("alignment_max_patch_pixels", DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS)),
            alignment_iterations=int(identification_params["alignment_iterations"]),
            alignment_template_image=identification_params.get("alignment_template_image"),
            template_pixel_size_x_nm=(
                float(identification_params["template_pixel_size_x_nm"])
                if identification_params.get("template_pixel_size_x_nm") is not None
                else None
            ),
            template_pixel_size_y_nm=(
                float(identification_params["template_pixel_size_y_nm"])
                if identification_params.get("template_pixel_size_y_nm") is not None
                else None
            ),
            progress_callback=self._origami_identification_worker_progress,
        )
        return "origami_random_roi", {
            "points_nm": points_nm,
            "locs": tile_locs,
            "picks": picks,
            "render_result": render_result,
            "roi_nm": roi_nm,
            "tile_index": tile_index,
            "available_tile_count": len(tile_lattice),
            "source": source,
            "source_path": source_params["source_path"],
            "identification_params": dict(identification_params),
        }

    def analyze_tiled_origamis(self, *, use_tile_limit: bool = False) -> None:
        context = self._validated_origami_tile_context()
        if context is None:
            return
        source_locs, pixelsize, tile_lattice, validated_params, source_params = context
        # Tiled analysis supersedes any validation-ROI overlay workers that may
        # still be queued. Give the aggregate run its own generation so a late
        # one-window result cannot replace it after completion.
        self.origami_identification_generation += 1
        validated_params = independent_origami_pipeline_params(validated_params)
        validated_params["identification_generation"] = self.origami_identification_generation
        try:
            if use_tile_limit:
                tile_indices = evenly_distributed_tile_indices(tile_lattice, int(self.origami_tile_count.get()))
                # Keep the already validated ROI in every limited run.  Apart
                # from providing a useful positive control, this makes an
                # N-tile run directly comparable with the Step 4 result the
                # user approved before launching it.
                validation_roi = tuple(
                    float(value)
                    for value in (self.__dict__.get("origami_loaded_roi_nm") or ())
                )
                validation_index = next(
                    (
                        index
                        for index, tile in enumerate(tile_lattice)
                        if len(validation_roi) == 4
                        and np.allclose(tile, validation_roi, rtol=0.0, atol=1e-9)
                    ),
                    None,
                )
                if (
                    validation_index is not None
                    and len(tile_indices)
                    and validation_index not in set(np.asarray(tile_indices, dtype=int))
                ):
                    tile_indices = np.asarray(tile_indices, dtype=int).copy()
                    tile_indices[-1] = int(validation_index)
            else:
                tile_indices = np.arange(len(tile_lattice), dtype=int)
            overlay_params = {
                "rows": int(validated_params["rows"]),
                "columns": int(validated_params["columns"]),
                "spacing_x_nm": float(validated_params["spacing_x_nm"]),
                "spacing_y_nm": float(validated_params["spacing_y_nm"]),
                "g5m_sigma_min_nm": float(self.origami_g5m_sigma_min_nm.get()),
                "g5m_sigma_max_nm": float(self.origami_g5m_sigma_max_nm.get()),
                "g5m_min_locs": int(self.origami_g5m_min_locs.get()),
                "g5m_bic_patience": int(self.origami_g5m_bic_patience.get()),
                "site_radius_nm": float(self.origami_site_radius_nm.get()),
                "direct_site_radius_nm": float(validated_params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
                "direct_min_site_localizations": int(validated_params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
                "direct_min_site_evidence": float(validated_params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
                "allow_mirror": bool(self.origami_allow_mirror.get()),
                "overlay_pixel_nm": float(self.origami_overlay_pixel_nm.get()),
                "overlay_padding_nm": float(self.origami_overlay_padding_nm.get()),
                "overlay_blur_nm": float(self.origami_overlay_blur_nm.get()),
                "source_path": source_params["source_path"],
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid tiled-analysis settings", str(exc))
            return
        if not tile_lattice:
            messagebox.showinfo("No complete tiles", "No ROI-sized tile fits completely inside the image.")
            return
        if (
            overlay_params["site_radius_nm"] <= 0
        ):
            messagebox.showerror("Invalid tiled-analysis settings", "Site radius must be positive.")
            return

        self.origami_identification_running = True
        self.origami_active_step = 5
        self._set_origami_step_progress(5, 0.0, "Analyzing tiles…")
        self.origami_identification_progress.set(0.0)
        tile_width = tile_lattice[0][1] - tile_lattice[0][0]
        tile_height = tile_lattice[0][3] - tile_lattice[0][2]
        selection_note = (
            f"{len(tile_indices):,} spatially distributed of {len(tile_lattice):,} available"
            if use_tile_limit and len(tile_indices) < len(tile_lattice)
            else f"all {len(tile_indices):,} available"
        )
        self.origami_identification_progress_text.set(
            f"Starting {selection_note} complete {tile_width:g} × {tile_height:g} nm tiles..."
        )
        self.origami_identify_button.state(["disabled"])
        self.origami_tiled_button.state(["disabled"])
        self.origami_n_tiles_button.state(["disabled"])
        self.origami_random_roi_button.state(["disabled"])
        self._refresh_origami_action_states()
        self.status.set(f"Tiled origami analysis: preparing {selection_note} ROI tiles...")
        self._run_worker(
            lambda: self._tiled_origami_worker(
                source_locs,
                pixelsize,
                tile_lattice,
                tile_indices,
                validated_params,
                source_params,
                overlay_params,
            )
        )

    def _tiled_origami_worker(
        self,
        source_locs: pd.DataFrame,
        pixelsize: float,
        tile_lattice: list[tuple[float, float, float, float]],
        tile_indices: np.ndarray,
        identification_params: dict[str, Any],
        source_params: dict[str, Any],
        overlay_params: dict[str, Any],
    ) -> tuple[str, Any]:
        selected = source_locs
        source = str(source_params["source"])
        if source.lower().startswith("filtered") and source_params["active_filters"]:
            keep = pd.Series(True, index=selected.index, dtype=bool)
            for mode, (left, right) in source_params["active_filters"]:
                series, _label = localization_series_for_mode(
                    selected,
                    mode,
                    pixelsize,
                    float(source_params["exposure_ms"]),
                    float(source_params["link_radius_nm"]),
                    int(source_params["max_gap_frames"]),
                )
                values = series.to_numpy(dtype=float)
                keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
            selected = selected.loc[keep]
        if selected.empty:
            raise ValueError("The whole-image source contains no points after applying the validated source filters.")

        x_nm = selected["x"].to_numpy(dtype=np.float32, copy=False) * np.float32(pixelsize)
        y_nm = selected["y"].to_numpy(dtype=np.float32, copy=False) * np.float32(pixelsize)
        x_starts = np.asarray(sorted({tile[0] for tile in tile_lattice}), dtype=float)
        y_starts = np.asarray(sorted({tile[2] for tile in tile_lattice}), dtype=float)
        tile_width = float(tile_lattice[0][1] - tile_lattice[0][0])
        tile_height = float(tile_lattice[0][3] - tile_lattice[0][2])
        nx = len(x_starts)
        ny = len(y_starts)
        tile_x = np.floor((x_nm - x_starts[0]) / tile_width).astype(np.int32)
        tile_y = np.floor((y_nm - y_starts[0]) / tile_height).astype(np.int32)
        valid = (
            np.isfinite(x_nm)
            & np.isfinite(y_nm)
            & (tile_x >= 0)
            & (tile_x < nx)
            & (tile_y >= 0)
            & (tile_y < ny)
            & (x_nm < x_starts[-1] + tile_width)
            & (y_nm < y_starts[-1] + tile_height)
        )
        source_indices = np.flatnonzero(valid)
        tile_ids = tile_y[valid] * nx + tile_x[valid]
        order = np.argsort(tile_ids, kind="stable")
        sorted_ids = tile_ids[order]
        sorted_indices = source_indices[order]
        del tile_ids, source_indices, order, tile_x, tile_y, valid

        accepted_regions: list[np.ndarray] = []
        accepted_centers: list[np.ndarray] = []
        tile_pick_results: list[OrigamiPickResult] = []
        multi_tile_payloads: list[dict[str, Any]] = []
        multi_template_mode = len(identification_params.get("custom_templates", ())) > 1
        analyzed_source_indices: list[np.ndarray] = []
        rejected_count = 0
        candidate_count = 0
        analyzed_nonempty_tiles = 0
        selected_tile_count = len(tile_indices)
        available_tile_count = len(tile_lattice)
        for selection_index, lattice_index_value in enumerate(tile_indices):
            lattice_index = int(lattice_index_value)
            left = int(np.searchsorted(sorted_ids, lattice_index, side="left"))
            right = int(np.searchsorted(sorted_ids, lattice_index, side="right"))
            if right <= left:
                self._origami_identification_worker_progress(
                    80.0 * (selection_index + 1) / selected_tile_count,
                    f"Selected tile {selection_index + 1}/{selected_tile_count} "
                    f"(lattice tile {lattice_index + 1}/{available_tile_count}): empty; skipped.",
                )
                continue
            core_indices = sorted_indices[left:right]
            core_x0, core_x1, core_y0, core_y1 = tile_lattice[lattice_index]
            # Run the same spatial input through Steps 1-4 that a validation
            # ROI uses.  The former footprint-sized halo was applied *before*
            # coarse connected-component detection, so signal or noise in an
            # adjacent tile could merge with an otherwise valid candidate and
            # make tiled classification disagree with Step 4.
            tile_points = np.column_stack((x_nm[core_indices], y_nm[core_indices])).astype(
                float, copy=False
            )
            analyzed_nonempty_tiles += 1
            analyzed_source_indices.append(core_indices)

            def tile_progress(
                percent: float,
                message: str,
                current: int = selection_index,
                lattice_tile: int = lattice_index,
            ) -> None:
                overall = 80.0 * (current + float(percent) / 100.0) / selected_tile_count
                self._origami_identification_worker_progress(
                    overall,
                    f"Selected tile {current + 1}/{selected_tile_count} "
                    f"(lattice tile {lattice_tile + 1}/{available_tile_count}): {message}",
                )

            if multi_template_mode:
                # Validation-ROI caches contain coordinates and fitted poses
                # from a different window. Every tile must build its own Step
                # 1 candidates and then run Steps 2–4 from those candidates.
                tile_identification_params = independent_origami_pipeline_params(
                    identification_params
                )
                tile_identification_params["_candidate_core_bounds_nm"] = (
                    core_x0,
                    core_x1,
                    core_y0,
                    core_y1,
                )
                result_kind, tile_payload = self._identify_origami_worker(
                    tile_points,
                    tile_identification_params,
                    tile_progress,
                )
                if result_kind != "origami_multi_picks":
                    raise RuntimeError("Multi-template tiled analysis returned an unexpected result.")
                multi_tile_payloads.append(tile_payload)
                tile_unique_count = (
                    int(np.sum(np.asarray(tile_payload["counts"], dtype=int)))
                    + int(tile_payload["unclassified_count"])
                    + int(tile_payload.get("suppressed_duplicate_count", 0))
                )
                candidate_count += tile_unique_count
                if selection_index % 10 == 0:
                    gc.collect()
                continue

            picks = identify_origami_regions(
                tile_points,
                pick_bin_size_nm=float(identification_params["pick_bin_size_nm"]),
                connect_distance_nm=float(identification_params["connect_distance_nm"]),
                density_threshold=float(identification_params["density_threshold"]),
                min_candidate_points=int(identification_params["min_candidate_points"]),
                max_candidate_points=int(identification_params["max_candidate_points"]),
                rows=int(identification_params["rows"]),
                columns=int(identification_params["columns"]),
                spacing_x_nm=float(identification_params["spacing_x_nm"]),
                spacing_y_nm=float(identification_params["spacing_y_nm"]),
                rectangle_margin_nm=float(identification_params["rectangle_margin_nm"]),
                min_rectangle_confidence=float(identification_params["min_rectangle_confidence"]),
                use_correlation_gate=bool(identification_params.get("use_correlation_gate", True)),
                site_mask_radius_nm=float(identification_params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
                min_supported_sites=int(identification_params.get("min_supported_sites", 0)),
                min_site_evidence=float(identification_params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
                min_site_localizations=int(identification_params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
                min_supported_rows=int(identification_params.get("min_supported_rows", 0)),
                min_supported_columns=int(identification_params.get("min_supported_columns", 0)),
                max_site_spacing_error_nm=float(identification_params.get("max_site_spacing_error_nm", float("inf"))),
                alignment_pixel_nm=float(identification_params["alignment_pixel_nm"]),
                alignment_max_patch_pixels=int(identification_params.get("alignment_max_patch_pixels", DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS)),
                alignment_iterations=int(identification_params["alignment_iterations"]),
                alignment_template_image=identification_params.get("alignment_template_image"),
                template_pixel_size_x_nm=(
                    float(identification_params["template_pixel_size_x_nm"])
                    if identification_params.get("template_pixel_size_x_nm") is not None
                    else None
                ),
                template_pixel_size_y_nm=(
                    float(identification_params["template_pixel_size_y_nm"])
                    if identification_params.get("template_pixel_size_y_nm") is not None
                    else None
                ),
                progress_callback=tile_progress,
            )
            candidate_centers = np.asarray(
                [np.median(region, axis=0) for region in picks.regions],
                dtype=float,
            ).reshape(-1, 2)
            in_core = (
                (candidate_centers[:, 0] >= core_x0)
                & (candidate_centers[:, 0] < core_x1)
                & (candidate_centers[:, 1] >= core_y0)
                & (candidate_centers[:, 1] < core_y1)
            )
            picks.accepted_mask = np.asarray(picks.accepted_mask, dtype=bool) & in_core
            candidate_count += int(np.count_nonzero(in_core))
            tile_pick_results.append(picks)
            rejected_count += int(np.count_nonzero(in_core)) - picks.accepted_count
            accepted_regions.extend(region.copy() for region in picks.accepted_aligned_regions)
            accepted_centers.extend(
                np.median(region, axis=0)
                for region, accepted in zip(picks.regions, picks.accepted_mask)
                if bool(accepted)
            )
            if selection_index % 10 == 0:
                gc.collect()

        if multi_template_mode:
            if not multi_tile_payloads:
                raise ValueError("No nonempty selected tiles were available for template classification.")
            template_names = [
                str(item["name"]) for item in multi_tile_payloads[0]["templates"]
            ]
            combined_templates: list[dict[str, Any]] = []
            sequence_param_names = (
                "classification_lattice_precision",
                "classification_empty_cell_fraction",
                "classification_bright_cell_probability",
                "classification_cell_pattern_correlation",
                "classification_cell_agreement",
                "classification_monte_carlo_posterior",
                "classification_monte_carlo_log_bayes_factor",
                "classification_scores",
                "classification_dispositions",
                "classification_winner_margins",
                "classification_runner_up_templates",
                "classification_template_probabilities",
                "classification_raw_template_probabilities",
                "classification_template_probability_vectors",
                "classification_raw_template_probability_vectors",
                "classification_crop_retained_fraction",
                "classification_logical_bit_probabilities",
                "digital_group_localization_evidence",
                "digital_group_prominences",
                "digital_pixel_probabilities",
            )
            for template_index, name in enumerate(template_names):
                tile_items = [payload["templates"][template_index] for payload in multi_tile_payloads]
                if any(str(item["name"]) != name for item in tile_items):
                    raise RuntimeError("Template order changed while aggregating tiled classifications.")
                combined_params = dict(tile_items[0]["params"])
                for parameter_name in sequence_param_names:
                    combined_params[parameter_name] = tuple(
                        value
                        for item in tile_items
                        for value in item["params"].get(parameter_name, ())
                    )
                combined_templates.append(
                    {
                        "name": name,
                        "picks": concatenate_origami_pick_results(
                            [item["picks"] for item in tile_items]
                        ),
                        "params": combined_params,
                    }
                )
            combined_counts = np.sum(
                np.stack(
                    [np.asarray(payload["counts"], dtype=int) for payload in multi_tile_payloads],
                    axis=0,
                ),
                axis=0,
            )
            assigned_count = int(np.sum(combined_counts))
            combined_indices = np.concatenate(analyzed_source_indices)
            combined_locs = selected.iloc[combined_indices].copy()
            combined_points_nm = combined_locs[["x", "y"]].to_numpy(dtype=float) * pixelsize
            limited_selection = selected_tile_count < available_tile_count
            selection_description = (
                f"{selected_tile_count} spatially distributed of {available_tile_count} available ROI tiles"
                if limited_selection
                else f"all {selected_tile_count} available ROI tiles"
            )
            unclassified_centers = [
                np.asarray(payload.get("unclassified_centers_nm", np.empty((0, 2))), dtype=float).reshape(-1, 2)
                for payload in multi_tile_payloads
            ]
            combined_unclassified_details = [
                dict(detail)
                for payload in multi_tile_payloads
                for detail in payload.get("unclassified_details", ())
            ]
            self._origami_identification_worker_progress(100.0, "Tiled template classification complete.")
            return "origami_multi_picks", {
                "templates": combined_templates,
                "counts": combined_counts,
                "unclassified_count": sum(
                    int(payload["unclassified_count"]) for payload in multi_tile_payloads
                ),
                "suppressed_duplicate_count": sum(
                    int(payload.get("suppressed_duplicate_count", 0)) for payload in multi_tile_payloads
                ),
                "unclassified_centers_nm": (
                    np.concatenate(unclassified_centers, axis=0)
                    if unclassified_centers
                    else np.empty((0, 2), dtype=float)
                ),
                "unclassified_details": combined_unclassified_details,
                "rejection_summary": summarize_origami_rejection_reasons(
                    combined_unclassified_details
                ),
                "source_path": overlay_params["source_path"],
                "params": dict(identification_params),
                "points_nm": combined_points_nm,
                "locs": combined_locs,
                "source": f"{source} ({selection_description})",
                "tile_count": selected_tile_count,
                "available_tile_count": available_tile_count,
                "nonempty_tile_count": analyzed_nonempty_tiles,
                "candidate_count": candidate_count,
                "accepted_count": assigned_count,
                "tile_size_nm": (tile_width, tile_height),
            }

        if not accepted_regions:
            raise ValueError(
                f"No origamis were accepted across {selected_tile_count} selected complete tiles. "
                "Revisit the validation ROI and identification thresholds."
            )
        self._origami_identification_worker_progress(
            82.0,
            f"Identified {len(accepted_regions):,}/{candidate_count:,} candidates; building fast overlay...",
        )

        def alignment_progress(message: str) -> None:
            self._origami_identification_worker_progress(90.0, message)

        result = align_picked_origamis(
            accepted_regions,
            rows=int(overlay_params["rows"]),
            columns=int(overlay_params["columns"]),
            spacing_x_nm=float(overlay_params["spacing_x_nm"]),
            spacing_y_nm=float(overlay_params["spacing_y_nm"]),
            site_radius_nm=float(overlay_params.get("direct_site_radius_nm", overlay_params["site_radius_nm"])),
            g5m_sigma_min_nm=float(overlay_params["g5m_sigma_min_nm"]),
            g5m_sigma_max_nm=float(overlay_params["g5m_sigma_max_nm"]),
            g5m_min_locs=int(overlay_params["g5m_min_locs"]),
            g5m_max_rounds_without_best_bic=int(overlay_params["g5m_bic_patience"]),
            prealigned=True,
            source_centers_nm=np.asarray(accepted_centers, dtype=float),
            allow_mirror=bool(overlay_params["allow_mirror"]),
            initially_rejected_count=rejected_count,
            use_g5m=False,
            direct_min_site_localizations=int(overlay_params.get("direct_min_site_localizations", 1)),
            direct_min_site_evidence=float(overlay_params.get("direct_min_site_evidence", 0.0)),
            grid_points_nm=(
                picks.template_points_nm
                if identification_params.get("alignment_template_image") is not None
                else None
            ),
            symmetrize_180=identification_params.get("alignment_template_image") is None,
            progress_callback=alignment_progress,
        )
        combined_picks = concatenate_origami_pick_results(tile_pick_results)
        combined_indices = np.concatenate(analyzed_source_indices)
        combined_locs = selected.iloc[combined_indices].copy()
        combined_points_nm = combined_locs[["x", "y"]].to_numpy(dtype=float) * pixelsize
        limited_selection = selected_tile_count < available_tile_count
        selection_description = (
            f"{selected_tile_count} spatially distributed of {available_tile_count} available ROI tiles"
            if limited_selection
            else f"all {selected_tile_count} available ROI tiles"
        )
        self._origami_identification_worker_progress(100.0, "Tiled origami analysis complete.")
        return "origami_tiled", {
            "result": result,
            "source": f"{source} ({selection_description})",
            "source_note": selection_description,
            "source_count": int(len(combined_locs)),
            "occupancy_threshold": 1,
            "render_settings": {
                "rows": int(overlay_params["rows"]),
                "columns": int(overlay_params["columns"]),
                "spacing_x_nm": float(overlay_params["spacing_x_nm"]),
                "spacing_y_nm": float(overlay_params["spacing_y_nm"]),
                "pixel_size_nm": float(overlay_params["overlay_pixel_nm"]),
                "padding_nm": float(overlay_params["overlay_padding_nm"]),
                "blur_nm": float(overlay_params["overlay_blur_nm"]),
            },
            "source_path": overlay_params["source_path"],
            "tile_count": selected_tile_count,
            "available_tile_count": available_tile_count,
            "tile_selection": "spatially distributed subset" if limited_selection else "all complete tiles",
            "nonempty_tile_count": analyzed_nonempty_tiles,
            "candidate_count": candidate_count,
            "accepted_count": len(accepted_regions),
            "tile_size_nm": (tile_width, tile_height),
            "points_nm": combined_points_nm,
            "locs": combined_locs,
            "picks": combined_picks,
            "identification_params": dict(identification_params),
        }

    def _apply_tiled_origami_result(self, payload: dict[str, Any]) -> None:
        """Install one aggregate tiled result without retaining ROI-only views."""
        self.origami_source_points_nm = np.asarray(payload["points_nm"], dtype=float)
        self.origami_source_locs = payload["locs"]
        self.origami_source_render_result = None
        self.origami_pick_result = payload["picks"]
        self.origami_identification_params = dict(payload["identification_params"])
        self.origami_loaded_source_label = str(payload["source"])
        self.origami_roi_history_position = -1

        # Multi-template caches here describe the validation ROI, not this
        # aggregate run. Leaving them active routes the combined spatial view
        # and type plots back to that single window.
        self.origami_multi_template_results = {}
        self.origami_multi_template_counts = {}
        self.origami_multi_template_overlay_results = {}
        self.origami_multi_template_overlays_building = set()
        self.origami_multi_template_unclassified_count = 0
        self.origami_multi_template_unclassified_centers_nm = np.empty((0, 2), dtype=float)
        self.origami_multi_template_unclassified_details = []
        self.origami_template_result_view.set("All templates")
        self.origami_template_result_combo.configure(values=("All templates",))
        self.origami_template_result_combo.state(["disabled"])

        self.origami_plot_option.set("Aligned density")
        self._plot_origami_analysis(payload)
        self._refresh_origami_action_states()

    def overlay_origamis(self) -> None:
        picks = self.origami_pick_result
        if picks is None:
            messagebox.showinfo("Origami not identified", "Click Identify Origami and inspect the colored outlines first.")
            return
        if picks.accepted_count == 0:
            messagebox.showinfo(
                "No accepted origami",
                "No candidates pass the point, site-prominence, grid-coverage, and spacing limits. Adjust the settings and run Identify Origami again.",
            )
            return
        identification_params = dict(self.origami_identification_params or {})
        try:
            params = {
                "rows": int(identification_params.get("rows", self.origami_rows.get())),
                "columns": int(identification_params.get("columns", self.origami_columns.get())),
                "spacing_x_nm": float(
                    identification_params.get("spacing_x_nm", self.origami_spacing_x_nm.get())
                ),
                "spacing_y_nm": float(
                    identification_params.get("spacing_y_nm", self.origami_spacing_y_nm.get())
                ),
                "g5m_sigma_min_nm": float(self.origami_g5m_sigma_min_nm.get()),
                "g5m_sigma_max_nm": float(self.origami_g5m_sigma_max_nm.get()),
                "g5m_min_locs": int(self.origami_g5m_min_locs.get()),
                "g5m_bic_patience": int(self.origami_g5m_bic_patience.get()),
                "site_radius_nm": float(self.origami_site_radius_nm.get()),
                "direct_site_radius_nm": float(identification_params.get("site_mask_radius_nm", DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM)),
                "direct_min_site_localizations": int(identification_params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
                "direct_min_site_evidence": float(identification_params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
                "template_points_nm": picks.template_points_nm.copy(),
                "symmetrize_180": identification_params.get("alignment_template_image") is None,
                "occupancy_threshold": 1,
                "allow_mirror": bool(self.origami_allow_mirror.get()),
                "overlay_pixel_nm": float(self.origami_overlay_pixel_nm.get()),
                "overlay_padding_nm": float(self.origami_overlay_padding_nm.get()),
                "overlay_blur_nm": float(self.origami_overlay_blur_nm.get()),
                "source_path": self.origami_loaded_source_path,
                "source_label": (
                    f"{self.origami_loaded_source_label} — {self.origami_template_result_view.get()}"
                    if self.origami_template_result_view.get() in self.origami_multi_template_results
                    else self.origami_loaded_source_label
                ),
                "template_name": (
                    self.origami_template_result_view.get()
                    if self.origami_template_result_view.get() in self.origami_multi_template_results
                    else None
                ),
                "source_count": len(self.origami_source_points_nm) if self.origami_source_points_nm is not None else 0,
                "identification_generation": identification_params.get("identification_generation"),
            }
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid overlay settings", str(exc))
            return
        if params["site_radius_nm"] <= 0:
            messagebox.showerror(
                "Invalid overlay settings",
                "Site-match radius must be positive.",
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
        self.status.set(f"Building fast overlay for {len(accepted_regions)} image-aligned origamis without G5M...")
        self._run_worker(lambda: self._overlay_origami_worker(accepted_regions, accepted_centers, rejected_count, params, False))

    def refine_origami_overlay_with_g5m(self) -> None:
        current = self.origami_result
        if current is None or current.origami_count == 0:
            messagebox.showinfo("No fast overlay", "Build the fast overlay first, then refine it with Picasso G5M if needed.")
            return
        try:
            params = {
                "rows": current.rows,
                "columns": current.columns,
                "spacing_x_nm": float(self.origami_result_render_settings["spacing_x_nm"]),
                "spacing_y_nm": float(self.origami_result_render_settings["spacing_y_nm"]),
                "g5m_sigma_min_nm": float(self.origami_g5m_sigma_min_nm.get()),
                "g5m_sigma_max_nm": float(self.origami_g5m_sigma_max_nm.get()),
                "g5m_min_locs": int(self.origami_g5m_min_locs.get()),
                "g5m_bic_patience": int(self.origami_g5m_bic_patience.get()),
                "site_radius_nm": float(self.origami_site_radius_nm.get()),
                "template_points_nm": current.grid_points_nm.copy(),
                "symmetrize_180": current.symmetrized_180,
                "occupancy_threshold": self.origami_result_occupancy_threshold,
                "allow_mirror": False,
                "overlay_pixel_nm": float(self.origami_result_render_settings["pixel_size_nm"]),
                "overlay_padding_nm": float(self.origami_result_render_settings["padding_nm"]),
                "overlay_blur_nm": float(self.origami_result_render_settings["blur_nm"]),
                "source_path": self.loaded.path if self.loaded is not None else None,
                "source_label": self.origami_result_source,
                "source_count": self.origami_result_source_count,
            }
        except (tk.TclError, TypeError, ValueError) as exc:
            messagebox.showerror("Invalid G5M settings", str(exc))
            return
        if (
            params["g5m_sigma_min_nm"] <= 0
            or params["g5m_sigma_max_nm"] < params["g5m_sigma_min_nm"]
            or params["g5m_min_locs"] < 1
            or params["g5m_bic_patience"] < 1
            or params["site_radius_nm"] <= 0
        ):
            messagebox.showerror("Invalid G5M settings", "Check the G5M sigma, localization, patience, and site-radius values.")
            return
        regions = [points.copy() for points in current.aligned_points]
        self.status.set(f"Refining {len(regions):,} aligned origamis with Picasso G5M...")
        self._run_worker(
            lambda: self._overlay_origami_worker(
                regions,
                current.centers_nm.copy(),
                current.rejected_candidate_count,
                params,
                True,
            )
        )

    def _overlay_origami_worker(
        self,
        accepted_regions: list[np.ndarray],
        accepted_centers: np.ndarray,
        rejected_count: int,
        params: dict[str, Any],
        use_g5m: bool,
    ) -> tuple[str, Any]:
        assignment_radius_nm = float(
            params["site_radius_nm"]
            if use_g5m
            else params.get("direct_site_radius_nm", params["site_radius_nm"])
        )
        result = align_picked_origamis(
            accepted_regions,
            rows=int(params["rows"]),
            columns=int(params["columns"]),
            spacing_x_nm=float(params["spacing_x_nm"]),
            spacing_y_nm=float(params["spacing_y_nm"]),
            site_radius_nm=assignment_radius_nm,
            g5m_sigma_min_nm=float(params["g5m_sigma_min_nm"]),
            g5m_sigma_max_nm=float(params["g5m_sigma_max_nm"]),
            g5m_min_locs=int(params["g5m_min_locs"]),
            g5m_max_rounds_without_best_bic=int(params["g5m_bic_patience"]),
            prealigned=True,
            source_centers_nm=accepted_centers,
            allow_mirror=bool(params["allow_mirror"]),
            initially_rejected_count=rejected_count,
            use_g5m=use_g5m,
            direct_min_site_localizations=int(params.get("direct_min_site_localizations", 1)),
            direct_min_site_evidence=float(params.get("direct_min_site_evidence", 0.0)),
            grid_points_nm=params.get("template_points_nm"),
            symmetrize_180=bool(params.get("symmetrize_180", True)),
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
            "template_name": params.get("template_name"),
            "identification_generation": params.get("identification_generation"),
        }

    def _correction_worker(self) -> tuple[str, Any]:
        assert self.loaded is not None
        cache_settings = self._current_drift_cache_settings()
        corrected_locs, drift, label = apply_drift_correction(
            self.loaded.locs,
            self.loaded.info,
            self.drift_method.get(),
            int(self.drift_segmentation.get()),
            float(self.aim_intersect_nm.get()),
            float(self.aim_roi_nm.get()),
            self._worker_status,
            float(self.rcc_lattice_pitch_nm.get()),
            self.drift_file_path,
        )
        cache_path = save_development_session_cache(
            self.loaded,
            corrected_locs,
            drift,
            label,
            cache_settings,
        )
        return "correction", {
            "locs": corrected_locs,
            "drift": drift,
            "label": label,
            "source_path": self.loaded.path,
            "cache_path": cache_path,
        }

    def _render_current_corrected_worker(self, disp_px_size_nm: float | None = None) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.corrected_locs is not None
        map_result = render_picasso_map(
            self.corrected_locs,
            self.loaded.info,
            float(disp_px_size_nm if disp_px_size_nm is not None else self.render_disp_px_nm.get()),
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
            "filtered_locs": filtered_locs,
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

    def _link_color_worker(self, disp_px_size_nm: float | None = None) -> tuple[str, Any]:
        assert self.loaded is not None
        assert self.linked_locs is not None
        linked_locs = self.linked_locs
        self._worker_status(f"Rendering linked map from cached {len(linked_locs):,} collapsed events...")
        map_result = render_picasso_map(
            linked_locs,
            self.loaded.info,
            float(disp_px_size_nm if disp_px_size_nm is not None else self.render_disp_px_nm.get()),
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
                elif kind == "load_progress":
                    percent, message = payload
                    self.load_progress_value.set(max(0.0, min(100.0, float(percent))))
                    self.load_progress_text.set(str(message))
                    self.status.set(str(message))
                elif kind == "origami_identification_progress":
                    percent, message = payload
                    self.origami_identification_progress.set(max(0.0, min(100.0, float(percent))))
                    self.origami_identification_progress_text.set(str(message))
                    if self.origami_active_step in self.origami_step_progress:
                        self.origami_step_progress[self.origami_active_step].set(
                            max(0.0, min(100.0, float(percent)))
                        )
                        self.origami_step_progress_text[self.origami_active_step].set(
                            str(message)
                        )
                    self.status.set(f"Identify Origami: {float(percent):.0f}% — {message}")
                elif kind == "error":
                    exc, details = payload
                    if self.origami_active_step in self.origami_step_progress_text:
                        self.origami_step_progress_text[self.origami_active_step].set(
                            f"Stopped: {exc}"
                        )
                    self.origami_single_overlay_building = False
                    getattr(self, "origami_multi_template_overlays_building", set()).clear()
                    if self.load_progress_bar.winfo_ismapped():
                        self.session_load_in_progress = False
                        self._hide_load_progress()
                    if self.origami_identification_running:
                        self._finish_origami_identification_progress("Identification stopped because of an error.")
                    if hasattr(self, "origami_run_candidates_button"):
                        self.origami_run_candidates_button.state(["!disabled"])
                    self.status.set("Error")
                    self._show_error_indicator(str(exc), str(details))
                    messagebox.showerror("Analysis error", f"{exc}\n\n{details}")
                elif kind == "result":
                    try:
                        result_kind, result_payload = payload
                        if result_kind == "cached_session":
                            self.session_load_in_progress = False
                            cached_loaded = result_payload["loaded"]
                            # The corrected map is the useful restored view.
                            # Rendering the full raw and corrected maps at the
                            # same time doubles peak CPU and memory pressure.
                            self._after_load(cached_loaded, render_raw=False)
                            self.development_session_cache_path = result_payload.get("cache_path")
                            cached_settings = dict(result_payload.get("cache_settings", {}))
                            cached_drift_file = cached_settings.get("drift_file")
                            if cached_drift_file is not None:
                                self.drift_file_path = Path(cached_drift_file["path"])
                                self.drift_file_label.set(self.drift_file_path.name)
                            self.load_progress_value.set(100.0)
                            self.load_progress_text.set(
                                f"Restored {len(cached_loaded.locs):,} localizations and drift correction"
                            )
                            self.load_progress_hide_id = self.after(1200, self._hide_load_progress)
                            self.worker_queue.put(("result", ("correction", result_payload)))
                        elif result_kind == "loaded":
                            self.session_load_in_progress = False
                            self._after_load(result_payload)
                            self.load_progress_value.set(100.0)
                            self.load_progress_text.set(f"Loaded {len(result_payload.locs):,} localizations")
                            self.load_progress_hide_id = self.after(1200, self._hide_load_progress)
                        elif result_kind == "correction":
                            if self.loaded is None or result_payload.get("source_path") != self.loaded.path:
                                continue
                            self.corrected_locs = result_payload["locs"]
                            self.linked_locs = None
                            self.filtered_map_locs = None
                            self.filtered_map_render_context = {}
                            self.linked_source_count = 0
                            self.linked_roi_nm = None
                            self.linked_params = None
                            self.linked_source_name = self.linking_source.get()
                            self.linked_scope_name = self.linking_scope.get()
                            self.origami_result = None
                            self.origami_result_source = ""
                            self.origami_source_points_nm = None
                            self.origami_source_locs = None
                            self.origami_source_render_result = None
                            self.origami_pick_result = None
                            self.origami_loaded_source_label = ""
                            self.origami_loaded_source_path = None
                            self.origami_loaded_roi_nm = None
                            self.origami_loaded_source_params = None
                            self.origami_identification_params = None
                            self.origami_tiled_button.state(["disabled"])
                            self.origami_n_tiles_button.state(["disabled"])
                            self.origami_random_roi_button.state(["disabled"])
                            self.origami_inspected_tile_indices.clear()
                            self.origami_roi_history.clear()
                            self.origami_roi_history_position = -1
                            self.origami_random_inspection_payload = None
                            self.origami_identification_baseline = None
                            self._on_origami_identification_setting_changed()
                            self._refresh_origami_action_states()
                            self.drift = result_payload["drift"]
                            self.correction_label = result_payload["label"]
                            if result_payload.get("cache_path") is not None:
                                self.development_session_cache_path = Path(result_payload["cache_path"])
                            cache_note = "Restored from development cache. " if result_payload.get("restored_from_cache") else ""
                            cached_map = result_payload.get("cached_map")
                            if cached_map is not None:
                                self.render_disp_px_nm.set(float(cached_map["disp_px_size_nm"]))
                                self.render_blur_method.set(str(cached_map["blur_method"]))
                                self.min_blur_width.set(float(cached_map.get("min_blur_width", 1.0)))
                                self.status.set(f"{cache_note}{self.correction_label} ready. Displaying cached map...")
                                self.after_idle(lambda result=cached_map: self._plot_map(result))
                            else:
                                self.status.set(
                                    f"{cache_note}{self.correction_label} ready. Rendering map with current render settings..."
                                )
                                if result_payload.get("restored_from_cache"):
                                    self.after_idle(
                                        lambda: self.show_current_map(manual_pixel_override=False)
                                    )
                                else:
                                    self.show_current_map(manual_pixel_override=False)
                        elif result_kind == "dynamic_map":
                            self.dynamic_render_running = False
                            request_id = int(result_payload.get("dynamic_request_id", -1))
                            is_current = (
                                self.loaded is not None
                                and result_payload.get("source_path") == self.loaded.path
                                and request_id == self.dynamic_render_request_id
                                and int(result_payload.get("dynamic_tab_index", -1)) == self.active_notebook_tab
                                and bool(self.dynamic_zoom_render.get())
                            )
                            if is_current and result_payload.get("dynamic_error"):
                                details = str(result_payload.get("dynamic_error_details", ""))
                                self.status.set(f"Dynamic zoom render failed: {result_payload['dynamic_error']}")
                                self._show_error_indicator(str(result_payload["dynamic_error"]), details)
                            elif is_current:
                                target_kind = result_payload.get("dynamic_target_kind")
                                if target_kind == "raw_map":
                                    self._plot_raw_map(result_payload)
                                elif target_kind == "map_only":
                                    self._plot_map(result_payload)
                                elif target_kind == "link_map":
                                    self._plot_link_map(result_payload)
                                elif target_kind == "filtered_map":
                                    self._plot_filtered_maps(result_payload)
                            if self.dynamic_render_pending:
                                self.dynamic_render_pending = False
                                self._schedule_dynamic_map_render()
                        elif result_kind == "origami_dynamic_render":
                            self.origami_zoom_render_running = False
                            is_current = (
                                self.loaded is not None
                                and result_payload.get("source_path") == self.loaded.path
                                and int(result_payload.get("request_id", -1)) == self.origami_zoom_render_request_id
                                and int(result_payload.get("roi_position", -999)) == self.origami_roi_history_position
                                and self.notebook.index(self.notebook.select()) == ORIGAMI_TAB
                                and self.origami_last_rendered_plot_option
                                in {"Loaded source data", "Identified origami template matches", "Random ROI inspection"}
                            )
                            if is_current and result_payload.get("dynamic_error"):
                                self.status.set(f"Origami zoom render failed: {result_payload['dynamic_error']}")
                                self._show_error_indicator(
                                    str(result_payload["dynamic_error"]),
                                    str(result_payload.get("dynamic_error_details", "")),
                                )
                            elif is_current:
                                self._apply_origami_dynamic_render(result_payload)
                            if self.origami_zoom_render_pending:
                                self.origami_zoom_render_pending = False
                                self._schedule_origami_zoom_render()
                        elif result_kind == "raw_map":
                            self._plot_raw_map(result_payload)
                        elif result_kind == "map_only":
                            self._plot_map(result_payload)
                            if (
                                result_payload.get("viewport_nm") is None
                                and self.development_session_cache_path is not None
                            ):
                                cache_path = self.development_session_cache_path
                                cached_result = dict(result_payload)
                                self._run_worker(
                                    lambda: (
                                        "development_map_cached",
                                        save_development_map_cache(cache_path, cached_result),
                                    )
                                )
                        elif result_kind == "development_map_cached":
                            pass
                        elif result_kind == "filtered_maps":
                            self._plot_filtered_maps(result_payload)
                        elif result_kind == "hist":
                            self._plot_histogram(result_payload)
                        elif result_kind == "link_summary":
                            self.linked_locs = result_payload.get("linked_locs")
                            self.filtered_map_locs = None
                            self.filtered_map_render_context = {}
                            self.linked_source_count = int(result_payload.get("selected_count", 0))
                            self.linked_roi_nm = result_payload.get("roi_nm")
                            self.linked_params = result_payload.get("linked_params")
                            self.linked_source_name = str(result_payload.get("link_source", self.linking_source.get()))
                            self.linked_scope_name = str(result_payload.get("link_scope", self.linking_scope.get()))
                            self.origami_result = None
                            self.origami_result_source = ""
                            self.origami_source_points_nm = None
                            self.origami_source_locs = None
                            self.origami_source_render_result = None
                            self.origami_pick_result = None
                            self.origami_loaded_source_label = ""
                            self.origami_loaded_source_path = None
                            self.origami_loaded_roi_nm = None
                            self.origami_loaded_source_params = None
                            self.origami_identification_params = None
                            self.origami_tiled_button.state(["disabled"])
                            self.origami_n_tiles_button.state(["disabled"])
                            self.origami_random_roi_button.state(["disabled"])
                            self.origami_inspected_tile_indices.clear()
                            self.origami_roi_history.clear()
                            self.origami_roi_history_position = -1
                            self.origami_random_inspection_payload = None
                            self.origami_identification_baseline = None
                            self._on_origami_identification_setting_changed()
                            self._refresh_origami_action_states()
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
                                self.origami_identification_generation += 1
                                self.origami_source_points_nm = result_payload["points_nm"]
                                self.origami_source_candidate_fingerprint = origami_source_fingerprint(
                                    self.origami_source_points_nm
                                )
                                self.origami_source_locs = result_payload["locs"]
                                self.origami_source_render_result = dict(result_payload["render_result"])
                                self.origami_loaded_source_label = str(result_payload["source_label"])
                                self.origami_loaded_source_path = result_payload["source_path"]
                                self.origami_loaded_roi_nm = result_payload.get("source_roi_nm")
                                self.origami_loaded_source_params = dict(result_payload["source_params"])
                                self.origami_identification_params = None
                                self.origami_active_step = 0
                                for step in range(1, 6):
                                    self.origami_step_progress[step].set(0.0)
                                    self.origami_step_progress_text[step].set("Not run")
                                self.origami_pick_result = None
                                self.origami_staged_candidates = None
                                self.origami_staged_candidate_signature = None
                                self.origami_staged_status.set(
                                    "Run Step 1, then continue through alignment, digital pixels, classification, and optional tiling."
                                )
                                self.origami_result = None
                                self.origami_multi_template_results = {}
                                self.origami_multi_template_counts = {}
                                self.origami_multi_template_overlay_results = {}
                                self.origami_multi_template_overlays_building = set()
                                self.origami_single_overlay_building = False
                                self.origami_multi_template_unclassified_count = 0
                                self.origami_multi_template_unclassified_centers_nm = np.empty((0, 2), dtype=float)
                                self.origami_multi_template_unclassified_details = []
                                self.origami_template_result_view.set("All templates")
                                self.origami_template_result_combo.state(["disabled"])
                                self.origami_match_roi.set(1)
                                self.origami_match_roi_label.set("Candidate –/–")
                                self.origami_tiled_button.state(["disabled"])
                                self.origami_n_tiles_button.state(["disabled"])
                                self.origami_random_roi_button.state(["disabled"])
                                self.origami_inspected_tile_indices.clear()
                                self.origami_roi_history.clear()
                                self.origami_roi_history_position = -1
                                self.origami_random_inspection_payload = None
                                self._plot_origami_source_data()
                                self.origami_identification_baseline = None
                                self._on_origami_identification_setting_changed()
                                self._show_origami_stage("Identify")
                                self._refresh_origami_action_states()
                        elif result_kind == "origami_stage_candidates":
                            for stage_button in (
                                self.origami_run_candidates_button,
                                self.origami_run_alignment_button,
                                self.origami_run_sites_button,
                                self.origami_run_acceptance_button,
                            ):
                                stage_button.state(["!disabled"])
                            self.origami_identification_running = False
                            if (
                                self.loaded is not None
                                and result_payload.get("source_path") == self.loaded.path
                            ):
                                self.origami_staged_candidates = result_payload["candidates"]
                                self.origami_staged_candidate_signature = tuple(
                                    result_payload["signature"]
                                )
                                candidate_count = len(self.origami_staged_candidates[0])
                                self.origami_identification_progress.set(100.0)
                                self.origami_step_progress[1].set(100.0)
                                self.origami_step_progress_text[1].set(
                                    f"Complete: {candidate_count:,} candidates"
                                )
                                self.origami_identification_progress_text.set(
                                    f"Step 1 complete: {candidate_count:,} connected candidates."
                                )
                                self.origami_staged_status.set(
                                    "Step 1 is cached. Adjust it and rerun, or load an alignment template and continue to Step 2."
                                )
                                self._plot_origami_coarse_density()
                                self.status.set(
                                    f"Step 1 complete: inspect {candidate_count:,} connected candidates before alignment."
                                )
                        elif result_kind == "origami_stage_preview":
                            if (
                                self.loaded is None
                                or result_payload.get("source_path") != self.loaded.path
                            ):
                                continue
                            completed_picks = result_payload["picks"]
                            completed_stage = int(result_payload["stage"])
                            self.origami_pick_result = completed_picks
                            self.origami_identification_params = dict(result_payload["params"])
                            self.origami_multi_template_results = {}
                            self.origami_multi_template_counts = {}
                            self.origami_result = None
                            self._set_origami_stage_preview_overlays(completed_stage)
                            self.origami_plot_option.set("Identified origami template matches")
                            self._plot_identified_origamis()
                            self.origami_identification_progress.set(100.0)
                            self.origami_step_progress[completed_stage].set(100.0)
                            self.origami_step_progress_text[completed_stage].set(
                                f"Complete: {len(completed_picks.regions):,} candidates"
                            )
                            self.origami_identification_running = False
                            self.origami_identify_button.state(["!disabled"])
                            for stage_button in (
                                self.origami_run_candidates_button,
                                self.origami_run_alignment_button,
                                self.origami_run_sites_button,
                                self.origami_run_acceptance_button,
                            ):
                                stage_button.state(["!disabled"])
                            self.origami_tiled_button.state(["disabled"])
                            self.origami_n_tiles_button.state(["disabled"])
                            self.origami_random_roi_button.state(["disabled"])
                            next_stage = completed_stage + 1
                            self.origami_identification_progress_text.set(
                                f"Step {completed_stage} complete: {len(completed_picks.regions):,} candidates available for inspection."
                            )
                            self.origami_staged_status.set(
                                f"Step {completed_stage} is ready to inspect. Adjust settings or continue to Step {next_stage}."
                            )
                            self.status.set(
                                f"Step {completed_stage} complete; showing every candidate before final acceptance."
                            )
                        elif result_kind == "origami_picks":
                            completed_picks: OrigamiPickResult = result_payload["picks"]
                            self.origami_identification_progress.set(100.0)
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self.origami_pick_result = completed_picks
                                self.origami_identification_params = dict(result_payload["params"])
                                self.origami_result = None
                                self.origami_match_roi.set(1)
                                self.origami_match_roi_label.set(
                                    f"Candidate 1/{len(completed_picks.regions):,}"
                                    if completed_picks.regions
                                    else "Candidate –/–"
                                )
                                self.origami_identification_baseline = (
                                    self.origami_pending_identification_snapshot
                                    if self.origami_pending_identification_snapshot is not None
                                    else self._origami_identification_snapshot()
                                )
                                self.origami_pending_identification_snapshot = None
                                self.origami_inspected_tile_indices.clear()
                                self.origami_roi_history.clear()
                                self.origami_roi_history_position = -1
                                self.origami_random_inspection_payload = None
                                self._on_origami_identification_setting_changed()
                                self._refresh_origami_action_states()
                                self.origami_show_theoretical_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY
                                )
                                self.origami_show_alignment_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY
                                )
                                self.origami_show_detected_sites_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY
                                )
                                self.origami_show_site_diagnostics.set(
                                    DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS
                                )
                                self.origami_show_localization_group_assignments.set(
                                    DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS
                                )
                                self.origami_show_prominence_geometry.set(
                                    DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY
                                )
                                self.origami_show_text_statistics.set(
                                    DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS
                                )
                                self._plot_identified_origamis()
                                if completed_picks.accepted_count > 0:
                                    self.after_idle(self._auto_build_active_template_overlay)
                            self._finish_origami_identification_progress(
                                f"Complete: {completed_picks.accepted_count}/{len(completed_picks.regions)} candidates passed point, site-prominence, grid-coverage, and spacing limits."
                            )
                        elif result_kind == "origami_multi_picks":
                            self.origami_identification_progress.set(100.0)
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                if "points_nm" in result_payload and "locs" in result_payload:
                                    # A tiled multi-template run owns a new
                                    # whole-selection spatial source. Install
                                    # it before drawing any classification so
                                    # views cannot fall back to the validation
                                    # ROI's one-window image.
                                    self.origami_source_points_nm = np.asarray(
                                        result_payload["points_nm"], dtype=float
                                    )
                                    self.origami_source_candidate_fingerprint = origami_source_fingerprint(
                                        self.origami_source_points_nm
                                    )
                                    self.origami_source_locs = result_payload["locs"]
                                    self.origami_source_render_result = None
                                    self.origami_loaded_source_label = str(
                                        result_payload.get("source", self.origami_loaded_source_label)
                                    )
                                    self.origami_roi_history_position = -1
                                template_results = list(result_payload["templates"])
                                names = [str(item["name"]) for item in template_results]
                                counts = np.asarray(result_payload["counts"], dtype=int)
                                shared_multi_params = dict(result_payload.get("params", {}))
                                self.origami_multi_template_results = {
                                    name: {
                                        "picks": item["picks"],
                                        # Keep the complete custom-template
                                        # collection available after choosing
                                        # an individual type. Tiled analysis
                                        # otherwise sees only that selected
                                        # template and loses classification.
                                        "params": {
                                            **shared_multi_params,
                                            **dict(item["params"]),
                                        },
                                    }
                                    for name, item in zip(names, template_results)
                                }
                                self.origami_multi_template_counts = {
                                    name: int(count) for name, count in zip(names, counts)
                                }
                                self.origami_multi_template_overlay_results = {
                                    str(name): dict(payload)
                                    for name, payload in result_payload.get("overlays", {}).items()
                                }
                                self.origami_multi_template_overlays_building = set()
                                self.origami_multi_template_unclassified_count = int(
                                    result_payload["unclassified_count"]
                                )
                                self.origami_multi_template_unclassified_centers_nm = np.asarray(
                                    result_payload.get("unclassified_centers_nm", np.empty((0, 2))),
                                    dtype=float,
                                ).reshape(-1, 2)
                                self.origami_multi_template_unclassified_details = [
                                    dict(detail)
                                    for detail in result_payload.get("unclassified_details", ())
                                ]
                                assigned_names = [
                                    name for name, count in zip(names, counts) if int(count) > 0
                                ]
                                active_name = assigned_names[0] if assigned_names else names[0]
                                active_payload = self.origami_multi_template_results[active_name]
                                self.origami_pick_result = active_payload["picks"]
                                self.origami_identification_params = dict(active_payload["params"])
                                active_overlay = self.origami_multi_template_overlay_results.get(active_name)
                                if active_overlay is None:
                                    self.origami_result = None
                                    self.origami_result_render_settings = None
                                else:
                                    self.origami_result = active_overlay["result"]
                                    self.origami_result_source = str(active_overlay["source"])
                                    self.origami_result_source_count = int(active_overlay["source_count"])
                                    self.origami_result_render_settings = dict(active_overlay["render_settings"])
                                    self.origami_result_occupancy_threshold = int(
                                        active_overlay["occupancy_threshold"]
                                    )
                                self.origami_template_result_view.set("All templates")
                                self.origami_template_result_combo.configure(
                                    values=("All templates", *names)
                                )
                                self.origami_template_result_combo.state(["!disabled", "readonly"])
                                self.origami_identification_baseline = (
                                    self.origami_pending_identification_snapshot
                                    if self.origami_pending_identification_snapshot is not None
                                    else self._origami_identification_snapshot()
                                )
                                self.origami_pending_identification_snapshot = None
                                self._on_origami_identification_setting_changed()
                                self.origami_show_theoretical_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY
                                )
                                self.origami_show_alignment_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_ALIGNMENT_OVERLAY
                                )
                                self.origami_show_detected_sites_overlay.set(
                                    DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY
                                )
                                self.origami_show_site_diagnostics.set(
                                    DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS
                                )
                                self.origami_show_localization_group_assignments.set(
                                    DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS
                                )
                                self.origami_show_prominence_geometry.set(
                                    DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY
                                )
                                self.origami_show_text_statistics.set(
                                    DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS
                                )
                                self.origami_result = None
                                self.origami_result_render_settings = None
                                self.origami_plot_option.set("Identified origami template matches")
                                self._plot_all_template_classifications()
                                self._refresh_origami_action_states()
                            assigned = int(np.sum(np.asarray(result_payload["counts"], dtype=int)))
                            rejection_summary = dict(
                                result_payload.get("rejection_summary", {})
                            )
                            rejection_note = ""
                            if rejection_summary:
                                rejection_note = "; rejection gates: " + ", ".join(
                                    f"{name} {int(count):,}"
                                    for name, count in sorted(
                                        rejection_summary.items(),
                                        key=lambda item: (-int(item[1]), str(item[0])),
                                    )
                                )
                            self._finish_origami_identification_progress(
                                f"Complete: {assigned:,} candidates classified across "
                                f"{len(result_payload['templates'])} templates; "
                                f"{int(result_payload['unclassified_count']):,} unclassified; "
                                f"{int(result_payload.get('suppressed_duplicate_count', 0)):,} "
                                f"weaker nearby duplicate fits suppressed{rejection_note}."
                            )
                        elif result_kind == "origami_random_roi":
                            completed_picks = result_payload["picks"]
                            self.origami_identification_progress.set(100.0)
                            self._finish_origami_identification_progress(
                                f"Random ROI complete: {completed_picks.accepted_count}/"
                                f"{len(completed_picks.regions)} candidates accepted."
                            )
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self.origami_inspected_tile_indices.add(int(result_payload["tile_index"]))
                                self.origami_roi_history.append(dict(result_payload))
                                self.origami_roi_history_position = len(self.origami_roi_history) - 1
                                pending_view = self.origami_pending_roi_view
                                self.origami_pending_roi_view = None
                                if pending_view == "Coarse identification density":
                                    self.origami_plot_option.set("Coarse identification density")
                                    self._plot_origami_coarse_density()
                                else:
                                    self._plot_random_origami_roi(result_payload)
                        elif result_kind == "origami_tiled":
                            self.origami_identification_progress.set(100.0)
                            tile_count = int(result_payload["tile_count"])
                            available_tile_count = int(result_payload.get("available_tile_count", tile_count))
                            tile_summary = (
                                f"{tile_count:,} selected of {available_tile_count:,} available tiles"
                                if tile_count < available_tile_count
                                else f"all {tile_count:,} available tiles"
                            )
                            self._finish_origami_identification_progress(
                                f"Complete: {result_payload['accepted_count']:,}/{result_payload['candidate_count']:,} "
                                f"candidates accepted across {tile_summary}."
                            )
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self._apply_tiled_origami_result(result_payload)
                                width_nm, height_nm = result_payload["tile_size_nm"]
                                self.status.set(
                                    f"Tiled analysis complete: {result_payload['accepted_count']:,} origamis from "
                                    f"{result_payload['candidate_count']:,} candidates in {result_payload['nonempty_tile_count']:,}/"
                                    f"{result_payload['tile_count']:,} selected complete {width_nm:g} × {height_nm:g} nm ROIs "
                                    f"({tile_summary})."
                                )
                        elif result_kind == "origami":
                            if self.loaded is not None and result_payload.get("source_path") == self.loaded.path:
                                self._plot_origami_analysis(result_payload)
                                self._refresh_origami_action_states()
                    except Exception as exc:
                        details = traceback.format_exc()
                        self.status.set("Error")
                        self._show_error_indicator(str(exc), details)
                        messagebox.showerror("Analysis error", f"{exc}\n\n{details}")
        except queue.Empty:
            pass
        self.after(100, self._poll_worker)

    def _after_load(self, loaded: LoadedData, *, render_raw: bool = True) -> None:
        self.dynamic_render_request_id += 1
        self.dynamic_render_pending = False
        if self.dynamic_render_after_id is not None:
            try:
                self.after_cancel(self.dynamic_render_after_id)
            except Exception:
                pass
            self.dynamic_render_after_id = None
        self.loaded = loaded
        self.development_session_cache_path = None
        self.corrected_locs = None
        self.linked_locs = None
        self.filtered_map_locs = None
        self.filtered_map_render_context = {}
        self.linked_source_count = 0
        self.linked_roi_nm = None
        self.linked_params = None
        self.linked_source_name = self.linking_source.get()
        self.linked_scope_name = self.linking_scope.get()
        self.drift = None
        self.drift_file_path = None
        self.drift_file_label.set("No drift file selected")
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
        self.origami_source_locs = None
        self.origami_source_render_result = None
        self.origami_pick_result = None
        self.origami_loaded_source_label = ""
        self.origami_loaded_source_path = None
        self.origami_loaded_roi_nm = None
        self.origami_loaded_source_params = None
        self.origami_identification_params = None
        self.origami_multi_template_results = {}
        self.origami_multi_template_counts = {}
        self.origami_multi_template_overlay_results = {}
        self.origami_multi_template_overlays_building = set()
        self.origami_single_overlay_building = False
        self.origami_multi_template_unclassified_count = 0
        self.origami_multi_template_unclassified_centers_nm = np.empty((0, 2), dtype=float)
        self.origami_multi_template_unclassified_details = []
        self.origami_template_result_view.set("All templates")
        self.origami_template_result_combo.state(["disabled"])
        self.origami_tiled_button.state(["disabled"])
        self.origami_n_tiles_button.state(["disabled"])
        self.origami_random_roi_button.state(["disabled"])
        self.origami_inspected_tile_indices.clear()
        self.origami_roi_history.clear()
        self.origami_roi_history_position = -1
        self.origami_random_inspection_payload = None
        self.origami_identification_baseline = None
        self._on_origami_identification_setting_changed()
        self._refresh_origami_action_states()
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
        if render_raw:
            self.status.set(f"Loaded {loaded.path}. Rendering raw uncorrected map...")
            self.show_raw_map(auto_fit=True)
        else:
            self.status.set(f"Restored cached localization tables for {loaded.path.name}.")

    def _clear_outputs_for_new_file(self) -> None:
        self.map_density_images.clear()
        if self.density_refresh_after_id is not None:
            try:
                self.after_cancel(self.density_refresh_after_id)
            except Exception:
                pass
            self.density_refresh_after_id = None
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
        self.temporal_annotation_artists = []
        self.temporal_axis = self.temporal_figure.add_subplot(111)
        self.temporal_axis.set_title("No temporal metric plotted")
        self.temporal_axis.set_xlabel("Frame")
        self.temporal_axis.set_ylabel("Metric")
        self.temporal_figure.tight_layout()
        self.temporal_canvas.draw_idle()
        self.origami_figure.clear()
        self.origami_figure.suptitle("1. Load source   2. Identify   3. Overlay")
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
        options = ["photons", "precision_nm", "precision_radial_nm", "lpx_nm", "lpy_nm", "frame", "frame_gap", "localizations_per_frame", "sx", "sy", "bg", "nearest_neighbor_nm", "event_length_frames", "event_length_ms", "event_locs", "event_photons"]
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
            self.map_density_images[RAW_MAP_TAB] = image
            display_image, density_limits = self._scale_map_density(image)
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
            f"Rendered raw map with {result['n_rendered']:,} uncorrected localizations at "
            f"{float(result['disp_px_size_nm']):.3g} nm/pixel. "
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
            self.map_density_images[CORRECTED_MAP_TAB] = image
            display_image, density_limits = self._scale_map_density(image)
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
        action = "Restored cached map for" if result.get("restored_from_cache") else "Rendered"
        self.status.set(
            f"{action} {result['n_rendered']:,} corrected localizations at "
            f"{float(result['disp_px_size_nm']):.3g} nm/pixel. "
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
        self.temporal_annotation_artists = []
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
        self._draw_temporal_annotations()
        self.temporal_figure.tight_layout()
        self.temporal_canvas.draw_idle()
        self.notebook.select(TEMPORAL_TAB)

    def _draw_origami_source_density(
        self,
        axis: Any,
        points_nm: np.ndarray,
        picks: OrigamiPickResult | None = None,
        render_result: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        cached_render = self.origami_source_render_result if render_result is None else render_result
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
        self.origami_source_density_artist = image
        axis.set_position(ORIGAMI_SOURCE_AXES_RECT)
        axis.set_anchor("C")
        colorbar_axis = self.origami_figure.add_axes(ORIGAMI_SOURCE_COLORBAR_RECT)
        colorbar = self.origami_figure.colorbar(image, cax=colorbar_axis)
        self.origami_source_colorbar = colorbar
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
        self.origami_match_panel_combo.state(["disabled"])
        self.origami_gallery_view_limits = None
        self.origami_gallery_home_limits = None
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        preview = self._draw_origami_source_density(axis, points)
        axis.callbacks.connect("xlim_changed", self._on_origami_view_limits_changed)
        axis.callbacks.connect("ylim_changed", self._on_origami_view_limits_changed)
        pixel_x = float(preview["effective_pixel_x_nm"])
        pixel_y = float(preview["effective_pixel_y_nm"])
        axis.set_title(
            f"Loaded source data: {self.origami_loaded_source_label}\n"
            f"{len(points):,} source points; display pixel {pixel_x:.3g} × {pixel_y:.3g} nm; "
            f"blur={preview.get('blur_method', 'smooth')}"
        )
        self.origami_canvas.draw_idle()
        self.origami_last_rendered_plot_option = "Loaded source data"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        # The cached preview was produced before the Origami tab became
        # visible, often using the map tab's zoom/render resolution. Wait
        # until Tk has laid out this canvas, then render its actual viewport.
        self.after_idle(self._rerender_loaded_origami_source_view)
        self.status.set(f"Loaded and displayed {len(points):,} source points. Tune identification settings, then click Identify Origami.")

    def _rerender_loaded_origami_source_view(self) -> None:
        if (
            self.origami_last_rendered_plot_option == "Loaded source data"
            and self._current_notebook_tab_index() == ORIGAMI_TAB
        ):
            self.update_idletasks()
            self._schedule_origami_zoom_render(delay_ms=0)

    def _draw_all_template_classifications(self, axis: Any) -> tuple[int, int]:
        for artist in self.origami_footprint_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.origami_footprint_artists = []
        x0, x1 = sorted(float(value) for value in axis.get_xlim())
        y0, y1 = sorted(float(value) for value in axis.get_ylim())
        show_sites = bool(self.origami_show_theoretical_overlay.get())
        alignment_toggle = getattr(self, "origami_show_alignment_overlay", None)
        show_alignment = bool(alignment_toggle.get()) if alignment_toggle is not None else False
        show_detected = bool(self.origami_show_detected_sites_overlay.get())
        show_diagnostics = bool(self.origami_show_site_diagnostics.get())
        assignment_toggle = getattr(
            self, "origami_show_localization_group_assignments", None
        )
        show_group_assignments = (
            bool(assignment_toggle.get()) if assignment_toggle is not None else False
        )
        show_prominence = bool(self.origami_show_prominence_geometry.get())
        show_labels = bool(self.origami_show_text_statistics.get())
        names = list(self.origami_multi_template_results)
        palette = matplotlib.colormaps["tab10"]
        legend_handles: list[Any] = []
        shown = 0
        total_visible = 0
        maximum_per_type = max(1, 500 // max(1, len(names)))
        detected_positions: list[np.ndarray] = []
        assignment_segments: list[np.ndarray] = []
        failed_site_positions: list[np.ndarray] = []
        failed_site_colors: list[str] = []
        diagnostic_annotations: list[tuple[np.ndarray, str, str]] = []
        prominence_boundaries: list[np.ndarray] = []
        prominence_links: list[np.ndarray] = []
        prominence_peaks: list[np.ndarray] = []
        prominence_references: list[np.ndarray] = []

        diagnostic_localization_limit = 50
        diagnostic_localizations_visible = 0
        if show_group_assignments:
            for name in names:
                candidate_picks: OrigamiPickResult = self.origami_multi_template_results[name]["picks"]
                candidate_bounds = np.asarray(candidate_picks.bounds_nm, dtype=float)
                diagnostic_localizations_visible += int(np.count_nonzero(
                    np.asarray(candidate_picks.accepted_mask, dtype=bool)
                    & (candidate_bounds[:, 1] >= x0)
                    & (candidate_bounds[:, 0] <= x1)
                    & (candidate_bounds[:, 3] >= y0)
                    & (candidate_bounds[:, 2] <= y1)
                ))
        draw_localization_assignments = (
            show_group_assignments
            and diagnostic_localizations_visible <= diagnostic_localization_limit
        )

        for type_index, name in enumerate(names):
            payload = self.origami_multi_template_results[name]
            picks: OrigamiPickResult = payload["picks"]
            params = dict(payload["params"])
            overlay_grid = classified_template_overlay_points(
                picks.template_points_nm,
                params,
            )
            alignment_grid = alignment_template_overlay_points(
                picks.template_points_nm,
                params,
            )
            color = palette(type_index % 10)
            bounds = np.asarray(picks.bounds_nm, dtype=float)
            visible = np.flatnonzero(
                np.asarray(picks.accepted_mask, dtype=bool)
                & (bounds[:, 1] >= x0)
                & (bounds[:, 0] <= x1)
                & (bounds[:, 3] >= y0)
                & (bounds[:, 2] <= y1)
            )
            total_visible += len(visible)
            if len(visible) > maximum_per_type:
                visible = visible[np.linspace(0, len(visible) - 1, maximum_per_type, dtype=int)]
            shown += len(visible)

            fitted_centers: list[np.ndarray] = []
            site_positions: list[np.ndarray] = []
            alignment_positions: list[np.ndarray] = []
            for region_index in visible:
                corners = picks.rectangle_corners_nm[region_index]
                center = np.mean(corners, axis=0)
                fitted_centers.append(center)
                fitted_grid = theoretical_grid_in_footprint(picks.template_points_nm, corners)
                if show_sites:
                    displayed_overlay_grid = overlay_grid
                    if show_diagnostics:
                        decision_overlay_grid = digital_group_decision_overlay_points(
                            picks.template_points_nm,
                            params,
                            int(region_index),
                        )
                        if decision_overlay_grid is not None:
                            displayed_overlay_grid = decision_overlay_grid
                    site_positions.append(
                        theoretical_grid_in_footprint(displayed_overlay_grid, corners)
                    )
                if show_alignment and len(alignment_grid):
                    alignment_positions.append(
                        theoretical_grid_in_footprint(alignment_grid, corners)
                    )
                if show_detected or show_diagnostics or show_group_assignments or show_prominence:
                    decisions = [
                        origami_site_decision_label(
                            int(count),
                            float(prominence),
                            int(params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS)),
                            float(params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE)),
                        )
                        for count, prominence in zip(
                            picks.site_localization_counts[region_index],
                            picks.site_prominence[region_index],
                        )
                    ]
                    measured_centroids = picks.site_centroids_nm[region_index]
                    measured = np.all(np.isfinite(measured_centroids), axis=1)
                    if show_detected and np.any(measured):
                        measured_world = theoretical_grid_in_footprint(
                            measured_centroids[measured], corners
                        )
                        detected_positions.append(measured_world)
                        assignment_segments.extend(
                            np.asarray((observed, target), dtype=float)
                            for observed, target in zip(measured_world, fitted_grid[measured])
                        )
                    if show_diagnostics or show_group_assignments:
                        digital_annotations = digital_group_decision_annotations(
                            picks.template_points_nm,
                            params,
                            int(region_index),
                        )
                        if digital_annotations is not None:
                            if show_diagnostics:
                                self.origami_footprint_artists.extend(
                                    PaintAnalysisApp._draw_digital_decision_blobs(
                                        self,
                                        axis,
                                        picks.template_points_nm,
                                        params,
                                        int(region_index),
                                        lambda points, corners=corners: theoretical_grid_in_footprint(
                                            points, corners
                                        ),
                                    )
                                )
                            aligned_regions = getattr(picks, "aligned_regions", ())
                            if draw_localization_assignments and region_index < len(aligned_regions):
                                self.origami_footprint_artists.extend(
                                    PaintAnalysisApp._draw_digital_group_localization_assignments(
                                        self,
                                        axis,
                                        aligned_regions[region_index],
                                        picks.template_points_nm,
                                        params,
                                        lambda points, corners=corners: theoretical_grid_in_footprint(
                                            points, corners
                                        ),
                                    )
                                )
                            if show_diagnostics:
                                for local_center, diagnostic, diagnostic_color in digital_annotations:
                                    target = theoretical_grid_in_footprint(
                                        local_center[None, :], corners
                                    )[0]
                                    diagnostic_annotations.append(
                                        (target, diagnostic, diagnostic_color)
                                    )
                        elif show_diagnostics:
                            for target, decision in zip(fitted_grid, decisions):
                                supported, diagnostic_color, diagnostic = decision
                                if not supported:
                                    failed_site_positions.append(target)
                                    failed_site_colors.append(diagnostic_color)
                                diagnostic_annotations.append((target, diagnostic, diagnostic_color))
                    if show_prominence:
                        peak_positions = picks.site_peak_positions_nm[region_index]
                        boundary_points = picks.site_boundary_points_nm[region_index]
                        boundary_references = picks.site_boundary_reference_positions_nm[region_index]
                        valid_peaks = np.all(np.isfinite(peak_positions), axis=1)
                        for site_index in np.flatnonzero(valid_peaks):
                            peak_world = theoretical_grid_in_footprint(
                                peak_positions[[site_index]], corners
                            )[0]
                            prominence_peaks.append(peak_world)
                            boundary = boundary_points[site_index]
                            finite_boundary = np.all(np.isfinite(boundary), axis=1)
                            if np.any(finite_boundary):
                                boundary_world = theoretical_grid_in_footprint(
                                    boundary[finite_boundary], corners
                                )
                                prominence_boundaries.append(
                                    np.vstack((boundary_world, boundary_world[0]))
                                )
                            reference = boundary_references[site_index]
                            if np.all(np.isfinite(reference)):
                                reference_world = theoretical_grid_in_footprint(
                                    reference[None, :], corners
                                )[0]
                                prominence_references.append(reference_world)
                                prominence_links.append(np.vstack((peak_world, reference_world)))
                if show_labels:
                    classification_scores = params.get("classification_scores", ())
                    monte_carlo_posteriors = params.get("classification_monte_carlo_posterior", ())
                    lattice_precisions = params.get("classification_lattice_precision", ())
                    empty_cell_fractions = params.get("classification_empty_cell_fraction", ())
                    pattern_correlations = params.get("classification_cell_pattern_correlation", ())
                    template_probabilities = params.get("classification_template_probabilities", ())
                    winner_margins = params.get("classification_winner_margins", ())
                    crop_retention = params.get("classification_crop_retained_fraction", ())
                    classification_score_text = (
                        f"; evidence={float(classification_scores[region_index]):.2f}"
                        if region_index < len(classification_scores)
                        else ""
                    )
                    evidence_abbreviation = (
                        "digital p"
                        if params.get("classification_evidence_model") in {"logical bits", "logical strokes", "direct digital groups"}
                        else "MC p"
                    )
                    monte_carlo_text = (
                        f"; {evidence_abbreviation}={float(monte_carlo_posteriors[region_index]):.2f}"
                        if region_index < len(monte_carlo_posteriors)
                        else ""
                    )
                    lattice_precision_text = (
                        f"; cell-match={float(lattice_precisions[region_index]):.2f}"
                        if region_index < len(lattice_precisions)
                        else ""
                    )
                    empty_cell_text = (
                        f"; black-empty={float(empty_cell_fractions[region_index]):.2f}"
                        if region_index < len(empty_cell_fractions)
                        else ""
                    )
                    pattern_text = (
                        f"; pattern={float(pattern_correlations[region_index]):.2f}"
                        if region_index < len(pattern_correlations)
                        else ""
                    )
                    template_probability_text = (
                        f"; class-p={float(template_probabilities[region_index]):.2f}"
                        if region_index < len(template_probabilities)
                        else ""
                    )
                    winner_margin_text = (
                        f"; margin={float(winner_margins[region_index]):.2f}"
                        if region_index < len(winner_margins)
                        and np.isfinite(winner_margins[region_index])
                        else ""
                    )
                    crop_retention_text = (
                        f"; retained={100.0 * float(crop_retention[region_index]):.0f}%"
                        if region_index < len(crop_retention)
                        else ""
                    )
                    label = axis.annotate(
                        f"{name}: n={int(picks.point_counts[region_index]):,}; "
                        f"corr={float(picks.rectangle_confidence[region_index]):.2f}; "
                        f"sites={int(picks.supported_site_count[region_index])}; "
                        f"spacing={float(picks.site_spacing_max_error_nm[region_index]):.1f} nm; "
                        f"gap={float(picks.site_gap_contrast[region_index]):.2f}"
                        f"{lattice_precision_text}"
                        f"{empty_cell_text}"
                        f"{pattern_text}"
                        f"{monte_carlo_text}"
                        f"{template_probability_text}"
                        f"{winner_margin_text}"
                        f"{crop_retention_text}"
                        f"{classification_score_text}",
                        xy=center,
                        xytext=(3, 3),
                        textcoords="offset points",
                        color=color,
                        fontsize=7,
                        fontweight="bold",
                        clip_on=True,
                        zorder=7,
                    )
                    label.set_in_layout(False)
                    self.origami_footprint_artists.append(label)
            if fitted_centers:
                centers = np.vstack(fitted_centers)
                center_artist = axis.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    s=22,
                    marker="D",
                    facecolors=[color],
                    edgecolors="#111827",
                    linewidths=0.5,
                    alpha=0.95,
                    zorder=4.5,
                )
                center_artist.set_in_layout(False)
                self.origami_footprint_artists.append(center_artist)
            if site_positions:
                sites = np.vstack(site_positions)
                site_artist = axis.scatter(
                    sites[:, 0],
                    sites[:, 1],
                    s=18,
                    marker="o",
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=0.8,
                    alpha=0.95,
                    zorder=5,
                )
                site_artist.set_in_layout(False)
                self.origami_footprint_artists.append(site_artist)
            if alignment_positions:
                alignment_sites = np.vstack(alignment_positions)
                alignment_artist = axis.scatter(
                    alignment_sites[:, 0],
                    alignment_sites[:, 1],
                    s=24,
                    marker="s",
                    facecolors="none",
                    edgecolors="#22d3ee",
                    linewidths=1.0,
                    alpha=0.95,
                    zorder=5.1,
                )
                alignment_artist.set_in_layout(False)
                self.origami_footprint_artists.append(alignment_artist)
            legend_handles.append(
                matplotlib.patches.Patch(
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.5,
                    label=f"{name} ({self.origami_multi_template_counts.get(name, 0):,})",
                )
            )

        if detected_positions:
            measured = np.vstack(detected_positions)
            links = LineCollection(
                assignment_segments,
                colors="#a3e635",
                linewidths=0.7,
                alpha=0.75,
                zorder=5.0,
            )
            axis.add_collection(links)
            links.set_in_layout(False)
            self.origami_footprint_artists.append(links)
            detected_artist = axis.scatter(
                measured[:, 0],
                measured[:, 1],
                s=18,
                marker="o",
                facecolors="#a3e635",
                edgecolors="#14532d",
                linewidths=0.7,
                alpha=0.95,
                zorder=5.2,
            )
            detected_artist.set_in_layout(False)
            self.origami_footprint_artists.append(detected_artist)
        if failed_site_positions:
            failed = np.vstack(failed_site_positions)
            failed_artist = axis.scatter(
                failed[:, 0],
                failed[:, 1],
                s=25,
                marker="x",
                c=failed_site_colors,
                linewidths=1.1,
                alpha=0.95,
                zorder=5.5,
            )
            failed_artist.set_in_layout(False)
            self.origami_footprint_artists.append(failed_artist)
        detailed_view = total_visible <= 12
        if show_diagnostics and detailed_view:
            for target, diagnostic, diagnostic_color in diagnostic_annotations:
                annotation = axis.annotate(
                    diagnostic,
                    xy=target,
                    xytext=(3, 3),
                    textcoords="offset points",
                    color=diagnostic_color,
                    fontsize=6,
                    fontweight="bold",
                    clip_on=True,
                    zorder=7,
                )
                annotation.set_in_layout(False)
                self.origami_footprint_artists.append(annotation)
        if show_prominence and detailed_view:
            if prominence_boundaries:
                boundaries = LineCollection(
                    prominence_boundaries,
                    colors="#22d3ee",
                    linewidths=0.65,
                    linestyles="dashed",
                    alpha=0.75,
                    zorder=5.6,
                )
                axis.add_collection(boundaries)
                boundaries.set_in_layout(False)
                self.origami_footprint_artists.append(boundaries)
            if prominence_links:
                reference_links = LineCollection(
                    prominence_links,
                    colors="#f472b6",
                    linewidths=0.55,
                    alpha=0.7,
                    zorder=5.7,
                )
                axis.add_collection(reference_links)
                reference_links.set_in_layout(False)
                self.origami_footprint_artists.append(reference_links)
            if prominence_peaks:
                peaks = np.vstack(prominence_peaks)
                peak_artist = axis.scatter(
                    peaks[:, 0], peaks[:, 1], s=20, marker="D",
                    facecolors="#22d3ee", edgecolors="#164e63",
                    linewidths=0.6, zorder=6.2,
                )
                peak_artist.set_in_layout(False)
                self.origami_footprint_artists.append(peak_artist)
            if prominence_references:
                references = np.vstack(prominence_references)
                reference_artist = axis.scatter(
                    references[:, 0], references[:, 1], s=14, marker="s",
                    facecolors="#f472b6", edgecolors="#831843",
                    linewidths=0.5, zorder=6.1,
                )
                reference_artist.set_in_layout(False)
                self.origami_footprint_artists.append(reference_artist)
        if (show_diagnostics or show_prominence) and not detailed_view:
            note = axis.text(
                0.01,
                0.01,
                "Zoom to 12 or fewer classified candidates for decision labels and prominence geometry.",
                transform=axis.transAxes,
                color="#f8fafc",
                fontsize=8,
                bbox={"facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.85},
                ha="left",
                va="bottom",
                zorder=8,
            )
            note.set_in_layout(False)
            self.origami_footprint_artists.append(note)
        if show_group_assignments and not draw_localization_assignments:
            note = axis.text(
                0.01,
                0.045,
                "Zoom to 50 or fewer classified candidates to color individual localizations by digital group.",
                transform=axis.transAxes,
                color="#f8fafc",
                fontsize=8,
                bbox={"facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.85},
                ha="left",
                va="bottom",
                zorder=8,
            )
            note.set_in_layout(False)
            self.origami_footprint_artists.append(note)

        unclassified = np.asarray(
            getattr(self, "origami_multi_template_unclassified_centers_nm", np.empty((0, 2))),
            dtype=float,
        ).reshape(-1, 2)
        if len(unclassified):
            visible_unclassified = unclassified[
                (unclassified[:, 0] >= x0)
                & (unclassified[:, 0] <= x1)
                & (unclassified[:, 1] >= y0)
                & (unclassified[:, 1] <= y1)
            ]
            unclassified_artist = axis.scatter(
                visible_unclassified[:, 0],
                visible_unclassified[:, 1],
                s=22,
                marker="x",
                c="#9ca3af",
                linewidths=1.0,
                alpha=0.9,
                zorder=4,
            )
            unclassified_artist.set_in_layout(False)
            self.origami_footprint_artists.append(unclassified_artist)
        if show_labels:
            for detail in getattr(self, "origami_multi_template_unclassified_details", ()):
                center = np.asarray(detail.get("center_nm", ()), dtype=float)
                if center.shape != (2,) or not np.all(np.isfinite(center)):
                    continue
                if not (x0 <= center[0] <= x1 and y0 <= center[1] <= y1):
                    continue
                cell_match = float(detail.get("cell_match", float("nan")))
                cell_match_text = f"; cell-match={cell_match:.2f}" if np.isfinite(cell_match) else ""
                empty_fraction = float(detail.get("empty_cell_fraction", float("nan")))
                empty_text = f"; black-empty={empty_fraction:.2f}" if np.isfinite(empty_fraction) else ""
                pattern = float(detail.get("cell_pattern_correlation", float("nan")))
                pattern_text = f"; pattern={pattern:.2f}" if np.isfinite(pattern) else ""
                monte_carlo = float(detail.get("monte_carlo_posterior", float("nan")))
                evidence_abbreviation = (
                    "digital p"
                    if detail.get("evidence_model") in {"logical bits", "logical strokes", "direct digital groups"}
                    else "MC p"
                )
                monte_carlo_text = (
                    f"; {evidence_abbreviation}={monte_carlo:.2f}" if np.isfinite(monte_carlo) else ""
                )
                retained = float(detail.get("crop_retained_fraction", float("nan")))
                retained_text = f"; retained={100.0 * retained:.0f}%" if np.isfinite(retained) else ""
                reasons = tuple(str(reason) for reason in detail.get("failure_reasons", ()))
                failure_text = ", ".join(reasons) if reasons else "no template passed all gates"
                label = axis.annotate(
                    f"UNCLASSIFIED — best {detail.get('template_name', 'template')}: "
                    f"n={int(detail.get('point_count', 0)):,}; "
                    f"corr={float(detail.get('correlation', float('nan'))):.2f}; "
                    f"sites={int(detail.get('supported_sites', 0))}; "
                    f"spacing={float(detail.get('spacing_error_nm', float('nan'))):.1f} nm; "
                    f"gap={float(detail.get('site_gap_contrast', float('nan'))):.2f}"
                    f"{cell_match_text}{empty_text}{pattern_text}{monte_carlo_text}{retained_text}; "
                    f"evidence={float(detail.get('classification_score', float('nan'))):.2f}\n"
                    f"FAIL: {failure_text}",
                    xy=center,
                    xytext=(3, 3),
                    textcoords="offset points",
                    color="#9ca3af",
                    fontsize=7,
                    fontweight="bold",
                    clip_on=True,
                    zorder=7,
                )
                label.set_in_layout(False)
                self.origami_footprint_artists.append(label)
        legend_handles.append(
            matplotlib.patches.Patch(
                facecolor="none",
                edgecolor="#9ca3af",
                label=f"Unclassified ({self.origami_multi_template_unclassified_count:,})",
            )
        )
        legend = axis.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)
        legend.set_in_layout(False)
        return shown, total_visible

    def _plot_all_template_classifications(self) -> None:
        points = self.origami_source_points_nm
        if points is None or not self.origami_multi_template_results:
            return
        self.origami_match_panel_combo.state(["disabled"])
        self.origami_roi_history_position = -1
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        preview = self._draw_origami_source_density(axis, points)
        shown, total_visible = self._draw_all_template_classifications(axis)
        axis.callbacks.connect("xlim_changed", self._on_origami_view_limits_changed)
        axis.callbacks.connect("ylim_changed", self._on_origami_view_limits_changed)
        axis.set_title(
            f"Origami template classification: {sum(self.origami_multi_template_counts.values()):,} assigned; "
            f"{self.origami_multi_template_unclassified_count:,} unclassified\n"
            f"Color identifies the winning template; display pixel "
            f"{float(preview['effective_pixel_x_nm']):.3g} × {float(preview['effective_pixel_y_nm']):.3g} nm; "
            f"showing {shown}/{total_visible} classified footprints in view"
        )
        self.origami_canvas.draw_idle()
        self.origami_last_rendered_plot_option = "Identified origami template matches"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(
            f"Showing all template classifications together: {shown:,}/{total_visible:,} visible assigned fits; "
            f"{self.origami_multi_template_unclassified_count:,} unclassified candidates are gray × markers"
            f"{' with best-fit failure statistics' if self.origami_show_text_statistics.get() else ''}."
        )

    def _plot_identified_origamis(self) -> None:
        points = self.origami_source_points_nm
        picks = self.origami_pick_result
        if points is None or picks is None:
            return
        # The panel choice is useful before opening a footprint too: it
        # determines which diagnostic panel appears when the user clicks a
        # match in this overview.
        self.origami_match_panel_combo.state(["!disabled", "readonly"])
        self.origami_roi_history_position = -1
        self.origami_match_roi.set(1)
        self.origami_match_roi_label.set("Validation ROI")
        if self.origami_result is None:
            self.origami_gallery_view_limits = None
            self.origami_gallery_home_limits = None
        self.origami_plot_option.set("Identified origami template matches")
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        axis.set_anchor("C")
        preview = self._draw_origami_source_density(axis, points, picks)
        visible_footprints, total_visible = self._draw_visible_origami_footprints(axis, picks)
        axis.callbacks.connect("xlim_changed", self._on_origami_view_limits_changed)
        axis.callbacks.connect("ylim_changed", self._on_origami_view_limits_changed)
        if self.origami_footprint_release_cid is not None:
            self.origami_canvas.mpl_disconnect(self.origami_footprint_release_cid)
        if self.origami_footprint_scroll_cid is not None:
            self.origami_canvas.mpl_disconnect(self.origami_footprint_scroll_cid)
        self.origami_footprint_release_cid = self.origami_canvas.mpl_connect(
            "button_release_event", lambda _event: self._schedule_origami_footprint_refresh()
        )
        self.origami_footprint_scroll_cid = self.origami_canvas.mpl_connect(
            "scroll_event", lambda _event: self._schedule_origami_footprint_refresh()
        )
        if len(picks.point_counts):
            count_summary = (
                f"region points min/median/max = {int(np.min(picks.point_counts)):,}/"
                f"{int(np.median(picks.point_counts)):,}/{int(np.max(picks.point_counts)):,}"
            )
        else:
            count_summary = "no connected regions"
        if len(picks.rectangle_confidence):
            confidence_summary = (
                f"theoretical-template correlation min/median/max = {np.min(picks.rectangle_confidence):.2f}/"
                f"{np.median(picks.rectangle_confidence):.2f}/{np.max(picks.rectangle_confidence):.2f}"
            )
        else:
            confidence_summary = "no theoretical-template matches"
        if len(picks.site_gap_contrast):
            site_gap_summary = (
                f"site-gap contrast min/median/max = {np.min(picks.site_gap_contrast):.2f}/"
                f"{np.median(picks.site_gap_contrast):.2f}/{np.max(picks.site_gap_contrast):.2f}"
            )
        else:
            site_gap_summary = "no site-gap scores"
        identification_params = dict(self.origami_identification_params or {})
        selected_template = self.origami_template_result_view.get()
        template_title = (
            f" — {selected_template}"
            if selected_template in self.origami_multi_template_results
            else ""
        )
        is_classification_view = selected_template in self.origami_multi_template_results
        accepted_label = "classified as this type" if is_classification_view else "accepted"
        outline_label = (
            "solid assigned to this type; dashed failed or assigned elsewhere"
            if is_classification_view
            else "solid accepted, dashed rejected"
        )
        use_correlation_gate = bool(identification_params.get("use_correlation_gate", True))
        correlation_gate_text = (
            f"correlation ≥ {float(identification_params.get('min_rectangle_confidence', self.origami_min_rectangle_confidence.get())):g}; "
            if use_correlation_gate
            else "correlation shown for QC only; "
        )
        bic_computed = bool(identification_params.get("compute_grid_blob_bic", True))
        bic_summary = (
            f"grid-vs-blob ΔBIC min/median/max = {np.min(picks.grid_vs_blob_delta_bic):.1f}/"
            f"{np.median(picks.grid_vs_blob_delta_bic):.1f}/{np.max(picks.grid_vs_blob_delta_bic):.1f}"
            if bic_computed
            else "grid-vs-blob ΔBIC deferred in multi-template mode"
        )
        support_summary = (
            f"supported sites min/median/max = {int(np.min(picks.supported_site_count))}/"
            f"{int(np.median(picks.supported_site_count))}/{int(np.max(picks.supported_site_count))}; "
            f"spacing max-error min/median/max = {np.min(picks.site_spacing_max_error_nm):.1f}/"
            f"{np.median(picks.site_spacing_max_error_nm):.1f}/{np.max(picks.site_spacing_max_error_nm):.1f} nm; "
            f"{bic_summary}"
            if len(picks.supported_site_count)
            else "no sparse-grid scores"
        )
        axis.set_title(
            f"Identified origami{template_title}: {picks.accepted_count}/{len(picks.regions)} {accepted_label} "
            f"({correlation_gate_text}sparse sites ≥ {int(identification_params.get('min_supported_sites', 0))}; "
            f"site-gap shown for QC only; {'ΔBIC shown for QC only' if bic_computed else 'ΔBIC deferred'}; "
            f"spacing error ≤ {float(identification_params.get('max_site_spacing_error_nm', float('inf'))):g} nm; "
            f"{outline_label})\n"
            f"{count_summary}; {confidence_summary}; {site_gap_summary}; {support_summary}\n"
            f"footprint {picks.rectangle_width_nm:g} × {picks.rectangle_height_nm:g} nm; alignment pixel "
            f"{picks.alignment_pixel_nm:.3g} nm; display pixel "
            f"{float(preview['effective_pixel_x_nm']):.3g} × {float(preview['effective_pixel_y_nm']):.3g} nm; "
            f"pick bin {float(self.origami_pick_bin_nm.get()):g} nm; density ≥ {picks.density_threshold:.3g}; "
            f"showing {visible_footprints}/{total_visible} footprints in view"
        )
        self.origami_canvas.draw_idle()
        self.origami_last_rendered_plot_option = "Identified origami template matches"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        if picks.accepted_count:
            self.status.set(
                f"Outlined {picks.accepted_count} origamis {accepted_label} from {len(picks.regions)} connected regions. "
                "Rejected fits are shown with failure text only when text statistics is enabled. Click a footprint for its template and negative-space diagnostics."
            )
        elif is_classification_view:
            self.status.set(
                f"No candidates were assigned to {selected_template}. Candidates that passed this template but won under another template are labeled OTHER TYPE when text statistics is enabled."
            )
        else:
            self.status.set(
                f"Found {len(picks.regions)} connected regions, but none pass the point, site-prominence, grid-coverage, and spacing limits; "
                f"{count_summary}; {confidence_summary}; {site_gap_summary}; {support_summary}. Adjust the identification settings and rerun."
            )

    def _plot_random_origami_roi(self, payload: dict[str, Any]) -> None:
        points = np.asarray(payload["points_nm"], dtype=float)
        picks: OrigamiPickResult = payload["picks"]
        roi_nm = tuple(float(value) for value in payload["roi_nm"])
        tile_index = int(payload["tile_index"])
        available_tile_count = int(payload["available_tile_count"])
        self.origami_random_inspection_payload = payload
        self.origami_plot_option.set("Identified origami template matches")
        self.origami_match_panel_combo.state(["disabled"])
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        axis = self.origami_figure.add_subplot(111)
        axis.set_anchor("C")
        preview = self._draw_origami_source_density(
            axis,
            points,
            picks,
            render_result=dict(payload["render_result"]),
        )
        visible_footprints, total_visible = self._draw_visible_origami_footprints(axis, picks)
        axis.callbacks.connect("xlim_changed", self._on_origami_view_limits_changed)
        axis.callbacks.connect("ylim_changed", self._on_origami_view_limits_changed)
        if self.origami_footprint_release_cid is not None:
            self.origami_canvas.mpl_disconnect(self.origami_footprint_release_cid)
        if self.origami_footprint_scroll_cid is not None:
            self.origami_canvas.mpl_disconnect(self.origami_footprint_scroll_cid)
        self.origami_footprint_release_cid = self.origami_canvas.mpl_connect(
            "button_release_event", lambda _event: self._schedule_origami_footprint_refresh()
        )
        self.origami_footprint_scroll_cid = self.origami_canvas.mpl_connect(
            "scroll_event", lambda _event: self._schedule_origami_footprint_refresh()
        )
        params = dict(payload["identification_params"])
        if len(picks.point_counts):
            count_summary = (
                f"points min/median/max = {int(np.min(picks.point_counts)):,}/"
                f"{int(np.median(picks.point_counts)):,}/{int(np.max(picks.point_counts)):,}"
            )
            correlation_summary = (
                f"correlation min/median/max = {np.min(picks.rectangle_confidence):.2f}/"
                f"{np.median(picks.rectangle_confidence):.2f}/{np.max(picks.rectangle_confidence):.2f}"
            )
            site_gap_summary = (
                f"site-gap min/median/max = {np.min(picks.site_gap_contrast):.2f}/"
                f"{np.median(picks.site_gap_contrast):.2f}/{np.max(picks.site_gap_contrast):.2f}"
            )
        else:
            count_summary = "no connected candidates"
            correlation_summary = "no template correlations"
            site_gap_summary = "no site-gap scores"
        support_summary = (
            f"supported sites min/median/max = {int(np.min(picks.supported_site_count))}/"
            f"{int(np.median(picks.supported_site_count))}/{int(np.max(picks.supported_site_count))}; "
            f"spacing max-error min/median/max = {np.min(picks.site_spacing_max_error_nm):.1f}/"
            f"{np.median(picks.site_spacing_max_error_nm):.1f}/{np.max(picks.site_spacing_max_error_nm):.1f} nm; "
            f"ΔBIC min/median/max = {np.min(picks.grid_vs_blob_delta_bic):.1f}/"
            f"{np.median(picks.grid_vs_blob_delta_bic):.1f}/{np.max(picks.grid_vs_blob_delta_bic):.1f}"
            if len(picks.supported_site_count)
            else "no sparse-grid scores"
        )
        correlation_gate_text = (
            f"correlation ≥ {float(params['min_rectangle_confidence']):g}; "
            if bool(params.get("use_correlation_gate", True))
            else "correlation QC only; "
        )
        axis.set_title(
            f"Random ROI inspection — tile {tile_index + 1}/{available_tile_count}; "
            f"{picks.accepted_count}/{len(picks.regions)} accepted\n"
            f"ROI x={roi_nm[0]:g}–{roi_nm[1]:g} nm, y={roi_nm[2]:g}–{roi_nm[3]:g} nm; "
            f"{len(points):,} source points; {count_summary}; {correlation_summary}; {site_gap_summary}; {support_summary}\n"
            f"validated settings: pick bin {float(params['pick_bin_size_nm']):g} nm; "
            f"density ≥ {float(params['density_threshold']):g}; {correlation_gate_text}"
            f"sites ≥ {int(params.get('min_supported_sites', 0))}; site-gap and ΔBIC QC only; spacing error ≤ "
            f"{float(params.get('max_site_spacing_error_nm', float('inf'))):g} nm; display pixel "
            f"{float(preview['effective_pixel_x_nm']):.3g} × {float(preview['effective_pixel_y_nm']):.3g} nm; "
            f"showing {visible_footprints}/{total_visible} footprints"
        )
        self.origami_last_rendered_plot_option = "Random ROI inspection"
        roi_number = self.origami_roi_history_position + 2
        self.origami_match_roi.set(roi_number)
        self.origami_match_roi_label.set(
            f"ROI {roi_number} • random tile {tile_index + 1}/{available_tile_count} • {len(picks.regions)} candidates"
        )
        self._configure_origami_navigation_controls()
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(
            f"Random ROI tile {tile_index + 1:,}/{available_tile_count:,}: "
            f"{picks.accepted_count:,}/{len(picks.regions):,} candidates accepted. "
            "Click a footprint for template and negative-space diagnostics, or use the ROI arrows."
        )

    def _show_identified_origami_overview(self) -> None:
        if self.origami_pick_result is None:
            messagebox.showinfo("No identified origami", "Run Identify Origami before viewing candidate ROIs.")
            return
        self._plot_identified_origamis()

    def _change_origami_match_roi(self, delta: int) -> None:
        picks = self.origami_pick_result
        if picks is None or not picks.regions:
            messagebox.showinfo("No identified origami", "Run Identify Origami before navigating candidate ROIs.")
            return
        count = len(picks.regions)
        try:
            requested = int(self.origami_match_roi.get())
        except (tk.TclError, ValueError):
            requested = 1
        if delta:
            requested = ((requested - 1 + int(delta)) % count) + 1
        else:
            requested = max(1, min(count, requested))
        self.origami_match_roi.set(requested)
        self._plot_identified_origami_match_roi(requested - 1)

    def _plot_template_competition(
        self,
        selected_picks: OrigamiPickResult,
        selected_region_index: int,
    ) -> None:
        """Compare every template fit for one spatial candidate side by side."""
        selected_center = np.median(selected_picks.regions[selected_region_index], axis=0)
        names = list(self.origami_multi_template_results)
        panel_count = len(names)
        columns = min(3, max(1, panel_count))
        rows = int(math.ceil(panel_count / columns))
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("constrained", w_pad=5 / 72, h_pad=6 / 72)
        axes = np.asarray(
            self.origami_figure.subplots(rows, columns, squeeze=False),
            dtype=object,
        ).ravel()
        palette = matplotlib.colormaps["tab10"]
        matched_panels = 0
        winning_name = ""
        winning_probability = -1.0

        for template_index, (name, axis) in enumerate(zip(names, axes)):
            payload = self.origami_multi_template_results[name]
            picks: OrigamiPickResult = payload["picks"]
            params = dict(payload["params"])
            centers = np.asarray(
                [np.median(region, axis=0) for region in picks.regions],
                dtype=float,
            ).reshape(-1, 2)
            if not len(centers):
                axis.text(0.5, 0.5, "No candidate fit", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(name, color=palette(template_index % 10), fontweight="bold")
                axis.set_axis_off()
                continue
            distances = np.linalg.norm(centers - selected_center, axis=1)
            candidate_index = int(np.argmin(distances))
            match_distance_nm = max(
                float(params.get("connect_distance_nm", DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM)),
                0.45 * max(float(picks.rectangle_width_nm), float(picks.rectangle_height_nm)),
            )
            if float(distances[candidate_index]) > match_distance_nm:
                axis.text(
                    0.5,
                    0.5,
                    f"No corresponding fit\nnearest center {distances[candidate_index]:.1f} nm away",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(name, color=palette(template_index % 10), fontweight="bold")
                axis.set_axis_off()
                continue
            matched_panels += 1
            candidate = np.asarray(picks.alignment_candidate_images[candidate_index], dtype=float)
            candidate_display = candidate - float(np.min(candidate))
            candidate_display /= max(float(np.max(candidate_display)), 1e-12)
            side_nm = float(
                getattr(
                    picks,
                    "alignment_canvas_side_nm",
                    np.hypot(picks.rectangle_width_nm, picks.rectangle_height_nm),
                )
            )
            extent = (-side_nm / 2.0, side_nm / 2.0, -side_nm / 2.0, side_nm / 2.0)
            axis.imshow(
                candidate_display,
                extent=extent,
                origin="lower",
                cmap="magma",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            template_sites = classified_template_overlay_points(
                picks.template_points_nm,
                params,
            )
            axis.scatter(
                template_sites[:, 0],
                template_sites[:, 1],
                s=32,
                facecolors="none",
                edgecolors="#22d3ee",
                linewidths=1.0,
                label="expected bright",
            )
            full_grid = ideal_grid_points(
                int(params["rows"]),
                int(params["columns"]),
                float(params["spacing_x_nm"]),
                float(params["spacing_y_nm"]),
            )
            supported = np.asarray(
                picks.lattice_supported_sites[candidate_index], dtype=bool
            )
            localization_counts = np.asarray(
                picks.lattice_site_localization_counts[candidate_index], dtype=float
            )
            expected_cells = np.zeros(len(full_grid), dtype=bool)
            if len(template_sites):
                _distances, cells = cKDTree(full_grid).query(template_sites, k=1)
                expected_cells[np.asarray(cells, dtype=int)] = True
            supported_expected = supported & expected_cells
            supported_unexpected = supported & ~expected_cells
            if np.any(supported_expected):
                axis.scatter(
                    full_grid[supported_expected, 0],
                    full_grid[supported_expected, 1],
                    s=12.0 + 5.0 * np.sqrt(np.maximum(localization_counts[supported_expected], 0.0)),
                    marker="o",
                    facecolors="#a3e635",
                    edgecolors="#365314",
                    linewidths=0.5,
                    label="detected expected",
                )
            if np.any(supported_unexpected):
                axis.scatter(
                    full_grid[supported_unexpected, 0],
                    full_grid[supported_unexpected, 1],
                    s=18.0 + 5.0 * np.sqrt(np.maximum(localization_counts[supported_unexpected], 0.0)),
                    marker="x",
                    color="#f43f5e",
                    linewidths=1.1,
                    label="detected in black space",
                )

            probabilities = params.get("classification_template_probabilities", ())
            raw_probabilities = params.get("classification_raw_template_probabilities", ())
            monte_carlo = params.get("classification_monte_carlo_posterior", ())
            scores = params.get("classification_scores", ())
            margins = params.get("classification_winner_margins", ())
            pattern_correlations = params.get("classification_cell_pattern_correlation", ())
            cell_agreements = params.get("classification_cell_agreement", ())
            crop_retention_values = params.get("classification_crop_retained_fraction", ())
            dispositions = params.get("classification_dispositions", ())
            probability = (
                float(probabilities[candidate_index])
                if candidate_index < len(probabilities)
                else float("nan")
            )
            raw_probability = (
                float(raw_probabilities[candidate_index])
                if candidate_index < len(raw_probabilities)
                else probability
            )
            mc_probability = (
                float(monte_carlo[candidate_index])
                if candidate_index < len(monte_carlo)
                else float("nan")
            )
            score = float(scores[candidate_index]) if candidate_index < len(scores) else float("nan")
            margin = float(margins[candidate_index]) if candidate_index < len(margins) else float("nan")
            pattern_correlation = (
                float(pattern_correlations[candidate_index])
                if candidate_index < len(pattern_correlations)
                else float("nan")
            )
            cell_agreement = (
                float(cell_agreements[candidate_index])
                if candidate_index < len(cell_agreements)
                else float("nan")
            )
            crop_retention = (
                float(crop_retention_values[candidate_index])
                if candidate_index < len(crop_retention_values)
                else float("nan")
            )
            crop_retention_title = (
                f" · retained {100.0 * crop_retention:.0f}%"
                if np.isfinite(crop_retention)
                else ""
            )
            disposition = (
                str(dispositions[candidate_index])
                if candidate_index < len(dispositions)
                else ""
            )
            assigned = bool(picks.accepted_mask[candidate_index])
            if probability > winning_probability:
                winning_probability = probability
                winning_name = name
            for spine in axis.spines.values():
                spine.set_color("#16a34a" if assigned else "#6b7280")
                spine.set_linewidth(2.4 if assigned else 0.8)
            margin_text = f" · margin {margin:.2f}" if np.isfinite(margin) else ""
            axis.set_title(
                f"{name}   class p {probability:.2f} · raw p {raw_probability:.2f} · MC {mc_probability:.2f}\n"
                f"score {score:.2f} · corr {picks.rectangle_confidence[candidate_index]:.2f} · "
                f"pattern {pattern_correlation:.2f} · cells {cell_agreement:.2f}"
                f"{crop_retention_title}"
                f"{margin_text}\n{disposition}",
                fontsize=8,
                color=palette(template_index % 10),
                fontweight="bold" if assigned else "normal",
            )
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])

        for axis in axes[panel_count:]:
            axis.set_axis_off()
        handles: list[Any] = []
        labels: list[str] = []
        for axis in axes[:panel_count]:
            panel_handles, panel_labels = axis.get_legend_handles_labels()
            for handle, label in zip(panel_handles, panel_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            self.origami_figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(3, len(handles)),
                fontsize=8,
                framealpha=0.9,
            )
        self.origami_figure.suptitle(
            f"Template competition for one candidate — green frame is the assigned winner\n"
            "Cyan rings: expected bright sites · green dots: supported expected sites · "
            "red ×: detected signal in template-black space",
            fontsize=10,
        )
        self.origami_match_panel_combo.state(["!disabled", "readonly"])
        self.origami_match_roi.set(selected_region_index + 1)
        self.origami_match_roi_label.set(
            f"Candidate {selected_region_index + 1}/{len(selected_picks.regions):,}"
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Identified origami match ROI"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(
            f"Compared {matched_panels}/{panel_count} template fits for this candidate; "
            f"largest equal-prior probability is {winning_name} ({winning_probability:.2f})."
        )

    def _plot_identified_origami_match_roi(self, region_index: int) -> None:
        """Show the exact candidate/template arrays and their correlation contributions."""
        active = self._active_origami_picks_and_params()
        if active is None:
            return
        picks, params = active
        if not picks.regions:
            return
        region_index = max(0, min(len(picks.regions) - 1, int(region_index)))
        self.origami_selected_match_index = region_index
        if (
            self.origami_match_panel.get() == "Template competition"
            and len(self.origami_multi_template_results) > 1
        ):
            self._plot_template_competition(picks, region_index)
            return
        rows = int(params.get("rows", self.origami_rows.get()))
        columns = int(params.get("columns", self.origami_columns.get()))
        spacing_x_nm = float(params.get("spacing_x_nm", self.origami_spacing_x_nm.get()))
        spacing_y_nm = float(params.get("spacing_y_nm", self.origami_spacing_y_nm.get()))
        grid = np.asarray(picks.template_points_nm, dtype=float)
        if not len(grid):
            grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
        overlay_grid = classified_template_overlay_points(grid, params)
        if self.origami_show_site_diagnostics.get():
            decision_overlay_grid = digital_group_decision_overlay_points(
                grid,
                params,
                region_index,
            )
            if decision_overlay_grid is not None:
                overlay_grid = decision_overlay_grid
        minimum_site_localizations = int(params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS))
        minimum_site_evidence = float(params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE))
        has_cached_site_evidence = (
            picks.site_localization_counts.shape == (len(picks.regions), len(grid))
            and picks.site_prominence.shape == (len(picks.regions), len(grid))
        )
        if has_cached_site_evidence:
            site_counts = picks.site_localization_counts[region_index]
            site_prominence = picks.site_prominence[region_index]
            peak_positions_nm = picks.site_peak_positions_nm[region_index]
            boundary_reference_positions_nm = picks.site_boundary_reference_positions_nm[region_index]
            boundary_points_nm = picks.site_boundary_points_nm[region_index]
        else:
            site_evidence = sparse_site_evidence_diagnostics(
                picks.aligned_regions[region_index],
                grid,
                site_radius_nm=float(picks.site_mask_radius_nm),
            )
            site_counts = site_evidence.counts
            site_prominence = site_evidence.prominence
            peak_positions_nm = site_evidence.peak_positions_nm
            boundary_reference_positions_nm = site_evidence.boundary_reference_positions_nm
            boundary_points_nm = site_evidence.boundary_points_nm
        site_decisions = [
            origami_site_decision_label(
                int(count),
                float(prominence),
                minimum_site_localizations,
                minimum_site_evidence,
            )
            for count, prominence in zip(site_counts, site_prominence)
        ]
        supported_sites = np.asarray([decision[0] for decision in site_decisions], dtype=bool)
        site_edge_colors = [decision[1] for decision in site_decisions]
        measured_site_centroids = picks.site_centroids_nm[region_index]
        measured_sites = np.all(np.isfinite(measured_site_centroids), axis=1)

        def draw_measured_assignments(target_axis: Any) -> None:
            for site_index in np.flatnonzero(measured_sites):
                measured = measured_site_centroids[site_index]
                theoretical = grid[site_index]
                target_axis.plot(
                    [measured[0], theoretical[0]],
                    [measured[1], theoretical[1]],
                    color="#a3e635",
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=4,
                )
            if np.any(measured_sites):
                target_axis.scatter(
                    measured_site_centroids[measured_sites, 0],
                    measured_site_centroids[measured_sites, 1],
                    s=22,
                    marker="o",
                    facecolors="#a3e635",
                    edgecolors="#14532d",
                    linewidths=0.7,
                    zorder=5,
                )

        def draw_site_diagnostics(target_axis: Any) -> None:
            if not self.origami_show_site_diagnostics.get():
                return
            digital_annotations = digital_group_decision_annotations(
                grid,
                params,
                region_index,
            )
            if digital_annotations is not None:
                PaintAnalysisApp._draw_digital_decision_blobs(
                    self,
                    target_axis,
                    grid,
                    params,
                    region_index,
                )
                for site, diagnostic, color in digital_annotations:
                    target_axis.annotate(
                        diagnostic,
                        xy=site,
                        xytext=(3, 4),
                        textcoords="offset points",
                        color=color,
                        fontsize=7,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        clip_on=True,
                        zorder=7,
                    )
                return
            for site_index, (site, decision) in enumerate(zip(grid, site_decisions), start=1):
                _supported, color, diagnostic = decision
                target_axis.annotate(
                    f"S{site_index} {diagnostic}",
                    xy=site,
                    xytext=(3, 4),
                    textcoords="offset points",
                    color=color,
                    fontsize=6,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    clip_on=True,
                    zorder=7,
                )

        def draw_group_assignments(target_axis: Any) -> None:
            toggle = getattr(
                self, "origami_show_localization_group_assignments", None
            )
            if toggle is None or not bool(toggle.get()):
                return
            aligned_regions = getattr(picks, "aligned_regions", ())
            if region_index < len(aligned_regions):
                PaintAnalysisApp._draw_digital_group_localization_assignments(
                    self,
                    target_axis,
                    aligned_regions[region_index],
                    grid,
                    params,
                )

        def draw_prominence_geometry(target_axis: Any) -> None:
            if not self.origami_show_prominence_geometry.get():
                return
            valid_peaks = np.all(np.isfinite(peak_positions_nm), axis=1)
            for site_index in np.flatnonzero(valid_peaks):
                boundary = boundary_points_nm[site_index]
                finite_boundary = np.all(np.isfinite(boundary), axis=1)
                if np.any(finite_boundary):
                    closed_boundary = np.vstack((boundary[finite_boundary], boundary[finite_boundary][0]))
                    target_axis.plot(
                        closed_boundary[:, 0],
                        closed_boundary[:, 1],
                        color="#22d3ee",
                        linewidth=0.65,
                        linestyle="--",
                        alpha=0.75,
                        zorder=5.6,
                    )
                peak = peak_positions_nm[site_index]
                reference = boundary_reference_positions_nm[site_index]
                if np.all(np.isfinite(reference)):
                    target_axis.plot(
                        [peak[0], reference[0]],
                        [peak[1], reference[1]],
                        color="#f472b6",
                        linewidth=0.55,
                        alpha=0.7,
                        zorder=5.7,
                    )
                    target_axis.scatter(
                        [reference[0]],
                        [reference[1]],
                        s=14,
                        marker="s",
                        facecolors="#f472b6",
                        edgecolors="#831843",
                        linewidths=0.5,
                        zorder=6.1,
                    )
            if np.any(valid_peaks):
                target_axis.scatter(
                    peak_positions_nm[valid_peaks, 0],
                    peak_positions_nm[valid_peaks, 1],
                    s=20,
                    marker="D",
                    facecolors="#22d3ee",
                    edgecolors="#164e63",
                    linewidths=0.6,
                    zorder=6.2,
                )
        width_nm = float(picks.rectangle_width_nm)
        height_nm = float(picks.rectangle_height_nm)
        candidate = np.asarray(picks.alignment_candidate_images[region_index], dtype=float)
        template = np.asarray(picks.alignment_reference_image, dtype=float)
        side_nm = float(
            getattr(picks, "alignment_canvas_side_nm", np.hypot(width_nm, height_nm))
        )
        image_extent = (-side_nm / 2.0, side_nm / 2.0, -side_nm / 2.0, side_nm / 2.0)
        candidate_norm = max(float(np.linalg.norm(candidate)), 1e-12)
        template_norm = max(float(np.linalg.norm(template)), 1e-12)
        normalized_candidate = candidate / candidate_norm
        normalized_template = template / template_norm
        contribution = normalized_candidate * normalized_template
        calculated_correlation = float(np.sum(contribution))

        candidate_positive = candidate - float(np.min(candidate))
        template_positive = template - float(np.min(template))
        candidate_positive /= max(float(np.max(candidate_positive)), 1e-12)
        template_positive /= max(float(np.max(template_positive)), 1e-12)
        pixel_height, pixel_width = candidate.shape
        x_centers = np.linspace(-side_nm / 2.0, side_nm / 2.0, pixel_width, endpoint=False) + side_nm / (2.0 * pixel_width)
        y_centers = np.linspace(-side_nm / 2.0, side_nm / 2.0, pixel_height, endpoint=False) + side_nm / (2.0 * pixel_height)
        mask_xx, mask_yy = np.meshgrid(x_centers, y_centers)
        mask_points = np.column_stack((mask_xx.ravel(), mask_yy.ravel()))
        nearest_site_distance, _nearest_site = cKDTree(grid).query(mask_points, k=1)
        inside_footprint = (
            (np.abs(mask_points[:, 0]) <= width_nm / 2.0)
            & (np.abs(mask_points[:, 1]) <= height_nm / 2.0)
        )
        positive_space = (inside_footprint & (nearest_site_distance <= picks.site_mask_radius_nm)).reshape(candidate.shape)
        negative_space = (inside_footprint & ~positive_space.ravel()).reshape(candidate.shape)
        space_overlay = np.zeros((*candidate.shape, 4), dtype=float)
        space_overlay[negative_space] = (1.0, 0.55, 0.0, 0.16)
        space_overlay[positive_space] = (0.0, 0.9, 1.0, 0.12)
        overlay = np.zeros((*candidate.shape, 3), dtype=float)
        overlay[..., 0] = candidate_positive
        overlay[..., 1] = template_positive
        overlay[..., 2] = template_positive
        overlay = np.clip(overlay, 0.0, 1.0)
        accepted = bool(picks.accepted_mask[region_index])
        decision = "ACCEPTED" if accepted else "REJECTED"
        decision_color = "#15803d" if accepted else "#b91c1c"

        self.origami_plot_option.set("Identified origami template matches")
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("none")
        canvas_widget = self.origami_canvas.get_tk_widget()
        canvas_width = max(1, int(canvas_widget.winfo_width()))
        canvas_height = max(1, int(canvas_widget.winfo_height()))
        self.origami_match_panel_combo.state(["!disabled", "readonly"])
        selected_panel = self.origami_match_panel.get()
        if canvas_width < 760 and selected_panel == "All panels":
            selected_panel = "Overlay"
            self.origami_match_panel.set(selected_panel)
        show_all_panels = selected_panel == "All panels"
        use_single_row = show_all_panels and canvas_width / canvas_height >= 2.4
        if use_single_row:
            all_axes = np.asarray(self.origami_figure.subplots(1, 4)).reshape(4)
            self.origami_figure.subplots_adjust(left=0.035, right=0.985, bottom=0.09, top=0.73, wspace=0.32)
            axes_by_panel = dict(zip(("Candidate", "Template", "Overlay", "Contributions"), all_axes))
        elif show_all_panels:
            all_axes = np.asarray(self.origami_figure.subplots(2, 2)).reshape(4)
            self.origami_figure.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.80, wspace=0.24, hspace=0.42)
            axes_by_panel = dict(zip(("Candidate", "Template", "Overlay", "Contributions"), all_axes))
        else:
            single_axis = self.origami_figure.subplots(1, 1)
            self.origami_figure.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.80)
            axes_by_panel = {selected_panel: single_axis}

        candidate_axis = axes_by_panel.get("Candidate")
        if candidate_axis is not None:
            candidate_axis.imshow(
                candidate_positive,
                extent=image_extent,
                origin="lower",
                cmap="magma",
                interpolation="nearest",
                aspect="equal",
                vmin=0.0,
                vmax=1.0,
            )
            candidate_axis.scatter(
                overlay_grid[:, 0], overlay_grid[:, 1], s=28,
                facecolors="none", edgecolors="#22d3ee", linewidths=0.9,
            )
            draw_measured_assignments(candidate_axis)
            draw_prominence_geometry(candidate_axis)
            draw_site_diagnostics(candidate_axis)
            draw_group_assignments(candidate_axis)
            candidate_axis.set_title(
                f"Aligned candidate: {decision}\n"
                f"{int(picks.point_counts[region_index]):,} points • {picks.rectangle_angles_deg[region_index]:.1f}°",
                color=decision_color,
                fontsize=9,
            )

        template_axis = axes_by_panel.get("Template")
        if template_axis is not None:
            template_axis.imshow(
                template_positive,
                extent=image_extent,
                origin="lower",
                cmap="magma",
                interpolation="nearest",
                aspect="equal",
                vmin=0.0,
                vmax=1.0,
            )
            template_axis.scatter(
                overlay_grid[:, 0], overlay_grid[:, 1], s=28,
                facecolors="none", edgecolors="#22d3ee", linewidths=0.9,
            )
            if params.get("template_mode") == "Custom image":
                template_title = f"Custom barcode template: {params.get('custom_template_name') or 'uploaded image'}"
                template_subtitle = f"{len(grid)} bright-component template sites"
            else:
                template_title = f"Simulated {rows} × {columns} grid template"
                template_subtitle = f"{spacing_x_nm:g} × {spacing_y_nm:g} nm site grid"
            template_axis.set_title(
                f"{template_title}\n{template_subtitle}",
                fontsize=9,
            )

        overlay_axis = axes_by_panel.get("Overlay")
        if overlay_axis is not None:
            overlay_axis.imshow(overlay, extent=image_extent, origin="lower", interpolation="nearest", aspect="equal")
            overlay_axis.imshow(space_overlay, extent=image_extent, origin="lower", interpolation="nearest", aspect="equal")
            overlay_axis.scatter(
                overlay_grid[:, 0], overlay_grid[:, 1], s=24,
                facecolors="none", edgecolors="#22d3ee", linewidths=0.9,
            )
            draw_measured_assignments(overlay_axis)
            draw_prominence_geometry(overlay_axis)
            draw_site_diagnostics(overlay_axis)
            draw_group_assignments(overlay_axis)
            overlay_axis.set_title(
                "Measured assignments + negative space\nfilled = measured • hollow = grid target • amber gaps",
                fontsize=9,
            )

        contribution_limit = max(float(np.max(np.abs(contribution))), 1e-12)
        contribution_axis = axes_by_panel.get("Contributions")
        if contribution_axis is not None:
            contribution_axis.imshow(
                contribution,
                extent=image_extent,
                origin="lower",
                cmap="PiYG",
                interpolation="nearest",
                aspect="equal",
                vmin=-contribution_limit,
                vmax=contribution_limit,
            )
            contribution_axis.set_title(
                "Pixel contributions\n"
                f"green raises • magenta lowers • sum = {calculated_correlation:.3f}",
                fontsize=9,
            )
        for axis in axes_by_panel.values():
            axis.tick_params(labelsize=7, length=2)
            axis.grid(False)
        self.origami_figure.suptitle(
            f"Candidate {region_index + 1}/{len(picks.regions)} • normalized template correlation = "
            f"Σ(candidate / ‖candidate‖ × template / ‖template‖) = {calculated_correlation:.3f}\n"
            f"site-gap contrast = {picks.site_gap_contrast[region_index]:.3f}; "
            f"on-site localizations = {100.0 * picks.on_site_fraction[region_index]:.1f}%; "
            f"supported sites = {picks.supported_site_count[region_index]}/{len(grid)} "
            f"across {picks.supported_row_count[region_index]}/{rows} rows and "
            f"{picks.supported_column_count[region_index]}/{columns} columns; "
            f"site-spacing RMS/max error = {picks.site_spacing_rms_nm[region_index]:.2f}/"
            f"{picks.site_spacing_max_error_nm[region_index]:.2f} nm; "
            f"grid-vs-blob ΔBIC = {picks.grid_vs_blob_delta_bic[region_index]:.1f}\n"
            "The first two panels are brightness-scaled for display; the contribution panel uses the exact scoring values.",
            fontsize=10,
            y=0.98,
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Identified origami match ROI"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(
            f"Showing candidate ROI {region_index + 1:,}/{len(picks.regions):,}: {decision.lower()}, "
            f"{int(picks.point_counts[region_index]):,} points, theoretical-template correlation "
            f"{picks.rectangle_confidence[region_index]:.3f}, {picks.supported_site_count[region_index]} supported sites, "
            f"maximum spacing error {picks.site_spacing_max_error_nm[region_index]:.2f} nm, "
            f"grid-vs-blob ΔBIC {picks.grid_vs_blob_delta_bic[region_index]:.1f}."
        )

    def _active_origami_picks_and_params(self) -> tuple[OrigamiPickResult, dict[str, Any]] | None:
        position = int(self.origami_roi_history_position)
        if 0 <= position < len(self.origami_roi_history):
            payload = self.origami_roi_history[position]
            return payload["picks"], dict(payload["identification_params"])
        if self.origami_pick_result is None:
            return None
        return self.origami_pick_result, dict(self.origami_identification_params or {})

    def _show_origami_coarse_density(self) -> None:
        self.origami_plot_option.set("Coarse identification density")
        self._plot_origami_coarse_density()

    def _plot_origami_coarse_density(self) -> None:
        """Show the cached picker density and its connected active-bin components."""
        history_position = int(self.origami_roi_history_position)
        history_payload = (
            self.origami_roi_history[history_position]
            if 0 <= history_position < len(self.origami_roi_history)
            else None
        )
        picks = history_payload["picks"] if history_payload is not None else self.origami_pick_result
        if picks is None and self.origami_staged_candidates is not None:
            staged_regions, density, contrast, extent, component_labels = (
                self.origami_staged_candidates
            )
            picks = SimpleNamespace(
                regions=staged_regions,
                density_image=density,
                density_contrast=contrast,
                density_extent_nm=extent,
                density_component_labels=component_labels,
                density_threshold=float(self.origami_min_density_contrast.get()),
            )
        if picks is None:
            messagebox.showinfo(
                "No identification density",
                "Run Step 1 once to build the coarse density map.",
            )
            return
        self.origami_match_panel_combo.state(["disabled"])
        params = (
            dict(history_payload["identification_params"])
            if history_payload is not None
            else self.origami_identification_params or {}
        )
        bin_size_nm = float(params.get("pick_bin_size_nm", self.origami_pick_bin_nm.get()))
        connect_distance_nm = float(params.get("connect_distance_nm", self.origami_connect_distance_nm.get()))
        x_min, x_max, y_min, y_max = picks.density_extent_nm
        density = np.asarray(picks.density_image, dtype=float)
        contrast = np.asarray(picks.density_contrast, dtype=float)
        component_labels = np.asarray(picks.density_component_labels, dtype=int)
        if density.size == 0 or contrast.size == 0 or component_labels.size == 0:
            messagebox.showinfo(
                "Tile-local coarse maps",
                "The distributed result combines spatial identifications from every selected tile, "
                "but its coarse candidate grids remain tile-local and cannot be displayed as one contiguous map. "
                "Choose Identified origami template matches to view all selected tiles together.",
            )
            return
        x_centers = np.linspace(x_min, x_max, density.shape[0], endpoint=False) + (x_max - x_min) / (2.0 * density.shape[0])
        y_centers = np.linspace(y_min, y_max, density.shape[1], endpoint=False) + (y_max - y_min) / (2.0 * density.shape[1])

        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("constrained", w_pad=6 / 72, h_pad=6 / 72)
        density_axis, component_axis = self.origami_figure.subplots(1, 2)
        density_image = density_axis.imshow(
            density.T,
            extent=picks.density_extent_nm,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        self.origami_figure.colorbar(
            density_image,
            ax=density_axis,
            shrink=0.82,
            label="Gaussian-smoothed source points / coarse bin",
        )
        if contrast.size and np.nanmin(contrast) <= picks.density_threshold <= np.nanmax(contrast):
            density_axis.contour(
                x_centers,
                y_centers,
                contrast.T,
                levels=[picks.density_threshold],
                colors=["#22d3ee"],
                linewidths=1.0,
            )
        density_axis.set_title(
            f"Smoothed coarse density ({bin_size_nm:g} nm bins)\n"
            f"cyan contour: contrast ≥ {picks.density_threshold:g}"
        )

        visible_components = np.ma.masked_less(component_labels.T, 0)
        component_count = int(np.max(component_labels) + 1) if np.any(component_labels >= 0) else 0
        component_cmap = matplotlib.colormaps["turbo"].copy()
        component_cmap.set_bad("#111827")
        component_axis.imshow(
            visible_components,
            extent=picks.density_extent_nm,
            origin="lower",
            cmap=component_cmap,
            vmin=0,
            vmax=max(1, component_count - 1),
            interpolation="nearest",
            aspect="equal",
        )
        if 0 < component_count <= 100:
            for component in range(component_count):
                bin_indices = np.argwhere(component_labels == component)
                if not len(bin_indices):
                    continue
                center_x = float(np.mean(x_centers[bin_indices[:, 0]]))
                center_y = float(np.mean(y_centers[bin_indices[:, 1]]))
                component_axis.text(
                    center_x,
                    center_y,
                    f"C{component + 1}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1},
                )
        component_axis.set_title(
            f"Active bins grouped into {component_count:,} candidates\n"
            f"bin centers within {connect_distance_nm:g} nm are connected"
        )
        for axis in (density_axis, component_axis):
            axis.set_xlabel("x position (nm)")
            axis.set_ylabel("y position (nm)")
            axis.grid(False)
        roi_label = "validation ROI" if history_payload is None else f"ROI {history_position + 2}"
        self.origami_figure.suptitle(
            f"Coarse identification map — {roi_label}; cached result, viewing does not rerun identification",
            fontsize=12,
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Coarse identification density"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        active_bin_count = int(np.count_nonzero(component_labels >= 0))
        self.status.set(
            f"Showing {active_bin_count:,} active coarse bins grouped into {component_count:,} candidate regions. "
            "Point, site-prominence, grid-coverage, spacing, and optional correlation tests occur after this candidate-seeding step; site-gap and ΔBIC are QC only."
        )

    def _draw_visible_origami_footprints(self, axis: Any, picks: OrigamiPickResult) -> tuple[int, int]:
        for artist in self.origami_footprint_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.origami_footprint_artists = []
        x0, x1 = sorted(float(value) for value in axis.get_xlim())
        y0, y1 = sorted(float(value) for value in axis.get_ylim())
        bounds = picks.bounds_nm
        visible = np.flatnonzero(
            (bounds[:, 1] >= x0) & (bounds[:, 0] <= x1) & (bounds[:, 3] >= y0) & (bounds[:, 2] <= y1)
        )
        total_visible = len(visible)
        show_labels = bool(self.origami_show_text_statistics.get())
        # Rejected poses are diagnostic results rather than identified
        # origamis. Keep every part of those fits hidden unless the user asks
        # for the text-statistics view that explains why they failed.
        if not show_labels:
            visible = visible[np.asarray(picks.accepted_mask[visible], dtype=bool)]
        displayed_visible = len(visible)
        maximum_outlines = 500
        if displayed_visible > maximum_outlines:
            visible = visible[np.linspace(0, displayed_visible - 1, maximum_outlines, dtype=int)]
        accepted_numbers = np.cumsum(picks.accepted_mask.astype(int))
        colors = matplotlib.colormaps.get_cmap("tab20")
        show_theoretical = bool(self.origami_show_theoretical_overlay.get())
        alignment_toggle = getattr(self, "origami_show_alignment_overlay", None)
        show_alignment = bool(alignment_toggle.get()) if alignment_toggle is not None else False
        show_detected_sites = bool(self.origami_show_detected_sites_overlay.get())
        show_site_diagnostics = bool(self.origami_show_site_diagnostics.get())
        assignment_toggle = getattr(
            self, "origami_show_localization_group_assignments", None
        )
        show_group_assignments = (
            bool(assignment_toggle.get()) if assignment_toggle is not None else False
        )
        prominence_requested = bool(self.origami_show_prominence_geometry.get())
        show_prominence_geometry = prominence_requested and displayed_visible <= 12
        params = dict(self.origami_identification_params or {})
        active = self._active_origami_picks_and_params()
        if active is not None and active[0] is picks:
            params = active[1]
        theoretical_grid = np.empty((0, 2), dtype=float)
        overlay_grid = np.empty((0, 2), dtype=float)
        theoretical_positions: list[np.ndarray] = []
        theoretical_colors: list[Any] = []
        alignment_positions: list[np.ndarray] = []
        detected_positions: list[np.ndarray] = []
        assignment_segments: list[np.ndarray] = []
        failed_site_positions: list[np.ndarray] = []
        failed_site_colors: list[str] = []
        site_diagnostic_annotations: list[tuple[np.ndarray, str, str]] = []
        prominence_boundaries: list[np.ndarray] = []
        prominence_links: list[np.ndarray] = []
        prominence_peaks: list[np.ndarray] = []
        prominence_references: list[np.ndarray] = []
        template_image = params.get("alignment_template_image")
        uses_custom_template = params.get("template_mode") == "Custom image" and template_image is not None
        show_site_diagnostic_text = show_site_diagnostics and displayed_visible <= 12
        show_digital_localization_assignments = (
            show_group_assignments and displayed_visible <= 50
        )
        if show_theoretical or show_alignment or show_detected_sites or show_site_diagnostics or show_group_assignments or show_prominence_geometry:
            theoretical_grid = np.asarray(picks.template_points_nm, dtype=float)
            if not len(theoretical_grid):
                theoretical_grid = ideal_grid_points(
                    int(params.get("rows", self.origami_rows.get())),
                    int(params.get("columns", self.origami_columns.get())),
                    float(params.get("spacing_x_nm", self.origami_spacing_x_nm.get())),
                    float(params.get("spacing_y_nm", self.origami_spacing_y_nm.get())),
                )
            overlay_grid = classified_template_overlay_points(theoretical_grid, params)
            alignment_grid = alignment_template_overlay_points(theoretical_grid, params)
            for region_index in visible:
                accepted = bool(picks.accepted_mask[region_index])
                accepted_number = int(accepted_numbers[region_index])
                color = colors((accepted_number - 1) % 20 / 19.0) if accepted else "#9ca3af"
                fitted_grid = theoretical_grid_in_footprint(
                    theoretical_grid,
                    picks.rectangle_corners_nm[region_index],
                )
                if show_theoretical:
                    displayed_overlay_grid = overlay_grid
                    if show_site_diagnostics:
                        decision_overlay_grid = digital_group_decision_overlay_points(
                            theoretical_grid,
                            params,
                            int(region_index),
                        )
                        if decision_overlay_grid is not None:
                            displayed_overlay_grid = decision_overlay_grid
                    theoretical_positions.append(
                        theoretical_grid_in_footprint(
                            displayed_overlay_grid,
                            picks.rectangle_corners_nm[region_index],
                        )
                    )
                    theoretical_colors.extend([color] * len(displayed_overlay_grid))
                if show_alignment and len(alignment_grid):
                    alignment_positions.append(
                        theoretical_grid_in_footprint(
                            alignment_grid,
                            picks.rectangle_corners_nm[region_index],
                        )
                    )
                if show_detected_sites or show_site_diagnostics or show_group_assignments or show_prominence_geometry:
                    minimum_site_localizations = int(params.get("min_site_localizations", DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS))
                    minimum_site_evidence = float(params.get("min_site_evidence", DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE))
                    site_counts = picks.site_localization_counts[region_index]
                    site_prominence = picks.site_prominence[region_index]
                    site_total = min(
                        len(fitted_grid),
                        len(site_counts),
                        len(site_prominence),
                        len(picks.site_centroids_nm[region_index]),
                    )
                    fitted_grid = fitted_grid[:site_total]
                    site_counts = site_counts[:site_total]
                    site_prominence = site_prominence[:site_total]
                    decisions = [
                        origami_site_decision_label(
                            int(count),
                            float(prominence),
                            minimum_site_localizations,
                            minimum_site_evidence,
                        )
                        for count, prominence in zip(site_counts, site_prominence)
                    ]
                    supported = np.asarray([decision[0] for decision in decisions], dtype=bool)
                    measured_centroids = picks.site_centroids_nm[region_index][:site_total]
                    measured = np.all(np.isfinite(measured_centroids), axis=1)
                    if show_detected_sites and np.any(measured):
                        measured_world = theoretical_grid_in_footprint(
                            measured_centroids[measured],
                            picks.rectangle_corners_nm[region_index],
                        )
                        detected_positions.append(measured_world)
                        assignment_segments.extend(
                            np.asarray((observed, target), dtype=float)
                            for observed, target in zip(measured_world, fitted_grid[measured])
                        )
                    if show_site_diagnostics or show_group_assignments:
                        digital_annotations = digital_group_decision_annotations(
                            theoretical_grid,
                            params,
                            int(region_index),
                        )
                        if digital_annotations is not None:
                            if show_site_diagnostics:
                                self.origami_footprint_artists.extend(
                                    PaintAnalysisApp._draw_digital_decision_blobs(
                                        self,
                                        axis,
                                        theoretical_grid,
                                        params,
                                        int(region_index),
                                        lambda points, corners=picks.rectangle_corners_nm[region_index]: theoretical_grid_in_footprint(
                                            points, corners
                                        ),
                                    )
                                )
                            aligned_regions = getattr(picks, "aligned_regions", ())
                            if show_digital_localization_assignments and region_index < len(aligned_regions):
                                self.origami_footprint_artists.extend(
                                    PaintAnalysisApp._draw_digital_group_localization_assignments(
                                        self,
                                        axis,
                                        aligned_regions[region_index],
                                        theoretical_grid,
                                        params,
                                        lambda points, corners=picks.rectangle_corners_nm[region_index]: theoretical_grid_in_footprint(
                                            points, corners
                                        ),
                                    )
                                )
                            if show_site_diagnostic_text:
                                for local_center, diagnostic, diagnostic_color in digital_annotations:
                                    target = theoretical_grid_in_footprint(
                                        local_center[None, :],
                                        picks.rectangle_corners_nm[region_index],
                                    )[0]
                                    site_diagnostic_annotations.append(
                                        (target, diagnostic, diagnostic_color)
                                    )
                        elif show_site_diagnostics:
                            for site_index, (target, decision) in enumerate(zip(fitted_grid, decisions), start=1):
                                site_supported, diagnostic_color, diagnostic = decision
                                if not site_supported:
                                    failed_site_positions.append(target)
                                    failed_site_colors.append(diagnostic_color)
                                if show_site_diagnostic_text:
                                    site_diagnostic_annotations.append(
                                        (target, f"S{site_index} {diagnostic}", diagnostic_color)
                                    )
                    if show_prominence_geometry:
                        peak_positions = picks.site_peak_positions_nm[region_index]
                        boundary_points = picks.site_boundary_points_nm[region_index]
                        boundary_references = picks.site_boundary_reference_positions_nm[region_index]
                        valid_peaks = np.all(np.isfinite(peak_positions), axis=1)
                        for site_index in np.flatnonzero(valid_peaks):
                            peak_world = theoretical_grid_in_footprint(
                                peak_positions[[site_index]],
                                picks.rectangle_corners_nm[region_index],
                            )[0]
                            boundary = boundary_points[site_index]
                            finite_boundary = np.all(np.isfinite(boundary), axis=1)
                            if np.any(finite_boundary):
                                boundary_world = theoretical_grid_in_footprint(
                                    boundary[finite_boundary],
                                    picks.rectangle_corners_nm[region_index],
                                )
                                prominence_boundaries.append(np.vstack((boundary_world, boundary_world[0])))
                            reference = boundary_references[site_index]
                            prominence_peaks.append(peak_world)
                            if np.all(np.isfinite(reference)):
                                reference_world = theoretical_grid_in_footprint(
                                    reference[None, :],
                                    picks.rectangle_corners_nm[region_index],
                                )[0]
                                prominence_references.append(reference_world)
                                prominence_links.append(np.vstack((peak_world, reference_world)))
        for region_index in visible:
            accepted = bool(picks.accepted_mask[region_index])
            accepted_number = int(accepted_numbers[region_index])
            corners = picks.rectangle_corners_nm[region_index]
            color = colors((accepted_number - 1) % 20 / 19.0) if accepted else "#9ca3af"
            if not uses_custom_template:
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
                self.origami_footprint_artists.append(rectangle)
            if show_labels:
                label_corner = corners[int(np.argmax(corners[:, 1]))]
                prefix = f"A{accepted_number}" if accepted else f"R{region_index + 1}"
                classification_scores = params.get("classification_scores", ())
                monte_carlo_posteriors = params.get("classification_monte_carlo_posterior", ())
                lattice_precisions = params.get("classification_lattice_precision", ())
                empty_cell_fractions = params.get("classification_empty_cell_fraction", ())
                pattern_correlations = params.get("classification_cell_pattern_correlation", ())
                crop_retention_values = params.get("classification_crop_retained_fraction", ())
                digital_pixel_ids = params.get("digital_pixel_ids", ())
                digital_pixel_probabilities = params.get("digital_pixel_probabilities", ())
                digital_pixel_text = ""
                if region_index < len(digital_pixel_probabilities):
                    digital_pixel_text = "; bits=" + ",".join(
                        f"{bit_id}:{float(probability):.2f}"
                        for bit_id, probability in zip(
                            digital_pixel_ids,
                            digital_pixel_probabilities[region_index],
                        )
                    )
                classification_score_text = (
                    f"; evidence={float(classification_scores[region_index]):.2f}"
                    if region_index < len(classification_scores)
                    else ""
                )
                evidence_abbreviation = (
                    "digital p"
                    if params.get("classification_evidence_model") in {"logical bits", "logical strokes", "direct digital groups"}
                    else "MC p"
                )
                monte_carlo_text = (
                    f"; {evidence_abbreviation}={float(monte_carlo_posteriors[region_index]):.2f}"
                    if region_index < len(monte_carlo_posteriors)
                    else ""
                )
                lattice_precision_text = (
                    f"; cell-match={float(lattice_precisions[region_index]):.2f}"
                    if region_index < len(lattice_precisions)
                    else ""
                )
                empty_cell_text = (
                    f"; black-empty={float(empty_cell_fractions[region_index]):.2f}"
                    if region_index < len(empty_cell_fractions)
                    else ""
                )
                pattern_text = (
                    f"; pattern={float(pattern_correlations[region_index]):.2f}"
                    if region_index < len(pattern_correlations)
                    else ""
                )
                crop_retention_text = (
                    f"; retained={100.0 * float(crop_retention_values[region_index]):.0f}%"
                    if region_index < len(crop_retention_values)
                    else ""
                )
                bic_text = (
                    f"{picks.grid_vs_blob_delta_bic[region_index]:.0f}"
                    if bool(params.get("compute_grid_blob_bic", True))
                    else "deferred"
                )
                annotation = axis.text(
                    label_corner[0],
                    label_corner[1],
                    f"{prefix}: n={picks.point_counts[region_index]:,}; {picks.rectangle_angles_deg[region_index]:.1f}°; "
                    f"corr={picks.rectangle_confidence[region_index]:.2f}; sites={picks.supported_site_count[region_index]}; "
                    f"spacing={picks.site_spacing_max_error_nm[region_index]:.1f}nm; "
                    f"gap={picks.site_gap_contrast[region_index]:.2f}; ΔBIC={bic_text}"
                    f"{lattice_precision_text}"
                    f"{empty_cell_text}"
                    f"{pattern_text}"
                    f"{crop_retention_text}"
                    f"{monte_carlo_text}"
                    f"{classification_score_text}"
                    f"{digital_pixel_text}",
                    color=color,
                    fontsize=7,
                    va="bottom",
                    ha="left",
                    clip_on=True,
                )
                annotation.set_in_layout(False)
                self.origami_footprint_artists.append(annotation)
                if not accepted:
                    failure_reasons = origami_candidate_failure_reasons(
                        point_count=int(picks.point_counts[region_index]),
                        correlation=float(picks.rectangle_confidence[region_index]),
                        supported_sites=int(picks.supported_site_count[region_index]),
                        supported_rows=int(picks.supported_row_count[region_index]),
                        supported_columns=int(picks.supported_column_count[region_index]),
                        spacing_error_nm=float(picks.site_spacing_max_error_nm[region_index]),
                        params=params,
                    )
                    dispositions = params.get("classification_dispositions", ())
                    disposition = (
                        str(dispositions[region_index])
                        if region_index < len(dispositions)
                        else ""
                    )
                    reassigned = not failure_reasons and disposition.startswith("classified as ")
                    failure_text = (
                        "OTHER TYPE: " + disposition.removeprefix("classified as ")
                        if reassigned
                        else "FAIL: " + "; ".join(failure_reasons or [disposition or "no template passed"])
                    )
                    failure_annotation = axis.text(
                        label_corner[0],
                        label_corner[1],
                        failure_text,
                        color="#f59e0b" if reassigned else "#ff3b30",
                        fontsize=7,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        clip_on=True,
                    )
                    failure_annotation.set_in_layout(False)
                    self.origami_footprint_artists.append(failure_annotation)
        if theoretical_positions:
            positions = np.vstack(theoretical_positions)
            site_overlay = axis.scatter(
                positions[:, 0],
                positions[:, 1],
                s=22,
                marker="o",
                facecolors="none",
                edgecolors=theoretical_colors,
                linewidths=0.9,
                alpha=0.95,
                zorder=4,
                label="Theoretical docking sites",
            )
            site_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(site_overlay)
        if alignment_positions:
            positions = np.vstack(alignment_positions)
            alignment_overlay = axis.scatter(
                positions[:, 0],
                positions[:, 1],
                s=28,
                marker="s",
                facecolors="none",
                edgecolors="#22d3ee",
                linewidths=1.0,
                alpha=0.95,
                zorder=4.2,
                label="Step 2 alignment template",
            )
            alignment_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(alignment_overlay)
        if failed_site_positions:
            positions = np.vstack(failed_site_positions)
            failed_overlay = axis.scatter(
                positions[:, 0],
                positions[:, 1],
                s=25,
                marker="x",
                c=failed_site_colors,
                linewidths=1.1,
                alpha=0.95,
                zorder=5.5,
                label="Unsupported site (color identifies failed gate)",
            )
            failed_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(failed_overlay)
        if prominence_boundaries:
            boundary_overlay = LineCollection(
                prominence_boundaries,
                colors="#22d3ee",
                linewidths=0.65,
                linestyles="dashed",
                alpha=0.75,
                zorder=5.6,
                label="Prominence boundary samples",
            )
            axis.add_collection(boundary_overlay)
            boundary_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(boundary_overlay)
        if prominence_links:
            reference_links = LineCollection(
                prominence_links,
                colors="#f472b6",
                linewidths=0.55,
                alpha=0.7,
                zorder=5.7,
            )
            axis.add_collection(reference_links)
            reference_links.set_in_layout(False)
            self.origami_footprint_artists.append(reference_links)
        if prominence_peaks:
            peaks = np.vstack(prominence_peaks)
            peak_overlay = axis.scatter(
                peaks[:, 0],
                peaks[:, 1],
                s=20,
                marker="D",
                facecolors="#22d3ee",
                edgecolors="#164e63",
                linewidths=0.6,
                zorder=6.2,
                label="Selected prominence peaks",
            )
            peak_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(peak_overlay)
        if prominence_references:
            references = np.vstack(prominence_references)
            reference_overlay = axis.scatter(
                references[:, 0],
                references[:, 1],
                s=14,
                marker="s",
                facecolors="#f472b6",
                edgecolors="#831843",
                linewidths=0.5,
                zorder=6.1,
                label="90th-percentile boundary reference",
            )
            reference_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(reference_overlay)
        for target, diagnostic, diagnostic_color in site_diagnostic_annotations:
            annotation = axis.annotate(
                diagnostic,
                xy=target,
                xytext=(3, 3),
                textcoords="offset points",
                color=diagnostic_color,
                fontsize=6,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=True,
                zorder=7,
            )
            annotation.set_in_layout(False)
            self.origami_footprint_artists.append(annotation)
        if prominence_requested and not show_prominence_geometry:
            annotation = axis.text(
                0.01,
                0.055 if show_site_diagnostics and not show_site_diagnostic_text else 0.01,
                "Zoom to 12 or fewer candidates to show prominence peaks and sampling rings.",
                transform=axis.transAxes,
                color="#67e8f9",
                fontsize=8,
                bbox={"facecolor": "#111827", "edgecolor": "#0891b2", "alpha": 0.85},
                ha="left",
                va="bottom",
                zorder=8,
            )
            annotation.set_in_layout(False)
            self.origami_footprint_artists.append(annotation)
        if show_site_diagnostics and not show_site_diagnostic_text:
            annotation = axis.text(
                0.01,
                0.01,
                "Zoom to 12 or fewer candidates for digital-group decision labels.",
                transform=axis.transAxes,
                color="#f8fafc",
                fontsize=8,
                bbox={"facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.85},
                ha="left",
                va="bottom",
                zorder=8,
            )
            annotation.set_in_layout(False)
            self.origami_footprint_artists.append(annotation)
        if show_group_assignments and not show_digital_localization_assignments:
            annotation = axis.text(
                0.01,
                0.045,
                "Zoom to 50 or fewer candidates to color individual localizations by digital group.",
                transform=axis.transAxes,
                color="#f8fafc",
                fontsize=8,
                bbox={"facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.85},
                ha="left",
                va="bottom",
                zorder=8,
            )
            annotation.set_in_layout(False)
            self.origami_footprint_artists.append(annotation)
        if detected_positions:
            positions = np.vstack(detected_positions)
            assignment_overlay = LineCollection(
                assignment_segments,
                colors="#a3e635",
                linewidths=0.7,
                alpha=0.75,
                zorder=4.5,
                label="Measured-to-grid assignments",
            )
            axis.add_collection(assignment_overlay)
            assignment_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(assignment_overlay)
            detected_overlay = axis.scatter(
                positions[:, 0],
                positions[:, 1],
                s=18,
                marker="o",
                facecolors="#a3e635",
                edgecolors="#14532d",
                linewidths=0.7,
                alpha=0.95,
                zorder=5,
                label="Measured supported-site centroids",
            )
            detected_overlay.set_in_layout(False)
            self.origami_footprint_artists.append(detected_overlay)
        return len(visible), total_visible

    def _on_origami_view_limits_changed(self, _axis: Any = None) -> None:
        self._schedule_origami_footprint_refresh()
        if not self.origami_zoom_render_applying:
            self._schedule_origami_zoom_render()

    def _schedule_origami_zoom_render(self, delay_ms: int = DYNAMIC_RENDER_DEBOUNCE_MS) -> None:
        if self.origami_last_rendered_plot_option not in {
            "Loaded source data",
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            return
        if self.loaded is None:
            return
        self.origami_zoom_render_request_id += 1
        if self.origami_zoom_render_after_id is not None:
            try:
                self.after_cancel(self.origami_zoom_render_after_id)
            except Exception:
                pass
        self.origami_zoom_render_after_id = self.after(
            max(0, int(delay_ms)),
            self._start_origami_zoom_render,
        )

    def _origami_home_viewport_nm(self) -> tuple[float, float, float, float] | None:
        """Return the stable initial extent for the active Origami source/ROI."""
        render_result: dict[str, Any] | None = None
        position = int(self.origami_roi_history_position)
        if 0 <= position < len(self.origami_roi_history):
            candidate = self.origami_roi_history[position].get("render_result")
            if isinstance(candidate, dict):
                render_result = candidate
        if render_result is None and self.origami_last_rendered_plot_option == "Random ROI inspection":
            payload = self.origami_random_inspection_payload
            candidate = payload.get("render_result") if isinstance(payload, dict) else None
            if isinstance(candidate, dict):
                render_result = candidate
        if render_result is None and isinstance(self.origami_source_render_result, dict):
            render_result = self.origami_source_render_result
        if render_result is None:
            points = self.origami_source_points_nm
            if points is None or not len(points):
                return None
            points = np.asarray(points, dtype=float)
            x0, y0 = np.min(points, axis=0)
            x1, y1 = np.max(points, axis=0)
            padding = max(1.0, 0.01 * max(x1 - x0, y1 - y0))
            return x0 - padding, x1 + padding, y0 - padding, y1 + padding
        extent = render_result.get("extent")
        if not isinstance(extent, (tuple, list, np.ndarray)) or len(extent) != 4:
            return None
        values = tuple(float(value) for value in extent)
        if not all(np.isfinite(value) for value in values):
            return None
        x0, x1, y0, y1 = values
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _active_origami_zoom_source(self) -> tuple[pd.DataFrame, dict[str, Any], int] | None:
        position = int(self.origami_roi_history_position)
        if 0 <= position < len(self.origami_roi_history):
            payload = self.origami_roi_history[position]
            locs = payload.get("locs")
            source_params = self.origami_loaded_source_params
            if isinstance(locs, pd.DataFrame) and source_params is not None:
                return locs, dict(source_params), position
        if self.origami_source_locs is None or self.origami_loaded_source_params is None:
            return None
        return self.origami_source_locs, dict(self.origami_loaded_source_params), -1

    def _start_origami_zoom_render(self) -> None:
        self.origami_zoom_render_after_id = None
        if (
            self.loaded is None
            or self.notebook.index(self.notebook.select()) != ORIGAMI_TAB
            or self.origami_last_rendered_plot_option
            not in {"Loaded source data", "Identified origami template matches", "Random ROI inspection"}
            or not self.origami_figure.axes
        ):
            return
        if self.origami_zoom_render_running:
            self.origami_zoom_render_pending = True
            return
        source = self._active_origami_zoom_source()
        if source is None:
            return
        locs, source_params, roi_position = source
        axis = self.origami_figure.axes[0]
        x0, x1 = (float(value) for value in axis.get_xlim())
        y0, y1 = (float(value) for value in axis.get_ylim())
        viewport = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        try:
            bounds = axis.get_window_extent()
            display_width = max(1.0, float(bounds.width))
            display_height = max(1.0, float(bounds.height))
            render_pixel_nm = float(
                f"{optimal_dynamic_render_pixel_nm(viewport, display_width, display_height):.6g}"
            )
        except ValueError:
            return
        request_id = int(self.origami_zoom_render_request_id)
        source_path = self.loaded.path
        info = self.loaded.info
        blur_method = str(source_params["render_blur_method"])
        min_blur_width = float(source_params["render_min_blur_width"])
        self.origami_zoom_render_running = True
        self.origami_zoom_render_pending = False
        render_subject = (
            "Loaded-source zoom render"
            if self.origami_last_rendered_plot_option == "Loaded source data"
            else "Origami zoom render"
        )
        self.status.set(f"{render_subject}: {render_pixel_nm:.3g} nm/pixel...")

        def worker() -> tuple[str, Any]:
            try:
                result = render_picasso_map(
                    locs,
                    info,
                    render_pixel_nm,
                    blur_method,
                    min_blur_width,
                    viewport,
                )
                viewport_min_density, viewport_max_density = histogram_density_limits(
                    np.asarray(result["image"], dtype=float)
                )
                result.update(
                    {
                        "min_density": viewport_min_density,
                        "max_density": viewport_max_density,
                        "source_path": source_path,
                        "request_id": request_id,
                        "roi_position": roi_position,
                    }
                )
                return "origami_dynamic_render", result
            except Exception as exc:
                return "origami_dynamic_render", {
                    "source_path": source_path,
                    "request_id": request_id,
                    "roi_position": roi_position,
                    "dynamic_error": str(exc),
                    "dynamic_error_details": traceback.format_exc(),
                }

        self._run_worker(worker)

    def _apply_origami_dynamic_render(self, render_result: dict[str, Any]) -> None:
        artist = self.origami_source_density_artist
        if artist is None or not self.origami_figure.axes:
            return
        raw_image = np.asarray(render_result["image"], dtype=float)
        contrast, density_limits = scale_density_like_picasso(
            raw_image,
            float(render_result["min_density"]),
            float(render_result["max_density"]),
        )
        extent = tuple(float(value) for value in render_result["extent"])
        axis = self.origami_figure.axes[0]
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        self.origami_zoom_render_applying = True
        try:
            artist.set_data(contrast)
            artist.set_extent(extent)
            axis.set_xlim(*xlim, emit=False)
            axis.set_ylim(*ylim, emit=False)
        finally:
            self.origami_zoom_render_applying = False
        pixel_x = (extent[1] - extent[0]) / max(1, raw_image.shape[1])
        pixel_y = (extent[3] - extent[2]) / max(1, raw_image.shape[0])
        if self.origami_source_colorbar is not None:
            self.origami_source_colorbar.set_label(
                f"density contrast ({density_limits[0]:.3g}-{density_limits[1]:.3g} locs/render px)"
            )
        title = axis.get_title()
        title = re.sub(
            r"(?:display|Picasso render) pixel [^;\n]+ nm",
            f"display pixel {pixel_x:.3g} × {pixel_y:.3g} nm",
            title,
        )
        axis.set_title(title)
        self.origami_canvas.draw_idle()
        render_subject = (
            "loaded source viewport"
            if self.origami_last_rendered_plot_option == "Loaded source data"
            else "visible origami ROI"
        )
        self.status.set(
            f"Rerendered the {render_subject} at {pixel_x:.3g} × {pixel_y:.3g} nm/pixel with "
            f"viewport density {density_limits[0]:.3g}–{density_limits[1]:.3g} locs/render pixel."
        )

    def _schedule_origami_footprint_refresh(self) -> None:
        if self.origami_last_rendered_plot_option not in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            return
        if self.origami_footprint_refresh_after_id is not None:
            try:
                self.after_cancel(self.origami_footprint_refresh_after_id)
            except Exception:
                pass
        self.origami_footprint_refresh_after_id = self.after(100, self._refresh_origami_footprints)

    def _refresh_origami_footprints(self) -> None:
        self.origami_footprint_refresh_after_id = None
        if (
            self.origami_last_rendered_plot_option
            not in {"Identified origami template matches", "Random ROI inspection"}
            or not self.origami_figure.axes
        ):
            return
        if self.origami_last_rendered_plot_option == "Random ROI inspection":
            payload = self.origami_random_inspection_payload
            picks = payload.get("picks") if payload is not None else None
        elif (
            self.origami_template_result_view.get() == "All templates"
            and self.origami_multi_template_results
        ):
            axis = self.origami_figure.axes[0]
            shown, visible = self._draw_all_template_classifications(axis)
            self.origami_canvas.draw_idle()
            self.status.set(
                f"Showing all template classifications together: {shown:,}/{visible:,} visible assigned fits; "
                f"{self.origami_multi_template_unclassified_count:,} unclassified candidates are gray × markers."
            )
            return
        else:
            picks = self.origami_pick_result
        if picks is None:
            return
        axis = self.origami_figure.axes[0]
        shown, visible = self._draw_visible_origami_footprints(axis, picks)
        self.origami_canvas.draw_idle()
        overlays: list[str] = []
        if self.origami_show_theoretical_overlay.get():
            overlays.append("classified-template overlay")
        if self.origami_show_alignment_overlay.get():
            overlays.append("Step 2 alignment-template overlay")
        if self.origami_show_detected_sites_overlay.get():
            overlays.append("detected sites")
        if self.origami_show_site_diagnostics.get():
            overlays.append("digital-group ON/OFF")
        assignment_toggle = getattr(
            self, "origami_show_localization_group_assignments", None
        )
        if assignment_toggle is not None and assignment_toggle.get():
            overlays.append("localization group assignments")
        if self.origami_show_prominence_geometry.get():
            overlays.append("prominence sampling")
        overlay_state = "with " + " and ".join(overlays) if overlays else "without site overlays"
        self.status.set(
            f"Showing {shown:,} of {visible:,} identified footprints {overlay_state}; "
            "zoom for additional local detail."
        )

    def _plot_origami_analysis(self, payload: dict[str, Any]) -> None:
        payload_generation = payload.get("identification_generation")
        if (
            payload_generation is not None
            and int(payload_generation) != int(self.origami_identification_generation)
        ):
            return
        result: OrigamiAnalysisResult = payload["result"]
        threshold = int(payload["occupancy_threshold"])
        result_template = payload.get("template_name")
        if result_template is None:
            self.origami_single_overlay_building = False
        if result_template in self.origami_multi_template_results:
            self.origami_multi_template_overlay_results[str(result_template)] = dict(payload)
            getattr(self, "origami_multi_template_overlays_building", set()).discard(result_template)
            if self.origami_template_result_view.get() != result_template:
                return
        self.origami_result = result
        self.origami_result_source = str(payload["source"])
        self.origami_result_source_count = int(payload["source_count"])
        self.origami_result_render_settings = dict(payload["render_settings"])
        self.origami_result_occupancy_threshold = threshold
        self.origami_gallery_view_limits = None
        self.origami_gallery_home_limits = None
        self.origami_gallery_page.set(1)
        self.origami_gallery_current_indices = np.empty(0, dtype=int)
        self.origami_gallery_tile_hitboxes = []
        self.origami_selected_index = None
        self.origami_density_cache_key = None
        self.origami_density_cache = None
        self.origami_last_rendered_plot_option = ""

        self.render_origami_plot()
        self.status.set(
            f"Overlaid all {result.origami_count} identified origamis using {result.clustering_method} and assigned "
            f"{sum(len(sites) for sites in result.cluster_site_indices)} docking-site groups; median alignment RMS "
            f"{np.median(result.alignment_rms_nm):.2f} nm and median grid match "
            f"{100.0 * np.median(result.grid_match_fraction):.1f}%."
        )

    def _origami_gallery_page_data(self) -> tuple[np.ndarray, int]:
        result = self.origami_result
        if result is None:
            return np.empty(0, dtype=int), 0
        min_match_text = self.origami_gallery_min_match.get().strip()
        max_rms_text = self.origami_gallery_max_rms.get().strip()
        min_match = float(min_match_text) / 100.0 if min_match_text else 0.0
        max_rms = float(max_rms_text) if max_rms_text else math.inf
        page_size = int(self.origami_gallery_page_size.get())
        indices = origami_gallery_indices(
            result,
            self.origami_gallery_sort.get(),
            min_grid_match_fraction=min_match,
            max_alignment_rms_nm=max_rms,
        )
        page_indices, page_number, page_count = origami_gallery_page(
            indices,
            int(self.origami_gallery_page.get()),
            page_size,
        )
        self.origami_gallery_page.set(page_number)
        self.origami_gallery_page_count = page_count
        start = (page_number - 1) * page_size + 1 if len(indices) else 0
        end = min(page_number * page_size, len(indices))
        self.origami_gallery_page_label.set(
            f"Page {page_number}/{page_count} • {start}-{end} of {len(indices):,} selected / {result.origami_count:,} total"
        )
        self.origami_gallery_current_indices = page_indices
        return page_indices, len(indices)

    def _apply_origami_gallery_view(self) -> None:
        self.origami_gallery_page.set(1)
        self.origami_gallery_view_limits = None
        if self.origami_plot_option.get() not in {"Individual origami gallery", "Individual site assignments"}:
            self.origami_plot_option.set("Individual origami gallery")
        self.render_origami_plot()

    def _change_origami_gallery_page(self, delta: int) -> None:
        try:
            current = int(self.origami_gallery_page.get())
        except (tk.TclError, ValueError):
            current = 1
        requested = current + int(delta) if delta else current
        page = max(1, min(int(self.origami_gallery_page_count), requested))
        if delta and page == current:
            self._configure_origami_navigation_controls()
            return
        self.origami_gallery_page.set(page)
        self.origami_gallery_view_limits = None
        if self.origami_plot_option.get() not in {"Individual origami gallery", "Individual site assignments"}:
            self.origami_plot_option.set("Individual origami gallery")
        self.render_origami_plot()

    def _show_origami_gallery(self) -> None:
        self.origami_plot_option.set("Individual origami gallery")
        self.render_origami_plot()

    def _on_origami_canvas_click(self, event: Any) -> None:
        if (
            event.button != 1
            or event.xdata is None
            or event.ydata is None
            or bool(getattr(self.origami_toolbar, "mode", ""))
        ):
            return
        if self.origami_last_rendered_plot_option in {
            "Identified origami template matches",
            "Random ROI inspection",
        }:
            active = self._active_origami_picks_and_params()
            if active is None:
                return
            picks, _params = active
            point = (float(event.xdata), float(event.ydata))
            containing = [
                index
                for index, corners in enumerate(picks.rectangle_corners_nm)
                if MatplotlibPath(corners).contains_point(point)
            ]
            if containing:
                selected = min(
                    containing,
                    key=lambda index: float(np.linalg.norm(np.mean(picks.rectangle_corners_nm[index], axis=0) - point)),
                )
                self._plot_identified_origami_match_roi(selected)
            return
        if self.origami_last_rendered_plot_option not in {
            "Individual origami gallery",
            "Individual site assignments",
        }:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        for x0, x1, y0, y1, result_index in self.origami_gallery_tile_hitboxes:
            if x0 <= x < x1 and y0 <= y < y1:
                self.origami_selected_index = result_index
                self.origami_plot_option.set("Selected origami detail")
                self.render_origami_plot()
                return

    def _cached_aligned_origami_density(
        self,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> dict[str, Any]:
        key = (
            id(result),
            int(render_settings["rows"]),
            int(render_settings["columns"]),
            float(render_settings["spacing_x_nm"]),
            float(render_settings["spacing_y_nm"]),
            float(render_settings["pixel_size_nm"]),
            float(render_settings["padding_nm"]),
            float(render_settings["blur_nm"]),
            bool(result.symmetrized_180),
        )
        if self.origami_density_cache_key != key or self.origami_density_cache is None:
            self.status.set(f"Building aggregate density from all {result.origami_count:,} origamis...")
            self.update_idletasks()
            self.origami_density_cache = render_aligned_origami_density(
                result.aligned_points,
                **render_settings,
                symmetrize_180=result.symmetrized_180,
            )
            self.origami_density_cache_key = key
        return self.origami_density_cache

    def render_origami_plot(self) -> None:
        option = self.origami_plot_option.get()
        self.origami_match_panel_combo.state(["disabled"])
        if option == "Origami type counts":
            self._plot_origami_type_counts()
            return
        if option == "Classification diagnostics":
            self._plot_classification_diagnostics()
            return
        if option == "Digital-group bias heatmap":
            self._plot_digital_group_bias_heatmap()
            return
        if option == "Coarse identification density":
            self._plot_origami_coarse_density()
            return
        if option == "Identified origami template matches":
            if (
                self.origami_template_result_view.get() == "All templates"
                and self.origami_multi_template_results
            ):
                self._plot_all_template_classifications()
                return
            if self.origami_pick_result is None or self.origami_source_points_nm is None:
                messagebox.showinfo("No identified origami", "Run Identify Origami before rendering the rectangle preview.")
                return
            self._plot_identified_origamis()
            return

        result = self.origami_result
        render_settings = self.origami_result_render_settings
        if result is None or render_settings is None:
            messagebox.showinfo("No origami overlay", "Build the fast overlay before rendering a result plot.")
            return

        gallery_options = {"Individual origami gallery", "Individual site assignments"}
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
            page_indices, _filtered_count = self._origami_gallery_page_data()
            self._plot_individual_origami_gallery(axis, result, render_settings, page_indices)
        elif option == "Individual site assignments":
            page_indices, _filtered_count = self._origami_gallery_page_data()
            self._plot_individual_origami_clusters(axis, result, render_settings, page_indices)
        elif option == "Selected origami detail":
            self._plot_selected_origami_detail(axis, result, render_settings)
        elif option == "Aligned density":
            overlay_render = self._cached_aligned_origami_density(result, render_settings)
            digital_grid = self._origami_digital_overlay_grid(result, render_settings)
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
            group_handles = self._draw_digital_group_blobs(axis, digital_grid)
            legend_handles = list(group_handles)
            if self._active_origami_digital_pixel_model() is None:
                legend_handles.insert(
                    0,
                    axis.scatter(
                        digital_grid[:, 0], digital_grid[:, 1], s=55,
                        facecolors="none", edgecolors="#22d3ee", linewidths=1.2,
                        label="analog sites", zorder=4,
                    ),
                )
            if legend_handles:
                axis.legend(
                    handles=legend_handles,
                    loc="upper right",
                    ncol=2 if len(group_handles) > 4 else 1,
                    fontsize=7,
                    framealpha=0.78,
                )
            effective_x = float(overlay_render["effective_pixel_x_nm"])
            effective_y = float(overlay_render["effective_pixel_y_nm"])
            pixel_text = f"{effective_x:.3g} nm/px" if math.isclose(effective_x, effective_y) else f"{effective_x:.3g} × {effective_y:.3g} nm/px"
            symmetry_text = "; 0°/180° equal-weight average" if result.symmetrized_180 else ""
            axis.set_title(
                f"Aligned density ({result.origami_count} origamis)\n"
                f"{pixel_text}; Gaussian σ={float(overlay_render['blur_nm']):.3g} nm; "
                f"{int(overlay_render['rendered_point_count'] / (2 if result.symmetrized_180 else 1)):,}/"
                f"{int(overlay_render['total_point_count'] / (2 if result.symmetrized_180 else 1)):,} physical source points in view"
                f"{symmetry_text}"
            )
            axis.set_xlabel("aligned x (nm)")
            axis.set_ylabel("aligned y (nm)")
        elif option == "Integrated density per site":
            overlay_render = self._cached_aligned_origami_density(result, render_settings)
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
                    f"Gaussian σ={float(overlay_render['blur_nm']):.3g} nm; includes unassigned points"
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
                "Site occupancy (0°/180° equal-weight assigned-group presence)",
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
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        if option in gallery_options:
            self.status.set(
                f"Rendered {len(self.origami_gallery_current_indices):,} thumbnails on {self.origami_gallery_page_label.get()}. "
                "Click a tile for full detail."
            )
        elif option == "Selected origami detail" and self.origami_selected_index is not None:
            self.status.set(f"Rendered detail for origami #{self.origami_selected_index + 1}. Click Back to Gallery to return.")
        else:
            self.status.set(f"Rendered {option.lower()} using all {result.origami_count:,} origamis.")

    def _plot_origami_type_counts(self) -> None:
        if not self.origami_multi_template_results:
            messagebox.showinfo(
                "No template classification",
                "Load multiple templates and run Identify Origami before viewing type counts.",
            )
            return
        names = list(self.origami_multi_template_results)
        counts = [int(self.origami_multi_template_counts.get(name, 0)) for name in names]
        labels = [*names, "Unclassified"]
        values = [*counts, int(self.origami_multi_template_unclassified_count)]
        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("constrained", w_pad=6 / 72, h_pad=6 / 72)
        axis = self.origami_figure.subplots(1, 1)
        colors = [matplotlib.colormaps["tab10"](index % 10) for index in range(len(names))]
        colors.append("#9ca3af")
        bars = axis.bar(np.arange(len(labels)), values, color=colors, edgecolor="white")
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
        probability_sums = np.asarray(
            [
                np.sum(
                    np.asarray(
                        self.origami_multi_template_results[name]
                        .get("params", {})
                        .get("classification_raw_template_probabilities", ()),
                        dtype=float,
                    )
                )
                for name in names
            ],
            dtype=float,
        )
        if np.any(probability_sums > 0.0):
            axis.scatter(
                np.arange(len(names)),
                probability_sums,
                marker="D",
                s=48,
                facecolors="white",
                edgecolors="black",
                linewidths=1.1,
                zorder=5,
                label="Σ P(template | candidate)",
            )
            for position, probability_sum in enumerate(probability_sums):
                axis.annotate(
                    f"Σp={probability_sum:.1f}",
                    (position, probability_sum),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )
            axis.legend(loc="upper right", fontsize=8)
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
        axis.set_ylabel("classified origami count")
        axis.set_title(
            f"Origami template classification ({sum(counts):,} assigned; "
            f"{self.origami_multi_template_unclassified_count:,} unclassified)\n"
            "Bars are hard assignments; diamonds sum equal-prior probabilities across all candidates"
        )
        axis.grid(True, axis="y", alpha=0.25)
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Origami type counts"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        self.status.set(
            "Template counts: "
            + ", ".join(f"{name}={count:,}" for name, count in zip(names, counts))
            + f"; unclassified={self.origami_multi_template_unclassified_count:,}. "
            "Choose a Classification view to inspect one type and build its separate overlays."
        )

    def _plot_digital_group_bias_heatmap(self) -> None:
        """Show aggregate digital-group ON probabilities by assignment cohort."""
        if not self.origami_multi_template_results:
            messagebox.showinfo(
                "No template classification",
                "Load digital-group templates and run Step 4 before viewing group bias.",
            )
            return
        names = list(self.origami_multi_template_results)
        first_payload = self.origami_multi_template_results[names[0]]
        first_params = dict(first_payload.get("params", {}))
        model = first_params.get("digital_pixel_model")
        if not isinstance(model, dict):
            model = first_params.get("logical_model")
        group_ids = tuple(str(value) for value in (model or {}).get("bit_ids", ()))
        probability_rows = np.asarray(
            first_params.get("digital_pixel_probabilities", ()), dtype=float
        )
        if (
            not group_ids
            or probability_rows.ndim != 2
            or probability_rows.shape[1] != len(group_ids)
        ):
            messagebox.showinfo(
                "No digital-group measurements",
                "Run Step 3 and Step 4 with a digital-pixel JSON model before viewing this heatmap.",
            )
            return

        candidate_count = probability_rows.shape[0]
        finite_probability = np.isfinite(probability_rows)
        group_evaluated_counts = np.sum(finite_probability, axis=0).astype(int)
        observed_on_counts = np.sum(
            finite_probability & (probability_rows >= 0.5), axis=0
        ).astype(int)
        cohort_names = ["All evaluated"]
        cohort_masks = [np.ones(candidate_count, dtype=bool)]
        assigned_any = np.zeros(candidate_count, dtype=bool)
        for name in names:
            picks = self.origami_multi_template_results[name].get("picks")
            mask = (
                np.asarray(picks.accepted_mask, dtype=bool)
                if picks is not None
                else np.empty(0, dtype=bool)
            )
            if mask.shape != (candidate_count,):
                continue
            cohort_names.append(name)
            cohort_masks.append(mask)
            assigned_any |= mask
        cohort_names.append("Not assigned")
        cohort_masks.append(~assigned_any)

        mean_probability = np.full((len(group_ids), len(cohort_masks)), np.nan)
        on_fraction = np.full_like(mean_probability, np.nan)
        sample_counts = np.zeros(len(cohort_masks), dtype=int)
        for cohort_index, mask in enumerate(cohort_masks):
            finite_rows = mask & np.any(np.isfinite(probability_rows), axis=1)
            sample_counts[cohort_index] = int(np.count_nonzero(finite_rows))
            if not np.any(finite_rows):
                continue
            values = probability_rows[finite_rows]
            with np.errstate(invalid="ignore"):
                mean_probability[:, cohort_index] = np.nanmean(values, axis=0)
                on_fraction[:, cohort_index] = np.nanmean(values >= 0.5, axis=0)

        expected_on = np.asarray(
            first_params.get("digital_group_expected_on_fractions", ()), dtype=float
        )
        expected_on_counts = np.full(len(group_ids), np.nan)
        expected_column_index: int | None = None
        if expected_on.shape == (len(group_ids),):
            expected_on_counts = expected_on * group_evaluated_counts
            cohort_names.insert(1, "Expected equal mix")
            mean_probability = np.insert(mean_probability, 1, expected_on, axis=1)
            on_fraction = np.insert(on_fraction, 1, expected_on, axis=1)
            sample_counts = np.insert(sample_counts, 1, 0)
            expected_column_index = 1

        self.origami_figure.clear()
        self.origami_figure.set_layout_engine(
            "constrained", w_pad=10 / 72, h_pad=8 / 72
        )
        axis = self.origami_figure.subplots(1, 1)
        image = axis.imshow(
            mean_probability,
            origin="upper",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        axis.set_xticks(
            np.arange(len(cohort_names)),
            labels=[
                (
                    f"{name}\nprior"
                    if name == "Expected equal mix"
                    else f"{name}\nn={sample_counts[index]:,}"
                )
                for index, name in enumerate(cohort_names)
            ],
            rotation=20,
            ha="right",
        )
        axis.set_yticks(np.arange(len(group_ids)), labels=group_ids)
        axis.set_xlabel("classification cohort")
        axis.set_ylabel("digital group")
        axis.set_title(
            "Digital-group state bias across evaluated origami\n"
            "Color = mean P(ON); text = hard ON rate (P ≥ 0.5)"
        )
        for group_index in range(len(group_ids)):
            for cohort_index in range(len(cohort_names)):
                probability = mean_probability[group_index, cohort_index]
                fraction = on_fraction[group_index, cohort_index]
                if not np.isfinite(probability):
                    label = "—"
                    text_color = "#374151"
                else:
                    label = f"p={probability:.2f}\n{100.0 * fraction:.0f}% ON"
                    if cohort_index == 0:
                        label += (
                            f"\nON={observed_on_counts[group_index]:,}/"
                            f"{group_evaluated_counts[group_index]:,}"
                        )
                    elif cohort_index == expected_column_index:
                        label += f"\nexpected={expected_on_counts[group_index]:.1f}"
                    text_color = "white" if probability < 0.25 or probability > 0.78 else "black"
                axis.text(
                    cohort_index,
                    group_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
        self.origami_figure.colorbar(
            image,
            ax=axis,
            label="mean digital-group P(ON)",
            pad=0.02,
            shrink=0.92,
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Digital-group bias heatmap"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        overall = mean_probability[:, 0]
        finite = np.flatnonzero(np.isfinite(overall))
        if len(finite):
            if np.any(np.isfinite(expected_on_counts)):
                differences = observed_on_counts - expected_on_counts
                largest = int(np.nanargmax(np.abs(differences)))
                self.status.set(
                    f"Digital-group ON totals across {sample_counts[0]:,} candidates: "
                    f"largest departure from the equal-template expectation is "
                    f"{group_ids[largest]} ({observed_on_counts[largest]:,} observed vs "
                    f"{expected_on_counts[largest]:.1f} expected; "
                    f"Δ {differences[largest]:+.1f})."
                )
            else:
                highest = int(finite[np.argmax(overall[finite])])
                lowest = int(finite[np.argmin(overall[finite])])
                self.status.set(
                    f"Digital-group bias across {sample_counts[0]:,} candidates: "
                    f"highest mean P(ON) is {group_ids[highest]} ({overall[highest]:.2f}); "
                    f"lowest is {group_ids[lowest]} ({overall[lowest]:.2f})."
                )
        else:
            self.status.set("No finite digital-group probabilities were available.")

    def _plot_classification_diagnostics(self) -> None:
        if not self.origami_multi_template_results:
            messagebox.showinfo(
                "No template classification",
                "Load multiple templates and run Identify Origami before viewing diagnostics.",
            )
            return
        names = list(self.origami_multi_template_results)
        diagnostics = classification_bias_diagnostics(
            names,
            [int(self.origami_multi_template_counts.get(name, 0)) for name in names],
            self.origami_multi_template_unclassified_details,
            int(self.origami_multi_template_unclassified_count),
        )
        assigned = np.asarray(diagnostics["assigned"], dtype=int)
        rejected = np.asarray(diagnostics["rejected_best"], dtype=int)
        rates = np.asarray(diagnostics["assignment_rates"], dtype=float)
        failures = np.asarray(diagnostics["failure_counts"], dtype=int)
        unresolved = int(diagnostics["unresolved_unclassified"])

        self.origami_figure.clear()
        self.origami_figure.set_layout_engine("constrained", w_pad=10 / 72, h_pad=8 / 72)
        dashboard = self.origami_figure.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.25),
            width_ratios=(1.08, 0.92),
        )
        count_axis = self.origami_figure.add_subplot(dashboard[0, 0])
        margin_axis = self.origami_figure.add_subplot(dashboard[0, 1])
        failure_axis = self.origami_figure.add_subplot(dashboard[1, :])
        positions = np.arange(len(names))
        palette = matplotlib.colormaps["tab10"]
        colors = [palette(index % 10) for index in range(len(names))]
        probability_sums = np.asarray(
            [
                np.sum(
                    np.asarray(
                        self.origami_multi_template_results[name]
                        .get("params", {})
                        .get("classification_template_probabilities", ()),
                        dtype=float,
                    )
                )
                for name in names
            ],
            dtype=float,
        )
        assigned_bars = count_axis.barh(
            positions,
            assigned,
            color=colors,
            edgecolor="white",
            label="Assigned",
        )
        rejected_bars = count_axis.barh(
            positions,
            rejected,
            left=assigned,
            color="#d1d5db",
            edgecolor="#6b7280",
            hatch="///",
            label="Best fit, rejected",
        )
        count_axis.scatter(
            probability_sums,
            positions,
            marker="D",
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=1.1,
            zorder=5,
            label="Σ raw model probability",
        )
        for position, probability_sum in zip(positions, probability_sums):
            count_axis.annotate(
                f"Σp {probability_sum:.1f}",
                (probability_sum, position),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        count_axis.bar_label(
            assigned_bars,
            labels=[f"{value:,}" if value else "" for value in assigned],
            label_type="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
        for bar, value, rate in zip(rejected_bars, rejected, rates):
            if value:
                count_axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"rej {value:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            count_axis.annotate(
                f"{100.0 * rate:.0f}% pass",
                (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color="#374151",
            )
        count_axis.set_yticks(positions, labels=names)
        count_axis.invert_yaxis()
        count_axis.set_xlabel("candidate equivalents")
        count_axis.set_title("Assignments and pre-gate model preference", fontsize=10)
        count_axis.legend(loc="lower right", fontsize=7, framealpha=0.9)
        count_axis.grid(True, axis="x", alpha=0.22)

        margin_values: list[np.ndarray] = []
        for name in names:
            payload = self.origami_multi_template_results[name]
            picks = payload.get("picks")
            params = dict(payload.get("params", {}))
            margins = np.asarray(params.get("classification_winner_margins", ()), dtype=float)
            accepted_mask = (
                np.asarray(picks.accepted_mask, dtype=bool)
                if picks is not None
                else np.zeros(len(margins), dtype=bool)
            )
            if len(margins) != len(accepted_mask):
                margin_values.append(np.empty(0, dtype=float))
            else:
                margin_values.append(margins[accepted_mask & np.isfinite(margins)])
        plotted_margins = [values if len(values) else np.asarray([np.nan]) for values in margin_values]
        boxes = margin_axis.boxplot(
            plotted_margins,
            positions=positions,
            widths=0.55,
            vert=False,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(colors[index])
            box.set_alpha(0.7)
        margin_axis.axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1.0)
        margin_axis.set_yticks(positions, labels=names)
        margin_axis.invert_yaxis()
        margin_axis.set_xlabel("winner − runner-up evidence")
        margin_axis.set_title("Assignment certainty (near 0 = ambiguous)", fontsize=10)
        margin_axis.grid(True, axis="x", alpha=0.22)

        rejection_denominator = np.maximum(rejected[:, None], 1)
        failure_percent = 100.0 * failures / rejection_denominator
        image = failure_axis.imshow(
            failure_percent.T,
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=max(1.0, float(np.max(failure_percent))),
            aspect="auto",
        )
        failure_axis.set_xticks(positions, labels=names)
        failure_axis.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=3,
        )
        failure_axis.set_yticks(
            np.arange(len(CLASSIFICATION_DIAGNOSTIC_GATES)),
            labels=CLASSIFICATION_DIAGNOSTIC_GATES,
        )
        failure_axis.set_title("Why best-fit candidates were rejected (count · %)", fontsize=10, pad=8)
        failure_axis.set_xlabel("A candidate may fail multiple gates.", fontsize=8)
        for template_index in range(len(names)):
            for gate_index in range(len(CLASSIFICATION_DIAGNOSTIC_GATES)):
                count = int(failures[template_index, gate_index])
                if count:
                    failure_axis.text(
                        template_index,
                        gate_index,
                        f"{count} · {failure_percent[template_index, gate_index]:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if failure_percent[template_index, gate_index] < 55 else "black",
                    )
        self.origami_figure.colorbar(
            image,
            ax=failure_axis,
            label="rejected best fits (%)",
            pad=0.015,
            shrink=0.92,
        )
        self.origami_figure.suptitle(
            f"Classification diagnostics: {int(np.sum(assigned)):,} assigned · "
            f"{int(np.sum(rejected)):,} diagnosed unclassified"
            f"{f'; {unresolved:,} without a recorded best-template diagnosis' if unresolved else ''}",
            fontsize=12,
        )
        self.origami_canvas.draw_idle()
        self.origami_toolbar.update()
        self.origami_last_rendered_plot_option = "Classification diagnostics"
        self._configure_origami_navigation_controls()
        self.notebook.select(ORIGAMI_TAB)
        lowest_index = int(np.argmin(rates)) if len(rates) else 0
        highest_probability_index = int(np.argmax(probability_sums)) if len(probability_sums) else 0
        total_probability = float(np.sum(probability_sums))
        probability_share = (
            100.0 * probability_sums[highest_probability_index] / total_probability
            if total_probability > 0.0
            else 0.0
        )
        self.status.set(
            f"Classification diagnostics: lowest best-template pass rate is "
            f"{names[lowest_index]} at {100.0 * rates[lowest_index]:.1f}%. "
            f"Largest summed raw model probability is {names[highest_probability_index]} "
            f"({probability_share:.1f}% of total probability mass)."
        )

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

    def _active_origami_digital_pixel_model(self) -> dict[str, Any] | None:
        params = getattr(self, "origami_identification_params", None)
        if not isinstance(params, dict):
            params = {}
        # Resolve directly from the selected classification payload as well.
        # This matters after tiled runs, where the shared run parameters may
        # temporarily replace the per-template parameters used by a result
        # plot while the template selector still identifies the active type.
        template_view_variable = self.__dict__.get("origami_template_result_view")
        try:
            selected_name = str(template_view_variable.get()) if template_view_variable is not None else ""
        except (AttributeError, tk.TclError):
            selected_name = ""
        selected_payload = self.__dict__.get("origami_multi_template_results", {}).get(
            selected_name
        )
        selected_params = (
            selected_payload.get("params", {})
            if isinstance(selected_payload, dict)
            else {}
        )
        # A selected classification template carries the same group schema as
        # Step 3 plus its own active-bit combination. Prefer that model for
        # result plots; the schema-only model would incorrectly show all bits.
        model = selected_params.get("logical_model")
        if not isinstance(model, dict):
            model = params.get("logical_model")
        if not isinstance(model, dict):
            model = selected_params.get("digital_pixel_model")
        if not isinstance(model, dict):
            model = params.get("digital_pixel_model")
        return model if isinstance(model, dict) else None

    def _origami_digital_overlay_grid(
        self,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> np.ndarray:
        """Return the complete analog lattice used by the custom digital groups."""
        grid = np.asarray(result.grid_points_nm, dtype=float)
        model = self._active_origami_digital_pixel_model()
        if not isinstance(model, dict):
            return grid
        try:
            rows, columns = (int(value) for value in model["physical_shape"])
            if len(grid) != rows * columns:
                grid = ideal_grid_points(
                    rows,
                    columns,
                    float(render_settings["spacing_x_nm"]),
                    float(render_settings["spacing_y_nm"]),
                )
        except (KeyError, TypeError, ValueError):
            pass
        return grid

    def _draw_digital_group_blobs(
        self,
        axis: Any,
        grid_points_nm: np.ndarray,
        transforms: list[Callable[[np.ndarray], np.ndarray]] | None = None,
        *,
        linewidth: float = 1.1,
    ) -> list[Any]:
        """Draw colored group envelopes while leaving analog site markers visible."""
        model = self._active_origami_digital_pixel_model()
        display_params = dict(self.origami_identification_params or {})
        outlines = digital_group_blob_contours(
            grid_points_nm,
            model,
            active_only=True,
            radius_nm=(
                float(display_params["site_mask_radius_nm"])
                if display_params.get("site_mask_radius_nm") is not None
                else None
            ),
        )
        if not outlines:
            return []
        if transforms is None:
            transforms = [lambda points: np.asarray(points, dtype=float)]
        palette = matplotlib.colormaps["tab10"]
        all_group_ids = tuple(str(value) for value in (model or {}).get("bit_ids", ()))
        legend_handles: list[Any] = []
        for visible_index, (group_id, contours) in enumerate(outlines):
            group_index = all_group_ids.index(group_id) if group_id in all_group_ids else visible_index
            color = palette(group_index % 10)
            polygons = [
                np.asarray(transform(contour), dtype=float)
                for transform in transforms
                for contour in contours
            ]
            if not polygons:
                continue
            collection = PolyCollection(
                polygons,
                closed=True,
                facecolors=[(*color[:3], 0.12)],
                edgecolors=[(*color[:3], 0.95)],
                linewidths=linewidth,
                zorder=3.5,
            )
            axis.add_collection(collection)
            legend_handles.append(
                matplotlib.patches.Patch(
                    facecolor=(*color[:3], 0.12),
                    edgecolor=(*color[:3], 0.95),
                    linewidth=linewidth,
                    label=group_id,
                )
            )
        return legend_handles

    def _draw_digital_decision_blobs(
        self,
        axis: Any,
        grid_points_nm: np.ndarray,
        params: dict[str, Any],
        candidate_index: int,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> list[Any]:
        """Shade groups by identity; encode inferred state with opacity and line style."""
        transform = transform or (lambda points: np.asarray(points, dtype=float))
        artists: list[Any] = []
        for _group_id, contours, color, alpha, linestyle in digital_group_decision_blobs(
            grid_points_nm,
            params,
            candidate_index,
        ):
            polygons = [np.asarray(transform(contour), dtype=float) for contour in contours]
            if not polygons:
                continue
            collection = PolyCollection(
                polygons,
                closed=True,
                facecolors=[matplotlib.colors.to_rgba(color, alpha)],
                edgecolors=[matplotlib.colors.to_rgba(color, 0.95)],
                linewidths=1.0,
                linestyles=linestyle,
                zorder=5.8,
            )
            axis.add_collection(collection)
            collection.set_in_layout(False)
            artists.append(collection)
        return artists

    def _draw_digital_group_localization_assignments(
        self,
        axis: Any,
        aligned_region: np.ndarray,
        grid_points_nm: np.ndarray,
        params: dict[str, Any],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> list[Any]:
        """Color localizations by the Step 3 digital groups they support.

        This deliberately reuses the exact fractional assignment calculation
        used to build ``digital_group_localization_evidence``.  A localization
        assigned exclusively to one group is a filled marker.  When it
        contributes to several overlapping groups, concentric colored rings
        show every contribution without allowing the last group drawn to hide
        the others.
        """
        model = params.get("digital_pixel_model")
        if not isinstance(model, dict):
            model = params.get("logical_model")
        if not isinstance(model, dict):
            return []
        group_ids = tuple(str(value) for value in model.get("bit_ids", ()))
        group_cells = tuple(
            model.get("bit_physical_cells", model.get("bit_cells", ()))
        )
        if not group_ids or len(group_ids) != len(group_cells):
            return []
        source_points = np.asarray(aligned_region, dtype=float)
        if source_points.ndim != 2 or source_points.shape[1:] != (2,) or not len(source_points):
            return []
        points = source_points[np.all(np.isfinite(source_points), axis=1)]
        if not len(points):
            return []
        source_grid = np.asarray(grid_points_nm, dtype=float)
        grid = source_grid
        try:
            rows, columns = (
                int(value) for value in model.get("physical_shape", (0, 0))
            )
            if rows * columns != len(grid):
                grid = ideal_grid_points(
                    rows,
                    columns,
                    float(params["spacing_x_nm"]),
                    float(params["spacing_y_nm"]),
                )
            radius_nm = float(params["site_mask_radius_nm"])
            group_signature = tuple(
                tuple(sorted({int(value) for value in cells}))
                for cells in group_cells
            )
            cache_key = (
                id(source_points),
                source_points.shape,
                id(source_grid),
                source_grid.shape,
                radius_nm,
                float(params["spacing_x_nm"]),
                float(params["spacing_y_nm"]),
                group_signature,
            )
            cache = self.__dict__.get("_origami_digital_assignment_display_cache")
            if not isinstance(cache, dict):
                cache = {}
                self._origami_digital_assignment_display_cache = cache
            cached = cache.get(cache_key)
            if (
                cached is not None
                and cached[0] is source_points
                and cached[1] is source_grid
            ):
                points, responsibilities = cached[2], cached[3]
            else:
                responsibilities = digital_group_localization_responsibilities(
                    points,
                    grid,
                    group_cells,
                    assignment_radius_nm=radius_nm,
                )
                cache[cache_key] = (
                    source_points,
                    source_grid,
                    points,
                    responsibilities,
                )
                while len(cache) > 256:
                    cache.pop(next(iter(cache)))
        except (KeyError, TypeError, ValueError):
            return []
        plotted = np.asarray((transform or (lambda value: value))(points), dtype=float)
        memberships = responsibilities > 1e-12
        membership_counts = np.sum(memberships, axis=1)
        artists: list[Any] = []

        # Exclusive evidence is a compact filled marker in its group color.
        for group_index in range(len(group_ids)):
            selected = memberships[:, group_index] & (membership_counts == 1)
            if not np.any(selected):
                continue
            color = digital_group_display_color(group_index)
            artist = axis.scatter(
                plotted[selected, 0],
                plotted[selected, 1],
                s=9,
                marker="o",
                facecolors=[matplotlib.colors.to_rgba(color, 0.82)],
                edgecolors=[matplotlib.colors.to_rgba(color, 1.0)],
                linewidths=0.25,
                zorder=6.35,
            )
            artist.set_in_layout(False)
            artists.append(artist)

        # Shared evidence uses one concentric ring per contributing group.
        # Draw weaker responsibilities farther outside and the strongest one
        # nearest the localization, so no group is overwritten.
        shared_indices = np.flatnonzero(membership_counts > 1)
        ring_batches: dict[tuple[int, int], list[int]] = {}
        for point_index in shared_indices:
            group_indices = np.flatnonzero(memberships[point_index])
            ordered = sorted(
                (int(value) for value in group_indices),
                key=lambda value: (float(responsibilities[point_index, value]), -value),
            )
            for outer_rank, group_index in enumerate(ordered):
                # Rank zero is the outermost ring.  Marker size is an area, so
                # a generous increment is needed to keep nested edges visible.
                ring_batches.setdefault((group_index, len(ordered) - outer_rank), []).append(
                    int(point_index)
                )
        for (group_index, size_rank), indices in sorted(
            ring_batches.items(), key=lambda item: -item[0][1]
        ):
            selected = np.asarray(indices, dtype=int)
            color = digital_group_display_color(group_index)
            artist = axis.scatter(
                plotted[selected, 0],
                plotted[selected, 1],
                s=8.0 + 10.0 * size_rank,
                marker="o",
                facecolors="none",
                edgecolors=[matplotlib.colors.to_rgba(color, 0.95)],
                linewidths=0.8,
                zorder=6.3 + 0.002 * size_rank,
            )
            artist.set_in_layout(False)
            artists.append(artist)
        return artists

    def _plot_individual_origami_gallery(
        self,
        axis: Any,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
        page_indices: np.ndarray,
    ) -> None:
        self.origami_gallery_tile_hitboxes = []
        if not len(page_indices):
            axis.text(0.5, 0.5, "No origamis pass the gallery quality filters.", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return
        settings = dict(render_settings)
        digital_grid = self._origami_digital_overlay_grid(result, settings)
        field_width_nm, field_height_nm, preview_pixel_nm, tile_width, tile_height, columns, rows, gap = (
            self._origami_gallery_geometry(result, settings, len(page_indices))
        )
        settings["pixel_size_nm"] = preview_pixel_nm
        rendered = [
            render_aligned_origami_density(
                [points],
                **settings,
            )
            for points in (result.aligned_points[index] for index in page_indices)
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
        grid_x = (digital_grid[:, 0] - x_min) * tile_width / (x_max - x_min)
        grid_y = (y_max - digital_grid[:, 1]) * tile_height / (y_max - y_min)
        show_analog_sites = self._active_origami_digital_pixel_model() is None
        expected_x: list[float] = []
        expected_y: list[float] = []
        for page_offset, ((x0, y0), result_index) in enumerate(zip(tile_origins, page_indices)):
            if show_analog_sites:
                expected_x.extend((x0 + grid_x).tolist())
                expected_y.extend((y0 + grid_y).tolist())
            self.origami_gallery_tile_hitboxes.append(
                (float(x0), float(x0 + tile_width), float(y0), float(y0 + tile_height), int(result_index))
            )
            label = axis.text(
                x0 + 2,
                y0 + 2,
                f"#{int(result_index) + 1}  n={result.source_point_counts[result_index]:,}",
                color="white",
                fontsize=6,
                va="top",
                ha="left",
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.5},
            )
            label.set_clip_path(axis.patch)
            label.set_in_layout(False)
        transforms: list[Callable[[np.ndarray], np.ndarray]] = []
        for x0, y0 in tile_origins:
            def to_gallery(points: np.ndarray, x0: int = x0, y0: int = y0) -> np.ndarray:
                converted = np.empty_like(np.asarray(points, dtype=float))
                converted[:, 0] = x0 + (points[:, 0] - x_min) * tile_width / (x_max - x_min)
                converted[:, 1] = y0 + (y_max - points[:, 1]) * tile_height / (y_max - y_min)
                return converted
            transforms.append(to_gallery)
        group_handles = self._draw_digital_group_blobs(
            axis,
            digital_grid,
            transforms,
            linewidth=0.65,
        )
        if show_analog_sites:
            axis.scatter(expected_x, expected_y, s=9, facecolors="none", edgecolors="#22d3ee", linewidths=0.5)
        if group_handles:
            axis.legend(
                handles=group_handles,
                loc="lower center",
                ncol=min(6, len(group_handles)),
                fontsize=6,
                framealpha=0.75,
            )
        axis.set_title(
            f"Individual aligned origami — {self.origami_gallery_page_label.get()}\n"
            + (
                "Click a tile for full detail; analog sites are outlined"
                if show_analog_sites
                else "Click a tile for full detail; overlays show active digital groups only"
            )
        )
        axis.set_axis_off()

    def _origami_gallery_geometry(
        self,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
        item_count: int | None = None,
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
        count = result.origami_count if item_count is None else max(1, int(item_count))
        columns = max(1, int(np.ceil(np.sqrt(count * tile_height / tile_width))))
        rows = int(np.ceil(count / columns))
        return width_nm, height_nm, preview_pixel_nm, tile_width, tile_height, columns, rows, 3

    def _plot_individual_origami_clusters(
        self,
        axis: Any,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
        page_indices: np.ndarray,
    ) -> None:
        self.origami_gallery_tile_hitboxes = []
        if not len(page_indices):
            axis.text(0.5, 0.5, "No origamis pass the gallery quality filters.", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return
        settings = dict(render_settings)
        digital_grid = self._origami_digital_overlay_grid(result, settings)
        width_nm, height_nm, _preview_pixel_nm, tile_width, tile_height, columns, rows, gap = (
            self._origami_gallery_geometry(result, settings, len(page_indices))
        )
        gallery = np.ones(
            (rows * tile_height + (rows - 1) * gap, columns * tile_width + (columns - 1) * gap, 3),
            dtype=float,
        )
        x_min, x_max = -width_nm / 2.0, width_nm / 2.0
        y_min, y_max = -height_nm / 2.0, height_nm / 2.0
        site_count = len(digital_grid)
        site_colors = matplotlib.colormaps["tab20"](np.linspace(0.0, 1.0, max(site_count, 2)))[:, :3]
        digital_model = self._active_origami_digital_pixel_model()
        digital_group_ids, site_memberships = digital_group_site_memberships(
            digital_model,
            site_count,
            active_only=True,
        )
        digital_group_palette = matplotlib.colormaps["tab10"]
        all_digital_group_ids = tuple(
            str(value) for value in (digital_model or {}).get("bit_ids", ())
        )
        digital_group_colors = np.asarray(
            [
                digital_group_palette(
                    (all_digital_group_ids.index(group_id)
                     if group_id in all_digital_group_ids else index) % 10
                )[:3]
                for index, group_id in enumerate(digital_group_ids)
            ],
            dtype=float,
        ).reshape(-1, 3)
        show_analog_sites = digital_model is None
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

        for page_offset, result_index in enumerate(page_indices):
            points = result.aligned_points[result_index]
            labels = result.cluster_labels[result_index]
            centers = result.cluster_centers_nm[result_index]
            sites = result.cluster_site_indices[result_index]
            row, column = divmod(page_offset, columns)
            y0 = row * (tile_height + gap)
            x0 = column * (tile_width + gap)
            self.origami_gallery_tile_hitboxes.append(
                (float(x0), float(x0 + tile_width), float(y0), float(y0 + tile_height), int(result_index))
            )
            tile = np.zeros((tile_height, tile_width, 3), dtype=float)
            detected_group_indices = {
                group_index
                for site_index in sites
                if 0 <= int(site_index) < len(site_memberships)
                for group_index in site_memberships[int(site_index)]
            }
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

                if digital_group_ids:
                    color_sum = np.zeros_like(tile)
                    intensity_sum = np.zeros((tile_height, tile_width), dtype=float)
                    for group_index in range(len(digital_group_ids)):
                        cluster_ids = [
                            cluster_label
                            for cluster_label, site_index in enumerate(oriented_sites)
                            if 0 <= int(site_index) < len(site_memberships)
                            and group_index in site_memberships[int(site_index)]
                        ]
                        if not cluster_ids:
                            continue
                        members = inside & np.isin(labels, cluster_ids)
                        if not np.any(members):
                            continue
                        flat = y_pixels[members] * tile_width + x_pixels[members]
                        density = np.bincount(
                            flat, minlength=tile_height * tile_width
                        ).reshape(tile_height, tile_width)
                        peak = float(np.max(density))
                        if peak > 0:
                            intensity = np.sqrt(density / peak)
                            color_sum += intensity[..., None] * digital_group_colors[group_index]
                            intensity_sum += intensity
                    has_group_signal = intensity_sum > 0.0
                    if np.any(has_group_signal):
                        mixed_color = color_sum[has_group_signal] / intensity_sum[has_group_signal, None]
                        brightness = np.clip(intensity_sum[has_group_signal], 0.0, 1.0)[:, None]
                        tile[has_group_signal] = np.maximum(
                            tile[has_group_signal],
                            mixed_color * brightness,
                        )
                    outside_cluster_ids = [
                        cluster_label
                        for cluster_label, site_index in enumerate(oriented_sites)
                        if not (
                            0 <= int(site_index) < len(site_memberships)
                            and site_memberships[int(site_index)]
                        )
                    ]
                    if outside_cluster_ids:
                        outside = inside & np.isin(labels, outside_cluster_ids)
                        if np.any(outside):
                            flat = y_pixels[outside] * tile_width + x_pixels[outside]
                            density = np.bincount(
                                flat, minlength=tile_height * tile_width
                            ).reshape(tile_height, tile_width)
                            peak = float(np.max(density))
                            if peak > 0:
                                tile = np.maximum(
                                    tile,
                                    (0.32 * np.sqrt(density / peak))[..., None],
                                )
                else:
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
                    for site_index in oriented_sites[cluster_inside]:
                        memberships = (
                            site_memberships[int(site_index)]
                            if 0 <= int(site_index) < len(site_memberships)
                            else ()
                        )
                        center_colors.append(
                            np.mean(digital_group_colors[list(memberships)], axis=0)
                            if memberships
                            else (
                                np.asarray([0.58, 0.64, 0.72])
                                if digital_group_ids
                                else site_colors[int(site_index)]
                            )
                        )

            gallery[y0 : y0 + tile_height, x0 : x0 + tile_width] = np.clip(tile, 0.0, 1.0)
            grid_x, grid_y, _grid_inside = to_tile_pixels(digital_grid)
            if show_analog_sites:
                expected_x.extend((x0 + grid_x).tolist())
                expected_y.extend((y0 + grid_y).tolist())
            label = axis.text(
                x0 + 2,
                y0 + 2,
                (
                    f"#{int(result_index) + 1}  n={result.source_point_counts[result_index]:,}; "
                    f"{len(detected_group_indices)} digital groups"
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
        transforms: list[Callable[[np.ndarray], np.ndarray]] = []
        for page_offset in range(len(page_indices)):
            row, column = divmod(page_offset, columns)
            x0 = column * (tile_width + gap)
            y0 = row * (tile_height + gap)

            def to_gallery(points: np.ndarray, x0: int = x0, y0: int = y0) -> np.ndarray:
                converted = np.empty_like(np.asarray(points, dtype=float))
                converted[:, 0] = x0 + (points[:, 0] - x_min) * tile_width / width_nm
                converted[:, 1] = y0 + (y_max - points[:, 1]) * tile_height / height_nm
                return converted

            transforms.append(to_gallery)
        group_handles = self._draw_digital_group_blobs(
            axis,
            digital_grid,
            transforms,
            linewidth=0.65,
        )
        if show_analog_sites:
            axis.scatter(expected_x, expected_y, s=8, facecolors="none", edgecolors="#67e8f9", linewidths=0.45)
        if center_x:
            axis.scatter(center_x, center_y, s=9, marker="x", c=np.asarray(center_colors), linewidths=0.7)
        if group_handles:
            axis.legend(
                handles=group_handles,
                loc="lower center",
                ncol=min(6, len(group_handles)),
                fontsize=6,
                framealpha=0.75,
            )
        detail = (
            f"G5M σ {result.g5m_sigma_min_nm:g}–{result.g5m_sigma_max_nm:g} nm, minimum "
            f"{result.g5m_min_locs} locs, BIC patience {result.g5m_max_rounds_without_best_bic}"
            if result.clustering_method == "Picasso G5M"
            else (
                f"Fast supported-site assignment: ≥{result.direct_min_site_localizations} locs and site "
                f"prominence ≥{result.direct_min_site_evidence:g}; use G5M refinement for model-selected components"
            )
        )
        axis.set_title(
            f"Docking-site assignments — {self.origami_gallery_page_label.get()}\n"
            f"{result.clustering_method}; {detail}; site match {result.site_match_radius_nm:g} nm\n"
            + (
                "Click a tile for full detail; analog sites are outlined; × = assigned center"
                if show_analog_sites
                else "Click a tile for full detail; active digital groups only; × = assigned center"
            )
        )
        axis.set_axis_off()

    def _plot_selected_origami_detail(
        self,
        axis: Any,
        result: OrigamiAnalysisResult,
        render_settings: dict[str, Any],
    ) -> None:
        index = self.origami_selected_index
        if index is None or index < 0 or index >= result.origami_count:
            axis.text(0.5, 0.5, "Click an origami in either gallery to inspect it.", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return
        self.origami_detail_number.set(index + 1)
        self.origami_detail_label.set(f"Origami {index + 1:,}/{result.origami_count:,}")
        points = result.aligned_points[index]
        labels = result.cluster_labels[index]
        sites = result.cluster_site_indices[index]
        centers = result.cluster_centers_nm[index]
        detail_settings = dict(render_settings)
        digital_grid = self._origami_digital_overlay_grid(result, detail_settings)
        detail_settings["pixel_size_nm"] = min(float(detail_settings["pixel_size_nm"]), 0.5)
        rendered = render_aligned_origami_density([points], **detail_settings)
        axis.imshow(
            rendered["image"],
            extent=rendered["extent"],
            origin="lower",
            cmap="gray_r",
            interpolation="nearest",
            aspect="equal",
        )
        site_count = result.rows * result.columns
        colors = matplotlib.colormaps["tab20"](np.linspace(0.0, 1.0, max(site_count, 2)))
        noise = labels < 0
        if np.any(noise):
            axis.scatter(points[noise, 0], points[noise, 1], s=5, color="#94a3b8", alpha=0.45, label="unassigned")
        for cluster_label, site_index in enumerate(sites):
            members = labels == cluster_label
            if np.any(members):
                axis.scatter(points[members, 0], points[members, 1], s=7, color=colors[int(site_index)], alpha=0.75)
        group_handles = self._draw_digital_group_blobs(axis, digital_grid)
        if self._active_origami_digital_pixel_model() is None:
            axis.scatter(
                digital_grid[:, 0], digital_grid[:, 1], s=80,
                facecolors="none", edgecolors="#06b6d4", linewidths=1.2,
                label="analog sites", zorder=4,
            )
        if len(centers):
            axis.scatter(centers[:, 0], centers[:, 1], s=45, marker="x", color=colors[sites], linewidths=1.3, label="assigned centers")
        occupied = float(np.sum(result.site_occupancy[index]))
        axis.set_title(
            f"Origami #{index + 1}: {result.source_point_counts[index]:,} source points; {occupied:g} occupied sites\n"
            f"RMS {result.alignment_rms_nm[index]:.3g} nm; grid match {100.0 * result.grid_match_fraction[index]:.1f}%; "
            f"{result.clustering_method}"
        )
        axis.set_xlabel("aligned x (nm)")
        axis.set_ylabel("aligned y (nm)")
        axis.legend(handles=[*axis.get_legend_handles_labels()[0], *group_handles], loc="upper right", fontsize=8)
        axis.grid(False)

    def _plot_filtered_maps(self, result: dict[str, Any]) -> None:
        if self.loaded is None or result.get("source_path") != self.loaded.path:
            return
        if "filtered_locs" in result:
            self.filtered_map_locs = result["filtered_locs"]
            self.filtered_map_render_context = {
                key: result[key]
                for key in ("map_source", "source_label", "scope_text", "filter_text")
                if key in result
            }
        filtered = result.get("map")
        if filtered is None:
            self.map_density_images.pop(FILTERED_MAP_TAB, None)
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
            self.map_density_images[FILTERED_MAP_TAB] = filtered_image
            filtered_display, filtered_limits = self._scale_map_density(
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
            self.map_density_images[LINKED_MAP_TAB] = image
            display_image, density_limits = self._scale_map_density(image)
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
            f"{roi_text} localizations at {float(result['disp_px_size_nm']):.3g} nm/pixel. "
            f"Density limits {density_limits[0]:.4g}-{density_limits[1]:.4g}."
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
            messagebox.showinfo("No origami overlay", "Build the fast overlay before exporting.")
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
                    "occupancy_definition": "mean of assigned-group presence at the 0-degree and 180-degree site counterparts",
                    "orientation_averaging": "0_and_180_degrees_equal_weight",
                    "clustering_method": result.clustering_method,
                    "g5m_sigma_min_nm": result.g5m_sigma_min_nm if result.clustering_method == "Picasso G5M" else np.nan,
                    "g5m_sigma_max_nm": result.g5m_sigma_max_nm if result.clustering_method == "Picasso G5M" else np.nan,
                    "g5m_minimum_localizations": result.g5m_min_locs if result.clustering_method == "Picasso G5M" else np.nan,
                    "g5m_bic_patience": result.g5m_max_rounds_without_best_bic if result.clustering_method == "Picasso G5M" else np.nan,
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

    def export_origami_gallery_pdf(self) -> None:
        result = self.origami_result
        if result is None or result.origami_count == 0:
            messagebox.showinfo("No origami overlay", "Build the fast overlay before exporting a gallery.")
            return
        path_text = filedialog.asksaveasfilename(
            title="Export paged origami gallery",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile="origami_gallery.pdf",
        )
        if not path_text:
            return
        path = Path(path_text)
        original_option = self.origami_plot_option.get()
        original_page = int(self.origami_gallery_page.get())
        gallery_option = original_option if original_option in {"Individual origami gallery", "Individual site assignments"} else "Individual origami gallery"
        self.origami_plot_option.set(gallery_option)
        self.origami_gallery_page.set(1)
        try:
            _first_page, selected_count = self._origami_gallery_page_data()
            page_size = int(self.origami_gallery_page_size.get())
            page_count = max(1, int(math.ceil(selected_count / page_size)))
            with PdfPages(path) as pdf:
                for page_number in range(1, page_count + 1):
                    self.origami_gallery_page.set(page_number)
                    self.render_origami_plot()
                    self.status.set(f"Exporting gallery PDF page {page_number:,}/{page_count:,}...")
                    self.update_idletasks()
                    pdf.savefig(self.origami_figure)
        except Exception as exc:
            messagebox.showerror("Gallery export failed", str(exc))
            return
        finally:
            self.origami_plot_option.set(original_option)
            self.origami_gallery_page.set(original_page)
            self.render_origami_plot()
        self._remember_file_dialog_dir(path)
        self.status.set(f"Exported {page_count:,}-page gallery PDF to {path.name} without allocating a full-population mosaic.")


def main() -> None:
    app = PaintAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
