"""Detection, alignment, and site-level statistics for DNA origami localizations."""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates, maximum_filter, rotate, shift as ndimage_shift
from scipy.optimize import linear_sum_assignment, minimize, nnls
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from scipy.spatial import cKDTree


@dataclass
class OrigamiAnalysisResult:
    aligned_points: list[np.ndarray]
    centers_nm: np.ndarray
    source_point_counts: np.ndarray
    site_counts: np.ndarray
    site_occupancy: np.ndarray
    cluster_labels: list[np.ndarray]
    cluster_centers_nm: list[np.ndarray]
    cluster_site_indices: list[np.ndarray]
    alignment_rms_nm: np.ndarray
    grid_match_fraction: np.ndarray
    grid_points_nm: np.ndarray
    rows: int
    columns: int
    g5m_sigma_min_nm: float
    g5m_sigma_max_nm: float
    g5m_min_locs: int
    g5m_max_rounds_without_best_bic: int
    site_match_radius_nm: float
    direct_min_site_localizations: int
    direct_min_site_evidence: float
    rejected_candidate_count: int
    symmetrized_180: bool
    clustering_method: str = "Picasso G5M"

    @property
    def origami_count(self) -> int:
        return len(self.aligned_points)


@dataclass
class OrigamiPickResult:
    regions: list[np.ndarray]
    aligned_regions: list[np.ndarray]
    accepted_mask: np.ndarray
    point_counts: np.ndarray
    original_point_counts: np.ndarray
    crop_retained_fractions: np.ndarray
    bounds_nm: np.ndarray
    rectangle_corners_nm: np.ndarray
    rectangle_angles_deg: np.ndarray
    rectangle_confidence: np.ndarray
    site_gap_contrast: np.ndarray
    on_site_fraction: np.ndarray
    site_mask_radius_nm: float
    template_points_nm: np.ndarray
    lattice_site_localization_counts: np.ndarray
    lattice_site_prominence: np.ndarray
    lattice_supported_sites: np.ndarray
    site_localization_counts: np.ndarray
    site_prominence: np.ndarray
    site_peak_positions_nm: np.ndarray
    site_boundary_reference_positions_nm: np.ndarray
    site_boundary_points_nm: np.ndarray
    site_centroids_nm: np.ndarray
    supported_site_count: np.ndarray
    supported_row_count: np.ndarray
    supported_column_count: np.ndarray
    site_spacing_rms_nm: np.ndarray
    site_spacing_max_error_nm: np.ndarray
    grid_vs_blob_delta_bic: np.ndarray
    rectangle_matched_site_count: np.ndarray
    rectangle_fit_rms_nm: np.ndarray
    rectangle_width_nm: float
    rectangle_height_nm: float
    density_image: np.ndarray
    density_contrast: np.ndarray
    density_component_labels: np.ndarray
    density_extent_nm: tuple[float, float, float, float]
    density_threshold: float
    alignment_pixel_nm: float
    alignment_canvas_side_nm: float
    alignment_reference_image: np.ndarray
    alignment_candidate_images: np.ndarray

    @property
    def accepted_regions(self) -> list[np.ndarray]:
        return [region for region, accepted in zip(self.regions, self.accepted_mask) if bool(accepted)]

    @property
    def accepted_aligned_regions(self) -> list[np.ndarray]:
        return [region for region, accepted in zip(self.aligned_regions, self.accepted_mask) if bool(accepted)]

    @property
    def accepted_rectangle_corners(self) -> list[np.ndarray]:
        return [corners for corners, accepted in zip(self.rectangle_corners_nm, self.accepted_mask) if bool(accepted)]

    @property
    def accepted_count(self) -> int:
        return int(np.count_nonzero(self.accepted_mask))


@dataclass
class SparseSiteEvidenceResult:
    counts: np.ndarray
    prominence: np.ndarray
    peak_positions_nm: np.ndarray
    peak_density: np.ndarray
    boundary_density: np.ndarray
    boundary_reference_density: np.ndarray
    boundary_reference_positions_nm: np.ndarray
    boundary_points_nm: np.ndarray


def concatenate_origami_pick_results(results: list[OrigamiPickResult]) -> OrigamiPickResult:
    """Combine same-template tile results into one world-coordinate pick result."""
    if not results:
        raise ValueError("At least one origami pick result is required.")
    first = results[0]
    for result in results[1:]:
        if result.template_points_nm.shape != first.template_points_nm.shape or not np.allclose(
            result.template_points_nm,
            first.template_points_nm,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Tiled origami results must use the same theoretical template.")

    def concatenate(attribute: str, *, axis: int = 0) -> np.ndarray:
        arrays = [np.asarray(getattr(result, attribute)) for result in results]
        populated = [array for array in arrays if array.ndim > axis and array.shape[axis] > 0]
        return np.concatenate(populated, axis=axis) if populated else arrays[0].copy()

    extents = np.asarray([result.density_extent_nm for result in results], dtype=float)
    density_extent = (
        float(np.min(extents[:, 0])),
        float(np.max(extents[:, 1])),
        float(np.min(extents[:, 2])),
        float(np.max(extents[:, 3])),
    )
    return OrigamiPickResult(
        regions=[region for result in results for region in result.regions],
        aligned_regions=[region for result in results for region in result.aligned_regions],
        accepted_mask=concatenate("accepted_mask"),
        point_counts=concatenate("point_counts"),
        original_point_counts=concatenate("original_point_counts"),
        crop_retained_fractions=concatenate("crop_retained_fractions"),
        bounds_nm=concatenate("bounds_nm"),
        rectangle_corners_nm=concatenate("rectangle_corners_nm"),
        rectangle_angles_deg=concatenate("rectangle_angles_deg"),
        rectangle_confidence=concatenate("rectangle_confidence"),
        site_gap_contrast=concatenate("site_gap_contrast"),
        on_site_fraction=concatenate("on_site_fraction"),
        site_mask_radius_nm=float(first.site_mask_radius_nm),
        template_points_nm=np.asarray(first.template_points_nm, dtype=float).copy(),
        lattice_site_localization_counts=concatenate("lattice_site_localization_counts"),
        lattice_site_prominence=concatenate("lattice_site_prominence"),
        lattice_supported_sites=concatenate("lattice_supported_sites"),
        site_localization_counts=concatenate("site_localization_counts"),
        site_prominence=concatenate("site_prominence"),
        site_peak_positions_nm=concatenate("site_peak_positions_nm"),
        site_boundary_reference_positions_nm=concatenate("site_boundary_reference_positions_nm"),
        site_boundary_points_nm=concatenate("site_boundary_points_nm"),
        site_centroids_nm=concatenate("site_centroids_nm"),
        supported_site_count=concatenate("supported_site_count"),
        supported_row_count=concatenate("supported_row_count"),
        supported_column_count=concatenate("supported_column_count"),
        site_spacing_rms_nm=concatenate("site_spacing_rms_nm"),
        site_spacing_max_error_nm=concatenate("site_spacing_max_error_nm"),
        grid_vs_blob_delta_bic=concatenate("grid_vs_blob_delta_bic"),
        rectangle_matched_site_count=concatenate("rectangle_matched_site_count"),
        rectangle_fit_rms_nm=concatenate("rectangle_fit_rms_nm"),
        rectangle_width_nm=float(first.rectangle_width_nm),
        rectangle_height_nm=float(first.rectangle_height_nm),
        # Coarse grids have tile-local shapes and coordinates. The combined
        # spatial view renders source points directly and intentionally omits
        # a misleading stitched coarse contour.
        density_image=np.empty((0, 0), dtype=float),
        density_contrast=np.empty((0, 0), dtype=float),
        density_component_labels=np.empty((0, 0), dtype=int),
        density_extent_nm=density_extent,
        density_threshold=float(first.density_threshold),
        alignment_pixel_nm=float(first.alignment_pixel_nm),
        alignment_canvas_side_nm=float(first.alignment_canvas_side_nm),
        alignment_reference_image=np.asarray(first.alignment_reference_image).copy(),
        alignment_candidate_images=concatenate("alignment_candidate_images"),
    )


@dataclass
class TemplateClassificationResult:
    """One-to-one assignments of spatial candidates to competing templates."""

    assignment_masks: list[np.ndarray]
    counts: np.ndarray
    unclassified_count: int
    suppressed_duplicate_count: int
    group_centers_nm: np.ndarray
    winning_template_indices: np.ndarray
    candidate_group_indices: list[np.ndarray]
    winning_scores: np.ndarray
    runner_up_template_indices: np.ndarray
    runner_up_scores: np.ndarray
    winning_score_margins: np.ndarray
    template_probabilities: np.ndarray
    raw_template_probabilities: np.ndarray


def bidirectional_template_classification_scores(
    correlations: np.ndarray,
    on_site_fractions: np.ndarray,
    supported_site_counts: np.ndarray,
    template_site_count: int,
    *,
    off_site_empty_fractions: np.ndarray | None = None,
    bright_site_probabilities: np.ndarray | None = None,
    cell_pattern_correlations: np.ndarray | None = None,
) -> np.ndarray:
    """Score observed signal, expected bright sites, and expected black cells.

    Correlation measures the raster fit. The harmonic precision/recall term
    additionally penalizes localization signal in template-black space and
    bright template sites that have no measured support. This prevents a
    dense superset template from winning solely because it overlaps every
    site in a sparser pattern.
    """
    correlations = np.asarray(correlations, dtype=float)
    precision = np.asarray(on_site_fractions, dtype=float)
    supported = np.asarray(supported_site_counts, dtype=float)
    if template_site_count < 1:
        raise ValueError("Template classification requires at least one theoretical site.")
    if correlations.shape != precision.shape or correlations.shape != supported.shape:
        raise ValueError("Correlation, on-site fraction, and supported-site arrays must match.")
    precision = np.clip(precision, 0.0, 1.0)
    recall = np.clip(supported / float(template_site_count), 0.0, 1.0)
    if bright_site_probabilities is not None:
        if off_site_empty_fractions is None:
            raise ValueError("Bright-site probabilities require off-site empty fractions.")
        bright_probability = np.asarray(bright_site_probabilities, dtype=float)
        specificity = np.asarray(off_site_empty_fractions, dtype=float)
        if bright_probability.shape != correlations.shape or specificity.shape != correlations.shape:
            raise ValueError("Cell-probability arrays must match the correlation array.")
        # Both terms come from one equal-prior candidate-specific brightness
        # model.  Equal weighting therefore does not encode a preference for
        # templates with either more bright cells or more black cells.
        agreement_terms = (
            np.clip(bright_probability, 0.0, 1.0)
            * np.clip(specificity, 0.0, 1.0)
        )
        if cell_pattern_correlations is None:
            agreement = np.sqrt(agreement_terms)
        else:
            pattern = np.asarray(cell_pattern_correlations, dtype=float)
            if pattern.shape != correlations.shape:
                raise ValueError("Cell-pattern correlations must match the correlation array.")
            agreement = np.cbrt(agreement_terms * np.clip(pattern, 0.0, 1.0))
    elif off_site_empty_fractions is None:
        denominator = precision + recall
        agreement = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
    else:
        specificity = np.asarray(off_site_empty_fractions, dtype=float)
        if specificity.shape != correlations.shape:
            raise ValueError("Off-site empty fractions must match the correlation array.")
        specificity = np.clip(specificity, 0.0, 1.0)
        # Treat all three directions as independent evidence.  Omitting
        # precision over-rewards sparse templates because most of their black
        # cells are easily empty; omitting specificity over-rewards dense
        # templates because they cover most observed signal.  Their geometric
        # mean requires a fit to explain observed localizations, expected
        # bright cells, and expected black cells without a density prior.
        agreement = np.cbrt(precision * recall * specificity)
    # Correlation is useful for pose quality, but its dynamic range is biased
    # toward dense superset templates: they can overlap nearly every bright
    # feature in a sparse candidate. Make bidirectional site agreement the
    # primary classification term and let correlation contribute only a
    # bounded 25% modulation.
    correlation_quality = np.clip(correlations, 0.0, 1.0)
    return agreement * (0.75 + 0.25 * correlation_quality)


def lattice_template_probability_agreement(
    aligned_regions: list[np.ndarray],
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score template cells using adaptive equal-prior brightness probabilities.

    For each candidate, localization counts on the complete lattice are split
    into dark and bright intensity populations.  The posterior uses equal
    class priors, so the cutoff is determined by the candidate's brightness
    distribution rather than by the number of occupied cells in a template.
    """
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    if template.ndim != 2 or template.shape[1] != 2 or not len(template):
        raise ValueError("Lattice classification requires at least one 2D template point.")
    grid_tree = cKDTree(full_grid)
    template_distances, template_cells = grid_tree.query(template, k=1)
    mapping_tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(template_distances > mapping_tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    occupied_cells = np.zeros(len(full_grid), dtype=bool)
    occupied_cells[np.asarray(template_cells, dtype=int)] = True
    black_cells = ~occupied_cells
    region_count = len(aligned_regions)
    cell_count = len(full_grid)
    lengths = np.fromiter((len(region) for region in aligned_regions), dtype=np.int64, count=region_count)
    counts = np.zeros((region_count, cell_count), dtype=float)
    if region_count and int(np.sum(lengths)):
        # The lattice is regular, so nearest row/column arithmetic replaces a
        # separate KD-tree query for every candidate.  One bincount then builds
        # the complete candidate-by-cell matrix.
        all_points = np.concatenate([np.asarray(region, dtype=float) for region in aligned_regions if len(region)])
        region_indices = np.repeat(np.arange(region_count, dtype=np.int64), lengths)
        x0 = -0.5 * (columns - 1) * float(spacing_x_nm)
        y0 = -0.5 * (rows - 1) * float(spacing_y_nm)
        column_indices = np.rint((all_points[:, 0] - x0) / float(spacing_x_nm)).astype(np.int64)
        row_indices = np.rint((all_points[:, 1] - y0) / float(spacing_y_nm)).astype(np.int64)
        inside = (
            (column_indices >= 0)
            & (column_indices < columns)
            & (row_indices >= 0)
            & (row_indices < rows)
        )
        clipped_columns = np.clip(column_indices, 0, columns - 1)
        clipped_rows = np.clip(row_indices, 0, rows - 1)
        nearest_x = x0 + clipped_columns * float(spacing_x_nm)
        nearest_y = y0 + clipped_rows * float(spacing_y_nm)
        maximum_cell_distance_sq = 0.25 * (
            float(spacing_x_nm) ** 2 + float(spacing_y_nm) ** 2
        )
        assigned = inside & (
            np.square(all_points[:, 0] - nearest_x) + np.square(all_points[:, 1] - nearest_y)
            <= maximum_cell_distance_sq
        )
        flat_cells = clipped_rows * columns + clipped_columns
        flat_keys = region_indices[assigned] * cell_count + flat_cells[assigned]
        counts = np.bincount(flat_keys, minlength=region_count * cell_count).reshape(region_count, cell_count)

    on_template_counts = np.sum(counts[:, occupied_cells], axis=1)
    on_site_fractions = np.divide(
        on_template_counts,
        lengths,
        out=np.zeros(region_count, dtype=float),
        where=lengths > 0,
    )
    intensity = np.log1p(counts)
    if not region_count:
        return np.empty(0), np.empty(0), on_site_fractions, np.empty(0)

    low = np.min(intensity, axis=1)
    high = np.max(intensity, axis=1)
    separable = high > low + 1e-12
    # Run all candidates' one-dimensional two-means fits together. The
    # midpoint remains the equal-prior/equal-variance decision boundary.
    for _ in range(24):
        boundary = 0.5 * (low + high)
        dark_group = intensity <= boundary[:, None]
        dark_count = np.sum(dark_group, axis=1)
        bright_count = cell_count - dark_count
        valid = separable & (dark_count > 0) & (bright_count > 0)
        next_low = np.divide(
            np.sum(np.where(dark_group, intensity, 0.0), axis=1),
            dark_count,
            out=low.copy(),
            where=valid,
        )
        next_high = np.divide(
            np.sum(np.where(~dark_group, intensity, 0.0), axis=1),
            bright_count,
            out=high.copy(),
            where=valid,
        )
        change = np.maximum(np.abs(next_low - low), np.abs(next_high - high))
        low = np.where(valid, next_low, low)
        high = np.where(valid, next_high, high)
        if not np.any(change[valid] >= 1e-8):
            break

    separation = np.maximum(high - low, 1e-6)
    boundary = 0.5 * (low + high)
    dark_group = intensity <= boundary[:, None]
    residual = np.where(dark_group, intensity - low[:, None], intensity - high[:, None])
    pooled_variance = np.mean(np.square(residual), axis=1)
    variance = np.maximum.reduce((pooled_variance, np.square(separation / 4.0), np.full(region_count, 1e-6)))
    log_odds = np.clip(
        separation[:, None] * (intensity - boundary[:, None]) / variance[:, None],
        -20.0,
        20.0,
    )
    bright_probability = 1.0 / (1.0 + np.exp(-log_odds))
    bright_probability[~separable] = 0.5
    bright_agreement = np.mean(bright_probability[:, occupied_cells], axis=1)
    dark_agreement = (
        np.mean(1.0 - bright_probability[:, black_cells], axis=1)
        if np.any(black_cells)
        else np.ones(region_count, dtype=float)
    )
    template_pattern = occupied_cells.astype(float)
    template_pattern -= np.mean(template_pattern)
    template_norm = float(np.linalg.norm(template_pattern))
    centered_probability = bright_probability - np.mean(bright_probability, axis=1, keepdims=True)
    probability_norm = np.linalg.norm(centered_probability, axis=1)
    pattern_denominator = probability_norm * template_norm
    if template_norm <= 1e-12:
        # An all-bright template has no spatial black/bright contrast to
        # correlate; leave the pattern term neutral rather than rejecting it.
        pattern_correlation = np.ones(region_count, dtype=float)
    else:
        pattern_correlation = np.divide(
            centered_probability @ template_pattern,
            pattern_denominator,
            out=np.zeros(region_count, dtype=float),
            where=pattern_denominator > 1e-12,
        )
    return bright_agreement, dark_agreement, on_site_fractions, pattern_correlation


def lattice_count_template_probability_agreement(
    lattice_site_counts: np.ndarray,
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare continuous per-cell counts with a template without hard site calls.

    Each candidate gets its own equal-prior, equal-variance bright/dark model.
    The resulting cell probabilities change smoothly when a cell gains or loses
    a localization, unlike the prominence/support mask used for QC display.
    """
    counts = np.asarray(lattice_site_counts, dtype=float)
    cell_count = int(rows) * int(columns)
    if counts.ndim != 2 or counts.shape[1] != cell_count:
        raise ValueError("Lattice-site counts must have one column per lattice cell.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("Lattice-site counts must be finite and nonnegative.")
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    if template.ndim != 2 or template.shape[1] != 2 or not len(template):
        raise ValueError("Lattice classification requires at least one 2D template point.")
    distances, cells = cKDTree(full_grid).query(template, k=1)
    tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(distances > tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    occupied = np.zeros(cell_count, dtype=bool)
    occupied[np.asarray(cells, dtype=int)] = True
    black = ~occupied

    intensity = np.log1p(counts)
    candidate_count = len(counts)
    if not candidate_count:
        return np.empty(0), np.empty(0), np.empty(0)
    low = np.min(intensity, axis=1)
    high = np.max(intensity, axis=1)
    separable = high > low + 1e-12
    for _iteration in range(24):
        boundary = 0.5 * (low + high)
        dark_group = intensity <= boundary[:, None]
        dark_count = np.sum(dark_group, axis=1)
        bright_count = cell_count - dark_count
        valid = separable & (dark_count > 0) & (bright_count > 0)
        next_low = np.divide(
            np.sum(np.where(dark_group, intensity, 0.0), axis=1),
            dark_count,
            out=low.copy(),
            where=valid,
        )
        next_high = np.divide(
            np.sum(np.where(~dark_group, intensity, 0.0), axis=1),
            bright_count,
            out=high.copy(),
            where=valid,
        )
        change = np.maximum(np.abs(next_low - low), np.abs(next_high - high))
        low = np.where(valid, next_low, low)
        high = np.where(valid, next_high, high)
        if not np.any(change[valid] >= 1e-8):
            break

    separation = np.maximum(high - low, 1e-6)
    boundary = 0.5 * (low + high)
    dark_group = intensity <= boundary[:, None]
    residual = np.where(dark_group, intensity - low[:, None], intensity - high[:, None])
    pooled_variance = np.mean(np.square(residual), axis=1)
    variance = np.maximum.reduce(
        (pooled_variance, np.square(separation / 4.0), np.full(candidate_count, 1e-6))
    )
    log_odds = np.clip(
        separation[:, None] * (intensity - boundary[:, None]) / variance[:, None],
        -20.0,
        20.0,
    )
    bright_probability = 1.0 / (1.0 + np.exp(-log_odds))
    bright_probability[~separable] = 0.5
    bright_agreement = np.mean(bright_probability[:, occupied], axis=1)
    dark_agreement = (
        np.mean(1.0 - bright_probability[:, black], axis=1)
        if np.any(black)
        else np.ones(candidate_count, dtype=float)
    )
    expected = occupied.astype(float) - float(np.mean(occupied))
    expected_norm = float(np.linalg.norm(expected))
    if expected_norm <= 1e-12:
        pattern_correlation = np.ones(candidate_count, dtype=float)
    else:
        observed = bright_probability - np.mean(bright_probability, axis=1, keepdims=True)
        denominator = np.linalg.norm(observed, axis=1) * expected_norm
        pattern_correlation = np.divide(
            observed @ expected,
            denominator,
            out=np.zeros(candidate_count, dtype=float),
            where=denominator > 1e-12,
        )
    return bright_agreement, dark_agreement, pattern_correlation


def logical_bit_template_evidence(
    lattice_site_counts: np.ndarray,
    logical_bit_cells: Sequence[Sequence[int]],
    active_logical_bits: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate physical extension sites into tolerant arbitrary logical bits.

    The physical lattice is still used for rigid registration.  Classification
    is performed on one probability per multi-site stroke, so a missing or
    blinking-heavy extension changes only a fraction of one logical bit.  Each
    logical bit contributes once regardless of how many physical sites draw it.

    Returns template-vs-unstructured posterior, log Bayes factor, per-candidate
    logical-bit probabilities, and logical-pattern correlation.
    """
    counts = np.asarray(lattice_site_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("Lattice-site counts must be a candidate-by-cell matrix.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("Lattice-site counts must be finite and nonnegative.")
    bit_cells = [np.asarray(cells, dtype=int) for cells in logical_bit_cells]
    active = np.asarray(active_logical_bits, dtype=bool)
    if not bit_cells or active.shape != (len(bit_cells),):
        raise ValueError("Logical templates require one active state per logical bit.")
    for cells in bit_cells:
        if not len(cells):
            raise ValueError("Every logical bit must contain at least one physical site.")
        if np.any(cells < 0) or np.any(cells >= counts.shape[1]):
            raise ValueError("Logical-bit physical site is outside the lattice.")

    candidate_count, cell_count = counts.shape
    if not candidate_count:
        return (
            np.empty(0),
            np.empty(0),
            np.empty((0, len(bit_cells))),
            np.empty(0),
        )

    # Reuse the candidate-specific equal-prior bright/dark separation, but do
    # not threshold individual physical sites.  Averaging these probabilities
    # within a stroke is the deliberate analog-to-logical conversion.
    intensity = np.log1p(counts)
    low = np.min(intensity, axis=1)
    high = np.max(intensity, axis=1)
    separable = high > low + 1e-12
    for _iteration in range(24):
        boundary = 0.5 * (low + high)
        dark_group = intensity <= boundary[:, None]
        dark_count = np.sum(dark_group, axis=1)
        bright_count = cell_count - dark_count
        valid = separable & (dark_count > 0) & (bright_count > 0)
        next_low = np.divide(
            np.sum(np.where(dark_group, intensity, 0.0), axis=1),
            dark_count,
            out=low.copy(),
            where=valid,
        )
        next_high = np.divide(
            np.sum(np.where(~dark_group, intensity, 0.0), axis=1),
            bright_count,
            out=high.copy(),
            where=valid,
        )
        change = np.maximum(np.abs(next_low - low), np.abs(next_high - high))
        low = np.where(valid, next_low, low)
        high = np.where(valid, next_high, high)
        if not np.any(change[valid] >= 1e-8):
            break
    separation = np.maximum(high - low, 1e-6)
    boundary = 0.5 * (low + high)
    dark_group = intensity <= boundary[:, None]
    residual = np.where(dark_group, intensity - low[:, None], intensity - high[:, None])
    variance = np.maximum.reduce(
        (
            np.mean(np.square(residual), axis=1),
            np.square(separation / 4.0),
            np.full(candidate_count, 1e-6),
        )
    )
    log_odds = np.clip(
        separation[:, None] * (intensity - boundary[:, None]) / variance[:, None],
        -20.0,
        20.0,
    )
    physical_bright_probability = 1.0 / (1.0 + np.exp(-log_odds))
    physical_bright_probability[~separable] = 0.5

    bit_probability = np.column_stack(
        [np.mean(physical_bright_probability[:, cells], axis=1) for cells in bit_cells]
    )
    bit_probability = np.clip(bit_probability, 1e-4, 1.0 - 1e-4)
    # Groups may overlap physically. A shared analog site is ON when any active
    # logical group contains it, so that site cannot be used as dark evidence
    # against an inactive group. Shared evidence is also divided among its
    # memberships to avoid counting one localization cluster multiple times.
    membership_count = np.zeros(cell_count, dtype=float)
    for cells in bit_cells:
        membership_count[cells] += 1.0
    membership_count = np.maximum(membership_count, 1.0)
    active_union = np.zeros(cell_count, dtype=bool)
    for cells, is_active in zip(bit_cells, active):
        if is_active:
            active_union[cells] = True

    expected_probability = np.full((candidate_count, len(bit_cells)), 0.5, dtype=float)
    evidence_weight = np.zeros(len(bit_cells), dtype=float)
    for bit_index, (cells, is_active) in enumerate(zip(bit_cells, active)):
        informative_cells = cells if is_active else cells[~active_union[cells]]
        if not len(informative_cells):
            continue
        if is_active:
            expected_probability[:, bit_index] = np.mean(
                physical_bright_probability[:, informative_cells], axis=1
            )
        else:
            expected_probability[:, bit_index] = np.mean(
                1.0 - physical_bright_probability[:, informative_cells], axis=1
            )
        evidence_weight[bit_index] = float(
            np.mean(1.0 / membership_count[informative_cells])
        )
    expected_probability = np.clip(expected_probability, 1e-4, 1.0 - 1e-4)
    # A non-overlapping logical bit contributes one observation regardless of
    # how many physical sites draw it. Overlap reduces only the duplicated
    # portion of its evidence.
    log_bayes_factor = np.sum(
        evidence_weight[None, :] * (np.log(expected_probability) - math.log(0.5)),
        axis=1,
    )
    posterior = 1.0 / (1.0 + np.exp(-np.clip(log_bayes_factor, -40.0, 40.0)))

    expected = active.astype(float) - float(np.mean(active))
    expected_norm = float(np.linalg.norm(expected))
    if expected_norm <= 1e-12:
        pattern_correlation = np.ones(candidate_count, dtype=float)
    else:
        observed = bit_probability - np.mean(bit_probability, axis=1, keepdims=True)
        denominator = np.linalg.norm(observed, axis=1) * expected_norm
        pattern_correlation = np.divide(
            observed @ expected,
            denominator,
            out=np.zeros(candidate_count, dtype=float),
            where=denominator > 1e-12,
        )
    return posterior, log_bayes_factor, bit_probability, pattern_correlation


def direct_digital_group_localization_evidence(
    aligned_regions: Sequence[np.ndarray],
    lattice_points_nm: np.ndarray,
    digital_group_cells: Sequence[Sequence[int]],
    *,
    assignment_radius_nm: float,
) -> np.ndarray:
    """Measure digital groups directly from aligned localization coordinates.

    The lattice coordinates describe the spatial footprint of each user-defined
    group, but no localization is first assigned to an individual lattice site.
    Instead, each localization receives an independent Gaussian affinity to
    every group from its distance to that group's footprint. A localization at
    a physical position shared by multiple logical groups contributes to every
    such group: shared analog positions encode membership in each logical bit
    and must not make those bits compete for a conserved weight.

    Values are returned as localization support per member position. This makes
    groups containing different numbers of positions comparable without turning
    those positions into independently classified analog bits.
    """
    grid = np.asarray(lattice_points_nm, dtype=float)
    if grid.ndim != 2 or grid.shape[1:] != (2,):
        raise ValueError("Lattice points must be an N-by-2 coordinate array.")
    radius = float(assignment_radius_nm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Digital-group assignment radius must be positive.")
    groups = [np.unique(np.asarray(cells, dtype=int)) for cells in digital_group_cells]
    if not groups:
        raise ValueError("At least one digital group is required.")
    for cells in groups:
        if not len(cells):
            raise ValueError("Every digital group must contain at least one position.")
        if np.any(cells < 0) or np.any(cells >= len(grid)):
            raise ValueError("Digital-group position is outside the template lattice.")

    evidence = np.zeros((len(aligned_regions), len(groups)), dtype=float)
    group_trees = [cKDTree(grid[cells]) for cells in groups]
    radius_squared = radius * radius
    for region_index, region_value in enumerate(aligned_regions):
        region = np.asarray(region_value, dtype=float)
        if region.ndim != 2 or region.shape[1:] != (2,) or not len(region):
            continue
        responsibilities = _digital_group_responsibilities_from_trees(
            region,
            group_trees,
            radius,
            radius_squared,
        )
        evidence[region_index] = np.sum(responsibilities, axis=0) / np.asarray(
            [len(cells) for cells in groups], dtype=float
        )
    return evidence


def _digital_group_responsibilities_from_trees(
    region: np.ndarray,
    group_trees: Sequence[cKDTree],
    radius: float,
    radius_squared: float,
) -> np.ndarray:
    affinity = np.zeros((len(region), len(group_trees)), dtype=float)
    for group_index, tree in enumerate(group_trees):
        distance, _nearest = tree.query(region, k=1)
        inside = distance <= radius
        affinity[inside, group_index] = np.exp(
            -0.5 * np.square(distance[inside]) / radius_squared
        )
    return affinity


def digital_group_localization_responsibilities(
    aligned_region: np.ndarray,
    lattice_points_nm: np.ndarray,
    digital_group_cells: Sequence[Sequence[int]],
    *,
    assignment_radius_nm: float,
) -> np.ndarray:
    """Return the exact per-localization group weights used by Step 3."""
    region = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(lattice_points_nm, dtype=float)
    if region.ndim != 2 or region.shape[1:] != (2,):
        raise ValueError("Aligned localizations must be an N-by-2 array.")
    if grid.ndim != 2 or grid.shape[1:] != (2,):
        raise ValueError("Lattice points must be an N-by-2 array.")
    radius = float(assignment_radius_nm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Digital-group assignment radius must be positive.")
    groups = [np.unique(np.asarray(cells, dtype=int)) for cells in digital_group_cells]
    if not groups or any(not len(cells) for cells in groups):
        raise ValueError("Every digital group must contain at least one position.")
    if any(np.any(cells < 0) or np.any(cells >= len(grid)) for cells in groups):
        raise ValueError("Digital-group position is outside the template lattice.")
    trees = [cKDTree(grid[cells]) for cells in groups]
    return _digital_group_responsibilities_from_trees(
        region, trees, radius, radius * radius
    )


def digital_group_template_evidence(
    digital_group_evidence: np.ndarray,
    active_digital_groups: Sequence[bool],
    digital_group_cells: Sequence[Sequence[int]],
    *,
    minimum_support_per_position: float = 0.0,
    minimum_group_prominence: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score a template using only one measured feature per digital group."""
    evidence = np.asarray(digital_group_evidence, dtype=float)
    active = np.asarray(active_digital_groups, dtype=bool)
    groups = [set(np.asarray(cells, dtype=int).tolist()) for cells in digital_group_cells]
    if evidence.ndim != 2:
        raise ValueError("Digital-group evidence must be a candidate-by-group matrix.")
    if evidence.shape[1] != len(groups) or active.shape != (len(groups),):
        raise ValueError("Digital-group evidence and template states have different sizes.")
    if not groups or any(not cells for cells in groups):
        raise ValueError("Every digital group must contain at least one position.")
    if not np.all(np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("Digital-group evidence must be finite and nonnegative.")
    minimum_support = float(minimum_support_per_position)
    minimum_prominence = float(minimum_group_prominence)
    if not np.isfinite(minimum_support) or minimum_support < 0.0:
        raise ValueError("Minimum digital-group support must be finite and nonnegative.")
    if not np.isfinite(minimum_prominence) or not 0.0 <= minimum_prominence <= 1.0:
        raise ValueError("Minimum digital-group prominence must be between zero and one.")
    candidate_count, group_count = evidence.shape
    if not candidate_count:
        return (
            np.empty(0),
            np.empty(0),
            np.empty((0, group_count)),
            np.empty(0),
        )

    # When the user supplies a visible support threshold, calibrate every group
    # independently against that common physical evidence scale.  Candidate-
    # relative normalization made strong/easy groups define the ON threshold
    # for weak/hard groups and created a systematic template bias.
    if minimum_support > 0.0:
        calibration_slope = math.log(19.0)
        log_odds = calibration_slope * (
            evidence / max(minimum_support, 1e-12) - 1.0
        )
        group_probability = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20.0, 20.0)))
    else:
        # Backward-compatible scale-free behavior for callers without a
        # support calibration (primarily legacy files and QC calculations).
        group_probability = None

    # Candidate-local separation now operates on the digital groups themselves.
    # Direct group assignment has a meaningful zero, so anchor the OFF state at
    # zero rather than forcing the dimmest observed group to be OFF. The latter
    # would make a uniformly populated all-ON template impossible to represent.
    intensity = np.log1p(evidence)
    low = np.zeros(candidate_count, dtype=float)
    high = np.max(intensity, axis=1)
    separable = high > low + 1e-12
    for _iteration in range(24):
        boundary = 0.5 * (low + high)
        dark = intensity <= boundary[:, None]
        bright_count = np.sum(~dark, axis=1)
        valid = separable & (bright_count > 0)
        next_high = np.divide(
            np.sum(np.where(~dark, intensity, 0.0), axis=1),
            bright_count,
            out=high.copy(),
            where=valid,
        )
        change = np.abs(next_high - high)
        high = np.where(valid, next_high, high)
        if not np.any(change[valid] >= 1e-8):
            break
    separation = np.maximum(high - low, 1e-6)
    boundary = 0.5 * (low + high)
    dark = intensity <= boundary[:, None]
    residual = np.where(dark, intensity - low[:, None], intensity - high[:, None])
    variance = np.maximum.reduce(
        (
            np.mean(np.square(residual), axis=1),
            np.square(separation / 4.0),
            np.full(candidate_count, 1e-6),
        )
    )
    log_odds = np.clip(
        separation[:, None] * (intensity - boundary[:, None]) / variance[:, None],
        -20.0,
        20.0,
    )
    relative_probability = 1.0 / (1.0 + np.exp(-log_odds))
    relative_probability[~separable] = 0.5
    if group_probability is None:
        group_probability = relative_probability
    group_probability = np.clip(group_probability, 1e-4, 1.0 - 1e-4)

    # Apply the user-visible Step 3 gates to the same probabilities consumed
    # by Step 4. Below-threshold evidence is smoothly capped below 0.5, rather
    # than being displayed as ON despite failing the configured minimum.
    raw_group_prominence = np.maximum(0.0, 2.0 * group_probability - 1.0)
    if minimum_support > 0.0:
        support_ceiling = np.where(
            evidence >= minimum_support,
            1.0,
            0.5 * np.clip(evidence / minimum_support, 0.0, 1.0),
        )
        group_probability = np.minimum(group_probability, support_ceiling)
    if minimum_prominence > 0.0:
        prominence_ceiling = np.where(
            raw_group_prominence >= minimum_prominence,
            1.0,
            0.5 * np.clip(raw_group_prominence / minimum_prominence, 0.0, 1.0),
        )
        group_probability = np.minimum(group_probability, prominence_ceiling)
    group_probability = np.clip(group_probability, 1e-4, 1.0 - 1e-4)

    # An inactive group cannot provide independent dark evidence where its
    # footprint overlaps an active group. This overlap correction uses only
    # the group definitions, never per-position localization measurements.
    active_union: set[int] = set()
    for cells, is_active in zip(groups, active):
        if is_active:
            active_union.update(cells)
    evidence_weight = np.ones(group_count, dtype=float)
    for group_index, (cells, is_active) in enumerate(zip(groups, active)):
        if not is_active:
            evidence_weight[group_index] = len(cells - active_union) / len(cells)
    agreement = np.where(active[None, :], group_probability, 1.0 - group_probability)
    log_bayes_factor = np.sum(
        evidence_weight[None, :] * (np.log(agreement) - math.log(0.5)), axis=1
    )
    posterior = 1.0 / (1.0 + np.exp(-np.clip(log_bayes_factor, -40.0, 40.0)))

    expected = active.astype(float) - float(np.mean(active))
    expected_norm = float(np.linalg.norm(expected))
    if expected_norm <= 1e-12:
        pattern_correlation = np.ones(candidate_count, dtype=float)
    else:
        observed = group_probability - np.mean(group_probability, axis=1, keepdims=True)
        denominator = np.linalg.norm(observed, axis=1) * expected_norm
        pattern_correlation = np.divide(
            observed @ expected,
            denominator,
            out=np.zeros(candidate_count, dtype=float),
            where=denominator > 1e-12,
        )
    return posterior, log_bayes_factor, group_probability, pattern_correlation


# Backward-compatible name from the original fixed-stroke implementation.
logical_stroke_template_evidence = logical_bit_template_evidence


def lattice_template_on_site_fractions(
    aligned_regions: list[np.ndarray],
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
) -> np.ndarray:
    """Measure signal assigned to the exact occupied cells of a grid template.

    The ordinary site mask may intentionally be wider than one lattice pitch
    so it remains tolerant during fit acceptance.  Such overlapping masks are
    unsuitable for distinguishing custom patterns: a point on an explicitly
    empty neighboring cell can otherwise count as on-template.  Nearest-cell
    assignment keeps that tolerance while preserving the template's black
    cells as negative evidence.
    """
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    if template.ndim != 2 or template.shape[1] != 2 or not len(template):
        raise ValueError("Lattice classification requires at least one 2D template point.")

    grid_tree = cKDTree(full_grid)
    template_distances, template_cells = grid_tree.query(template, k=1)
    mapping_tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(template_distances > mapping_tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    occupied_cells = np.zeros(len(full_grid), dtype=bool)
    occupied_cells[np.asarray(template_cells, dtype=int)] = True

    # Every point inside a lattice cell can be assigned even at a cell corner;
    # points farther away are in the footprint margin and count as off-template.
    maximum_cell_distance_nm = 0.5 * math.hypot(float(spacing_x_nm), float(spacing_y_nm))
    fractions = np.zeros(len(aligned_regions), dtype=float)
    for index, region in enumerate(aligned_regions):
        points = np.asarray(region, dtype=float)
        if not len(points):
            continue
        distances, cells = grid_tree.query(points, k=1)
        matches = (distances <= maximum_cell_distance_nm) & occupied_cells[np.asarray(cells, dtype=int)]
        fractions[index] = float(np.mean(matches))
    return fractions


def detected_lattice_template_agreement(
    lattice_supported_sites: np.ndarray,
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare a template with the same detected-site mask shown in the UI."""
    supported = np.asarray(lattice_supported_sites, dtype=bool)
    cell_count = int(rows) * int(columns)
    if supported.ndim != 2 or supported.shape[1] != cell_count:
        raise ValueError("Detected lattice-site masks must have one column per lattice cell.")
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    if template.ndim != 2 or template.shape[1] != 2 or not len(template):
        raise ValueError("Lattice classification requires at least one 2D template point.")
    distances, cells = cKDTree(full_grid).query(template, k=1)
    tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(distances > tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    expected = np.zeros(cell_count, dtype=bool)
    expected[np.asarray(cells, dtype=int)] = True
    black = ~expected
    bright_agreement = np.mean(supported[:, expected], axis=1)
    dark_agreement = (
        np.mean(~supported[:, black], axis=1)
        if np.any(black)
        else np.ones(len(supported), dtype=float)
    )
    expected_centered = expected.astype(float) - float(np.mean(expected))
    expected_norm = float(np.linalg.norm(expected_centered))
    if expected_norm <= 1e-12:
        pattern_correlation = np.ones(len(supported), dtype=float)
    else:
        observed = supported.astype(float)
        observed -= np.mean(observed, axis=1, keepdims=True)
        denominator = np.linalg.norm(observed, axis=1) * expected_norm
        pattern_correlation = np.divide(
            observed @ expected_centered,
            denominator,
            out=np.zeros(len(supported), dtype=float),
            where=denominator > 1e-12,
        )
    return bright_agreement, dark_agreement, pattern_correlation


@lru_cache(maxsize=128)
def _monte_carlo_template_log_probabilities(
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    occupied_cells: tuple[int, ...],
    simulation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic reusable bank of generative template models."""
    grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    sites = grid[np.asarray(occupied_cells, dtype=int)]
    # Stable geometry-derived seed makes reruns reproducible without making
    # different masks share identical nuisance samples.
    seed = int(
        (rows * 73_856_093)
        ^ (columns * 19_349_663)
        ^ sum((index + 1) * (cell + 17) for index, cell in enumerate(occupied_cells))
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    detection_probability = 0.10 + 0.88 * rng.beta(3.0, 2.0, simulation_count)
    background_probability = 0.001 + 0.18 * rng.beta(1.0, 14.0, simulation_count)
    minimum_spacing = min(float(spacing_x_nm), float(spacing_y_nm))
    localization_sigma_nm = rng.uniform(0.8, max(0.81, 0.32 * minimum_spacing), simulation_count)
    registration_offset_nm = rng.normal(0.0, 0.10 * minimum_spacing, size=(simulation_count, 2))
    probabilities = np.empty((simulation_count, len(grid)), dtype=float)
    for simulation in range(simulation_count):
        displaced_sites = sites + registration_offset_nm[simulation]
        distance_sq = np.sum(
            np.square(grid[:, None, :] - displaced_sites[None, :, :]),
            axis=2,
        )
        influence = np.max(
            np.exp(-distance_sq / (2.0 * localization_sigma_nm[simulation] ** 2)),
            axis=1,
        )
        probabilities[simulation] = background_probability[simulation] + (
            detection_probability[simulation] - background_probability[simulation]
        ) * influence
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities), np.log1p(-probabilities)


@lru_cache(maxsize=32)
def _monte_carlo_blob_log_probabilities(
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    simulation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reusable family of compact/elongated non-template blobs."""
    grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    rng = np.random.default_rng(
        int(rows * 83_492_791 + columns * 2_654_435_761) & 0xFFFFFFFF
    )
    span_x = max(float(np.ptp(grid[:, 0])), float(spacing_x_nm))
    span_y = max(float(np.ptp(grid[:, 1])), float(spacing_y_nm))
    scale = max(min(span_x, span_y), min(float(spacing_x_nm), float(spacing_y_nm)))
    detection_probability = 0.15 + 0.83 * rng.beta(3.0, 2.0, simulation_count)
    background_probability = 0.001 + 0.18 * rng.beta(1.0, 14.0, simulation_count)
    major_sigma = rng.uniform(0.12 * scale, 0.75 * scale, simulation_count)
    minor_sigma = major_sigma * rng.uniform(0.35, 1.0, simulation_count)
    angles = rng.uniform(0.0, math.pi, simulation_count)
    centers = np.column_stack(
        (
            rng.normal(0.0, 0.12 * span_x, simulation_count),
            rng.normal(0.0, 0.12 * span_y, simulation_count),
        )
    )
    probabilities = np.empty((simulation_count, len(grid)), dtype=float)
    for simulation in range(simulation_count):
        centered = grid - centers[simulation]
        cosine = math.cos(float(angles[simulation]))
        sine = math.sin(float(angles[simulation]))
        major = centered[:, 0] * cosine + centered[:, 1] * sine
        minor = -centered[:, 0] * sine + centered[:, 1] * cosine
        influence = np.exp(
            -0.5
            * (
                np.square(major / major_sigma[simulation])
                + np.square(minor / minor_sigma[simulation])
            )
        )
        probabilities[simulation] = background_probability[simulation] + (
            detection_probability[simulation] - background_probability[simulation]
        ) * influence
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities), np.log1p(-probabilities)


def monte_carlo_template_evidence(
    lattice_supported_sites: np.ndarray,
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    simulation_count: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate tolerant template-vs-blob evidence from detected site clusters.

    Samples span labeling efficiency, nonspecific detections, localization
    spread, and small registration offsets. Evidence uses a tempered spatial
    count likelihood: repeated localizations add information, while log-count
    compression and a bounded effective sample size avoid treating every blink
    as an independent molecule.

    The returned probability is the equal-prior logistic transform of the
    template-versus-null evidence. A value above 0.5 means the spatial
    pattern is closer to the simulated template family than to a smooth blob
    or an unstructured (zero-correlation) pattern.
    """
    if simulation_count < 16:
        raise ValueError("Monte Carlo template classification requires at least 16 simulations.")
    supported = np.asarray(lattice_supported_sites)
    cell_count = int(rows) * int(columns)
    if supported.ndim != 2 or supported.shape[1] != cell_count:
        raise ValueError("Lattice-site evidence must have one column per lattice cell.")
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    distances, cells = cKDTree(full_grid).query(template, k=1)
    tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(distances > tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    occupied_cells = tuple(sorted(set(int(cell) for cell in np.asarray(cells, dtype=int))))
    log_bright, log_dark = _monte_carlo_template_log_probabilities(
        int(rows),
        int(columns),
        float(spacing_x_nm),
        float(spacing_y_nm),
        occupied_cells,
        int(simulation_count),
    )
    if np.issubdtype(supported.dtype, np.bool_):
        observed = supported.astype(float)
    else:
        observed = np.asarray(supported, dtype=float)
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("Lattice-site evidence must be finite and nonnegative.")
        # Retain graded evidence from localization clusters without allowing a
        # single blinking-heavy site to dominate the complete spatial pattern.
        observed = np.log1p(observed)
    blob_log_bright, blob_log_dark = _monte_carlo_blob_log_probabilities(
        int(rows), int(columns), float(spacing_x_nm), float(spacing_y_nm), int(simulation_count)
    )
    del log_dark, blob_log_dark

    # Condition on the candidate's total signal and compare its spatial count
    # distribution with every simulated nuisance realization. log1p above
    # limits blinking-heavy sites; the square-root effective sample size lets
    # additional localizations increase certainty without treating repeated
    # blinks as independent molecules.
    observed_total = np.sum(observed, axis=1, keepdims=True)
    observed_distribution = np.divide(
        observed,
        observed_total,
        out=np.zeros_like(observed),
        where=observed_total > 0.0,
    )
    effective_count = np.clip(
        np.sqrt(np.sum(np.asarray(supported, dtype=float), axis=1)),
        1.0,
        32.0,
    )

    def normalized_log_cell_probabilities(log_probabilities: np.ndarray) -> np.ndarray:
        return log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)

    template_log_cells = normalized_log_cell_probabilities(log_bright)
    blob_log_cells = normalized_log_cell_probabilities(blob_log_bright)
    template_log_likelihood = effective_count[:, None] * (
        observed_distribution @ template_log_cells.T
    )
    blob_log_likelihood = effective_count[:, None] * (
        observed_distribution @ blob_log_cells.T
    )
    template_log_evidence = logsumexp(template_log_likelihood, axis=1) - math.log(simulation_count)
    blob_log_evidence = logsumexp(blob_log_likelihood, axis=1) - math.log(simulation_count)
    uniform_log_evidence = -effective_count * math.log(cell_count)
    null_log_evidence = np.maximum(uniform_log_evidence, blob_log_evidence)
    log_bayes_factor = template_log_evidence - null_log_evidence
    posterior = 1.0 / (1.0 + np.exp(-np.clip(log_bayes_factor, -30.0, 30.0)))
    no_spatial_information = observed_total[:, 0] <= 1e-12
    log_bayes_factor[no_spatial_information] = -30.0
    posterior[no_spatial_information] = 0.0
    return posterior, log_bayes_factor


def lattice_template_empty_cell_fractions(
    aligned_regions: list[np.ndarray],
    template_points_nm: np.ndarray,
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    min_site_localizations: int,
) -> np.ndarray:
    """Return the mean emptiness of cells designated black by a template."""
    if min_site_localizations < 1:
        raise ValueError("Minimum site localizations must be positive.")
    full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    template = np.asarray(template_points_nm, dtype=float)
    if template.ndim != 2 or template.shape[1] != 2 or not len(template):
        raise ValueError("Lattice classification requires at least one 2D template point.")
    grid_tree = cKDTree(full_grid)
    template_distances, template_cells = grid_tree.query(template, k=1)
    mapping_tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
    if np.any(template_distances > mapping_tolerance_nm):
        raise ValueError("Custom-template sites do not map to the configured row/column lattice.")
    occupied_cells = np.zeros(len(full_grid), dtype=bool)
    occupied_cells[np.asarray(template_cells, dtype=int)] = True
    black_cells = ~occupied_cells
    if not np.any(black_cells):
        return np.ones(len(aligned_regions), dtype=float)

    maximum_cell_distance_nm = 0.5 * math.hypot(float(spacing_x_nm), float(spacing_y_nm))
    fractions = np.ones(len(aligned_regions), dtype=float)
    for index, region in enumerate(aligned_regions):
        points = np.asarray(region, dtype=float)
        if not len(points):
            continue
        distances, cells = grid_tree.query(points, k=1)
        assigned_cells = np.asarray(cells[distances <= maximum_cell_distance_nm], dtype=int)
        counts = np.bincount(assigned_cells, minlength=len(full_grid))
        occupancy_evidence = np.clip(
            counts.astype(float) / float(min_site_localizations),
            0.0,
            1.0,
        )
        fractions[index] = float(np.mean(1.0 - occupancy_evidence[black_cells]))
    return fractions


def classify_template_candidates(
    candidate_centers_by_template: list[np.ndarray],
    accepted_masks: list[np.ndarray],
    correlations: list[np.ndarray],
    *,
    match_distance_nm: float,
    deduplication_distance_nm: float | None = None,
    minimum_winner_probability: float | None = None,
) -> TemplateClassificationResult:
    """Match duplicate detections across templates and select one accepted fit per object."""
    template_count = len(candidate_centers_by_template)
    if template_count < 1 or len(accepted_masks) != template_count or len(correlations) != template_count:
        raise ValueError("Classification requires at least one equally described template.")
    if match_distance_nm <= 0:
        raise ValueError("Template candidate match distance must be positive.")
    if deduplication_distance_nm is not None and deduplication_distance_nm <= 0:
        raise ValueError("Template deduplication distance must be positive.")
    if minimum_winner_probability is not None and not 0.0 <= minimum_winner_probability <= 1.0:
        raise ValueError("Minimum winner probability must be between zero and one.")

    centers_by_template: list[np.ndarray] = []
    for centers, accepted, scores in zip(candidate_centers_by_template, accepted_masks, correlations):
        centers = np.asarray(centers, dtype=float)
        accepted = np.asarray(accepted, dtype=bool)
        scores = np.asarray(scores, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Each template candidate-center array must be N x 2.")
        if len(centers) != len(accepted) or len(centers) != len(scores):
            raise ValueError("Candidate centers, acceptance masks, and correlations must have matching lengths.")
        centers_by_template.append(centers)

    # The normal multi-template path evaluates the exact same physical
    # candidates in the same order. Preserve those stable candidate IDs
    # directly; spatial rematching can otherwise depend on template order.
    shares_candidate_ids = all(
        len(centers) == len(centers_by_template[0])
        and np.allclose(centers, centers_by_template[0], rtol=0.0, atol=1e-9)
        for centers in centers_by_template[1:]
    )
    if shares_candidate_ids:
        groups = [
            {template_index: candidate_index for template_index in range(template_count)}
            for candidate_index in range(len(centers_by_template[0]))
        ]
        group_centers = [center.copy() for center in centers_by_template[0]]
        candidate_group_indices = [
            np.arange(len(groups), dtype=int) for _template_index in range(template_count)
        ]
        first_spatial_template = template_count
    else:
        # Compatibility fallback for callers that generated independent
        # candidate sets. Production classification avoids this path.
        groups = [{0: index} for index in range(len(centers_by_template[0]))]
        group_centers = [center.copy() for center in centers_by_template[0]]
        candidate_group_indices = [np.arange(len(centers_by_template[0]), dtype=int)]
        first_spatial_template = 1
    for template_index in range(first_spatial_template, template_count):
        centers = centers_by_template[template_index]
        template_group_indices = np.full(len(centers), -1, dtype=int)
        matched_candidates: set[int] = set()
        if groups and len(centers):
            distances = np.linalg.norm(
                np.asarray(group_centers, dtype=float)[:, None, :] - centers[None, :, :],
                axis=2,
            )
            group_indices, candidate_indices = linear_sum_assignment(distances)
            for group_index, candidate_index in zip(group_indices, candidate_indices):
                if distances[group_index, candidate_index] > match_distance_nm:
                    continue
                groups[int(group_index)][template_index] = int(candidate_index)
                template_group_indices[int(candidate_index)] = int(group_index)
                matched_candidates.add(int(candidate_index))
                member_centers = [
                    centers_by_template[member_template][member_index]
                    for member_template, member_index in groups[int(group_index)].items()
                ]
                group_centers[int(group_index)] = np.mean(member_centers, axis=0)
        for candidate_index, center in enumerate(centers):
            if candidate_index not in matched_candidates:
                template_group_indices[candidate_index] = len(groups)
                groups.append({template_index: candidate_index})
                group_centers.append(center.copy())
        candidate_group_indices.append(template_group_indices)

    assignment_masks = [np.zeros(len(centers), dtype=bool) for centers in centers_by_template]
    raw_template_probabilities = np.zeros((len(groups), template_count), dtype=float)
    template_probabilities = np.zeros((len(groups), template_count), dtype=float)
    for group_index, group in enumerate(groups):
        group_scores = np.full(template_count, -float("inf"), dtype=float)
        eligible = np.zeros(template_count, dtype=bool)
        for template_index, candidate_index in group.items():
            group_scores[template_index] = float(correlations[template_index][candidate_index])
            eligible[template_index] = bool(accepted_masks[template_index][candidate_index])
        finite = np.isfinite(group_scores)
        if np.any(finite):
            shifted = group_scores[finite] - float(np.max(group_scores[finite]))
            weights = np.exp(np.clip(shifted, -700.0, 0.0))
            raw_template_probabilities[group_index, finite] = weights / float(np.sum(weights))
        eligible &= finite
        if np.any(eligible):
            shifted = group_scores[eligible] - float(np.max(group_scores[eligible]))
            weights = np.exp(np.clip(shifted, -700.0, 0.0))
            template_probabilities[group_index, eligible] = weights / float(np.sum(weights))
    winning_template_indices = np.full(len(groups), -1, dtype=int)
    winning_scores = np.full(len(groups), -float("inf"), dtype=float)
    runner_up_template_indices = np.full(len(groups), -1, dtype=int)
    runner_up_scores = np.full(len(groups), -float("inf"), dtype=float)
    winning_score_margins = np.full(len(groups), np.nan, dtype=float)
    winning_candidates: list[tuple[int, int] | None] = [None] * len(groups)
    for group_index, group in enumerate(groups):
        passing = [
            (float(correlations[template_index][candidate_index]), template_index, candidate_index)
            for template_index, candidate_index in group.items()
            if bool(accepted_masks[template_index][candidate_index])
        ]
        if not passing:
            continue
        _score, template_index, candidate_index = max(passing)
        # This probability is normalized across every loaded template with
        # equal priors.  Applying the gate here lets templates compete first
        # and prevents the subsequent duplicate suppression from discarding
        # a neighbor on behalf of a fit that will ultimately be rejected.
        if (
            minimum_winner_probability is not None
            and raw_template_probabilities[group_index, template_index] + 1e-9
            < minimum_winner_probability
        ):
            continue
        assignment_masks[template_index][candidate_index] = True
        winning_template_indices[group_index] = template_index
        winning_scores[group_index] = _score
        winning_candidates[group_index] = (template_index, candidate_index)
        competing = sorted(
            (
                (float(correlations[other_template][other_candidate]), other_template)
                for other_template, other_candidate in group.items()
                if other_template != template_index
                and bool(accepted_masks[other_template][other_candidate])
                and np.isfinite(correlations[other_template][other_candidate])
            ),
            reverse=True,
        )
        if competing:
            runner_up_scores[group_index], runner_up_template_indices[group_index] = competing[0]
            winning_score_margins[group_index] = _score - runner_up_scores[group_index]
        else:
            winning_score_margins[group_index] = float("inf")

    suppressed_duplicate_count = 0
    if deduplication_distance_nm is not None:
        retained_groups: list[int] = []
        passing_groups = np.flatnonzero(winning_template_indices >= 0)
        for group_index in passing_groups[np.argsort(-winning_scores[passing_groups], kind="stable")]:
            if any(
                np.linalg.norm(group_centers[int(group_index)] - group_centers[retained])
                < deduplication_distance_nm
                for retained in retained_groups
            ):
                winner = winning_candidates[int(group_index)]
                assert winner is not None
                assignment_masks[winner[0]][winner[1]] = False
                winning_template_indices[int(group_index)] = -2
                suppressed_duplicate_count += 1
            else:
                retained_groups.append(int(group_index))

    counts = np.asarray([np.count_nonzero(mask) for mask in assignment_masks], dtype=int)
    return TemplateClassificationResult(
        assignment_masks=assignment_masks,
        counts=counts,
        unclassified_count=int(np.count_nonzero(winning_template_indices == -1)),
        suppressed_duplicate_count=suppressed_duplicate_count,
        group_centers_nm=np.asarray(group_centers, dtype=float).reshape(-1, 2),
        winning_template_indices=winning_template_indices,
        candidate_group_indices=candidate_group_indices,
        winning_scores=winning_scores,
        runner_up_template_indices=runner_up_template_indices,
        runner_up_scores=runner_up_scores,
        winning_score_margins=winning_score_margins,
        template_probabilities=template_probabilities,
        raw_template_probabilities=raw_template_probabilities,
    )


def ideal_grid_points(rows: int, columns: int, spacing_x_nm: float, spacing_y_nm: float, column_offsets_nm: Sequence[float] | None = None) -> np.ndarray:
    if rows < 1 or columns < 1:
        raise ValueError("Grid rows and columns must both be at least 1.")
    if spacing_x_nm <= 0 or spacing_y_nm <= 0:
        raise ValueError("Grid spacing must be greater than zero.")
    x = (np.arange(columns, dtype=float) - (columns - 1) / 2.0) * spacing_x_nm
    if column_offsets_nm is not None and len(column_offsets_nm):
        offsets = np.asarray(column_offsets_nm, dtype=float)
        if offsets.shape != (columns,) or not np.all(np.isfinite(offsets)):
            raise ValueError("Column offsets must contain one finite value per column.")
        x += offsets
        if np.any(np.diff(x) <= 0):
            raise ValueError("Column positions must be strictly increasing.")
        x -= (x[0] + x[-1]) / 2.0
    y = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * spacing_y_nm
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def render_aligned_origami_density(
    aligned_points: list[np.ndarray],
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    pixel_size_nm: float,
    padding_nm: float,
    blur_nm: float,
    max_pixels: int = 4_000_000,
    symmetrize_180: bool = False,
    chunk_origamis: int = 256,
) -> dict[str, object]:
    """Render aligned points with bounded working memory at a known resolution."""
    if not aligned_points:
        raise ValueError("No aligned origami points are available to render.")
    if pixel_size_nm <= 0:
        raise ValueError("Overlay pixel size must be greater than zero.")
    if padding_nm < 0 or blur_nm < 0:
        raise ValueError("Overlay padding and blur cannot be negative.")
    if chunk_origamis < 1:
        raise ValueError("Render chunk size must be at least one origami.")
    grid_width_nm = max(spacing_x_nm, (columns - 1) * spacing_x_nm)
    grid_height_nm = max(spacing_y_nm, (rows - 1) * spacing_y_nm)
    width_nm = grid_width_nm + 2.0 * padding_nm
    height_nm = grid_height_nm + 2.0 * padding_nm
    x_bins = max(1, int(np.ceil(width_nm / pixel_size_nm)))
    y_bins = max(1, int(np.ceil(height_nm / pixel_size_nm)))
    if x_bins * y_bins > max_pixels:
        scale = np.sqrt((x_bins * y_bins) / max_pixels)
        x_bins = max(1, int(np.floor(x_bins / scale)))
        y_bins = max(1, int(np.floor(y_bins / scale)))
    effective_x_nm = width_nm / x_bins
    effective_y_nm = height_nm / y_bins
    extent = (-width_nm / 2.0, width_nm / 2.0, -height_nm / 2.0, height_nm / 2.0)
    density = np.zeros((x_bins, y_bins), dtype=float)
    rendered_point_count = 0
    total_point_count = 0
    orientation_count = 2 if symmetrize_180 else 1
    for start in range(0, len(aligned_points), chunk_origamis):
        chunk = np.vstack(aligned_points[start : start + chunk_origamis])
        orientations = (chunk, -chunk) if symmetrize_180 else (chunk,)
        for points in orientations:
            chunk_density, _x_edges, _y_edges = np.histogram2d(
                points[:, 0],
                points[:, 1],
                bins=(x_bins, y_bins),
                range=((extent[0], extent[1]), (extent[2], extent[3])),
            )
            density += chunk_density
            in_view = (
                (points[:, 0] >= extent[0])
                & (points[:, 0] <= extent[1])
                & (points[:, 1] >= extent[2])
                & (points[:, 1] <= extent[3])
            )
            rendered_point_count += int(np.count_nonzero(in_view))
            total_point_count += int(len(points))
    density /= len(aligned_points) * orientation_count
    if blur_nm > 0:
        density = gaussian_filter(
            density,
            sigma=(blur_nm / effective_x_nm, blur_nm / effective_y_nm),
            mode="constant",
        )
    return {
        "image": density.T,
        "extent": extent,
        "effective_pixel_x_nm": float(effective_x_nm),
        "effective_pixel_y_nm": float(effective_y_nm),
        "rendered_point_count": rendered_point_count,
        "total_point_count": total_point_count,
        "blur_nm": float(blur_nm),
    }


def origami_gallery_indices(
    result: OrigamiAnalysisResult,
    sort_mode: str = "Origami ID",
    min_grid_match_fraction: float = 0.0,
    max_alignment_rms_nm: float = math.inf,
) -> np.ndarray:
    """Return stable result indices for a filtered, quality-sorted gallery."""
    if not 0.0 <= min_grid_match_fraction <= 1.0:
        raise ValueError("Minimum grid-match fraction must be between zero and one.")
    if max_alignment_rms_nm <= 0:
        raise ValueError("Maximum alignment RMS must be positive.")
    indices = np.flatnonzero(
        (result.grid_match_fraction >= min_grid_match_fraction)
        & (result.alignment_rms_nm <= max_alignment_rms_nm)
    )
    if sort_mode == "Origami ID":
        order = indices
    elif sort_mode == "Source position":
        local_order = np.lexsort((result.centers_nm[indices, 0], result.centers_nm[indices, 1]))
        order = indices[local_order]
    elif sort_mode == "Most localizations":
        order = indices[np.argsort(-result.source_point_counts[indices], kind="stable")]
    elif sort_mode == "Highest alignment RMS":
        order = indices[np.argsort(-result.alignment_rms_nm[indices], kind="stable")]
    elif sort_mode == "Lowest grid match":
        order = indices[np.argsort(result.grid_match_fraction[indices], kind="stable")]
    elif sort_mode == "Fewest occupied sites":
        occupied = np.sum(result.site_occupancy[indices], axis=1)
        order = indices[np.argsort(occupied, kind="stable")]
    elif sort_mode == "QC sample":
        if not len(indices):
            order = indices
        else:
            occupied = np.sum(result.site_occupancy[indices], axis=1)
            rankings = (
                indices,
                indices[np.argsort(result.source_point_counts[indices], kind="stable")],
                indices[np.argsort(result.alignment_rms_nm[indices], kind="stable")],
                indices[np.argsort(result.grid_match_fraction[indices], kind="stable")],
                indices[np.argsort(occupied, kind="stable")],
            )
            sample_positions = np.linspace(0, len(indices) - 1, min(32, len(indices)), dtype=int)
            representative: list[int] = []
            seen: set[int] = set()
            for position in sample_positions:
                for ranking in rankings:
                    candidate = int(ranking[position])
                    if candidate not in seen:
                        representative.append(candidate)
                        seen.add(candidate)
            order = np.asarray(representative, dtype=int)
    else:
        raise ValueError(f"Unknown origami gallery sort: {sort_mode}")
    return np.asarray(order, dtype=int)


def origami_gallery_page(indices: np.ndarray, page_number: int, page_size: int) -> tuple[np.ndarray, int, int]:
    """Return one one-based gallery page plus normalized page metadata."""
    if page_size < 1:
        raise ValueError("Gallery page size must be at least one.")
    indices = np.asarray(indices, dtype=int)
    page_count = max(1, int(np.ceil(len(indices) / page_size)))
    normalized_page = min(max(1, int(page_number)), page_count)
    start = (normalized_page - 1) * page_size
    return indices[start : start + page_size], normalized_page, page_count


def integrate_rendered_density_at_sites(
    rendered_density: dict[str, object],
    grid_points_nm: np.ndarray,
    site_radius_nm: float,
) -> np.ndarray:
    """Integrate a rendered mean-density image inside circles around grid sites."""
    if site_radius_nm <= 0:
        raise ValueError("Site radius must be greater than zero.")
    image = np.asarray(rendered_density["image"], dtype=float)
    if image.ndim != 2 or not image.size:
        raise ValueError("Rendered density must contain a non-empty 2D image.")
    grid_points_nm = np.asarray(grid_points_nm, dtype=float)
    if grid_points_nm.ndim != 2 or grid_points_nm.shape[1] != 2:
        raise ValueError("Grid points must be an N x 2 coordinate array.")
    x_min, x_max, y_min, y_max = (float(value) for value in rendered_density["extent"])
    pixel_x_nm = (x_max - x_min) / image.shape[1]
    pixel_y_nm = (y_max - y_min) / image.shape[0]
    x_centers = x_min + (np.arange(image.shape[1]) + 0.5) * pixel_x_nm
    y_centers = y_min + (np.arange(image.shape[0]) + 0.5) * pixel_y_nm
    xx, yy = np.meshgrid(x_centers, y_centers)
    radius_squared = float(site_radius_nm) ** 2
    return np.asarray([
        float(np.sum(image[np.square(xx - point[0]) + np.square(yy - point[1]) <= radius_squared]))
        for point in grid_points_nm
    ])


def fit_picasso_g5m_components(
    points_nm: np.ndarray,
    *,
    min_locs: int,
    sigma_min_nm: float,
    sigma_max_nm: float,
    max_rounds_without_best_bic: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Picasso's exact 2D G5M core to one pre-isolated origami."""
    if min_locs < 1 or max_rounds_without_best_bic < 1:
        raise ValueError("G5M minimum localizations and BIC patience must be at least 1.")
    if sigma_min_nm <= 0 or sigma_max_nm < sigma_min_nm:
        raise ValueError("G5M sigma bounds must be positive and ordered minimum to maximum.")
    points_nm = np.ascontiguousarray(np.asarray(points_nm, dtype=np.float64))
    if len(points_nm) < min_locs:
        return np.full(len(points_nm), -1, dtype=int), np.empty((0, 2), dtype=float)

    from picasso import g5m as picasso_g5m

    model = picasso_g5m._find_optimal_G5M_2D(
        points_nm,
        min_locs=int(min_locs),
        sigma_bounds=(float(sigma_min_nm), float(sigma_max_nm)),
        lp=np.ones(len(points_nm), dtype=np.float64),
        loc_prec_handle="abs",
        max_rounds_without_best_bic=int(max_rounds_without_best_bic),
    )
    if model is None or len(model.valid_idx) == 0:
        return np.full(len(points_nm), -1, dtype=int), np.empty((0, 2), dtype=float)
    return np.asarray(model.predict(points_nm), dtype=int), np.asarray(model.means, dtype=float)


def _pca_aligned_candidates(
    points_nm: np.ndarray,
    alignment_points_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    allow_mirror: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    center = np.mean(alignment_points_nm, axis=0)
    centered = points_nm - center
    centered_alignment = alignment_points_nm - center
    covariance = np.cov(centered_alignment, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor = np.asarray([-major[1], major[0]])
    base = np.column_stack([centered @ major, centered @ minor])
    base_alignment = np.column_stack([centered_alignment @ major, centered_alignment @ minor])

    grid_extent = np.ptp(grid_points_nm, axis=0)
    if grid_extent[1] > grid_extent[0]:
        base = base[:, [1, 0]]
        base_alignment = base_alignment[:, [1, 0]]

    candidates = [(base, base_alignment), (-base, -base_alignment)]
    if allow_mirror:
        mirrored = base * np.asarray([-1.0, 1.0])
        mirrored_alignment = base_alignment * np.asarray([-1.0, 1.0])
        candidates.extend([(mirrored, mirrored_alignment), (-mirrored, -mirrored_alignment)])
    return candidates


def _rectangle_aligned_candidates(
    points_nm: np.ndarray,
    alignment_points_nm: np.ndarray,
    rectangle_corners_nm: np.ndarray,
    allow_mirror: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    center = np.mean(rectangle_corners_nm, axis=0)
    x_axis = rectangle_corners_nm[1] - rectangle_corners_nm[0]
    y_axis = rectangle_corners_nm[3] - rectangle_corners_nm[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack([x_axis, y_axis])
    base = (points_nm - center) @ basis
    base_alignment = (alignment_points_nm - center) @ basis
    candidates = [(base, base_alignment), (-base, -base_alignment)]
    if allow_mirror:
        mirror = np.asarray([-1.0, 1.0])
        candidates.extend([(base * mirror, base_alignment * mirror), (-base * mirror, -base_alignment * mirror)])
    return candidates


def _translated_grid_score(
    alignment_points_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    site_radius_nm: float,
) -> tuple[np.ndarray, tuple[int, int, float]]:
    shifted = alignment_points_nm.copy()
    for _ in range(6):
        distances = np.linalg.norm(shifted[:, None, :] - grid_points_nm[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        residuals = shifted - grid_points_nm[nearest]
        nearest_distance = distances[np.arange(len(shifted)), nearest]
        inliers = nearest_distance <= max(site_radius_nm * 2.0, 1.0)
        if not np.any(inliers):
            inliers = nearest_distance <= np.percentile(nearest_distance, 60.0)
        translation = np.median(residuals[inliers], axis=0)
        shifted -= translation
        if float(np.linalg.norm(translation)) < 0.005:
            break
    distances = np.linalg.norm(shifted[:, None, :] - grid_points_nm[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(shifted)), nearest]
    matched = nearest_distance <= site_radius_nm
    unique_sites = len(np.unique(nearest[matched])) if np.any(matched) else 0
    rms = float(np.sqrt(np.mean(np.square(nearest_distance[matched])))) if np.any(matched) else float("inf")
    translation = np.median(alignment_points_nm - shifted, axis=0)
    return translation, (unique_sites, int(np.count_nonzero(matched)), -rms)


def _refine_rotation_to_grid(
    points_nm: np.ndarray,
    alignment_points_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    site_radius_nm: float,
) -> np.ndarray:
    best_angle = 0.0
    best_translation = np.zeros(2, dtype=float)
    best_score = (-1, -1, -float("inf"))

    def consider(angle_degrees: float) -> None:
        nonlocal best_angle, best_translation, best_score
        angle = np.deg2rad(angle_degrees)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_alignment = alignment_points_nm @ rotation.T
        translation, score = _translated_grid_score(rotated_alignment, grid_points_nm, site_radius_nm)
        if score > best_score:
            best_angle = angle_degrees
            best_translation = translation
            best_score = score

    for angle_degrees in np.linspace(-45.0, 45.0, 91):
        consider(float(angle_degrees))
    coarse_angle = best_angle
    for angle_degrees in np.linspace(coarse_angle - 1.0, coarse_angle + 1.0, 21):
        consider(float(angle_degrees))

    angle = np.deg2rad(best_angle)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points_nm @ rotation.T - best_translation


def _fit_translation_and_sites(
    points_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    site_radius_nm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    shifted = points_nm.copy()
    for _ in range(4):
        distances = np.linalg.norm(shifted[:, None, :] - grid_points_nm[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        residuals = shifted - grid_points_nm[nearest]
        inliers = np.linalg.norm(residuals, axis=1) <= max(site_radius_nm * 2.0, 1.0)
        if not np.any(inliers):
            break
        translation = np.median(residuals[inliers], axis=0)
        shifted -= translation
        if float(np.linalg.norm(translation)) < 0.01:
            break

    distances = np.linalg.norm(shifted[:, None, :] - grid_points_nm[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(shifted)), nearest]
    accepted = nearest_distance <= site_radius_nm
    counts = np.bincount(nearest[accepted], minlength=len(grid_points_nm)).astype(int)
    if np.any(accepted):
        rms = float(np.sqrt(np.mean(np.square(nearest_distance[accepted]))))
    else:
        rms = float("inf")
    return shifted, counts, rms


def cluster_aligned_origami_sites(
    points_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    g5m_sigma_min_nm: float,
    g5m_sigma_max_nm: float,
    g5m_min_locs: int,
    g5m_max_rounds_without_best_bic: int,
    site_match_radius_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Picasso G5M to one aligned origami and assign components to grid sites."""
    if site_match_radius_nm <= 0:
        raise ValueError("Site-match radius must be greater than zero.")

    points_nm = np.asarray(points_nm, dtype=float)
    raw_labels, raw_centers = fit_picasso_g5m_components(
        points_nm,
        min_locs=g5m_min_locs,
        sigma_min_nm=g5m_sigma_min_nm,
        sigma_max_nm=g5m_sigma_max_nm,
        max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
    )
    display_labels = np.full(len(points_nm), -1, dtype=int)
    counts = np.zeros(len(grid_points_nm), dtype=int)
    centers: list[np.ndarray] = []
    site_indices: list[int] = []
    for raw_label, center in enumerate(raw_centers):
        members = raw_labels == raw_label
        distances = np.linalg.norm(grid_points_nm - center, axis=1)
        site_index = int(np.argmin(distances))
        if float(distances[site_index]) > site_match_radius_nm:
            continue
        display_label = len(centers)
        display_labels[members] = display_label
        centers.append(center)
        site_indices.append(site_index)
        counts[site_index] += int(np.count_nonzero(members))

    center_array = np.vstack(centers) if centers else np.empty((0, 2), dtype=float)
    return counts, display_labels, center_array, np.asarray(site_indices, dtype=int)


def density_map_for_origami_picking(
    points_nm: np.ndarray,
    bin_size_nm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], np.ndarray, np.ndarray]:
    """Build a lightly smoothed density map with Picasso-style auto contrast."""
    points_nm = np.asarray(points_nm, dtype=float)
    x_min, y_min = np.floor(np.min(points_nm, axis=0) / bin_size_nm) * bin_size_nm
    x_max, y_max = np.ceil(np.max(points_nm, axis=0) / bin_size_nm) * bin_size_nm
    if x_max <= x_min:
        x_max = x_min + bin_size_nm
    if y_max <= y_min:
        y_max = y_min + bin_size_nm
    x_edges = np.arange(x_min, x_max + bin_size_nm * 1.01, bin_size_nm)
    y_edges = np.arange(y_min, y_max + bin_size_nm * 1.01, bin_size_nm)
    density, _x_edges, _y_edges = np.histogram2d(points_nm[:, 0], points_nm[:, 1], bins=(x_edges, y_edges))
    density = gaussian_filter(density.astype(float), sigma=1.0, mode="constant")
    # A single aggregate should not redefine the detection threshold for every
    # other object in a large ROI. Retain the historical behavior for tiny
    # diagnostic maps, but use a robust high percentile for normal images.
    if density.size >= 400:
        local_peaks = density[
            (density > 0.0)
            & (density >= maximum_filter(density, size=3, mode="constant"))
        ]
        reference_peak = (
            float(np.quantile(local_peaks, 0.90))
            if len(local_peaks) >= 5
            else float(np.max(density))
        )
    else:
        reference_peak = float(np.max(density)) if density.size else 0.0
    auto_max = 0.5 * reference_peak
    if auto_max <= 0:
        auto_max = 1.0
    contrast = np.clip(density / auto_max, 0.0, 1.0)
    return density, contrast, (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])), x_edges, y_edges


def render_localization_preview(
    points_nm: np.ndarray,
    *,
    pixel_size_nm: float = 1.0,
    blur_nm: float = 1.0,
    max_pixels: int = 4_000_000,
) -> dict[str, object]:
    """Render original localization coordinates independently of the coarse picker grid."""
    points_nm = np.asarray(points_nm, dtype=float)
    points_nm = points_nm[np.all(np.isfinite(points_nm), axis=1)]
    if not len(points_nm):
        raise ValueError("There are no finite localizations to preview.")
    if pixel_size_nm <= 0 or blur_nm < 0:
        raise ValueError("Preview pixel size must be positive and blur cannot be negative.")
    lower = np.floor(np.min(points_nm, axis=0) / pixel_size_nm) * pixel_size_nm
    upper = np.ceil(np.max(points_nm, axis=0) / pixel_size_nm) * pixel_size_nm
    span = np.maximum(upper - lower, pixel_size_nm)
    x_bins = max(1, int(np.ceil(span[0] / pixel_size_nm)))
    y_bins = max(1, int(np.ceil(span[1] / pixel_size_nm)))
    if x_bins * y_bins > max_pixels:
        scale = np.sqrt((x_bins * y_bins) / max_pixels)
        x_bins = max(1, int(np.floor(x_bins / scale)))
        y_bins = max(1, int(np.floor(y_bins / scale)))
    effective_x_nm = float(span[0] / x_bins)
    effective_y_nm = float(span[1] / y_bins)
    density, _x_edges, _y_edges = np.histogram2d(
        points_nm[:, 0],
        points_nm[:, 1],
        bins=(x_bins, y_bins),
        range=((lower[0], upper[0]), (lower[1], upper[1])),
    )
    if blur_nm > 0:
        density = gaussian_filter(
            density,
            sigma=(blur_nm / effective_x_nm, blur_nm / effective_y_nm),
            mode="constant",
        )
    auto_max = 0.5 * float(np.max(density)) if density.size else 1.0
    if auto_max <= 0:
        auto_max = 1.0
    return {
        "contrast": np.clip(density.T / auto_max, 0.0, 1.0),
        "extent": (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])),
        "effective_pixel_x_nm": effective_x_nm,
        "effective_pixel_y_nm": effective_y_nm,
        "blur_nm": float(blur_nm),
    }


def _regular_grid_active_components(
    active_mask: np.ndarray,
    *,
    bin_size_nm: float,
    connect_distance_nm: float,
) -> tuple[np.ndarray, int]:
    """Label active coarse bins without materializing every nearby-bin pair.

    The bins lie on a regular lattice. First label directly adjacent bins, then
    merge those compact components only where a longer permitted lattice offset
    bridges a gap. This preserves the Euclidean connection rule while avoiding
    the potentially enormous KD-tree pair list produced by dense/noisy maps.
    """
    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 2:
        raise ValueError("The coarse active-bin mask must be two-dimensional.")
    if bin_size_nm <= 0.0 or connect_distance_nm <= 0.0:
        raise ValueError("Coarse component distances must be positive.")
    structure = np.zeros((3, 3), dtype=np.uint8)
    structure[1, 1] = 1
    tolerance = 1e-12 * max(1.0, float(connect_distance_nm))
    for delta_x in range(-1, 2):
        for delta_y in range(-1, 2):
            if math.hypot(delta_x, delta_y) * bin_size_nm <= connect_distance_nm + tolerance:
                structure[delta_x + 1, delta_y + 1] = 1
    base_labels, base_count = label(active, structure=structure)
    if base_count == 0:
        return np.full(active.shape, -1, dtype=np.int32), 0

    maximum_offset = int(math.floor((connect_distance_nm + tolerance) / bin_size_nm))
    encoded_edges: list[np.ndarray] = []
    size_x, size_y = active.shape
    for delta_x in range(0, maximum_offset + 1):
        for delta_y in range(-maximum_offset, maximum_offset + 1):
            if delta_x == 0 and delta_y <= 0:
                continue
            if math.hypot(delta_x, delta_y) * bin_size_nm > connect_distance_nm + tolerance:
                continue
            if abs(delta_x) <= 1 and abs(delta_y) <= 1 and structure[delta_x + 1, delta_y + 1]:
                continue
            if delta_x >= size_x or abs(delta_y) >= size_y:
                continue
            left_x = slice(0, size_x - delta_x)
            right_x = slice(delta_x, size_x)
            if delta_y >= 0:
                left_y = slice(0, size_y - delta_y)
                right_y = slice(delta_y, size_y)
            else:
                left_y = slice(-delta_y, size_y)
                right_y = slice(0, size_y + delta_y)
            left = base_labels[left_x, left_y]
            right = base_labels[right_x, right_y]
            bridges = (left > 0) & (right > 0) & (left != right)
            if not np.any(bridges):
                continue
            first = left[bridges].astype(np.int64) - 1
            second = right[bridges].astype(np.int64) - 1
            low = np.minimum(first, second)
            high = np.maximum(first, second)
            encoded_edges.append(np.unique(low * base_count + high))

    if encoded_edges:
        edge_codes = np.unique(np.concatenate(encoded_edges))
        adjacency = csr_matrix(
            (
                np.ones(len(edge_codes), dtype=np.uint8),
                (edge_codes // base_count, edge_codes % base_count),
            ),
            shape=(base_count, base_count),
        )
        component_count, base_components = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
    else:
        component_count = base_count
        base_components = np.arange(base_count, dtype=np.int32)
    output = np.full(active.shape, -1, dtype=np.int32)
    output[active] = base_components[base_labels[active] - 1]
    return output, int(component_count)


def _pick_origami_regions(
    points_nm: np.ndarray,
    bin_size_nm: float,
    connect_distance_nm: float,
    density_threshold: float,
    *,
    component_connect_distance_nm: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """Connect supported bins, then recover original points near each object."""
    density, contrast, extent, x_edges, y_edges = density_map_for_origami_picking(points_nm, bin_size_nm)
    component_labels = np.full(contrast.shape, -1, dtype=np.int32)
    active_indices = np.argwhere(contrast >= density_threshold)
    if progress_callback is not None:
        progress_callback(
            25.0,
            f"Density binning complete: {len(active_indices):,} active bins; connecting components…",
        )
    if len(active_indices) == 0:
        return [], density, contrast, extent, component_labels
    cell_centers = np.column_stack(
        [
            x_edges[active_indices[:, 0]] + bin_size_nm / 2.0,
            y_edges[active_indices[:, 1]] + bin_size_nm / 2.0,
        ]
    )
    if len(cell_centers) == 1:
        distances = np.linalg.norm(points_nm - cell_centers[0], axis=1)
        region = points_nm[distances <= connect_distance_nm]
        if len(region):
            component_labels[tuple(active_indices[0])] = 0
            return [region], density, contrast, extent, component_labels
        return [], density, contrast, extent, component_labels

    component_connect_distance = (
        float(connect_distance_nm)
        if component_connect_distance_nm is None
        else max(float(connect_distance_nm), float(component_connect_distance_nm))
    )
    cell_components, _component_count = _regular_grid_active_components(
        contrast >= density_threshold,
        bin_size_nm=float(bin_size_nm),
        connect_distance_nm=component_connect_distance,
    )
    cell_components = cell_components[active_indices[:, 0], active_indices[:, 1]]
    if progress_callback is not None:
        progress_callback(
            60.0,
            f"Connected {len(active_indices):,} active bins into {_component_count:,} components; recovering source points…",
        )
    nearest_distance, nearest_cell = cKDTree(cell_centers).query(points_nm, k=1)
    assigned = nearest_distance <= connect_distance_nm
    point_components = cell_components[nearest_cell[assigned]]
    assigned_points = points_nm[assigned]
    order = np.argsort(point_components, kind="stable")
    sorted_components = point_components[order]
    boundaries = np.flatnonzero(np.diff(sorted_components)) + 1
    regions = [region for region in np.split(assigned_points[order], boundaries) if len(region)]
    present_components = np.unique(sorted_components)
    component_to_display = np.full(_component_count, -1, dtype=np.int32)
    component_to_display[present_components] = np.arange(
        len(present_components), dtype=np.int32
    )
    # Assign every active bin in one indexed operation. Iterating over
    # components repeatedly rescanned the entire active-bin array and became
    # another large cost on noisy whole-image sources.
    component_labels[active_indices[:, 0], active_indices[:, 1]] = (
        component_to_display[cell_components]
    )
    if progress_callback is not None:
        progress_callback(100.0, f"Recovered source points for {len(regions):,} coarse components.")
    return regions, density, contrast, extent, component_labels


def pick_origami_candidates(
    points_nm: np.ndarray,
    *,
    bin_size_nm: float,
    connect_distance_nm: float,
    density_threshold: float,
    minimum_points: int = 1,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """Generate physical candidates, omitting components that cannot meet a point minimum."""
    points = np.asarray(points_nm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Origami candidate generation requires an N x 2 coordinate array.")
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError("There are no finite source points for origami candidate generation.")
    if bin_size_nm <= 0.0 or connect_distance_nm <= 0.0:
        raise ValueError("Pick-bin size and connection distance must be positive.")
    if not 0.0 <= density_threshold <= 1.0:
        raise ValueError("Minimum density contrast must be between 0 and 1.")
    if minimum_points < 1:
        raise ValueError("Minimum candidate points must be at least one.")
    if progress_callback is not None:
        progress_callback(5.0, "Binning source points for coarse candidate detection…")
    candidates = _pick_origami_regions(
        points,
        float(bin_size_nm),
        float(connect_distance_nm),
        float(density_threshold),
        component_connect_distance_nm=None,
        progress_callback=(
            (lambda percent, message: progress_callback(5.0 + 0.8 * percent, message))
            if progress_callback is not None
            else None
        ),
    )
    if progress_callback is not None:
        progress_callback(85.0, "Filtering small connected components…")
    filtered = filter_origami_candidates(candidates, minimum_points)[0]
    if progress_callback is not None:
        progress_callback(100.0, f"Retained {len(filtered[0]):,} coarse candidates.")
    return filtered


def filter_origami_candidates(
    candidates: tuple[
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        tuple[float, float, float, float],
        np.ndarray,
    ],
    minimum_points: int,
) -> tuple[
    tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[float, float, float, float], np.ndarray],
    int,
]:
    """Drop connected components that cannot possibly pass the post-crop minimum.

    Alignment crops points from a component but never adds them. Therefore an
    original component below ``minimum_points`` is guaranteed to fail and can
    be removed before thumbnail rendering and pose search.
    """
    if minimum_points < 1:
        raise ValueError("Minimum candidate points must be at least one.")
    regions, density, contrast, extent, component_labels = candidates
    keep_indices = [index for index, region in enumerate(regions) if len(region) >= minimum_points]
    rejected_count = len(regions) - len(keep_indices)
    if rejected_count == 0:
        return candidates, 0
    filtered_regions = [regions[index] for index in keep_indices]
    filtered_labels = np.full_like(component_labels, -1)
    for new_index, old_index in enumerate(keep_indices):
        filtered_labels[component_labels == old_index] = new_index
    return (
        filtered_regions,
        density,
        contrast,
        extent,
        filtered_labels,
    ), rejected_count


def _fit_grid_rectangle(
    region_nm: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    g5m_sigma_min_nm: float,
    g5m_sigma_max_nm: float,
    g5m_min_locs: int,
    g5m_max_rounds_without_best_bic: int,
    site_match_radius_nm: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    """Fit a fixed-size, freely rotating grid rectangle and retain points inside it."""
    _labels, centers = fit_picasso_g5m_components(
        region_nm,
        min_locs=g5m_min_locs,
        sigma_min_nm=g5m_sigma_min_nm,
        sigma_max_nm=g5m_sigma_max_nm,
        max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
    )
    if len(centers) < 3:
        stride = max(1, int(np.ceil(len(region_nm) / 500)))
        centers = region_nm[::stride]

    best_angle = 0.0
    best_translation = np.mean(centers, axis=0)
    best_score = (-1, -1, -float("inf"), -float("inf"))

    def consider(angle_degrees: float) -> None:
        nonlocal best_angle, best_translation, best_score
        angle = np.deg2rad(angle_degrees)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = centers @ rotation.T
        translations = (rotated[:, None, :] - grid_points_nm[None, :, :]).reshape(-1, 2)
        shifted = rotated[None, :, :] - translations[:, None, :]
        distances = np.linalg.norm(
            shifted[:, :, None, :] - grid_points_nm[None, None, :, :],
            axis=3,
        )
        nearest = np.argmin(distances, axis=2)
        nearest_distance = np.take_along_axis(distances, nearest[:, :, None], axis=2)[:, :, 0]
        matched = nearest_distance <= site_match_radius_nm
        matched_count = np.sum(matched, axis=1)
        valid = matched_count > 0
        if not np.any(valid):
            return
        unique_sites = np.sum(
            np.any(
                (nearest[:, :, None] == np.arange(len(grid_points_nm))[None, None, :]) & matched[:, :, None],
                axis=1,
            ),
            axis=1,
        )
        rms = np.sqrt(
            np.sum(np.square(nearest_distance) * matched, axis=1)
            / np.maximum(matched_count, 1)
        )
        mean_position = np.sum(shifted * matched[:, :, None], axis=1) / np.maximum(matched_count[:, None], 1)
        centeredness = np.linalg.norm(mean_position, axis=1)
        valid_indices = np.flatnonzero(valid)
        order = np.lexsort(
            (
                centeredness[valid_indices],
                rms[valid_indices],
                -matched_count[valid_indices],
                -unique_sites[valid_indices],
            )
        )
        candidate_index = int(valid_indices[int(order[0])])
        score = (
            int(unique_sites[candidate_index]),
            int(matched_count[candidate_index]),
            -float(rms[candidate_index]),
            -float(centeredness[candidate_index]),
        )
        if score > best_score:
            best_angle = angle_degrees
            best_translation = translations[candidate_index].copy()
            best_score = score

    for angle_degrees in np.arange(0.0, 180.0, 2.0):
        consider(float(angle_degrees))
    coarse_angle = best_angle
    for angle_degrees in np.arange(coarse_angle - 2.0, coarse_angle + 2.01, 0.2):
        consider(float(angle_degrees % 180.0))

    angle = np.deg2rad(best_angle)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local_points = region_nm @ rotation.T - best_translation
    inside = (
        (np.abs(local_points[:, 0]) <= rectangle_width_nm / 2.0)
        & (np.abs(local_points[:, 1]) <= rectangle_height_nm / 2.0)
    )
    local_corners = np.asarray(
        [
            [-rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
            [rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
            [rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
            [-rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
        ]
    )
    world_corners = (local_corners + best_translation) @ rotation
    matched_sites = max(0, int(best_score[0]))
    matched_clusters = max(0, int(best_score[1]))
    fit_rms_nm = max(0.0, float(-best_score[2])) if np.isfinite(best_score[2]) else float("inf")
    coverage = matched_sites / max(1, len(grid_points_nm))
    agreement = matched_clusters / max(1, len(centers))
    rms_quality = float(np.exp(-0.5 * np.square(fit_rms_nm / site_match_radius_nm))) if np.isfinite(fit_rms_nm) else 0.0
    confidence = float(np.clip(coverage * np.sqrt(agreement * rms_quality), 0.0, 1.0))
    return region_nm[inside], world_corners, float((-best_angle) % 180.0), confidence, matched_sites, fit_rms_nm


def _render_candidate_image(
    region_nm: np.ndarray,
    center_nm: np.ndarray,
    side_nm: float,
    pixel_nm: float,
    blur_nm: float,
) -> np.ndarray:
    bins = max(16, int(np.ceil(side_nm / pixel_nm)))
    half = side_nm / 2.0
    image, _x_edges, _y_edges = np.histogram2d(
        region_nm[:, 0],
        region_nm[:, 1],
        bins=(bins, bins),
        range=((center_nm[0] - half, center_nm[0] + half), (center_nm[1] - half, center_nm[1] + half)),
    )
    image = image.T
    if blur_nm > 0:
        image = gaussian_filter(image, blur_nm / pixel_nm, mode="constant")
    image = np.sqrt(image)
    image -= np.mean(image)
    norm = float(np.linalg.norm(image))
    return image / norm if norm > 0 else image


def alignment_corner_counts(regions, template_points_nm, radius_nm):
    """Count support at template marks nearest each bounding-box corner."""
    sites = np.asarray(template_points_nm, dtype=float).reshape(-1, 2)
    if not len(sites):
        return np.empty((len(regions), 0), dtype=int)
    low, high = sites.min(axis=0), sites.max(axis=0)
    corners = np.asarray([low, [high[0], low[1]], high, [low[0], high[1]]])
    indices = np.unique(cKDTree(sites).query(corners)[1])
    required = sites[indices]
    counts = np.zeros((len(regions), len(required)), dtype=int)
    for index, points in enumerate(regions):
        if len(points):
            counts[index] = cKDTree(points).query_ball_point(required, radius_nm, return_length=True)
    return counts


def alignment_dark_boundary(reference, canvas_side_nm):
    """Pixel-edge rectangle used by the correlation's exterior mask, in nm."""
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or not reference.size or np.ptp(reference) <= 0:
        return np.empty((0, 2))
    signal = reference - reference.min()
    y, x = np.nonzero(signal >= 0.35 * signal.max())
    height, width = reference.shape
    x0, x1 = (np.asarray([x.min(), x.max() + 1]) / width - 0.5) * canvas_side_nm
    y0, y1 = (np.asarray([y.min(), y.max() + 1]) / height - 0.5) * canvas_side_nm
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


def _boundary_template_correlation(image: np.ndarray, template: np.ndarray) -> float:
    """Match bright marks and exterior darkness, ignoring unmarked interior.

    The outermost bright pixels define a rectangle in template coordinates.
    Only bright marks and pixels outside that rectangle enter the cosine score;
    exterior reference values are zero, so exterior signal increases the
    denominator without increasing agreement. The 35% contrast cutoff matches
    template component detection. Undo rendering's global centering/scaling
    before masking so ignored interior signal cannot affect normalization.
    """
    reference = np.asarray(template, dtype=float)
    candidate = np.asarray(image, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Alignment image and template shapes must match.")
    reference = reference - np.min(reference)
    candidate = candidate - np.min(candidate)
    peak = float(np.max(reference))
    if peak <= 0.0:
        return 0.0
    bright = reference >= 0.35 * peak
    coordinates = np.nonzero(bright)
    bounds = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
    exterior = np.ones(reference.shape, dtype=bool)
    exterior[bounds] = False
    expected = reference[bright]
    observed = candidate[bright]
    observed_norm = np.sqrt(np.sum(observed**2) + np.sum(candidate[exterior]**2))
    denominator = float(np.linalg.norm(expected) * observed_norm)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(observed, expected) / denominator, 0.0, 1.0))


def prepare_custom_alignment_template(
    template_image: np.ndarray,
    *,
    output_shape: tuple[int, int],
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    canvas_side_nm: float,
    spacing_x_nm: float | None = None,
    spacing_y_nm: float | None = None,
    active_width_nm: float | None = None,
    active_height_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
) -> np.ndarray:
    """Map a bright-on-dark raster template onto the physical candidate footprint."""
    source = np.asarray(template_image, dtype=float)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("A custom alignment template must be a two-dimensional image at least 2 x 2 pixels.")
    if not np.all(np.isfinite(source)):
        raise ValueError("A custom alignment template cannot contain NaN or infinite values.")
    source = source - float(np.min(source))
    maximum = float(np.max(source))
    if maximum <= 0.0:
        raise ValueError("A custom alignment template must contain both dark background and bright signal.")
    source /= maximum
    output_height, output_width = (int(value) for value in output_shape)
    source_x_nm, source_y_nm = custom_template_physical_axes(
        source,
        rectangle_width_nm=rectangle_width_nm,
        rectangle_height_nm=rectangle_height_nm,
        spacing_x_nm=spacing_x_nm,
        spacing_y_nm=spacing_y_nm,
        active_width_nm=active_width_nm,
        active_height_nm=active_height_nm,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    canvas_x_nm = np.linspace(-canvas_side_nm / 2.0, canvas_side_nm / 2.0, output_width)
    canvas_y_nm = np.linspace(-canvas_side_nm / 2.0, canvas_side_nm / 2.0, output_height)
    source_columns = np.interp(canvas_x_nm, source_x_nm, np.arange(source.shape[1], dtype=float))
    source_rows = np.interp(canvas_y_nm, source_y_nm, np.arange(source.shape[0], dtype=float))
    sample_columns, sample_rows = np.meshgrid(source_columns, source_rows)
    inside = (
        (canvas_x_nm[None, :] >= source_x_nm[0])
        & (canvas_x_nm[None, :] <= source_x_nm[-1])
        & (canvas_y_nm[:, None] >= source_y_nm[0])
        & (canvas_y_nm[:, None] <= source_y_nm[-1])
    )
    canvas = map_coordinates(source, (sample_rows, sample_columns), order=1, mode="nearest")
    canvas[~inside] = 0.0
    canvas = np.sqrt(np.clip(canvas, 0.0, None))
    canvas -= np.mean(canvas)
    norm = float(np.linalg.norm(canvas))
    if norm <= 1e-12:
        raise ValueError("A custom alignment template has no usable contrast after scaling.")
    return canvas / norm


def _custom_template_component_centers(template_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a validated raster and bright-component centers as row/column pixels."""
    source = np.asarray(template_image, dtype=float)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("A custom site template must be a two-dimensional image at least 2 x 2 pixels.")
    if not np.all(np.isfinite(source)):
        raise ValueError("A custom site template cannot contain NaN or infinite values.")
    minimum = float(np.min(source))
    maximum = float(np.max(source))
    if maximum <= minimum:
        raise ValueError("A custom site template must contain both dark background and bright signal.")
    component_labels, component_count = label(
        source >= minimum + 0.35 * (maximum - minimum),
        structure=np.ones((3, 3), dtype=int),
    )
    centers: list[tuple[float, float]] = []
    signal = source - minimum
    for component in range(1, component_count + 1):
        rows, columns = np.nonzero(component_labels == component)
        weights = signal[rows, columns]
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            centers.append(
                (float(np.sum(rows * weights) / weight_sum), float(np.sum(columns * weights) / weight_sum))
            )
    if not centers:
        raise ValueError("The custom alignment template contains no separable bright sites.")
    return source, np.asarray(centers, dtype=float)


def _template_axis_coordinates(
    component_positions: np.ndarray,
    pixel_count: int,
    fallback_span_nm: float,
    expected_spacing_nm: float | None,
    active_span_nm: float | None = None,
    pixel_size_nm: float | None = None,
) -> np.ndarray:
    if pixel_size_nm is not None and pixel_size_nm > 0.0:
        # An explicitly calibrated raster keeps its own center, border, and
        # aspect ratio. Pixel centers are separated by exactly this distance.
        return (
            np.arange(pixel_count, dtype=float) - 0.5 * (pixel_count - 1)
        ) * float(pixel_size_nm)
    center_pixel = 0.5 * (float(np.min(component_positions)) + float(np.max(component_positions)))
    component_span_pixels = float(np.max(component_positions) - np.min(component_positions))
    if active_span_nm is not None and active_span_nm > 0.0 and component_span_pixels > 0.0:
        # The user-configured rows/columns define the physical footprint. The
        # raster may contain any number of bright marks, so its mark count must
        # not multiply the configured pitch.
        scale_nm_per_pixel = float(active_span_nm) / component_span_pixels
    else:
        sorted_positions = np.sort(np.asarray(component_positions, dtype=float))
        differences = np.diff(sorted_positions)
        differences = differences[differences > max(1.0, 0.002 * pixel_count)]
        if expected_spacing_nm is not None and expected_spacing_nm > 0.0 and len(differences):
            pixel_pitch = float(np.percentile(differences, 20.0))
            scale_nm_per_pixel = float(expected_spacing_nm) / pixel_pitch
        else:
            scale_nm_per_pixel = float(fallback_span_nm) / max(pixel_count - 1, 1)
    return (np.arange(pixel_count, dtype=float) - center_pixel) * scale_nm_per_pixel


def custom_template_physical_axes(
    template_image: np.ndarray,
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    spacing_x_nm: float | None = None,
    spacing_y_nm: float | None = None,
    active_width_nm: float | None = None,
    active_height_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate raster axes from its bright bounds instead of image borders."""
    source, centers = _custom_template_component_centers(template_image)
    x_nm = _template_axis_coordinates(
        centers[:, 1], source.shape[1], rectangle_width_nm, spacing_x_nm, active_width_nm, pixel_size_x_nm
    )
    y_nm = _template_axis_coordinates(
        centers[:, 0], source.shape[0], rectangle_height_nm, spacing_y_nm, active_height_nm, pixel_size_y_nm
    )
    return x_nm, y_nm


def custom_template_site_points(
    template_image: np.ndarray,
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    spacing_x_nm: float | None = None,
    spacing_y_nm: float | None = None,
    active_width_nm: float | None = None,
    active_height_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
) -> np.ndarray:
    """Extract bright connected components as physical docking-site positions."""
    source, centers = _custom_template_component_centers(template_image)
    x_nm, y_nm = custom_template_physical_axes(
        source,
        rectangle_width_nm=rectangle_width_nm,
        rectangle_height_nm=rectangle_height_nm,
        spacing_x_nm=spacing_x_nm,
        spacing_y_nm=spacing_y_nm,
        active_width_nm=active_width_nm,
        active_height_nm=active_height_nm,
        pixel_size_x_nm=pixel_size_x_nm,
        pixel_size_y_nm=pixel_size_y_nm,
    )
    result = np.column_stack(
        (
            np.interp(centers[:, 1], np.arange(source.shape[1]), x_nm),
            np.interp(centers[:, 0], np.arange(source.shape[0]), y_nm),
        )
    )
    return result[np.lexsort((result[:, 0], result[:, 1]))]


def site_gap_contrast_for_regions(
    aligned_regions: list[np.ndarray],
    grid_points_nm: np.ndarray,
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    site_radius_nm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Measure expected-site enrichment relative to explicit interior negative space."""
    if site_radius_nm <= 0:
        raise ValueError("Site-mask radius must be greater than zero.")
    grid = np.asarray(grid_points_nm, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid):
        raise ValueError("Site-gap scoring requires at least one 2D template point.")

    def design_bounds(values: np.ndarray, fallback_span: float) -> tuple[float, float]:
        unique = np.unique(np.asarray(values, dtype=float))
        if len(unique) > 1:
            step = float(np.median(np.diff(unique)))
            return float(unique[0] - step / 2.0), float(unique[-1] + step / 2.0)
        return float(unique[0] - fallback_span / 2.0), float(unique[0] + fallback_span / 2.0)

    x_lower, x_upper = design_bounds(grid[:, 0], rectangle_width_nm)
    y_lower, y_upper = design_bounds(grid[:, 1], rectangle_height_nm)
    sample_count = 256
    sample_x = np.linspace(x_lower, x_upper, sample_count)
    sample_y = np.linspace(y_lower, y_upper, sample_count)
    sample_xx, sample_yy = np.meshgrid(sample_x, sample_y)
    sample_points = np.column_stack((sample_xx.ravel(), sample_yy.ravel()))
    sample_distances, _sample_sites = cKDTree(grid).query(sample_points, k=1)
    site_area_fraction = float(np.mean(sample_distances <= site_radius_nm))
    site_area_fraction = float(np.clip(site_area_fraction, 1e-6, 1.0 - 1e-6))

    contrasts = np.zeros(len(aligned_regions), dtype=float)
    on_site_fractions = np.zeros(len(aligned_regions), dtype=float)
    grid_tree = cKDTree(grid)
    for index, region in enumerate(aligned_regions):
        points = np.asarray(region, dtype=float)
        if not len(points):
            continue
        distances, _sites = grid_tree.query(points, k=1)
        on_site_fraction = float(np.mean(distances <= site_radius_nm))
        on_density = on_site_fraction / site_area_fraction
        gap_density = (1.0 - on_site_fraction) / (1.0 - site_area_fraction)
        denominator = on_density + gap_density
        contrasts[index] = (on_density - gap_density) / denominator if denominator > 0 else 0.0
        on_site_fractions[index] = on_site_fraction
    return contrasts, on_site_fractions, site_area_fraction


def sparse_site_evidence_diagnostics(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
) -> SparseSiteEvidenceResult:
    """Return per-site support measurements and prominence-sampling geometry.

    Each expected site seeds a bounded peak search on one shared Gaussian-
    smoothed localization-density image.
    Prominence is the fractional drop from that local maximum to the high end
    of the surrounding boundary density.  A dim but isolated peak can therefore
    score highly, while a shoulder or arbitrary location inside a broad blob
    scores poorly.  The separate localization-count floor protects against
    declaring one-point fluctuations to be occupied sites.
    """
    points = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    if site_radius_nm <= 0:
        raise ValueError("Site-mask radius must be greater than zero.")
    if not len(points):
        return SparseSiteEvidenceResult(
            counts=np.zeros(len(grid), dtype=int),
            prominence=np.zeros(len(grid), dtype=float),
            peak_positions_nm=np.full((len(grid), 2), np.nan, dtype=float),
            peak_density=np.zeros(len(grid), dtype=float),
            boundary_density=np.zeros((len(grid), 32), dtype=float),
            boundary_reference_density=np.zeros(len(grid), dtype=float),
            boundary_reference_positions_nm=np.full((len(grid), 2), np.nan, dtype=float),
            boundary_points_nm=np.full((len(grid), 32, 2), np.nan, dtype=float),
        )
    distances = np.linalg.norm(points[:, None, :] - grid[None, :, :], axis=2)
    nearest_sites = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(points)), nearest_sites]
    assigned_sites = nearest_sites[nearest_distances <= site_radius_nm]
    on_counts = np.bincount(assigned_sites, minlength=len(grid)).astype(int)
    if len(grid) > 1:
        nearest_spacing = cKDTree(grid).query(grid, k=2)[0][:, 1]
        typical_spacing = float(np.median(nearest_spacing))
    else:
        typical_spacing = max(2.0 * site_radius_nm, 1.0)
        nearest_spacing = np.full(len(grid), typical_spacing, dtype=float)
    bandwidth_nm = float(np.clip(0.15 * typical_spacing, 1.5, 3.5))
    boundary_angles = 2.0 * math.pi * np.arange(32, dtype=float) / 32.0
    prominence = np.zeros(len(grid), dtype=float)
    peak_positions = np.full((len(grid), 2), np.nan, dtype=float)
    peak_densities = np.zeros(len(grid), dtype=float)
    boundary_densities = np.zeros((len(grid), len(boundary_angles)), dtype=float)
    boundary_reference_densities = np.zeros(len(grid), dtype=float)
    boundary_reference_positions = np.full((len(grid), 2), np.nan, dtype=float)
    boundary_points = np.full((len(grid), len(boundary_angles), 2), np.nan, dtype=float)
    render_margin_nm = max(typical_spacing, site_radius_nm + 3.0 * bandwidth_nm)
    x_min = float(np.min(grid[:, 0]) - render_margin_nm)
    x_max = float(np.max(grid[:, 0]) + render_margin_nm)
    y_min = float(np.min(grid[:, 1]) - render_margin_nm)
    y_max = float(np.max(grid[:, 1]) + render_margin_nm)
    requested_pixel_nm = min(1.0, bandwidth_nm / 2.0)
    x_bins = max(16, int(np.ceil((x_max - x_min) / requested_pixel_nm)))
    y_bins = max(16, int(np.ceil((y_max - y_min) / requested_pixel_nm)))
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    effective_x_nm = float((x_max - x_min) / x_bins)
    effective_y_nm = float((y_max - y_min) / y_bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    density, _x_edges, _y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(x_edges, y_edges),
    )
    density = gaussian_filter(
        density.T,
        sigma=(bandwidth_nm / effective_y_nm, bandwidth_nm / effective_x_nm),
        mode="constant",
    )

    for site_index, site in enumerate(grid):
        if on_counts[site_index] == 0:
            continue
        search_radius = min(site_radius_nm, 0.40 * float(nearest_spacing[site_index]))
        x_indices = np.flatnonzero(np.abs(x_centers - site[0]) <= search_radius)
        y_indices = np.flatnonzero(np.abs(y_centers - site[1]) <= search_radius)
        if not len(x_indices) or not len(y_indices):
            continue
        local_density = density[np.ix_(y_indices, x_indices)].copy()
        local_xx, local_yy = np.meshgrid(x_centers[x_indices], y_centers[y_indices])
        inside_search = (local_xx - site[0]) ** 2 + (local_yy - site[1]) ** 2 <= search_radius**2
        local_density[~inside_search] = -float("inf")
        peak_row, peak_column = np.unravel_index(np.argmax(local_density), local_density.shape)
        peak = np.asarray(
            (x_centers[x_indices[peak_column]], y_centers[y_indices[peak_row]]),
            dtype=float,
        )
        peak_density = float(local_density[peak_row, peak_column])
        peak_positions[site_index] = peak
        peak_densities[site_index] = peak_density
        boundary_radius = min(
            0.48 * float(nearest_spacing[site_index]),
            max(site_radius_nm, 2.5 * bandwidth_nm),
        )
        boundary = peak + boundary_radius * np.column_stack(
            (np.cos(boundary_angles), np.sin(boundary_angles))
        )
        boundary_points[site_index] = boundary
        boundary_rows = (boundary[:, 1] - y_min) / effective_y_nm - 0.5
        boundary_columns = (boundary[:, 0] - x_min) / effective_x_nm - 0.5
        boundary_density = map_coordinates(
            density,
            (boundary_rows, boundary_columns),
            order=1,
            mode="constant",
            cval=0.0,
        )
        saddle_density = float(np.percentile(boundary_density, 90.0))
        boundary_densities[site_index] = boundary_density
        boundary_reference_densities[site_index] = saddle_density
        reference_index = int(np.argmin(np.abs(boundary_density - saddle_density)))
        boundary_reference_positions[site_index] = boundary[reference_index]
        if peak_density > 1e-12:
            prominence[site_index] = np.clip(
                (peak_density - saddle_density) / peak_density,
                0.0,
                1.0,
            )
    return SparseSiteEvidenceResult(
        counts=on_counts,
        prominence=prominence,
        peak_positions_nm=peak_positions,
        peak_density=peak_densities,
        boundary_density=boundary_densities,
        boundary_reference_density=boundary_reference_densities,
        boundary_reference_positions_nm=boundary_reference_positions,
        boundary_points_nm=boundary_points,
    )


def sparse_site_evidence(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-site localization counts and local peak prominence."""
    result = sparse_site_evidence_diagnostics(
        aligned_region,
        grid_points_nm,
        site_radius_nm=site_radius_nm,
    )
    return result.counts, result.prominence


def supported_grid_site_mask(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    min_site_localizations: int,
    min_site_evidence: float,
) -> np.ndarray:
    """Return which theoretical grid sites have sufficient localized evidence."""
    counts, evidence = sparse_site_evidence(
        aligned_region,
        grid_points_nm,
        site_radius_nm=site_radius_nm,
    )
    return (counts >= int(min_site_localizations)) & (evidence >= float(min_site_evidence))


def supported_site_spacing_errors(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    supported_sites: np.ndarray,
    *,
    site_radius_nm: float,
) -> tuple[float, float]:
    """Return RMS and maximum pairwise spacing errors for supported site centroids."""
    centroids = supported_site_centroids(
        aligned_region,
        grid_points_nm,
        supported_sites,
        site_radius_nm=site_radius_nm,
    )
    retained_indices = np.flatnonzero(np.all(np.isfinite(centroids), axis=1))
    if len(retained_indices) < 2:
        return float("inf"), float("inf")
    observed = centroids[retained_indices]
    expected = np.asarray(grid_points_nm, dtype=float)[retained_indices]
    pair_rows, pair_columns = np.triu_indices(len(observed), k=1)
    observed_spacing = np.linalg.norm(observed[pair_rows] - observed[pair_columns], axis=1)
    expected_spacing = np.linalg.norm(expected[pair_rows] - expected[pair_columns], axis=1)
    errors = np.abs(observed_spacing - expected_spacing)
    return float(np.sqrt(np.mean(np.square(errors)))), float(np.max(errors))


def supported_site_centroids(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    supported_sites: np.ndarray,
    *,
    site_radius_nm: float,
) -> np.ndarray:
    """Return measured centroid for each supported grid assignment; other rows are NaN."""
    points = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    supported = np.asarray(supported_sites, dtype=bool)
    if supported.shape != (len(grid),):
        raise ValueError("Supported-site mask must have one value per grid site.")
    centroids = np.full((len(grid), 2), np.nan, dtype=float)
    supported_indices = np.flatnonzero(supported)
    if not len(supported_indices) or not len(points):
        return centroids
    distances, nearest_sites = cKDTree(grid).query(points, k=1)
    for site_index in supported_indices:
        assigned = points[(nearest_sites == site_index) & (distances <= site_radius_nm)]
        if not len(assigned):
            continue
        centroids[site_index] = np.mean(assigned, axis=0)
    return centroids


@lru_cache(maxsize=32)
def _grid_blob_model_basis(
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    pixel_nm: float,
    psf_sigma_nm: float,
    grid_coordinates: tuple[float, ...],
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the fixed grid-model design shared by every candidate in a run."""
    x_bins = max(16, int(np.ceil(rectangle_width_nm / pixel_nm)))
    y_bins = max(16, int(np.ceil(rectangle_height_nm / pixel_nm)))
    x_edges = np.linspace(-rectangle_width_nm / 2.0, rectangle_width_nm / 2.0, x_bins + 1)
    y_edges = np.linspace(-rectangle_height_nm / 2.0, rectangle_height_nm / 2.0, y_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    grid = np.asarray(grid_coordinates, dtype=float).reshape(-1, 2)
    grid_columns = [
        np.exp(-0.5 * ((xx - site[0]) ** 2 + (yy - site[1]) ** 2) / psf_sigma_nm**2).ravel()
        for site in grid
    ]
    grid_design = np.column_stack([*grid_columns, np.ones(xx.size)])
    return x_bins, y_bins, xx, yy, grid_design


def grid_vs_blob_delta_bic(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    pixel_nm: float,
) -> float:
    """Positive values favor an adaptive-width sparse grid over one broad blob."""
    points = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    if not len(points):
        return -float("inf")
    if pixel_nm <= 0:
        raise ValueError("Grid-versus-blob scoring pixel must be positive.")
    # Model selection uses one stable physical sampling grid.  It must not
    # change merely because the user requests a different alignment thumbnail.
    pixel_nm = 2.0
    x_bins = max(16, int(np.ceil(rectangle_width_nm / pixel_nm)))
    y_bins = max(16, int(np.ceil(rectangle_height_nm / pixel_nm)))
    image, _x_edges, _y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(x_bins, y_bins),
        range=(
            (-rectangle_width_nm / 2.0, rectangle_width_nm / 2.0),
            (-rectangle_height_nm / 2.0, rectangle_height_nm / 2.0),
        ),
    )
    observation = image.T.ravel().astype(float)
    if len(grid) > 1:
        nearest_grid_spacing = float(np.median(cKDTree(grid).query(grid, k=2)[0][:, 1]))
    else:
        nearest_grid_spacing = max(rectangle_width_nm, rectangle_height_nm)
    minimum_sigma = max(1.5, 0.75 * pixel_nm)
    maximum_sigma = max(minimum_sigma, min(8.0, 0.40 * nearest_grid_spacing))
    candidate_sigmas = np.unique(
        np.concatenate(
            (
                np.geomspace(minimum_sigma, maximum_sigma, 7),
                np.asarray([max(2.0, pixel_nm)]),
            )
        )
    )
    correlation_sigma = max(1.5, pixel_nm)
    footprint_area = max(float(rectangle_width_nm * rectangle_height_nm), pixel_nm**2)
    independent_resolution_elements = footprint_area / (math.pi * correlation_sigma**2)
    effective_sample_count = max(
        16.0,
        min(float(observation.size), float(len(points)), independent_resolution_elements),
    )
    mean_square_denominator = float(observation.size)
    best_grid_bic = float("inf")
    xx = yy = None
    for psf_sigma in candidate_sigmas:
        _x_bins, _y_bins, xx, yy, grid_design = _grid_blob_model_basis(
            float(rectangle_width_nm),
            float(rectangle_height_nm),
            pixel_nm,
            float(psf_sigma),
            tuple(float(value) for value in grid.ravel()),
        )
        grid_coefficients, _grid_residual = nnls(grid_design, observation)
        grid_fit = grid_design @ grid_coefficients
        grid_rss = max(float(np.sum(np.square(observation - grid_fit))), 1e-12)
        # Active amplitudes plus background and one fitted shared-width parameter.
        grid_parameters = max(2, int(np.count_nonzero(grid_coefficients[:-1] > 1e-8)) + 2)
        grid_bic = effective_sample_count * math.log(grid_rss / mean_square_denominator) + (
            grid_parameters * math.log(effective_sample_count)
        )
        best_grid_bic = min(best_grid_bic, grid_bic)

    assert xx is not None and yy is not None
    psf_sigma = max(2.0, pixel_nm)

    center = np.mean(points, axis=0)
    covariance = np.cov(points.T) if len(points) > 1 else np.eye(2) * psf_sigma**2
    covariance = np.asarray(covariance, dtype=float) + np.eye(2) * psf_sigma**2
    try:
        inverse_covariance = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse_covariance = np.linalg.pinv(covariance)
    offsets = np.stack((xx - center[0], yy - center[1]), axis=-1)
    exponent = np.einsum("...i,ij,...j->...", offsets, inverse_covariance, offsets)
    blob = np.exp(-0.5 * exponent).ravel()
    blob_design = np.column_stack((blob, np.ones(observation.size)))
    blob_coefficients, _blob_residual = nnls(blob_design, observation)
    blob_fit = blob_design @ blob_coefficients
    blob_rss = max(float(np.sum(np.square(observation - blob_fit))), 1e-12)

    # Mean and covariance were estimated from the points in addition to amplitude/background.
    blob_bic = effective_sample_count * math.log(blob_rss / mean_square_denominator) + (
        7 * math.log(effective_sample_count)
    )
    return float(blob_bic - best_grid_bic)


def _polar_image(image: np.ndarray, angle_count: int = 180) -> np.ndarray:
    """Sample a square localization image in polar coordinates for fast rotation search."""
    radius_count = max(12, image.shape[0] // 2 - 2)
    radii = np.linspace(2.0, image.shape[0] / 2.0 - 2.0, radius_count)
    angles = np.arange(angle_count, dtype=float) * (2.0 * np.pi / angle_count)
    center = (image.shape[0] - 1.0) / 2.0
    # Histogram rows increase with physical y, matching the localization coordinate system.
    yy = center + np.sin(angles)[:, None] * radii[None, :]
    xx = center + np.cos(angles)[:, None] * radii[None, :]
    polar = map_coordinates(image, [yy, xx], order=1, mode="constant", cval=0.0)
    polar -= np.mean(polar, axis=0, keepdims=True)
    return polar


def _rotation_candidates_from_polar(
    image: np.ndarray,
    reference_polar_fft: np.ndarray,
    *,
    maximum_candidates: int = 4,
    minimum_separation_deg: float = 20.0,
    image_polar_fft: np.ndarray | None = None,
) -> np.ndarray:
    """Return separated angular-correlation peaks for full-pose scoring."""
    if maximum_candidates < 1:
        raise ValueError("At least one rotation candidate is required.")
    polar_fft = (
        np.asarray(image_polar_fft)
        if image_polar_fft is not None
        else np.fft.rfft(_polar_image(image), axis=0)
    )
    correlation = np.fft.irfft(
        np.sum(polar_fft * np.conj(reference_polar_fft), axis=1),
        n=180,
    )
    local_maxima = np.flatnonzero(
        (correlation >= np.roll(correlation, 1)) & (correlation >= np.roll(correlation, -1))
    )
    ranked = local_maxima[np.argsort(correlation[local_maxima])[::-1]]
    if not len(ranked):
        ranked = np.asarray([int(np.argmax(correlation))], dtype=int)
    minimum_separation_bins = max(1, int(np.ceil(float(minimum_separation_deg) / 2.0)))
    selected: list[int] = []
    for peak_value in ranked:
        peak = int(peak_value)
        if any(
            min((peak - other) % len(correlation), (other - peak) % len(correlation))
            < minimum_separation_bins
            for other in selected
        ):
            continue
        selected.append(peak)
        if len(selected) >= maximum_candidates:
            break

    angles: list[float] = []
    for peak in selected:
        left = float(correlation[(peak - 1) % len(correlation)])
        middle = float(correlation[peak])
        right = float(correlation[(peak + 1) % len(correlation)])
        denominator = left - 2.0 * middle + right
        subpixel = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        refined_peak = peak + float(np.clip(subpixel, -0.5, 0.5))
        if refined_peak > 90.0:
            refined_peak -= 180.0
        angles.append(refined_peak * 2.0)
    return np.asarray(angles, dtype=float)


def _rotation_from_polar(image: np.ndarray, reference_polar_fft: np.ndarray) -> float:
    """Return the strongest polar-correlation angle for compatibility."""
    return float(_rotation_candidates_from_polar(image, reference_polar_fft, maximum_candidates=1)[0])


def _principal_axis_angle(image: np.ndarray) -> tuple[float, float]:
    """Return the rendered signal's principal-axis angle and anisotropy ratio."""
    weights = np.asarray(image, dtype=float) - float(np.min(image))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return 0.0, 1.0
    yy, xx = np.indices(weights.shape, dtype=float)
    xx -= (weights.shape[1] - 1.0) / 2.0
    yy -= (weights.shape[0] - 1.0) / 2.0
    center_x = float(np.sum(weights * xx) / weight_sum)
    center_y = float(np.sum(weights * yy) / weight_sum)
    centered_x = xx - center_x
    centered_y = yy - center_y
    covariance = np.asarray(
        [
            [np.sum(weights * centered_x * centered_x), np.sum(weights * centered_x * centered_y)],
            [np.sum(weights * centered_x * centered_y), np.sum(weights * centered_y * centered_y)],
        ],
        dtype=float,
    ) / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    smallest = max(float(np.min(eigenvalues)), 1e-12)
    anisotropy = float(np.max(eigenvalues)) / smallest
    return float(np.rad2deg(np.arctan2(principal[1], principal[0]))), anisotropy


def _full_pose_rotation_candidates(
    image: np.ndarray,
    template: np.ndarray,
    template_polar_fft: np.ndarray,
    *,
    rotational_period_deg: float = 180.0,
    image_polar_fft: np.ndarray | None = None,
    image_axis_info: tuple[float, float] | None = None,
) -> np.ndarray:
    """Combine data-driven rotation peaks with contamination-resistant coarse coverage."""
    if rotational_period_deg not in (180.0, 360.0):
        raise ValueError("Rotation-search period must be either 180 or 360 degrees.")
    polar_angles = _rotation_candidates_from_polar(
        image,
        template_polar_fft,
        image_polar_fft=image_polar_fft,
    )
    image_axis, image_anisotropy = (
        image_axis_info if image_axis_info is not None else _principal_axis_angle(image)
    )
    template_axis, template_anisotropy = _principal_axis_angle(template)
    candidates = list(float(value) for value in polar_angles)
    if image_anisotropy >= 1.05 and template_anisotropy >= 1.05:
        principal_rotation = image_axis - template_axis
        candidates.extend(principal_rotation + offset for offset in np.arange(-20.0, 20.1, 5.0))
        if rotational_period_deg == 360.0:
            # A principal axis has no direction, so preserve its opposite
            # direction when an asymmetric template distinguishes the two.
            candidates.extend(principal_rotation + 180.0 + offset for offset in np.arange(-20.0, 20.1, 5.0))
    # A neighboring object can dominate both the polar and principal-axis
    # estimates. Sparse inlier refinement can recover the target only if the
    # correct basin is still represented, so retain inexpensive full-period
    # coverage at 20-degree intervals. The inlier refinement safely closes the
    # remaining angular gap while keeping large tiled runs practical.
    half_period = rotational_period_deg / 2.0
    candidates.extend(float(value) for value in np.arange(-half_period, half_period, 20.0))

    unique: list[float] = []
    for angle in candidates:
        if any(
            abs(((angle - existing + half_period) % rotational_period_deg) - half_period) < 0.25
            for existing in unique
        ):
            continue
        unique.append(float(angle))
    return np.asarray(unique, dtype=float)


def _translation_hypotheses_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    maximum_candidates: int = 3,
    minimum_separation_pixels: int = 4,
    maximum_shift_pixels: int | None = None,
) -> list[tuple[float, float]]:
    """Return separated subpixel translation peaks instead of only the global maximum."""
    correlation = fftconvolve(image, reference[::-1, ::-1], mode="full")
    zero_y = reference.shape[0] - 1
    zero_x = reference.shape[1] - 1
    max_shift = max(
        2,
        image.shape[0] // 8 if maximum_shift_pixels is None else int(maximum_shift_pixels),
    )
    y_slice = slice(max(0, zero_y - max_shift), min(correlation.shape[0], zero_y + max_shift + 1))
    x_slice = slice(max(0, zero_x - max_shift), min(correlation.shape[1], zero_x + max_shift + 1))
    search = correlation[y_slice, x_slice]
    local_maxima = np.argwhere(
        (search >= np.roll(search, 1, axis=0))
        & (search >= np.roll(search, -1, axis=0))
        & (search >= np.roll(search, 1, axis=1))
        & (search >= np.roll(search, -1, axis=1))
    )
    if not len(local_maxima):
        local_maxima = np.asarray([np.unravel_index(int(np.argmax(search)), search.shape)])
    ranked = local_maxima[np.argsort(search[local_maxima[:, 0], local_maxima[:, 1]])[::-1]]
    selected: list[tuple[int, int]] = []
    for local_y, local_x in ranked:
        peak_y = int(local_y + y_slice.start)
        peak_x = int(local_x + x_slice.start)
        if any(
            (peak_y - other_y) ** 2 + (peak_x - other_x) ** 2
            < int(minimum_separation_pixels) ** 2
            for other_y, other_x in selected
        ):
            continue
        selected.append((peak_y, peak_x))
        if len(selected) >= max(1, int(maximum_candidates)):
            break

    def subpixel_offset(before: float, middle: float, after: float) -> float:
        denominator = before - 2.0 * middle + after
        if abs(denominator) <= 1e-12:
            return 0.0
        return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))

    hypotheses: list[tuple[float, float]] = []
    for peak_y, peak_x in selected:
        offset_y = 0.0
        offset_x = 0.0
        if 0 < peak_y < correlation.shape[0] - 1:
            offset_y = subpixel_offset(
                float(correlation[peak_y - 1, peak_x]),
                float(correlation[peak_y, peak_x]),
                float(correlation[peak_y + 1, peak_x]),
            )
        if 0 < peak_x < correlation.shape[1] - 1:
            offset_x = subpixel_offset(
                float(correlation[peak_y, peak_x - 1]),
                float(correlation[peak_y, peak_x]),
                float(correlation[peak_y, peak_x + 1]),
            )
        hypotheses.append((peak_y - zero_y + offset_y, peak_x - zero_x + offset_x))
    return hypotheses


def _translation_to_reference(image: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Find the strongest bounded, subpixel linear-correlation shift."""
    return _translation_hypotheses_to_reference(image, reference, maximum_candidates=1)[0]


def _score_complete_image_pose(
    image: np.ndarray,
    template: np.ndarray,
    angle_deg: float,
) -> tuple[float, float, float, np.ndarray]:
    """Optimize translation and score one complete rotation/translation pose."""
    rotated = rotate(image, angle_deg, reshape=False, order=1, mode="constant", prefilter=False)
    shift_y, shift_x = _translation_to_reference(rotated, template)
    aligned = ndimage_shift(
        rotated,
        shift=(-shift_y, -shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    score = _boundary_template_correlation(aligned, template)
    return score, float(shift_x), float(shift_y), aligned


def _sparse_pose_quality(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    required_sites: int,
    minimum_site_localizations: int,
    saturate_site_counts: bool = False,
) -> float:
    """Score a pose using both present and missing alignment-site evidence.

    The count-based coverage terms alone are nearly flat while a bright stroke
    remains anywhere inside a comparatively large site mask.  That made poses
    a few degrees apart look equivalent.  ``site_evidence`` samples only the
    closest ``minimum_site_localizations`` at every expected alignment mark,
    so extra brightness cannot compensate for putting the mark beside the
    localization ridge.  Its complement is an explicit missing-site penalty.
    One weakest mark is ignored to tolerate a genuinely absent extension.
    """
    points = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    if not len(points) or not len(grid):
        return -1.0
    distances, sites = cKDTree(grid).query(points, k=1)
    counts = np.bincount(
        sites[distances <= site_radius_nm], minlength=len(grid)
    ).astype(float)
    supported = counts >= minimum_site_localizations
    supported_indices = np.flatnonzero(supported)
    if required_sites <= 0:
        return 0.0
    required_coverage = min(len(supported_indices) / max(required_sites, 1), 1.0)
    total_coverage = len(supported_indices) / max(len(grid), 1)
    centroid_precision = 0.0
    site_evidence = np.zeros(len(grid), dtype=float)
    evidence_sigma_nm = max(0.75, min(float(site_radius_nm) / 3.0, 2.5))
    if len(supported_indices):
        residuals: list[float] = []
        for site_index in supported_indices:
            assigned_mask = (sites == site_index) & (distances <= site_radius_nm)
            assigned = points[assigned_mask]
            if len(assigned):
                residuals.append(float(np.linalg.norm(np.mean(assigned, axis=0) - grid[site_index])))
        if residuals:
            centroid_precision = float(
                np.mean(np.clip(1.0 - np.asarray(residuals) / site_radius_nm, 0.0, 1.0))
            )
    # Use only the strongest K localization samples at each alignment mark.
    # This is deliberately brightness-saturated: 500 localizations several
    # nanometers away must not beat three localizations centered on the mark.
    required_localizations = max(1, int(minimum_site_localizations))
    for site_index in range(len(grid)):
        assigned_distances = distances[
            (sites == site_index) & (distances <= site_radius_nm)
        ]
        if not len(assigned_distances):
            continue
        weights = np.exp(
            -0.5 * np.square(assigned_distances / evidence_sigma_nm)
        )
        strongest = np.sort(weights)[-required_localizations:]
        site_evidence[site_index] = float(
            np.sum(strongest) / required_localizations
        )
    continuous_support = float(np.mean(site_evidence))
    missing = 1.0 - site_evidence
    # Missing one extension should not ruin an otherwise decisive L-shaped
    # alignment, but systematic darkness caused by a rotated pose must count.
    allowed_missing_sites = 1 if len(missing) >= 5 else 0
    if allowed_missing_sites:
        missing_for_score = np.sort(missing)[:-allowed_missing_sites]
    else:
        missing_for_score = missing
    missing_penalty = float(np.mean(missing_for_score)) if len(missing_for_score) else 0.0
    # Saturate each alignment site independently. A digital group containing
    # hundreds of localizations must not outweigh an alignment mark merely by
    # contributing more points near a competing pose.
    if saturate_site_counts:
        inlier_support = float(
            np.mean(
                np.clip(
                    counts / max(float(minimum_site_localizations), 1.0),
                    0.0,
                    1.0,
                )
            )
        )
    else:
        # Legacy full-grid alignment uses the dominant connected object's
        # inlier fraction to reject a separate nearby distractor.
        inlier_support = float(np.mean(distances <= site_radius_nm))
    # Required coverage dominates so genuinely sparse origami remain viable.
    # Extra distributed sites and precise centroids break ties, while the modest
    # soft per-site term rewards partially supported marks without restoring a
    # localization-count weighting.
    return float(
        1.5 * required_coverage
        + 0.5 * total_coverage
        + 0.6 * centroid_precision
        + 1.0 * continuous_support
        + 0.2 * inlier_support
        - 0.9 * missing_penalty
    )


def _refine_sparse_grid_pose(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    minimum_site_localizations: int,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine a provisional pose with Kabsch fits to grid-supported site centroids."""
    original = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    refined = original.copy()
    accumulated_rotation = np.eye(2, dtype=float)
    accumulated_offset = np.zeros(2, dtype=float)
    grid_tree = cKDTree(grid)
    for _iteration in range(max(0, int(iterations))):
        distances, sites = grid_tree.query(refined, k=1)
        observed_centroids: list[np.ndarray] = []
        expected_sites: list[np.ndarray] = []
        for site_index in range(len(grid)):
            assigned = refined[(sites == site_index) & (distances <= site_radius_nm)]
            if len(assigned) < minimum_site_localizations:
                continue
            observed_centroids.append(np.mean(assigned, axis=0))
            expected_sites.append(grid[site_index])
        if len(observed_centroids) < 2:
            break
        observed = np.asarray(observed_centroids, dtype=float)
        expected = np.asarray(expected_sites, dtype=float)
        observed_center = np.mean(observed, axis=0)
        expected_center = np.mean(expected, axis=0)
        covariance = (observed - observed_center).T @ (expected - expected_center)
        left, _singular_values, right_transpose = np.linalg.svd(covariance)
        correction = right_transpose.T @ left.T
        if np.linalg.det(correction) < 0:
            right_transpose[-1, :] *= -1.0
            correction = right_transpose.T @ left.T
        offset = expected_center - observed_center @ correction.T
        refined = refined @ correction.T + offset
        accumulated_rotation = correction @ accumulated_rotation
        accumulated_offset = accumulated_offset @ correction.T + offset
        correction_angle = abs(float(np.rad2deg(np.arctan2(correction[1, 0], correction[0, 0]))))
        if correction_angle < 0.05 and float(np.linalg.norm(offset)) < 0.05:
            break
    return refined, accumulated_rotation, accumulated_offset


def _refine_pose_by_alignment_site_contrast(
    aligned_region: np.ndarray,
    grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    minimum_site_localizations: int,
    maximum_angle_correction_deg: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fine-tune a provisional pose by rewarding bright expected marks and dark misses.

    This final, bounded refinement works in localization coordinates, avoiding
    the angular quantization of the rendered alignment thumbnail.  Only the
    uploaded alignment sites are queried; signal from other digital groups is
    neither rewarded nor treated as expected background.
    """
    points = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(grid_points_nm, dtype=float)
    if not len(points) or not len(grid):
        return np.eye(2, dtype=float), np.zeros(2, dtype=float), -float("inf")
    tree = cKDTree(points)
    sample_count = min(max(1, int(minimum_site_localizations)), len(points))
    sigma_nm = max(0.75, min(float(site_radius_nm) / 3.0, 2.5))
    translation_bound_nm = max(0.5, min(float(site_radius_nm) / 2.0, 4.0))
    allowed_missing_sites = 1 if len(grid) >= 5 else 0

    def evidence_score(parameters: np.ndarray) -> float:
        angle_deg, offset_x, offset_y = (float(value) for value in parameters)
        angle = np.deg2rad(angle_deg)
        correction = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        offset = np.asarray([offset_x, offset_y], dtype=float)
        # If corrected = points @ correction.T + offset, these are the
        # equivalent query positions in the uncorrected point coordinates.
        query_sites = (grid - offset) @ correction
        distances, _indices = tree.query(query_sites, k=sample_count)
        distances = np.asarray(distances, dtype=float)
        if distances.ndim == 1:
            distances = distances[:, None]
        weights = np.exp(-0.5 * np.square(distances / sigma_nm))
        weights[distances > site_radius_nm] = 0.0
        evidence = np.mean(weights, axis=1)
        if allowed_missing_sites:
            evidence = np.sort(evidence)[allowed_missing_sites:]
        bright_support = float(np.mean(evidence)) if len(evidence) else 0.0
        missing_penalty = float(np.mean(1.0 - evidence)) if len(evidence) else 1.0
        # Writing both terms explicitly makes a dark expected mark reduce the
        # objective instead of merely failing to add positive correlation.
        return bright_support - 0.9 * missing_penalty

    initial = np.zeros(3, dtype=float)
    initial_score = evidence_score(initial)
    result = minimize(
        lambda values: -evidence_score(np.asarray(values, dtype=float)),
        initial,
        method="Powell",
        bounds=(
            (-float(maximum_angle_correction_deg), float(maximum_angle_correction_deg)),
            (-translation_bound_nm, translation_bound_nm),
            (-translation_bound_nm, translation_bound_nm),
        ),
        options={"xtol": 0.02, "ftol": 1e-4, "maxiter": 35},
    )
    values = np.asarray(result.x, dtype=float) if result.success else initial
    final_score = evidence_score(values)
    if not np.isfinite(final_score) or final_score <= initial_score + 1e-6:
        values = initial
        final_score = initial_score
    angle = np.deg2rad(float(values[0]))
    correction = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    return correction, np.asarray(values[1:3], dtype=float), float(final_score)


def _refine_pose_by_full_lattice(
    aligned_region: np.ndarray,
    full_grid_points_nm: np.ndarray,
    alignment_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    minimum_site_localizations: int,
    angular_block_deg: float = 12.0,
    maximum_angle_correction_deg: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine an L_L pose against occupied sites on the complete physical lattice.

    Empty lattice cells are neutral because their digital state is unknown.
    The angular search grows by blocks while its best solution remains on the
    current boundary, allowing recovery from L_L errors well above ten degrees.
    """
    points = np.asarray(aligned_region, dtype=float)
    full_grid = np.asarray(full_grid_points_nm, dtype=float)
    alignment_grid = np.asarray(alignment_points_nm, dtype=float)
    if not len(points) or not len(full_grid):
        return np.eye(2, dtype=float), np.zeros(2, dtype=float), -float("inf")
    tree = cKDTree(points)
    sample_count = min(max(1, int(minimum_site_localizations)), len(points))
    sigma_nm = max(0.75, min(float(site_radius_nm) / 3.0, 2.5))
    translation_bound_nm = max(0.5, min(float(site_radius_nm) / 2.0, 4.0))

    def site_evidence(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        angle_deg, offset_x, offset_y = (float(value) for value in values)
        angle = np.deg2rad(angle_deg)
        correction = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        query_sites = (grid - np.asarray([offset_x, offset_y])) @ correction
        distances, _indices = tree.query(query_sites, k=sample_count)
        distances = np.asarray(distances, dtype=float)
        if distances.ndim == 1:
            distances = distances[:, None]
        weights = np.exp(-0.5 * np.square(distances / sigma_nm))
        weights[distances > site_radius_nm] = 0.0
        return np.mean(weights, axis=1)

    def score(values: np.ndarray) -> float:
        full_evidence = site_evidence(full_grid, values)
        # A capped sum rewards every occupied analog position without assuming
        # that any of the other 12x8 positions should be dark or bright.
        occupied_lattice_score = float(np.sum(full_evidence) / np.sqrt(len(full_grid)))
        alignment_evidence = site_evidence(alignment_grid, values)
        if len(alignment_evidence) >= 5:
            alignment_evidence = np.sort(alignment_evidence)[1:]
        alignment_support = float(np.mean(alignment_evidence)) if len(alignment_evidence) else 0.0
        alignment_missing = (
            float(np.mean(1.0 - alignment_evidence)) if len(alignment_evidence) else 1.0
        )
        return occupied_lattice_score + 0.35 * (alignment_support - 0.9 * alignment_missing)

    # Grow the coarse search only when evidence continues improving at an edge.
    # Every angle gets a small translation search as well: otherwise an L_L
    # translation error can make the correct angular basin look artificially
    # weak before the continuous optimizer ever sees it.
    zero_pose = np.zeros(3, dtype=float)
    initial_score = score(zero_pose)
    tested: dict[float, tuple[float, float, float]] = {}
    coarse_offsets = (
        -0.5 * translation_bound_nm,
        0.0,
        0.5 * translation_bound_nm,
    )
    lower = -float(angular_block_deg)
    upper = float(angular_block_deg)
    while True:
        for angle in np.arange(lower, upper + 0.001, 1.0):
            key = float(round(angle, 6))
            if key not in tested:
                best_trial = (-float("inf"), 0.0, 0.0)
                for offset_x in coarse_offsets:
                    for offset_y in coarse_offsets:
                        trial_score = score(
                            np.asarray([angle, offset_x, offset_y], dtype=float)
                        )
                        if trial_score > best_trial[0]:
                            best_trial = (trial_score, offset_x, offset_y)
                tested[key] = best_trial
        best_angle = max(tested, key=lambda value: tested[value][0])
        expanded = False
        if best_angle <= lower + 1.0 and abs(lower) < maximum_angle_correction_deg:
            lower = max(-maximum_angle_correction_deg, lower - angular_block_deg)
            expanded = True
        if best_angle >= upper - 1.0 and upper < maximum_angle_correction_deg:
            upper = min(maximum_angle_correction_deg, upper + angular_block_deg)
            expanded = True
        if not expanded:
            break

    local_lower = max(-maximum_angle_correction_deg, best_angle - 2.0)
    local_upper = min(maximum_angle_correction_deg, best_angle + 2.0)
    _coarse_score, coarse_x, coarse_y = tested[best_angle]
    result = minimize(
        lambda values: -score(np.asarray(values, dtype=float)),
        np.asarray([best_angle, coarse_x, coarse_y], dtype=float),
        method="Powell",
        bounds=(
            (local_lower, local_upper),
            (-translation_bound_nm, translation_bound_nm),
            (-translation_bound_nm, translation_bound_nm),
        ),
        options={"xtol": 0.02, "ftol": 1e-4, "maxiter": 35},
    )
    values = np.asarray(result.x, dtype=float) if result.success else np.zeros(3, dtype=float)
    final_score = score(values)
    if not np.isfinite(final_score) or final_score <= initial_score + 1e-4:
        values = np.zeros(3, dtype=float)
        final_score = initial_score
    angle = np.deg2rad(float(values[0]))
    correction = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    return correction, np.asarray(values[1:3], dtype=float), float(final_score)


def _refine_pose_from_lattice_centroids(
    aligned_region: np.ndarray,
    full_grid_points_nm: np.ndarray,
    *,
    site_radius_nm: float,
    minimum_site_localizations: int,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Refine a nearly aligned pose from equal-weight robust analog-site centers."""
    original = np.asarray(aligned_region, dtype=float)
    grid = np.asarray(full_grid_points_nm, dtype=float)
    if not len(original) or not len(grid):
        return np.eye(2, dtype=float), np.zeros(2, dtype=float), 0, float("inf")
    refined = original.copy()
    accumulated_rotation = np.eye(2, dtype=float)
    accumulated_offset = np.zeros(2, dtype=float)
    grid_tree = cKDTree(grid)
    retained_count = 0
    retained_rms = float("inf")
    for _iteration in range(max(0, int(iterations))):
        distances, sites = grid_tree.query(refined, k=1)
        observed_centroids: list[np.ndarray] = []
        expected_sites: list[np.ndarray] = []
        for site_index in range(len(grid)):
            assigned = refined[
                (sites == site_index) & (distances <= float(site_radius_nm))
            ]
            if len(assigned) < int(minimum_site_localizations):
                continue
            # A trimmed center resists background localizations while retaining
            # subnanometer precision for a dense analog pixel.
            median = np.median(assigned, axis=0)
            radial = np.linalg.norm(assigned - median, axis=1)
            keep_count = max(
                int(minimum_site_localizations), int(np.ceil(0.8 * len(assigned)))
            )
            keep = np.argsort(radial)[:keep_count]
            observed_centroids.append(np.mean(assigned[keep], axis=0))
            expected_sites.append(grid[site_index])
        if len(observed_centroids) < 3:
            break
        observed = np.asarray(observed_centroids, dtype=float)
        expected = np.asarray(expected_sites, dtype=float)
        residuals = np.linalg.norm(observed - expected, axis=1)
        residual_median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - residual_median)))
        robust_limit = min(
            0.9 * float(site_radius_nm),
            residual_median + max(1.0, 2.5 * 1.4826 * mad),
        )
        inliers = residuals <= robust_limit
        if int(np.count_nonzero(inliers)) < 3:
            inliers = np.argsort(residuals)[: min(3, len(residuals))]
            observed = observed[inliers]
            expected = expected[inliers]
        else:
            observed = observed[inliers]
            expected = expected[inliers]
        observed_center = np.mean(observed, axis=0)
        expected_center = np.mean(expected, axis=0)
        covariance = (observed - observed_center).T @ (expected - expected_center)
        left, _singular_values, right_transpose = np.linalg.svd(covariance)
        correction = right_transpose.T @ left.T
        if np.linalg.det(correction) < 0:
            right_transpose[-1, :] *= -1.0
            correction = right_transpose.T @ left.T
        offset = expected_center - observed_center @ correction.T
        corrected_centroids = observed @ correction.T + offset
        retained_count = len(observed)
        retained_rms = float(
            np.sqrt(np.mean(np.sum(np.square(corrected_centroids - expected), axis=1)))
        )
        refined = refined @ correction.T + offset
        accumulated_rotation = correction @ accumulated_rotation
        accumulated_offset = accumulated_offset @ correction.T + offset
        correction_angle = abs(
            float(np.rad2deg(np.arctan2(correction[1, 0], correction[0, 0])))
        )
        if correction_angle < 0.01 and float(np.linalg.norm(offset)) < 0.02:
            break
    return accumulated_rotation, accumulated_offset, retained_count, retained_rms


def _align_regions_by_image_correlation(
    regions: list[np.ndarray],
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    requested_pixel_nm: float,
    iterations: int,
    template_points_nm: np.ndarray | None = None,
    template_image: np.ndarray | None = None,
    template_spacing_x_nm: float | None = None,
    template_spacing_y_nm: float | None = None,
    template_active_width_nm: float | None = None,
    template_active_height_nm: float | None = None,
    template_pixel_size_x_nm: float | None = None,
    template_pixel_size_y_nm: float | None = None,
    max_patch_pixels: int = 128,
    sparse_pose_site_count: int = 0,
    sparse_site_radius_nm: float = 7.5,
    sparse_min_site_localizations: int = 3,
    refinement_grid_points_nm: np.ndarray | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    aligned_image_output: list[np.ndarray] | None = None,
    candidate_image_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Independently classify and rigidly align candidates to the theoretical grid image."""
    if requested_pixel_nm <= 0:
        raise ValueError("Alignment pixel size must be greater than zero.")
    if iterations < 1:
        raise ValueError("Theoretical-template alignment passes must be at least 1.")
    side_nm = float(np.hypot(rectangle_width_nm, rectangle_height_nm))
    uses_custom_template = template_image is not None
    # A sparse asymmetric raster can have its localization median near an edge
    # or corner (an L shape is the extreme case). Give custom templates enough
    # surrounding canvas to render the complete footprint around that median.
    canvas_side_nm = 2.0 * side_nm if uses_custom_template else side_nm
    pixel_nm = max(float(requested_pixel_nm), canvas_side_nm / max_patch_pixels)
    cache_key = (
        id(regions),
        len(regions),
        float(canvas_side_nm),
        float(pixel_nm),
        int(max_patch_pixels),
    )
    cached_candidates = candidate_image_cache.get(cache_key) if candidate_image_cache is not None else None
    if cached_candidates is not None:
        cached_regions = cached_candidates[0]
        if len(cached_regions) != len(regions) or any(
            cached_region is not region for cached_region, region in zip(cached_regions, regions)
        ):
            cached_candidates = None
    if cached_candidates is None:
        centers = np.asarray([np.median(region, axis=0) for region in regions], dtype=float)
        rendered_images: list[np.ndarray] = []
        render_progress_every = max(1, len(regions) // 25)
        for index, (region, center) in enumerate(zip(regions, centers), start=1):
            rendered_images.append(
                _render_candidate_image(region, center, canvas_side_nm, pixel_nm, max(pixel_nm, 1.0))
            )
            if progress_callback and (index == len(regions) or index % render_progress_every == 0):
                progress_callback(
                    15.0 + 5.0 * index / max(len(regions), 1),
                    f"Rendering alignment thumbnails: {index:,}/{len(regions):,} candidates...",
                )
        images = np.asarray(rendered_images, dtype=np.float32)
        candidate_polar_ffts = tuple(np.fft.rfft(_polar_image(image), axis=0) for image in images)
        candidate_axis_info = tuple(_principal_axis_angle(image) for image in images)
        if candidate_image_cache is not None:
            candidate_image_cache[cache_key] = (
                tuple(regions),
                centers,
                images,
                candidate_polar_ffts,
                candidate_axis_info,
            )
    else:
        _cached_regions, centers, images, candidate_polar_ffts, candidate_axis_info = cached_candidates
        centers = np.asarray(centers, dtype=float)
        images = np.asarray(images, dtype=np.float32)
        if progress_callback:
            progress_callback(20.0, f"Reusing {len(regions):,} cached candidate thumbnails and polar transforms...")
    if len(images) == 0:
        return [], centers, np.empty(0), np.empty((0, 2)), np.empty(0), pixel_nm, np.empty((0, 0))

    # Build one fixed reference from the configured physical design. Candidates
    # are aligned and scored independently, so the result does not depend on
    # how many other objects happen to be present in the ROI.
    if template_points_nm is None:
        template_x = np.linspace(-rectangle_width_nm * 0.3, rectangle_width_nm * 0.3, 4)
        template_y = np.linspace(-rectangle_height_nm * 0.25, rectangle_height_nm * 0.25, 3)
        template_xx, template_yy = np.meshgrid(template_x, template_y)
        template_points_nm = np.column_stack([template_xx.ravel(), template_yy.ravel()])
    if uses_custom_template:
        template = prepare_custom_alignment_template(
            np.asarray(template_image, dtype=float),
            output_shape=images.shape[1:3],
            rectangle_width_nm=rectangle_width_nm,
            rectangle_height_nm=rectangle_height_nm,
            canvas_side_nm=canvas_side_nm,
            spacing_x_nm=template_spacing_x_nm,
            spacing_y_nm=template_spacing_y_nm,
            active_width_nm=template_active_width_nm,
            active_height_nm=template_active_height_nm,
            pixel_size_x_nm=template_pixel_size_x_nm,
            pixel_size_y_nm=template_pixel_size_y_nm,
        )
    else:
        template = _render_candidate_image(
            np.asarray(template_points_nm, dtype=float),
            np.zeros(2),
            side_nm,
            pixel_nm,
            max(pixel_nm, 1.5),
        )
    template_polar_fft = np.fft.rfft(_polar_image(template), axis=0)
    angles = np.zeros(len(images), dtype=float)
    shifts = np.zeros((len(images), 2), dtype=float)
    aligned_images = images.copy()
    correlations = np.full(len(images), -1.0, dtype=float)
    total_alignment_work = iterations * len(images)
    alignment_progress_every = max(1, total_alignment_work // 40)
    for iteration in range(iterations):
        for index, image in enumerate(images):
            if iteration == 0:
                trial_angles = _full_pose_rotation_candidates(
                    image,
                    template,
                    template_polar_fft,
                    rotational_period_deg=360.0 if uses_custom_template else 180.0,
                    image_polar_fft=candidate_polar_ffts[index],
                    image_axis_info=candidate_axis_info[index],
                )
            else:
                refinement_step = 2.0 / (2.0 ** (iteration - 1))
                trial_angles = angles[index] + refinement_step * np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
            best_score = -float("inf")
            best_pose_quality = -float("inf")
            best_shift_x = 0.0
            best_shift_y = 0.0
            best_angle = float(angles[index])
            best_image = aligned_images[index]
            for trial_angle in trial_angles:
                rotated_image = rotate(
                    image,
                    float(trial_angle),
                    reshape=False,
                    order=1,
                    mode="constant",
                    prefilter=False,
                )
                translation_hypotheses = _translation_hypotheses_to_reference(
                    rotated_image,
                    template,
                    maximum_candidates=(3 if iteration == 0 else 2) if sparse_pose_site_count > 0 else 1,
                    minimum_separation_pixels=max(3, int(round(sparse_site_radius_nm / pixel_nm))),
                    maximum_shift_pixels=images.shape[1] // 3 if uses_custom_template else None,
                )
                for shift_y, shift_x in translation_hypotheses:
                    fitted_angle = float(trial_angle)
                    fitted_shift = np.asarray([shift_x, shift_y], dtype=float) * pixel_nm
                    if sparse_pose_site_count > 0:
                        raw_angle = np.deg2rad(-float(trial_angle))
                        raw_rotation = np.asarray(
                            [[np.cos(raw_angle), -np.sin(raw_angle)], [np.sin(raw_angle), np.cos(raw_angle)]]
                        )
                        trial_points = (regions[index] - centers[index]) @ raw_rotation.T - fitted_shift
                        refined_points, correction, correction_offset = _refine_sparse_grid_pose(
                            trial_points,
                            np.asarray(template_points_nm, dtype=float),
                            site_radius_nm=sparse_site_radius_nm,
                            minimum_site_localizations=sparse_min_site_localizations,
                            iterations=2 if iteration == 0 else 1,
                        )
                        combined_rotation = correction @ raw_rotation
                        fitted_shift = fitted_shift @ correction.T - correction_offset
                        fitted_angle = -float(
                            np.rad2deg(np.arctan2(combined_rotation[1, 0], combined_rotation[0, 0]))
                        )
                        inside = (
                            (np.abs(refined_points[:, 0]) <= rectangle_width_nm / 2.0)
                            & (np.abs(refined_points[:, 1]) <= rectangle_height_nm / 2.0)
                        )
                        pose_quality = _sparse_pose_quality(
                            refined_points[inside],
                            np.asarray(template_points_nm, dtype=float),
                            site_radius_nm=sparse_site_radius_nm,
                            required_sites=sparse_pose_site_count,
                            minimum_site_localizations=sparse_min_site_localizations,
                            saturate_site_counts=uses_custom_template,
                        )
                        if not uses_custom_template and pose_quality < best_pose_quality - 1e-12:
                            continue
                        candidate_image = _render_candidate_image(
                            refined_points,
                            np.zeros(2, dtype=float),
                            canvas_side_nm,
                            pixel_nm,
                            max(pixel_nm, 1.0),
                        )
                        score = _boundary_template_correlation(candidate_image, template)
                    else:
                        candidate_image = ndimage_shift(
                            rotated_image,
                            shift=(-shift_y, -shift_x),
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )
                        score = _boundary_template_correlation(candidate_image, template)
                        pose_quality = score
                    # Once a sparse alignment template is available, choose
                    # the pose exclusively from its site-consensus geometry.
                    # Adding whole-candidate image correlation here lets very
                    # bright non-alignment digital groups pull an L_L template
                    # toward their strokes. Keep raster correlation only as a
                    # deterministic tie-breaker and reported QC measurement.
                    selection_quality = pose_quality
                    if selection_quality > best_pose_quality or (
                        abs(selection_quality - best_pose_quality) <= 1e-12 and score > best_score
                    ):
                        best_pose_quality = selection_quality
                        best_score = score
                        best_shift_x = float(fitted_shift[0] / pixel_nm)
                        best_shift_y = float(fitted_shift[1] / pixel_nm)
                        best_angle = fitted_angle
                        best_image = candidate_image
            angles[index] = best_angle
            shifts[index] = (best_shift_x, best_shift_y)
            aligned_images[index] = best_image
            correlations[index] = best_score
            completed_alignment_work = iteration * len(images) + index + 1
            if progress_callback and (
                completed_alignment_work == total_alignment_work
                or completed_alignment_work % alignment_progress_every == 0
            ):
                progress_callback(
                    20.0 + 52.0 * completed_alignment_work / total_alignment_work,
                    f"Theoretical-template alignment pass {iteration + 1}/{iterations}: "
                    f"candidate {index + 1:,}/{len(images):,}...",
                )

    # The raster search above finds the correct pose basin efficiently, but
    # its effective pixel size can leave a residual angular error.  Finish in
    # localization coordinates, where expected alignment marks landing in
    # dark regions explicitly lower the score.
    if sparse_pose_site_count > 0 and template_points_nm is not None and len(template_points_nm):
        for index, (region, center) in enumerate(zip(regions, centers)):
            raw_angle = np.deg2rad(-float(angles[index]))
            raw_rotation = np.asarray(
                [[np.cos(raw_angle), -np.sin(raw_angle)], [np.sin(raw_angle), np.cos(raw_angle)]],
                dtype=float,
            )
            shift_nm = np.asarray(shifts[index], dtype=float) * pixel_nm
            provisional = (region - center) @ raw_rotation.T - shift_nm
            correction, correction_offset, _contrast_score = (
                _refine_pose_by_alignment_site_contrast(
                    provisional,
                    np.asarray(template_points_nm, dtype=float),
                    site_radius_nm=sparse_site_radius_nm,
                    minimum_site_localizations=sparse_min_site_localizations,
                )
            )
            corrected = provisional @ correction.T + correction_offset
            combined_rotation = correction @ raw_rotation
            corrected_shift_nm = shift_nm @ correction.T - correction_offset
            if refinement_grid_points_nm is not None and len(refinement_grid_points_nm):
                lattice_correction, lattice_offset, _lattice_score = (
                    _refine_pose_by_full_lattice(
                        corrected,
                        np.asarray(refinement_grid_points_nm, dtype=float),
                        np.asarray(template_points_nm, dtype=float),
                        site_radius_nm=sparse_site_radius_nm,
                        minimum_site_localizations=sparse_min_site_localizations,
                    )
                )
                corrected = corrected @ lattice_correction.T + lattice_offset
                combined_rotation = lattice_correction @ combined_rotation
                corrected_shift_nm = (
                    corrected_shift_nm @ lattice_correction.T - lattice_offset
                )
                centroid_correction, centroid_offset, _site_count, _site_rms = (
                    _refine_pose_from_lattice_centroids(
                        corrected,
                        np.asarray(refinement_grid_points_nm, dtype=float),
                        site_radius_nm=sparse_site_radius_nm,
                        minimum_site_localizations=sparse_min_site_localizations,
                    )
                )
                corrected = corrected @ centroid_correction.T + centroid_offset
                combined_rotation = centroid_correction @ combined_rotation
                corrected_shift_nm = (
                    corrected_shift_nm @ centroid_correction.T - centroid_offset
                )
            angles[index] = -float(
                np.rad2deg(
                    np.arctan2(combined_rotation[1, 0], combined_rotation[0, 0])
                )
            )
            shifts[index] = corrected_shift_nm / pixel_nm
            corrected_image = _render_candidate_image(
                corrected,
                np.zeros(2, dtype=float),
                canvas_side_nm,
                pixel_nm,
                max(pixel_nm, 1.0),
            )
            aligned_images[index] = corrected_image
            correlations[index] = _boundary_template_correlation(corrected_image, template)

    if aligned_image_output is not None:
        aligned_image_output.extend(image.copy() for image in aligned_images)
    aligned_regions: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    local_corners = np.asarray([
        [-rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
        [rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
        [rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
        [-rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
    ])
    transform_progress_every = max(1, len(regions) // 20)
    for index, (region, center, angle_deg, (shift_x, shift_y)) in enumerate(
        zip(regions, centers, angles, shifts), start=1
    ):
        raw_angle = np.deg2rad(-angle_deg)
        raw_rotation = np.asarray([[np.cos(raw_angle), -np.sin(raw_angle)], [np.sin(raw_angle), np.cos(raw_angle)]])
        shift_nm = np.asarray([shift_x, shift_y]) * pixel_nm
        aligned = (region - center) @ raw_rotation.T - shift_nm
        inside = (np.abs(aligned[:, 0]) <= rectangle_width_nm / 2.0) & (np.abs(aligned[:, 1]) <= rectangle_height_nm / 2.0)
        aligned_regions.append(aligned[inside])
        corners.append((local_corners + shift_nm) @ raw_rotation + center)
        if progress_callback and (index == len(regions) or index % transform_progress_every == 0):
            progress_callback(
                72.0 + 3.0 * index / max(len(regions), 1),
                f"Applying fitted poses: {index:,}/{len(regions):,} candidates...",
            )
    reported_angles = np.mod(angles, 360.0)
    return aligned_regions, centers, np.asarray(corners), reported_angles, correlations, pixel_nm, template


def identify_origami_regions(
    points_nm: np.ndarray,
    *,
    pick_bin_size_nm: float,
    connect_distance_nm: float,
    density_threshold: float,
    min_candidate_points: int,
    max_candidate_points: int,
    rows: int | None = None,
    columns: int | None = None,
    spacing_x_nm: float | None = None,
    spacing_y_nm: float | None = None,
    rectangle_margin_nm: float = 0.0,
    g5m_sigma_min_nm: float = 1.0,
    g5m_sigma_max_nm: float = 8.0,
    g5m_min_locs: int = 20,
    g5m_max_rounds_without_best_bic: int = 3,
    site_match_radius_nm: float = 7.5,
    min_rectangle_confidence: float = 0.40,
    use_correlation_gate: bool = True,
    site_mask_radius_nm: float = 7.5,
    min_supported_sites: int = 0,
    min_site_evidence: float = 0.10,
    min_site_localizations: int = 3,
    min_supported_rows: int = 0,
    min_supported_columns: int = 0,
    max_site_spacing_error_nm: float = float("inf"),
    alignment_pixel_nm: float = 1.0,
    alignment_max_patch_pixels: int = 128,
    alignment_iterations: int = 3,
    column_offsets_nm: Sequence[float] | None = None,
    alignment_template_image: np.ndarray | None = None,
    template_pixel_size_x_nm: float | None = None,
    template_pixel_size_y_nm: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    compute_grid_blob_bic: bool = True,
    measure_sites: bool = True,
    precomputed_candidates: tuple[
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        tuple[float, float, float, float],
        np.ndarray,
    ] | None = None,
    candidate_image_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
) -> OrigamiPickResult:
    points_nm = np.asarray(points_nm, dtype=float)
    if points_nm.ndim != 2 or points_nm.shape[1] != 2:
        raise ValueError("Origami identification requires an N x 2 coordinate array.")
    points_nm = points_nm[np.all(np.isfinite(points_nm), axis=1)]
    if points_nm.size == 0:
        raise ValueError("There are no finite source points for origami identification.")
    if pick_bin_size_nm <= 0 or connect_distance_nm <= 0:
        raise ValueError("Pick-bin size and connection distance must be greater than zero.")
    if not 0.0 <= density_threshold <= 1.0:
        raise ValueError("Minimum density contrast must be between 0 and 1.")
    if min_candidate_points < 1 or max_candidate_points < min_candidate_points:
        raise ValueError("Invalid candidate point limits.")
    if not 0.0 <= min_rectangle_confidence <= 1.0:
        raise ValueError("Minimum rectangle confidence must be between 0 and 1.")
    if site_mask_radius_nm <= 0:
        raise ValueError("Site-mask radius must be greater than zero.")
    if min_supported_sites < 0 or min_supported_rows < 0 or min_supported_columns < 0 or min_site_localizations < 1:
        raise ValueError("Sparse-grid support counts cannot be negative, and minimum localizations per site must be positive.")
    if not 0.0 <= min_site_evidence <= 1.0:
        raise ValueError("Minimum site prominence must be between 0 and 1.")
    if max_site_spacing_error_nm <= 0:
        raise ValueError("Maximum site-spacing error must be greater than zero.")
    if alignment_max_patch_pixels < 16:
        raise ValueError("Maximum alignment image size must be at least 16 pixels.")
    if alignment_template_image is not None and (
        template_pixel_size_x_nm is not None and template_pixel_size_x_nm <= 0.0
        or template_pixel_size_y_nm is not None and template_pixel_size_y_nm <= 0.0
    ):
        raise ValueError("Custom-template pixel sizes must be greater than zero.")

    if progress_callback:
        progress_callback(2.0, "Building the spatial density map...")
    component_connect_distance_nm = None
    if (
        alignment_template_image is not None
        and spacing_x_nm is not None
        and spacing_y_nm is not None
    ):
        # Custom patterns can be hollow and have no dense central blob. Join
        # their supported arms across one expected site pitch, including the
        # positional uncertainty introduced by coarse spatial bins. Point
        # recovery still uses the user's narrower Connect distance, so this
        # changes component topology without sweeping in extra background.
        component_connect_distance_nm = (
            max(float(spacing_x_nm), float(spacing_y_nm))
            + math.sqrt(2.0) * float(pick_bin_size_nm)
        )
    if precomputed_candidates is None:
        raw_candidates = _pick_origami_regions(
            points_nm,
            pick_bin_size_nm,
            connect_distance_nm,
            density_threshold,
            component_connect_distance_nm=component_connect_distance_nm,
            progress_callback=(
                (lambda percent, message: progress_callback(2.0 + 0.1 * percent, message))
                if progress_callback is not None
                else None
            ),
        )
    else:
        raw_candidates = precomputed_candidates
        if progress_callback:
            progress_callback(12.0, f"Reusing {len(raw_candidates[0]):,} cached spatial candidates...")
    filtered_candidates, undersized_count = filter_origami_candidates(
        raw_candidates,
        min_candidate_points,
    )
    regions, density, contrast, extent, component_labels = filtered_candidates
    if progress_callback:
        skipped = (
            f"; skipped {undersized_count:,} components below {min_candidate_points:,} points"
            if undersized_count
            else ""
        )
        progress_callback(
            15.0,
            f"Found {len(regions):,} viable candidate regions{skipped}; building bounded image thumbnails...",
        )
    rectangle_width_nm = 0.0
    rectangle_height_nm = 0.0
    rectangle_corners: list[np.ndarray] = []
    rectangle_angles: list[float] = []
    rectangle_confidences: list[float] = []
    rectangle_matched_sites: list[int] = []
    rectangle_fit_rms: list[float] = []
    aligned_candidate_images: list[np.ndarray] = []
    grid = np.empty((0, 2), dtype=float)
    if (
        rows is not None
        and columns is not None
        and spacing_x_nm is not None
        and spacing_y_nm is not None
    ):
        if rectangle_margin_nm < 0:
            raise ValueError("Rectangle margin cannot be negative.")
        rectangle_width_nm = max(spacing_x_nm, (columns - 1) * spacing_x_nm) + 2.0 * rectangle_margin_nm
        rectangle_height_nm = max(spacing_y_nm, (rows - 1) * spacing_y_nm) + 2.0 * rectangle_margin_nm
        active_width_nm = max(spacing_x_nm, (columns - 1) * spacing_x_nm)
        active_height_nm = max(spacing_y_nm, (rows - 1) * spacing_y_nm)
        full_grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm, column_offsets_nm)
        active_width_nm = max(spacing_x_nm, float(np.ptp(full_grid[:, 0])))
        rectangle_width_nm = active_width_nm + 2.0 * rectangle_margin_nm
        if alignment_template_image is not None:
            grid = custom_template_site_points(
                alignment_template_image,
                rectangle_width_nm=rectangle_width_nm,
                rectangle_height_nm=rectangle_height_nm,
                spacing_x_nm=spacing_x_nm,
                spacing_y_nm=spacing_y_nm,
                active_width_nm=active_width_nm,
                active_height_nm=active_height_nm,
                pixel_size_x_nm=template_pixel_size_x_nm,
                pixel_size_y_nm=template_pixel_size_y_nm,
            )
        else:
            grid = full_grid
        template_grid_distances, template_lattice_indices = cKDTree(full_grid).query(grid, k=1)
        template_mapping_tolerance_nm = 0.35 * min(float(spacing_x_nm), float(spacing_y_nm))
        if np.any(template_grid_distances > template_mapping_tolerance_nm):
            raise ValueError("The alignment template does not map onto the configured lattice.")
        template_lattice_indices = np.asarray(template_lattice_indices, dtype=int)
        site_row_ids = template_lattice_indices // columns
        site_column_ids = template_lattice_indices % columns
        aligned_regions, _centers, fitted_corners, fitted_angles, correlations, effective_alignment_pixel, reference = (
            _align_regions_by_image_correlation(
                regions,
                rectangle_width_nm=rectangle_width_nm,
                rectangle_height_nm=rectangle_height_nm,
                requested_pixel_nm=alignment_pixel_nm,
                max_patch_pixels=alignment_max_patch_pixels,
                sparse_pose_site_count=min_supported_sites,
                sparse_site_radius_nm=site_mask_radius_nm,
                sparse_min_site_localizations=min_site_localizations,
                refinement_grid_points_nm=full_grid,
                iterations=alignment_iterations,
                template_points_nm=grid,
                template_image=alignment_template_image,
                template_spacing_x_nm=spacing_x_nm,
                template_spacing_y_nm=spacing_y_nm,
                template_active_width_nm=active_width_nm,
                template_active_height_nm=active_height_nm,
                template_pixel_size_x_nm=template_pixel_size_x_nm,
                template_pixel_size_y_nm=template_pixel_size_y_nm,
                progress_callback=progress_callback,
                aligned_image_output=aligned_candidate_images,
                candidate_image_cache=candidate_image_cache,
            )
        )
        rectangle_corners = list(fitted_corners)
        rectangle_angles = list(fitted_angles)
        rectangle_confidences = list(correlations)
        rectangle_matched_sites = [0] * len(regions)
        rectangle_fit_rms = [float("nan")] * len(regions)
    else:
        aligned_regions = [region - np.median(region, axis=0) for region in regions]
        effective_alignment_pixel = float(alignment_pixel_nm)
        reference = np.empty((0, 0))
        for region in regions:
            x_min, y_min = np.min(region, axis=0)
            x_max, y_max = np.max(region, axis=0)
            rectangle_corners.append(np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
            rectangle_angles.append(0.0)
            rectangle_confidences.append(1.0)
            rectangle_matched_sites.append(0)
            rectangle_fit_rms.append(0.0)

    if (
        measure_sites
        and rows is not None
        and columns is not None
        and spacing_x_nm is not None
        and spacing_y_nm is not None
    ):
        if progress_callback:
            progress_callback(
                76.0,
                f"Measuring site-versus-gap density for {len(aligned_regions):,} candidates...",
            )
        site_gap_contrast, on_site_fraction, _site_area_fraction = site_gap_contrast_for_regions(
            aligned_regions,
            grid,
            rectangle_width_nm=rectangle_width_nm,
            rectangle_height_nm=rectangle_height_nm,
            site_radius_nm=site_mask_radius_nm,
        )
    else:
        site_gap_contrast = np.ones(len(aligned_regions), dtype=float)
        on_site_fraction = np.ones(len(aligned_regions), dtype=float)
    if progress_callback:
        progress_callback(
            79.0,
            "Site-versus-gap density complete; evaluating individual candidates..."
            if measure_sites
            else "Rigid alignment complete; site measurement deferred to Stage 4.",
        )

    supported_site_counts = np.zeros(len(aligned_regions), dtype=int)
    supported_row_counts = np.zeros(len(aligned_regions), dtype=int)
    supported_column_counts = np.zeros(len(aligned_regions), dtype=int)
    site_spacing_rms = np.full(len(aligned_regions), float("inf"), dtype=float)
    site_spacing_max_errors = np.full(len(aligned_regions), float("inf"), dtype=float)
    grid_blob_delta_bic = np.full(len(aligned_regions), float("inf"), dtype=float)
    site_count_matrix = np.empty((len(aligned_regions), 0), dtype=int)
    site_prominence_matrix = np.empty((len(aligned_regions), 0), dtype=float)
    lattice_site_count_matrix = np.empty((len(aligned_regions), 0), dtype=int)
    lattice_site_prominence_matrix = np.empty((len(aligned_regions), 0), dtype=float)
    lattice_supported_matrix = np.empty((len(aligned_regions), 0), dtype=bool)
    site_peak_positions = np.empty((len(aligned_regions), 0, 2), dtype=float)
    site_boundary_reference_positions = np.empty((len(aligned_regions), 0, 2), dtype=float)
    site_boundary_points = np.empty((len(aligned_regions), 0, 32, 2), dtype=float)
    site_centroids = np.empty((len(aligned_regions), 0, 2), dtype=float)
    if (
        measure_sites
        and rows is not None
        and columns is not None
        and spacing_x_nm is not None
        and spacing_y_nm is not None
    ):
        site_count_matrix = np.zeros((len(aligned_regions), len(grid)), dtype=int)
        site_prominence_matrix = np.zeros((len(aligned_regions), len(grid)), dtype=float)
        lattice_site_count_matrix = np.zeros((len(aligned_regions), len(full_grid)), dtype=int)
        lattice_site_prominence_matrix = np.zeros((len(aligned_regions), len(full_grid)), dtype=float)
        lattice_supported_matrix = np.zeros((len(aligned_regions), len(full_grid)), dtype=bool)
        site_peak_positions = np.full((len(aligned_regions), len(grid), 2), np.nan, dtype=float)
        site_boundary_reference_positions = np.full(
            (len(aligned_regions), len(grid), 2), np.nan, dtype=float
        )
        site_boundary_points = np.full(
            (len(aligned_regions), len(grid), 32, 2), np.nan, dtype=float
        )
        site_centroids = np.full((len(aligned_regions), len(grid), 2), np.nan, dtype=float)
        candidate_progress_every = max(1, len(aligned_regions) // 40)
        for index, region in enumerate(aligned_regions):
            site_evidence = sparse_site_evidence_diagnostics(
                region,
                full_grid,
                site_radius_nm=site_mask_radius_nm,
            )
            lattice_site_count_matrix[index] = site_evidence.counts
            lattice_site_prominence_matrix[index] = site_evidence.prominence
            lattice_supported = (
                (site_evidence.counts >= int(min_site_localizations))
                & (site_evidence.prominence >= float(min_site_evidence))
            )
            lattice_supported_matrix[index] = lattice_supported
            site_count_matrix[index] = site_evidence.counts[template_lattice_indices]
            site_prominence_matrix[index] = site_evidence.prominence[template_lattice_indices]
            site_peak_positions[index] = site_evidence.peak_positions_nm[template_lattice_indices]
            site_boundary_reference_positions[index] = site_evidence.boundary_reference_positions_nm[
                template_lattice_indices
            ]
            site_boundary_points[index] = site_evidence.boundary_points_nm[template_lattice_indices]
            supported = lattice_supported[template_lattice_indices]
            site_centroids[index] = supported_site_centroids(
                region,
                grid,
                supported,
                site_radius_nm=site_mask_radius_nm,
            )
            supported_indices = np.flatnonzero(supported)
            supported_site_counts[index] = len(supported_indices)
            if len(supported_indices):
                supported_row_counts[index] = len(np.unique(site_row_ids[supported_indices]))
                supported_column_counts[index] = len(np.unique(site_column_ids[supported_indices]))
            site_spacing_rms[index], site_spacing_max_errors[index] = supported_site_spacing_errors(
                region,
                grid,
                supported,
                site_radius_nm=site_mask_radius_nm,
            )
            if compute_grid_blob_bic:
                grid_blob_delta_bic[index] = grid_vs_blob_delta_bic(
                    region,
                    grid,
                    rectangle_width_nm=rectangle_width_nm,
                    rectangle_height_nm=rectangle_height_nm,
                    pixel_nm=effective_alignment_pixel,
                )
            completed_candidates = index + 1
            if progress_callback and (
                completed_candidates == len(aligned_regions)
                or completed_candidates % candidate_progress_every == 0
            ):
                progress_callback(
                    79.0 + 20.0 * completed_candidates / max(len(aligned_regions), 1),
                    "Evaluating site prominence, spacing, and ΔBIC QC: "
                    f"{completed_candidates:,}/{len(aligned_regions):,} candidates...",
                )
    elif progress_callback:
        progress_callback(99.0, "Candidate measurements complete; applying acceptance limits...")

    point_counts = np.asarray([len(region) for region in aligned_regions], dtype=int)
    original_point_counts = np.asarray([len(region) for region in regions], dtype=int)
    crop_retained_fractions = np.divide(
        point_counts,
        original_point_counts,
        out=np.zeros(len(point_counts), dtype=float),
        where=original_point_counts > 0,
    )
    confidence_array = np.asarray(rectangle_confidences, dtype=float)
    accepted_mask = (
        (point_counts >= min_candidate_points)
        & (point_counts <= max_candidate_points)
        & ((confidence_array >= min_rectangle_confidence) if use_correlation_gate else True)
        & (supported_site_counts >= min_supported_sites)
        & (supported_row_counts >= min_supported_rows)
        & (supported_column_counts >= min_supported_columns)
        & (site_spacing_max_errors <= max_site_spacing_error_nm)
    )
    if alignment_template_image is not None:
        corner_counts = alignment_corner_counts(aligned_regions, grid, site_mask_radius_nm)
        accepted_mask &= np.all(corner_counts >= min_site_localizations, axis=1)
    bounds = np.asarray(
        [
            [
                float(np.min(region[:, 0])),
                float(np.max(region[:, 0])),
                float(np.min(region[:, 1])),
                float(np.max(region[:, 1])),
            ]
            for region in regions
        ],
        dtype=float,
    ) if regions else np.empty((0, 4), dtype=float)
    result = OrigamiPickResult(
        regions=regions,
        aligned_regions=aligned_regions,
        accepted_mask=accepted_mask,
        point_counts=point_counts,
        original_point_counts=original_point_counts,
        crop_retained_fractions=crop_retained_fractions,
        bounds_nm=bounds,
        rectangle_corners_nm=np.asarray(rectangle_corners, dtype=float) if rectangle_corners else np.empty((0, 4, 2)),
        rectangle_angles_deg=np.asarray(rectangle_angles, dtype=float),
        rectangle_confidence=confidence_array,
        site_gap_contrast=np.asarray(site_gap_contrast, dtype=float),
        on_site_fraction=np.asarray(on_site_fraction, dtype=float),
        site_mask_radius_nm=float(site_mask_radius_nm),
        template_points_nm=np.asarray(grid, dtype=float),
        lattice_site_localization_counts=lattice_site_count_matrix,
        lattice_site_prominence=lattice_site_prominence_matrix,
        lattice_supported_sites=lattice_supported_matrix,
        site_localization_counts=site_count_matrix,
        site_prominence=site_prominence_matrix,
        site_peak_positions_nm=site_peak_positions,
        site_boundary_reference_positions_nm=site_boundary_reference_positions,
        site_boundary_points_nm=site_boundary_points,
        site_centroids_nm=site_centroids,
        supported_site_count=supported_site_counts,
        supported_row_count=supported_row_counts,
        supported_column_count=supported_column_counts,
        site_spacing_rms_nm=site_spacing_rms,
        site_spacing_max_error_nm=site_spacing_max_errors,
        grid_vs_blob_delta_bic=grid_blob_delta_bic,
        rectangle_matched_site_count=np.asarray(rectangle_matched_sites, dtype=int),
        rectangle_fit_rms_nm=np.asarray(rectangle_fit_rms, dtype=float),
        rectangle_width_nm=float(rectangle_width_nm),
        rectangle_height_nm=float(rectangle_height_nm),
        density_image=density,
        density_contrast=contrast,
        density_component_labels=component_labels,
        density_extent_nm=extent,
        density_threshold=float(density_threshold),
        alignment_pixel_nm=float(effective_alignment_pixel),
        alignment_canvas_side_nm=(
            (2.0 if alignment_template_image is not None else 1.0)
            * float(np.hypot(rectangle_width_nm, rectangle_height_nm))
        ),
        alignment_reference_image=reference,
        alignment_candidate_images=np.asarray(aligned_candidate_images, dtype=np.float32),
    )
    if progress_callback:
        progress_callback(
            100.0,
            f"Identification complete: {result.accepted_count}/{len(result.regions)} candidates passed point, site-prominence, grid-coverage, and spacing limits.",
        )
    return result


def align_picked_origamis(
    picked_regions: list[np.ndarray],
    *,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    site_radius_nm: float,
    g5m_sigma_min_nm: float = 1.0,
    g5m_sigma_max_nm: float = 8.0,
    g5m_min_locs: int = 10,
    g5m_max_rounds_without_best_bic: int = 3,
    rectangle_corners_nm: list[np.ndarray] | None = None,
    prealigned: bool = False,
    source_centers_nm: np.ndarray | None = None,
    allow_mirror: bool = False,
    initially_rejected_count: int = 0,
    use_g5m: bool = True,
    direct_min_site_localizations: int = 1,
    direct_min_site_evidence: float = 0.0,
    grid_points_nm: np.ndarray | None = None,
    symmetrize_180: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> OrigamiAnalysisResult:
    if not picked_regions:
        raise ValueError("No identified origamis are available. Adjust the identification settings and run Identify Origami again.")
    if site_radius_nm <= 0:
        raise ValueError("Site radius must be greater than zero.")
    if use_g5m and (g5m_sigma_min_nm <= 0 or g5m_sigma_max_nm < g5m_sigma_min_nm):
        raise ValueError("G5M sigma bounds must be positive and ordered minimum to maximum.")
    if use_g5m and (g5m_min_locs < 1 or g5m_max_rounds_without_best_bic < 1):
        raise ValueError("G5M minimum localizations and BIC patience must be at least 1.")
    if direct_min_site_localizations < 1 or not 0.0 <= direct_min_site_evidence <= 1.0:
        raise ValueError("Direct assignment requires a positive site count and site prominence from 0 to 1.")

    if grid_points_nm is None:
        grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
    else:
        grid = np.asarray(grid_points_nm, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid) or not np.all(np.isfinite(grid)):
            raise ValueError("Custom template sites must be a non-empty N x 2 coordinate array.")
    if rectangle_corners_nm is not None and len(rectangle_corners_nm) != len(picked_regions):
        raise ValueError("Each picked origami must have one fitted rectangle.")
    supplied_corners = rectangle_corners_nm if rectangle_corners_nm is not None else [None] * len(picked_regions)
    if source_centers_nm is not None and len(source_centers_nm) != len(picked_regions):
        raise ValueError("Each picked origami must have one source center.")
    centers = np.asarray(source_centers_nm, dtype=float) if source_centers_nm is not None else np.asarray(
        [np.mean(region, axis=0) for region in picked_regions]
    )
    candidates = [(center, region, corners) for center, region, corners in zip(centers, picked_regions, supplied_corners)]
    rejected = int(initially_rejected_count)

    # Preserve the accepted-source order so one physical origami keeps the same
    # stable ID across identification, galleries, statistics, and CSV export.
    running_pattern: np.ndarray | None = None
    accepted_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, (center, region, rectangle_corners) in enumerate(candidates, start=1):
        choices: list[tuple[np.ndarray, np.ndarray, float]] = []
        if prealigned:
            aligned_candidates = [(region, region)]
            if allow_mirror:
                mirrored = region * np.asarray([-1.0, 1.0])
                aligned_candidates.append((mirrored, mirrored))
        elif use_g5m:
            _g5m_labels, cluster_centers = fit_picasso_g5m_components(
                region,
                min_locs=g5m_min_locs,
                sigma_min_nm=g5m_sigma_min_nm,
                sigma_max_nm=g5m_sigma_max_nm,
                max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
            )
            alignment_points = cluster_centers if len(cluster_centers) >= 3 else region
            if rectangle_corners is None:
                aligned_candidates = _pca_aligned_candidates(region, alignment_points, grid, allow_mirror)
            else:
                aligned_candidates = _rectangle_aligned_candidates(region, alignment_points, rectangle_corners, allow_mirror)
        else:
            if rectangle_corners is None:
                aligned_candidates = _pca_aligned_candidates(region, region, grid, allow_mirror)
            else:
                aligned_candidates = _rectangle_aligned_candidates(region, region, rectangle_corners, allow_mirror)
        for candidate, candidate_alignment in aligned_candidates:
            refined = candidate if prealigned else _refine_rotation_to_grid(candidate, candidate_alignment, grid, site_radius_nm)
            choices.append(_fit_translation_and_sites(refined, grid, site_radius_nm))
        match_fractions = [float(np.sum(item[1]) / len(item[0])) for item in choices]
        best_match = max(match_fractions)
        eligible = [
            item for item, match_fraction in zip(choices, match_fractions)
            if match_fraction >= best_match - 1e-9
        ]
        if running_pattern is None:
            chosen = min(eligible, key=lambda item: item[2])
        else:
            reference = np.log1p(running_pattern)

            def choice_score(item: tuple[np.ndarray, np.ndarray, float]) -> tuple[float, float]:
                pattern = np.log1p(item[1].astype(float))
                scale = float(np.linalg.norm(reference) * np.linalg.norm(pattern))
                similarity = float(np.dot(reference, pattern) / scale) if scale > 0 else 0.0
                return similarity, -item[2]

            chosen = max(eligible, key=choice_score)
        aligned, counts, rms = chosen
        match_fraction = float(np.sum(counts) / len(aligned))
        if use_g5m:
            raw_cluster_counts, cluster_labels, cluster_centers, cluster_sites = cluster_aligned_origami_sites(
                aligned,
                grid,
                g5m_sigma_min_nm=g5m_sigma_min_nm,
                g5m_sigma_max_nm=g5m_sigma_max_nm,
                g5m_min_locs=g5m_min_locs,
                g5m_max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
                site_match_radius_nm=site_radius_nm,
            )
        else:
            distances, nearest_sites = cKDTree(grid).query(aligned, k=1)
            supported_sites = supported_grid_site_mask(
                aligned,
                grid,
                site_radius_nm=site_radius_nm,
                min_site_localizations=direct_min_site_localizations,
                min_site_evidence=direct_min_site_evidence,
            )
            matched = (distances <= site_radius_nm) & supported_sites[nearest_sites]
            cluster_sites = np.unique(nearest_sites[matched]).astype(int)
            cluster_labels = np.full(len(aligned), -1, dtype=int)
            cluster_centers_rows: list[np.ndarray] = []
            raw_cluster_counts = np.zeros(len(grid), dtype=int)
            for cluster_label, site_index in enumerate(cluster_sites):
                members = matched & (nearest_sites == site_index)
                cluster_labels[members] = cluster_label
                raw_cluster_counts[site_index] = int(np.count_nonzero(members))
                cluster_centers_rows.append(np.mean(aligned[members], axis=0))
            cluster_centers = (
                np.vstack(cluster_centers_rows) if cluster_centers_rows else np.empty((0, 2), dtype=float)
            )
        if symmetrize_180:
            # A rectangular grid is invariant under 180-degree rotation. Give
            # both directions equal statistical weight when no directional
            # custom template resolves that ambiguity.
            rotated_cluster_counts = raw_cluster_counts[::-1]
            cluster_counts = 0.5 * (raw_cluster_counts.astype(float) + rotated_cluster_counts.astype(float))
            cluster_occupancy = 0.5 * (
                (raw_cluster_counts > 0).astype(float) + (rotated_cluster_counts > 0).astype(float)
            )
        else:
            cluster_counts = raw_cluster_counts.astype(float)
            cluster_occupancy = (raw_cluster_counts > 0).astype(float)
        accepted_rows.append(
            (
                aligned,
                center,
                cluster_counts,
                cluster_occupancy,
                len(region),
                rms,
                match_fraction,
                cluster_labels,
                cluster_centers,
                cluster_sites,
            )
        )
        running_pattern = cluster_counts.astype(float) if running_pattern is None else running_pattern + cluster_counts
        if progress_callback and (index == len(candidates) or index % max(1, len(candidates) // 20) == 0):
            stage = "Picasso G5M docking-site clustering" if use_g5m else "fast grid-site assignment"
            progress_callback(f"Origami analysis: {stage} {index}/{len(candidates)} candidates...")

    return OrigamiAnalysisResult(
        aligned_points=[row[0] for row in accepted_rows],
        centers_nm=np.vstack([row[1] for row in accepted_rows]),
        site_counts=np.vstack([row[2] for row in accepted_rows]),
        site_occupancy=np.vstack([row[3] for row in accepted_rows]),
        cluster_labels=[row[7] for row in accepted_rows],
        cluster_centers_nm=[row[8] for row in accepted_rows],
        cluster_site_indices=[row[9] for row in accepted_rows],
        source_point_counts=np.asarray([row[4] for row in accepted_rows], dtype=int),
        alignment_rms_nm=np.asarray([row[5] for row in accepted_rows], dtype=float),
        grid_match_fraction=np.asarray([row[6] for row in accepted_rows], dtype=float),
        grid_points_nm=grid,
        rows=rows,
        columns=columns,
        g5m_sigma_min_nm=float(g5m_sigma_min_nm),
        g5m_sigma_max_nm=float(g5m_sigma_max_nm),
        g5m_min_locs=int(g5m_min_locs),
        g5m_max_rounds_without_best_bic=int(g5m_max_rounds_without_best_bic),
        site_match_radius_nm=float(site_radius_nm),
        direct_min_site_localizations=int(direct_min_site_localizations),
        direct_min_site_evidence=float(direct_min_site_evidence),
        rejected_candidate_count=rejected,
        symmetrized_180=bool(symmetrize_180),
        clustering_method="Picasso G5M" if use_g5m else "Supported-site direct assignment",
    )


def analyze_origami_regions(
    points_nm: np.ndarray,
    *,
    pick_bin_size_nm: float,
    connect_distance_nm: float,
    density_threshold: float,
    min_candidate_points: int,
    max_candidate_points: int,
    rows: int,
    columns: int,
    spacing_x_nm: float,
    spacing_y_nm: float,
    site_radius_nm: float,
    g5m_sigma_min_nm: float = 1.0,
    g5m_sigma_max_nm: float = 8.0,
    g5m_min_locs: int = 10,
    g5m_max_rounds_without_best_bic: int = 3,
    allow_mirror: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> OrigamiAnalysisResult:
    if progress_callback:
        progress_callback(f"Origami analysis: binning {len(points_nm):,} source points for fast whole-origami picking...")
    picks = identify_origami_regions(
        points_nm,
        pick_bin_size_nm=pick_bin_size_nm,
        connect_distance_nm=connect_distance_nm,
        density_threshold=density_threshold,
        min_candidate_points=min_candidate_points,
        max_candidate_points=max_candidate_points,
    )
    return align_picked_origamis(
        picks.accepted_regions,
        rows=rows,
        columns=columns,
        spacing_x_nm=spacing_x_nm,
        spacing_y_nm=spacing_y_nm,
        site_radius_nm=site_radius_nm,
        g5m_sigma_min_nm=g5m_sigma_min_nm,
        g5m_sigma_max_nm=g5m_sigma_max_nm,
        g5m_min_locs=g5m_min_locs,
        g5m_max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
        allow_mirror=allow_mirror,
        initially_rejected_count=len(picks.regions) - picks.accepted_count,
        progress_callback=progress_callback,
    )
