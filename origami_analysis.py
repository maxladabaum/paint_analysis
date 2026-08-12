"""Detection, alignment, and site-level statistics for DNA origami localizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
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
    rejected_candidate_count: int
    symmetrized_180: bool

    @property
    def origami_count(self) -> int:
        return len(self.aligned_points)


@dataclass
class OrigamiPickResult:
    regions: list[np.ndarray]
    aligned_regions: list[np.ndarray]
    accepted_mask: np.ndarray
    point_counts: np.ndarray
    bounds_nm: np.ndarray
    rectangle_corners_nm: np.ndarray
    rectangle_angles_deg: np.ndarray
    rectangle_confidence: np.ndarray
    rectangle_matched_site_count: np.ndarray
    rectangle_fit_rms_nm: np.ndarray
    rectangle_width_nm: float
    rectangle_height_nm: float
    density_image: np.ndarray
    density_contrast: np.ndarray
    density_extent_nm: tuple[float, float, float, float]
    density_threshold: float
    alignment_pixel_nm: float
    alignment_reference_image: np.ndarray

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


def ideal_grid_points(rows: int, columns: int, spacing_x_nm: float, spacing_y_nm: float) -> np.ndarray:
    if rows < 1 or columns < 1:
        raise ValueError("Grid rows and columns must both be at least 1.")
    if spacing_x_nm <= 0 or spacing_y_nm <= 0:
        raise ValueError("Grid spacing must be greater than zero.")
    x = (np.arange(columns, dtype=float) - (columns - 1) / 2.0) * spacing_x_nm
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
) -> dict[str, object]:
    """Render aligned points over a grid-defined field at a known resolution."""
    if not aligned_points:
        raise ValueError("No aligned origami points are available to render.")
    if pixel_size_nm <= 0:
        raise ValueError("Overlay pixel size must be greater than zero.")
    if padding_nm < 0 or blur_nm < 0:
        raise ValueError("Overlay padding and blur cannot be negative.")
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
    points = np.vstack(aligned_points)
    density, _x_edges, _y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(x_bins, y_bins),
        range=((extent[0], extent[1]), (extent[2], extent[3])),
    )
    density /= len(aligned_points)
    if blur_nm > 0:
        density = gaussian_filter(
            density,
            sigma=(blur_nm / effective_x_nm, blur_nm / effective_y_nm),
            mode="constant",
        )
    in_view = (
        (points[:, 0] >= extent[0])
        & (points[:, 0] <= extent[1])
        & (points[:, 1] >= extent[2])
        & (points[:, 1] <= extent[3])
    )
    return {
        "image": density.T,
        "extent": extent,
        "effective_pixel_x_nm": float(effective_x_nm),
        "effective_pixel_y_nm": float(effective_y_nm),
        "rendered_point_count": int(np.count_nonzero(in_view)),
        "total_point_count": int(len(points)),
        "blur_nm": float(blur_nm),
    }


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
    auto_max = 0.5 * float(np.max(density)) if density.size else 1.0
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


def _pick_origami_regions(
    points_nm: np.ndarray,
    bin_size_nm: float,
    connect_distance_nm: float,
    density_threshold: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Connect only dense bins, then recover original points near each object."""
    density, contrast, extent, x_edges, y_edges = density_map_for_origami_picking(points_nm, bin_size_nm)
    active_indices = np.argwhere(contrast >= density_threshold)
    if len(active_indices) == 0:
        return [], density, contrast, extent
    cell_centers = np.column_stack(
        [
            x_edges[active_indices[:, 0]] + bin_size_nm / 2.0,
            y_edges[active_indices[:, 1]] + bin_size_nm / 2.0,
        ]
    )
    if len(cell_centers) == 1:
        distances = np.linalg.norm(points_nm - cell_centers[0], axis=1)
        return [points_nm[distances <= connect_distance_nm]], density, contrast, extent

    pairs = cKDTree(cell_centers).query_pairs(connect_distance_nm, output_type="ndarray")
    parents = np.arange(len(cell_centers), dtype=np.int64)
    ranks = np.zeros(len(cell_centers), dtype=np.uint8)

    def find(item: int) -> int:
        root = item
        while parents[root] != root:
            root = int(parents[root])
        while parents[item] != item:
            parent = int(parents[item])
            parents[item] = root
            item = parent
        return root

    for left_value, right_value in pairs:
        left = find(int(left_value))
        right = find(int(right_value))
        if left == right:
            continue
        if ranks[left] < ranks[right]:
            left, right = right, left
        parents[right] = left
        if ranks[left] == ranks[right]:
            ranks[left] += 1

    roots = np.asarray([find(index) for index in range(len(cell_centers))], dtype=np.int64)
    _unique_roots, cell_components = np.unique(roots, return_inverse=True)
    nearest_distance, nearest_cell = cKDTree(cell_centers).query(points_nm, k=1)
    assigned = nearest_distance <= connect_distance_nm
    point_components = cell_components[nearest_cell[assigned]]
    assigned_points = points_nm[assigned]
    order = np.argsort(point_components, kind="stable")
    sorted_components = point_components[order]
    boundaries = np.flatnonzero(np.diff(sorted_components)) + 1
    regions = [region for region in np.split(assigned_points[order], boundaries) if len(region)]
    return regions, density, contrast, extent


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


def _rotation_from_polar(image: np.ndarray, reference_polar_fft: np.ndarray) -> float:
    polar_fft = np.fft.rfft(_polar_image(image), axis=0)
    correlation = np.fft.irfft(
        np.sum(polar_fft * np.conj(reference_polar_fft), axis=1),
        n=180,
    )
    peak = int(np.argmax(correlation))
    left = float(correlation[(peak - 1) % len(correlation)])
    middle = float(correlation[peak])
    right = float(correlation[(peak + 1) % len(correlation)])
    denominator = left - 2.0 * middle + right
    subpixel = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
    refined_peak = peak + float(np.clip(subpixel, -0.5, 0.5))
    if refined_peak > 90.0:
        refined_peak -= 180.0
    return refined_peak * 2.0


def _translation_to_reference(image: np.ndarray, reference_fft: np.ndarray) -> tuple[int, int]:
    cross_power = np.fft.fft2(image) * np.conj(reference_fft)
    correlation = np.fft.ifft2(cross_power).real
    max_shift = max(2, image.shape[0] // 8)
    allowed = np.zeros(image.shape, dtype=bool)
    allowed[: max_shift + 1, : max_shift + 1] = True
    allowed[: max_shift + 1, -max_shift:] = True
    allowed[-max_shift:, : max_shift + 1] = True
    allowed[-max_shift:, -max_shift:] = True
    peak = np.unravel_index(int(np.argmax(np.where(allowed, correlation, -np.inf))), image.shape)
    shifts = [int(value) for value in peak]
    for axis in range(2):
        if shifts[axis] > image.shape[axis] // 2:
            shifts[axis] -= image.shape[axis]
    return shifts[0], shifts[1]


def _align_regions_by_image_correlation(
    regions: list[np.ndarray],
    *,
    rectangle_width_nm: float,
    rectangle_height_nm: float,
    requested_pixel_nm: float,
    iterations: int,
    template_points_nm: np.ndarray | None = None,
    max_patch_pixels: int = 64,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Classify and rigidly align candidate images to an iteratively refined template."""
    if requested_pixel_nm <= 0:
        raise ValueError("Alignment pixel size must be greater than zero.")
    if iterations < 1:
        raise ValueError("Template refinement iterations must be at least 1.")
    side_nm = float(np.hypot(rectangle_width_nm, rectangle_height_nm))
    pixel_nm = max(float(requested_pixel_nm), side_nm / max_patch_pixels)
    centers = np.asarray([np.median(region, axis=0) for region in regions], dtype=float)
    images = np.asarray([
        _render_candidate_image(region, center, side_nm, pixel_nm, max(pixel_nm, 1.0))
        for region, center in zip(regions, centers)
    ], dtype=np.float32)
    if len(images) == 0:
        return [], centers, np.empty(0), np.empty((0, 2)), np.empty(0), pixel_nm, np.empty((0, 0))

    contrast = np.asarray([np.percentile(image, 99) - np.percentile(image, 50) for image in images])
    reference = images[int(np.argmax(contrast))].copy()
    angles = np.zeros(len(images), dtype=float)
    shifts = np.zeros((len(images), 2), dtype=float)
    aligned_images = images.copy()
    for iteration in range(iterations):
        reference_polar_fft = np.fft.rfft(_polar_image(reference), axis=0)
        reference_fft = np.fft.fft2(reference)
        for index, image in enumerate(images):
            angle = _rotation_from_polar(image, reference_polar_fft)
            rotated = rotate(image, angle, reshape=False, order=1, mode="constant", prefilter=False)
            shift_y, shift_x = _translation_to_reference(rotated, reference_fft)
            aligned_images[index] = np.roll(rotated, (-shift_y, -shift_x), axis=(0, 1))
            angles[index] = angle
            shifts[index] = (shift_x, shift_y)
        reference = np.median(aligned_images, axis=0)
        reference -= np.mean(reference)
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm > 0:
            reference /= reference_norm
        if progress_callback:
            progress_callback(
                25.0 + 55.0 * (iteration + 1) / iterations,
                f"Image cross-correlation pass {iteration + 1}/{iterations}: {len(regions):,} candidates...",
            )

    correlations = np.asarray([
        float(np.clip(np.dot(image.ravel(), reference.ravel()) / max(np.linalg.norm(image), 1e-12), -1.0, 1.0))
        for image in aligned_images
    ])
    # Use the known rectangular aspect ratio only to create an absolute orientation
    # target. Candidate-to-candidate alignment above remains independent of sites.
    if template_points_nm is None:
        template_x = np.linspace(-rectangle_width_nm * 0.3, rectangle_width_nm * 0.3, 4)
        template_y = np.linspace(-rectangle_height_nm * 0.25, rectangle_height_nm * 0.25, 3)
        template_xx, template_yy = np.meshgrid(template_x, template_y)
        template_points_nm = np.column_stack([template_xx.ravel(), template_yy.ravel()])
    template = _render_candidate_image(
        np.asarray(template_points_nm, dtype=float),
        np.zeros(2),
        side_nm,
        pixel_nm,
        max(pixel_nm, 1.5),
    )
    canonical_image_angle_deg = _rotation_from_polar(reference, np.fft.rfft(_polar_image(template), axis=0))
    canonical_angle_deg = -canonical_image_angle_deg
    canonical_angle = np.deg2rad(canonical_angle_deg)
    canonical_rotation = np.asarray([
        [np.cos(canonical_angle), -np.sin(canonical_angle)],
        [np.sin(canonical_angle), np.cos(canonical_angle)],
    ])
    reference = rotate(reference, -canonical_angle_deg, reshape=False, order=1, mode="constant", prefilter=False)
    aligned_regions: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    local_corners = np.asarray([
        [-rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
        [rectangle_width_nm / 2.0, -rectangle_height_nm / 2.0],
        [rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
        [-rectangle_width_nm / 2.0, rectangle_height_nm / 2.0],
    ])
    for region, center, angle_deg, (shift_x, shift_y) in zip(regions, centers, angles, shifts):
        raw_angle = np.deg2rad(-angle_deg)
        raw_rotation = np.asarray([[np.cos(raw_angle), -np.sin(raw_angle)], [np.sin(raw_angle), np.cos(raw_angle)]])
        shift_nm = np.asarray([shift_x, shift_y]) * pixel_nm
        aligned = ((region - center) @ raw_rotation.T - shift_nm) @ canonical_rotation.T
        inside = (np.abs(aligned[:, 0]) <= rectangle_width_nm / 2.0) & (np.abs(aligned[:, 1]) <= rectangle_height_nm / 2.0)
        aligned_regions.append(aligned[inside])
        corners.append((local_corners @ canonical_rotation + shift_nm) @ raw_rotation + center)
    reported_angles = np.mod(angles - canonical_angle_deg, 360.0)
    return aligned_regions, centers, np.asarray(corners), reported_angles, correlations, pixel_nm, reference


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
    min_rectangle_confidence: float = 0.0,
    alignment_pixel_nm: float = 1.0,
    alignment_iterations: int = 2,
    progress_callback: Callable[[float, str], None] | None = None,
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

    if progress_callback:
        progress_callback(2.0, "Building the spatial density map...")
    regions, density, contrast, extent = _pick_origami_regions(
        points_nm, pick_bin_size_nm, connect_distance_nm, density_threshold
    )
    if progress_callback:
        progress_callback(15.0, f"Found {len(regions)} candidate regions; building bounded image thumbnails...")
    rectangle_width_nm = 0.0
    rectangle_height_nm = 0.0
    rectangle_corners: list[np.ndarray] = []
    rectangle_angles: list[float] = []
    rectangle_confidences: list[float] = []
    rectangle_matched_sites: list[int] = []
    rectangle_fit_rms: list[float] = []
    if rows is not None and columns is not None and spacing_x_nm is not None and spacing_y_nm is not None:
        if rectangle_margin_nm < 0:
            raise ValueError("Rectangle margin cannot be negative.")
        rectangle_width_nm = max(spacing_x_nm, (columns - 1) * spacing_x_nm) + 2.0 * rectangle_margin_nm
        rectangle_height_nm = max(spacing_y_nm, (rows - 1) * spacing_y_nm) + 2.0 * rectangle_margin_nm
        grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
        aligned_regions, _centers, fitted_corners, fitted_angles, correlations, effective_alignment_pixel, reference = (
            _align_regions_by_image_correlation(
                regions,
                rectangle_width_nm=rectangle_width_nm,
                rectangle_height_nm=rectangle_height_nm,
                requested_pixel_nm=alignment_pixel_nm,
                iterations=alignment_iterations,
                template_points_nm=grid,
                progress_callback=progress_callback,
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

    point_counts = np.asarray([len(region) for region in aligned_regions], dtype=int)
    confidence_array = np.asarray(rectangle_confidences, dtype=float)
    accepted_mask = (
        (point_counts >= min_candidate_points)
        & (point_counts <= max_candidate_points)
        & (confidence_array >= min_rectangle_confidence)
    )
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
        bounds_nm=bounds,
        rectangle_corners_nm=np.asarray(rectangle_corners, dtype=float) if rectangle_corners else np.empty((0, 4, 2)),
        rectangle_angles_deg=np.asarray(rectangle_angles, dtype=float),
        rectangle_confidence=confidence_array,
        rectangle_matched_site_count=np.asarray(rectangle_matched_sites, dtype=int),
        rectangle_fit_rms_nm=np.asarray(rectangle_fit_rms, dtype=float),
        rectangle_width_nm=float(rectangle_width_nm),
        rectangle_height_nm=float(rectangle_height_nm),
        density_image=density,
        density_contrast=contrast,
        density_extent_nm=extent,
        density_threshold=float(density_threshold),
        alignment_pixel_nm=float(effective_alignment_pixel),
        alignment_reference_image=reference,
    )
    if progress_callback:
        progress_callback(100.0, f"Identification complete: {result.accepted_count}/{len(result.regions)} image-matched origamis accepted.")
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
    progress_callback: Callable[[str], None] | None = None,
) -> OrigamiAnalysisResult:
    if not picked_regions:
        raise ValueError("No identified origamis are available. Adjust the identification settings and run Identify Origami again.")
    if site_radius_nm <= 0:
        raise ValueError("Site radius must be greater than zero.")
    if g5m_sigma_min_nm <= 0 or g5m_sigma_max_nm < g5m_sigma_min_nm:
        raise ValueError("G5M sigma bounds must be positive and ordered minimum to maximum.")
    if g5m_min_locs < 1 or g5m_max_rounds_without_best_bic < 1:
        raise ValueError("G5M minimum localizations and BIC patience must be at least 1.")

    grid = ideal_grid_points(rows, columns, spacing_x_nm, spacing_y_nm)
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
        else:
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
        raw_cluster_counts, cluster_labels, cluster_centers, cluster_sites = cluster_aligned_origami_sites(
            aligned,
            grid,
            g5m_sigma_min_nm=g5m_sigma_min_nm,
            g5m_sigma_max_nm=g5m_sigma_max_nm,
            g5m_min_locs=g5m_min_locs,
            g5m_max_rounds_without_best_bic=g5m_max_rounds_without_best_bic,
            site_match_radius_nm=site_radius_nm,
        )
        # A rectangular grid is invariant under 180-degree rotation. Give the
        # selected pose and its 180-degree counterpart equal statistical weight
        # so brightness or missing sites cannot impose an arbitrary direction.
        rotated_cluster_counts = raw_cluster_counts[::-1]
        cluster_counts = 0.5 * (raw_cluster_counts.astype(float) + rotated_cluster_counts.astype(float))
        cluster_occupancy = 0.5 * (
            (raw_cluster_counts > 0).astype(float) + (rotated_cluster_counts > 0).astype(float)
        )
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
            progress_callback(f"Origami analysis: Picasso G5M docking-site clustering {index}/{len(candidates)} candidates...")

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
        rejected_candidate_count=rejected,
        symmetrized_180=True,
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
