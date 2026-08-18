from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal
from scipy.optimize import curve_fit


def suppress_lattice_periodicity(
    segments: np.ndarray,
    pitch_camera_pixels: float,
    harmonics: int = 3,
    max_directions: int = 4,
) -> np.ndarray:
    """Remove detected lattice peaks while preserving nonperiodic image detail."""
    pitch = float(pitch_camera_pixels)
    if not np.isfinite(pitch) or pitch <= 2.0:
        raise ValueError("RCC lattice pitch must be finite and greater than two camera pixels.")
    if harmonics < 1 or max_directions < 1:
        raise ValueError("Lattice harmonics and direction count must be positive.")

    images = np.asarray(segments, dtype=float)
    if images.ndim != 3:
        raise ValueError("RCC segments must be a stack of two-dimensional images.")
    _segment_count, height, width = images.shape
    window = np.outer(
        signal.windows.tukey(height, alpha=0.04),
        signal.windows.tukey(width, alpha=0.04),
    )

    centered = images - images.mean(axis=(1, 2), keepdims=True)
    detection_spectra = np.fft.fftshift(
        np.fft.fft2(centered * window, axes=(-2, -1)),
        axes=(-2, -1),
    )
    # Sum power, not complex spectra or images: power is invariant to the
    # translation-dependent Fourier phase, so drift cannot cancel a lattice
    # direction during automatic orientation detection.
    reference_spectrum = np.sum(np.abs(detection_spectra) ** 2, axis=0)
    center_y, center_x = height // 2, width // 2
    yy, xx = np.mgrid[:height, :width]
    dy = yy - center_y
    dx = xx - center_x
    # Radius is measured in cycles/pixel even for rectangular images.
    radial_frequency = np.hypot(dy / height, dx / width)
    expected_frequency = 1.0 / pitch
    frequency_half_width = max(2.0 / min(height, width), 0.15 * expected_frequency)
    annulus = np.abs(radial_frequency - expected_frequency) <= frequency_half_width
    local_maxima = reference_spectrum == ndimage.maximum_filter(reference_spectrum, size=3)
    candidates = np.argwhere(annulus & local_maxima)
    if candidates.size == 0:
        raise ValueError(
            "No Fourier lattice peak was found near the requested pitch. Check the lattice pitch and pixel size."
        )
    candidate_power = reference_spectrum[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(candidate_power)[::-1]
    candidates = candidates[order]
    candidate_power = candidate_power[order]
    minimum_peak_power = 0.03 * float(candidate_power[0])

    directions: list[tuple[float, float, float]] = []
    minimum_angle = np.deg2rad(12.0)
    for (candidate_y, candidate_x), power in zip(candidates, candidate_power):
        if power < minimum_peak_power:
            break
        offset_y = float(candidate_y - center_y)
        offset_x = float(candidate_x - center_x)
        angle = float(np.mod(np.arctan2(offset_y / height, offset_x / width), np.pi))
        if any(
            min(abs(angle - existing), np.pi - abs(angle - existing)) < minimum_angle
            for existing, _offset_y, _offset_x in directions
        ):
            continue
        directions.append((angle, offset_y, offset_x))
        if len(directions) >= max_directions:
            break
    if not directions:
        raise ValueError("The Fourier lattice orientation could not be determined.")

    notch_mask = np.ones((height, width), dtype=float)
    # Scale the notch to the separation of reciprocal-lattice peaks. This is
    # wide enough for finite-size/disordered peak broadening without allowing
    # adjacent notches to erase the nonperiodic spectrum in smaller images.
    fundamental_radius_bins = min(np.hypot(offset_y, offset_x) for _angle, offset_y, offset_x in directions)
    notch_sigma_bins = float(np.clip(0.20 * fundamental_radius_bins, 1.5, 5.0))
    notch_centers: list[tuple[float, float]] = []
    for _angle, base_dy, base_dx in directions:
        for harmonic in range(1, harmonics + 1):
            for sign in (-1.0, 1.0):
                notch_centers.append((sign * harmonic * base_dy, sign * harmonic * base_dx))

    # A 2D lattice also produces peaks at integer combinations of its two
    # reciprocal basis vectors. Remove those narrow peaks without discarding
    # the complete frequency ring containing nonperiodic defect information.
    if len(directions) >= 2:
        _angle_1, basis_1_y, basis_1_x = directions[0]
        _angle_2, basis_2_y, basis_2_x = directions[1]
        for first_multiple in range(-harmonics, harmonics + 1):
            for second_multiple in range(-harmonics, harmonics + 1):
                if first_multiple == 0 and second_multiple == 0:
                    continue
                notch_centers.append(
                    (
                        first_multiple * basis_1_y + second_multiple * basis_2_y,
                        first_multiple * basis_1_x + second_multiple * basis_2_x,
                    )
                )

    for offset_y, offset_x in notch_centers:
        target_y = center_y + offset_y
        target_x = center_x + offset_x
        if not (1.0 <= target_y < height - 1.0 and 1.0 <= target_x < width - 1.0):
            continue
        distance_squared = (yy - target_y) ** 2 + (xx - target_x) ** 2
        notch_mask *= 1.0 - np.exp(-0.5 * distance_squared / notch_sigma_bins**2)

    # Apply the learned notch to the unwindowed data. The taper is valuable
    # for locating compact spectral peaks, but applying a fixed camera-space
    # window to every segment would itself favor zero displacement.
    spectra = np.fft.fftshift(np.fft.fft2(centered, axes=(-2, -1)), axes=(-2, -1))
    filtered = np.fft.ifft2(
        np.fft.ifftshift(spectra * notch_mask[None, :, :], axes=(-2, -1)),
        axes=(-2, -1),
    ).real
    return filtered


def _edge_safe_image_shift(
    image_a: np.ndarray,
    image_b: np.ndarray,
    box: int = 5,
    roi: int | None = 32,
) -> tuple[float, float]:
    """Picasso-compatible peak fitting that fits from the full correlation image."""
    from picasso import imageprocess

    if np.all(image_a == 0) or np.all(image_b == 0):
        return 0.0, 0.0

    full_correlation = imageprocess.xcorr(image_a, image_b)
    height, width = image_a.shape
    search = full_correlation
    y_offset = x_offset = 0
    if roi is not None:
        y_margin = int((height - roi) / 2)
        x_margin = int((width - roi) / 2)
        if y_margin > 0:
            search = search[y_margin:-y_margin, :]
            y_offset = y_margin
        if x_margin > 0:
            search = search[:, x_margin:-x_margin]
            x_offset = x_margin

    peak_y_search, peak_x_search = np.unravel_index(search.argmax(), search.shape)
    peak_y = peak_y_search + y_offset
    peak_x = peak_x_search + x_offset
    fit_radius = int(box / 2)
    fit_roi = full_correlation[
        peak_y - fit_radius : peak_y + fit_radius + 1,
        peak_x - fit_radius : peak_x + fit_radius + 1,
    ]
    expected_shape = (2 * fit_radius + 1, 2 * fit_radius + 1)
    if fit_roi.shape != expected_shape:
        raise ValueError(
            "RCC correlation peak reached the full image boundary; the shift cannot be fitted safely."
        )

    grid_y, grid_x = np.mgrid[-fit_radius : fit_radius + 1, -fit_radius : fit_radius + 1]

    def flat_gaussian(
        coords: tuple[np.ndarray, np.ndarray],
        amplitude: float,
        center_x: float,
        center_y: float,
        sigma: float,
        background: float,
    ) -> np.ndarray:
        x_coord, y_coord = coords
        values = amplitude * np.exp(
            -0.5 * ((x_coord - center_x) ** 2 + (y_coord - center_y) ** 2) / sigma**2
        ) + background
        return values.ravel()

    def parabolic_offset(left: float, center: float, right: float) -> float:
        """Estimate a bounded subpixel peak offset from three samples."""
        denominator = left - 2.0 * center + right
        scale = max(abs(left), abs(center), abs(right), 1.0)
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps * scale:
            return 0.0
        offset = 0.5 * (left - right) / denominator
        if not np.isfinite(offset):
            return 0.0
        return float(np.clip(offset, -1.0, 1.0))

    background = float(fit_roi.min())
    initial = [float(fit_roi.max()) - background, 0.0, 0.0, 1.0, background]
    try:
        fitted, _covariance = curve_fit(
            flat_gaussian,
            (grid_x, grid_y),
            fit_roi.ravel(),
            p0=initial,
            bounds=(
                [0.0, -fit_radius, -fit_radius, np.finfo(float).eps, -np.inf],
                [np.inf, fit_radius, fit_radius, 2.0 * box, np.inf],
            ),
            max_nfev=5000,
        )
        subpixel_x = float(fitted[1])
        subpixel_y = float(fitted[2])
    except (RuntimeError, ValueError, FloatingPointError):
        # A low-pass-filtered array envelope is often broad or asymmetric and
        # therefore not well described by a 2D Gaussian. A local parabola is
        # less precise but always gives a stable, bounded subpixel estimate.
        center = fit_radius
        subpixel_x = parabolic_offset(
            float(fit_roi[center, center - 1]),
            float(fit_roi[center, center]),
            float(fit_roi[center, center + 1]),
        )
        subpixel_y = parabolic_offset(
            float(fit_roi[center - 1, center]),
            float(fit_roi[center, center]),
            float(fit_roi[center + 1, center]),
        )

    shift_x = subpixel_x + peak_x - np.floor(width / 2)
    shift_y = subpixel_y + peak_y - np.floor(height / 2)
    return -shift_y, -shift_x


def _edge_safe_rcc(
    segments: np.ndarray,
    max_shift: int = 32,
    callback: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from picasso import lib

    segment_count = len(segments)
    shifts_x = np.zeros((segment_count, segment_count))
    shifts_y = np.zeros((segment_count, segment_count))
    completed = 0
    if callback is not None:
        callback(0)
    for first in range(segment_count - 1):
        for second in range(first + 1, segment_count):
            shifts_y[first, second], shifts_x[first, second] = _edge_safe_image_shift(
                segments[first], segments[second], box=5, roi=max_shift
            )
            completed += 1
            if callback is not None:
                callback(completed)
    return lib.minimize_shifts(shifts_x, shifts_y)


def undrift_rcc_with_lattice_suppression(
    locs: pd.DataFrame,
    info: list[dict[str, Any]],
    segmentation: int,
    lattice_pitch_nm: float,
    segmentation_callback: Callable[[int], None] | None = None,
    rcc_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Picasso RCC after suppressing spatial detail at the lattice scale."""
    from picasso import postprocess

    pixel_size_nm = float(info[0]["Pixelsize"])
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
        raise ValueError("Pixel size must be positive for RCC lattice suppression.")

    bounds, segments = postprocess.segment(
        locs,
        info,
        int(segmentation),
        {"blur_method": "gaussian", "min_blur_width": 1},
        segmentation_callback,
    )
    filtered_segments = suppress_lattice_periodicity(segments, float(lattice_pitch_nm) / pixel_size_nm)
    shift_y, shift_x = _edge_safe_rcc(filtered_segments, 32, rcc_callback)

    sample_times = (bounds[1:] + bounds[:-1]) / 2
    drift_x_spline = interpolate.InterpolatedUnivariateSpline(sample_times, shift_x, k=3)
    drift_y_spline = interpolate.InterpolatedUnivariateSpline(sample_times, shift_y, k=3)
    frame_times = np.arange(int(info[0]["Frames"]))
    drift = pd.DataFrame(
        {
            "x": drift_x_spline(frame_times),
            "y": drift_y_spline(frame_times),
        }
    )
    corrected_locs = postprocess.apply_drift(locs, info, drift=drift)
    return drift, corrected_locs
