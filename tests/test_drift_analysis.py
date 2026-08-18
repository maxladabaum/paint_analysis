import unittest
from unittest import mock

import numpy as np

from drift_analysis import _edge_safe_image_shift, suppress_lattice_periodicity


class DriftAnalysisTests(unittest.TestCase):
    def test_lattice_suppression_reduces_periodic_contrast_without_mixing_segments(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        lattice = np.cos(2.0 * np.pi * xx / 8.0) + np.cos(2.0 * np.pi * yy / 8.0)
        defect = 2.0 * np.exp(-0.5 * ((xx - 21.0) ** 2 + (yy - 39.0) ** 2) / 3.0**2)
        segments = np.stack([lattice + defect, 3.0 * (lattice + defect)])

        filtered = suppress_lattice_periodicity(segments, pitch_camera_pixels=8.0)

        self.assertEqual(filtered.shape, segments.shape)
        np.testing.assert_allclose(filtered[1], 3.0 * filtered[0], atol=1e-12)

        before_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(segments[0])))
        after_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(filtered[0])))
        center = 32
        self.assertLess(after_spectrum[center, center + 8], 0.1 * before_spectrum[center, center + 8])
        self.assertGreater(float(np.max(filtered[0])), 0.1)

    def test_fourier_suppression_recovers_shift_from_defect_fingerprint(self) -> None:
        size = 128
        yy, xx = np.mgrid[:size, :size]
        lattice = 2.0 * np.cos(2.0 * np.pi * xx / 8.0) + 1.6 * np.cos(2.0 * np.pi * yy / 8.0)
        defects = (
            3.0 * np.exp(-0.5 * ((xx - 31.0) ** 2 + (yy - 44.0) ** 2) / 3.0**2)
            + 2.0 * np.exp(-0.5 * ((xx - 91.0) ** 2 + (yy - 72.0) ** 2) / 5.0**2)
            - 1.5 * np.exp(-0.5 * ((xx - 68.0) ** 2 + (yy - 99.0) ** 2) / 4.0**2)
        )
        image = lattice + defects
        shifted = np.roll(image, shift=(3, -4), axis=(0, 1))

        filtered = suppress_lattice_periodicity(np.stack([image, shifted]), pitch_camera_pixels=8.0)
        shift_y, shift_x = _edge_safe_image_shift(filtered[0], filtered[1], box=5, roi=32)

        self.assertAlmostEqual(abs(shift_y), 3.0, places=1)
        self.assertAlmostEqual(abs(shift_x), 4.0, places=1)

    def test_invalid_lattice_pitch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than two"):
            suppress_lattice_periodicity(np.zeros((2, 3, 4)), -1.0)

    def test_edge_safe_peak_fit_handles_search_window_boundary(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        image = np.exp(-0.5 * ((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 2.0**2)
        shifted = np.roll(image, shift=(0, 14), axis=(0, 1))

        shift_y, shift_x = _edge_safe_image_shift(image, shifted, box=5, roi=32)

        self.assertAlmostEqual(abs(shift_y), 0.0, places=1)
        self.assertAlmostEqual(abs(shift_x), 14.0, places=1)

    def test_peak_fit_falls_back_when_gaussian_optimizer_does_not_converge(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        image = np.exp(-0.5 * ((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 2.0**2)
        shifted = np.roll(image, shift=(3, -4), axis=(0, 1))

        with mock.patch("drift_analysis.curve_fit", side_effect=RuntimeError("did not converge")):
            shift_y, shift_x = _edge_safe_image_shift(image, shifted, box=5, roi=32)

        self.assertAlmostEqual(abs(shift_y), 3.0, places=1)
        self.assertAlmostEqual(abs(shift_x), 4.0, places=1)


if __name__ == "__main__":
    unittest.main()
