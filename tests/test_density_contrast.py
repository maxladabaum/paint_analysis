import unittest

import numpy as np

from paint_analysis_gui import PaintAnalysisApp, histogram_density_limits, scale_density_like_picasso


class FakeVariable:
    def __init__(self, value: float | bool) -> None:
        self.value = value

    def get(self) -> float | bool:
        return self.value

    def set(self, value: float | bool) -> None:
        self.value = value


class DensityContrastTests(unittest.TestCase):
    def test_automatic_limits_keep_histogram_population_below_saturation(self) -> None:
        populated = np.concatenate((np.linspace(1.0, 10.0, 10_000), np.full(5, 1_000.0)))
        image = np.concatenate((np.zeros(1_000), populated, [np.nan]))

        _scaled, limits = scale_density_like_picasso(image, 0.0, 0.0)

        saturated_fraction = np.count_nonzero(populated >= limits[1]) / populated.size
        self.assertLessEqual(saturated_fraction, 0.001)
        self.assertGreater(limits[1], 10.0)
        self.assertLess(limits[1], 1_000.0)

    def test_manual_density_limits_are_preserved(self) -> None:
        image = np.asarray([[0.0, 1.0], [10.0, 100.0]])

        _scaled, limits = scale_density_like_picasso(image, 5.0, 20.0)

        self.assertEqual(limits, (5.0, 20.0))

    def test_constant_populated_density_remains_visible(self) -> None:
        image = np.asarray([[0.0, 5.0], [5.0, 0.0]])

        scaled, limits = scale_density_like_picasso(image, 0.0, 0.0)

        self.assertEqual(limits, (0.0, 5.0))
        np.testing.assert_allclose(scaled, [[0.0, 1.0], [1.0, 0.0]])

    def test_empty_density_has_safe_limits(self) -> None:
        self.assertEqual(histogram_density_limits(np.zeros((2, 3))), (0.0, 1.0))

    def test_automatic_render_updates_displayed_density_values(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.auto_density_contrast = FakeVariable(True)
        app.auto_density_multiplier = FakeVariable(2.0)
        app.render_min_density = FakeVariable(999.0)
        app.render_max_density = FakeVariable(1000.0)
        image = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 100.0]])

        _scaled, limits = app._scale_map_density(image)

        expected_max = histogram_density_limits(image)[1] * 2.0
        self.assertEqual(limits, (0.0, expected_max))
        self.assertEqual(app.render_min_density.get(), 0.0)
        self.assertEqual(app.render_max_density.get(), expected_max)


if __name__ == "__main__":
    unittest.main()
