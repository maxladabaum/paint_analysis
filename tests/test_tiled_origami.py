import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from paint_analysis_gui import (
    PaintAnalysisApp,
    evenly_distributed_tile_indices,
    fully_fitting_roi_tiles,
    randomized_tile_order,
)


class TiledOrigamiTests(unittest.TestCase):
    def test_tiles_are_anchored_to_validated_roi_and_only_include_full_tiles(self) -> None:
        validation_roi = (20.0, 40.0, 30.0, 50.0)

        tiles = fully_fitting_roi_tiles(100.0, 100.0, validation_roi)

        self.assertEqual(len(tiles), 20)
        self.assertIn(validation_roi, tiles)
        self.assertTrue(all(0 <= x0 < x1 <= 100 for x0, x1, _y0, _y1 in tiles))
        self.assertTrue(all(0 <= y0 < y1 <= 100 for _x0, _x1, y0, y1 in tiles))
        self.assertTrue(all(x1 - x0 == 20 for x0, x1, _y0, _y1 in tiles))
        self.assertTrue(all(y1 - y0 == 20 for _x0, _x1, y0, y1 in tiles))

    def test_100k_image_with_10k_roi_produces_100_tiles(self) -> None:
        tiles = fully_fitting_roi_tiles(
            100_000.0,
            100_000.0,
            (40_000.0, 50_000.0, 30_000.0, 40_000.0),
        )

        self.assertEqual(len(tiles), 100)

    def test_rejects_zero_area_validation_roi(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive width and height"):
            fully_fitting_roi_tiles(100.0, 100.0, (10.0, 10.0, 20.0, 40.0))

    def test_distributed_subset_has_requested_unique_deterministic_tiles(self) -> None:
        tiles = fully_fitting_roi_tiles(100.0, 100.0, (0.0, 10.0, 0.0, 10.0))

        first = evenly_distributed_tile_indices(tiles, 10)
        second = evenly_distributed_tile_indices(tiles, 10)

        self.assertEqual(len(first), 10)
        self.assertEqual(len(np.unique(first)), 10)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first >= 0) & (first < len(tiles))))

    def test_distributed_subset_spans_the_tile_field(self) -> None:
        tiles = fully_fitting_roi_tiles(100.0, 100.0, (0.0, 10.0, 0.0, 10.0))

        selected = evenly_distributed_tile_indices(tiles, 5)
        centers = np.asarray(
            [
                [
                    (tiles[index][0] + tiles[index][1]) / 2.0,
                    (tiles[index][2] + tiles[index][3]) / 2.0,
                ]
                for index in selected
            ]
        )

        self.assertGreaterEqual(np.ptp(centers[:, 0]), 80.0)
        self.assertGreaterEqual(np.ptp(centers[:, 1]), 80.0)

    def test_request_at_or_above_available_returns_every_tile(self) -> None:
        tiles = fully_fitting_roi_tiles(100.0, 100.0, (0.0, 10.0, 0.0, 10.0))

        np.testing.assert_array_equal(evenly_distributed_tile_indices(tiles, 1000), np.arange(100))

    def test_rejects_nonpositive_requested_tile_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            evenly_distributed_tile_indices([(0.0, 10.0, 0.0, 10.0)], 0)

    def test_random_tile_order_excludes_validation_and_previously_inspected_tiles(self) -> None:
        order = randomized_tile_order(8, {1, 3, 6}, np.random.default_rng(12))

        self.assertEqual(set(order), {0, 2, 4, 5, 7})
        self.assertEqual(len(order), 5)

    def test_random_roi_worker_skips_empty_tile_and_reuses_validated_settings(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.loaded = SimpleNamespace(info=[{"Pixelsize": 1.0}])
        app._origami_identification_worker_progress = mock.Mock()
        source = pd.DataFrame({"x": [12.0, 13.0], "y": [2.0, 3.0]})
        tiles = [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)]
        identification_params = {
            "pick_bin_size_nm": 5.0,
            "connect_distance_nm": 20.0,
            "density_threshold": 0.2,
            "min_candidate_points": 1,
            "max_candidate_points": 10,
            "rows": 3,
            "columns": 4,
            "spacing_x_nm": 20.0,
            "spacing_y_nm": 20.0,
            "rectangle_margin_nm": 10.0,
            "min_rectangle_confidence": 0.4,
            "alignment_pixel_nm": 2.0,
            "alignment_iterations": 2,
            "alignment_template_image": np.eye(3),
        }
        source_params = {
            "source": "Corrected localizations",
            "active_filters": [],
            "render_pixel_nm": 2.0,
            "render_blur_method": "smooth",
            "render_min_blur_width": 0.0,
            "render_min_density": 0.0,
            "render_max_density": 1.0,
            "source_path": "example.hdf5",
        }
        fake_picks = SimpleNamespace(accepted_count=1, regions=[np.empty((0, 2))])

        with (
            mock.patch("paint_analysis_gui.render_picasso_map", return_value={"image": np.ones((2, 2))}),
            mock.patch("paint_analysis_gui.identify_origami_regions", return_value=fake_picks) as identify,
        ):
            kind, payload = app._random_origami_roi_worker(
                source,
                1.0,
                tiles,
                np.asarray([0, 1]),
                identification_params,
                source_params,
            )

        self.assertEqual(kind, "origami_random_roi")
        self.assertEqual(payload["tile_index"], 1)
        np.testing.assert_allclose(payload["points_nm"], [[12.0, 2.0], [13.0, 3.0]])
        self.assertEqual(identify.call_args.kwargs["density_threshold"], 0.2)
        self.assertEqual(identify.call_args.kwargs["alignment_iterations"], 2)
        np.testing.assert_array_equal(
            identify.call_args.kwargs["alignment_template_image"],
            np.eye(3),
        )


if __name__ == "__main__":
    unittest.main()
