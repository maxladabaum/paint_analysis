import unittest

from paint_analysis_gui import fully_fitting_roi_tiles


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


if __name__ == "__main__":
    unittest.main()
