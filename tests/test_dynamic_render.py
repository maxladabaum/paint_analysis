import unittest

from paint_analysis_gui import PaintAnalysisApp, optimal_dynamic_render_pixel_nm


class FakeVariable:
    def __init__(self, value: float) -> None:
        self.value = value

    def get(self) -> float:
        return self.value

    def set(self, value: float) -> None:
        self.value = value


class FakeBounds:
    width = 1000.0
    height = 500.0


class FakeAxis:
    def get_window_extent(self) -> FakeBounds:
        return FakeBounds()


class DynamicRenderTests(unittest.TestCase):
    def test_matches_render_resolution_to_displayed_viewport(self) -> None:
        pixel_nm = optimal_dynamic_render_pixel_nm(
            (0.0, 100_000.0, 0.0, 100_000.0),
            display_width_px=1000,
            display_height_px=500,
        )
        self.assertEqual(pixel_nm, 200.0)

    def test_never_renders_finer_than_one_nm(self) -> None:
        pixel_nm = optimal_dynamic_render_pixel_nm(
            (100.0, 600.0, 200.0, 600.0),
            display_width_px=1000,
            display_height_px=800,
        )
        self.assertEqual(pixel_nm, 1.0)

    def test_rejects_empty_viewport(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive width and height"):
            optimal_dynamic_render_pixel_nm((1.0, 1.0, 0.0, 10.0), 100, 100)

    def test_dynamic_resolution_updates_render_pixel_control(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.render_disp_px_nm = FakeVariable(10.0)

        pixel_nm = app._dynamic_render_pixel_nm(
            FakeAxis(),
            (0.0, 100_000.0, 0.0, 100_000.0),
        )

        self.assertEqual(pixel_nm, 200.0)
        self.assertEqual(app.render_disp_px_nm.get(), 200.0)

    def test_explicit_render_request_preserves_manual_pixel_value(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.render_disp_px_nm = FakeVariable(7.5)
        app.dynamic_zoom_render = FakeVariable(True)

        pixel_nm = app._render_pixel_nm_for_request(
            FakeAxis(),
            (0.0, 100_000.0, 0.0, 100_000.0),
            allow_dynamic=False,
        )

        self.assertEqual(pixel_nm, 7.5)
        self.assertEqual(app.render_disp_px_nm.get(), 7.5)

    def test_automatic_render_request_still_updates_pixel_value(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.render_disp_px_nm = FakeVariable(7.5)
        app.dynamic_zoom_render = FakeVariable(True)
        app._full_map_viewport_nm = lambda: None

        pixel_nm = app._render_pixel_nm_for_request(
            FakeAxis(),
            (0.0, 100_000.0, 0.0, 100_000.0),
            allow_dynamic=True,
        )

        self.assertEqual(pixel_nm, 200.0)
        self.assertEqual(app.render_disp_px_nm.get(), 200.0)


if __name__ == "__main__":
    unittest.main()
