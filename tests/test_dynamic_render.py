import unittest
from unittest import mock

import numpy as np

from paint_analysis_gui import (
    DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS,
    DEFAULT_ORIGAMI_ALIGNMENT_PASSES,
    DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM,
    DEFAULT_ORIGAMI_MIN_POINTS,
    PaintAnalysisApp,
    optimal_dynamic_render_pixel_nm,
    origami_candidate_failure_reasons,
    origami_site_decision_label,
    responsive_column_count,
    theoretical_grid_in_footprint,
)


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
    def test_origami_acceptance_defaults_match_validated_examples(self) -> None:
        self.assertEqual(DEFAULT_ORIGAMI_MIN_POINTS, 100)
        self.assertEqual(DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM, 6.0)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS, 128)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_PASSES, 3)

    def test_site_decision_labels_explain_each_gate(self) -> None:
        self.assertEqual(
            origami_site_decision_label(6, 0.42, 3, 0.25),
            (True, "#84cc16", "6 loc; p=0.42 ✓"),
        )
        count_failure = origami_site_decision_label(2, 0.42, 3, 0.25)
        prominence_failure = origami_site_decision_label(6, 0.12, 3, 0.25)
        both_failure = origami_site_decision_label(1, 0.08, 3, 0.25)
        self.assertEqual(count_failure[2], "loc 2<3")
        self.assertEqual(prominence_failure[2], "p 0.12<0.25")
        self.assertEqual(both_failure[2], "loc 1<3; p 0.08<0.25")

    def test_rejected_origami_label_lists_each_failed_gate(self) -> None:
        reasons = origami_candidate_failure_reasons(
            point_count=87,
            correlation=0.1,
            supported_sites=4,
            supported_rows=1,
            supported_columns=2,
            spacing_error_nm=7.2,
            params={
                "min_candidate_points": 100,
                "max_candidate_points": 1000,
                "use_correlation_gate": False,
                "min_rectangle_confidence": 0.5,
                "min_supported_sites": 5,
                "min_supported_rows": 2,
                "min_supported_columns": 2,
                "max_site_spacing_error_nm": 6.0,
            },
        )

        self.assertEqual(
            reasons,
            [
                "points 87 < 100",
                "sites 4 < 5",
                "rows 1 < 2",
                "spacing 7.2 > 6 nm",
            ],
        )

    def test_every_origami_view_has_an_explicit_navigation_mode(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        expected_modes = {
            "Loaded source data": "none",
            "Coarse identification density": "roi",
            "Random ROI inspection": "roi",
            "Identified origami template matches": "roi",
            "Identified origami match ROI": "roi",
            "Individual origami gallery": "gallery",
            "Individual site assignments": "gallery",
            "Selected origami detail": "detail",
            "Aligned density": "none",
            "Integrated density per site": "none",
            "Mean site counts": "none",
            "Site occupancy": "none",
            "Occupied-site completeness": "none",
        }

        for view, expected_mode in expected_modes.items():
            app.origami_last_rendered_plot_option = view
            self.assertEqual(app._origami_navigation_mode(), expected_mode, view)

    def test_shared_origami_arrows_dispatch_to_gallery_pages(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Individual origami gallery"
        app._change_origami_gallery_page = mock.Mock()
        app._change_origami_match_roi = mock.Mock()

        app._change_origami_navigation(1)

        app._change_origami_gallery_page.assert_called_once_with(1)
        app._change_origami_match_roi.assert_not_called()

    def test_shared_origami_arrows_dispatch_to_roi_windows(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Identified origami template matches"
        app._change_origami_gallery_page = mock.Mock()
        app._change_origami_roi_window = mock.Mock()

        app._change_origami_navigation(-1)

        app._change_origami_roi_window.assert_called_once_with(-1)
        app._change_origami_gallery_page.assert_not_called()

    def test_shared_origami_arrows_dispatch_to_selected_origami_detail(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Selected origami detail"
        app._change_origami_gallery_page = mock.Mock()
        app._change_origami_match_roi = mock.Mock()
        app._change_origami_detail = mock.Mock()

        app._change_origami_navigation(1)

        app._change_origami_detail.assert_called_once_with(1)
        app._change_origami_gallery_page.assert_not_called()
        app._change_origami_match_roi.assert_not_called()

    def test_theoretical_grid_is_transformed_into_rotated_fitted_footprint(self) -> None:
        grid = np.asarray([[-10.0, -5.0], [10.0, 5.0]])
        corners = np.asarray([[110.0, 180.0], [110.0, 220.0], [90.0, 220.0], [90.0, 180.0]])

        transformed = theoretical_grid_in_footprint(grid, corners)

        np.testing.assert_allclose(transformed, [[105.0, 190.0], [95.0, 210.0]])

    def test_responsive_controls_reflow_without_three_column_or_clipped_layouts(self) -> None:
        self.assertEqual(responsive_column_count(1000, 235, 4), 4)
        self.assertEqual(responsive_column_count(700, 235, 4), 2)
        self.assertEqual(responsive_column_count(300, 235, 4), 1)

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

    def test_loaded_origami_source_schedules_viewport_rerendering(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Loaded source data"
        app.loaded = object()
        app.origami_zoom_render_request_id = 4
        app.origami_zoom_render_after_id = None
        app.after = mock.Mock(return_value="scheduled-render")

        app._schedule_origami_zoom_render()

        self.assertEqual(app.origami_zoom_render_request_id, 5)
        self.assertEqual(app.origami_zoom_render_after_id, "scheduled-render")
        app.after.assert_called_once()


if __name__ == "__main__":
    unittest.main()
