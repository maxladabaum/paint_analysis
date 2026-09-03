import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tifffile
from PIL import Image, PngImagePlugin
from matplotlib.figure import Figure

from paint_analysis_gui import (
    DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS,
    DEFAULT_ORIGAMI_ALIGNMENT_PASSES,
    DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM,
    DEFAULT_ORIGAMI_CORRELATION_THRESHOLD,
    DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM,
    DEFAULT_ORIGAMI_MIN_POINTS,
    DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE,
    DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY,
    DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS,
    DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY,
    DEFAULT_ORIGAMI_USE_CORRELATION_GATE,
    FILTERED_MAP_TAB,
    ORIGAMI_TAB,
    OrigamiToolbar,
    PaintAnalysisApp,
    PicassoAimStatusProgress,
    TemporalVLineAnnotation,
    custom_template_contours_nm,
    load_custom_template_image,
    load_custom_template_metadata,
    optimal_dynamic_render_pixel_nm,
    origami_candidate_failure_reasons,
    origami_site_decision_label,
    parse_temporal_vline_annotation,
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
    def test_aim_progress_accounts_for_all_3d_passes_and_never_exceeds_100_percent(self) -> None:
        updates: list[str] = []
        progress = PicassoAimStatusProgress(updates.append, phase_count=4)

        for phase in range(4):
            if phase:
                progress.zero_progress(f"AIM pass {phase + 1}/4")
            list(progress.get_iterator(0, 2))
            progress.set_value(1)

        percentages = [float(message.split(":", 1)[1].split("%", 1)[0]) for message in updates]
        self.assertTrue(percentages)
        self.assertTrue(all(0.0 <= value <= 100.0 for value in percentages))
        self.assertEqual(percentages[-1], 100.0)

    def test_aim_progress_remains_bounded_if_picasso_resets_an_extra_time(self) -> None:
        updates: list[str] = []
        progress = PicassoAimStatusProgress(updates.append, phase_count=2)
        progress.zero_progress("AIM pass 2/2")
        progress.zero_progress("unexpected extra reset")
        list(progress.get_iterator(0, 1))

        self.assertIn("100.0% overall", updates[-1])

    def test_temporal_vline_annotation_parses_frame_and_custom_text(self) -> None:
        self.assertEqual(
            parse_temporal_vline_annotation("2000", "power 20% exposure 100ms"),
            TemporalVLineAnnotation(2000, "power 20% exposure 100ms"),
        )
        self.assertEqual(
            parse_temporal_vline_annotation("2000.0", "  laser changed  "),
            TemporalVLineAnnotation(2000, "laser changed"),
        )
        for frame in ("", "-1", "2.5", "not a frame"):
            with self.subTest(frame=frame):
                with self.assertRaisesRegex(ValueError, "non-negative whole number"):
                    parse_temporal_vline_annotation(frame, "event")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_temporal_vline_annotation("2000", "   ")

    def test_temporal_vlines_are_drawn_on_every_temporal_axis(self) -> None:
        figure = Figure()
        first_axis = figure.add_subplot(211)
        second_axis = figure.add_subplot(212)
        app = SimpleNamespace(
            temporal_figure=figure,
            temporal_annotations=[
                TemporalVLineAnnotation(2000, "power 20% exposure 100ms"),
                TemporalVLineAnnotation(4000, "power 30%"),
            ],
            temporal_annotation_artists=[],
        )
        PaintAnalysisApp._draw_temporal_annotations(app)
        self.assertEqual(len(app.temporal_annotation_artists), 8)
        for axis in (first_axis, second_axis):
            self.assertEqual(len(axis.lines), 2)
            self.assertEqual([text.get_text() for text in axis.texts], [
                "power 20% exposure 100ms",
                "power 30%",
            ])
            self.assertEqual(float(axis.lines[0].get_xdata()[0]), 2000.0)

    def test_custom_template_loader_converts_to_grayscale_and_physical_y_up(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[0, 1] = (255, 255, 255)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "barcode.tif"
            tifffile.imwrite(path, image)
            loaded = load_custom_template_image(path)

        self.assertEqual(loaded.shape, (4, 5))
        self.assertGreater(float(loaded[-1, 1]), 0.0)
        self.assertEqual(float(loaded[0, 1]), 0.0)

    def test_custom_template_loader_reads_embedded_picklist_calibration(self) -> None:
        metadata = {
            "format": "paint-analysis-origami-template-v1",
            "rows": 8,
            "columns": 12,
            "spacing_x_nm": 120.0 / 11.0,
            "spacing_y_nm": 40.0 / 7.0,
            "margin_nm": 20.0,
            "width_nm": 160.0,
            "height_nm": 80.0,
            "width_px": 500,
            "height_px": 250,
            "pixel_size_x_nm": 160.0 / 499.0,
            "pixel_size_y_nm": 80.0 / 249.0,
        }
        png_metadata = PngImagePlugin.PngInfo()
        png_metadata.add_text("paint_analysis_template", json.dumps(metadata))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "barcode.png"
            Image.fromarray(np.zeros((250, 500), dtype=np.uint8)).save(path, pnginfo=png_metadata)
            loaded = load_custom_template_metadata(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual((loaded["rows"], loaded["columns"]), (8, 12))
        self.assertAlmostEqual(loaded["pixel_size_x_nm"], 160.0 / 499.0)
        self.assertAlmostEqual(loaded["pixel_size_y_nm"], 80.0 / 249.0)

    def test_importing_generated_template_restores_physical_geometry(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._file_dialog_initial_dir = mock.Mock(return_value=Path("/tmp"))
        app._remember_file_dialog_dir = mock.Mock()
        app.origami_rows = FakeVariable(3)
        app.origami_columns = FakeVariable(4)
        app.origami_spacing_x_nm = FakeVariable(20.0)
        app.origami_spacing_y_nm = FakeVariable(20.0)
        app.origami_rectangle_margin_nm = FakeVariable(20.0)
        app.origami_template_pixel_x_nm = FakeVariable(1.0)
        app.origami_template_pixel_y_nm = FakeVariable(1.0)
        app.origami_custom_template_name = FakeVariable("")
        app.origami_template_mode = FakeVariable("Simulated grid")
        app.origami_custom_templates = []
        app.origami_multi_template_results = {}
        app.origami_multi_template_counts = {}
        app.origami_template_result_view = FakeVariable("All templates")
        app.origami_template_result_combo = mock.Mock()
        app.status = FakeVariable("")
        metadata = {
            "width_px": 500,
            "height_px": 250,
            "rows": 8,
            "columns": 12,
            "spacing_x_nm": 120.0 / 11.0,
            "spacing_y_nm": 40.0 / 7.0,
            "margin_nm": 20.0,
            "pixel_size_x_nm": 160.0 / 499.0,
            "pixel_size_y_nm": 80.0 / 249.0,
        }
        with (
            mock.patch("paint_analysis_gui.filedialog.askopenfilename", return_value="/tmp/template.png"),
            mock.patch("paint_analysis_gui.load_custom_template_image", return_value=np.zeros((250, 500))),
            mock.patch("paint_analysis_gui.load_custom_template_metadata", return_value=metadata),
        ):
            PaintAnalysisApp._load_origami_custom_template(app)

        self.assertEqual((app.origami_rows.get(), app.origami_columns.get()), (8, 12))
        self.assertAlmostEqual(app.origami_spacing_x_nm.get(), 120.0 / 11.0)
        self.assertAlmostEqual(app.origami_spacing_y_nm.get(), 40.0 / 7.0)
        self.assertEqual(app.origami_rectangle_margin_nm.get(), 20.0)
        self.assertAlmostEqual(app.origami_template_pixel_x_nm.get(), 160.0 / 499.0)
        self.assertAlmostEqual(app.origami_template_pixel_y_nm.get(), 80.0 / 249.0)
        self.assertEqual(app.origami_template_mode.get(), "Custom image")

    def test_loading_multiple_templates_requires_and_retains_each_calibration(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._file_dialog_initial_dir = mock.Mock(return_value=Path("/tmp"))
        app._remember_file_dialog_dir = mock.Mock()
        app.origami_rows = FakeVariable(3)
        app.origami_columns = FakeVariable(4)
        app.origami_spacing_x_nm = FakeVariable(20.0)
        app.origami_spacing_y_nm = FakeVariable(20.0)
        app.origami_rectangle_margin_nm = FakeVariable(20.0)
        app.origami_template_pixel_x_nm = FakeVariable(1.0)
        app.origami_template_pixel_y_nm = FakeVariable(1.0)
        app.origami_custom_template_name = FakeVariable("")
        app.origami_template_mode = FakeVariable("Simulated grid")
        app.origami_template_result_view = FakeVariable("All templates")
        app.origami_template_result_combo = mock.Mock()
        app.status = FakeVariable("")
        metadata = {
            "format": "paint-analysis-origami-template-v1",
            "width_px": 100,
            "height_px": 50,
            "rows": 8,
            "columns": 12,
            "spacing_x_nm": 120.0 / 11.0,
            "spacing_y_nm": 40.0 / 7.0,
            "margin_nm": 20.0,
            "pixel_size_x_nm": 160.0 / 99.0,
            "pixel_size_y_nm": 80.0 / 49.0,
        }
        with (
            mock.patch(
                "paint_analysis_gui.filedialog.askopenfilenames",
                return_value=("/tmp/type_a.png", "/tmp/type_b.png"),
            ),
            mock.patch(
                "paint_analysis_gui.load_custom_template_image",
                side_effect=(np.zeros((50, 100)), np.ones((50, 100))),
            ),
            mock.patch("paint_analysis_gui.load_custom_template_metadata", return_value=metadata),
        ):
            PaintAnalysisApp._load_multiple_origami_custom_templates(app)

        self.assertEqual([item["name"] for item in app.origami_custom_templates], ["type_a", "type_b"])
        self.assertEqual(app.origami_template_mode.get(), "Custom image")
        self.assertAlmostEqual(app.origami_spacing_x_nm.get(), 120.0 / 11.0)
        app.origami_template_result_combo.configure.assert_called_once_with(
            values=("All templates", "type_a", "type_b")
        )

    def test_template_count_plot_includes_each_type_and_unclassified(self) -> None:
        figure = Figure()
        app = SimpleNamespace(
            origami_multi_template_results={"type_a": {}, "type_b": {}},
            origami_multi_template_counts={"type_a": 12, "type_b": 7},
            origami_multi_template_unclassified_count=3,
            origami_figure=figure,
            origami_canvas=SimpleNamespace(draw_idle=mock.Mock()),
            origami_toolbar=SimpleNamespace(update=mock.Mock()),
            _configure_origami_navigation_controls=mock.Mock(),
            notebook=SimpleNamespace(select=mock.Mock()),
            status=FakeVariable(""),
        )

        PaintAnalysisApp._plot_origami_type_counts(app)

        self.assertEqual([patch.get_height() for patch in figure.axes[0].patches], [12, 7, 3])
        self.assertIn("type_a=12", app.status.get())
        self.assertIn("unclassified=3", app.status.get())

    def test_selecting_a_classified_type_restores_its_fits_and_cached_overlay(self) -> None:
        picks = object()
        result = object()
        app = SimpleNamespace(
            origami_template_result_view=FakeVariable("type_b"),
            origami_multi_template_results={
                "type_b": {"picks": picks, "params": {"rows": 8, "columns": 12}}
            },
            origami_multi_template_overlay_results={
                "type_b": {
                    "result": result,
                    "source": "source — type_b",
                    "source_count": 123,
                    "render_settings": {"rows": 8},
                    "occupancy_threshold": 1,
                }
            },
            origami_plot_option=FakeVariable("Origami type counts"),
            origami_pick_result=None,
            origami_identification_params=None,
            origami_result=None,
            _plot_identified_origamis=mock.Mock(),
            _refresh_origami_action_states=mock.Mock(),
        )

        PaintAnalysisApp._on_origami_template_result_selection(app)

        self.assertIs(app.origami_pick_result, picks)
        self.assertIs(app.origami_result, result)
        self.assertEqual(app.origami_result_source, "source — type_b")
        self.assertEqual(app.origami_plot_option.get(), "Identified origami template matches")
        app._plot_identified_origamis.assert_called_once_with()

    def test_multi_template_plot_menu_includes_identification_counts_and_overlay_views(self) -> None:
        combo = mock.Mock()
        app = SimpleNamespace(
            origami_pick_result=object(),
            origami_multi_template_results={"type_a": {}, "type_b": {}},
            origami_result=object(),
            origami_all_plot_options=(
                "Coarse identification density",
                "Identified origami template matches",
                "Origami type counts",
                "Individual origami gallery",
                "Aligned density",
            ),
            origami_plot_combo=combo,
            origami_refine_button=mock.Mock(),
            origami_back_gallery_button=mock.Mock(),
        )

        PaintAnalysisApp._refresh_origami_action_states(app)

        combo.configure.assert_called_once_with(
            values=(
                "Coarse identification density",
                "Identified origami template matches",
                "Origami type counts",
                "Individual origami gallery",
                "Aligned density",
            )
        )
        combo.state.assert_called_once_with(["!disabled", "readonly"])

    def test_origami_header_keeps_classification_selector_next_to_plot_controls(self) -> None:
        widgets = {
            name: mock.Mock()
            for name in (
                "origami_sidebar_toggle_button",
                "origami_view_label",
                "origami_plot_combo",
                "origami_template_result_label",
                "origami_template_result_combo",
                "origami_match_label",
                "origami_match_panel_combo",
                "origami_popout_button",
                "origami_fullscreen_button",
            )
        }
        app = SimpleNamespace(**widgets)

        PaintAnalysisApp._layout_origami_plot_header(app, SimpleNamespace(width=1200))

        widgets["origami_plot_combo"].grid_configure.assert_called_once_with(
            row=0, column=2, columnspan=1, padx=(6, 8), pady=0, sticky="ew"
        )
        widgets["origami_template_result_label"].grid_configure.assert_called_once_with(
            row=0, column=3, columnspan=1, padx=(0, 4), pady=0
        )
        widgets["origami_template_result_combo"].grid_configure.assert_called_once_with(
            row=0, column=4, columnspan=1, padx=(0, 8), pady=0, sticky="ew"
        )

    def test_multi_template_identification_builds_fast_overlay_for_each_populated_type(self) -> None:
        region_a = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        region_b = np.asarray([[10.0, 10.0], [11.0, 11.0]])

        def picks(regions: list[np.ndarray]) -> SimpleNamespace:
            return SimpleNamespace(
                accepted_count=len(regions),
                accepted_aligned_regions=regions,
                regions=regions,
                accepted_mask=np.ones(len(regions), dtype=bool),
                template_points_nm=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            )

        template_params = {
            "rows": 1,
            "columns": 2,
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 10.0,
            "site_mask_radius_nm": 5.0,
            "min_site_localizations": 2,
            "min_site_evidence": 0.2,
        }
        template_results = [
            {"name": "type_a", "picks": picks([region_a]), "params": template_params},
            {"name": "type_b", "picks": picks([region_b]), "params": template_params},
            {"name": "empty", "picks": picks([]), "params": template_params},
        ]
        overlay_worker = mock.Mock(
            side_effect=(
                ("origami", {"result": "overlay_a"}),
                ("origami", {"result": "overlay_b"}),
            )
        )
        app = SimpleNamespace(
            origami_loaded_source_label="corrected ROI",
            _origami_identification_worker_progress=mock.Mock(),
            _overlay_origami_worker=overlay_worker,
        )
        params = {
            "source_path": Path("source.hdf5"),
            "fast_overlay_settings": {
                "g5m_sigma_min_nm": 0.5,
                "g5m_sigma_max_nm": 5.0,
                "g5m_min_locs": 3,
                "g5m_bic_patience": 2,
                "site_radius_nm": 5.0,
                "allow_mirror": False,
                "overlay_pixel_nm": 0.5,
                "overlay_padding_nm": 10.0,
                "overlay_blur_nm": 1.0,
            },
        }

        overlays = PaintAnalysisApp._build_multi_template_fast_overlays(
            app,
            template_results,
            np.vstack((region_a, region_b)),
            params,
        )

        self.assertEqual(overlays, {"type_a": {"result": "overlay_a"}, "type_b": {"result": "overlay_b"}})
        self.assertEqual(overlay_worker.call_count, 2)
        self.assertEqual(overlay_worker.call_args_list[0].args[3]["source_label"], "corrected ROI — type_a")
        self.assertEqual(overlay_worker.call_args_list[1].args[3]["source_label"], "corrected ROI — type_b")

    def test_custom_template_overlay_contours_follow_hollow_l_shape(self) -> None:
        template = np.zeros((9, 9), dtype=float)
        template[1:8, 1:3] = 1.0
        template[1:3, 1:8] = 1.0

        contours = custom_template_contours_nm(template, 90.0, 90.0)

        self.assertTrue(contours)
        points = np.vstack(contours)
        self.assertLess(float(np.min(points[:, 0])), -25.0)
        self.assertGreater(float(np.max(points[:, 0])), 25.0)
        self.assertLess(float(np.min(points[:, 1])), -25.0)
        self.assertGreater(float(np.max(points[:, 1])), 25.0)
        # The empty upper-right interior must remain outside the contour rather
        # than being replaced by a rectangular bounding box.
        self.assertFalse(np.any(np.all(points > np.asarray([20.0, 20.0]), axis=1)))

    def test_custom_template_overview_draws_one_theoretical_overlay(self) -> None:
        figure = Figure()
        axis = figure.add_subplot(111)
        axis.set_xlim(-100.0, 100.0)
        axis.set_ylim(-100.0, 100.0)
        grid = np.asarray([[-80.0, -37.5], [0.0, -37.5], [80.0, 37.5]])
        accepted_corners = np.asarray(
            [[-80.0, -37.5], [80.0, -37.5], [80.0, 37.5], [-80.0, 37.5]]
        )
        picks = SimpleNamespace(
            bounds_nm=np.asarray([
                [-90.0, 90.0, -45.0, 45.0],
                [-50.0, 130.0, -25.0, 65.0],
            ]),
            accepted_mask=np.asarray([True, False]),
            template_points_nm=grid,
            rectangle_corners_nm=np.asarray(
                [
                    accepted_corners,
                    accepted_corners + np.asarray([40.0, 20.0]),
                ]
            ),
        )
        app = SimpleNamespace(
            origami_footprint_artists=[],
            origami_show_text_statistics=FakeVariable(False),
            origami_show_theoretical_overlay=FakeVariable(True),
            origami_show_detected_sites_overlay=FakeVariable(False),
            origami_show_site_diagnostics=FakeVariable(False),
            origami_show_prominence_geometry=FakeVariable(False),
            origami_identification_params={
                "template_mode": "Custom image",
                "alignment_template_image": np.eye(3),
            },
            _active_origami_picks_and_params=lambda: None,
        )

        shown, total = PaintAnalysisApp._draw_visible_origami_footprints(app, axis, picks)

        self.assertEqual((shown, total), (1, 2))
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(len(axis.patches), 0)
        np.testing.assert_allclose(axis.collections[0].get_offsets(), grid)

        # Text statistics is also the explicit diagnostic opt-in for rejected
        # fits, so enabling it restores both the rejected sites and failure text.
        app.origami_show_text_statistics.set(True)
        picks.point_counts = np.asarray([100, 50])
        picks.rectangle_angles_deg = np.asarray([0.0, 0.0])
        picks.rectangle_confidence = np.asarray([0.8, 0.1])
        picks.supported_site_count = np.asarray([3, 1])
        picks.supported_row_count = np.asarray([2, 1])
        picks.supported_column_count = np.asarray([2, 1])
        picks.site_spacing_max_error_nm = np.asarray([1.0, 9.0])
        picks.site_gap_contrast = np.asarray([0.8, 0.1])
        picks.grid_vs_blob_delta_bic = np.asarray([20.0, -5.0])

        shown, total = PaintAnalysisApp._draw_visible_origami_footprints(app, axis, picks)

        self.assertEqual((shown, total), (2, 2))
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(len(axis.collections[0].get_offsets()), 2 * len(grid))
        self.assertTrue(any(text.get_text().startswith("FAIL:") for text in axis.texts))

    def test_origami_acceptance_defaults_match_validated_examples(self) -> None:
        self.assertEqual(DEFAULT_ORIGAMI_MIN_POINTS, 100)
        self.assertEqual(DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM, 8.0)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS, 128)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_PASSES, 3)
        self.assertEqual(DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM, 20.0)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE, 0.10)
        self.assertEqual(DEFAULT_ORIGAMI_CORRELATION_THRESHOLD, 0.40)
        self.assertTrue(DEFAULT_ORIGAMI_USE_CORRELATION_GATE)
        self.assertTrue(DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS)

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
            "Origami type counts": "none",
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

    def test_filtered_map_uses_cached_filtered_points_for_dynamic_zoom_render(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.dynamic_render_after_id = "queued"
        app.dynamic_zoom_render = FakeVariable(True)
        app.loaded = SimpleNamespace(path="source.hdf5", info=[{"Pixelsize": 100.0}])
        app.dynamic_render_running = False
        app.dynamic_render_pending = False
        app.dynamic_render_request_id = 7
        app.active_notebook_tab = FILTERED_MAP_TAB
        app.filtered_map_locs = ["filtered-localizations"]
        app.filtered_map_render_context = {
            "map_source": "Corrected map",
            "source_label": "corrected",
            "scope_text": "full map",
            "filter_text": "precision: 0-10",
        }
        app.render_min_density = FakeVariable(0.0)
        app.render_max_density = FakeVariable(10.0)
        app.render_blur_method = FakeVariable("none")
        app.min_blur_width = FakeVariable(0.0)
        app.status = FakeVariable("")
        app._axis_canvas_for_tab = mock.Mock(return_value=(FakeAxis(), object()))
        app._full_map_viewport_nm = mock.Mock(return_value=(0.0, 1000.0, 0.0, 1000.0))
        app._shared_map_viewport_nm = mock.Mock(return_value=(100.0, 300.0, 200.0, 500.0))
        app._dynamic_render_pixel_nm = mock.Mock(return_value=1.0)
        app._run_worker = mock.Mock()

        app._start_dynamic_map_render()

        worker = app._run_worker.call_args.args[0]
        rendered = {
            "image": np.ones((3, 2)),
            "extent": (100.0, 300.0, 200.0, 500.0),
        }
        with mock.patch("paint_analysis_gui.render_filtered_map_with_settings", return_value=rendered) as render:
            result_kind, payload = worker()

        self.assertEqual(result_kind, "dynamic_map")
        self.assertEqual(payload["dynamic_target_kind"], "filtered_map")
        self.assertIs(payload["map"], rendered)
        render.assert_called_once_with(
            app.filtered_map_locs,
            app.loaded.info,
            1.0,
            "none",
            0.0,
            (100.0, 300.0, 200.0, 500.0),
        )

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

    def test_loaded_source_switch_rerenders_from_visible_origami_viewport(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Loaded source data"
        app._current_notebook_tab_index = mock.Mock(return_value=ORIGAMI_TAB)
        app.update_idletasks = mock.Mock()
        app._schedule_origami_zoom_render = mock.Mock()

        app._rerender_loaded_origami_source_view()

        app._schedule_origami_zoom_render.assert_called_once_with(delay_ms=0)
        app.update_idletasks.assert_called_once_with()

    def test_loaded_source_switch_does_not_rerender_after_view_changed(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_last_rendered_plot_option = "Selected origami detail"
        app._current_notebook_tab_index = mock.Mock(return_value=ORIGAMI_TAB)
        app.update_idletasks = mock.Mock()
        app._schedule_origami_zoom_render = mock.Mock()

        app._rerender_loaded_origami_source_view()

        app._schedule_origami_zoom_render.assert_not_called()

    def test_entering_origami_tab_queues_visible_source_rerender(self) -> None:
        app = SimpleNamespace(
            active_notebook_tab=FILTERED_MAP_TAB,
            _axis_canvas_for_tab=mock.Mock(return_value=None),
            _current_notebook_tab_index=mock.Mock(return_value=ORIGAMI_TAB),
            _sync_global_sidebar_visibility=mock.Mock(),
            after_idle=mock.Mock(),
            _rerender_loaded_origami_source_view=mock.Mock(),
        )

        PaintAnalysisApp._on_notebook_tab_changed(app, None)

        self.assertEqual(app.active_notebook_tab, ORIGAMI_TAB)
        app.after_idle.assert_called_once_with(app._rerender_loaded_origami_source_view)

    def test_origami_home_restores_stable_extent_and_forces_rerender(self) -> None:
        axis = mock.Mock()
        canvas = SimpleNamespace(
            figure=SimpleNamespace(axes=[axis]),
            draw_idle=mock.Mock(),
        )
        app = SimpleNamespace(
            origami_gallery_home_limits=None,
            origami_last_rendered_plot_option="Identified origami template matches",
            _origami_home_viewport_nm=mock.Mock(return_value=(10.0, 210.0, 20.0, 320.0)),
            _schedule_origami_footprint_refresh=mock.Mock(),
            _schedule_origami_zoom_render=mock.Mock(),
        )
        toolbar = OrigamiToolbar.__new__(OrigamiToolbar)
        toolbar.app = app
        toolbar.canvas = canvas

        toolbar.home()

        axis.set_xlim.assert_called_once_with(10.0, 210.0, emit=False)
        axis.set_ylim.assert_called_once_with(20.0, 320.0, emit=False)
        canvas.draw_idle.assert_called_once()
        app._schedule_origami_footprint_refresh.assert_called_once_with()
        app._schedule_origami_zoom_render.assert_called_once_with(delay_ms=0)

    def test_origami_home_extent_uses_original_render_not_zoomed_artist(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_roi_history_position = -1
        app.origami_roi_history = []
        app.origami_random_inspection_payload = None
        app.origami_last_rendered_plot_option = "Identified origami template matches"
        app.origami_source_render_result = {"extent": (100.0, 900.0, 200.0, 800.0)}

        self.assertEqual(
            app._origami_home_viewport_nm(),
            (100.0, 900.0, 200.0, 800.0),
        )


if __name__ == "__main__":
    unittest.main()
