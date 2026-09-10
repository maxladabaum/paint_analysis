import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from paint_analysis_gui import (
    PaintAnalysisApp,
    evenly_distributed_tile_indices,
    fully_fitting_roi_tiles,
    independent_origami_pipeline_params,
    randomized_tile_order,
)


class TiledOrigamiTests(unittest.TestCase):
    def test_independent_tile_pipeline_discards_validation_roi_caches(self) -> None:
        cached_object = object()
        alignment_template = {"name": "L_L", "image": np.ones((3, 3))}
        classification_templates = (
            {"name": "full", "image": np.ones((3, 3))},
            {"name": "square", "image": np.eye(3)},
        )
        digital_model = {"bit_ids": ("left", "right")}
        fresh = independent_origami_pipeline_params(
            {
                "pick_bin_size_nm": 10.0,
                "shared_alignment_template": alignment_template,
                "custom_templates": classification_templates,
                "digital_pixel_model": digital_model,
                "min_rectangle_confidence": 0.3,
                "min_monte_carlo_probability": 0.5,
                "_inspection_stage": 4,
                "_display_stage": 4,
                "_precomputed_candidates": cached_object,
                "_prealigned_picks": cached_object,
                "_premeasured_picks": cached_object,
                "_candidate_core_bounds_nm": (0.0, 1.0, 0.0, 1.0),
                "_alignment_accepted_mask": (True, False),
            }
        )

        self.assertEqual(fresh["pick_bin_size_nm"], 10.0)
        self.assertEqual(fresh["_inspection_stage"], 5)
        self.assertFalse(any(key.startswith("_pre") for key in fresh))
        self.assertNotIn("_candidate_core_bounds_nm", fresh)
        self.assertNotIn("_alignment_accepted_mask", fresh)
        self.assertIs(fresh["shared_alignment_template"], alignment_template)
        self.assertIs(fresh["custom_templates"], classification_templates)
        self.assertIs(fresh["digital_pixel_model"], digital_model)
        self.assertEqual(fresh["min_rectangle_confidence"], 0.3)
        self.assertEqual(fresh["min_monte_carlo_probability"], 0.5)

    def test_distributed_analysis_uses_multi_template_aware_validated_context(self) -> None:
        variable = lambda value: SimpleNamespace(get=lambda: value, set=mock.Mock())
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._validated_origami_tile_context = mock.Mock(
            return_value=(
                pd.DataFrame({"x": [1.0], "y": [1.0]}),
                1.0,
                [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)],
                {
                    "rows": 8,
                    "columns": 12,
                    "spacing_x_nm": 10.0,
                    "spacing_y_nm": 6.0,
                    "site_mask_radius_nm": 7.5,
                    "min_site_localizations": 3,
                    "min_site_evidence": 0.1,
                },
                {"source": "Corrected", "source_path": "/tmp/source.hdf5"},
            )
        )
        app.origami_tile_count = variable(1)
        app.origami_loaded_roi_nm = (10.0, 20.0, 0.0, 10.0)
        app.origami_identification_generation = 4
        app.origami_g5m_sigma_min_nm = variable(1.0)
        app.origami_g5m_sigma_max_nm = variable(4.0)
        app.origami_g5m_min_locs = variable(2)
        app.origami_g5m_bic_patience = variable(3)
        app.origami_site_radius_nm = variable(7.5)
        app.origami_allow_mirror = variable(True)
        app.origami_overlay_pixel_nm = variable(1.0)
        app.origami_overlay_padding_nm = variable(10.0)
        app.origami_overlay_blur_nm = variable(1.0)
        app.origami_identification_progress = variable(0.0)
        app.origami_identification_progress_text = variable("")
        app.origami_identify_button = mock.Mock()
        app.origami_tiled_button = mock.Mock()
        app.origami_n_tiles_button = mock.Mock()
        app.origami_random_roi_button = mock.Mock()
        app._refresh_origami_action_states = mock.Mock()
        app.status = variable("")
        app._run_worker = mock.Mock()

        app.analyze_tiled_origamis(use_tile_limit=True)

        app._validated_origami_tile_context.assert_called_once_with()
        app._run_worker.assert_called_once()
        self.assertTrue(app.origami_identification_running)

        limited_task = app._run_worker.call_args.args[0]
        app._tiled_origami_worker = mock.Mock(return_value=("ignored", {}))
        limited_task()
        limited_indices = app._tiled_origami_worker.call_args.args[3]
        np.testing.assert_array_equal(limited_indices, np.asarray([1]))

        app.origami_identification_running = False
        app._run_worker.reset_mock()
        app._tiled_origami_worker.reset_mock()
        app.analyze_tiled_origamis(use_tile_limit=False)

        app._run_worker.assert_called_once()
        whole_image_task = app._run_worker.call_args.args[0]
        whole_image_task()
        whole_image_indices = app._tiled_origami_worker.call_args.args[3]
        np.testing.assert_array_equal(whole_image_indices, np.asarray([0, 1]))
        self.assertEqual(app._validated_origami_tile_context.call_count, 2)

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

    def test_partial_edge_tiles_cover_image_once_and_preserve_validation_roi(self) -> None:
        tiles = fully_fitting_roi_tiles(23.0, 17.0, (3.0, 13.0, 2.0, 8.0), include_partial_edges=True)
        self.assertIn((3.0, 13.0, 2.0, 8.0), tiles)
        self.assertIn((0.0, 3.0, 0.0, 2.0), tiles)
        self.assertIn((13.0, 23.0, 14.0, 17.0), tiles)
        self.assertAlmostEqual(sum((x1-x0)*(y1-y0) for x0,x1,y0,y1 in tiles), 23*17)
        for x in np.arange(0, 23, 0.5):
            for y in np.arange(0, 17, 0.5):
                self.assertEqual(sum(x0 <= x < x1 and y0 <= y < y1 for x0,x1,y0,y1 in tiles), 1)
        self.assertEqual(fully_fitting_roi_tiles(4, 3, (0, 10, 0, 10), include_partial_edges=True), [(0, 4, 0, 3)])

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

    def test_tiled_worker_retains_spatial_results_from_every_selected_tile(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._origami_identification_worker_progress = mock.Mock()
        source = pd.DataFrame(
            {
                "x": [2.0, 3.0, 12.0, 13.0, 21.0, 25.0],
                "y": [2.0, 3.0, 2.0, 3.0, 13.0, 15.0],
            }
        )
        tiles = [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)]
        identification_params = {
            "pick_bin_size_nm": 5.0,
            "connect_distance_nm": 20.0,
            "density_threshold": 0.2,
            "min_candidate_points": 1,
            "max_candidate_points": 10,
            "rows": 2,
            "columns": 2,
            "spacing_x_nm": 5.0,
            "spacing_y_nm": 5.0,
            "rectangle_margin_nm": 2.0,
            "min_rectangle_confidence": 0.4,
            "alignment_pixel_nm": 2.0,
            "alignment_iterations": 2,
            "alignment_template_image": np.eye(2),
            "template_pixel_size_x_nm": 1.0,
            "template_pixel_size_y_nm": 1.0,
        }
        source_params = {
            "source": "Corrected localizations",
            "active_filters": [],
        }
        overlay_params = {
            "rows": 2,
            "columns": 2,
            "spacing_x_nm": 5.0,
            "spacing_y_nm": 5.0,
            "g5m_sigma_min_nm": 1.0,
            "g5m_sigma_max_nm": 8.0,
            "g5m_min_locs": 2,
            "g5m_bic_patience": 2,
            "site_radius_nm": 3.0,
            "direct_site_radius_nm": 3.0,
            "direct_min_site_localizations": 1,
            "direct_min_site_evidence": 0.0,
            "allow_mirror": False,
            "overlay_pixel_nm": 1.0,
            "overlay_padding_nm": 2.0,
            "overlay_blur_nm": 1.0,
            "source_path": "example.hdf5",
        }
        grid = np.asarray([[-2.5, -2.5], [2.5, 2.5]])
        first = SimpleNamespace(
            accepted_count=1,
            regions=[np.asarray([[2.0, 2.0], [3.0, 3.0]])],
            accepted_aligned_regions=[np.asarray([[-0.5, -0.5], [0.5, 0.5]])],
            accepted_mask=np.asarray([True]),
            template_points_nm=grid,
        )
        second = SimpleNamespace(
            accepted_count=1,
            regions=[np.asarray([[12.0, 2.0], [13.0, 3.0]])],
            accepted_aligned_regions=[np.asarray([[-0.5, -0.5], [0.5, 0.5]])],
            accepted_mask=np.asarray([True]),
            template_points_nm=grid,
        )
        combined_picks = object()
        analysis_result = object()

        with (
            mock.patch(
                "paint_analysis_gui.identify_origami_regions",
                side_effect=[first, second],
            ) as identify,
            mock.patch(
                "paint_analysis_gui.concatenate_origami_pick_results",
                return_value=combined_picks,
            ) as concatenate,
            mock.patch(
                "paint_analysis_gui.align_picked_origamis",
                return_value=analysis_result,
            ) as align,
        ):
            kind, payload = app._tiled_origami_worker(
                source,
                1.0,
                tiles,
                np.asarray([0, 1]),
                identification_params,
                source_params,
                overlay_params,
            )

        self.assertEqual(kind, "origami_tiled")
        self.assertEqual(identify.call_count, 2)
        self.assertEqual(len(identify.call_args_list[0].args[0]), 2)
        self.assertEqual(len(identify.call_args_list[1].args[0]), 2)
        concatenate.assert_called_once_with([first, second])
        self.assertEqual(len(align.call_args.args[0]), 2)
        self.assertIs(payload["picks"], combined_picks)
        self.assertIs(payload["result"], analysis_result)
        np.testing.assert_allclose(payload["points_nm"], source[["x", "y"]].to_numpy())
        self.assertEqual(len(payload["locs"]), 6)
        # Edge points remain in the overview without entering any tile's classifier.
        np.testing.assert_allclose(payload["points_nm"], source[["x", "y"]].to_numpy())
        self.assertEqual(len(payload["locs"]), 6)


    def test_tiled_completion_replaces_validation_roi_plot_caches(self) -> None:
        def variable(value: object) -> SimpleNamespace:
            result = SimpleNamespace(value=value)
            result.get = lambda: result.value
            result.set = lambda new_value: setattr(result, "value", new_value)
            return result
        points = np.asarray([[2.0, 3.0], [102.0, 103.0]])
        locs = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})
        picks = object()
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_source_points_nm = np.asarray([[1.0, 1.0]])
        app.origami_source_locs = locs.iloc[:1]
        app.origami_source_render_result = {"extent": (0.0, 10.0, 0.0, 10.0)}
        app.origami_pick_result = object()
        app.origami_identification_params = {"rows": 1}
        app.origami_loaded_source_label = "validation ROI"
        app.origami_roi_history_position = 0
        app.origami_multi_template_results = {"validation type": {"picks": object()}}
        app.origami_multi_template_counts = {"validation type": 1}
        app.origami_multi_template_overlay_results = {"validation type": {}}
        app.origami_multi_template_overlays_building = {"validation type"}
        app.origami_multi_template_unclassified_count = 2
        app.origami_multi_template_unclassified_centers_nm = np.ones((2, 2))
        app.origami_multi_template_unclassified_details = [{}, {}]
        app.origami_template_result_view = variable("validation type")
        app.origami_template_result_combo = mock.Mock()
        app.origami_plot_option = variable("Identified origami template matches")
        app._plot_origami_analysis = mock.Mock()
        app._refresh_origami_action_states = mock.Mock()
        payload = {
            "points_nm": points,
            "locs": locs,
            "picks": picks,
            "identification_params": {"rows": 8, "columns": 12},
            "source": "Corrected (10 spatially distributed tiles)",
        }

        app._apply_tiled_origami_result(payload)

        np.testing.assert_allclose(app.origami_source_points_nm, points)
        self.assertIs(app.origami_source_locs, locs)
        self.assertIs(app.origami_pick_result, picks)
        self.assertIsNone(app.origami_source_render_result)
        self.assertEqual(app.origami_roi_history_position, -1)
        self.assertEqual(app.origami_multi_template_results, {})
        self.assertEqual(app.origami_multi_template_counts, {})
        self.assertEqual(app.origami_template_result_view.get(), "All templates")
        self.assertEqual(app.origami_plot_option.get(), "Aligned density")
        app.origami_template_result_combo.configure.assert_called_once_with(
            values=("All templates",)
        )
        app.origami_template_result_combo.state.assert_called_once_with(["disabled"])
        app._plot_origami_analysis.assert_called_once_with(payload)
        app._refresh_origami_action_states.assert_called_once_with()

    def test_multi_template_tiled_worker_aggregates_every_tile_by_template(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._origami_identification_worker_progress = mock.Mock()
        source = pd.DataFrame(
            {"x": [2.0, 3.0, 12.0, 13.0, 21.0, 25.0], "y": [2.0, 3.0, 2.0, 3.0, 13.0, 15.0]}
        )
        tiles = [(0.0, 4.0, 0.0, 10.0), (4.0, 14.0, 0.0, 10.0)]
        templates = [{"name": "square"}, {"name": "full"}]
        validation_cache = object()
        identification_params = {
            "custom_templates": templates,
            "_inspection_stage": 4,
            "_precomputed_candidates": validation_cache,
            "_prealigned_picks": validation_cache,
            "_premeasured_picks": validation_cache,
        }
        source_params = {
            "source": "Corrected localizations",
            "source_path": "example.hdf5",
            "active_filters": [],
        }
        overlay_params = {"source_path": "example.hdf5"}

        def tile_payload(square_count: int, full_count: int, x_offset: float) -> tuple[str, dict]:
            return "origami_multi_picks", {
                "templates": [
                    {
                        "name": "square",
                        "picks": SimpleNamespace(tile=(x_offset, "square")),
                        "params": {
                            "classification_scores": (0.8,),
                            "classification_dispositions": ("classified as square",),
                            "classification_logical_bit_probabilities": ((0.2, 0.8),),
                            "digital_group_localization_evidence": ((x_offset + 1.0, x_offset + 2.0),),
                            "digital_group_prominences": ((0.6, 0.7),),
                            "digital_pixel_probabilities": ((0.25, 0.75),),
                        },
                    },
                    {
                        "name": "full",
                        "picks": SimpleNamespace(tile=(x_offset, "full")),
                        "params": {
                            "classification_scores": (0.7,),
                            "classification_dispositions": ("classified as full",),
                        },
                    },
                ],
                "counts": np.asarray([square_count, full_count]),
                "unclassified_count": 1,
                "suppressed_duplicate_count": 0,
                "unclassified_centers_nm": np.asarray([[x_offset + 4.0, 4.0]]),
                "unclassified_details": [{"center_nm": np.asarray([x_offset + 4.0, 4.0])}],
            }

        app._identify_origami_worker = mock.Mock(
            side_effect=[tile_payload(1, 2, 0.0), tile_payload(3, 4, 10.0)]
        )
        combined_square = object()
        combined_full = object()
        with mock.patch(
            "paint_analysis_gui.concatenate_origami_pick_results",
            side_effect=[combined_square, combined_full],
        ) as concatenate:
            kind, payload = app._tiled_origami_worker(
                source,
                1.0,
                tiles,
                np.asarray([0, 1]),
                identification_params,
                source_params,
                overlay_params,
            )

        self.assertEqual(kind, "origami_multi_picks")
        np.testing.assert_array_equal(payload["counts"], [4, 6])
        self.assertEqual(payload["unclassified_count"], 2)
        self.assertEqual(payload["accepted_count"], 10)
        self.assertEqual(len(payload["points_nm"]), 6)
        self.assertIs(payload["templates"][0]["picks"], combined_square)
        self.assertIs(payload["templates"][1]["picks"], combined_full)
        self.assertEqual(
            payload["templates"][0]["params"]["classification_scores"],
            (0.8, 0.8),
        )
        self.assertEqual(
            payload["templates"][0]["params"]["classification_logical_bit_probabilities"],
            ((0.2, 0.8), (0.2, 0.8)),
        )
        self.assertEqual(
            payload["templates"][0]["params"]["digital_group_localization_evidence"],
            ((1.0, 2.0), (11.0, 12.0)),
        )
        self.assertEqual(
            payload["templates"][0]["params"]["digital_group_prominences"],
            ((0.6, 0.7), (0.6, 0.7)),
        )
        self.assertEqual(
            payload["templates"][0]["params"]["digital_pixel_probabilities"],
            ((0.25, 0.75), (0.25, 0.75)),
        )
        self.assertEqual(app._identify_origami_worker.call_count, 2)
        expected_tile_points = (
            np.asarray([[2.0, 2.0], [3.0, 3.0]]),
            np.asarray([[12.0, 2.0], [13.0, 3.0]]),
        )
        for worker_call, expected_points in zip(
            app._identify_origami_worker.call_args_list, expected_tile_points
        ):
            np.testing.assert_allclose(worker_call.args[0], expected_points)
            tile_params = worker_call.args[1]
            self.assertEqual(tile_params["_inspection_stage"], 5)
            self.assertFalse(any(key.startswith("_pre") for key in tile_params))
            self.assertIn("_candidate_core_bounds_nm", tile_params)
        self.assertEqual(concatenate.call_count, 2)
        # Edge points remain in the overview without entering any tile's classifier.
        np.testing.assert_allclose(payload["points_nm"], source[["x", "y"]].to_numpy())
        self.assertEqual(len(payload["locs"]), 6)


    def test_multi_template_tiled_worker_returns_unclassified_only_results(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._origami_identification_worker_progress = mock.Mock()
        source = pd.DataFrame(
            {"x": [2.0, 3.0, 12.0, 13.0], "y": [2.0, 3.0, 2.0, 3.0]}
        )
        tiles = [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)]
        identification_params = {
            "custom_templates": [{"name": "square"}, {"name": "full"}],
            "_inspection_stage": 4,
        }
        source_params = {
            "source": "Corrected localizations",
            "source_path": "example.hdf5",
            "active_filters": [],
        }
        overlay_params = {"source_path": "example.hdf5"}

        def rejected_tile(x_offset: float) -> tuple[str, dict]:
            return "origami_multi_picks", {
                "templates": [
                    {
                        "name": name,
                        "picks": SimpleNamespace(tile=(x_offset, name)),
                        "params": {
                            "classification_scores": (0.2,),
                            "classification_dispositions": ("unclassified",),
                        },
                    }
                    for name in ("square", "full")
                ],
                "counts": np.asarray([0, 0]),
                "unclassified_count": 1,
                "suppressed_duplicate_count": 0,
                "unclassified_centers_nm": np.asarray([[x_offset + 4.0, 4.0]]),
                "unclassified_details": [{
                    "center_nm": np.asarray([x_offset + 4.0, 4.0]),
                    "failure_reasons": ("correlation",),
                }],
            }

        app._identify_origami_worker = mock.Mock(
            side_effect=[rejected_tile(0.0), rejected_tile(10.0)]
        )
        combined_picks = (object(), object())
        with mock.patch(
            "paint_analysis_gui.concatenate_origami_pick_results",
            side_effect=combined_picks,
        ):
            kind, payload = app._tiled_origami_worker(
                source,
                1.0,
                tiles,
                np.asarray([0, 1]),
                identification_params,
                source_params,
                overlay_params,
            )

        self.assertEqual(kind, "origami_multi_picks")
        np.testing.assert_array_equal(payload["counts"], [0, 0])
        self.assertEqual(payload["accepted_count"], 0)
        self.assertEqual(payload["unclassified_count"], 2)
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(len(payload["unclassified_details"]), 2)
        self.assertIs(payload["templates"][0]["picks"], combined_picks[0])
        self.assertIs(payload["templates"][1]["picks"], combined_picks[1])


if __name__ == "__main__":
    unittest.main()
