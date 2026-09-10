import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, PngImagePlugin
from matplotlib.figure import Figure

from origami_analysis import OrigamiPickResult, classify_template_candidates

from paint_analysis_gui import (
    DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS,
    DEFAULT_ORIGAMI_ALIGNMENT_PASSES,
    DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM,
    DEFAULT_ORIGAMI_CORRELATION_THRESHOLD,
    DEFAULT_ORIGAMI_MAX_POINTS,
    DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM,
    DEFAULT_ORIGAMI_MIN_DENSITY,
    DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR,
    DEFAULT_ORIGAMI_MIN_POINTS,
    DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS,
    DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE,
    DEFAULT_ORIGAMI_PICK_BIN_NM,
    DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY,
    DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS,
    DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY,
    DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS,
    DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS,
    DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY,
    DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM,
    DEFAULT_ORIGAMI_USE_CORRELATION_GATE,
    DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE,
    FILTERED_MAP_TAB,
    LoadedData,
    ORIGAMI_TAB,
    OrigamiToolbar,
    ORIGAMI_ALIGNMENT_CACHE_KEYS,
    ORIGAMI_SITE_CACHE_KEYS,
    PaintAnalysisApp,
    PicassoAimStatusProgress,
    TemporalVLineAnnotation,
    classification_bias_diagnostics,
    alignment_template_overlay_points,
    classified_template_overlay_indices,
    classified_template_overlay_points,
    origami_grid_points,
    digital_group_blob_contours,
    digital_group_decision_blobs,
    digital_group_decision_annotations,
    digital_group_decision_overlay_points,
    digital_group_site_memberships,
    custom_template_contours_nm,
    custom_template_display_name,
    load_custom_template_image,
    load_custom_template_metadata,
    logical_stroke_model_from_metadata,
    load_development_session_cache,
    optimal_dynamic_render_pixel_nm,
    origami_stage_cache_matches,
    origami_candidate_failure_reasons,
    origami_source_fingerprint,
    origami_site_decision_label,
    parse_temporal_vline_annotation,
    responsive_column_count,
    save_development_session_cache,
    save_development_map_cache,
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
    def test_alignment_overlay_is_extracted_from_step_two_template(self) -> None:
        image = np.zeros((9, 9), dtype=float)
        image[2, 2] = 1.0
        image[6, 6] = 1.0
        fallback = np.zeros((4, 2), dtype=float)
        params = {
            "shared_alignment_template": {
                "image": image,
                "rows": 3,
                "columns": 3,
                "spacing_x_nm": 10.0,
                "spacing_y_nm": 10.0,
                "rectangle_margin_nm": 5.0,
                "template_pixel_size_x_nm": 1.0,
                "template_pixel_size_y_nm": 1.0,
            }
        }

        points = alignment_template_overlay_points(fallback, params)

        self.assertEqual(points.shape, (2, 2))

    def test_recorded_step_two_points_override_full_measurement_grid(self) -> None:
        full_grid = np.column_stack((np.arange(96, dtype=float), np.zeros(96)))
        recorded = ((-10.0, 0.0), (0.0, 0.0), (0.0, 10.0))

        points = alignment_template_overlay_points(
            full_grid,
            {
                "alignment_template_overlay_points_nm": recorded,
                "shared_alignment_template": {},
            },
        )

        np.testing.assert_allclose(points, recorded)

    def test_step_one_cache_uses_exact_source_and_skips_an_unchanged_rerun(self) -> None:
        first = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        second = np.asarray([[0.0, 0.0], [2.0, 2.0]])
        self.assertEqual(origami_source_fingerprint(first), origami_source_fingerprint(first.copy()))
        self.assertNotEqual(origami_source_fingerprint(first), origami_source_fingerprint(second))

        signature = (10.0, 20.0, 0.8, 200, len(first), origami_source_fingerprint(first))
        candidates = ([first], np.zeros((1, 1)), np.zeros((1, 1)), (0.0, 1.0, 0.0, 1.0), np.zeros((1, 1)))
        app = SimpleNamespace(
            origami_identification_running=False,
            origami_source_points_nm=first,
            origami_loaded_source_path=Path("source.hdf5"),
            origami_staged_candidates=candidates,
            origami_staged_candidate_signature=signature,
            origami_active_step=0,
            origami_identification_progress=FakeVariable(0.0),
            origami_identification_progress_text=FakeVariable(""),
            origami_step_progress={1: FakeVariable(0.0)},
            origami_step_progress_text={1: FakeVariable("")},
            origami_staged_status=FakeVariable(""),
            status=FakeVariable(""),
            _origami_candidate_stage_signature=lambda: signature,
            _plot_origami_coarse_density=mock.Mock(),
            _run_worker=mock.Mock(),
        )

        PaintAnalysisApp._run_origami_candidate_stage(app)

        app._run_worker.assert_not_called()
        app._plot_origami_coarse_density.assert_called_once_with()
        self.assertEqual(app.origami_step_progress[1].get(), 100.0)

    def test_step_four_cache_ignores_classification_template_names(self) -> None:
        previous = {
            key: f"stable-{index}"
            for index, key in enumerate(
                ORIGAMI_ALIGNMENT_CACHE_KEYS + ORIGAMI_SITE_CACHE_KEYS
            )
        }
        current = dict(previous)
        previous["custom_template_name"] = "L_L"
        current["custom_template_name"] = "square"
        previous["template_mode"] = "Custom image"
        current["template_mode"] = "Custom image"

        self.assertTrue(
            origami_stage_cache_matches(
                previous,
                current,
                ORIGAMI_ALIGNMENT_CACHE_KEYS + ORIGAMI_SITE_CACHE_KEYS,
            )
        )
        current["_alignment_template_signature"] = "different-alignment"
        self.assertFalse(
            origami_stage_cache_matches(
                previous,
                current,
                ORIGAMI_ALIGNMENT_CACHE_KEYS + ORIGAMI_SITE_CACHE_KEYS,
            )
        )

    def test_classified_template_overlay_uses_alignment_and_active_logical_sites(self) -> None:
        grid = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
        params = {
            "logical_model": {
                "physical_shape": (2, 4),
                "alignment_cells": (0, 1),
                "bit_physical_cells": ((1, 2, 3), (3, 4), (6, 7)),
                "active_bits": (True, False, True),
            }
        }

        indices = classified_template_overlay_indices(grid, params)

        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 6, 7])

        sparse_alignment = grid[[0, 1]]
        calibrated_params = {
            **params,
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 20.0,
        }
        points = classified_template_overlay_points(sparse_alignment, calibrated_params)
        full_grid = np.asarray(
            [
                [-15.0, -10.0], [-5.0, -10.0], [5.0, -10.0], [15.0, -10.0],
                [-15.0, 10.0], [-5.0, 10.0], [5.0, 10.0], [15.0, 10.0],
            ]
        )
        np.testing.assert_allclose(points, full_grid[[0, 1, 2, 3, 6, 7]])

    def test_digital_group_decisions_replace_physical_site_labels(self) -> None:
        grid = np.column_stack((np.arange(6, dtype=float), np.zeros(6)))
        params = {
            "digital_pixel_model": {
                "bit_ids": ("slash", "bottom"),
                "bit_physical_cells": ((0, 1, 2), (2, 4, 5)),
                "physical_shape": (1, 6),
            },
            "digital_pixel_probabilities": ((0.82, 0.21),),
        }

        annotations = digital_group_decision_annotations(grid, params, 0)

        self.assertIsNotNone(annotations)
        assert annotations is not None
        self.assertEqual([item[1] for item in annotations], [
            "slash",
            "bottom — FAIL: ON probability 0.21 < 0.50",
        ])
        self.assertEqual([item[2] for item in annotations], ["#22c55e", "#ff3030"])
        np.testing.assert_allclose(annotations[0][0], [1.0, 0.0])
        np.testing.assert_allclose(annotations[1][0], [11.0 / 3.0, 0.0])

        selected_points = digital_group_decision_overlay_points(grid, params, 0)
        assert selected_points is not None
        np.testing.assert_allclose(selected_points, grid[[0, 1, 2]])
        blobs = digital_group_decision_blobs(grid, params, 0)
        self.assertEqual([item[0] for item in blobs], ["slash", "bottom"])
        self.assertEqual([item[2] for item in blobs], ["#22c55e", "#ff3030"])
        self.assertGreater(blobs[0][3], blobs[1][3])
        self.assertEqual([item[4] for item in blobs], ["solid", "solid"])

    def test_digital_group_boundaries_include_support_and_prominence_gates(self) -> None:
        grid = np.column_stack((np.arange(3, dtype=float), np.zeros(3)))
        params = {
            "digital_pixel_model": {
                "bit_ids": ("on", "low_support", "low_prominence"),
                "bit_physical_cells": ((0,), (1,), (2,)),
                "physical_shape": (1, 3),
            },
            "digital_pixel_probabilities": ((0.8, 0.8, 0.8),),
            "digital_group_localization_evidence": ((5.0, 2.5, 5.0),),
            "digital_group_prominences": ((0.5, 0.5, 0.0),),
            "min_site_localizations": 5,
            "min_site_evidence": 0.1,
        }
        axis = Figure().subplots()
        artists = PaintAnalysisApp._draw_digital_decision_blobs(
            PaintAnalysisApp.__new__(PaintAnalysisApp), axis, grid, params, 0
        )
        self.assertEqual(len(artists), 3)
        np.testing.assert_allclose(artists[0].get_edgecolors()[0], [34 / 255, 197 / 255, 94 / 255, 0.95])
        for artist in artists[1:]:
            np.testing.assert_allclose(artist.get_edgecolors()[0], [1, 48 / 255, 48 / 255, 0.95])

    def test_off_boundaries_remain_red_under_overlapping_on_alignment_group(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        grid = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        for group_ids, probabilities in (
            (("off", "full_align"), (0.1, 0.9)),
            (("full_align", "off"), (0.9, 0.1)),
        ):
            with self.subTest(group_ids=group_ids):
                params = {
                    "digital_pixel_model": {
                        "bit_ids": group_ids,
                        "bit_physical_cells": ((0, 1), (0, 1)),
                        "physical_shape": (1, 2),
                    },
                    "digital_pixel_probabilities": (probabilities,),
                    "site_mask_radius_nm": 2.0,
                }
                figure = Figure(figsize=(3, 2), dpi=100)
                canvas = FigureCanvasAgg(figure)
                axis = figure.subplots()
                axis.set_facecolor("black")
                axis.set_xlim(-4, 14)
                axis.set_ylim(-4, 4)
                PaintAnalysisApp._draw_digital_decision_blobs(
                    PaintAnalysisApp.__new__(PaintAnalysisApp), axis, grid, params, 0,
                )
                canvas.draw()
                pixels = np.asarray(canvas.buffer_rgba()).astype(float)
                red = (pixels[..., 0] > 150) & (pixels[..., 0] > 2 * pixels[..., 1])
                self.assertGreater(np.count_nonzero(red), 100)

    def test_single_template_classification_preserves_failed_fits(self) -> None:
        classified = classify_template_candidates(
            [np.asarray([[0.0, 0.0], [100.0, 0.0]])],
            [np.asarray([True, False])],
            [np.asarray([-3.0, 2.0])],
            match_distance_nm=10.0,
            minimum_winner_probability=0.25,
        )
        np.testing.assert_array_equal(classified.assignment_masks[0], [True, False])
        np.testing.assert_array_equal(classified.counts, [1])
        self.assertEqual(classified.unclassified_count, 1)

    def test_digital_pixel_statistics_include_on_and_off_measurements(self) -> None:
        params = {
            "digital_pixel_model": {
                "physical_shape": (1, 2), "bit_ids": ("on", "off"),
                "bit_physical_cells": ((0,), (1,)),
            },
            "digital_pixel_probabilities": ((0.9, 0.2),),
            "digital_group_localization_evidence": ((8.0, 2.0),),
            "digital_group_prominences": ((0.8, 0.0),),
            "min_site_localizations": 5, "min_site_evidence": 0.1,
        }
        labels = digital_group_decision_annotations(
            np.asarray([[0, 0], [10, 0]]), params, 0, include_statistics=True,
        )
        self.assertIn("on ON · P(ON)=0.90", labels[0][1])
        self.assertIn("locs/position=8.0; prominence=0.80", labels[0][1])
        self.assertIn("off OFF · P(ON)=0.20", labels[1][1])
        self.assertIn("FAIL: locs/position 2.0 < 5", labels[1][1])

    def test_column_gap_is_shared_by_measurement_grid_and_digital_labels(self) -> None:
        params = {
            "spacing_x_nm": 120.0 / 11,
            "spacing_y_nm": 5.0,
            "digital_pixel_model": {
                "physical_shape": (8, 12),
                "column_offsets_nm": (0.0,) * 6 + (5.0,) * 6,
                "bit_ids": ("left", "right"),
                "bit_physical_cells": ((5,), (6,)),
            },
            "digital_pixel_probabilities": ((0.9, 0.9),),
        }
        grid = origami_grid_points(8, 12, 120.0 / 11, 5.0, params)
        intervals = np.diff(grid[:12, 0])
        expected = np.full(11, 120.0 / 11)
        expected[5] += 5
        np.testing.assert_allclose(intervals, expected)
        np.testing.assert_allclose(grid.mean(axis=0), [0, 0], atol=1e-12)
        self.assertAlmostEqual(np.ptp(grid[:, 0]), 125.0)
        # A sparse alignment grid must reconstruct the same full geometry.
        labels = digital_group_decision_annotations(grid[[0, 1]], params, 0)
        np.testing.assert_allclose([item[0] for item in labels], grid[[5, 6]])
        self.assertFalse(origami_stage_cache_matches(
            {"column_offsets_nm": (0.0,) * 12},
            {"column_offsets_nm": (0.0,) * 6 + (5.0,) * 6},
            ORIGAMI_ALIGNMENT_CACHE_KEYS,
        ))

    def test_digital_group_decisions_fall_back_for_legacy_templates(self) -> None:
        self.assertIsNone(
            digital_group_decision_annotations(
                np.zeros((4, 2)),
                {"digital_pixel_probabilities": ((0.5,),)},
                0,
            )
        )

    def test_digital_group_decision_display_colors_localizations_and_rings_overlaps(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        grid = np.asarray(((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)))
        params = {
            "digital_pixel_model": {
                "bit_ids": ("left", "right"),
                "bit_physical_cells": ((0, 1), (1, 2)),
                "physical_shape": (1, 3),
            },
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 10.0,
            "site_mask_radius_nm": 2.0,
        }
        axis = Figure().subplots()

        artists = PaintAnalysisApp._draw_digital_group_localization_assignments(
            app,
            axis,
            np.asarray(((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))),
            grid,
            params,
        )

        self.assertEqual(len(artists), 4)
        filled = [artist for artist in artists if len(artist.get_facecolors())]
        rings = [artist for artist in artists if not len(artist.get_facecolors())]
        self.assertEqual(len(filled), 2)
        self.assertEqual(len(rings), 2)
        self.assertEqual(sorted(float(artist.get_sizes()[0]) for artist in filled), [9.0, 9.0])
        self.assertEqual(sorted(float(artist.get_sizes()[0]) for artist in rings), [18.0, 28.0])

    def test_digital_group_decisions_show_direct_count_and_prominence(self) -> None:
        grid = np.column_stack((np.arange(3, dtype=float), np.zeros(3)))
        params = {
            "digital_pixel_model": {
                "bit_ids": ("stroke",),
                "bit_physical_cells": ((0, 1, 2),),
                "physical_shape": (1, 3),
            },
            "digital_pixel_probabilities": ((0.8,),),
            "digital_group_localization_evidence": ((2.5,),),
            "min_site_localizations": 2,
            "min_site_evidence": 0.5,
        }

        annotations = digital_group_decision_annotations(grid, params, 0)

        assert annotations is not None
        self.assertEqual(annotations[0][1], "stroke")
        self.assertEqual(annotations[0][2], "#22c55e")

    def test_failed_digital_group_label_lists_only_failed_gates_in_red(self) -> None:
        grid = np.column_stack((np.arange(3, dtype=float), np.zeros(3)))
        params = {
            "digital_pixel_model": {
                "bit_ids": ("stroke",),
                "bit_physical_cells": ((0, 1, 2),),
                "physical_shape": (1, 3),
            },
            "digital_pixel_probabilities": ((0.3,),),
            "digital_group_localization_evidence": ((1.5,),),
            "digital_group_prominences": ((0.2,),),
            "min_site_localizations": 2,
            "min_site_evidence": 0.5,
        }

        annotations = digital_group_decision_annotations(grid, params, 0)

        assert annotations is not None
        self.assertEqual(
            annotations[0][1],
            "stroke — FAIL: locs/position 1.5 < 2; prominence 0.20 < 0.5",
        )
        self.assertEqual(annotations[0][2], "#ff3030")

    def test_digital_group_blobs_outline_overlapping_analog_memberships(self) -> None:
        grid = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 10.0]],
            dtype=float,
        )
        model = {
            "bit_ids": ("horizontal", "corner"),
            "bit_physical_cells": ((0, 1, 2), (2, 3)),
        }

        outlines = digital_group_blob_contours(grid, model)

        self.assertEqual([name for name, _contours in outlines], ["horizontal", "corner"])
        self.assertTrue(all(contours for _name, contours in outlines))
        horizontal_points = np.vstack(outlines[0][1])
        corner_points = np.vstack(outlines[1][1])
        self.assertLess(float(np.min(horizontal_points[:, 0])), 0.0)
        self.assertGreater(float(np.max(horizontal_points[:, 0])), 20.0)
        self.assertGreater(float(np.max(corner_points[:, 1])), 10.0)

    def test_digital_group_circle_matches_site_mask_radius(self) -> None:
        grid = np.asarray(((0.0, 0.0),))
        model = {
            "bit_ids": ("single",),
            "bit_physical_cells": ((0,),),
        }

        outlines = digital_group_blob_contours(grid, model, radius_nm=10.0)

        points = np.vstack(outlines[0][1])
        radii = np.linalg.norm(points, axis=1)
        self.assertAlmostEqual(float(np.median(radii)), 10.0, delta=0.15)

    def test_analog_sites_map_to_digital_groups_with_shared_membership(self) -> None:
        group_ids, memberships = digital_group_site_memberships(
            {
                "bit_ids": ("slash", "bottom"),
                "bit_physical_cells": ((0, 1, 2), (2, 3)),
            },
            5,
        )

        self.assertEqual(group_ids, ("slash", "bottom"))
        self.assertEqual(memberships, ((0,), (0,), (0, 1), (1,), ()))

    def test_result_overlays_include_only_active_groups_from_selected_template(self) -> None:
        grid = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        model = {
            "bit_ids": ("selected", "inactive"),
            "bit_physical_cells": ((0, 1), (1, 2)),
            "active_bits": (True, False),
        }

        outlines = digital_group_blob_contours(grid, model, active_only=True)
        group_ids, memberships = digital_group_site_memberships(
            model, len(grid), active_only=True
        )

        self.assertEqual([name for name, _contours in outlines], ["selected"])
        self.assertEqual(group_ids, ("selected",))
        self.assertEqual(memberships, ((0,), (0,), ()))

        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_identification_params = {
            "digital_pixel_model": {
                "bit_ids": ("selected", "inactive"),
                "active_bits": (True, True),
            },
            "logical_model": model,
        }
        self.assertIs(PaintAnalysisApp._active_origami_digital_pixel_model(app), model)
        axis = Figure().subplots()
        handles = PaintAnalysisApp._draw_digital_group_blobs(app, axis, grid)
        self.assertEqual([handle.get_label() for handle in handles], ["selected"])

    def test_result_overlay_resolves_selected_template_after_tiled_run(self) -> None:
        selected_model = {
            "bit_ids": ("selected", "inactive"),
            "bit_physical_cells": ((0,), (1,)),
            "active_bits": (True, False),
        }
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_identification_params = {
            "digital_pixel_model": {
                "bit_ids": ("selected", "inactive"),
                "active_bits": (True, True),
            }
        }
        app.origami_template_result_view = FakeVariable("slash_slash")
        app.origami_multi_template_results = {
            "slash_slash": {"params": {"logical_model": selected_model}}
        }

        self.assertIs(
            PaintAnalysisApp._active_origami_digital_pixel_model(app),
            selected_model,
        )

    def test_digital_overlay_reconstructs_full_analog_lattice_and_draws_groups(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_identification_params = {
            "digital_pixel_model": {
                "bit_ids": ("shared",),
                "bit_physical_cells": ((0, 1, 2),),
                "physical_shape": (2, 3),
            }
        }
        result = SimpleNamespace(grid_points_nm=np.asarray([[0.0, 0.0], [10.0, 0.0]]))
        settings = {"spacing_x_nm": 10.0, "spacing_y_nm": 20.0}

        grid = PaintAnalysisApp._origami_digital_overlay_grid(app, result, settings)
        axis = Figure().subplots()
        handles = PaintAnalysisApp._draw_digital_group_blobs(app, axis, grid)

        self.assertEqual(grid.shape, (6, 2))
        self.assertEqual([handle.get_label() for handle in handles], ["shared"])
        self.assertEqual(len(axis.collections), 1)

    def test_individual_assignments_color_clusters_by_digital_group(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_identification_params = {
            "digital_pixel_model": {
                "bit_ids": ("one_group",),
                "bit_physical_cells": ((0, 1),),
                "physical_shape": (1, 2),
            }
        }
        app.origami_show_theoretical_overlay = FakeVariable(True)
        app.origami_show_text_statistics = FakeVariable(True)
        app.origami_gallery_tile_hitboxes = []
        app.origami_gallery_page_label = FakeVariable("Page 1/1")
        result = SimpleNamespace(
            origami_count=1,
            aligned_points=[np.asarray([[-5.0, 0.0], [5.0, 0.0]])],
            cluster_labels=[np.asarray([0, 1])],
            cluster_centers_nm=[np.asarray([[-5.0, 0.0], [5.0, 0.0]])],
            cluster_site_indices=[np.asarray([0, 1])],
            source_point_counts=np.asarray([2]),
            rows=1,
            columns=2,
            grid_points_nm=np.asarray([[-5.0, 0.0], [5.0, 0.0]]),
            clustering_method="direct",
            direct_min_site_localizations=1,
            direct_min_site_evidence=0.0,
            site_match_radius_nm=5.0,
            g5m_sigma_min_nm=0.5,
            g5m_sigma_max_nm=5.0,
            g5m_min_locs=3,
            g5m_max_rounds_without_best_bic=2,
        )
        settings = {
            "rows": 1,
            "columns": 2,
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 10.0,
            "pixel_size_nm": 1.0,
            "padding_nm": 5.0,
            "blur_nm": 0.0,
        }
        axis = Figure().subplots()

        PaintAnalysisApp._plot_individual_origami_clusters(
            app, axis, result, settings, np.asarray([0])
        )

        gallery = np.asarray(axis.images[0].get_array())
        np.testing.assert_allclose(gallery[10, 5], gallery[10, 15])
        self.assertEqual(
            [text.get_text() for text in axis.get_legend().get_texts()],
            ["one_group"],
        )
        self.assertIn("1 digital groups", axis.texts[0].get_text())

        for plot in (PaintAnalysisApp._plot_individual_origami_gallery, PaintAnalysisApp._plot_individual_origami_clusters):
            for enabled in (False, True):
                app.origami_show_theoretical_overlay.set(enabled)
                app.origami_show_text_statistics.set(enabled)
                axis = Figure().subplots()
                plot(app, axis, result, settings, np.asarray([0]))
                self.assertEqual(any(type(item).__name__ == "PolyCollection" for item in axis.collections), enabled)
                self.assertEqual(any("n=" in item.get_text() for item in axis.texts), enabled)
                self.assertEqual(len(app.origami_gallery_tile_hitboxes), 1)

        app.origami_identification_params.update({
            "rows": 1, "columns": 2, "spacing_x_nm": 10.0, "spacing_y_nm": 10.0,
            "site_mask_radius_nm": 2.0, "min_site_localizations": 5, "min_site_evidence": 0.1,
        })
        app._active_origami_picks_and_params = mock.Mock(return_value=(None, app.origami_identification_params))
        for name in ("alignment_overlay", "detected_sites_overlay", "site_diagnostics", "localization_group_assignments", "prominence_geometry"):
            setattr(app, "origami_show_" + name, FakeVariable(True))
        axis = Figure().subplots()
        app._draw_gallery_display_overlays(axis, result, 0, result.grid_points_nm, lambda points: points)
        self.assertTrue(any("one_group OFF" in label.get_text() for label in axis.texts))
        self.assertTrue(any(type(item).__name__ == "PolyCollection" for item in axis.collections))
        app.origami_show_text_statistics.set(False)
        axis = Figure().subplots()
        app._draw_gallery_display_overlays(axis, result, 0, result.grid_points_nm, lambda points: points)
        self.assertEqual(len(axis.texts), 0)
        self.assertTrue(any(type(item).__name__ == "PolyCollection" for item in axis.collections))

        app.origami_show_text_statistics.set(True)
        axis = Figure().subplots()
        app._draw_aligned_density_display_overlays(axis, result, result.grid_points_nm)
        self.assertTrue(any("one_group: ON 0.0%" in label.get_text() for label in axis.texts))
        self.assertTrue(any(type(item).__name__ == "PolyCollection" for item in axis.collections))
        app.origami_show_text_statistics.set(False)
        axis = Figure().subplots()
        app._draw_aligned_density_display_overlays(axis, result, result.grid_points_nm)
        self.assertEqual(len(axis.texts), 0)
        self.assertGreater(len(axis.collections), 0)
        for name in ("theoretical_overlay", "alignment_overlay", "detected_sites_overlay", "site_diagnostics", "localization_group_assignments", "prominence_geometry"):
            getattr(app, "origami_show_" + name).set(False)
        axis = Figure().subplots()
        app._draw_aligned_density_display_overlays(axis, result, result.grid_points_nm)
        self.assertEqual(len(axis.collections), 0)
        self.assertIsNone(axis.get_legend())

    def test_tile_progress_updates_step_bar_footer_and_running_label(self) -> None:
        app = SimpleNamespace(
            origami_identification_running=True,
            origami_step_progress={4: FakeVariable(100), 5: FakeVariable(0)},
            origami_step_progress_text={4: FakeVariable("Complete"), 5: FakeVariable("")},
            origami_staged_status=FakeVariable("Running Step 4"),
            origami_primary_help=mock.Mock(), status=FakeVariable("Old plot status"),
        )
        message = "Selected tile 5/10: Step 2 · shared alignment"
        PaintAnalysisApp._set_origami_step_progress(app, 5, 42, message)
        self.assertEqual(app.origami_step_progress[5].get(), 42)
        self.assertEqual(app.origami_step_progress[4].get(), 100)
        self.assertEqual(app.origami_step_progress_text[5].get(), message)
        self.assertEqual(app.status.get(), "Tile analysis: 42% — " + message)
        app.origami_primary_help.configure.assert_called_once_with(text=app.status.get())
        self.assertEqual(app.origami_staged_status.get(), "Running tile analysis · 42%")

    def test_gallery_display_toggles_rerender_current_view(self) -> None:
        for name in ("theoretical_overlay", "alignment_overlay", "detected_sites_overlay", "site_diagnostics", "localization_group_assignments", "prominence_geometry", "text_statistics"):
            for view in ("Individual origami gallery", "Individual site assignments", "Aligned density"):
                app = SimpleNamespace(origami_last_rendered_plot_option=view, render_origami_plot=mock.Mock())
                getattr(PaintAnalysisApp, "_toggle_origami_" + name)(app)
                app.render_origami_plot.assert_called_once()

    def test_symbolic_template_filenames_have_readable_display_names(self) -> None:
        self.assertEqual(custom_template_display_name("/tmp/:.png.png"), "/")
        self.assertEqual(custom_template_display_name("/tmp/:\\.png"), "\\")
        self.assertEqual(custom_template_display_name("/tmp/almost_full.png"), "almost_full")
        self.assertEqual(
            custom_template_display_name("/tmp/filesystem-safe.png", {"display_name": "/"}),
            "/",
        )

    def test_development_session_cache_restores_numeric_locs_and_invalidates_changed_source(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.hdf5"
            source.write_bytes(b"source-v1")
            raw = pd.DataFrame(
                {
                    "frame": np.asarray([0, 1], dtype=np.uint32),
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                }
            )
            corrected = raw.assign(x=[0.5, 1.5])
            drift = pd.DataFrame({"x": [0.5, 0.5], "y": [0.0, 0.0]})
            loaded = LoadedData(
                source,
                raw,
                [{"Frames": 2, "Pixelsize": 130.0}],
                {"source": "test"},
            )
            settings = {
                "method": "aim",
                "segmentation": 1000,
                "aim_intersect_nm": 20.0,
                "aim_roi_nm": 60.0,
                "rcc_lattice_pitch_nm": 0.0,
                "drift_file": None,
            }

            cache_path = save_development_session_cache(
                loaded,
                corrected,
                drift,
                "Picasso AIM test",
                settings,
                folder / "cache",
            )
            restored = load_development_session_cache(source, settings, folder / "cache")

            self.assertIsNotNone(cache_path)
            self.assertIsNotNone(restored)
            assert restored is not None
            pd.testing.assert_frame_equal(restored["loaded"].locs, raw)
            pd.testing.assert_frame_equal(restored["locs"], corrected)
            pd.testing.assert_frame_equal(restored["drift"], drift)
            self.assertEqual(restored["label"], "Picasso AIM test")

            map_result = {
                "result_type": "map",
                "image": np.arange(12, dtype=float).reshape(3, 4),
                "extent": (0.0, 40.0, 0.0, 30.0),
                "viewport_nm": None,
                "n_rendered": 2,
                "disp_px_size_nm": 10.0,
                "blur_method": "smooth",
            }
            assert cache_path is not None
            self.assertTrue(save_development_map_cache(cache_path, map_result))
            restored_with_map = load_development_session_cache(source, settings, folder / "cache")
            assert restored_with_map is not None
            np.testing.assert_array_equal(restored_with_map["cached_map"]["image"], map_result["image"])
            self.assertEqual(restored_with_map["cached_map"]["extent"], map_result["extent"])
            self.assertEqual(restored_with_map["cached_map"]["disp_px_size_nm"], 10.0)

            source.write_bytes(b"source-v2-is-different")
            self.assertIsNone(load_development_session_cache(source, settings, folder / "cache"))

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

    def test_colored_logical_groups_have_equal_alignment_intensity(self) -> None:
        metadata = {
            "format": "paint-analysis-origami-template-v1",
            "rows": 2,
            "columns": 2,
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 10.0,
            "margin_nm": 5.0,
            "width_nm": 20.0,
            "height_nm": 20.0,
            "width_px": 5,
            "height_px": 4,
            "logical_model": {
                "format": "paint-analysis-logical-bits-v1",
                "physical_rows": 2,
                "physical_columns": 2,
                "logical_bits": [
                    {"id": "red bit", "physical_sites": [[1, 1]], "display_color": "#ff0000"},
                    {"id": "green bit", "physical_sites": [[1, 2]], "display_color": "#00ff00"},
                ],
                "active_logical_bits": ["red bit", "green bit"],
            },
        }
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[0, 1] = (255, 0, 0)
        rgb[0, 3] = (0, 255, 0)
        png_metadata = PngImagePlugin.PngInfo()
        png_metadata.add_text("paint_analysis_template", json.dumps(metadata))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "colored_groups.png"
            Image.fromarray(rgb).save(path, pnginfo=png_metadata)
            loaded = load_custom_template_image(path)

        self.assertEqual(float(loaded[-1, 1]), float(loaded[-1, 3]))
        self.assertGreater(float(loaded[-1, 1]), 0.0)

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
        self.assertAlmostEqual(loaded["spacing_y_nm"], 5.0)
        self.assertAlmostEqual(loaded["height_nm"], 75.0)
        self.assertAlmostEqual(loaded["pixel_size_x_nm"], 160.0 / 499.0)
        self.assertAlmostEqual(loaded["pixel_size_y_nm"], 75.0 / 249.0)
        self.assertTrue(loaded["geometry_migrated_from_120x40"])

    def test_logical_stroke_metadata_is_normalized_to_lattice_indices(self) -> None:
        metadata = {
            "rows": 8,
            "columns": 12,
            "logical_model": {
                "format": "paint-analysis-logical-bits-v1",
                "physical_rows": 8,
                "physical_columns": 12,
                "alignment_groups": [
                    {"id": "alignment", "physical_sites": [[1, 1], [8, 12]]}
                ],
                "logical_bits": [
                    {"id": "left.slash", "physical_sites": [[2, 2], [3, 3]]},
                    {"id": "right.bottom", "physical_sites": [[8, 8], [8, 9]]},
                ],
                "active_logical_bits": ["left.slash"],
            },
        }
        model = logical_stroke_model_from_metadata(metadata)
        self.assertEqual(model["bit_ids"], ("left.slash", "right.bottom"))
        self.assertEqual(model["bit_cells"], ((62, 73), (7, 8)))
        self.assertEqual(model["alignment_cells"], (11, 84))
        self.assertEqual(model["active_bits"], (True, False))

    def test_loading_separate_shared_alignment_template_preserves_calibration(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._file_dialog_initial_dir = mock.Mock(return_value=Path("/tmp"))
        app._remember_file_dialog_dir = mock.Mock()
        app.origami_shared_alignment_template_name = FakeVariable("")
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
        image = np.zeros((250, 500), dtype=float)
        with (
            mock.patch("paint_analysis_gui.filedialog.askopenfilename", return_value="/tmp/L_L.png"),
            mock.patch("paint_analysis_gui.load_custom_template_image", return_value=image),
            mock.patch("paint_analysis_gui.load_custom_template_metadata", return_value=metadata),
        ):
            PaintAnalysisApp._load_origami_shared_alignment_template(app)

        self.assertEqual(app.origami_shared_alignment_template["name"], "L_L")
        self.assertEqual(app.origami_shared_alignment_template["rows"], 8)
        self.assertAlmostEqual(
            app.origami_shared_alignment_template["template_pixel_size_x_nm"],
            160.0 / 499.0,
        )
        self.assertIn("Shared alignment: L_L", app.origami_shared_alignment_template_name.get())

    def test_clearing_shared_alignment_restores_independent_fitting(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app.origami_shared_alignment_template = {"name": "L_L"}
        app.origami_shared_alignment_template_name = FakeVariable("Shared alignment: L_L")
        app.status = FakeVariable("")

        PaintAnalysisApp._clear_origami_shared_alignment_template(app)

        self.assertIsNone(app.origami_shared_alignment_template)
        self.assertEqual(
            app.origami_shared_alignment_template_name.get(),
            "No alignment template loaded",
        )
        self.assertIn("Load one", app.status.get())

    def test_loading_standalone_digital_pixel_json(self) -> None:
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._file_dialog_initial_dir = mock.Mock(return_value=Path("/tmp"))
        app._remember_file_dialog_dir = mock.Mock()
        app.origami_shared_alignment_template = {"rows": 2, "columns": 3}
        app.origami_custom_templates = []
        app.origami_digital_pixel_schema_name = FakeVariable("")
        app.status = FakeVariable("")
        schema = {
            "format": "paint-analysis-logical-bits-v1",
            "physical_rows": 2,
            "physical_columns": 3,
            "alignment_groups": [],
            "logical_bits": [
                {"id": "bit_a", "physical_sites": [[1, 1], [1, 2]]},
                {"id": "bit_b", "physical_sites": [[2, 2], [2, 3]]},
            ],
        }
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "digital_groups.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            with mock.patch(
                "paint_analysis_gui.filedialog.askopenfilename",
                return_value=str(path),
            ):
                PaintAnalysisApp._load_origami_digital_pixel_schema(app)

        self.assertEqual(app.origami_digital_pixel_model["bit_ids"], ("bit_a", "bit_b"))
        self.assertIn("2 digital groups", app.origami_digital_pixel_schema_name.get())

    def test_inspection_applies_correlation_gate_and_preserves_it_during_detection(self) -> None:
        empty = np.empty(0, dtype=float)
        site_count = 4
        picks = OrigamiPickResult(
            regions=[],
            aligned_regions=[],
            accepted_mask=np.empty(0, dtype=bool),
            point_counts=np.empty(0, dtype=int),
            original_point_counts=np.empty(0, dtype=int),
            crop_retained_fractions=empty,
            bounds_nm=np.empty((0, 4)),
            rectangle_corners_nm=np.empty((0, 4, 2)),
            rectangle_angles_deg=empty,
            rectangle_confidence=empty,
            site_gap_contrast=empty,
            on_site_fraction=empty,
            site_mask_radius_nm=1.0,
            template_points_nm=np.zeros((site_count, 2)),
            lattice_site_localization_counts=np.empty((0, site_count)),
            lattice_site_prominence=np.empty((0, site_count)),
            lattice_supported_sites=np.empty((0, site_count), dtype=bool),
            site_localization_counts=np.empty((0, site_count)),
            site_prominence=np.empty((0, site_count)),
            site_peak_positions_nm=np.empty((0, site_count, 2)),
            site_boundary_reference_positions_nm=np.empty((0, site_count, 2)),
            site_boundary_points_nm=np.empty((0, site_count, 32, 2)),
            site_centroids_nm=np.empty((0, site_count, 2)),
            supported_site_count=np.empty(0, dtype=int),
            supported_row_count=np.empty(0, dtype=int),
            supported_column_count=np.empty(0, dtype=int),
            site_spacing_rms_nm=empty,
            site_spacing_max_error_nm=empty,
            grid_vs_blob_delta_bic=empty,
            rectangle_matched_site_count=np.empty(0, dtype=int),
            rectangle_fit_rms_nm=empty,
            rectangle_width_nm=1.0,
            rectangle_height_nm=1.0,
            density_image=np.empty((0, 0)),
            density_contrast=np.empty((0, 0)),
            density_component_labels=np.empty((0, 0), dtype=int),
            density_extent_nm=(0.0, 1.0, 0.0, 1.0),
            density_threshold=0.1,
            alignment_pixel_nm=1.0,
            alignment_canvas_side_nm=1.0,
            alignment_reference_image=np.empty((0, 0)),
            alignment_candidate_images=np.empty((0, 0, 0)),
        )

        regions = [np.zeros((10, 2)) for _ in range(4)]
        picks = replace(
            picks, regions=regions, aligned_regions=regions,
            point_counts=np.full(4, 10),
            rectangle_confidence=np.asarray([0.35, 0.4, 0.47, np.nan]),
            accepted_mask=np.ones(4, dtype=bool),
        )
        params = {
            "_inspection_stage": 3, "_display_stage": 2,
            "source_path": Path("source.hdf5"),
            "min_rectangle_confidence": 0.4, "use_correlation_gate": True,
            "min_candidate_points": 1,
            "rows": 2, "columns": 2, "spacing_x_nm": 1.0, "spacing_y_nm": 1.0,
            "site_mask_radius_nm": 1.0, "min_site_localizations": 1,
            "min_site_evidence": 0.1, "min_supported_sites": 5,
        }
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._identify_origami_with_params = mock.Mock(return_value=picks)
        _, preview = PaintAnalysisApp._identify_origami_worker(
            app, np.empty((0, 2)), params, lambda *_args: None,
        )
        np.testing.assert_array_equal(preview["picks"].accepted_mask, [False, True, True, False])
        measured = PaintAnalysisApp._remeasure_origami_sites(
            app, preview["picks"], params, lambda *_args: None,
        )
        np.testing.assert_array_equal(measured.accepted_mask, [False, True, True, False])
        self.assertEqual(params["_alignment_accepted_mask"], (False, True, True, False))
        for change in ({"min_rectangle_confidence": 0.3}, {"use_correlation_gate": False}):
            updated = {**params, **change}
            self.assertFalse(origami_stage_cache_matches(params, updated, ORIGAMI_ALIGNMENT_CACHE_KEYS))
        disabled = PaintAnalysisApp._apply_origami_alignment_filters(
            picks, {**params, "use_correlation_gate": False},
        )
        np.testing.assert_array_equal(disabled.accepted_mask, [True, True, True, True])
        reasons = origami_candidate_failure_reasons(
            point_count=10, correlation=0.35, supported_sites=0,
            supported_rows=0, supported_columns=0, spacing_error_nm=float("inf"),
            params=params,
        )
        self.assertEqual(reasons, ["corr 0.35 < 0.4"])

    def test_shared_alignment_fits_once_and_preserves_each_digital_model(self) -> None:
        empty = np.empty(0, dtype=float)
        site_count = 4
        picks = OrigamiPickResult(
            regions=[],
            aligned_regions=[],
            accepted_mask=np.empty(0, dtype=bool),
            point_counts=np.empty(0, dtype=int),
            original_point_counts=np.empty(0, dtype=int),
            crop_retained_fractions=empty,
            bounds_nm=np.empty((0, 4)),
            rectangle_corners_nm=np.empty((0, 4, 2)),
            rectangle_angles_deg=empty,
            rectangle_confidence=empty,
            site_gap_contrast=empty,
            on_site_fraction=empty,
            site_mask_radius_nm=1.0,
            template_points_nm=np.zeros((site_count, 2)),
            lattice_site_localization_counts=np.empty((0, site_count)),
            lattice_site_prominence=np.empty((0, site_count)),
            lattice_supported_sites=np.empty((0, site_count), dtype=bool),
            site_localization_counts=np.empty((0, site_count)),
            site_prominence=np.empty((0, site_count)),
            site_peak_positions_nm=np.empty((0, site_count, 2)),
            site_boundary_reference_positions_nm=np.empty((0, site_count, 2)),
            site_boundary_points_nm=np.empty((0, site_count, 32, 2)),
            site_centroids_nm=np.empty((0, site_count, 2)),
            supported_site_count=np.empty(0, dtype=int),
            supported_row_count=np.empty(0, dtype=int),
            supported_column_count=np.empty(0, dtype=int),
            site_spacing_rms_nm=empty,
            site_spacing_max_error_nm=empty,
            grid_vs_blob_delta_bic=empty,
            rectangle_matched_site_count=np.empty(0, dtype=int),
            rectangle_fit_rms_nm=empty,
            rectangle_width_nm=1.0,
            rectangle_height_nm=1.0,
            density_image=np.empty((0, 0)),
            density_contrast=np.empty((0, 0)),
            density_component_labels=np.empty((0, 0), dtype=int),
            density_extent_nm=(0.0, 1.0, 0.0, 1.0),
            density_threshold=0.1,
            alignment_pixel_nm=1.0,
            alignment_canvas_side_nm=1.0,
            alignment_reference_image=np.empty((0, 0)),
            alignment_candidate_images=np.empty((0, 0, 0)),
        )

        def digital_template(name: str, active: bool) -> dict[str, object]:
            return {
                "name": name,
                "image": np.zeros((2, 2)),
                "rows": 2,
                "columns": 2,
                "spacing_x_nm": 1.0,
                "spacing_y_nm": 1.0,
                "rectangle_margin_nm": 1.0,
                "template_pixel_size_x_nm": 1.0,
                "template_pixel_size_y_nm": 1.0,
                "logical_model": {
                    "bit_ids": ("bit",),
                    "bit_cells": ((0,),),
                    "bit_physical_cells": ((0,),),
                    "alignment_cells": (),
                    "physical_shape": (2, 2),
                    "active_bits": (active,),
                },
            }

        templates = [digital_template("off", False), digital_template("on", True)]
        params = {
            "custom_templates": templates,
            "shared_alignment_template": {
                **digital_template("L_L", False),
                "name": "L_L",
            },
            "pick_bin_size_nm": 1.0,
            "connect_distance_nm": 1.0,
            "density_threshold": 0.1,
            "min_candidate_points": 1,
            "min_supported_sites": 5,
            "min_supported_rows": 2,
            "min_supported_columns": 2,
            "max_site_spacing_error_nm": 8.0,
            "min_monte_carlo_probability": 0.0,
            "use_cell_pattern_gate": False,
            "digital_pixel_model": templates[0]["logical_model"],
            "source_path": Path("source.hdf5"),
        }
        app = PaintAnalysisApp.__new__(PaintAnalysisApp)
        app._identify_origami_with_params = mock.Mock(return_value=picks)
        measured = PaintAnalysisApp._remeasure_origami_sites(
            app,
            replace(picks, template_points_nm=np.zeros((2, 2))),
            {
                "rows": 2,
                "columns": 2,
                "spacing_x_nm": 1.0,
                "spacing_y_nm": 1.0,
                "site_mask_radius_nm": 1.0,
                "min_site_localizations": 1,
                "min_site_evidence": 0.0,
                "digital_pixel_model": templates[0]["logical_model"],
            },
            lambda _percent, _message: None,
        )
        self.assertEqual(measured.template_points_nm.shape, (4, 2))
        self.assertEqual(measured.site_localization_counts.shape, (0, 4))
        app._remeasure_origami_sites = mock.Mock(return_value=measured)
        candidate_payload = (
            [],
            np.empty((0, 4)),
            np.empty((0, 0)),
            np.empty((0, 0), dtype=int),
            (0.0, 1.0, 0.0, 1.0),
            0.1,
        )
        with mock.patch(
            "paint_analysis_gui.pick_origami_candidates",
            return_value=candidate_payload,
        ):
            result_kind, payload = PaintAnalysisApp._identify_origami_worker(
                app,
                np.empty((0, 2)),
                params,
                lambda _percent, _message: None,
            )

        self.assertEqual(result_kind, "origami_multi_picks")
        self.assertEqual(app._identify_origami_with_params.call_count, 1)
        self.assertEqual(
            app._identify_origami_with_params.call_args.args[1]["_inspection_stage"],
            3,
        )
        alignment_params = app._identify_origami_with_params.call_args.args[1]
        self.assertEqual(alignment_params["min_supported_sites"], 0)
        self.assertEqual(alignment_params["min_supported_rows"], 0)
        self.assertEqual(alignment_params["min_supported_columns"], 0)
        self.assertEqual(alignment_params["max_site_spacing_error_nm"], float("inf"))
        app._remeasure_origami_sites.assert_called_once()
        self.assertEqual(
            [item["params"]["logical_model"]["active_bits"] for item in payload["templates"]],
            [(False,), (True,)],
        )

        with mock.patch("paint_analysis_gui.pick_origami_candidates", return_value=candidate_payload):
            single_kind, single_payload = PaintAnalysisApp._identify_origami_worker(
                app, np.empty((0, 2)), {**params, "custom_templates": templates[:1]},
                lambda _percent, _message: None,
            )
        self.assertEqual(single_kind, "origami_multi_picks")
        self.assertEqual(len(single_payload["templates"]), 1)

        app._identify_origami_with_params.reset_mock()
        app._remeasure_origami_sites.reset_mock()
        params["_inspection_stage"] = 3
        with mock.patch(
            "paint_analysis_gui.pick_origami_candidates",
            return_value=candidate_payload,
        ):
            preview_kind, preview_payload = PaintAnalysisApp._identify_origami_worker(
                app,
                np.empty((0, 2)),
                params,
                lambda _percent, _message: None,
            )

        self.assertEqual(preview_kind, "origami_stage_preview")
        self.assertEqual(preview_payload["stage"], 3)
        self.assertEqual(app._identify_origami_with_params.call_count, 1)
        app._remeasure_origami_sites.assert_not_called()

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

    def test_classification_bias_diagnostics_groups_best_rejections_and_gates(self) -> None:
        diagnostics = classification_bias_diagnostics(
            ["type_a", "type_b"],
            [8, 4],
            [
                {
                    "template_name": "type_a",
                    "failure_reasons": (
                        "corr 0.2 < 0.4",
                        "Monte Carlo template probability 0.3 < 0.5",
                        "cell-pattern correlation 0.04 < 0.30",
                    ),
                },
                {"template_name": "type_a", "failure_reasons": ("points 20 < 500",)},
                {"template_name": "type_b", "failure_reasons": ("sites 3 < 5",)},
            ],
            4,
        )

        np.testing.assert_array_equal(diagnostics["assigned"], [8, 4])
        np.testing.assert_array_equal(diagnostics["rejected_best"], [2, 1])
        np.testing.assert_allclose(diagnostics["assignment_rates"], [0.8, 0.8])
        self.assertEqual(int(diagnostics["failure_counts"][0, 0]), 1)
        self.assertEqual(int(diagnostics["failure_counts"][0, 1]), 1)
        self.assertEqual(int(diagnostics["failure_counts"][0, 2]), 1)
        self.assertEqual(int(diagnostics["failure_counts"][0, 3]), 1)
        self.assertEqual(int(diagnostics["failure_counts"][1, 4]), 1)
        self.assertEqual(diagnostics["unresolved_unclassified"], 1)

    def test_classification_diagnostics_plot_uses_recorded_best_template_failures(self) -> None:
        figure = Figure()
        app = SimpleNamespace(
            origami_multi_template_results={"type_a": {}, "type_b": {}},
            origami_multi_template_counts={"type_a": 8, "type_b": 4},
            origami_multi_template_unclassified_count=3,
            origami_multi_template_unclassified_details=[
                {"template_name": "type_a", "failure_reasons": ("corr 0.2 < 0.4",)},
                {"template_name": "type_a", "failure_reasons": ("points 20 < 500",)},
                {"template_name": "type_b", "failure_reasons": ("sites 3 < 5",)},
            ],
            origami_figure=figure,
            origami_canvas=SimpleNamespace(draw_idle=mock.Mock()),
            origami_toolbar=SimpleNamespace(update=mock.Mock()),
            _configure_origami_navigation_controls=mock.Mock(),
            notebook=SimpleNamespace(select=mock.Mock()),
            status=FakeVariable(""),
        )

        PaintAnalysisApp._plot_classification_diagnostics(app)

        self.assertEqual(len(figure.axes), 4)  # counts, margins, failure heatmap, colorbar
        self.assertEqual([patch.get_width() for patch in figure.axes[0].patches], [8, 4, 2, 1])
        self.assertIn("lowest best-template pass rate", app.status.get())

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
            origami_plot_option=FakeVariable("Aligned density"),
            origami_pick_result=None,
            origami_identification_params=None,
            origami_result=None,
            _plot_identified_origamis=mock.Mock(),
            render_origami_plot=mock.Mock(),
            _refresh_origami_action_states=mock.Mock(),
        )

        PaintAnalysisApp._on_origami_template_result_selection(app)

        self.assertIs(app.origami_pick_result, picks)
        self.assertIs(app.origami_result, result)
        self.assertEqual(app.origami_result_source, "source — type_b")
        self.assertEqual(app.origami_plot_option.get(), "Aligned density")
        app.render_origami_plot.assert_called_once_with()
        app._plot_identified_origamis.assert_not_called()

    def test_classification_selector_recovers_from_stale_disabled_match_state(self) -> None:
        combo = mock.Mock()
        selected = FakeVariable("square")
        app = SimpleNamespace(
            origami_template_result_combo=combo,
            origami_template_result_view=selected,
            origami_multi_template_results={"square": {}, "LL": {}},
        )

        PaintAnalysisApp._sync_origami_classification_selector_state(app)

        combo.configure.assert_called_once_with(values=("All templates", "square", "LL"))
        combo.state.assert_called_once_with(["!disabled", "readonly"])
        self.assertEqual(selected.get(), "square")

    def test_classification_selector_falls_back_when_selected_type_is_missing(self) -> None:
        combo = mock.Mock()
        selected = FakeVariable("old template")
        app = SimpleNamespace(
            origami_template_result_combo=combo,
            origami_template_result_view=selected,
            origami_multi_template_results={"square": {}, "LL": {}},
        )

        PaintAnalysisApp._sync_origami_classification_selector_state(app)

        self.assertEqual(selected.get(), "All templates")
        combo.state.assert_called_once_with(["!disabled", "readonly"])

    def test_selecting_all_templates_opens_combined_image_instead_of_histogram(self) -> None:
        retained_picks = object()
        retained_params = {"rows": 3}
        app = SimpleNamespace(
            origami_template_result_view=FakeVariable("All templates"),
            origami_multi_template_results={"type_a": {}, "type_b": {}},
            origami_plot_option=FakeVariable("Identified origami template matches"),
            origami_pick_result=retained_picks,
            origami_identification_params=retained_params,
            origami_result=object(),
            _plot_all_template_classifications=mock.Mock(),
            _plot_origami_type_counts=mock.Mock(),
            _refresh_origami_action_states=mock.Mock(),
        )

        PaintAnalysisApp._on_origami_template_result_selection(app)

        self.assertIs(app.origami_pick_result, retained_picks)
        self.assertIs(app.origami_identification_params, retained_params)
        self.assertEqual(app.origami_plot_option.get(), "Identified origami template matches")
        app._plot_all_template_classifications.assert_called_once_with()
        app._plot_origami_type_counts.assert_not_called()

    def test_multi_template_completion_enables_roi_tile_actions(self) -> None:
        buttons = [mock.Mock(), mock.Mock(), mock.Mock()]
        app = SimpleNamespace(
            origami_identification_running=True,
            status=FakeVariable("Running"),
            origami_pending_identification_snapshot=object(),
            origami_identify_button=mock.Mock(),
            origami_pick_result=None,
            origami_identification_params=None,
            origami_multi_template_counts={"type_a": 4, "type_b": 3},
            origami_multi_template_results={
                "type_a": {"picks": object(), "params": {"rows": 8}}
            },
            origami_loaded_roi_nm=(0.0, 100.0, 0.0, 80.0),
            origami_tiled_button=buttons[0],
            origami_n_tiles_button=buttons[1],
            origami_random_roi_button=buttons[2],
            origami_identification_progress_text=FakeVariable(""),
            _refresh_origami_action_states=mock.Mock(),
        )

        PaintAnalysisApp._finish_origami_identification_progress(app, "done")

        for button in buttons:
            button.state.assert_called_once_with(["!disabled"])
        self.assertFalse(app.origami_identification_running)
        self.assertEqual(app.origami_identification_progress_text.get(), "done")

    def test_all_template_view_keeps_spatial_plot_options_without_active_type(self) -> None:
        combo = mock.Mock()
        app = SimpleNamespace(
            origami_pick_result=None,
            origami_multi_template_results={"type_a": {}, "type_b": {}},
            origami_result=None,
            origami_all_plot_options=(
                "Coarse identification density",
                "Identified origami template matches",
                "Origami type counts",
                "Classification diagnostics",
                "Digital-group bias heatmap",
                "Digital-pixel spatial heatmap",
                "Individual origami gallery",
            ),
            origami_plot_combo=combo,
            origami_refine_button=mock.Mock(),
            origami_back_gallery_button=mock.Mock(),
        )

        PaintAnalysisApp._refresh_origami_action_states(app)

        combo.configure.assert_called_once_with(
            values=(
                "Identified origami template matches",
                "Origami type counts",
                "Classification diagnostics",
                "Digital-group bias heatmap",
                "Digital-pixel spatial heatmap",
            )
        )

    def test_selected_classification_exposes_views_before_overlay_ready(self):
        combo = mock.Mock()
        picks = SimpleNamespace(accepted_count=3)
        app = SimpleNamespace(
            origami_pick_result=picks,
            origami_multi_template_results={"full_align": {"picks": picks}},
            origami_template_result_view=SimpleNamespace(get=lambda: "full_align"),
            origami_result=None,
            origami_all_plot_options=("Coarse identification density", "Identified origami template matches",
                "Origami type counts", "Classification diagnostics", "Digital-group bias heatmap",
                "Digital-pixel spatial heatmap", "Individual origami gallery", "Aligned density"),
            origami_plot_combo=combo,
        )
        PaintAnalysisApp._refresh_origami_action_states(app)
        self.assertEqual(combo.configure.call_args.kwargs['values'], app.origami_all_plot_options)
        picks.accepted_count = 0
        PaintAnalysisApp._refresh_origami_action_states(app)
        self.assertNotIn("Aligned density", combo.configure.call_args.kwargs['values'])

    def test_selected_classification_plot_prepares_missing_overlay(self):
        app = SimpleNamespace(
            origami_plot_option=SimpleNamespace(get=lambda: "Aligned density"),
            origami_template_result_view=SimpleNamespace(get=lambda: "full_align"),
            origami_multi_template_results={"full_align": {"picks": SimpleNamespace(accepted_count=3)}},
            origami_result=None, origami_result_render_settings=None,
            origami_match_panel_combo=mock.Mock(), status=mock.Mock(),
            _auto_build_active_template_overlay=mock.Mock(),
        )
        with mock.patch("paint_analysis_gui.messagebox.showinfo") as info:
            PaintAnalysisApp.render_origami_plot(app)
        info.assert_not_called()
        app._auto_build_active_template_overlay.assert_called_once()

    def test_combined_classification_draws_each_type_and_unclassified_candidates(self) -> None:
        figure = Figure()
        axis = figure.add_subplot(111)
        axis.set_xlim(-100.0, 100.0)
        axis.set_ylim(-100.0, 100.0)

        def picks(center_x: float) -> SimpleNamespace:
            corners = np.asarray(
                [[center_x - 10.0, -5.0], [center_x + 10.0, -5.0],
                 [center_x + 10.0, 5.0], [center_x - 10.0, 5.0]]
            )
            return SimpleNamespace(
                bounds_nm=np.asarray([[center_x - 10.0, center_x + 10.0, -5.0, 5.0]]),
                accepted_mask=np.asarray([True]),
                rectangle_corners_nm=np.asarray([corners]),
                rectangle_width_nm=20.0,
                rectangle_height_nm=10.0,
                template_points_nm=np.asarray([[-5.0, 0.0], [5.0, 0.0]]),
                site_localization_counts=np.asarray([[3, 0]]),
                site_prominence=np.asarray([[0.5, 0.0]]),
                site_centroids_nm=np.asarray([[[-5.0, 0.5], [np.nan, np.nan]]]),
                site_peak_positions_nm=np.asarray([[[-5.0, 0.0], [np.nan, np.nan]]]),
                site_boundary_reference_positions_nm=np.asarray(
                    [[[-5.0, 2.0], [np.nan, np.nan]]]
                ),
                site_boundary_points_nm=np.asarray(
                    [[[[-6.0, -1.0], [-4.0, -1.0], [-4.0, 1.0], [-6.0, 1.0]],
                      [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]]]
                ),
                aligned_regions=[np.asarray([[-5.0, 0.0], [5.0, 0.0]])],
                point_counts=np.asarray([30]),
                rectangle_confidence=np.asarray([0.8]),
                supported_site_count=np.asarray([1]),
                site_spacing_max_error_nm=np.asarray([2.0]),
                site_gap_contrast=np.asarray([0.6]),
            )

        app = SimpleNamespace(
            origami_footprint_artists=[],
            origami_show_theoretical_overlay=FakeVariable(True),
            origami_show_detected_sites_overlay=FakeVariable(False),
            origami_show_site_diagnostics=FakeVariable(False),
            origami_show_localization_group_assignments=FakeVariable(False),
            origami_show_prominence_geometry=FakeVariable(False),
            origami_show_text_statistics=FakeVariable(False),
            origami_multi_template_results={
                "type_a": {
                    "picks": picks(-30.0),
                    "params": {
                        "alignment_template_image": np.eye(3),
                        "digital_pixel_model": {
                            "bit_ids": ("bit_a",),
                            "bit_physical_cells": ((0, 1),),
                            "physical_shape": (1, 2),
                        },
                        "digital_pixel_probabilities": ((0.75,),),
                        "spacing_x_nm": 10.0,
                        "spacing_y_nm": 10.0,
                        "site_mask_radius_nm": 2.0,
                    },
                },
                "type_b": {
                    "picks": picks(30.0),
                    "params": {"alignment_template_image": np.fliplr(np.eye(3))},
                },
            },
            origami_multi_template_counts={"type_a": 1, "type_b": 1},
            origami_multi_template_unclassified_count=1,
            origami_multi_template_unclassified_centers_nm=np.asarray([[0.0, 40.0]]),
            origami_multi_template_unclassified_details=[
                {
                    "center_nm": np.asarray([0.0, 40.0]),
                    "template_name": "type_b",
                    "point_count": 24,
                    "correlation": 0.35,
                    "supported_sites": 1,
                    "spacing_error_nm": 9.0,
                    "site_gap_contrast": 0.2,
                    "cell_match": 0.3,
                    "classification_score": 0.31,
                    "failure_reasons": ("corr 0.35 < 0.4", "sites 1 < 2"),
                }
            ],
        )

        shown, visible = PaintAnalysisApp._draw_all_template_classifications(app, axis)

        self.assertEqual((shown, visible), (2, 2))
        self.assertEqual(len(axis.collections), 5)
        np.testing.assert_allclose(axis.collections[1].get_offsets(), [[-35.0, 0.0], [-25.0, 0.0]])
        self.assertTrue(all(type(collection).__name__ == "PathCollection" for collection in axis.collections))
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        self.assertEqual(legend_labels, ["type_a (1)", "type_b (1)", "Unclassified (1)"])

        app.origami_show_text_statistics.set(True)
        PaintAnalysisApp._draw_all_template_classifications(app, axis)
        self.assertTrue(any("corr=" in text.get_text() for text in axis.texts))
        unclassified_labels = [
            text.get_text() for text in axis.texts if text.get_text().startswith("UNCLASSIFIED")
        ]
        self.assertEqual(len(unclassified_labels), 1)
        self.assertIn("best type_b", unclassified_labels[0])
        self.assertIn("FAIL: corr 0.35 < 0.4, sites 1 < 2", unclassified_labels[0])

        base_collection_count = len(axis.collections)
        app.origami_show_detected_sites_overlay.set(True)
        PaintAnalysisApp._draw_all_template_classifications(app, axis)
        self.assertGreater(len(axis.collections), base_collection_count)

        detected_collection_count = len(axis.collections)
        app.origami_show_site_diagnostics.set(True)
        app.origami_show_prominence_geometry.set(True)
        PaintAnalysisApp._draw_all_template_classifications(app, axis)
        self.assertGreater(len(axis.collections), detected_collection_count)
        self.assertTrue(any(text.get_text().startswith("bit_a ON") for text in axis.texts))
        self.assertTrue(any(type(collection).__name__ == "PolyCollection" for collection in axis.collections))

        app.origami_show_text_statistics.set(False)
        PaintAnalysisApp._draw_all_template_classifications(app, axis)
        self.assertFalse(any(text.get_text().startswith("bit_a ON") or "FAIL:" in text.get_text() for text in axis.texts))
        self.assertTrue(any(type(collection).__name__ == "PolyCollection" for collection in axis.collections))
        app.origami_show_text_statistics.set(True)

        app.origami_show_site_diagnostics.set(False)
        app.origami_show_localization_group_assignments.set(True)
        PaintAnalysisApp._draw_all_template_classifications(app, axis)
        self.assertFalse(any(type(collection).__name__ == "PolyCollection" for collection in axis.collections))
        self.assertTrue(any(
            len(collection.get_facecolors())
            and np.allclose(collection.get_facecolors()[0, :3], (0.12156863, 0.46666667, 0.70588235))
            for collection in axis.collections
        ))

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
                "Classification diagnostics",
                "Digital-group bias heatmap",
                "Digital-pixel spatial heatmap",
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
                "Classification diagnostics",
                "Digital-group bias heatmap",
                "Digital-pixel spatial heatmap",
                "Individual origami gallery",
                "Aligned density",
            )
        )
        combo.state.assert_called_once_with(["!disabled", "readonly"])

    def test_digital_group_bias_heatmap_uses_measured_candidate_probabilities(self) -> None:
        probabilities = ((0.9, 0.1), (0.8, 0.7), (0.2, 0.4))
        model = {"bit_ids": ("left", "right")}
        app = SimpleNamespace(
            origami_multi_template_results={
                "type_a": {
                    "picks": SimpleNamespace(
                        accepted_mask=np.asarray([True, False, False])
                    ),
                    "params": {
                        "digital_pixel_model": model,
                        "digital_pixel_probabilities": probabilities,
                        "digital_group_expected_on_fractions": (0.5, 0.25),
                    },
                },
                "type_b": {
                    "picks": SimpleNamespace(
                        accepted_mask=np.asarray([False, True, False])
                    ),
                    "params": {
                        "digital_pixel_model": model,
                        "digital_pixel_probabilities": probabilities,
                    },
                },
            },
            origami_figure=Figure(),
            origami_canvas=SimpleNamespace(draw_idle=mock.Mock()),
            origami_toolbar=SimpleNamespace(update=mock.Mock()),
            _configure_origami_navigation_controls=mock.Mock(),
            notebook=SimpleNamespace(select=mock.Mock()),
            status=SimpleNamespace(set=mock.Mock()),
        )

        PaintAnalysisApp._plot_digital_group_bias_heatmap(app)

        heatmap = np.asarray(app.origami_figure.axes[0].images[0].get_array())
        np.testing.assert_allclose(
            heatmap,
            np.asarray(
                [
                    [(0.9 + 0.8 + 0.2) / 3.0, 0.5, 0.9, 0.8, 0.2],
                    [(0.1 + 0.7 + 0.4) / 3.0, 0.25, 0.1, 0.7, 0.4],
                ]
            ),
        )
        labels = [text.get_text() for text in app.origami_figure.axes[0].texts]
        self.assertIn("p=0.63\n67% ON\nON=2/3", labels)
        self.assertIn("p=0.50\n50% ON\nexpected=1.5", labels)
        self.assertEqual(
            app.origami_last_rendered_plot_option,
            "Digital-group bias heatmap",
        )

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

    def test_origami_header_reflows_qc_display_controls_above_plot(self) -> None:
        header_widgets = {
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
        controls = tuple(mock.Mock() for _index in range(7))
        app = SimpleNamespace(
            **header_widgets,
            origami_qc_display_controls=controls,
            origami_qc_display_bar=mock.Mock(),
            origami_qc_display_note=mock.Mock(),
        )

        PaintAnalysisApp._layout_origami_plot_header(app, SimpleNamespace(width=800))

        for index, control in enumerate(controls):
            control.grid_configure.assert_called_once_with(
                row=1 + index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 12),
                pady=1,
            )
        app.origami_qc_display_note.grid_configure.assert_called_once_with(columnspan=2)

    def test_active_template_fast_overlay_is_built_lazily_only_once(self) -> None:
        app = SimpleNamespace(
            origami_template_result_view=FakeVariable("type_a"),
            origami_multi_template_results={"type_a": {}},
            origami_multi_template_overlay_results={},
            origami_pick_result=SimpleNamespace(accepted_count=2),
            overlay_origamis=mock.Mock(),
        )

        PaintAnalysisApp._auto_build_active_template_overlay(app)
        PaintAnalysisApp._auto_build_active_template_overlay(app)

        app.overlay_origamis.assert_called_once_with()
        self.assertEqual(app.origami_multi_template_overlays_building, {"type_a"})

        app.origami_multi_template_overlay_results["type_a"] = {"result": object()}
        app.origami_multi_template_overlays_building.clear()
        PaintAnalysisApp._auto_build_active_template_overlay(app)
        app.overlay_origamis.assert_called_once_with()

    def test_single_template_fast_overlay_is_rebuilt_once_after_identification(self) -> None:
        app = SimpleNamespace(
            origami_template_result_view=FakeVariable("All templates"),
            origami_multi_template_results={},
            origami_result=None,
            origami_pick_result=SimpleNamespace(accepted_count=3),
            origami_single_overlay_building=False,
            overlay_origamis=mock.Mock(),
        )

        PaintAnalysisApp._auto_build_active_template_overlay(app)
        PaintAnalysisApp._auto_build_active_template_overlay(app)

        app.overlay_origamis.assert_called_once_with()
        self.assertTrue(app.origami_single_overlay_building)

    def test_completed_fast_overlay_preserves_the_selected_plot_view(self) -> None:
        result = SimpleNamespace(
            origami_count=2,
            clustering_method="direct",
            cluster_site_indices=[np.asarray([0]), np.asarray([1])],
            alignment_rms_nm=np.asarray([1.0, 2.0]),
            grid_match_fraction=np.asarray([0.8, 0.9]),
        )
        app = SimpleNamespace(
            origami_template_result_view=FakeVariable("type_a"),
            origami_multi_template_results={"type_a": {}},
            origami_multi_template_overlay_results={},
            origami_multi_template_overlays_building={"type_a"},
            origami_plot_option=FakeVariable("Aligned density"),
            origami_gallery_page=FakeVariable(3),
            origami_density_cache_key=object(),
            origami_density_cache=object(),
            origami_last_rendered_plot_option="Individual origami gallery",
            render_origami_plot=mock.Mock(),
            status=FakeVariable(""),
        )
        payload = {
            "result": result,
            "source": "corrected ROI — type_a",
            "source_count": 200,
            "render_settings": {"pixel_size_nm": 0.5},
            "occupancy_threshold": 1,
            "template_name": "type_a",
        }

        PaintAnalysisApp._plot_origami_analysis(app, payload)

        self.assertEqual(app.origami_plot_option.get(), "Aligned density")
        app.render_origami_plot.assert_called_once_with()
        self.assertIs(app.origami_multi_template_overlay_results["type_a"]["result"], result)

    def test_stale_overlay_from_previous_identification_is_ignored(self) -> None:
        app = SimpleNamespace(origami_identification_generation=2)

        PaintAnalysisApp._plot_origami_analysis(
            app,
            {"identification_generation": 1},
        )

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
            aligned_regions=[np.repeat(grid, 5, axis=0), np.repeat(grid, 5, axis=0)],
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
                "classification_dispositions": (
                    "classified as type_a",
                    "classified as type_b",
                ),
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
        self.assertTrue(any(text.get_text() == "OTHER TYPE: type_b" for text in axis.texts))
        self.assertFalse(any("unclassified gate" in text.get_text() for text in axis.texts))

    def test_template_competition_draws_each_model_and_negative_space_conflicts(self) -> None:
        def make_picks(*, accepted: bool, supported: list[bool], counts: list[int]):
            return SimpleNamespace(
                regions=[np.asarray([[100.0, 200.0], [102.0, 198.0]])],
                alignment_candidate_images=np.asarray(
                    [[[0.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.0, 0.2, 0.0]]]
                ),
                alignment_canvas_side_nm=50.0,
                rectangle_width_nm=30.0,
                rectangle_height_nm=30.0,
                template_points_nm=np.asarray([[-5.0, -5.0], [5.0, 5.0]]),
                lattice_supported_sites=np.asarray([supported], dtype=bool),
                lattice_site_localization_counts=np.asarray([counts], dtype=float),
                accepted_mask=np.asarray([accepted], dtype=bool),
                rectangle_confidence=np.asarray([0.72]),
            )

        common_params = {
            "rows": 2,
            "columns": 2,
            "spacing_x_nm": 10.0,
            "spacing_y_nm": 10.0,
            "connect_distance_nm": 35.0,
            "classification_monte_carlo_posterior": (0.75,),
            "classification_scores": (1.2,),
            "classification_winner_margins": (0.4,),
            "classification_cell_pattern_correlation": (0.65,),
            "classification_cell_agreement": (0.70,),
        }
        first_params = {
            **common_params,
            "classification_template_probabilities": (0.8,),
            "classification_dispositions": ("classified as square",),
        }
        second_params = {
            **common_params,
            "classification_template_probabilities": (0.2,),
            "classification_dispositions": ("classified as square",),
        }
        app = SimpleNamespace(
            origami_multi_template_results={
                "square": {
                    "picks": make_picks(
                        accepted=True,
                        supported=[True, False, False, True],
                        counts=[8, 0, 0, 7],
                    ),
                    "params": first_params,
                },
                "full": {
                    "picks": make_picks(
                        accepted=False,
                        supported=[True, True, False, True],
                        counts=[8, 5, 0, 7],
                    ),
                    "params": second_params,
                },
            },
            origami_figure=Figure(),
            origami_match_panel_combo=SimpleNamespace(state=mock.Mock()),
            origami_match_roi=FakeVariable(0),
            origami_match_roi_label=FakeVariable(""),
            origami_canvas=SimpleNamespace(draw_idle=mock.Mock()),
            origami_toolbar=SimpleNamespace(update=mock.Mock()),
            _configure_origami_navigation_controls=mock.Mock(),
            notebook=SimpleNamespace(select=mock.Mock()),
            status=FakeVariable(""),
        )

        PaintAnalysisApp._plot_template_competition(
            app,
            app.origami_multi_template_results["square"]["picks"],
            0,
        )

        self.assertEqual(len(app.origami_figure.axes), 2)
        self.assertIn("class p 0.80", app.origami_figure.axes[0].get_title())
        self.assertIn("class p 0.20", app.origami_figure.axes[1].get_title())
        self.assertIn("pattern 0.65", app.origami_figure.axes[0].get_title())
        self.assertTrue(any(collection.get_label() == "detected in black space"
                            for collection in app.origami_figure.axes[1].collections))
        self.assertIn("largest equal-prior probability is square", app.status.get())

    def test_origami_acceptance_defaults_match_validated_examples(self) -> None:
        self.assertEqual(DEFAULT_ORIGAMI_PICK_BIN_NM, 10.0)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_DENSITY, 0.80)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_POINTS, 200)
        self.assertEqual(DEFAULT_ORIGAMI_MAX_POINTS, 3000)
        self.assertEqual(DEFAULT_ORIGAMI_MAX_SITE_SPACING_ERROR_NM, 8.0)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_MAX_PIXELS, 128)
        self.assertEqual(DEFAULT_ORIGAMI_ALIGNMENT_PASSES, 3)
        self.assertEqual(DEFAULT_ORIGAMI_CONNECT_DISTANCE_NM, 20.0)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_SITE_PROMINENCE, 0.10)
        self.assertEqual(DEFAULT_ORIGAMI_SITE_MASK_RADIUS_NM, 5.0)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_SITE_LOCALIZATIONS, 5)
        self.assertEqual(DEFAULT_ORIGAMI_CORRELATION_THRESHOLD, 0.30)
        self.assertEqual(DEFAULT_ORIGAMI_MIN_MONTE_CARLO_POSTERIOR, 0.25)
        self.assertTrue(DEFAULT_ORIGAMI_USE_CORRELATION_GATE)
        self.assertFalse(DEFAULT_ORIGAMI_USE_CELL_PATTERN_GATE)
        self.assertTrue(DEFAULT_ORIGAMI_SHOW_THEORETICAL_OVERLAY)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_DETECTED_SITES_OVERLAY)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_SITE_DIAGNOSTICS)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_LOCALIZATION_GROUP_ASSIGNMENTS)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_PROMINENCE_GEOMETRY)
        self.assertFalse(DEFAULT_ORIGAMI_SHOW_TEXT_STATISTICS)

    def test_step_three_preview_selects_digital_group_decisions(self) -> None:
        app = SimpleNamespace(
            origami_show_theoretical_overlay=FakeVariable(True),
            origami_show_alignment_overlay=FakeVariable(True),
            origami_show_detected_sites_overlay=FakeVariable(True),
            origami_show_site_diagnostics=FakeVariable(False),
            origami_show_localization_group_assignments=FakeVariable(True),
            origami_show_prominence_geometry=FakeVariable(True),
            origami_show_text_statistics=FakeVariable(True),
        )

        PaintAnalysisApp._set_origami_stage_preview_overlays(app, 3)

        self.assertFalse(app.origami_show_theoretical_overlay.get())
        self.assertTrue(app.origami_show_site_diagnostics.get())
        self.assertFalse(app.origami_show_localization_group_assignments.get())
        self.assertFalse(app.origami_show_alignment_overlay.get())
        self.assertFalse(app.origami_show_detected_sites_overlay.get())
        self.assertFalse(app.origami_show_prominence_geometry.get())
        self.assertFalse(app.origami_show_text_statistics.get())

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
            "Classification diagnostics": "none",
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
