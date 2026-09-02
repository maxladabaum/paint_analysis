import unittest
from unittest import mock

import numpy as np

from origami_analysis import (
    _align_regions_by_image_correlation,
    align_picked_origamis,
    analyze_origami_regions,
    cluster_aligned_origami_sites,
    fit_picasso_g5m_components,
    grid_vs_blob_delta_bic,
    ideal_grid_points,
    identify_origami_regions,
    integrate_rendered_density_at_sites,
    origami_gallery_indices,
    origami_gallery_page,
    render_aligned_origami_density,
    render_localization_preview,
    site_gap_contrast_for_regions,
    sparse_site_evidence,
    supported_grid_site_mask,
    supported_site_centroids,
    supported_site_spacing_errors,
)


class OrigamiAnalysisTests(unittest.TestCase):
    def test_sparse_site_support_is_not_penalized_for_unoccupied_sites(self) -> None:
        rng = np.random.default_rng(41)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        sparse_origami = np.vstack([rng.normal(grid[index], 1.5, size=(24, 2)) for index in occupied])
        counts, evidence = sparse_site_evidence(sparse_origami, grid, site_radius_nm=7.5)
        supported = supported_grid_site_mask(
            sparse_origami,
            grid,
            site_radius_nm=7.5,
            min_site_localizations=3,
            min_site_evidence=0.25,
        )

        np.testing.assert_array_equal(np.flatnonzero(supported), occupied)
        self.assertEqual(int(np.count_nonzero(supported)), 5)
        self.assertTrue(np.all(evidence[occupied] > 0.8))

    def test_site_prominence_accepts_a_broad_but_distinct_peak(self) -> None:
        rng = np.random.default_rng(4_102)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        regions = [rng.normal(grid[index], 1.5, size=(32, 2)) for index in occupied[:-1]]
        broad_site = rng.normal(grid[occupied[-1]], 5.0, size=(32, 2))
        regions.append(broad_site)

        counts, evidence = sparse_site_evidence(
            np.vstack(regions), grid, site_radius_nm=7.5
        )

        self.assertGreaterEqual(int(counts[occupied[-1]]), 5)
        self.assertGreaterEqual(float(evidence[occupied[-1]]), 0.25)

    def test_sparse_site_support_survives_diffuse_inter_site_background(self) -> None:
        rng = np.random.default_rng(4_103)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        sites = [rng.normal(grid[index], 1.5, size=(24, 2)) for index in occupied]
        diffuse_background = np.column_stack(
            (rng.uniform(-50.0, 50.0, 100), rng.uniform(-40.0, 40.0, 100))
        )
        supported = supported_grid_site_mask(
            np.vstack((*sites, diffuse_background)),
            grid,
            site_radius_nm=7.5,
            min_site_localizations=5,
            min_site_evidence=0.25,
        )

        np.testing.assert_array_equal(np.flatnonzero(supported), occupied)

    def test_site_prominence_does_not_penalize_a_dim_distinct_site(self) -> None:
        rng = np.random.default_rng(4_104)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        bright_sites = [rng.normal(grid[index], 1.5, size=(40, 2)) for index in occupied[:-1]]
        dim_site = rng.normal(grid[occupied[-1]], 1.5, size=(6, 2))

        counts, prominence = sparse_site_evidence(
            np.vstack((*bright_sites, dim_site)), grid, site_radius_nm=7.5
        )

        self.assertEqual(int(counts[occupied[-1]]), 6)
        self.assertGreaterEqual(float(prominence[occupied[-1]]), 0.25)

    def test_grid_vs_blob_bic_favors_sparse_sites_and_rejects_a_wide_blob(self) -> None:
        rng = np.random.default_rng(43)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        sparse_origami = np.vstack([rng.normal(grid[index], 2.0, size=(28, 2)) for index in occupied])
        wide_blob = rng.normal((0.0, 0.0), (16.0, 11.0), size=(len(sparse_origami), 2))

        sparse_score = grid_vs_blob_delta_bic(
            sparse_origami,
            grid,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            pixel_nm=2.0,
        )
        blob_score = grid_vs_blob_delta_bic(
            wide_blob,
            grid,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            pixel_nm=2.0,
        )

        self.assertGreater(sparse_score, 10.0)
        self.assertLess(blob_score, 10.0)

    def test_adaptive_grid_bic_accepts_broad_sites_independent_of_alignment_pixel(self) -> None:
        rng = np.random.default_rng(43_001)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        broad_grid = np.vstack([rng.normal(site, 5.5, size=(35, 2)) for site in grid])

        fine_alignment_score = grid_vs_blob_delta_bic(
            broad_grid,
            grid,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            pixel_nm=1.0,
        )
        coarse_alignment_score = grid_vs_blob_delta_bic(
            broad_grid,
            grid,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            pixel_nm=4.0,
        )

        self.assertGreater(fine_alignment_score, 0.0)
        self.assertAlmostEqual(fine_alignment_score, coarse_alignment_score, places=12)

    def test_negative_grid_blob_bic_is_qc_only_and_does_not_reject_a_candidate(self) -> None:
        rng = np.random.default_rng(10)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        points = np.vstack([rng.normal(site, 4.0, size=(12, 2)) for site in grid])
        points += np.asarray((250.0, 400.0))
        progress_updates: list[tuple[float, str]] = []

        picks = identify_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.03,
            min_candidate_points=80,
            max_candidate_points=250,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=20.0,
            min_rectangle_confidence=0.0,
            use_correlation_gate=False,
            site_mask_radius_nm=7.5,
            min_supported_sites=5,
            min_site_evidence=0.20,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
            max_site_spacing_error_nm=10.0,
            progress_callback=lambda percent, message: progress_updates.append((percent, message)),
        )

        self.assertEqual(picks.accepted_count, 1)
        self.assertLess(float(picks.grid_vs_blob_delta_bic[0]), 0.0)
        self.assertEqual(progress_updates[-1][0], 100.0)
        self.assertTrue(
            any("candidate 1/1" in message for _percent, message in progress_updates)
        )
        self.assertTrue(
            any("site prominence, spacing, and ΔBIC QC" in message for _percent, message in progress_updates)
        )
        self.assertTrue(
            all(left[0] <= right[0] for left, right in zip(progress_updates, progress_updates[1:]))
        )

    def test_site_prominence_recovers_dense_eleven_and_sparse_nine_site_origami(self) -> None:
        rng = np.random.default_rng(44)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)

        def candidate(occupied: np.ndarray, background_count: int) -> np.ndarray:
            peaks = np.vstack([rng.normal(grid[index], 1.8, size=(24, 2)) for index in occupied])
            background = np.column_stack(
                (
                    rng.uniform(-50.0, 50.0, size=background_count),
                    rng.uniform(-40.0, 40.0, size=background_count),
                )
            )
            return np.vstack((peaks, background))

        eleven_sites = np.arange(11, dtype=int)
        nine_sites = np.asarray([0, 1, 2, 4, 5, 6, 8, 9, 10])
        dense_left = candidate(eleven_sites, 180)
        sparse_right = candidate(nine_sites, 50)

        dense_mask = supported_grid_site_mask(
            dense_left,
            grid,
            site_radius_nm=7.5,
            min_site_localizations=5,
            min_site_evidence=0.25,
        )
        sparse_mask = supported_grid_site_mask(
            sparse_right,
            grid,
            site_radius_nm=7.5,
            min_site_localizations=5,
            min_site_evidence=0.25,
        )

        np.testing.assert_array_equal(np.flatnonzero(dense_mask), eleven_sites)
        np.testing.assert_array_equal(np.flatnonzero(sparse_mask), nine_sites)

    def test_supported_sites_must_retain_theoretical_grid_spacing(self) -> None:
        rng = np.random.default_rng(45)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        supported = np.zeros(len(grid), dtype=bool)
        supported[occupied] = True
        regular = np.vstack([rng.normal(grid[index], 0.6, size=(20, 2)) for index in occupied])
        distorted_grid = grid.copy()
        distorted_grid[11, 0] += 6.0
        distorted = np.vstack([rng.normal(distorted_grid[index], 0.6, size=(20, 2)) for index in occupied])

        regular_rms, regular_max = supported_site_spacing_errors(
            regular, grid, supported, site_radius_nm=7.5
        )
        distorted_rms, distorted_max = supported_site_spacing_errors(
            distorted, grid, supported, site_radius_nm=7.5
        )

        self.assertLess(regular_rms, 1.0)
        self.assertLess(regular_max, 1.0)
        self.assertGreater(distorted_rms, regular_rms)
        self.assertGreater(distorted_max, 5.0)

    def test_supported_site_centroids_show_the_measured_assignment_location(self) -> None:
        grid = ideal_grid_points(1, 2, 20.0, 20.0)
        points = np.asarray(
            [
                grid[0] + (-1.0, 2.0),
                grid[0] + (1.0, 2.0),
                grid[1] + (2.0, -1.0),
                grid[1] + (4.0, 1.0),
            ]
        )
        centroids = supported_site_centroids(
            points,
            grid,
            np.asarray([True, True]),
            site_radius_nm=7.5,
        )

        np.testing.assert_allclose(centroids[0], grid[0] + (0.0, 2.0))
        np.testing.assert_allclose(centroids[1], grid[1] + (3.0, 0.0))

    def test_site_spacing_error_is_an_independent_acceptance_gate(self) -> None:
        rng = np.random.default_rng(57)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        distorted_grid = grid.copy()
        distorted_grid[11, 0] += 7.0
        points = np.vstack([rng.normal(distorted_grid[index], 0.6, size=(32, 2)) for index in occupied])
        points += (250.0, 400.0)
        settings = dict(
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.05,
            min_candidate_points=100,
            max_candidate_points=300,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=20.0,
            min_rectangle_confidence=0.0,
            use_correlation_gate=False,
            site_mask_radius_nm=7.5,
            min_supported_sites=5,
            min_site_evidence=0.25,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
        )

        loose = identify_origami_regions(points, max_site_spacing_error_nm=100.0, **settings)
        strict = identify_origami_regions(points, max_site_spacing_error_nm=5.0, **settings)

        self.assertEqual(loose.accepted_count, 1)
        self.assertEqual(strict.accepted_count, 0)
        self.assertGreater(strict.site_spacing_max_error_nm[0], 5.0)

    def test_sparse_acceptance_does_not_require_full_template_correlation(self) -> None:
        rng = np.random.default_rng(47)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 3, 5, 8, 11])
        angle = np.deg2rad(37.0)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sparse_origami = np.vstack([rng.normal(grid[index], 1.4, size=(32, 2)) for index in occupied])
        sparse_origami = sparse_origami @ rotation.T + (250.0, 400.0)
        picks = identify_origami_regions(
            sparse_origami,
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.05,
            min_candidate_points=100,
            max_candidate_points=300,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=20.0,
            min_rectangle_confidence=0.99,
            use_correlation_gate=False,
            site_mask_radius_nm=7.5,
            min_supported_sites=5,
            min_site_evidence=0.25,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
            max_site_spacing_error_nm=5.0,
        )

        self.assertEqual(len(picks.regions), 1)
        self.assertEqual(picks.accepted_count, 1)
        self.assertEqual(picks.supported_site_count[0], 5)
        self.assertLess(picks.rectangle_confidence[0], 0.99)

        wide_blob = rng.normal((250.0, 400.0), (16.0, 11.0), size=(len(sparse_origami), 2))
        blob_picks = identify_origami_regions(
            wide_blob,
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.05,
            min_candidate_points=100,
            max_candidate_points=300,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=20.0,
            min_rectangle_confidence=0.0,
            use_correlation_gate=False,
            site_mask_radius_nm=7.5,
            min_supported_sites=5,
            min_site_evidence=0.25,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
            max_site_spacing_error_nm=5.0,
        )
        self.assertEqual(len(blob_picks.regions), 1)
        self.assertEqual(blob_picks.accepted_count, 0)
        self.assertLess(blob_picks.grid_vs_blob_delta_bic[0], 10.0)

    def test_site_gap_contrast_rejects_a_wide_blob(self) -> None:
        rng = np.random.default_rng(5)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        structured = np.vstack([rng.normal(site, 2.0, size=(30, 2)) for site in grid])
        wide_blob = rng.normal((0.0, 0.0), (16.0, 11.2), size=(360, 2))

        contrasts, on_site_fractions, area_fraction = site_gap_contrast_for_regions(
            [structured, wide_blob],
            grid,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            site_radius_nm=7.5,
        )

        self.assertGreater(contrasts[0], 0.9)
        self.assertLess(contrasts[1], 0.6)
        self.assertGreater(on_site_fractions[0], on_site_fractions[1])
        self.assertGreater(area_fraction, 0.0)
        self.assertLess(area_fraction, 1.0)

    def test_low_site_gap_contrast_is_qc_only_and_does_not_reject_a_candidate(self) -> None:
        rng = np.random.default_rng(10)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        signal = np.vstack([rng.normal(site, 6.0, size=(25, 2)) for site in grid])
        signal += np.asarray((250.0, 400.0))
        picks = identify_origami_regions(
            signal,
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.03,
            min_candidate_points=150,
            max_candidate_points=400,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=20.0,
            min_rectangle_confidence=0.0,
            use_correlation_gate=False,
            site_mask_radius_nm=7.5,
            min_supported_sites=5,
            min_site_evidence=0.20,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
            max_site_spacing_error_nm=10.0,
        )

        self.assertEqual(picks.accepted_count, 1)
        self.assertLess(float(picks.site_gap_contrast[0]), 0.30)

    def test_fast_overlay_assigns_sites_without_running_g5m(self) -> None:
        grid = ideal_grid_points(2, 2, 20.0, 20.0)
        points = np.vstack([np.repeat(site[None, :], 5, axis=0) for site in grid])

        with mock.patch("origami_analysis.fit_picasso_g5m_components", side_effect=AssertionError("G5M ran")):
            result = align_picked_origamis(
                [points],
                rows=2,
                columns=2,
                spacing_x_nm=20.0,
                spacing_y_nm=20.0,
                site_radius_nm=5.0,
                prealigned=True,
                use_g5m=False,
            )

        self.assertEqual(result.clustering_method, "Supported-site direct assignment")
        np.testing.assert_allclose(result.site_counts[0], [5.0, 5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result.cluster_site_indices[0], [0, 1, 2, 3])

    def test_fast_overlay_does_not_turn_single_background_points_into_occupied_sites(self) -> None:
        grid = ideal_grid_points(2, 2, 20.0, 20.0)
        points = np.vstack(
            (
                np.repeat(grid[0][None, :], 20, axis=0),
                np.repeat(grid[1][None, :], 15, axis=0),
                grid[2][None, :],
                grid[3][None, :],
            )
        )

        result = align_picked_origamis(
            [points],
            rows=2,
            columns=2,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            site_radius_nm=7.5,
            prealigned=True,
            use_g5m=False,
            direct_min_site_localizations=3,
            direct_min_site_evidence=0.25,
        )

        np.testing.assert_array_equal(result.cluster_site_indices[0], [0, 1])
        self.assertEqual(int(np.count_nonzero(result.site_counts[0])), 4)
        self.assertEqual(float(np.sum(result.site_occupancy[0])), 2.0)

    def test_integrated_density_per_site_uses_rendered_mean_image(self) -> None:
        grid = ideal_grid_points(1, 2, 10.0, 10.0)
        rendered = render_aligned_origami_density(
            [
                np.asarray([grid[0], grid[0], grid[1]]),
                np.asarray([grid[0]]),
            ],
            rows=1,
            columns=2,
            spacing_x_nm=10.0,
            spacing_y_nm=10.0,
            pixel_size_nm=1.0,
            padding_nm=5.0,
            blur_nm=0.0,
        )
        integrated = integrate_rendered_density_at_sites(rendered, grid, 1.0)
        np.testing.assert_allclose(integrated, [1.5, 0.5])

    def test_localization_preview_is_independent_of_pick_bin_resolution(self) -> None:
        points = np.asarray([[0.1, 0.1], [0.9, 0.9], [99.1, 79.1]])
        preview = render_localization_preview(points, pixel_size_nm=1.0, blur_nm=1.0)
        self.assertEqual(preview["contrast"].shape, (80, 100))
        self.assertAlmostEqual(preview["effective_pixel_x_nm"], 1.0)
        self.assertAlmostEqual(preview["effective_pixel_y_nm"], 1.0)

    def test_docking_site_occupancy_comes_from_per_origami_clusters(self) -> None:
        rng = np.random.default_rng(23)
        grid = np.asarray([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]])
        points = np.vstack(
            [
                rng.normal((0.0, 0.0), 0.7, size=(30, 2)),
                rng.normal((20.0, 0.0), 0.7, size=(18, 2)),
            ]
        )
        counts, labels, centers, sites = cluster_aligned_origami_sites(
            points,
            grid,
            g5m_sigma_min_nm=0.5,
            g5m_sigma_max_nm=3.0,
            g5m_min_locs=5,
            g5m_max_rounds_without_best_bic=3,
            site_match_radius_nm=5.0,
        )
        np.testing.assert_array_equal(counts, [30, 18, 0])
        np.testing.assert_array_equal(np.sort(sites), [0, 1])
        self.assertEqual(centers.shape, (2, 2))
        self.assertTrue(np.all(labels >= 0))

    def test_g5m_wrapper_matches_picasso_model_selection_core(self) -> None:
        from picasso import g5m as picasso_g5m

        rng = np.random.default_rng(24)
        points = np.vstack(
            [rng.normal((0.0, 0.0), 0.8, size=(30, 2)), rng.normal((20.0, 0.0), 0.8, size=(25, 2))]
        )
        labels, centers = fit_picasso_g5m_components(
            points,
            min_locs=5,
            sigma_min_nm=0.5,
            sigma_max_nm=3.0,
            max_rounds_without_best_bic=3,
        )
        model = picasso_g5m._find_optimal_G5M_2D(
            np.ascontiguousarray(points, dtype=np.float64),
            min_locs=5,
            sigma_bounds=(0.5, 3.0),
            lp=np.ones(len(points), dtype=np.float64),
            loc_prec_handle="abs",
            max_rounds_without_best_bic=3,
        )
        np.testing.assert_allclose(centers, model.means)
        np.testing.assert_array_equal(labels, model.predict(points))

    def test_cluster_centers_receive_continuous_rotational_refinement(self) -> None:
        rng = np.random.default_rng(51)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        regions = []
        for angle_degrees in (7.0, 23.0, 41.0, 68.0, 113.0):
            angle = np.deg2rad(angle_degrees)
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            signal = np.vstack([rng.normal(site, 0.8, size=(25, 2)) for site in grid])
            background = rng.uniform((-55.0, -45.0), (55.0, 45.0), size=(60, 2))
            regions.append(np.vstack([signal, background]) @ rotation.T + np.asarray([240.0, 180.0]))

        result = align_picked_origamis(
            regions,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            site_radius_nm=7.5,
            g5m_sigma_min_nm=0.5,
            g5m_sigma_max_nm=3.0,
            g5m_min_locs=10,
            g5m_max_rounds_without_best_bic=3,
        )
        residuals = []
        for centers, sites in zip(result.cluster_centers_nm, result.cluster_site_indices):
            residuals.extend(np.linalg.norm(centers - result.grid_points_nm[sites], axis=1))
        self.assertEqual(len(residuals), 5 * 12)
        self.assertLess(float(np.median(residuals)), 1.0)
        self.assertLess(float(np.max(residuals)), 7.5)

    def test_identification_uses_image_correlation_and_crops_background(self) -> None:
        rng = np.random.default_rng(71)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        angle_degrees = 31.0
        angle = np.deg2rad(angle_degrees)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.asarray([150.0, 120.0])
        signal = np.vstack([rng.normal(site, 0.8, size=(25, 2)) for site in grid]) @ rotation.T + center
        background = rng.uniform(center - (50.0, 45.0), center + (50.0, 45.0), size=(120, 2))
        points = np.vstack([signal, background])

        progress_updates: list[tuple[float, str]] = []
        picks = identify_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=25.0,
            density_threshold=0.05,
            min_candidate_points=250,
            max_candidate_points=380,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=8.0,
            g5m_sigma_min_nm=0.5,
            g5m_sigma_max_nm=3.0,
            g5m_min_locs=10,
            g5m_max_rounds_without_best_bic=3,
            site_match_radius_nm=7.5,
            progress_callback=lambda percent, message: progress_updates.append((percent, message)),
        )
        self.assertEqual(picks.accepted_count, 1)
        self.assertAlmostEqual(picks.rectangle_width_nm, 76.0)
        self.assertAlmostEqual(picks.rectangle_height_nm, 56.0)
        self.assertGreater(picks.rectangle_confidence[0], 0.8)
        self.assertLess(picks.point_counts[0], len(points))
        self.assertLessEqual(picks.alignment_reference_image.shape[0], 128)
        self.assertEqual(picks.alignment_candidate_images.shape[0], 1)
        candidate = picks.alignment_candidate_images[0]
        template = picks.alignment_reference_image
        displayed_score = float(
            np.sum(
                (candidate / np.linalg.norm(candidate))
                * (template / np.linalg.norm(template))
            )
        )
        self.assertAlmostEqual(displayed_score, float(picks.rectangle_confidence[0]), places=6)
        self.assertEqual(progress_updates[-1][0], 100.0)
        self.assertTrue(any("Rendering alignment thumbnails" in message for _percent, message in progress_updates))
        self.assertTrue(any("Theoretical-template alignment pass" in message for _percent, message in progress_updates))
        self.assertTrue(any("Applying fitted poses" in message for _percent, message in progress_updates))
        self.assertTrue(any("Measuring site-versus-gap density" in message for _percent, message in progress_updates))
        self.assertTrue(any("site prominence, spacing, and ΔBIC QC" in message for _percent, message in progress_updates))
        self.assertTrue(all(left[0] <= right[0] for left, right in zip(progress_updates, progress_updates[1:])))
        result = align_picked_origamis(
            picks.accepted_aligned_regions,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            site_radius_nm=7.5,
            g5m_sigma_min_nm=0.5,
            g5m_sigma_max_nm=3.0,
            g5m_min_locs=10,
            g5m_max_rounds_without_best_bic=3,
            prealigned=True,
        )
        residuals = np.linalg.norm(
            result.cluster_centers_nm[0] - result.grid_points_nm[result.cluster_site_indices[0]],
            axis=1,
        )
        self.assertGreaterEqual(len(residuals), 12)
        self.assertLessEqual(len(residuals), 14)
        self.assertLess(float(np.max(residuals)), 7.5)

    def test_image_classifier_rejects_non_origami_candidates(self) -> None:
        rng = np.random.default_rng(72)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        regions = []
        for _index in range(12):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            regions.append(np.vstack([rng.normal(site, 1.2, size=(20, 2)) for site in grid]) @ rotation.T)
        regions.extend([rng.uniform(-50.0, 50.0, size=(240, 2)) for _index in range(3)])
        scoring_images: list[np.ndarray] = []
        aligned, _centers, _corners, _angles, correlations, pixel_nm, _reference = _align_regions_by_image_correlation(
            regions,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            requested_pixel_nm=0.25,
            iterations=2,
            template_points_nm=grid,
            aligned_image_output=scoring_images,
        )
        self.assertEqual(len(aligned), 15)
        self.assertTrue(np.all(correlations[:12] > 0.55))
        self.assertTrue(np.all(correlations[12:] < 0.55))
        self.assertGreaterEqual(pixel_nm, np.hypot(100.0, 80.0) / 128.0)
        self.assertEqual(len(scoring_images), len(regions))
        displayed_score = float(
            np.sum(
                (scoring_images[0] / np.linalg.norm(scoring_images[0]))
                * (_reference / np.linalg.norm(_reference))
            )
        )
        self.assertAlmostEqual(displayed_score, float(correlations[0]), places=8)

    def test_partial_grid_uses_best_complete_pose_instead_of_strongest_polar_peak(self) -> None:
        rng = np.random.default_rng(990)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        expected_angle = 33.0
        angle = np.deg2rad(expected_angle)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # This asymmetric seven-site subset has a stronger polar-only peak near
        # 123 degrees, even though its complete image/template score is highest
        # at the physically correct 33-degree pose.
        region = np.vstack([rng.normal(grid[index], 1.0, size=(20, 2)) for index in range(7)]) @ rotation.T

        aligned, _centers, _corners, angles, correlations, _pixel_nm, _reference = (
            _align_regions_by_image_correlation(
                [region],
                rectangle_width_nm=100.0,
                rectangle_height_nm=80.0,
                requested_pixel_nm=2.0,
                iterations=2,
                template_points_nm=grid,
            )
        )

        angle_error = abs(((float(angles[0]) - expected_angle + 90.0) % 180.0) - 90.0)
        self.assertLess(angle_error, 2.0)
        self.assertGreater(float(correlations[0]), 0.68)
        self.assertEqual(len(aligned), 1)

    def test_theoretical_template_score_is_independent_of_roi_population(self) -> None:
        rng = np.random.default_rng(73)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        angle = np.deg2rad(37.0)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = np.vstack([rng.normal(site, 1.0, size=(20, 2)) for site in grid]) @ rotation.T
        distractors = [rng.uniform(-50.0, 50.0, size=(240, 2)) for _index in range(4)]
        settings = dict(
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            requested_pixel_nm=1.0,
            iterations=2,
            template_points_nm=grid,
        )

        alone = _align_regions_by_image_correlation([target], **settings)
        with_population = _align_regions_by_image_correlation([target, *distractors], **settings)

        self.assertAlmostEqual(float(alone[4][0]), float(with_population[4][0]), places=12)
        np.testing.assert_allclose(alone[0][0], with_population[0][0])

    def test_overlay_renderer_uses_grid_field_and_requested_resolution(self) -> None:
        points = [np.asarray([[-30.0, -20.0], [30.0, 20.0], [500.0, 500.0]])]
        rendered = render_aligned_origami_density(
            points,
            rows=3,
            columns=4,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            pixel_size_nm=0.5,
            padding_nm=20.0,
            blur_nm=1.0,
        )
        self.assertEqual(rendered["image"].shape, (160, 200))
        self.assertAlmostEqual(rendered["effective_pixel_x_nm"], 0.5)
        self.assertAlmostEqual(rendered["effective_pixel_y_nm"], 0.5)
        self.assertEqual(rendered["rendered_point_count"], 2)
        self.assertEqual(rendered["total_point_count"], 3)

    def test_streamed_symmetry_matches_explicit_orientations(self) -> None:
        points = [np.asarray([[-10.0, 0.0], [5.0, 7.0]]), np.asarray([[12.0, -3.0]])]
        settings = dict(
            rows=2,
            columns=2,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            pixel_size_nm=1.0,
            padding_nm=10.0,
            blur_nm=0.0,
        )
        streamed = render_aligned_origami_density(points, **settings, symmetrize_180=True, chunk_origamis=1)
        explicit = render_aligned_origami_density(
            [orientation for region in points for orientation in (region, -region)],
            **settings,
        )
        np.testing.assert_allclose(streamed["image"], explicit["image"])
        self.assertEqual(streamed["rendered_point_count"], explicit["rendered_point_count"])
        self.assertEqual(streamed["total_point_count"], explicit["total_point_count"])

    def test_gallery_filter_sort_and_page_are_stable(self) -> None:
        grid = ideal_grid_points(1, 2, 20.0, 20.0)
        regions = [np.repeat(grid, repeats, axis=0) for repeats in (1, 2, 3, 4)]
        result = align_picked_origamis(
            regions,
            rows=1,
            columns=2,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            site_radius_nm=5.0,
            prealigned=True,
            use_g5m=False,
        )
        result.grid_match_fraction[:] = [0.9, 0.4, 0.8, 0.7]
        indices = origami_gallery_indices(result, "Lowest grid match", min_grid_match_fraction=0.5)
        np.testing.assert_array_equal(indices, [3, 2, 0])
        page, page_number, page_count = origami_gallery_page(indices, page_number=2, page_size=2)
        np.testing.assert_array_equal(page, [0])
        self.assertEqual((page_number, page_count), (2, 2))

    def test_overlay_retains_identified_origami_with_low_grid_match(self) -> None:
        rng = np.random.default_rng(13)
        regions = [rng.normal((0.0, 0.0), 30.0, size=(100, 2)) for _ in range(3)]
        result = align_picked_origamis(
            regions,
            rows=3,
            columns=4,
            spacing_x_nm=15.0,
            spacing_y_nm=15.0,
            site_radius_nm=3.0,
        )
        self.assertEqual(result.origami_count, 3)
        self.assertTrue(np.all(result.grid_match_fraction < 0.5))

    def test_density_threshold_breaks_sparse_background_bridge(self) -> None:
        rng = np.random.default_rng(9)
        left = rng.normal((0.0, 0.0), 5.0, size=(500, 2))
        right = rng.normal((180.0, 0.0), 5.0, size=(500, 2))
        bridge = np.column_stack([np.linspace(25.0, 155.0, 25), rng.normal(0.0, 1.0, size=25)])
        background = rng.uniform((-40.0, -80.0), (220.0, 80.0), size=(150, 2))
        points = np.vstack([left, right, bridge, background])

        picks = identify_origami_regions(
            points,
            pick_bin_size_nm=10.0,
            connect_distance_nm=35.0,
            density_threshold=0.15,
            min_candidate_points=100,
            max_candidate_points=800,
        )

        self.assertEqual(picks.accepted_count, 2)
        self.assertTrue(np.all(np.sort(picks.point_counts[picks.accepted_mask]) > 450))
        self.assertEqual(picks.density_component_labels.shape, picks.density_contrast.shape)
        self.assertEqual(len(np.unique(picks.density_component_labels[picks.density_component_labels >= 0])), 2)
        self.assertTrue(
            np.all(
                picks.density_contrast[picks.density_component_labels >= 0]
                >= picks.density_threshold
            )
        )

    def test_rotated_origamis_preserve_a_shared_missing_site(self) -> None:
        rng = np.random.default_rng(42)
        grid = ideal_grid_points(3, 4, 15.0, 15.0)
        point_groups = []
        for index in range(24):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            center = np.asarray([(index % 6) * 180.0, (index // 6) * 180.0])
            for site_index, site in enumerate(grid):
                count = 0 if site_index == 2 else 12
                local = site + rng.normal(0.0, 1.2, size=(count, 2))
                point_groups.append(local @ rotation.T + center)

        points = np.vstack(point_groups)
        picks = identify_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=18.0,
            density_threshold=0.10,
            min_candidate_points=80,
            max_candidate_points=300,
        )
        self.assertEqual(len(picks.regions), 24)
        self.assertEqual(picks.accepted_count, 24)

        rejected_picks = identify_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=18.0,
            density_threshold=0.10,
            min_candidate_points=1000,
            max_candidate_points=2000,
        )
        self.assertEqual(len(rejected_picks.regions), 24)
        self.assertEqual(rejected_picks.accepted_count, 0)
        self.assertTrue(np.all(rejected_picks.point_counts == 132))

        result = analyze_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=18.0,
            density_threshold=0.10,
            min_candidate_points=80,
            max_candidate_points=300,
            rows=3,
            columns=4,
            spacing_x_nm=15.0,
            spacing_y_nm=15.0,
            site_radius_nm=6.0,
        )

        occupancy = np.mean(result.site_occupancy, axis=0)
        self.assertEqual(result.origami_count, 24)
        self.assertTrue(result.symmetrized_180)
        self.assertEqual(np.count_nonzero((occupancy > 0.4) & (occupancy < 0.6)), 2)
        self.assertGreaterEqual(np.count_nonzero(occupancy > 0.9), 10)
        np.testing.assert_allclose(result.site_counts[:, 2], result.site_counts[:, 9])


if __name__ == "__main__":
    unittest.main()
