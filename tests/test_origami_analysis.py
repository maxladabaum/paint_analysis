import unittest

import numpy as np

from origami_analysis import (
    _align_regions_by_image_correlation,
    align_picked_origamis,
    analyze_origami_regions,
    cluster_aligned_origami_sites,
    fit_picasso_g5m_components,
    ideal_grid_points,
    identify_origami_regions,
    integrate_rendered_density_at_sites,
    render_aligned_origami_density,
    render_localization_preview,
)


class OrigamiAnalysisTests(unittest.TestCase):
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
        self.assertLessEqual(picks.alignment_reference_image.shape[0], 64)
        self.assertEqual(progress_updates[-1][0], 100.0)
        self.assertTrue(any("cross-correlation pass" in message for _percent, message in progress_updates))
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
        aligned, _centers, _corners, _angles, correlations, pixel_nm, _reference = _align_regions_by_image_correlation(
            regions,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            requested_pixel_nm=0.25,
            iterations=2,
            template_points_nm=grid,
        )
        self.assertEqual(len(aligned), 15)
        self.assertTrue(np.all(correlations[:12] > 0.55))
        self.assertTrue(np.all(correlations[12:] < 0.55))
        self.assertGreaterEqual(pixel_nm, np.hypot(100.0, 80.0) / 64.0)

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
