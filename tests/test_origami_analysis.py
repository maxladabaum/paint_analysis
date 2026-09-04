import unittest
from unittest import mock

import numpy as np
import origami_analysis

from origami_analysis import (
    classify_template_candidates,
    _align_regions_by_image_correlation,
    align_picked_origamis,
    analyze_origami_regions,
    bidirectional_template_classification_scores,
    detected_lattice_template_agreement,
    cluster_aligned_origami_sites,
    custom_template_site_points,
    fit_picasso_g5m_components,
    grid_vs_blob_delta_bic,
    ideal_grid_points,
    identify_origami_regions,
    integrate_rendered_density_at_sites,
    lattice_template_on_site_fractions,
    lattice_template_empty_cell_fractions,
    lattice_count_template_probability_agreement,
    logical_stroke_template_evidence,
    lattice_template_probability_agreement,
    monte_carlo_template_evidence,
    origami_gallery_indices,
    origami_gallery_page,
    pick_origami_candidates,
    prepare_custom_alignment_template,
    render_aligned_origami_density,
    render_localization_preview,
    site_gap_contrast_for_regions,
    sparse_site_evidence,
    sparse_site_evidence_diagnostics,
    supported_grid_site_mask,
    supported_site_centroids,
    supported_site_spacing_errors,
)


class OrigamiAnalysisTests(unittest.TestCase):
    def test_logical_stroke_evidence_aggregates_physical_sites_per_bit(self) -> None:
        bit_cells = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14))
        counts = np.zeros((1, 20), dtype=float)
        counts[0, [0, 1, 2, 3]] = 12.0  # One missing extension on active bit 0.
        counts[0, [10, 11, 12, 13, 14]] = 10.0
        correct_p, correct_score, bit_p, _pattern = logical_stroke_template_evidence(
            counts, bit_cells, (True, False, True)
        )
        wrong_p, wrong_score, _wrong_bits, _wrong_pattern = logical_stroke_template_evidence(
            counts, bit_cells, (False, True, False)
        )
        self.assertGreater(bit_p[0, 0], 0.5)
        self.assertGreater(bit_p[0, 2], 0.5)
        self.assertGreater(correct_score[0], wrong_score[0])
        self.assertGreater(correct_p[0], 0.5)

    def test_overlapping_logical_bits_use_physical_union_semantics(self) -> None:
        # Cell 1 belongs to both bits. When A is ON and B is OFF, brightness at
        # that shared cell is explained by A and must not count against B.
        counts = np.zeros((1, 4), dtype=float)
        counts[0, [0, 1]] = 12.0
        bit_cells = ((0, 1), (1, 2))
        correct_p, correct_score, _bits, _pattern = logical_stroke_template_evidence(
            counts, bit_cells, (True, False)
        )
        wrong_p, wrong_score, _wrong_bits, _wrong_pattern = logical_stroke_template_evidence(
            counts, bit_cells, (True, True)
        )
        self.assertGreater(correct_score[0], wrong_score[0])
        self.assertGreater(correct_p[0], wrong_p[0])

    def test_candidate_detection_is_stable_in_presence_of_one_bright_aggregate(self) -> None:
        rng = np.random.default_rng(4)
        ordinary = np.vstack(
            [rng.normal((x, 0.0), 3.0, size=(150, 2)) for x in np.arange(0.0, 800.0, 100.0)]
        )
        aggregate = rng.normal((1000.0, 0.0), 3.0, size=(3000, 2))
        baseline = pick_origami_candidates(
            ordinary,
            bin_size_nm=5.0,
            connect_distance_nm=20.0,
            density_threshold=0.1,
        )[0]
        contaminated = pick_origami_candidates(
            np.vstack((ordinary, aggregate)),
            bin_size_nm=5.0,
            connect_distance_nm=20.0,
            density_threshold=0.1,
        )[0]

        self.assertEqual(len(baseline), 8)
        self.assertEqual(sum(len(region) == 150 for region in contaminated), 8)

    def test_candidate_point_minimum_is_applied_before_alignment(self) -> None:
        rng = np.random.default_rng(405)
        small = rng.normal((0.0, 0.0), 2.0, size=(40, 2))
        viable = rng.normal((150.0, 0.0), 2.0, size=(120, 2))
        regions, _density, _contrast, _extent, labels = pick_origami_candidates(
            np.vstack((small, viable)),
            bin_size_nm=5.0,
            connect_distance_nm=20.0,
            density_threshold=0.1,
            minimum_points=100,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0]), 120)
        self.assertEqual(set(np.unique(labels)), {-1, 0})

    def test_alignment_cache_reuses_candidate_images_and_polar_transforms(self) -> None:
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(909)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        region = np.vstack([rng.normal(site, 1.2, size=(12, 2)) for site in grid])
        regions = [region]
        templates: list[np.ndarray] = []
        for omitted in (0, 11):
            image = np.zeros((80, 100), dtype=float)
            for x_nm, y_nm in np.delete(grid, omitted, axis=0):
                column = int(round((x_nm + 50.0) / 100.0 * 99.0))
                row = int(round((y_nm + 40.0) / 80.0 * 79.0))
                image[row, column] = 1.0
            templates.append(gaussian_filter(image, 2.0))

        cache: dict[tuple[object, ...], tuple[object, ...]] = {}
        with (
            mock.patch(
                "origami_analysis._render_candidate_image",
                wraps=origami_analysis._render_candidate_image,
            ) as render_mock,
            mock.patch(
                "origami_analysis._polar_image",
                wraps=origami_analysis._polar_image,
            ) as polar_mock,
        ):
            _align_regions_by_image_correlation(
                regions,
                rectangle_width_nm=100.0,
                rectangle_height_nm=80.0,
                requested_pixel_nm=1.0,
                iterations=1,
                template_points_nm=grid,
                template_image=templates[0],
                sparse_pose_site_count=3,
                candidate_image_cache=cache,
            )
            _align_regions_by_image_correlation(
                regions,
                rectangle_width_nm=100.0,
                rectangle_height_nm=80.0,
                requested_pixel_nm=1.0,
                iterations=1,
                template_points_nm=grid,
                template_image=templates[1],
                sparse_pose_site_count=3,
                candidate_image_cache=cache,
            )
            # The base region thumbnail is rendered once. Additional renders
            # are template-specific refined pose trials centered at zero.
            self.assertEqual(
                sum(call.args[0] is region for call in render_mock.call_args_list),
                1,
            )
            # Polar conversion runs once for that candidate plus once for each
            # distinct template.
            self.assertEqual(polar_mock.call_count, 3)

    def test_bidirectional_classification_penalizes_dense_superset_templates(self) -> None:
        sparse_scores = bidirectional_template_classification_scores(
            # The dense template has a much higher raw correlation for the
            # sparse object, while the sparse template has a higher raw
            # correlation for the full object. Bidirectional evidence must
            # overcome both one-sided correlation mistakes.
            np.asarray([0.45, 0.95]),
            np.asarray([0.90, 0.45]),
            np.asarray([5, 5]),
            5,
        )
        full_scores = bidirectional_template_classification_scores(
            np.asarray([0.99, 0.70]),
            np.asarray([0.95, 0.90]),
            np.asarray([5, 10]),
            10,
        )
        classification = classify_template_candidates(
            [np.asarray([[0.0, 0.0], [100.0, 0.0]]), np.asarray([[0.0, 0.0], [100.0, 0.0]])],
            [np.asarray([True, True]), np.asarray([True, True])],
            [sparse_scores, full_scores],
            match_distance_nm=5.0,
        )

        np.testing.assert_array_equal(classification.counts, [1, 1])
        self.assertGreater(sparse_scores[0], full_scores[0])
        self.assertGreater(full_scores[1], sparse_scores[1])

    def test_classification_balances_observed_bright_and_black_evidence(self) -> None:
        sparse_score = bidirectional_template_classification_scores(
            np.asarray([0.5]),
            np.asarray([0.75]),
            np.asarray([13]),
            26,
            off_site_empty_fractions=np.asarray([0.80]),
        )[0]
        dense_score = bidirectional_template_classification_scores(
            np.asarray([0.5]),
            np.asarray([0.90]),
            np.asarray([32]),
            64,
            off_site_empty_fractions=np.asarray([0.80]),
        )[0]

        expected_sparse = np.cbrt(0.75 * 0.50 * 0.80) * (0.75 + 0.25 * 0.5)
        expected_dense = np.cbrt(0.90 * 0.50 * 0.80) * (0.75 + 0.25 * 0.5)
        self.assertAlmostEqual(sparse_score, expected_sparse, places=12)
        self.assertAlmostEqual(dense_score, expected_dense, places=12)
        self.assertGreater(dense_score, sparse_score)

    def test_classification_prefers_the_correct_spatial_cell_pattern(self) -> None:
        correct = bidirectional_template_classification_scores(
            np.asarray([0.5]),
            np.asarray([0.75]),
            np.asarray([12]),
            24,
            off_site_empty_fractions=np.asarray([0.75]),
            bright_site_probabilities=np.asarray([0.75]),
            cell_pattern_correlations=np.asarray([0.80]),
        )[0]
        wrong = bidirectional_template_classification_scores(
            np.asarray([0.5]),
            np.asarray([0.75]),
            np.asarray([12]),
            24,
            off_site_empty_fractions=np.asarray([0.75]),
            bright_site_probabilities=np.asarray([0.75]),
            cell_pattern_correlations=np.asarray([0.20]),
        )[0]

        self.assertGreater(correct, wrong * 1.5)

    def test_detected_site_mask_prefers_matching_template_layout(self) -> None:
        grid = ideal_grid_points(3, 4, 10.0, 8.0)
        square_cells = np.asarray([0, 1, 2, 3, 4, 7, 8, 9, 10, 11])
        diagonal_cells = np.asarray([0, 1, 2, 3, 5, 6, 8, 11])
        detected = np.zeros((1, 12), dtype=bool)
        detected[0, square_cells] = True

        square_bright, square_dark, square_pattern = detected_lattice_template_agreement(
            detected,
            grid[square_cells],
            rows=3,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=8.0,
        )
        diagonal_bright, diagonal_dark, diagonal_pattern = detected_lattice_template_agreement(
            detected,
            grid[diagonal_cells],
            rows=3,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=8.0,
        )

        self.assertAlmostEqual(square_bright[0], 1.0)
        self.assertAlmostEqual(square_dark[0], 1.0)
        self.assertAlmostEqual(square_pattern[0], 1.0)
        self.assertGreater(square_pattern[0], diagonal_pattern[0] + 0.50)
        self.assertGreater(square_bright[0], diagonal_bright[0])
        self.assertGreater(square_dark[0], diagonal_dark[0])

    def test_monte_carlo_evidence_recovers_template_and_rejects_blob(self) -> None:
        rows, columns = 6, 8
        grid = ideal_grid_points(rows, columns, 10.0, 7.0)
        row_ids = np.arange(rows * columns) // columns
        column_ids = np.arange(rows * columns) % columns
        square_cells = np.flatnonzero(
            (row_ids == 0) | (row_ids == rows - 1) | (column_ids == 0) | (column_ids == columns - 1)
        )
        diagonal_cells = np.flatnonzero(
            (column_ids == row_ids) | (column_ids == columns - row_ids - 1)
        )
        observed_square = np.zeros((1, rows * columns), dtype=bool)
        observed_square[0, square_cells] = True

        square_posterior, square_evidence = monte_carlo_template_evidence(
            observed_square,
            grid[square_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=7.0,
        )
        _diagonal_posterior, diagonal_evidence = monte_carlo_template_evidence(
            observed_square,
            grid[diagonal_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=7.0,
        )
        blob = np.sum(np.square(grid / np.asarray([18.0, 12.0])), axis=1) < 1.0
        blob_posterior, _blob_evidence = monte_carlo_template_evidence(
            blob[None, :],
            grid[square_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=7.0,
        )

        self.assertGreater(square_posterior[0], 0.95)
        self.assertGreater(square_evidence[0], diagonal_evidence[0] + 5.0)
        self.assertLess(blob_posterior[0], 0.70)

    def test_monte_carlo_evidence_tolerates_dropout_without_zero_probability(self) -> None:
        rows, columns = 8, 12
        grid = ideal_grid_points(rows, columns, 10.0, 6.0)
        row_ids = np.arange(rows * columns) // columns
        column_ids = np.arange(rows * columns) % columns
        square_cells = np.flatnonzero(
            (row_ids == 0)
            | (row_ids == rows - 1)
            | (column_ids == 0)
            | (column_ids == columns - 1)
        )
        observed = np.zeros((2, rows * columns), dtype=bool)
        # Only about one third of the expected sites are detected in the first
        # footprint, as commonly happens with sparse localization data.
        observed[0, square_cells[::3]] = True

        posterior, evidence = monte_carlo_template_evidence(
            observed,
            grid[square_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=6.0,
        )

        self.assertGreater(posterior[0], 0.50)
        self.assertTrue(np.isfinite(evidence[0]))
        self.assertEqual(posterior[1], 0.0)
        self.assertLess(evidence[1], 0.0)

    def test_monte_carlo_evidence_uses_graded_localization_counts(self) -> None:
        rows, columns = 6, 8
        grid = ideal_grid_points(rows, columns, 10.0, 7.0)
        row_ids = np.arange(rows * columns) // columns
        column_ids = np.arange(rows * columns) % columns
        square_cells = np.flatnonzero(
            (row_ids == 0)
            | (row_ids == rows - 1)
            | (column_ids == 0)
            | (column_ids == columns - 1)
        )
        diagonal_cells = np.flatnonzero(
            (column_ids == row_ids) | (column_ids == columns - row_ids - 1)
        )
        counts = np.ones((1, rows * columns), dtype=float)
        counts[0, square_cells] = np.linspace(8.0, 40.0, len(square_cells))

        square_probability, square_evidence = monte_carlo_template_evidence(
            counts,
            grid[square_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=7.0,
        )
        _diagonal_probability, diagonal_evidence = monte_carlo_template_evidence(
            counts,
            grid[diagonal_cells],
            rows=rows,
            columns=columns,
            spacing_x_nm=10.0,
            spacing_y_nm=7.0,
        )

        self.assertGreater(square_probability[0], 0.90)
        self.assertGreater(square_evidence[0], diagonal_evidence[0])

    def test_equal_prior_cell_probabilities_recover_bright_and_dark_pattern(self) -> None:
        grid = ideal_grid_points(2, 4, 10.0, 10.0)
        occupied = np.asarray([0, 2, 5, 7])
        rng = np.random.default_rng(31)
        region = np.vstack([rng.normal(grid[cell], 0.4, size=(8, 2)) for cell in occupied])

        bright, dark, on_site, pattern = lattice_template_probability_agreement(
            [region],
            grid[occupied],
            rows=2,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=10.0,
        )
        wrong_bright, wrong_dark, wrong_on_site, wrong_pattern = lattice_template_probability_agreement(
            [region],
            grid[[0, 1, 4, 7]],
            rows=2,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=10.0,
        )

        self.assertGreater(bright[0], 0.95)
        self.assertGreater(dark[0], 0.95)
        self.assertAlmostEqual(on_site[0], 1.0)
        self.assertAlmostEqual(wrong_on_site[0], 0.5)
        self.assertGreater(pattern[0], 0.95)
        self.assertGreater(pattern[0], wrong_pattern[0] + 0.50)
        self.assertGreater(np.sqrt(bright[0] * dark[0]), np.sqrt(wrong_bright[0] * wrong_dark[0]) + 0.30)

    def test_continuous_count_agreement_uses_graded_cell_evidence(self) -> None:
        grid = ideal_grid_points(2, 4, 10.0, 10.0)
        occupied = np.asarray([0, 2, 5, 7])
        counts = np.asarray([[12, 0, 8, 1, 0, 10, 0, 7]], dtype=float)
        bright, dark, pattern = lattice_count_template_probability_agreement(
            counts,
            grid[occupied],
            rows=2,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=10.0,
        )
        wrong_bright, wrong_dark, wrong_pattern = lattice_count_template_probability_agreement(
            counts,
            grid[[0, 1, 4, 7]],
            rows=2,
            columns=4,
            spacing_x_nm=10.0,
            spacing_y_nm=10.0,
        )

        self.assertGreater(bright[0], wrong_bright[0])
        self.assertGreater(dark[0], wrong_dark[0])
        self.assertGreater(pattern[0], wrong_pattern[0] + 0.5)

    def test_lattice_classification_distinguishes_neighboring_black_cells(self) -> None:
        rng = np.random.default_rng(202_609_03)
        full_grid = ideal_grid_points(8, 12, 10.9090909091, 5.7142857143)
        almost_full_cells = np.asarray(
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 18, 19, 24, 26, 29, 30, 32, 35, 36, 39,
                40, 42, 45, 46, 48, 51, 52, 54, 57, 58, 60, 62,
                65, 66, 68, 71, 72, 73, 74, 75, 76, 77,
            ]
        )
        square_cells = np.asarray(
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 17, 18, 23, 24, 29, 30, 35, 36, 41, 42, 47,
                48, 53, 54, 59, 60, 65, 66, 71, 72, 77, 78, 83,
                84, 89, 90, 95, 85, 86, 87, 88, 91, 92, 93, 94,
            ]
        )
        square_region = np.vstack(
            [rng.normal(full_grid[cell], 1.0, size=(12, 2)) for cell in square_cells]
        )

        almost_precision = lattice_template_on_site_fractions(
            [square_region],
            full_grid[almost_full_cells],
            rows=8,
            columns=12,
            spacing_x_nm=10.9090909091,
            spacing_y_nm=5.7142857143,
        )[0]
        square_precision = lattice_template_on_site_fractions(
            [square_region],
            full_grid[square_cells],
            rows=8,
            columns=12,
            spacing_x_nm=10.9090909091,
            spacing_y_nm=5.7142857143,
        )[0]

        self.assertGreater(square_precision, 0.90)
        self.assertLess(almost_precision, 0.75)
        self.assertGreater(square_precision, almost_precision + 0.20)

        almost_black_empty = lattice_template_empty_cell_fractions(
            [square_region],
            full_grid[almost_full_cells],
            rows=8,
            columns=12,
            spacing_x_nm=10.9090909091,
            spacing_y_nm=5.7142857143,
            min_site_localizations=3,
        )[0]
        square_black_empty = lattice_template_empty_cell_fractions(
            [square_region],
            full_grid[square_cells],
            rows=8,
            columns=12,
            spacing_x_nm=10.9090909091,
            spacing_y_nm=5.7142857143,
            min_site_localizations=3,
        )[0]

        self.assertGreater(square_black_empty, almost_black_empty + 0.20)

    def test_multi_template_classification_assigns_each_object_once(self) -> None:
        classification = classify_template_candidates(
            [
                np.asarray([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]),
                np.asarray([[1.0, -1.0], [101.0, 1.0], [201.0, 0.0]]),
            ],
            [np.asarray([True, True, False]), np.asarray([True, True, False])],
            [np.asarray([0.8, 0.45, 0.2]), np.asarray([0.5, 0.9, 0.3])],
            match_distance_nm=10.0,
        )

        np.testing.assert_array_equal(classification.assignment_masks[0], [True, False, False])
        np.testing.assert_array_equal(classification.assignment_masks[1], [False, True, False])
        np.testing.assert_array_equal(classification.counts, [1, 1])
        self.assertEqual(classification.unclassified_count, 1)
        np.testing.assert_array_equal(classification.candidate_group_indices[0], [0, 1, 2])
        np.testing.assert_array_equal(classification.candidate_group_indices[1], [0, 1, 2])
        np.testing.assert_allclose(
            np.sum(classification.template_probabilities, axis=1),
            np.asarray([1.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(
            np.sum(classification.raw_template_probabilities, axis=1),
            np.ones(3),
        )
        self.assertGreater(classification.template_probabilities[0, 0], 0.5)
        self.assertGreater(classification.template_probabilities[1, 1], 0.5)
        self.assertAlmostEqual(classification.winning_score_margins[0], 0.3)
        self.assertAlmostEqual(classification.winning_score_margins[1], 0.45)

    def test_multi_template_classification_prefers_a_passing_fit(self) -> None:
        classification = classify_template_candidates(
            [np.asarray([[0.0, 0.0]]), np.asarray([[1.0, 0.0]])],
            [np.asarray([True]), np.asarray([False])],
            [np.asarray([0.41]), np.asarray([0.95])],
            match_distance_nm=10.0,
        )

        np.testing.assert_array_equal(classification.counts, [1, 0])
        self.assertEqual(classification.unclassified_count, 0)
        np.testing.assert_allclose(classification.template_probabilities[0], [1.0, 0.0])
        self.assertGreater(classification.raw_template_probabilities[0, 1], 0.5)
        self.assertEqual(classification.runner_up_template_indices[0], -1)
        self.assertTrue(np.isinf(classification.winning_score_margins[0]))

    def test_shared_candidate_classification_is_template_order_invariant(self) -> None:
        centers = np.asarray([[0.0, 0.0], [100.0, 20.0]])
        accepted = [np.asarray([True, True]), np.asarray([True, False]), np.asarray([False, True])]
        scores = [np.asarray([0.4, 0.5]), np.asarray([0.9, 1.2]), np.asarray([1.5, 0.8])]
        baseline = classify_template_candidates(
            [centers.copy() for _ in range(3)],
            accepted,
            scores,
            match_distance_nm=20.0,
        )
        permutation = np.asarray([2, 0, 1])
        reordered = classify_template_candidates(
            [centers.copy() for _ in range(3)],
            [accepted[index] for index in permutation],
            [scores[index] for index in permutation],
            match_distance_nm=20.0,
        )

        np.testing.assert_array_equal(
            permutation[reordered.winning_template_indices],
            baseline.winning_template_indices,
        )
        np.testing.assert_allclose(
            reordered.template_probabilities[:, np.argsort(permutation)],
            baseline.template_probabilities,
        )

    def test_multi_template_classification_suppresses_nearby_duplicate_fragments(self) -> None:
        classification = classify_template_candidates(
            [
                np.asarray([[0.0, 0.0], [35.0, 0.0]]),
                np.empty((0, 2), dtype=float),
            ],
            [np.asarray([True, True]), np.empty(0, dtype=bool)],
            [np.asarray([0.8, 0.6]), np.empty(0, dtype=float)],
            match_distance_nm=20.0,
            deduplication_distance_nm=48.0,
        )

        np.testing.assert_array_equal(classification.counts, [1, 0])
        np.testing.assert_array_equal(classification.assignment_masks[0], [True, False])
        self.assertEqual(classification.unclassified_count, 0)
        self.assertEqual(classification.suppressed_duplicate_count, 1)
        np.testing.assert_array_equal(classification.winning_template_indices, [0, -2])

    def test_two_custom_templates_classify_a_mixed_image(self) -> None:
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(20_260_902)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        # The second design is a strict bright-site superset of the first.
        # Raw overlap alone must not classify both objects as the dense type.
        patterns = [np.asarray([0, 1, 4, 5, 9]), np.arange(len(grid))]
        templates: list[np.ndarray] = []
        for occupied in patterns:
            image = np.zeros((80, 100), dtype=float)
            for x_nm, y_nm in grid[occupied]:
                column = int(round((x_nm + 50.0) / 100.0 * 99.0))
                row = int(round((y_nm + 40.0) / 80.0 * 79.0))
                image[row, column] = 1.0
            templates.append(gaussian_filter(image, 2.0))

        mixed_regions: list[np.ndarray] = []
        for occupied, angle_degrees, center in zip(
            patterns,
            (23.0, -31.0),
            ((250.0, 300.0), (600.0, 500.0)),
        ):
            angle = np.deg2rad(angle_degrees)
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            mixed_regions.append(
                np.vstack([rng.normal(grid[index], 1.4, size=(30, 2)) for index in occupied])
                @ rotation.T
                + np.asarray(center)
            )
        points = np.vstack(mixed_regions)

        picks_by_template = [
            identify_origami_regions(
                points,
                pick_bin_size_nm=5.0,
                connect_distance_nm=15.0,
                density_threshold=0.02,
                min_candidate_points=100,
                max_candidate_points=500,
                rows=3,
                columns=4,
                spacing_x_nm=20.0,
                spacing_y_nm=20.0,
                rectangle_margin_nm=20.0,
                min_rectangle_confidence=0.4,
                site_mask_radius_nm=7.5,
                min_supported_sites=4,
                min_site_evidence=0.10,
                min_site_localizations=3,
                min_supported_rows=2,
                min_supported_columns=2,
                max_site_spacing_error_nm=8.0,
                alignment_pixel_nm=1.0,
                alignment_template_image=template,
                template_pixel_size_x_nm=100.0 / 99.0,
                template_pixel_size_y_nm=80.0 / 79.0,
                compute_grid_blob_bic=False,
            )
            for template in templates
        ]
        classification = classify_template_candidates(
            [
                np.asarray([np.median(region, axis=0) for region in picks.regions])
                for picks in picks_by_template
            ],
            [picks.accepted_mask for picks in picks_by_template],
            [
                bidirectional_template_classification_scores(
                    picks.rectangle_confidence,
                    picks.on_site_fraction,
                    picks.supported_site_count,
                    len(picks.template_points_nm),
                )
                for picks in picks_by_template
            ],
            match_distance_nm=20.0,
        )

        np.testing.assert_array_equal(classification.counts, [1, 1])
        self.assertEqual(classification.unclassified_count, 0)
    def test_custom_template_sites_use_configured_pitch_not_black_image_border(self) -> None:
        template = np.zeros((100, 200), dtype=float)
        for row in (20, 30, 40):
            for column in (50, 80, 110):
                template[row, column] = 1.0

        sites = custom_template_site_points(
            template,
            rectangle_width_nm=180.0,
            rectangle_height_nm=120.0,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
        )

        np.testing.assert_allclose(np.unique(sites[:, 0]), [-20.0, 0.0, 20.0])
        np.testing.assert_allclose(np.unique(sites[:, 1]), [-20.0, 0.0, 20.0])

    def test_custom_template_dot_count_does_not_expand_configured_footprint(self) -> None:
        template = np.zeros((220, 500), dtype=float)
        for column in range(60, 435, 34):
            template[170, column] = 1.0
        for row in range(58, 171, 16):
            template[row, 230] = 1.0
            template[row, 434] = 1.0

        sites = custom_template_site_points(
            template,
            rectangle_width_nm=120.0,
            rectangle_height_nm=100.0,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            active_width_nm=100.0,
            active_height_nm=80.0,
        )

        self.assertAlmostEqual(float(np.ptp(sites[:, 0])), 100.0)
        self.assertAlmostEqual(float(np.ptp(sites[:, 1])), 80.0)

    def test_custom_template_uses_explicit_nanometres_per_pixel(self) -> None:
        template = np.zeros((10, 20), dtype=float)
        template[2, 3] = 1.0
        template[7, 13] = 1.0

        sites = custom_template_site_points(
            template,
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            pixel_size_x_nm=2.0,
            pixel_size_y_nm=3.0,
        )

        np.testing.assert_allclose(sites, [[-13.0, -7.5], [7.0, 7.5]])

    def test_generated_grid_preserves_spacing_inside_image_margin(self) -> None:
        width_px, height_px = 500, 250
        width_nm, height_nm = 160.0, 80.0
        grid_width_nm, grid_height_nm = 120.0, 40.0
        template = np.zeros((height_px, width_px), dtype=float)
        left = int(round(20.0 * (width_px - 1) / width_nm))
        right = int(round(140.0 * (width_px - 1) / width_nm))
        bottom = int(round(20.0 * (height_px - 1) / height_nm))
        top = int(round(60.0 * (height_px - 1) / height_nm))
        template[bottom, left] = 1.0
        template[top, right] = 1.0

        sites = custom_template_site_points(
            template,
            rectangle_width_nm=width_nm,
            rectangle_height_nm=height_nm,
            pixel_size_x_nm=width_nm / (width_px - 1),
            pixel_size_y_nm=height_nm / (height_px - 1),
        )

        np.testing.assert_allclose(
            np.ptp(sites, axis=0),
            [grid_width_nm, grid_height_nm],
            atol=max(
                width_nm / (width_px - 1),
                height_nm / (height_px - 1),
            ),
        )

    def test_custom_template_is_scaled_into_the_physical_footprint(self) -> None:
        source = np.zeros((8, 10), dtype=float)
        source[2:4, 6:8] = 1.0

        prepared = prepare_custom_alignment_template(
            source,
            output_shape=(128, 128),
            rectangle_width_nm=100.0,
            rectangle_height_nm=80.0,
            canvas_side_nm=float(np.hypot(100.0, 80.0)),
        )

        self.assertEqual(prepared.shape, (128, 128))
        self.assertAlmostEqual(float(np.linalg.norm(prepared)), 1.0)
        self.assertAlmostEqual(float(np.mean(prepared)), 0.0, places=12)
        self.assertGreater(float(np.max(prepared)), 0.0)
        self.assertLess(float(np.min(prepared)), 0.0)

    def test_custom_barcode_template_drives_alignment_and_correlation(self) -> None:
        rng = np.random.default_rng(333)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        occupied = np.asarray([0, 1, 5, 6, 11])
        barcode = np.zeros((80, 100), dtype=float)
        for x_nm, y_nm in grid[occupied]:
            column = int(round((x_nm + 50.0) / 100.0 * 99.0))
            row = int(round((y_nm + 40.0) / 80.0 * 79.0))
            barcode[row, column] = 1.0
        from scipy.ndimage import gaussian_filter

        barcode = gaussian_filter(barcode, 2.0)
        expected_angle = 27.0
        angle = np.deg2rad(expected_angle)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        region = np.vstack(
            [rng.normal(grid[index], 1.4, size=(25, 2)) for index in occupied]
        ) @ rotation.T

        aligned, _centers, _corners, angles, correlations, _pixel_nm, reference = (
            _align_regions_by_image_correlation(
                [region],
                rectangle_width_nm=100.0,
                rectangle_height_nm=80.0,
                requested_pixel_nm=1.0,
                iterations=3,
                template_points_nm=grid,
                template_image=barcode,
                sparse_pose_site_count=3,
                sparse_site_radius_nm=7.5,
                sparse_min_site_localizations=3,
            )
        )

        angle_error = abs(((float(angles[0]) - expected_angle + 90.0) % 180.0) - 90.0)
        self.assertLess(angle_error, 1.0)
        self.assertGreater(float(correlations[0]), 0.9)
        self.assertEqual(len(aligned[0]), len(region))
        self.assertAlmostEqual(float(np.linalg.norm(reference)), 1.0)

    def test_asymmetric_custom_template_distinguishes_180_degree_orientation(self) -> None:
        rng = np.random.default_rng(334)
        grid = ideal_grid_points(7, 7, 20.0, 20.0)
        occupied = np.unique(np.r_[np.arange(7), np.arange(6, 49, 7)])
        barcode = np.zeros((140, 140), dtype=float)
        for x_nm, y_nm in grid[occupied]:
            column = int(round((x_nm + 70.0) / 140.0 * 139.0))
            row = int(round((y_nm + 70.0) / 140.0 * 139.0))
            barcode[row, column] = 1.0
        from scipy.ndimage import gaussian_filter

        barcode = gaussian_filter(barcode, 2.0)
        expected_angle = 207.0
        angle = np.deg2rad(expected_angle)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = np.vstack(
            [rng.normal(grid[index], 1.4, size=(25, 2)) for index in occupied]
        ) @ rotation.T + np.asarray([500.0, 500.0])

        picks = identify_origami_regions(
            points,
            pick_bin_size_nm=5.0,
            connect_distance_nm=10.0,
            density_threshold=0.02,
            min_candidate_points=200,
            max_candidate_points=400,
            rows=7,
            columns=7,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            rectangle_margin_nm=10.0,
            min_rectangle_confidence=0.3,
            site_mask_radius_nm=7.5,
            min_supported_sites=8,
            min_site_evidence=0.10,
            min_site_localizations=3,
            min_supported_rows=2,
            min_supported_columns=2,
            max_site_spacing_error_nm=8.0,
            alignment_pixel_nm=1.0,
            alignment_template_image=barcode,
        )

        self.assertEqual(len(picks.regions), 1)
        self.assertEqual(picks.accepted_count, 1)
        angle_error = abs(
            ((float(picks.rectangle_angles_deg[0]) - expected_angle + 180.0) % 360.0) - 180.0
        )
        self.assertLess(angle_error, 1.0)
        self.assertGreater(float(picks.rectangle_confidence[0]), 0.7)
        self.assertEqual(len(picks.aligned_regions[0]), len(points))
        self.assertEqual(picks.template_points_nm.shape, (len(occupied), 2))
        self.assertEqual(picks.site_localization_counts.shape, (1, len(occupied)))
        self.assertEqual(picks.lattice_supported_sites.shape, (1, 49))
        np.testing.assert_array_equal(np.flatnonzero(picks.lattice_supported_sites[0]), occupied)
        nearest_template_distance = np.min(
            np.linalg.norm(
                picks.template_points_nm[:, None, :] - grid[occupied][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        self.assertLess(float(np.max(nearest_template_distance)), 1.0)

        overlay = align_picked_origamis(
            picks.accepted_aligned_regions,
            rows=7,
            columns=7,
            spacing_x_nm=20.0,
            spacing_y_nm=20.0,
            site_radius_nm=7.5,
            prealigned=True,
            use_g5m=False,
            direct_min_site_localizations=3,
            grid_points_nm=picks.template_points_nm,
            symmetrize_180=False,
        )
        np.testing.assert_allclose(overlay.grid_points_nm, picks.template_points_nm)
        self.assertEqual(overlay.site_counts.shape, (1, len(occupied)))
        self.assertFalse(overlay.symmetrized_180)

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

    def test_site_prominence_diagnostics_expose_the_measured_geometry(self) -> None:
        rng = np.random.default_rng(4_100)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        points = rng.normal(grid[5], 1.5, size=(30, 2))

        diagnostics = sparse_site_evidence_diagnostics(points, grid, site_radius_nm=7.5)

        self.assertEqual(diagnostics.boundary_points_nm.shape, (12, 32, 2))
        self.assertTrue(np.all(np.isfinite(diagnostics.peak_positions_nm[5])))
        self.assertTrue(np.all(np.isfinite(diagnostics.boundary_reference_positions_nm[5])))
        expected_prominence = (
            diagnostics.peak_density[5] - diagnostics.boundary_reference_density[5]
        ) / diagnostics.peak_density[5]
        self.assertAlmostEqual(float(diagnostics.prominence[5]), float(expected_prominence))
        boundary_radii = np.linalg.norm(
            diagnostics.boundary_points_nm[5] - diagnostics.peak_positions_nm[5],
            axis=1,
        )
        self.assertAlmostEqual(float(np.min(boundary_radii)), float(np.max(boundary_radii)))

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
            any("site prominence, spacing, and ΔBIC" in message for _percent, message in progress_updates)
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
        self.assertGreaterEqual(picks.original_point_counts[0], picks.point_counts[0])
        self.assertAlmostEqual(
            picks.crop_retained_fractions[0],
            picks.point_counts[0] / picks.original_point_counts[0],
        )
        self.assertLessEqual(picks.alignment_reference_image.shape[0], 128)
        self.assertEqual(picks.alignment_candidate_images.shape[0], 1)
        self.assertEqual(picks.site_localization_counts.shape, (1, 12))
        self.assertEqual(picks.site_prominence.shape, (1, 12))
        self.assertEqual(picks.site_peak_positions_nm.shape, (1, 12, 2))
        self.assertEqual(picks.site_boundary_points_nm.shape, (1, 12, 32, 2))
        self.assertEqual(picks.site_centroids_nm.shape, (1, 12, 2))
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
        self.assertTrue(any("site prominence, spacing, and ΔBIC" in message for _percent, message in progress_updates))
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

    def test_sparse_pose_alignment_rejects_a_nearby_origami_distractor(self) -> None:
        rng = np.random.default_rng(2026)
        grid = ideal_grid_points(3, 4, 20.0, 20.0)
        expected_angle = 34.0
        angle = np.deg2rad(expected_angle)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target_sites = np.asarray([0, 1, 2, 4, 5, 7, 9, 11])
        target = np.vstack(
            [rng.normal(grid[index], 1.8, size=(22, 2)) for index in target_sites]
        ) @ rotation.T

        distractor_angle = np.deg2rad(-25.0)
        distractor_rotation = np.asarray(
            [
                [np.cos(distractor_angle), -np.sin(distractor_angle)],
                [np.sin(distractor_angle), np.cos(distractor_angle)],
            ]
        )
        distractor_sites = np.asarray([0, 1, 4, 5, 8, 9])
        distractor = (
            np.vstack(
                [rng.normal(grid[index], 2.0, size=(14, 2)) for index in distractor_sites]
            )
            @ distractor_rotation.T
            + np.asarray([50.0, -45.0])
        )

        aligned, _centers, _corners, angles, correlations, _pixel_nm, _reference = (
            _align_regions_by_image_correlation(
                [np.vstack((target, distractor))],
                rectangle_width_nm=100.0,
                rectangle_height_nm=80.0,
                requested_pixel_nm=1.0,
                iterations=3,
                template_points_nm=grid,
                sparse_pose_site_count=5,
                sparse_site_radius_nm=7.5,
                sparse_min_site_localizations=3,
            )
        )

        angle_error = abs(((float(angles[0]) - expected_angle + 90.0) % 180.0) - 90.0)
        # The robust search must recover the target basin rather than the
        # distractor's roughly 60-degree-different orientation. Subsequent site
        # assignment tolerates this small residual angular error.
        self.assertLess(angle_error, 4.0)
        self.assertGreater(float(correlations[0]), 0.3)
        self.assertLess(len(aligned[0]), len(target) + len(distractor))

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
        self.assertEqual(len(rejected_picks.regions), 0)
        self.assertEqual(rejected_picks.accepted_count, 0)
        self.assertEqual(rejected_picks.point_counts.size, 0)

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
