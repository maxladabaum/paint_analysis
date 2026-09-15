from dataclasses import dataclass

import numpy as np

from origami_analysis import fitted_footprint_overlap_fractions
from paint_analysis_gui import PaintAnalysisApp


def rectangle(x=0., y=0., angle=0., width=120., height=40.):
    corners = np.array([[-width/2, -height/2], [width/2, -height/2],
                        [width/2, height/2], [-width/2, height/2]])
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    return corners @ rotation.T + (x, y)


def test_crossing_fits_report_overlap_for_both_objects():
    fractions = fitted_footprint_overlap_fractions(
        np.array([rectangle(), rectangle(angle=90), rectangle(x=300)]))
    np.testing.assert_allclose(fractions, [1/3, 1/3, 0], atol=1e-12)


def test_touching_or_nearby_objects_are_not_overlaps():
    # Bounding boxes overlap, but the parallel rotated rectangles do not.
    shift = np.array([-1, 1]) * 45 / np.sqrt(2)
    fractions = fitted_footprint_overlap_fractions(
        np.array([rectangle(angle=45), rectangle(*shift, angle=45)]))
    np.testing.assert_allclose(fractions, 0, atol=1e-12)
    np.testing.assert_allclose(fitted_footprint_overlap_fractions(
        np.array([rectangle(), rectangle(x=120)])), 0, atol=1e-12)


def test_image_padding_is_excluded_and_invalid_fit_does_not_reject_neighbor():
    corners = np.array([rectangle(width=160, height=80),
                        rectangle(x=140, width=160, height=80)])
    assert np.all(fitted_footprint_overlap_fractions(corners) > .1)
    np.testing.assert_allclose(fitted_footprint_overlap_fractions(corners, margin_nm=20), 0)
    np.testing.assert_allclose(fitted_footprint_overlap_fractions(corners, eligible=[True, False]), 0)


@dataclass
class Picks:
    point_counts: np.ndarray
    rectangle_confidence: np.ndarray
    accepted_mask: np.ndarray
    rectangle_corners_nm: np.ndarray
    footprint_overlap_fraction: np.ndarray


def test_alignment_rejects_both_overlapping_fits_and_retains_rejection():
    picks = Picks(np.full(3, 100), np.full(3, .99), np.ones(3, bool),
                  np.array([rectangle(), rectangle(angle=90), rectangle(x=300)]), np.zeros(3))
    params = dict(require_corner_support=False, min_candidate_points=50,
                  max_candidate_points=500, min_rectangle_confidence=.3)
    result = PaintAnalysisApp._apply_origami_alignment_filters(picks, params)
    assert result.accepted_mask.tolist() == [False, False, True]
    # Later rejection of one member does not resurrect the contaminated fit.
    result.rectangle_confidence[1] = 0
    repeated = PaintAnalysisApp._apply_origami_alignment_filters(result, params)
    assert repeated.accepted_mask.tolist() == [False, False, True]


def test_overlap_threshold_defaults_to_zero_and_can_be_relaxed():
    picks = Picks(np.full(2, 100), np.full(2, .99), np.ones(2, bool),
                  np.array([rectangle(), rectangle(x=114)]), np.zeros(2))
    params = dict(require_corner_support=False, min_candidate_points=50,
                  max_candidate_points=500, min_rectangle_confidence=.3)
    strict = PaintAnalysisApp._apply_origami_alignment_filters(picks, params)
    assert strict.accepted_mask.tolist() == [False, False]
    np.testing.assert_allclose(strict.footprint_overlap_fraction, .05)
    # Equality is allowed; only overlap exceeding the threshold fails.
    params['max_footprint_overlap_fraction'] = .05
    relaxed = PaintAnalysisApp._apply_origami_alignment_filters(strict, params)
    assert relaxed.accepted_mask.tolist() == [True, True]
    params['max_footprint_overlap_fraction'] = 1.
    assert PaintAnalysisApp._apply_origami_alignment_filters(strict, params).accepted_mask.all()


def test_overlap_threshold_rejects_invalid_values():
    import pytest
    from origami_analysis import identify_origami_regions

    picks = Picks(np.full(2, 100), np.full(2, .99), np.ones(2, bool),
                  np.array([rectangle(), rectangle(x=114)]), np.zeros(2))
    for threshold in (-.01, 1.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='overlap'):
            PaintAnalysisApp._apply_origami_alignment_filters(
                picks, dict(require_corner_support=False, max_footprint_overlap_fraction=threshold))
        with pytest.raises(ValueError, match='overlap'):
            identify_origami_regions(np.zeros((1, 2)), pick_bin_size_nm=5.,
                                     connect_distance_nm=15., density_threshold=.2,
                                     min_candidate_points=1, max_candidate_points=100,
                                     max_footprint_overlap_fraction=threshold)


def test_connected_coarse_cluster_produces_only_one_candidate():
    from scipy.ndimage import label
    from origami_analysis import pick_origami_candidates

    rng = np.random.default_rng(9)
    fiducials = np.array([(x, y) for x in (-60, 60) for y in (-15, 0, 15)])
    full = np.array([(x, y) for x in np.arange(-60, 61, 10) for y in (-15, 0, 15)])
    # Two plausible template poses linked by above-threshold signal form one
    # blue contour; they must not become two independent fitting inputs.
    bridge = np.column_stack((np.arange(70, 121, 10), np.zeros(6)))
    sites = np.vstack((full, full + [190, 0], bridge))
    cluster = np.vstack([rng.normal(site, 1, (20, 2)) for site in sites])
    separate = np.vstack([rng.normal(site + [500, 0], 1, (20, 2)) for site in fiducials])
    result = pick_origami_candidates(
        np.vstack((cluster, separate)), bin_size_nm=5, connect_distance_nm=15,
        density_threshold=.2, minimum_points=100, candidate_template_points_nm=fiducials)
    assert {frozenset(map(tuple, region)) for region in result[0]} == {
        frozenset(map(tuple, cluster)), frozenset(map(tuple, separate))}
    coarse_labels, count = label(result[2] >= .2, structure=np.ones((3, 3)))
    for component in range(1, count + 1):
        owners = np.unique(result[4][coarse_labels == component])
        assert len(owners) == 1
