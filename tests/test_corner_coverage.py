from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from origami_analysis import alignment_corner_counts, alignment_dark_boundary
from paint_analysis_gui import PaintAnalysisApp, draw_alignment_dark_boundary


def test_each_corner_requires_support_even_with_bright_interior():
    sites = np.array([[-20., -10.], [20., -10.], [20., 10.], [-20., 10.]])
    complete = np.repeat(sites, 5, axis=0)
    missing = np.vstack([complete[5:], np.zeros((500, 2))])
    counts = alignment_corner_counts([complete, missing], sites, 2.)
    assert np.all(counts[0] == 5)
    assert np.count_nonzero(counts[1] == 0) == 1


@dataclass
class Picks:
    point_counts: np.ndarray
    rectangle_confidence: np.ndarray
    accepted_mask: np.ndarray
    aligned_regions: list
    template_points_nm: np.ndarray


def test_step_two_gate_rejects_missing_corner_despite_high_correlation():
    sites = np.array([[-20., -10.], [20., -10.], [20., 10.], [-20., 10.]])
    complete = np.repeat(sites, 5, axis=0)
    picks = Picks(np.array([20, 15]), np.array([.99, .99]), np.ones(2, bool),
                  [complete, complete[5:]], sites)
    params = dict(alignment_template_image=np.ones((5, 5)),
                  min_site_localizations=5, site_mask_radius_nm=2., min_rectangle_confidence=.3)
    result = PaintAnalysisApp._apply_origami_alignment_filters(picks, params)
    assert result.accepted_mask.tolist() == [True, False]
    params['alignment_template_overlay_points_nm'] = sites
    picks.template_points_nm = np.zeros((1, 2))  # Later detection can replace the grid.
    params['use_correlation_gate'] = False
    assert PaintAnalysisApp._apply_origami_alignment_filters(picks, params).accepted_mask.tolist() == [True, False]


def test_boundary_matches_exterior_mask_pixel_edges_and_transforms():
    reference = np.zeros((10, 10))
    reference[2, 3] = reference[7, 8] = 1
    boundary = alignment_dark_boundary(reference, 100.)
    expected = np.array([[-20, -30], [40, -30], [40, 30], [-20, 30], [-20, -30]])
    np.testing.assert_allclose(boundary, expected)
    picks = SimpleNamespace(alignment_reference_image=reference, alignment_canvas_side_nm=100.)
    axis = Figure().subplots()
    artists = draw_alignment_dark_boundary(axis, picks, lambda points: points + [200, 300])
    np.testing.assert_allclose(artists[0].get_xydata(), expected + [200, 300])
    assert artists[0].get_linestyle() == '--'
