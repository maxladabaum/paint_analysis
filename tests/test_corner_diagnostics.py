import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from origami_analysis import alignment_corner_sites
from paint_analysis_gui import (draw_corner_support_diagnostics, required_corner_mask,
                                PaintAnalysisApp, ORIGAMI_ALIGNMENT_CACHE_KEYS,
                                origami_stage_cache_matches)


class CornerDiagnosticsTests(unittest.TestCase):
    def test_counts_colors_and_radii_match_gate_at_rotated_world_positions(self):
        sites = np.array([[-20., -10.], [20., -10.], [20., 10.], [-20., 10.]])
        region = np.repeat(sites, [3, 2, 4, 3], axis=0)
        rotation = np.array([[0., -1.], [1., 0.]])
        translation = np.array([100., 200.])
        footprint = np.array([[-30., -20.], [30., -20.], [30., 20.], [-30., 20.]])
        picks = SimpleNamespace(template_points_nm=sites, aligned_regions=[region],
                                point_counts=np.array([len(region)]),
                                rectangle_corners_nm=np.array([footprint @ rotation.T + translation]))
        params = dict(alignment_template_image=np.ones((5, 5)),
                      min_site_localizations=3, site_mask_radius_nm=2.)
        self.assertFalse(required_corner_mask(picks, params)[0])
        axis = Figure().subplots()
        artists = draw_corner_support_diagnostics(axis, picks, params, 0)
        circles = [artist for artist in artists if isinstance(artist, Circle)]
        np.testing.assert_allclose([circle.center for circle in circles], sites @ rotation.T + translation)
        self.assertTrue(all(circle.radius == 2 for circle in circles))
        labels = [text.get_text() for text in axis.texts]
        self.assertEqual(labels, ['C1: 3/3 PASS', 'C2: 2/3 FAIL', 'C3: 4/3 PASS', 'C4: 3/3 PASS'])
        self.assertNotEqual(circles[0].get_edgecolor(), circles[1].get_edgecolor())
        local_axis = Figure().subplots()
        draw_corner_support_diagnostics(local_axis, picks, params, 0, world_coordinates=False)
        np.testing.assert_allclose([circle.center for circle in local_axis.patches], sites)
        for artist in artists:
            artist.remove()
        self.assertEqual(len(axis.patches) + len(axis.texts) + len(axis.lines), 0)

    def test_disabled_corner_gate_allows_missing_corners_in_both_stages(self):
        @dataclass
        class Picks:
            point_counts: np.ndarray
            rectangle_confidence: np.ndarray
            accepted_mask: np.ndarray
            aligned_regions: list
            template_points_nm: np.ndarray

        sites = np.array([[-20., -10.], [20., -10.], [20., 10.], [-20., 10.]])
        picks = Picks(np.array([9]), np.array([.9]), np.array([False]),
                      [np.repeat(sites[1:], 3, axis=0)], sites)
        params = dict(alignment_template_image=np.ones((5, 5)),
                      site_mask_radius_nm=2., min_site_localizations=3,
                      min_rectangle_confidence=.3, digital_pixel_model={})
        self.assertFalse(required_corner_mask(picks, params)[0])
        params['require_corner_support'] = False
        self.assertTrue(required_corner_mask(picks, params, 0)[0])
        self.assertTrue(PaintAnalysisApp._apply_origami_alignment_filters(picks, params).accepted_mask[0])
        self.assertTrue(PaintAnalysisApp._apply_origami_acceptance_filters(picks, params).accepted_mask[0])
        params['min_rectangle_confidence'] = .95
        self.assertFalse(PaintAnalysisApp._apply_origami_alignment_filters(picks, params).accepted_mask[0])
        self.assertFalse(PaintAnalysisApp._apply_origami_acceptance_filters(picks, params).accepted_mask[0])
        self.assertFalse(origami_stage_cache_matches(
            dict(params, require_corner_support=True), params, ORIGAMI_ALIGNMENT_CACHE_KEYS))
        picks.rectangle_corners_nm = np.array([sites])
        axis = Figure().subplots()
        draw_corner_support_diagnostics(axis, picks, params, 0)
        self.assertTrue(all('(gate off)' in text.get_text() for text in axis.texts))
        self.assertTrue(any('FAIL' in text.get_text() for text in axis.texts))

    def test_inactive_gate_and_duplicate_corner_marks(self):
        sites = np.array([[0., 0.], [10., 0.]])
        self.assertEqual(len(alignment_corner_sites(sites)), 2)
        axis = Figure().subplots()
        self.assertEqual(draw_corner_support_diagnostics(axis, None, {}, 0), [])
        self.assertEqual(alignment_corner_sites([]).shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
