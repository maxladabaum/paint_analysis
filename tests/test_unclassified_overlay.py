import unittest
from types import SimpleNamespace
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from paint_analysis_gui import unclassified_theoretical_points, draw_unclassified_theoretical_overlay


class UnclassifiedOverlayTests(unittest.TestCase):
    def test_overlay_uses_observed_pattern_and_locked_pose(self):
        grid = np.array([[-10., -5.], [10., -5.], [-10., 5.], [10., 5.]])
        corners = np.array([[-20., -15.], [20., -15.], [20., 15.], [-20., 15.]])
        rotation = np.array([[0., -1.], [1., 0.]])
        translation = np.array([100., 200.])
        picks = SimpleNamespace(template_points_nm=grid,
                                rectangle_corners_nm=np.array([corners @ rotation.T + translation]))
        model = dict(bit_ids=('a', 'b'), bit_physical_cells=((1,), (2,)),
                     active_bits=(True, False), alignment_cells=(0,), physical_shape=(2, 2))
        params = dict(digital_pixel_model=model, logical_model=model,
                      digital_pixel_ids=('b', 'a'),
                      classification_observed_digital_states=((True, False),))
        points = unclassified_theoretical_points(picks, params, 0)
        np.testing.assert_allclose(points, grid[[0, 2]] @ rotation.T + translation)
        axis = Figure().subplots()
        axis.set_xlim(80, 120)
        axis.set_ylim(180, 220)
        artists = draw_unclassified_theoretical_overlay(axis, [{'theoretical_points_nm': points}])
        np.testing.assert_allclose(artists[0].get_offsets(), points)
        np.testing.assert_allclose(artists[0].get_edgecolors()[0], to_rgba('#9ca3af'))
        self.assertEqual(len(artists[0].get_facecolors()), 0)
        artists[0].remove()
        self.assertEqual(len(axis.collections), 0)
        axis.set_xlim(500, 600)
        self.assertEqual(draw_unclassified_theoretical_overlay(axis, [{'theoretical_points_nm': points}]), [])

    def test_missing_older_details_and_overview_limit(self):
        axis = Figure().subplots()
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 10)
        self.assertEqual(draw_unclassified_theoretical_overlay(axis, [{}]), [])
        details = [{'theoretical_points_nm': [[1, 1], [2, 2]]}] * 10
        artist = draw_unclassified_theoretical_overlay(axis, details, maximum_candidates=3)[0]
        self.assertEqual(len(artist.get_offsets()), 6)


if __name__ == '__main__':
    unittest.main()
