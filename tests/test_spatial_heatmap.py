import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from matplotlib.figure import Figure

from paint_analysis_gui import PaintAnalysisApp, digital_group_presence_counts, origami_grid_points, spatial_digital_pixel_params


class SpatialHeatmapTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            'spacing_x_nm': 10.1, 'spacing_y_nm': 5.1,
            'column_offsets_nm': [0] * 6 + [10] * 6,
            'site_mask_radius_nm': 2.5,
            'digital_pixel_model': {
                'physical_shape': (8, 12), 'bit_ids': ('a', 'b'),
                'bit_physical_cells': ((5,), (6,)), 'active_bits': (True, False),
            },
            'digital_pixel_probabilities': [[0.9, 0.8], [0.8, np.nan], [0.9, 0.9]],
            'digital_group_localization_evidence': [[6, 2], [6, 6], [6, 6]],
            'digital_group_prominences': [[0.5, 0.5], [0.01, 0.5], [0.5, 0.5]],
            'min_site_localizations': 5, 'min_site_evidence': 0.1,
        }
        self.grid = origami_grid_points(8, 12, 10.1, 5.1, self.params)

    def test_assigned_only_full_decisions_and_missing_measurements(self):
        counts = digital_group_presence_counts(self.grid, self.params, [True, True, False])
        self.assertEqual(counts, {'a': [1, 2], 'b': [0, 1]})
        self.assertEqual(digital_group_presence_counts(self.grid, self.params, [False]*3),
                         {'a': [0, 0], 'b': [0, 0]})

    def test_spatial_plot_geometry_color_and_text_control(self):
        app = SimpleNamespace(
            origami_template_result_view=Mock(get=lambda: 'full_align'),
            origami_multi_template_results={'full_align': {
                'params': self.params, 'picks': SimpleNamespace(accepted_mask=np.array([True, True, False]))}},
            origami_figure=Figure(), origami_show_text_statistics=Mock(get=lambda: True),
            origami_canvas=Mock(), origami_toolbar=Mock(), notebook=Mock(), status=Mock(),
            _configure_origami_navigation_controls=Mock(),
        )
        PaintAnalysisApp._plot_digital_pixel_spatial_heatmap(app)
        axis = app.origami_figure.axes[0]
        self.assertEqual(axis.get_aspect(), 1.0)
        self.assertEqual(len(axis.collections), 2)  # Includes inactive expected pixels.
        self.assertIn('50% (1/2)', axis.texts[0].get_text())
        self.assertIn('0% (0/1)', axis.texts[1].get_text())
        centers = [collection.get_paths()[0].vertices[:, 0].mean() for collection in axis.collections]
        self.assertAlmostEqual(centers[1] - centers[0], 20.1, delta=0.1)
        app.origami_show_text_statistics.get = lambda: False
        PaintAnalysisApp._plot_digital_pixel_spatial_heatmap(app)
        self.assertEqual(len(app.origami_figure.axes[0].texts), 0)

        app.origami_multi_template_results['empty'] = {
            'params': self.params, 'picks': SimpleNamespace(accepted_mask=np.zeros(3, dtype=bool))}
        app.origami_template_result_view.get = lambda: 'All templates'
        PaintAnalysisApp._plot_digital_pixel_spatial_heatmap(app)
        self.assertEqual(len(app.origami_figure.axes), 3)  # Two cohorts and shared color scale.
        self.assertIn('0 assigned', app.origami_figure.axes[1].get_title())

    def test_full_align_aggregate_is_not_copied_to_every_site(self):
        params = {
            'rows': 1, 'columns': 2, 'spacing_x_nm': 30., 'spacing_y_nm': 5.1,
            'site_mask_radius_nm': 2.5, 'min_site_localizations': 5, 'min_site_evidence': 0.1,
            'digital_pixel_model': {'physical_shape': (1, 2), 'bit_ids': ('full_align',),
                                    'bit_physical_cells': ((0, 1),)},
            'digital_pixel_probabilities': ((1.,), (1.,)),
        }
        grid = origami_grid_points(1, 2, 30., 5.1, params)
        picks = SimpleNamespace(accepted_mask=np.array([True, True]),
            aligned_regions=[np.repeat(grid[:1], 30, axis=0), np.repeat(grid, 30, axis=0)])
        measured = spatial_digital_pixel_params(params, picks)
        self.assertEqual(measured['digital_pixel_model']['bit_ids'], ('R1C1', 'R1C2'))
        self.assertEqual(digital_group_presence_counts(grid, measured, picks.accepted_mask),
                         {'R1C1': [2, 2], 'R1C2': [1, 2]})
        self.assertEqual(params['digital_pixel_probabilities'], ((1.,), (1.,)))

    def test_full_align_without_localizations_does_not_invent_site_presence(self):
        params = {'digital_pixel_model': {'physical_shape': (1, 2),
            'bit_ids': ('full_align', '1'), 'bit_physical_cells': ((0, 1), (0,))},
            'digital_pixel_probabilities': ((1., 0.2),)}
        measured = spatial_digital_pixel_params(params, SimpleNamespace(accepted_mask=[True]))
        self.assertEqual(measured['digital_pixel_model']['bit_ids'], ('1', 'R1C2'))
        self.assertEqual(measured['digital_pixel_probabilities'][0, 0], 0.2)
        self.assertTrue(np.isnan(measured['digital_pixel_probabilities'][0, 1]))
