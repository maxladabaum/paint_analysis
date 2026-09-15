import unittest
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from paint_analysis_gui import PaintAnalysisApp


class RoiSelectionTests(unittest.TestCase):
    def test_raw_roi_survives_release_and_render_and_can_be_reselected(self):
        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        app = SimpleNamespace(
            loaded=SimpleNamespace(path=Path('locs.csv')),
            raw_map_axis=axis, raw_map_canvas=canvas, raw_map_figure=figure,
            map_density_images={}, raw_selector=None, roi_nm=None,
            raw_roi_patch=None, roi_patch=None, linked_roi_patch=None,
            filtered_roi_patch=None, map_axis=Figure().add_subplot(111),
            linked_map_axis=Figure().add_subplot(111),
            filtered_map_axis=Figure().add_subplot(111),
            map_canvas=Mock(), linked_map_canvas=Mock(), notebook=Mock(),
            status=Mock(), roi_label=Mock(),
            _remove_raw_map_colorbar=Mock(), _remove_raw_roi_highlight=Mock(),
            _add_fixed_colorbar=Mock(), _center_map_axis=Mock(),
            _apply_shared_map_limits=Mock(),
            _scale_map_density=lambda image: (image, (0, 1)),
        )
        for name in ('_enable_raw_roi_selector', '_draw_roi_patch', '_remove_roi_patch',
                     '_on_roi_select', '_update_roi_label', '_plot_raw_map', 'clear_roi'):
            setattr(app, name, MethodType(getattr(PaintAnalysisApp, name), app))
        def highlight():
            canvas.draw_idle()
            return 0
        app._highlight_raw_roi_locs = highlight
        result = dict(source_path=app.loaded.path, image=np.ones((10, 10)),
                      extent=(0, 100, 0, 100), n_rendered=100, disp_px_size_nm=10)
        def select(low, high):
            for event_name, point in [('button_press_event', low),
                                      ('motion_notify_event', high),
                                      ('button_release_event', high)]:
                x, y = axis.transData.transform(point)
                event = MouseEvent(event_name, canvas, x, y, button=1)
                canvas.callbacks.process(event_name, event)
            canvas.draw()
        app._plot_raw_map(result)
        select((10, 20), (40, 60))
        np.testing.assert_allclose(app.roi_nm, (10, 40, 20, 60))
        self.assertIn(app.raw_roi_patch, axis.patches)
        self.assertTrue(app.raw_roi_patch.get_visible())
        previous_selector = app.raw_selector
        app._plot_raw_map(result)
        self.assertIsNot(previous_selector, app.raw_selector)
        self.assertIn(app.raw_roi_patch, axis.patches)
        self.assertTrue(app.raw_roi_patch.get_visible())
        select((50, 50), (80, 90))
        np.testing.assert_allclose(app.roi_nm, (50, 80, 50, 90))
        self.assertIn(app.raw_roi_patch, axis.patches)
        app.clear_roi()
        self.assertIsNone(app.roi_nm)
        self.assertIsNone(app.raw_roi_patch)


if __name__ == '__main__':
    unittest.main()
