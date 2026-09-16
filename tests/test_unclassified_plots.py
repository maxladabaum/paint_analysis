from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from paint_analysis_gui import (PaintAnalysisApp, selected_classification_has_assignments,
                                unclassified_display_payload)


@dataclass
class Picks:
    regions: list
    accepted_mask: np.ndarray

    @property
    def accepted_count(self):
        return int(self.accepted_mask.sum())


def results():
    regions = [np.array([[i, 0.], [i, 1.]]) for i in range(4)]
    params = dict(logical_model=dict(bit_ids=('a', 'b'), active_bits=(True, False)))
    return {'code1': dict(picks=Picks(regions, np.array([1, 0, 0, 0], bool)), params=params),
            'full': dict(picks=Picks(regions, np.array([0, 1, 0, 0], bool)), params=params)}


def test_unclassified_mask_is_union_complement_without_changing_classes():
    saved = results()
    payload = unclassified_display_payload(saved)
    np.testing.assert_array_equal(payload['picks'].accepted_mask, [0, 0, 1, 1])
    assert payload['picks'].regions is saved['code1']['picks'].regions
    assert payload['params']['logical_model']['active_bits'] == (True, True)
    assert saved['code1']['params']['logical_model']['active_bits'] == (True, False)
    assert set(saved) == {'code1', 'full'}
    assert saved['code1']['picks'].accepted_count == 1
    assert saved['full']['picks'].accepted_count == 1


def test_no_unclassified_and_inconsistent_ordering():
    saved = results()
    saved['full']['picks'].accepted_mask[2:] = True
    assert unclassified_display_payload(saved)['picks'].accepted_count == 0
    saved['full']['picks'].regions = saved['full']['picks'].regions[::-1]
    with pytest.raises(ValueError, match='shared candidate'):
        unclassified_display_payload(saved)


def test_selection_preserves_gallery_and_automatically_builds_separate_overlay():
    saved = results()
    app = SimpleNamespace(
        origami_template_result_view=SimpleNamespace(get=lambda: 'Unclassified'),
        origami_plot_option=SimpleNamespace(get=lambda: 'Individual origami gallery'),
        origami_multi_template_results=saved, origami_multi_template_overlay_results={},
        status=mock.Mock(), _refresh_origami_action_states=mock.Mock(),
        after_idle=mock.Mock(), _auto_build_active_template_overlay=mock.Mock())
    PaintAnalysisApp._on_origami_template_result_selection(app)
    assert app.origami_pick_result.accepted_count == 2
    assert selected_classification_has_assignments(app)
    app.after_idle.assert_called_once_with(app._auto_build_active_template_overlay)
    app.overlay_origamis = mock.Mock()
    PaintAnalysisApp._auto_build_active_template_overlay(app)
    app.overlay_origamis.assert_called_once()
    assert 'Unclassified' in app.origami_multi_template_overlays_building


def test_selection_reuses_unclassified_overlay():
    app = SimpleNamespace(
        origami_template_result_view=SimpleNamespace(get=lambda: 'Unclassified'),
        origami_plot_option=SimpleNamespace(get=lambda: 'Aligned density'),
        origami_multi_template_results=results(),
        origami_multi_template_overlay_results={'Unclassified': dict(result=object(), source='test',
            source_count=4, render_settings={}, occupancy_threshold=1)},
        render_origami_plot=mock.Mock(), _refresh_origami_action_states=mock.Mock(), after_idle=mock.Mock())
    PaintAnalysisApp._on_origami_template_result_selection(app)
    app.render_origami_plot.assert_called_once()
    app.after_idle.assert_not_called()


def test_selector_keeps_unclassified_selection():
    variable = mock.Mock(get=lambda: 'Unclassified')
    app = SimpleNamespace(origami_multi_template_results=results(),
        origami_template_result_view=variable, origami_template_result_combo=mock.Mock())
    PaintAnalysisApp._sync_origami_classification_selector_state(app)
    variable.set.assert_not_called()
    app.origami_template_result_combo.configure.assert_called_once_with(
        values=('All templates', 'code1', 'full', 'Unclassified'))
