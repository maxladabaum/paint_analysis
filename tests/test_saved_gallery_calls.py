from types import SimpleNamespace
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from origami_analysis import align_picked_origamis
from paint_analysis_gui import PaintAnalysisApp, saved_gallery_classification


def fixture():
    regions = [np.array([[i, 0.], [i, 2.]]) for i in range(4)]
    picks = SimpleNamespace(regions=regions, accepted_mask=np.array([False, True, False, True]))
    params = dict(classification_method='exact digital ON/OFF lookup', _unclassified_display=True,
        digital_pixel_ids=('b', 'a'), digital_pixel_model=dict(bit_ids=('a', 'b'), bit_physical_cells=((0,), (1,))),
        digital_pixel_probabilities=((.1, .1), (.2, .9), (.9, .9), (.9, .9)),
        digital_group_localization_evidence=((1, 1), (1, 6), (8, 8), (9, 9)),
        digital_group_prominences=((0, 0), (.1, .8), (.9, .9), (.9, .9)),
        min_site_localizations=3, min_site_evidence=.1)
    app = SimpleNamespace(origami_identification_params=params,
        _active_origami_picks_and_params=lambda: (picks, params),
        _active_origami_digital_pixel_model=lambda: params['digital_pixel_model'],
        origami_multi_template_unclassified_details=[dict(center_nm=[3, 1], template_name='full', failure_reasons=['correlation below threshold'])])
    result = SimpleNamespace(aligned_points=[regions[1], regions[3]], centers_nm=np.array([[1, 1], [3, 1]]))
    return app, result


def test_saved_mapping_uses_filtered_order_and_reorders_ids():
    app, result = fixture()
    params, index, reason = saved_gallery_classification(app, result, 0)
    assert index == 1
    np.testing.assert_equal(params['digital_pixel_probabilities'], [[.9, .2]])
    assert app.origami_identification_params['digital_pixel_probabilities'][1] == (.2, .9)
    assert saved_gallery_classification(app, result, 1)[2] == 'Unclassified — full: correlation below threshold'
    result.centers_nm[0] = [100, 100]
    assert saved_gallery_classification(app, result, 0) is None


def test_gallery_never_remeasures_classification_calls():
    app, result = fixture()
    app.origami_show_site_diagnostics = SimpleNamespace(get=lambda: True)
    with mock.patch.object(PaintAnalysisApp, '_measure_direct_digital_groups') as measure, \
         mock.patch.object(PaintAnalysisApp, '_draw_digital_decision_blobs') as draw:
        PaintAnalysisApp._draw_gallery_display_overlays(app, Figure().subplots(), result, 0,
            np.array([[0., 0.], [10., 0.]]), lambda x: x)
    measure.assert_not_called()
    np.testing.assert_equal(draw.call_args.args[3]['digital_pixel_probabilities'], [[.9, .2]])


def test_pre_aligned_classification_pose_is_preserved_even_when_mirror_allowed():
    grid = np.array([[-10., 0.], [0., 0.], [10., 0.]])
    points = np.repeat(grid, 4, axis=0) + [1.5, .5]
    kwargs = dict(rows=1, columns=3, spacing_x_nm=10, spacing_y_nm=10,
        site_radius_nm=3, grid_points_nm=grid, prealigned=True, use_g5m=False,
        allow_mirror=True, symmetrize_180=False)
    locked = align_picked_origamis([points], preserve_pose=True, **kwargs)
    np.testing.assert_array_equal(locked.aligned_points[0], points)
    refined = align_picked_origamis([points], **kwargs)
    assert not np.array_equal(refined.aligned_points[0], points)
