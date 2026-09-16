import copy
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from origami_analysis import digital_group_template_evidence
from origami_review import review_tables, review_mask, threshold_sweep, plot_evidence, plot_sweep, plot_full_detail


def fixture():
    # full, near-full unmatched, empty, QC-rejected full, invalid
    evidence = np.array([[8,8], [8,2.5], [0,0], [8,8], [np.nan,0.]])
    relative = np.array([[.9,.9], [.9,.9], [0,0], [.9,.9], [np.nan,0.]])
    probability = np.full_like(evidence, np.nan)
    probability[:4] = digital_group_template_evidence(evidence[:4], [False]*2, [[0],[1]], minimum_support_per_position=3, minimum_group_prominence=.1)[2]
    regions = [np.array([[i*100.,0.], [i*100.,1.]]) for i in range(5)]
    model = dict(bit_ids=('a','b'), bit_physical_cells=((0,),(1,)), active_bits=(0,0))
    params = dict(digital_pixel_ids=('a','b'), digital_pixel_model=model, digital_pixel_probabilities=probability,
        digital_group_localization_evidence=evidence, digital_group_prominences=relative,
        min_site_localizations=3, min_site_evidence=.1, classification_qc_eligible=(1,1,1,0,1))
    results = {}
    for name, bits, mask in [('empty',(0,0),[0,0,1,0,0]), ('full',(1,1),[1,0,0,0,0])]:
        picks = SimpleNamespace(regions=regions, aligned_regions=[np.array([[0.,0.],[10.,0.]])]*5,
            template_points_nm=np.array([[0.,0.],[10.,0.]]), accepted_mask=np.array(mask,bool))
        results[name] = dict(picks=picks, params=dict(params, logical_model=dict(model, active_bits=bits)))
    details = [dict(center_nm=[300.,.5], failure_reasons=['corr 0.2 < 0.4'])]
    return results, details


def test_review_filters_are_disjoint_by_primary_status_and_keep_failure_subgroups():
    results, details = fixture()
    audit = review_tables(results, details)
    assert review_mask(audit['candidates'], 'All unclassified').tolist() == [False,True,False,True,True]
    assert review_mask(audit['candidates'], 'QC: correlation').tolist() == [False,False,False,True,False]
    assert review_mask(audit['candidates'], 'No exact match: 1 groups differ').tolist() == [False,True,False,False,False]
    assert review_mask(audit['candidates'], 'Invalid measurements').sum() == 1


def test_sweep_recovers_near_full_but_never_rescues_fixed_qc_reject():
    results, _ = fixture()
    before = copy.deepcopy(results)
    rows, counts, base = threshold_sweep(results, 'support', [2,3,10])
    assert base == 3
    assert rows.query('threshold == 2 and candidate_id == 2').iloc[0].transition == 'recovered'
    assert rows.query('candidate_id == 4').predicted.unique().tolist() == ['Unclassified']
    assert rows.query('candidate_id == 5').predicted.unique().tolist() == ['Unclassified']
    assert rows.query('threshold == 10 and candidate_id == 1').iloc[0].transition == 'switched'
    assert (rows.query('threshold == 3').transition == 'unchanged').all()
    np.testing.assert_equal(results['full']['params']['digital_pixel_probabilities'], before['full']['params']['digital_pixel_probabilities'])
    np.testing.assert_equal(results['full']['picks'].accepted_mask, before['full']['picks'].accepted_mask)


def test_missing_eligibility_and_inconsistent_baseline_refuse_predictions():
    results, _ = fixture()
    results['full']['params'].pop('classification_qc_eligible')
    with pytest.raises(ValueError, match='eligibility'):
        threshold_sweep(results)
    results, _ = fixture()
    results['full']['params']['classification_qc_eligible'] = (1,1,1,1,1)
    with pytest.raises(ValueError, match='baseline'):
        threshold_sweep(results)


def test_invalid_sweep_values_and_prominence_sweep():
    results, _ = fixture()
    for values in ([np.nan],[-1],[1.1]):
        with pytest.raises(ValueError):
            threshold_sweep(results, 'prominence', values)
    rows, _, _ = threshold_sweep(results, 'prominence', [0,.1,1])
    assert (rows.query('threshold == .1').transition == 'unchanged').all()


def test_new_plots_render():
    results, details = fixture()
    audit = review_tables(results, details)
    for draw in (lambda f: plot_evidence(f,audit), lambda f: plot_full_detail(f,results,audit,2),
                 lambda f: plot_sweep(f,*threshold_sweep(results,'support',[2,3,4]),'support')):
        figure = Figure(figsize=(13,8))
        FigureCanvasAgg(figure)
        draw(figure)
        figure.canvas.draw()


def test_sweep_tracks_lost_assignments_and_keeps_ambiguous_matches_unassigned():
    results, _ = fixture()
    for payload in results.values():
        evidence = payload['params']['digital_group_localization_evidence'].copy()
        evidence[0] = [4,8]
        payload['params']['digital_group_localization_evidence'] = evidence
        probability = payload['params']['digital_pixel_probabilities'].copy()
        probability[0] = digital_group_template_evidence(evidence[:1], [False]*2, [[0],[1]],
            minimum_support_per_position=3, minimum_group_prominence=.1)[2][0]
        payload['params']['digital_pixel_probabilities'] = probability
    rows, _, _ = threshold_sweep(results, 'support', [3,5])
    assert rows.query('candidate_id == 1 and threshold == 5').iloc[0].transition == 'lost'
    results, _ = fixture()
    results['full']['picks'].accepted_mask[:] = False
    results['duplicate_full'] = copy.deepcopy(results['full'])
    rows, _, _ = threshold_sweep(results, 'support', [2,3])
    assert (rows.query('candidate_id == 1').predicted == 'Unclassified').all()


def test_near_full_filter_and_exportable_transition_ids():
    results, details = fixture()
    audit = review_tables(results, details)
    assert review_mask(audit['candidates'], 'Near full: 1–2 groups OFF').tolist() == [False,True,False,False,False]
    rows, _, _ = threshold_sweep(results, values=[2,3])
    assert rows.query("transition == 'recovered'").candidate_id.unique().tolist() == [2]


def test_gallery_filter_maps_original_ids_before_pagination():
    from unittest import mock
    from paint_analysis_gui import PaintAnalysisApp
    class Var:
        def __init__(self, value): self.value = value
        def get(self): return self.value
        def set(self, value): self.value = value
    results, details = fixture()
    audit = review_tables(results, details)
    app = SimpleNamespace(origami_result=SimpleNamespace(origami_count=3),
        origami_identification_params={'_unclassified_display':True},
        origami_pick_result=SimpleNamespace(accepted_mask=np.array([0,1,0,1,1],bool)),
        origami_gallery_min_match=Var(''), origami_gallery_max_rms=Var(''),
        origami_gallery_page_size=Var(25), origami_gallery_sort=Var('Origami ID'),
        origami_review_filter=Var('QC: correlation'), origami_review_combo=mock.Mock(),
        _get_origami_review=lambda: audit, origami_gallery_page=Var(1), origami_gallery_page_label=Var(''))
    with mock.patch('paint_analysis_gui.origami_gallery_indices', return_value=np.array([2,1,0])):
        indices, count = PaintAnalysisApp._origami_gallery_page_data(app)
    assert indices.tolist() == [1]  # gallery index 1 is original candidate #4
    assert count == 1
