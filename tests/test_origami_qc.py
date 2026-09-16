import copy
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from origami_analysis import digital_group_template_evidence
from origami_qc import build_classification_audits, export_classification_audits, plot_classification_audit
from paint_analysis_gui import exact_digital_template_matches


def fixture_results():
    # Assigned empty, rejected full, unmatched 10, and invalid (not empty).
    probability = np.array([[.1, .1], [.9, .9], [.9, .1], [np.nan, .1]])
    params = dict(digital_pixel_ids=('a', 'b'), digital_pixel_probabilities=probability,
        digital_group_localization_evidence=np.full((4, 2), 5.),
        digital_group_prominences=np.full((4, 2), .8), min_site_localizations=3,
        min_site_evidence=.1)
    regions = [np.array([[i, 0.], [i, 1.]]) for i in range(4)]
    results = {}
    for name, bits, mask in [('empty', (False, False), [1, 0, 0, 0]),
                              ('full', (True, True), [0, 0, 0, 0])]:
        results[name] = dict(params=dict(params, logical_model=dict(bit_ids=('a', 'b'), active_bits=bits)),
            picks=SimpleNamespace(regions=regions, accepted_mask=np.array(mask, dtype=bool)))
    return results


def test_states_match_classifier_and_shared_candidates_count_once():
    results = fixture_results()
    original = copy.deepcopy(results)
    audit = build_classification_audits(results)
    assert audit['candidates'].status.tolist() == ['assigned', 'rejected_exact', 'unmatched', 'invalid']
    assert len(audit['candidates']) == 4
    assert len(audit['candidate_groups']) == 8
    for name, payload in results.items():
        matches = exact_digital_template_matches(copy.deepcopy(payload['params']), 4)
        rows = audit['candidate_template_distances']
        got = rows[(rows.template == name) & (rows.distance == 0)].candidate_id.tolist()
        assert got == (np.flatnonzero(matches) + 1).tolist()
        assert set(payload['params']) == set(original[name]['params'])
        np.testing.assert_array_equal(payload['picks'].accepted_mask, original[name]['picks'].accepted_mask)
    assert audit['metadata']['invalid_count'] == 1
    assert audit['group_summary'].iloc[0].candidates == 3


def test_unmatched_ties_and_missing_extra_are_not_assignments():
    audit = build_classification_audits(fixture_results())
    pattern = audit['unmatched_patterns'].iloc[0]
    assert pattern['pattern'] == '10'
    assert pattern.nearest_templates == 'empty;full'
    assert pattern.distance == 1
    rows = audit['candidate_template_distances'].query('candidate_id == 3').set_index('template')
    assert rows.loc['full', 'missing_groups'] == 'b'
    assert rows.loc['empty', 'extra_groups'] == 'a'
    assert audit['candidates'].iloc[2].assigned_template == ''
    assert audit['full_dropout_comparisons'].iloc[0].absent_groups == 'a;b'


def test_bit_ids_reorder_measurements_and_template_states():
    results = fixture_results()
    p = results['full']['params']
    p['digital_pixel_ids'] = ('b', 'a')
    p['logical_model'] = dict(bit_ids=('b', 'a'), active_bits=(True, True))
    for key in ('digital_pixel_probabilities', 'digital_group_localization_evidence', 'digital_group_prominences'):
        p[key] = p[key][:, ::-1]
    assert build_classification_audits(results)['unmatched_patterns'].iloc[0]['pattern'] == '10'


def test_exposes_support_pass_relative_prominence_failure():
    evidence = np.array([[100.] * 7 + [5.]])
    cells = [[i] for i in range(8)]
    _, _, relative, _ = digital_group_template_evidence(evidence, [False]*8, cells)
    _, _, probability, _ = digital_group_template_evidence(evidence, [False]*8, cells,
        minimum_support_per_position=3, minimum_group_prominence=.1)
    params = dict(digital_pixel_ids=tuple('abcdefgh'), logical_model=dict(bit_ids=tuple('abcdefgh'), active_bits=[True]*8),
        digital_pixel_probabilities=probability, digital_group_localization_evidence=evidence,
        digital_group_prominences=np.maximum(0, 2*relative-1), min_site_localizations=3, min_site_evidence=.1)
    results = {'full': dict(params=params, picks=SimpleNamespace(regions=[np.zeros((2, 2))], accepted_mask=np.array([False])))}
    audit = build_classification_audits(results)
    weak = audit['candidate_groups'].iloc[-1]
    assert weak.support_pass and weak.probability_pass
    assert weak.support_pass_prominence_fail
    assert not weak.final_on
    assert audit['candidate_template_distances'].iloc[0].missing_groups == 'h'


@pytest.mark.parametrize('change', ['threshold', 'ordering', 'measurement', 'assignment'])
def test_rejects_inconsistent_snapshots(change):
    results = fixture_results()
    if change == 'threshold':
        results['full']['params']['min_site_evidence'] = .2
    elif change == 'ordering':
        results['full']['picks'].regions = results['full']['picks'].regions[::-1]
    elif change == 'measurement':
        results['full']['params']['digital_pixel_probabilities'] = np.zeros((4, 2))
    else:
        results['full']['picks'].accepted_mask[0] = True
    with pytest.raises(ValueError):
        build_classification_audits(results)


def test_export_and_plots(tmp_path):
    audit = build_classification_audits(fixture_results())
    path = tmp_path / 'audit.zip'
    export_classification_audits(audit, path)
    with ZipFile(path) as archive:
        assert len(archive.namelist()) == 7
        assert json.loads(archive.read('metadata.json'))['group_order'] == ['a', 'b']
        assert b'support_pass_prominence_fail' in archive.read('candidate_groups.csv')
    for view in ('Digital-group threshold audit', 'Unmatched-pattern audit'):
        figure = Figure(figsize=(12, 8))
        FigureCanvasAgg(figure)
        plot_classification_audit(figure, audit, view)
        figure.canvas.draw()


def test_empty_candidates_and_no_unmatched_plot():
    results = fixture_results()
    for payload in results.values():
        payload['picks'].regions = []
        payload['picks'].accepted_mask = np.zeros(0, dtype=bool)
        for key in ('digital_pixel_probabilities', 'digital_group_localization_evidence', 'digital_group_prominences'):
            payload['params'][key] = np.empty((0, 2))
    audit = build_classification_audits(results)
    assert len(audit['candidates']) == 0
    assert audit['group_summary'].iloc[0].candidates == 0
    figure = Figure()
    plot_classification_audit(figure, audit, 'Unmatched-pattern audit')


def test_sample_mixture_reference_weights_and_group_id_order():
    from origami_qc import expected_template_fractions, expected_group_fractions
    names = ['align_fid', 'code1', 'code2', 'Code_3', 'code4', 'full']
    np.testing.assert_allclose(expected_template_fractions(names), np.array([1, 1, 1, 2, 1, 1])/7)
    patterns = [(0, 0), (0, 1), (0, 0), (1, 0), (0, 0), (1, 1)]
    results = {name: {'params': {'logical_model': {'bit_ids': ('a', 'b'), 'active_bits': bits},
                                'digital_group_expected_on_fractions': (0, 0)}}
               for name, bits in zip(names, patterns)}
    np.testing.assert_allclose(expected_group_fractions(results, ('b', 'a')), [2/7, 3/7])


def test_audit_export_uses_weighted_mixture():
    saved = fixture_results()
    saved['code3'] = saved.pop('full')
    audit = build_classification_audits(saved)
    assert audit['metadata']['expected_template_fractions'] == {'empty': 1/3, 'code3': 2/3}
    assert audit['group_summary'].iloc[0].mixture_expected_on_fraction == pytest.approx(2/3)
