import numpy as np

from paint_analysis_gui import (
    classification_bias_diagnostics,
    unclassified_template_attribution,
)


def test_unmatched_scores_do_not_choose_the_last_template():
    attempts = [(-np.inf, index, 0) for index in range(6)]
    for ordering in (attempts, attempts[::-1]):
        _, status = unclassified_template_attribution(ordering)
        assert status == 'no_match'


def test_tied_scores_have_no_unique_best_template():
    _, status = unclassified_template_attribution([(0., 0, 0), (0., 5, 0)])
    assert status == 'ambiguous'
    selected, status = unclassified_template_attribution([(-np.inf, 0, 0), (0., 5, 0)])
    assert status == 'unique_best'
    assert selected == (0., 5, 0)


def test_diagnostics_separate_unmatched_ties_and_real_rejections_including_old_records():
    details = [
        {'template_name': 'full', 'classification_score': -np.inf,
         'failure_reasons': ('digital ON/OFF pattern has no exact template match',)},
        {'template_name': '', 'template_attribution': 'no_match'},
        {'template_name': '', 'template_attribution': 'ambiguous'},
        {'template_name': 'full', 'classification_score': 0.,
         'failure_reasons': ('ambiguous digital ON/OFF pattern: multiple templates match',)},
        {'template_name': 'full', 'classification_score': 0.,
         'failure_reasons': ('corr 0.1 < 0.3',)},
    ]
    result = classification_bias_diagnostics(['empty', 'full'], [40, 14], details, 6)
    np.testing.assert_array_equal(result['rejected_best'], [0, 1])
    np.testing.assert_allclose(result['assignment_rates'], [1., 14/15])
    assert result['unmatched_count'] == 2
    assert result['ambiguous_count'] == 2
    assert result['unresolved_unclassified'] == 1
    assert result['failure_counts'].sum() == 1
