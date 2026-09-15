import unittest

import numpy as np

from origami_analysis import classify_template_candidates
from paint_analysis_gui import exact_digital_template_matches


class ExactDigitalClassificationTests(unittest.TestCase):
    def params(self, expected, probabilities, ids=('a', 'b'), template_ids=('a', 'b')):
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1, 2)
        return dict(logical_model=dict(bit_ids=template_ids, active_bits=expected),
                    digital_pixel_ids=ids, digital_pixel_probabilities=probabilities,
                    digital_group_localization_evidence=np.full(probabilities.shape, 5.),
                    digital_group_prominences=np.full(probabilities.shape, .8),
                    min_site_localizations=3, min_site_evidence=.1)

    def test_full_pattern_required_and_ids_matched_by_name(self):
        params = self.params((True, False), [[.9, .1], [.9, .9], [.1, .1], [.5, .49]])
        np.testing.assert_array_equal(exact_digital_template_matches(params, 4), [True, False, False, True])
        reordered = self.params((False, True), [[.9, .1]], template_ids=('b', 'a'))
        self.assertTrue(exact_digital_template_matches(reordered, 1)[0])

    def test_support_and_prominence_control_the_displayed_on_off_states(self):
        params = self.params((False, False), [[.9, .9]])
        params['digital_group_localization_evidence'][0, 0] = 2
        params['digital_group_prominences'][0, 1] = .09
        self.assertTrue(exact_digital_template_matches(params, 1)[0])
        self.assertEqual(params['classification_observed_digital_states'], ((False, False),))
        params['digital_pixel_probabilities'][0, 0] = np.nan
        self.assertFalse(exact_digital_template_matches(params, 1)[0])

    def test_single_template_never_accepts_an_unknown_pattern(self):
        params = self.params((True, False), [[.99, .99], [.6, .4]])
        mask = exact_digital_template_matches(params, 2)
        result = classify_template_candidates([np.array([[0., 0.], [100., 0.]])],
                    [mask], [np.where(mask, 0., -np.inf)], match_distance_nm=1., require_unique_match=True)
        np.testing.assert_array_equal(result.winning_template_indices, [-1, 0])
        self.assertEqual(result.unclassified_count, 1)

    def test_unknown_and_ambiguous_patterns_remain_unclassified(self):
        probabilities = [[.9, .1], [.1, .9], [.9, .9]]
        masks = [exact_digital_template_matches(self.params(pattern, probabilities), 3)
                 for pattern in [(True, False), (False, True), (True, False)]]
        centers = np.array([[0., 0.], [100., 0.], [200., 0.]])
        result = classify_template_candidates([centers] * 3, masks,
                    [np.where(mask, 0., -np.inf) for mask in masks],
                    match_distance_nm=1., require_unique_match=True)
        np.testing.assert_array_equal(result.winning_template_indices, [-1, 1, -1])
        self.assertEqual(result.counts.tolist(), [0, 1, 0])

    def test_missing_measurements_and_incompatible_ids_are_rejected(self):
        params = self.params((True, False), [[.9, .1]], template_ids=('a', 'c'))
        with self.assertRaisesRegex(ValueError, 'IDs'):
            exact_digital_template_matches(params, 1)
        params = self.params((True, False), [[.9, .1]])
        params.pop('digital_pixel_probabilities')
        with self.assertRaisesRegex(ValueError, 'Step 3'):
            exact_digital_template_matches(params, 1)


if __name__ == '__main__':
    unittest.main()
