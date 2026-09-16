import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from origami_dropout import _fit, _pattern_probabilities, fit_dropout_models, plot_dropout_models, MODEL_NAMES
from origami_qc import expected_template_fractions


def fixture(n=2400, full_penalty=1, seed=41):
    templates = np.array([[1,0,0,0,0],[1,1,0,1,0],[1,0,1,0,1],[1,1,1,1,0],[1,1,1,0,0],[1,1,1,1,1]],bool)
    names = ['align_fid','code1','code2','code3','code4','full']
    q = np.where(templates, [.98,.87,.9,.82,.86], .025)
    q[-1,1:] = .025+(q[-1,1:]-.025)*full_penalty
    rng = np.random.default_rng(seed)
    true = rng.choice(len(names),n,p=expected_template_fractions(names))
    states = rng.random((n,5)) < q[true]
    patterns = [''.join('1' if bit else '0' for bit in row) for row in states]
    results = {name: {'params': {'logical_model':{'bit_ids':tuple('abcde'),'active_bits':bits}}} for name,bits in zip(names,templates)}
    audit = {'metadata':{'group_order':tuple('abcde')}, 'candidates':pd.DataFrame({'candidate_id':np.arange(1,n+1),'status':['unmatched']*n,'pattern':patterns})}
    return results,audit,templates,q


def test_shared_group_dropout_recovers_rates_without_full_penalty():
    results,audit,templates,q = fixture()
    tables = fit_dropout_models(results,audit)
    models = tables['dropout_models']
    assert models.iloc[1].bic < models.iloc[0].bic
    assert models.iloc[1].bic < models.iloc[2].bic
    rates = tables['dropout_group_rates'][tables['dropout_group_rates'].model == MODEL_NAMES[1]]
    np.testing.assert_allclose(rates.on_detection, [.98,.87,.9,.82,.86], atol=.06)
    assert np.isnan(rates.iloc[0].false_on) # No OFF examples exist for the common alignment group.
    for _, rows in tables['dropout_pattern_counts'].groupby('model'):
        assert rows.predicted.sum() == pytest.approx(len(audit['candidates']))
    assert len(tables['dropout_held_out_scores']) == len(audit['candidates'])
    figure = Figure(figsize=(13,9)); FigureCanvasAgg(figure)
    plot_dropout_models(figure,tables)
    figure.canvas.draw()


def test_full_specific_effect_improves_held_out_prediction():
    results,audit,_,_ = fixture(n=5000,full_penalty=.3)
    tables = fit_dropout_models(results,audit)
    models = tables['dropout_models']
    assert models.iloc[2].bic < models.iloc[1].bic
    assert models.iloc[2].held_out_gain_vs_group > 0
    assert models.iloc[2].full_signal_retained < .6


def test_invalid_rows_excluded_and_small_samples_refused():
    results,audit,_,_ = fixture(n=20)
    with pytest.raises(ValueError,match='30 valid'):
        fit_dropout_models(results,audit)
    audit['candidates'].loc[0,['status','pattern']] = ['invalid','']
    with pytest.raises(ValueError,match='30 valid'):
        fit_dropout_models(results,audit)


def test_fitted_pattern_probabilities_normalize():
    _,_,templates,q = fixture()
    patterns = ((np.arange(32)[:,None] & (2**np.arange(4,-1,-1)))>0).astype(float)
    weights = expected_template_fractions(['a','b','c','code3','d','full'])
    probability = _pattern_probabilities(patterns,q,weights)
    assert probability.sum() == pytest.approx(1)
    fit = _fit(patterns, probability*10000, templates,weights,1)
    np.testing.assert_allclose(fit['q'],q,atol=.002)
