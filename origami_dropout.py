"""Fixed-mixture latent-template models for saved binary patterns; never reassigns candidates."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from origami_qc import expected_template_fractions

MODEL_NAMES = ('Shared error rates', 'Group-specific error rates', 'Group rates + full penalty')


def _fit(patterns, counts, templates, weights, kind):
    """MLE with ON >= false-ON, analytic gradients, and multiple starting points."""
    t, g = templates.shape
    has_off = ~templates.all(axis=0)
    on_index = np.zeros(g, int) if kind == 0 else np.arange(g)
    n_on = 1 if kind == 0 else g
    off_groups = np.flatnonzero(has_off)
    n_off = int(bool(len(off_groups))) if kind == 0 else len(off_groups)
    off_index = np.zeros(len(off_groups), int) if kind == 0 else np.arange(len(off_groups))
    full = templates.all(axis=1)
    penalty_mask = full[:, None] & has_off[None, :]
    use_penalty = kind == 2
    size = n_on + n_off + int(use_penalty)

    def probabilities(theta):
        on = expit(theta[on_index])
        ratio = np.zeros(g)
        if n_off:
            ratio[off_groups] = expit(theta[n_on + off_index])
        attenuation = np.ones((t,g))
        if use_penalty:
            attenuation[penalty_mask] = np.exp(-theta[-1])
        # On a full template, shrink variable groups' ON signal toward false-ON.
        factor = np.where(templates, ratio + (1-ratio)*attenuation, ratio)
        q = np.clip(on * factor, 1e-10, 1-1e-10)
        return q, on, ratio, attenuation

    def objective(theta):
        q, on, ratio, attenuation = probabilities(theta)
        likelihood = patterns @ np.log(q).T + (1-patterns) @ np.log1p(-q).T + np.log(weights)
        normalizer = logsumexp(likelihood, axis=1)
        responsibility = np.exp(likelihood-normalizer[:,None]) * counts[:,None]
        observed_on = responsibility.T @ patterns
        totals = responsibility.sum(axis=0)[:,None]
        dq = (observed_on-totals*q) / (q*(1-q))
        gradient = np.zeros(size)
        for j in range(g):
            gradient[on_index[j]] += np.sum(dq[:,j] * q[:,j] * (1-on[j]))
        for k,j in enumerate(off_groups):
            derivative = on[j]*ratio[j]*(1-ratio[j])*np.where(templates[:,j], 1-attenuation[:,j], 1)
            gradient[n_on+off_index[k]] += np.sum(dq[:,j]*derivative)
        if use_penalty:
            gradient[-1] = np.sum(dq * (-on*(1-ratio)*attenuation) * penalty_mask)
        return -float(counts @ normalizer), -gradient

    bounds = [(-9,9)]*(n_on+n_off) + ([(0,9)] if use_penalty else [])
    fits = []
    for sensitivity, false_on in ((.9,.03),(.65,.08),(.98,.005)):
        on_start = np.log(sensitivity/(1-sensitivity))
        r = false_on/sensitivity
        theta = np.r_[np.full(n_on,on_start), np.full(n_off,np.log(r/(1-r))), ([.1] if use_penalty else [])]
        fit = minimize(objective, theta, jac=True, bounds=bounds, method='L-BFGS-B', options={'maxiter':500, 'ftol':1e-10})
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise ValueError('Dropout model fitting did not converge; no interpretation was produced.')
    fit = min(fits, key=lambda x:x.fun)
    q, on, ratio, _ = probabilities(fit.x)
    return dict(q=q, on=on, false_on=on*ratio, size=size, log_likelihood=-fit.fun,
                full_attenuation=float(np.exp(-fit.x[-1])) if use_penalty else 1.,
                boundary=bool(any(abs(v-lo)<.001 or abs(v-hi)<.001 for v,(lo,hi) in zip(fit.x,bounds))))


def _pattern_probabilities(patterns, q, weights):
    return np.exp(logsumexp(patterns @ np.log(q).T + (1-patterns) @ np.log1p(-q).T + np.log(weights), axis=1))


def fit_dropout_models(results, audit, seed=1729):
    ids = tuple(audit['metadata']['group_order'])
    g = len(ids)
    if g > 12:
        raise ValueError('The exhaustive pattern check supports up to 12 digital groups.')
    names = list(results)
    templates = []
    for name in names:
        model = results[name]['params']['logical_model']
        model_ids = tuple(map(str,model['bit_ids']))
        templates.append(np.asarray(model['active_bits'], bool)[[model_ids.index(bit) for bit in ids]])
    templates = np.asarray(templates)
    if templates.sum() == 0 or templates.all() or np.sum(templates.all(axis=1)) != 1:
        raise ValueError('Load distinct coded templates including exactly one all-ON full template.')
    if len(np.unique(templates,axis=0)) != len(templates):
        raise ValueError('Dropout comparison requires distinct template patterns.')
    candidates = audit['candidates']
    valid = candidates.status != 'invalid'
    observed = np.array([[int(bit) for bit in p] for p in candidates.loc[valid,'pattern']], dtype=float)
    n = len(observed)
    if n < max(30, 3*g):
        raise ValueError('At least 30 valid candidates (and three per group) are required for this exploratory model check.')
    powers = 2**np.arange(g-1,-1,-1)
    codes = (observed @ powers).astype(int)
    patterns = ((np.arange(2**g)[:,None] & powers) > 0).astype(float)
    counts = np.bincount(codes,minlength=len(patterns))
    weights = expected_template_fractions(names)
    # Deterministic folds include every candidate exactly once as held-out data.
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order,3)
    records, rates, predictions, residuals = [], [], [], []
    log_scores = []
    full_index = int(np.flatnonzero(templates.all(axis=1))[0])
    exact_codes = templates @ powers
    matched = np.isin(np.arange(len(patterns)),exact_codes)
    for kind, name in enumerate(MODEL_NAMES):
        fit = _fit(patterns,counts,templates,weights,kind)
        probability = _pattern_probabilities(patterns,fit['q'],weights)
        held_out = np.zeros(n)
        for indices in folds:
            test_counts = np.bincount(codes[indices],minlength=len(patterns))
            trained = _fit(patterns,counts-test_counts,templates,weights,kind)
            predicted = _pattern_probabilities(patterns,trained['q'],weights)
            held_out[indices] = np.log(np.maximum(predicted[codes[indices]],1e-300))
        log_scores.append(held_out)
        records.append(dict(model=name, parameters=fit['size'], log_likelihood=fit['log_likelihood'],
            bic=fit['size']*np.log(n)-2*fit['log_likelihood'], held_out_log_score=float(held_out.mean()),
            full_signal_retained=fit['full_attenuation'], parameter_at_boundary=fit['boundary'], candidates=n))
        for j,bit in enumerate(ids):
            rates.append(dict(model=name,group_id=bit,on_detection=fit['on'][j],on_dropout=1-fit['on'][j],
                false_on=fit['false_on'][j] if not templates[:,j].all() else np.nan,
                full_on_detection=fit['q'][full_index,j]))
        for label, code in zip(names,exact_codes):
            predictions.append(dict(model=name,pattern_category=label,observed=int(counts[code]),predicted=n*probability[code]))
        predictions.append(dict(model=name,pattern_category='No exact template',observed=int(counts[~matched].sum()),predicted=n*probability[~matched].sum()))
        for index, pattern in enumerate(patterns):
            residuals.append(dict(model=name,pattern=''.join(str(int(v)) for v in pattern),observed=int(counts[index]),
                predicted=float(n*probability[index]),residual=float(counts[index]-n*probability[index])))
    models = pd.DataFrame(records)
    for k in range(len(models)):
        difference = log_scores[k]-log_scores[1]
        models.loc[k,'held_out_gain_vs_group'] = difference.mean()
        models.loc[k,'gain_standard_error'] = difference.std(ddof=1)/np.sqrt(n)
    return dict(dropout_models=models,dropout_group_rates=pd.DataFrame(rates),
        dropout_pattern_counts=pd.DataFrame(predictions),dropout_pattern_residuals=pd.DataFrame(residuals),
        dropout_held_out_scores=pd.DataFrame({'candidate_id': candidates.loc[valid,'candidate_id'].to_numpy(),
                                            **dict(zip(MODEL_NAMES,log_scores))}))


def plot_dropout_models(figure, tables):
    """Compact, resize-aware layout: charts, separate tables, and bounded notes."""
    figure.clear()
    figure.set_layout_engine('constrained', h_pad=.10, w_pad=.10, hspace=.08, wspace=.08)
    layout = figure.add_gridspec(3, 2, height_ratios=[1, 1, .24])
    count_axis = figure.add_subplot(layout[0, 0])
    rate_axis = figure.add_subplot(layout[0, 1])
    score_axis = figure.add_subplot(layout[1, 0])
    table_layout = layout[1, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=.35)
    model_axis = figure.add_subplot(table_layout[0])
    residual_axis = figure.add_subplot(table_layout[1])
    note_axis = figure.add_subplot(layout[2, :])
    models = tables['dropout_models']
    rates = tables['dropout_group_rates']
    predictions = tables['dropout_pattern_counts']
    short_names = ('Shared', 'By group', 'By group + full')
    for name, short in zip(MODEL_NAMES, short_names):
        rows = predictions[predictions.model == name]
        count_axis.plot(np.arange(len(rows)), rows.predicted, marker='.', label=short, linewidth=1.3)
    count_axis.plot(np.arange(len(rows)), rows.observed, 'ko', label='Observed', markersize=5)
    count_axis.set(xticks=np.arange(len(rows)),
                   xticklabels=[str(v).replace('No exact template', 'No exact\ntemplate') for v in rows.pattern_category],
                   ylabel='Candidates')
    count_axis.set_title('Exact patterns before QC rejection', fontsize=11)
    count_axis.tick_params(axis='x', rotation=0)
    count_axis.legend(fontsize=8, ncol=2, loc='best')

    group = rates[rates.model == MODEL_NAMES[1]]
    rate_axis.plot(group.group_id, group.on_dropout, 'o-', label='ON → OFF', markersize=4)
    rate_axis.plot(group.group_id, group.false_on, 'o-', label='OFF → ON', markersize=4)
    rate_axis.set(ylim=(0, 1), ylabel='Estimated probability')
    rate_axis.set_title('Group error estimates (shared across templates)', fontsize=11)
    rate_axis.legend(fontsize=8, ncol=2, loc='upper right')

    score_axis.errorbar(models.held_out_gain_vs_group, np.arange(3),
                        xerr=2*models.gain_standard_error, fmt='o', color='#0891b2',
                        ecolor='#374151', capsize=3, markersize=5)
    score_axis.axvline(0, color='gray', linestyle='--', linewidth=1)
    score_axis.set(yticks=np.arange(3), yticklabels=short_names, ylim=(-.5, 2.5),
                   xlabel='Held-out gain per candidate (higher is better)')
    score_axis.set_title('Prediction gain over By group; ±2 approximate SE', fontsize=11)

    def add_table(axis, title, headings, values, widths):
        axis.set_axis_off()
        axis.set_title(title, fontsize=10, loc='left', pad=5)
        table = axis.table(cellText=values, colLabels=headings, colWidths=widths,
                           cellLoc='left', bbox=[0, 0, 1, .96])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor('#d1d5db')
            cell.set_linewidth(.5)
            if row == 0:
                cell.set_facecolor('#f1f5f9')
                cell.set_text_props(weight='bold')
        return table

    add_table(model_axis, 'Model comparison — lower BIC is better',
              ['Model', 'BIC', 'ΔBIC', 'Boundary fit'],
              [[short, f'{r.bic:.1f}', f'{r.bic-models.bic.min():.1f}', 'Yes' if r.parameter_at_boundary else 'No']
               for short, r in zip(short_names, models.itertuples())], [.43, .18, .16, .23])
    residuals = tables['dropout_pattern_residuals']
    group_residuals = residuals[residuals.model == MODEL_NAMES[1]]
    largest = group_residuals.loc[group_residuals.residual.abs().nlargest(3).index]
    add_table(residual_axis, 'Largest pattern discrepancies — By group',
              ['Saved bit pattern', 'Observed', 'Predicted'],
              [[r.pattern, str(r.observed), f'{r.predicted:.1f}'] for r in largest.itertuples()], [.50, .25, .25])
    note_axis.set_axis_off()
    retained = models.iloc[2].full_signal_retained
    # Three short lines in their own layout cell avoid overflow on smaller canvases.
    notes = (
        f'Full term: {retained:.0%} of ON signal above false-ON retained (model parameter, not physical yield).',
        'Shared: common error rates. By group: group-specific rates. + full: additional full-specific loss.',
        'Assumes fixed mixture and independent group calls. Detection bias, damage and correlated loss are not modeled.'
    )
    for y, line in zip((.95, .6, .25), notes):
        note_axis.text(0, y, line, va='top', fontsize=8, transform=note_axis.transAxes)
    for axis in (count_axis, rate_axis, score_axis):
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    figure.suptitle(f'ON-dropout model check · {int(models.iloc[0].candidates):,} valid candidates · code3 ×2', fontsize=12)
