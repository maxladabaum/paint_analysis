"""Read-only investigation of unclassified candidates and threshold sensitivity."""
import numpy as np
import pandas as pd
from origami_qc import build_classification_audits
from origami_analysis import digital_group_template_evidence


def review_tables(results, details=()):
    audit = build_classification_audits(results)
    candidates = audit['candidates'].copy()
    distances = audit['candidate_template_distances']
    lookup = {tuple(np.round(d['center_nm'], 6)): d for d in details if 'center_nm' in d}
    first = next(iter(results.values()))
    groups = []
    reasons = []
    minimum = []
    for row in candidates.itertuples():
        center = np.median(first['picks'].regions[row.candidate_id - 1], axis=0)
        detail = lookup.get(tuple(np.round(center, 6)), {})
        failures = '; '.join(detail.get('failure_reasons', ()))
        subset = distances[distances.candidate_id == row.candidate_id]
        distance = int(subset.distance.min()) if len(subset) else -1
        minimum.append(distance)
        group = {'assigned': 'Assigned', 'unmatched': f'No exact match: {distance} groups differ',
                 'rejected_exact': 'Exact match: QC rejected', 'ambiguous_exact': 'Ambiguous exact matches',
                 'invalid': 'Invalid measurements'}[row.status]
        groups.append(group)
        reasons.append(failures or group)
    candidates['review_group'] = groups
    candidates['rejection_reason'] = reasons
    candidates['nearest_distance'] = minimum
    full_names = [name for name, p in results.items() if all(p['params']['logical_model']['active_bits'])]
    full_distances = distances[distances.template.isin(full_names)].groupby('candidate_id').distance.min()
    candidates['full_distance'] = candidates.candidate_id.map(full_distances)
    audit['candidates'] = candidates
    return audit


def review_filter_options(audit):
    options = ['All unclassified', 'Exact match: QC rejected', 'No exact match',
               'Ambiguous exact matches', 'Invalid measurements', 'Near full: 1–2 groups OFF']
    options += sorted(set(audit['candidates'].review_group) - set(options) - {'Assigned'})
    options += ['QC: correlation', 'QC: overlap', 'QC: point limits', 'QC: other']
    return options


def review_mask(candidates, choice):
    unclassified = candidates.status != 'assigned'
    if choice == 'All unclassified':
        return unclassified.to_numpy()
    if choice == 'Near full: 1–2 groups OFF':
        return (unclassified & candidates.full_distance.between(1, 2)).to_numpy()
    if choice == 'No exact match':
        return (candidates.status == 'unmatched').to_numpy()
    if choice.startswith('QC:'):
        reason = candidates.rejection_reason.str.lower()
        known = reason.str.contains('corr|overlap|point', regex=True)
        key = choice.removeprefix('QC: ')
        mask = (~known if key == 'other' else reason.str.contains(
            {'correlation': 'corr', 'overlap': 'overlap', 'point limits': 'point'}[key]))
        return ((candidates.status == 'rejected_exact') & mask).to_numpy()
    return (unclassified & (candidates.review_group == choice)).to_numpy()


def threshold_sweep(results, parameter='support', values=None):
    """Replay exact digital classification, freezing all non-digital eligibility.

    Thresholds are varied one at a time. No candidate detection/alignment or
    assignment is mutated. Suppression/tile exclusions remain fixed.
    """
    audit = build_classification_audits(results)
    names = list(results)
    first = results[names[0]]['params']
    ids = tuple(first['digital_pixel_ids'])
    n = len(audit['candidates'])
    support = float(first.get('min_site_localizations', 0))
    prominence = float(first.get('min_site_evidence', 0))
    if parameter not in ('support', 'prominence'):
        raise ValueError('Choose support or prominence for the sweep.')
    base = support if parameter == 'support' else prominence
    if values is None:
        values = np.linspace(max(0, base * .5), min(1, max(base * 1.5, .2)) if parameter == 'prominence' else max(base * 1.5, 1), 9)
    values = np.unique(np.r_[np.asarray(values, dtype=float), base])
    if not len(values) or len(values) > 101 or not np.isfinite(values).all() or np.any(values < 0) or (parameter == 'prominence' and np.any(values > 1)):
        raise ValueError('Use up to 100 finite nonnegative thresholds; prominence must be between 0 and 1.')
    evidence = np.asarray(first['digital_group_localization_evidence'], dtype=float).reshape(n, len(ids))
    relative = np.asarray(first['digital_group_prominences'], dtype=float).reshape(n, len(ids))
    valid = audit['candidates'].status.to_numpy() != 'invalid'
    expected, eligible = [], []
    for name in names:
        p = results[name]['params']
        model = p['logical_model']
        model_ids = tuple(model['bit_ids'])
        expected.append(np.asarray(model['active_bits'], bool)[[model_ids.index(bit) for bit in ids]])
        mask = np.asarray(p.get('classification_qc_eligible', p.get('_alignment_accepted_mask', ())), bool)
        if mask.shape != (n,):
            raise ValueError('Saved QC eligibility is missing. Rerun Step 4 (including tiled analysis if applicable) before sweeping thresholds.')
        dispositions = p.get('classification_dispositions', ())
        if len(dispositions) == n:
            mask = mask & np.array([not ('suppressed duplicate' in str(d) or 'tile halo' in str(d)) for d in dispositions])
        eligible.append(mask)
    expected = np.asarray(expected)
    eligible = np.asarray(eligible).T
    baseline = audit['candidates'].assigned_template.replace('', 'Unclassified').to_numpy()
    cells = [[j] for j in range(len(ids))]  # The scoring function uses group evidence; cell IDs do not change ON scores.

    def predict(value):
        low = value if parameter == 'support' else support
        prom = value if parameter == 'prominence' else prominence
        probabilities = np.full_like(evidence, np.nan)
        if np.any(valid):
            probabilities[valid] = digital_group_template_evidence(evidence[valid], [False]*len(ids), cells,
                minimum_support_per_position=low, minimum_group_prominence=prom)[2]
        states = (probabilities >= .5) & (evidence >= low) & (relative >= prom)
        matches = np.all(states[:, None] == expected[None], axis=2) & eligible & valid[:, None]
        labels = np.full(n, 'Unclassified', dtype=object)
        unique = matches.sum(axis=1) == 1
        labels[unique] = np.asarray(names)[matches[unique].argmax(axis=1)]
        return labels

    if not np.array_equal(predict(base), baseline):
        raise ValueError('Sweep baseline does not reproduce saved assignments. Rerun Steps 3 and 4 with shared alignment; no sweep predictions were applied.')
    rows, counts = [], []
    for value in values:
        predicted = predict(float(value))
        for i, (old, new) in enumerate(zip(baseline, predicted)):
            transition = ('unchanged' if old == new else 'recovered' if old == 'Unclassified'
                          else 'lost' if new == 'Unclassified' else 'switched')
            rows.append(dict(candidate_id=i+1, parameter=parameter, threshold=float(value), baseline_threshold=base,
                             original=old, predicted=new, transition=transition))
        counts.append(dict(threshold=float(value), **{name: int(np.sum(predicted == name)) for name in [*names, 'Unclassified']},
                           recovered=int(np.sum((baseline == 'Unclassified') & (predicted != 'Unclassified'))),
                           lost=int(np.sum((baseline != 'Unclassified') & (predicted == 'Unclassified'))),
                           switched=int(np.sum((baseline != 'Unclassified') & (predicted != 'Unclassified') & (baseline != predicted)))))
    return pd.DataFrame(rows), pd.DataFrame(counts), base


def plot_evidence(figure, audit):
    figure.clear()
    ids = audit['metadata']['group_order']
    axes = figure.subplots(len(ids), 2, squeeze=False)
    data = audit['candidate_groups']
    for j, bit in enumerate(ids):
        rows = data[(data.group_id == bit) & data.valid_candidate & (data.status != 'assigned')]
        for k, (field, threshold) in enumerate([('evidence', 'support_threshold'), ('relative_prominence', 'prominence_threshold')]):
            axis = axes[j, k]
            finite = rows[field].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            axis.hist(finite, bins=min(30, max(1, len(finite))), color='#22b8cf')
            value = audit['metadata']['support_threshold' if k == 0 else 'prominence_threshold']
            axis.axvline(value, color='red', linestyle='--')
            if k == 0:
                axis.set_xscale('symlog', linthresh=max(value, 1))
            axis.set_ylabel(str(bit), rotation=0, labelpad=18, fontsize=8)
            axis.tick_params(labelsize=7)
            if j == 0:
                axis.set_title(('Localization support' if k == 0 else 'Relative prominence') + f' (threshold {value:g})', fontsize=10)
    figure.suptitle('Unclassified evidence distributions — red = saved threshold; counts include legitimately OFF groups')
    figure.tight_layout(rect=(0, 0, 1, .95))


def plot_sweep(figure, transitions, counts, baseline, parameter):
    figure.clear()
    left, right = figure.subplots(1, 2)
    for name in counts.columns:
        if name not in ('threshold', 'recovered', 'lost', 'switched'):
            left.plot(counts.threshold, counts[name], marker='.', label=name)
    for name in ('recovered', 'lost', 'switched'):
        right.plot(counts.threshold, counts[name], marker='.', label=name)
    for axis in (left, right):
        axis.axvline(baseline, linestyle='--', color='gray', label='Saved threshold')
        axis.set_xlabel(parameter + ' threshold')
        axis.set_ylabel('Candidates')
        axis.legend(fontsize=8)
    left.set_title('Predicted assignments')
    right.set_title('Changes from saved assignments')
    figure.suptitle('Threshold sensitivity — saved detection, alignment and other QC gates fixed; no changes applied')
    figure.tight_layout(rect=(0, 0, 1, .94))


def plot_full_detail(figure, results, audit, candidate_id, *, grid_points=None, qc_reasons=()):
    figure.clear()
    payload = next(iter(results.values()))
    params, picks = payload['params'], payload['picks']
    index = candidate_id - 1
    left, right = figure.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})
    points = np.asarray(picks.aligned_regions[index])
    left.scatter(points[:, 0], points[:, 1], s=2, color='#343a40', alpha=.4)
    grid = np.asarray(picks.template_points_nm if grid_points is None else grid_points)
    model = params.get('digital_pixel_model') or params['logical_model']
    cells = model.get('bit_physical_cells', model.get('bit_cells', ()))
    from matplotlib.patches import Circle
    radius = float(params.get('site_mask_radius_nm', 5))
    groups = audit['candidate_groups'].query('candidate_id == @candidate_id').set_index('group_id')
    for bit, group in zip(model['bit_ids'], cells):
        sites = grid[np.asarray(group, int)]
        row = groups.loc[str(bit)]
        color = '#9ca3af' if not row.valid_candidate else '#22c55e' if pd.notna(row.final_on) and bool(row.final_on) else '#ef4444'
        for x, y in sites:
            left.add_patch(Circle((x, y), radius, fill=False, color=color, linewidth=.8))
        center = sites.mean(axis=0)
        left.text(*center, str(bit), fontsize=8, color=color)
    left.set(aspect='equal', xlabel='Saved aligned x (nm)', ylabel='Saved aligned y (nm)', title='Original localization positions and measurement regions')
    right.set_axis_off()
    rows = []
    for bit, row in groups.iterrows():
        failed = [name for name, passed in [('support', row.support_pass), ('score', row.probability_pass), ('prominence', row.prominence_pass)] if not passed]
        rows.append([bit, 'invalid' if not row.valid_candidate else 'ON' if row.final_on else 'OFF',
                     f'{row.evidence:.2f} / {row.support_threshold:g}', f'{row.stored_on_score:.2f} / 0.5',
                     f'{row.relative_prominence:.2f} / {row.prominence_threshold:g}', ', '.join(failed)])
    table = right.table(cellText=rows, colLabels=['Group', 'Saved', 'Support / min', 'Score / min', 'Prom. / min', 'Failed gates'],
                        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    full_names = [name for name, p in results.items() if all(p['params']['logical_model']['active_bits'])]
    distances = audit['candidate_template_distances']
    text = []
    for name in full_names:
        row = distances[(distances.candidate_id == candidate_id) & (distances.template == name)]
        text.append(f"{name}: missing ON groups: {row.iloc[0].missing_groups or 'none'}" if len(row) else f'{name}: invalid measurements')
    if not full_names:
        text.append('No all-ON template is loaded.')
    text.extend(qc_reasons)
    candidate = audit['candidates'].iloc[index]
    import textwrap
    right.set_title('\n'.join(textwrap.wrap(' | '.join(text), 80)), fontsize=10)
    figure.suptitle(f"Why wasn't this full? Saved candidate #{candidate_id}\n" + '\n'.join(textwrap.wrap(candidate.rejection_reason, 120)), fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .9))
