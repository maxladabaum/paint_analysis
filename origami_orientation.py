"""Read-only half-turn comparison of saved alignment poses."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from origami_analysis import (direct_digital_group_localization_evidence, digital_group_template_evidence,
                             alignment_corner_sites, _render_candidate_image, _boundary_template_correlation)


def compare_half_turn(results, audit, grid, fiducials):
    first = next(iter(results.values()))
    params, picks = first['params'], first['picks']
    ids = tuple(audit['metadata']['group_order'])
    model = params.get('digital_pixel_model') or params['logical_model']
    model_ids = tuple(map(str, model['bit_ids']))
    cells = model.get('bit_physical_cells', model.get('bit_cells', ()))
    order = [model_ids.index(bit) for bit in ids]
    groups = [cells[j] for j in order]
    regions = picks.aligned_regions
    n, g = len(regions), len(ids)
    if not n:
        raise ValueError("No saved candidates are available for orientation comparison.")
    grid, fiducials = np.asarray(grid,float), np.asarray(fiducials,float).reshape(-1,2)
    if not len(fiducials):
        raise ValueError('The saved shared alignment template has no fiducial geometry. Reload it before comparing orientations.')
    reference = np.asarray(picks.alignment_reference_image,float)
    side = float(picks.alignment_canvas_side_nm)
    pixel = float(picks.alignment_pixel_nm)
    if reference.ndim != 2 or not reference.size or not np.isfinite(reference).all() or min(side,pixel) <= 0:
        raise ValueError('Saved alignment raster is missing. Rerun shared alignment before this test.')
    radius = float(params.get('site_mask_radius_nm',5))
    support = float(params.get('min_site_localizations',0))
    prominence = float(params.get('min_site_evidence',0))
    corners = alignment_corner_sites(fiducials)
    names = list(results)
    expected, eligibility = [], []
    for name in names:
        p = results[name]['params']
        m = p['logical_model']; mids = tuple(map(str,m['bit_ids']))
        expected.append(np.asarray(m['active_bits'],bool)[[mids.index(bit) for bit in ids]])
        mask = np.asarray(p.get('classification_qc_eligible',p.get('_alignment_accepted_mask',())),bool)
        if mask.shape != (n,):
            mask = np.zeros(n,bool)  # Unknown is reported separately, never treated as passing.
            eligibility.append(None)
        else:
            disposition = p.get('classification_dispositions',())
            if len(disposition)==n:
                mask = mask & np.array([not ('suppressed duplicate' in str(d) or 'tile halo' in str(d)) for d in disposition])
            eligibility.append(mask)
    expected = np.asarray(expected)
    original_correlations = np.asarray(getattr(picks, 'rectangle_confidence', np.full(n,np.nan)),float)
    candidate_rows, group_rows, fiducial_rows = [], [], []
    valid = audit['candidates'].status.to_numpy() != 'invalid'
    for index, original in enumerate(regions):
        comparisons = []
        for degrees in (0,180):
            points = np.asarray(original,float) * (1 if degrees==0 else -1)
            evidence = direct_digital_group_localization_evidence([points],grid,groups,assignment_radius_nm=radius)
            raw = digital_group_template_evidence(evidence,[False]*g,groups)[2]
            prob = digital_group_template_evidence(evidence,[False]*g,groups,
                minimum_support_per_position=support,minimum_group_prominence=prominence)[2]
            prom = np.maximum(0,2*raw-1)
            states = (prob[0]>=.5)&(evidence[0]>=support)&(prom[0]>=prominence)
            pattern = ''.join('1' if b else '0' for b in states)
            if degrees==0 and valid[index] and pattern != audit['candidates'].iloc[index].pattern:
                raise ValueError(f'Saved calls do not reproduce for candidate {index+1}. Rerun Steps 3 and 4 before interpreting a rotation comparison.')
            image = _render_candidate_image(points,np.zeros(2),side,pixel,max(pixel,1.0))
            correlation = _boundary_template_correlation(image,reference)
            tree = cKDTree(points)
            counts = np.asarray(tree.query_ball_point(fiducials,radius,return_length=True),int)
            corner_counts = np.asarray(tree.query_ball_point(corners,radius,return_length=True),int)
            matched = np.flatnonzero(np.all(expected==states,axis=1)) if valid[index] else []
            matches = ';'.join(names[t] for t in matched)
            eligible_matches = [names[t] for t in matched if eligibility[t] is not None and eligibility[t][index]]
            comparisons.append(dict(pattern=pattern, correlation=correlation,
                supported_fiducials=int(np.sum(counts>=support)), corner_support_pass=bool(np.all(corner_counts>=support)),
                matches=matches, eligible_matches=';'.join(eligible_matches),
                qc_unknown=any(eligibility[t] is None for t in matched), states=states))
            for j,bit in enumerate(ids):
                group_rows.append(dict(candidate_id=index+1,orientation_deg=degrees,group_id=bit,evidence=evidence[0,j],
                    on_score=prob[0,j],relative_prominence=prom[0,j],on=bool(states[j]),
                    support_threshold=support,prominence_threshold=prominence,valid_saved_candidate=bool(valid[index])))
            for j,count in enumerate(counts):
                fiducial_rows.append(dict(candidate_id=index+1,orientation_deg=degrees,fiducial_id=j+1,
                    x_nm=fiducials[j,0],y_nm=fiducials[j,1],localizations=int(count),supported=bool(count>=support)))
        zero, half = comparisons
        assigned = audit['candidates'].iloc[index].assigned_template
        delta_corr = half['correlation']-zero['correlation']
        delta_fid = half['supported_fiducials']-zero['supported_fiducials']
        candidate_rows.append(dict(candidate_id=index+1,original_assignment=assigned or 'Unclassified',
            valid_saved_candidate=bool(valid[index]), original_search_correlation=float(original_correlations[index]) if len(original_correlations)==n else np.nan, pattern_0=zero['pattern'],pattern_180=half['pattern'],
            exact_matches_0=zero['matches'],exact_matches_180=half['matches'],
            eligible_matches_180=half['eligible_matches'],qc_eligibility_unknown=half['qc_unknown'],
            paired_correlation_0=zero['correlation'],paired_correlation_180=half['correlation'],correlation_change=delta_corr,
            supported_fiducials_0=zero['supported_fiducials'],supported_fiducials_180=half['supported_fiducials'],fiducial_support_change=delta_fid,
            corner_support_pass_0=zero['corner_support_pass'],corner_support_pass_180=half['corner_support_pass'],
            gained_on_groups=';'.join(np.asarray(ids)[~zero['states']&half['states']]),
            lost_on_groups=';'.join(np.asarray(ids)[zero['states']&~half['states']]),
            gained_on_count=int(np.sum(~zero['states']&half['states'])),
            alternative_unique_match=bool(len(half['matches'].split(';'))==1 and half['matches']),
            both_alignment_metrics_improve=bool(delta_corr>1e-6 and delta_fid>0)))
    return dict(orientation_candidates=pd.DataFrame(candidate_rows),orientation_groups=pd.DataFrame(group_rows),
                orientation_fiducials=pd.DataFrame(fiducial_rows))


def plot_half_turn(figure,tables):
    figure.clear()
    figure.set_layout_engine('constrained',h_pad=.1,w_pad=.1)
    layout=figure.add_gridspec(3,2,height_ratios=[1,1,.24])
    axes=[figure.add_subplot(layout[i,j]) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    data=tables['orientation_candidates']
    valid=data[data.valid_saved_candidate]
    for label,mask,color in [('Assigned',valid.original_assignment!='Unclassified','#64748b'),
                             ('Unclassified',valid.original_assignment=='Unclassified','#06b6d4')]:
        rows=valid[mask]
        axes[0].scatter(rows.paired_correlation_0,rows.paired_correlation_180,s=12,alpha=.65,label=label,color=color)
    axes[0].plot([0,1],[0,1],'k--',linewidth=.7)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Saved orientation: paired correlation',ylabel='180°: paired correlation',title='Does the half-turn improve alignment?')
    axes[0].legend(fontsize=8)
    dots=axes[1].scatter(valid.fiducial_support_change,valid.correlation_change,s=15,c=valid.gained_on_count,cmap='viridis')
    bar=figure.colorbar(dots,ax=axes[1],shrink=.8)
    bar.set_label('Groups OFF → ON',fontsize=8)
    bar.ax.tick_params(labelsize=7)
    axes[1].axhline(0,color='gray',linewidth=.7); axes[1].axvline(0,color='gray',linewidth=.7)
    axes[1].set(xlabel='Change in supported fiducials',ylabel='Change in paired correlation',title='Upper right: both alignment metrics improve')
    unclassified=valid[valid.original_assignment=='Unclassified']
    match=unclassified.alternative_unique_match
    counts=[len(unclassified),int(match.sum()),int((match&unclassified.both_alignment_metrics_improve).sum()),
            int((match&(unclassified.eligible_matches_180!='')&~unclassified.qc_eligibility_unknown).sum())]
    bars=axes[2].barh(range(4),counts,color='#22b8cf')
    axes[2].bar_label(bars,padding=3,fontsize=8)
    axes[2].set(yticks=range(4),yticklabels=['All unclassified','Unique match at 180°','Match + both metrics improve','Match + saved QC eligibility'],xlabel='Candidates (overlapping categories)',title='Alternative matches are not reassignments')
    axes[2].margins(x=.15)
    ranked=unclassified[match].sort_values(['both_alignment_metrics_improve','correlation_change'],ascending=False).head(6)
    axes[3].set_axis_off(); axes[3].set_title('Candidates to inspect',fontsize=10)
    if len(ranked):
        table=axes[3].table(cellText=[[r.candidate_id,r.exact_matches_180,f'{r.correlation_change:+.3f}',f'{r.fiducial_support_change:+d}'] for r in ranked.itertuples()],
            colLabels=['Saved ID','180° match','Δ correlation','Δ fiducials'],cellLoc='center',bbox=[0,.05,1,.85])
        table.auto_set_font_size(False);table.set_fontsize(8)
    else:
        axes[3].text(.5,.5,'No unclassified candidate has a unique 180° match.',ha='center',va='center',fontsize=8)
    note=figure.add_subplot(layout[2,:]);note.set_axis_off()
    note.text(0,.95,'Exact half-turn about the saved origin; no translation, refitting or assignment changes.',va='top',fontsize=8)
    note.text(0,.55,'Paired correlation uses cropped saved localizations; it is not the original search score. Saved QC eligibility stays fixed.',va='top',fontsize=8)
    note.text(0,.15,'Extra orientation trials can create chance matches. Inspect fiducials and raw signal; all candidate/group/site comparisons are exported.',va='top',fontsize=8)
    for axis in axes[:3]:
        axis.tick_params(labelsize=8);axis.title.set_fontsize(10);axis.xaxis.label.set_size(8);axis.yaxis.label.set_size(8)
    figure.suptitle(f'180° orientation check · {len(valid):,} valid candidates · {len(data)-len(valid):,} invalid excluded',fontsize=12)
