import copy
from types import SimpleNamespace
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from origami_analysis import _render_candidate_image, direct_digital_group_localization_evidence, digital_group_template_evidence
from origami_qc import build_classification_audits
from origami_orientation import compare_half_turn, plot_half_turn


def fixture():
    # Asymmetric physical sites: a half-turn cannot merely permute group IDs.
    grid=np.array([[-20.,-10.],[-20.,10.],[10.,10.]])
    fiducials=grid[:1]
    original=np.repeat(grid,12,axis=0)
    regions=[original,-original]
    evidence=direct_digital_group_localization_evidence(regions,grid,[[0],[1],[2]],assignment_radius_nm=2)
    prob=digital_group_template_evidence(evidence,[False]*3,[[0],[1],[2]],minimum_support_per_position=3,minimum_group_prominence=.1)[2]
    raw=digital_group_template_evidence(evidence,[False]*3,[[0],[1],[2]])[2]
    model=dict(bit_ids=('align','a','b'),bit_physical_cells=((0,),(1,),(2,)))
    params=dict(digital_pixel_ids=model['bit_ids'],digital_pixel_model=model,
        digital_pixel_probabilities=prob,digital_group_localization_evidence=evidence,digital_group_prominences=np.maximum(0,2*raw-1),
        min_site_localizations=3,min_site_evidence=.1,site_mask_radius_nm=2,classification_qc_eligible=(True,True))
    results={}
    for name,bits,mask in [('align_fid',(1,0,0),(False,False)),('full',(1,1,1),(True,False))]:
        picks=SimpleNamespace(regions=regions,aligned_regions=regions,accepted_mask=np.array(mask),
            alignment_reference_image=_render_candidate_image(np.repeat(fiducials,12,axis=0),np.zeros(2),80,1,1),
            alignment_canvas_side_nm=80,alignment_pixel_nm=1)
        results[name]=dict(picks=picks,params=dict(params,logical_model=dict(model,active_bits=bits)))
    return results,grid,fiducials


def test_half_turn_recovers_full_pattern_and_fiducials_without_reassigning():
    results,grid,fiducials=fixture()
    before=copy.deepcopy(results)
    audit=build_classification_audits(results)
    tables=compare_half_turn(results,audit,grid,fiducials)
    rows=tables['orientation_candidates']
    assert rows.iloc[1].exact_matches_0==''
    assert rows.iloc[1].exact_matches_180=='full'
    assert rows.iloc[1].eligible_matches_180=='full'
    assert rows.iloc[1].both_alignment_metrics_improve
    assert rows.iloc[1].gained_on_count==3
    assert rows.iloc[0].correlation_change<0
    for name in results:
        np.testing.assert_array_equal(results[name]['picks'].accepted_mask,before[name]['picks'].accepted_mask)
        np.testing.assert_array_equal(results[name]['picks'].aligned_regions,before[name]['picks'].aligned_regions)
    fig=Figure(figsize=(12,8));FigureCanvasAgg(fig)
    plot_half_turn(fig,tables);fig.canvas.draw()


def test_fixed_qc_and_missing_qc_are_not_automatic_recoveries():
    results,grid,fiducials=fixture()
    results['full']['params']['classification_qc_eligible']=(True,False)
    audit=build_classification_audits(results)
    row=compare_half_turn(results,audit,grid,fiducials)['orientation_candidates'].iloc[1]
    assert row.exact_matches_180=='full' and row.eligible_matches_180==''
    results['full']['params'].pop('classification_qc_eligible')
    row=compare_half_turn(results,audit,grid,fiducials)['orientation_candidates'].iloc[1]
    assert row.qc_eligibility_unknown and row.eligible_matches_180==''


def test_stale_measurements_and_missing_alignment_refuse_comparison():
    results,grid,fiducials=fixture()
    audit=build_classification_audits(results)
    audit['candidates'].loc[0,'pattern']='000'
    with pytest.raises(ValueError,match='do not reproduce'):
        compare_half_turn(results,audit,grid,fiducials)
    with pytest.raises(ValueError,match='fiducial geometry'):
        compare_half_turn(results,audit,grid,[])


def test_symmetric_full_pattern_does_not_gain_groups():
    results,grid,fiducials=fixture()
    # Reuse the same asymmetric group map with a symmetric localization population.
    symmetric=np.vstack((np.repeat(grid,12,axis=0),-np.repeat(grid,12,axis=0)))
    regions=[symmetric,symmetric]
    evidence=direct_digital_group_localization_evidence(regions,grid,[[0],[1],[2]],assignment_radius_nm=2)
    prob=digital_group_template_evidence(evidence,[False]*3,[[0],[1],[2]],minimum_support_per_position=3,minimum_group_prominence=.1)[2]
    raw=digital_group_template_evidence(evidence,[False]*3,[[0],[1],[2]])[2]
    for name,payload in results.items():
        payload['picks'].regions=regions
        payload['picks'].aligned_regions=regions
        payload['picks'].accepted_mask=np.array([name=='full']*2)
        payload['params'].update(digital_pixel_probabilities=prob,digital_group_localization_evidence=evidence,
                                 digital_group_prominences=np.maximum(0,2*raw-1))
    audit=build_classification_audits(results)
    rows=compare_half_turn(results,audit,grid,fiducials)['orientation_candidates']
    assert (rows.gained_on_count==0).all()
    assert (rows.exact_matches_0==rows.exact_matches_180).all()
    np.testing.assert_allclose(rows.correlation_change,0)
