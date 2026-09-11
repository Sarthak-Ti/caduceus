"""Synthetic checks for interval annotation, scoring, dependencies and weighted AP."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
import ep_linking10_pipeline as p
import ep_linking10_analysis as a


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'CACHE',tmp_path);monkeypatch.setattr(a,'CACHE',tmp_path)


def test_overlap_weights_and_terminal_boundaries():
    b,w=p.interval_bins(100,300,3)
    assert b.tolist()==[0,1,2] and w.tolist()==[28,128,44]
    assert p.interval_bins(128,256,2)[0].tolist()==[1]
    for lo,hi in [(-1,5),(2,2),(0,385)]:
        with pytest.raises(ValueError):p.interval_bins(lo,hi,3)


def test_packed_lookup_little_endian_and_weighted_interval():
    calls=np.array([[1,0,1,0,0,0,0,1,1],[0,1,0,0,0,0,0,0,1]],np.uint8)
    packed=np.packbits(calls,axis=1,bitorder='little')
    np.testing.assert_array_equal(p.calls_at_bins(packed,np.array([0,1,7,8])),calls[:,[0,1,7,8]])
    bins,weights=p.interval_bins(100,300,9)
    np.testing.assert_allclose(p.calls_at_bins(packed,bins) @ weights/200,[.36,.64])


def test_classification_reference_and_missing_and_ties():
    enh=pd.DataFrame(dict(mapping_status=['covered']*4+['missing_chromosome']))
    f=np.array([[1,1,1,0,np.nan],[1,1,1,0,np.nan],[0,1,1,1,np.nan],[0,0,1,1,np.nan]],float)
    r,_=p.classify_enhancers(enh,f,[0,1],anchor_fraction=.3)
    assert r.breadth_median.eq(3).all()
    assert r.specificity.tolist()==['higher','higher','lower','inactive','missing']
    assert r.n_reference_enhancers.eq(3).all() and np.isnan(r.breadth.iloc[-1])
    assert p.classify_enhancers(enh,f,[0,1],presence_fraction=.5)[0].specificity.tolist()==r.specificity.tolist()


def test_no_active_reference_not_silent_zeros():
    enh=pd.DataFrame(dict(mapping_status=['covered','missing']))
    r,_=p.classify_enhancers(enh,np.array([[0,np.nan],[0,np.nan]]),[0,1])
    assert r.breadth_median.isna().all() and r.specificity.tolist()==['inactive','missing']


@pytest.mark.parametrize('channels',[2,4])
def test_scores_match_original_notebook_function(channels):
    # Execute ONLY the isolated scorer definition, never the notebook cells.
    n=json.loads(Path(__file__).with_name('EP_linking9.ipynb').read_text())
    funcs=[node for c in n['cells'] if c['cell_type']=='code' for node in ast.parse(''.join(c['source'])).body
           if isinstance(node,ast.FunctionDef) and node.name=='score_slab']
    assert len(funcs)==1
    rng=np.random.default_rng(3);w=rng.random((20,channels,37));strand=np.array(['+','-']*10);ctx=np.ones(20,bool);ctx[2]=False
    ns=dict(np=np,strand_arr=strand,CAP=18)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'EP9 scorer','exec'),ns)
    expected=ns['score_slab'](w,ctx,32,576)
    np.testing.assert_array_equal(p.score_window(w,strand,ctx),expected)


def test_true_zero_kept_and_out_of_context_placeholder():
    w=np.ones((3,4,3));w[1,2]*=2
    scores=p.score_window(w,np.array(['+','+','-']),np.array([1,1,0],bool))
    assert scores[0]==scores[2]==0 and scores[1]>0
    with pytest.raises(AssertionError):p.score_window(-w,['+']*3,np.ones(3,bool))


def test_strict_zarr_does_not_fill_missing_chunk(tmp_path):
    meta=dict(zarr_format=2,order='C',filters=None,chunks=[1,4,3],dtype='<f4',compressor=None)
    (tmp_path/'.zarray').write_text(json.dumps(meta))
    with pytest.raises(FileNotFoundError):p.read_zarr_chunk(tmp_path,[0,0,0])
    with pytest.raises(AssertionError):p.read_zarr_chunk(tmp_path,[0.,0,0])
    x=np.ones((1,4,3),np.float32);(tmp_path/'0.0.0').write_bytes(x.tobytes())
    np.testing.assert_array_equal(p.read_zarr_chunk(tmp_path,[0,0,0]),x)


def test_shared_gene_and_overlapping_enhancers_form_joint_components():
    q=pd.DataFrame(dict(gene_id=['g1','g1','g2','g3','g4'],enhancer_id=list('abcde'),chrom=['chr1']*5,
                        enhancer_start=[0,200,250,400,500],enhancer_end=[100,300,350,500,600]))
    c=a.dependency_components(q)
    assert c[0]==c[1]==c[2] and c[3]!=c[4] and c[2]!=c[3]
    # Touching half-open intervals are not overlapping.


def test_weighted_ap_ties_matches_sklearn_and_explicit_resampling():
    y=np.array([1,0,1,0,1,0]);s=np.array([.8,.8,.2,.1,.1,0]);w=np.array([2,0,3,1,2,1])
    assert a.ap_with_weights(y,s,w)==pytest.approx(average_precision_score(y,s,sample_weight=w))
    ids=np.repeat(np.arange(len(y)),w)
    assert a.ap_with_weights(y,s,w)==pytest.approx(average_precision_score(y[ids],s[ids]))
    assert np.isnan(a.ap_with_weights(y,s,(1-y)))


def test_cohorts_keep_completion_and_context_separate():
    n=4;ctx=np.ones((n,len(p.RUNS)),bool);valid=ctx.copy()
    enf=next(i for i,r in enumerate(p.RUNS) if r[0]=='Enformer');ctx[0,enf]=False;valid[1,0]=False
    masks=a.cohort_masks(dict(context=ctx,valid=valid))
    assert masks['common']['context'].tolist()==[False,True,True,True]
    assert masks['common']['support'].tolist()==[False,False,True,True]
    assert masks['reach262']['support'].tolist()==[True,False,True,True]
    assert enf not in masks['reach262']['columns']


def test_bootstrap_identical_arms_have_zero_paired_differences(tmp_path):
    rng=np.random.default_rng(8);n=60
    q=pd.DataFrame(dict(pair_id=[str(i) for i in range(n)],gene_id=[f'g{i//2}' for i in range(n)],
        enhancer_id=[f'e{i}' for i in range(n)],chrom=['chr1']*n,enhancer_start=np.arange(n)*200,
        enhancer_end=np.arange(n)*200+100,enhancer_length=[100]*n,label=np.tile([1,0,0],20),
        k562_active=[True]*n,specificity=np.tile(['higher','lower'],30),distance_to_TSS=np.arange(n),
        k562_signal_mean=rng.random(n),breadth=rng.integers(1,100,n)))
    s=np.tile(rng.random((n,1)),(1,len(p.RUNS)))
    data=dict(signed=s,context=np.ones_like(s,bool),valid=np.ones_like(s,bool),shuffled=np.ones_like(s,np.int8))
    (tmp_path/'complete.json').write_text('{}')
    q,table,counts,draws=a.evaluate(q,data,n_boot=20)
    # Each component spans BOTH strata; their weights must agree across groups.
    assert q.groupby('gene_id').component.nunique().eq(1).all()
    comp=a.paired_comparisons(table,draws)
    np.testing.assert_allclose(comp.delta_AP,0,atol=1e-12)
    np.testing.assert_allclose(comp[['ci_low','ci_high']],0,atol=1e-12)
    assert table[(table.stratum=='higher') & (table.cohort=='common')].n.eq(30).all()


def test_load_missing_cache_never_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError,match='build stage'):p.load_cache()
    assert not (tmp_path/'complete.json').exists()


def test_model_definitions_match_current_ep9():
    n=json.loads(Path(__file__).with_name('EP_linking9.ipynb').read_text())
    nodes=[node for c in n['cells'] if c['cell_type']=='code' for node in ast.parse(''.join(c['source'])).body
           if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in node.targets)]
    assert len(nodes)==1
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'EP9 MODELS','exec'),ns)
    assert ns['MODELS']==p.MODELS and len(p.RUNS)==12


def test_annotator_uses_unique_intervals_and_cached_calls_only(tmp_path,monkeypatch):
    dnase=tmp_path/'dnase';folder=dnase/'chr1';folder.mkdir(parents=True)
    tracks=pd.DataFrame(dict(index=np.arange(674),description=['other']*674))
    anchors=[121,122,123,625];tracks.loc[anchors,'description']='DNASE:K562'
    tracks.to_csv(dnase/'tracks.tsv',sep='\t',index=False)
    (dnase/'coverage_manifest.json').write_text(json.dumps(dict(bin_bp=128,chromosomes=['chr1'])))
    (dnase/'track_thresholds.tsv').write_text('cached calls already computed\n')
    calls=np.zeros((674,4),np.uint8);calls[np.ix_(anchors,[0,1])]=1;calls[:10,1]=1
    np.save(folder/'calls.packed.npy',np.packbits(calls,axis=1,bitorder='little'))
    np.save(folder/'coverage.npy',calls.astype(np.float32)*128)
    monkeypatch.setattr(p,'DNASE',dnase);monkeypatch.setattr(p,'input_provenance',lambda:{'synthetic':'fixture'})
    q=pd.DataFrame(dict(enhancer_id=['a','a','b','c','d'],chrom=['chr1']*4+['chrM'],
                        enhancer_start=[0,0,128,500,0],enhancer_end=[128,128,256,520,128],
                        enhancer_length=[128,128,128,20,128]))
    enh=p.annotate_enhancers(q)
    assert len(enh)==4 and enh.n_reference_enhancers.eq(2).all()
    assert enh.breadth_median.eq(9).all()
    assert enh.mapping_status.tolist()==['covered','covered','outside_cached_domain','missing_chromosome']
    assert enh.breadth.iloc[:2].tolist()==[4,14]
    assert enh.k562_signal_mean.iloc[:2].eq(1).all()
    fractions=np.load(tmp_path/'track_enhancer_fraction.npy');assert fractions[:,2:].shape==(674,2)
    assert np.isnan(fractions[:,2:]).all()


def test_notebook_is_valid_python_and_no_automatic_cache_build():
    import nbformat
    nb=nbformat.read(Path(__file__).with_name('EP_linking10.ipynb'),as_version=4);nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':
            ast.parse(c.source)
            assert 'build_cache(' not in c.source


def test_within_gene_ranking_recovers_opposite_group_behavior(tmp_path):
    """Macro AUROC must use only within-gene/dataset comparisons."""
    rng=np.random.default_rng(31)
    rows=[]
    # Four informative gene-dataset units per specificity group. Model 0 ranks
    # higher-specificity units perfectly and lower-specificity units backwards.
    for group in ['higher','lower']:
        for gene in range(4):
            for label in [0,0,1,1]:
                rows.append(dict(dataset='study',gene_id=f'{group}_{gene}',specificity=group,
                    label=label,k562_active=True,pair_id=str(len(rows)),enhancer_id=f'e{len(rows)}',
                    chrom='chr1',enhancer_start=len(rows)*200,enhancer_end=len(rows)*200+100))
    q=pd.DataFrame(rows);q['component']=np.arange(len(q))//4
    n=len(q);scores=rng.normal(size=(n,len(p.RUNS)))
    for i,r in q.iterrows():
        scores[i,0]=r.label if r.specificity=='higher' else 1-r.label
        scores[i,1]=scores[i,0]  # identical comparator tests paired covariance
    data=dict(signed=scores,context=np.ones_like(scores,bool),valid=np.ones_like(scores,bool),
              shuffled=np.ones_like(scores,np.int8))
    (tmp_path/'complete.json').write_text('{}')
    summary,diffs,modeldiff,strict=a.within_gene_ranking(q,data,n_boot=100,seed=1)
    first=summary[(summary.cohort=='common')&(summary.model==p.RUNS[0][0])&
                  (summary.arm==p.RUNS[0][1])].set_index('specificity')
    assert first.loc['higher','macro_AUROC']==1 and first.loc['lower','macro_AUROC']==0
    d=diffs[(diffs.cohort=='common')&(diffs.model==p.RUNS[0][0])&
            (diffs.arm==p.RUNS[0][1])].iloc[0]
    assert d.higher_minus_lower==1 and d.ci_low==d.ci_high==1
    assert strict.n_shared_gene_datasets.eq(0).all()


def test_macro_draws_weight_units_equally_and_share_components():
    # Component zero has two units, each remains a macro unit rather than being
    # pooled into one candidate-heavy score.
    values=np.array([0.,1.,1.]);ids=np.array([0,0,1]);weights=np.array([[1,1],[2,0],[0,2]])
    got=a._macro_draws(values,ids,weights)
    np.testing.assert_allclose(got,[2/3,1/2,1])
