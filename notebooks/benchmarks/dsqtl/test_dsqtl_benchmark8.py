"""Numerical checks for the genome-wide cache builder (no real coverage reads)."""
import tempfile
from pathlib import Path
import numpy as np
from scipy.stats import false_discovery_control, poisson
import dsQTL_benchmark8_pipeline as p

def test_bh_histogram():
    rng=np.random.default_rng(12)
    for lam in [0.,.2,2.,10.,80.]:
        for alpha in [.01,.05,.2]:
            x=np.r_[rng.poisson(lam,1500),rng.integers(0,200,200),np.zeros(200,dtype=int)]
            hist=np.bincount(x,minlength=201)
            threshold=p.bh_count_threshold(hist,lam,alpha)
            expected=false_discovery_control(poisson.sf(x-1,lam))<=alpha
            np.testing.assert_array_equal(x>=threshold,expected)

def test_chunk_carry_and_calls():
    original=(p.CACHE,p.SOURCE,p.CHROMS)
    with tempfile.TemporaryDirectory() as d:
        p.CACHE=Path(d)/'cache';p.CACHE.mkdir()
        p.SOURCE=Path(d)/'source';(p.SOURCE/'chrToy').mkdir(parents=True)
        p.CHROMS=['chrToy']
        # Deliberately non-aligned chunks, one terminal partial bin, one cross-chunk peak.
        x=np.zeros((2,583),dtype=np.float16)
        x[0,128:256]=20;x[1,256:384]=15
        x[0,0]=1;x[1,-1]=7
        m=dict(shape=list(x.shape),chunks=[2,192],dtype='<f2',compressor=None)
        for j,start in enumerate(range(0,x.shape[1],192)):
            part=np.zeros((2,192),dtype=np.float16)
            part[:,:min(192,x.shape[1]-start)]=x[:,start:start+192]
            (p.SOURCE/'chrToy'/f'0.{j}').write_bytes(part.tobytes())
        sparse_meta = dict(m, fill_value=0)
        assert not p.read_chunk(p.SOURCE/'chrToy', sparse_meta, (0,99)).any()
        p.pool_chrom('chrToy',m)
        actual=np.load(p.CACHE/'chrToy'/'coverage.npy')
        expected=x[:,:512].reshape(2,4,128).sum(axis=2,dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)
        with np.load(p.CACHE/'chrToy'/'histogram.npz') as h:
            np.testing.assert_array_equal(h['sums'],expected.sum(axis=1))
            assert np.all(h['histogram'].sum(axis=1)==4)
        bits=np.packbits(actual>=1,axis=1,bitorder='little')
        np.testing.assert_array_equal(np.unpackbits(bits,axis=1,count=4,bitorder='little'),actual>=1)
        # Exercise full peak construction and interval assignment on the tiny genome.
        tracks=p.pd.DataFrame({'index':[0,1],'identifier':['a','b'],'description':['DNASE:GM12878','other']})
        p.call_peaks(tracks,np.array([0]))
        peaks=p.pd.read_csv(p.CACHE/'gm12878_bins.tsv.gz',sep='\t')
        assert peaks.start.tolist()==[128] and peaks.end.tolist()==[256]
        assert peaks.breadth.tolist()==[1]
        merged=np.load(p.CACHE/'chrToy'/'tracks_x_merged_presence.npy')
        np.testing.assert_array_equal(merged,[[1],[0]])
        import dsQTL_benchmark8_analysis as a
        old_cache=a.CACHE;a.CACHE=p.CACHE
        q=p.pd.DataFrame(dict(chrom_hg38=['chrToy']*4,pos0_hg38=[127,128,255,256],mapping_status=['mapped']*4))
        v=a.annotate_variants(q)
        assert v.bins_specificity.tolist()==['outside_GM12878_peaks','high_specificity','high_specificity','outside_GM12878_peaks']
        assert v.bins_specificity.equals(v.merged_specificity)
        a.CACHE=old_cache
        # Completed chromosome is reused even if its source becomes unavailable.
        p.SOURCE=Path(d)/'absent'
        assert p.pool_chrom('chrToy',m)['bins']==4
    p.CACHE,p.SOURCE,p.CHROMS=original

if __name__=='__main__':
    test_bh_histogram();test_chunk_carry_and_calls()
    print('PASS: histogram BH equals scipy BH; cross-chunk pooling, terminal bins, bit packing, peak matrices, interval joins, cache reuse.')

def test_analysis_outputs():
    import dsQTL_benchmark8_analysis as a
    old_cache=a.CACHE
    with tempfile.TemporaryDirectory() as d:
        a.CACHE=Path(d)
        n=40; rng=np.random.default_rng(7)
        v=p.pd.DataFrame(dict(variant_row=np.arange(n),label_binary=np.arange(n)%2,
                            chrom_hg38=['chrToy']*n,bin_index=np.arange(n),
                            bins_specificity=['high_specificity']*20+['low_specificity']*20,
                            merged_specificity=['high_specificity']*20+['low_specificity']*20,
                            merged_peak_id=[f'peak{i//2}' for i in range(n)]))
        scores=rng.random((n,len(a.ARMS)));scores[:,1]=scores[:,0]
        scorable=np.ones_like(scores,dtype=bool);scorable[0,-1]=False
        table=a.stratified_metrics(v,scores,scorable,n_boot=10)
        assert table.loc[table.stratum=='all_common','n'].eq(39).all()
        assert table.loc[table.stratum=='outside_GM12878_peaks','AP'].isna().all()
        paired=p.pd.read_csv(a.CACHE/'paired_model_differences.tsv',sep='\t')
        valid=paired[(paired.model==a.ARMS[1][0])&(paired.stratum!='outside_GM12878_peaks')]
        assert valid[['delta_AP','ci_low','ci_high']].eq(0).all().all()
        a.variant_comparisons(v,scores,scorable);a.plot_results(table)
        assert (a.CACHE/'figures'/'bins_model_comparison.png').exists()
    a.CACHE=old_cache

if __name__=='__main__':
    test_analysis_outputs()
    print('PASS: common support, empty groups, paired bootstrap, exports and figures.')
