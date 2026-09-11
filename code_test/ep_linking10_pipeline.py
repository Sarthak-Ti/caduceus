"""EP_linking10: derive enhancer annotations and scores from existing cached data.

Never reads the source DNase Zarr or changes dsQTL/eQTL caches. First build reads
existing prediction arrays (no model inference); subsequent analysis is cache-only.
"""
from pathlib import Path
import os, json, hashlib, time, fcntl
import numpy as np
import pandas as pd
import numcodecs

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('EP10_CACHE', ROOT / 'EP_linking10_cache'))
DNASE = ROOT / 'dsQTL_benchmark8_cache'
E2G = Path('/data1/lesliec/sarthak/data/joint_playground/e2g')
BENCHMARK = Path('/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv')
GENE_BOUNDS = Path('/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed')
TSS_INFO = Path('/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json')
BIN_BP = 128
PRESENCE_FRACTION = .5
ANCHOR_FRACTION = .3
HALF_BP = 576
PSEUDOCOUNT = 1e-2
MODELS = {
    'ours-borzoi': dict(bin=32, reach=262144, kind='joint', arms={
        'acc':'k562_stranded_borzoi_cage_tss_centered',
        'shuffle':'k562_stranded_borzoi_cage_tss_centered_dinuc',
        'shuffle+acc':'k562_stranded_borzoi_cage_tss_centered_dinuc_acc'}),
    'ours-enformer': dict(bin=128, reach=262144, kind='joint', arms={
        'acc':'k562_unstranded_enformer_tss_centered',
        'shuffle':'k562_unstranded_enformer_tss_centered_dinuc',
        'shuffle+acc':'k562_unstranded_enformer_tss_centered_dinuc_acc'}),
    'ours-striped': dict(bin=32, reach=1048576, kind='joint', arms={
        'acc':'k562_striped_tss_centered',
        'shuffle':'k562_striped_tss_centered_dinuc',
        'shuffle+acc':'k562_striped_tss_centered_dinuc_acc'}),
    'Borzoi': dict(bin=32, reach=262144, kind='seq', arms={'shuffle':'k562_borzoi_cage_tss_centered_dinuc'}),
    'Enformer': dict(bin=128, reach=98304, kind='seq', arms={'shuffle':'k562_enformer_e2g_tss_centered_dinuc'}),
    'AlphaGenome': dict(bin=32, reach=524288, kind='seq', arms={'shuffle':'k562_alphagenome_e2g_tss_centered_dinuc.zarr'}),
}
RUNS = [(m, arm, base) for m,c in MODELS.items() for arm,base in c['arms'].items()]
NAMES = [f'{m} / {arm}' for m,arm,_ in RUNS]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    p=Path(path); s=p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def atomic_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)


def progress(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def input_provenance():
    return {str(p):digest(p) for p in [BENCHMARK,GENE_BOUNDS,TSS_INFO]}


def benchmark_pairs():
    """Exactly the EP_linking9 mapping/filter order; preserve both original row IDs."""
    q=pd.read_csv(BENCHMARK, sep='\t'); assert q.CellType.eq('K562').all()
    q['original_row_id']=np.arange(len(q))
    bounds=pd.read_csv(GENE_BOUNDS,sep='\t'); tss=json.loads(TSS_INFO.read_text())
    lookup=bounds.set_index('name').Ensembl_ID.to_dict()
    lookup={name:g for name,g in lookup.items() if g in tss}
    q['gene_id']=q.measuredGeneSymbol.map(lookup)
    q=q.dropna(subset=['gene_id','Regulated']).reset_index(drop=True)
    assert q.Regulated.isin([True,False,0,1]).all()
    q['mapped_row_id']=np.arange(len(q)); q['label']=q.Regulated.astype(int)
    q['pair_id']='row:'+q.original_row_id.astype(str)
    q['enhancer_start']=pd.to_numeric(q.chromStart).astype(np.int64)
    q['enhancer_end']=pd.to_numeric(q.chromEnd).astype(np.int64)
    q['enhancer_id']=q.chrom+':'+q.enhancer_start.astype(str)+'-'+q.enhancer_end.astype(str)
    q['tss']=q.gene_id.map(lambda g:int(tss[g]['tss']))
    q['strand']=q.gene_id.map(lambda g:tss[g]['strand']); assert q.strand.isin(['+','-']).all()
    q['distance_to_TSS']=np.abs(q.tss-(q.enhancer_start+q.enhancer_end)//2)
    # `name` / pair_uid can contain old-build coordinates: never parse them for overlaps.
    q['enhancer_length']=q.enhancer_end-q.enhancer_start
    return q


def interval_bins(start, end, n_bins):
    """Global 128-bp bin IDs and exact overlap weights for a fully represented interval."""
    if start<0 or end<=start or end>n_bins*BIN_BP:
        raise ValueError('Invalid or not fully represented interval')
    bins=np.arange(start//BIN_BP,(end-1)//BIN_BP+1,dtype=np.int64)
    weights=np.minimum(end,(bins+1)*BIN_BP)-np.maximum(start,bins*BIN_BP)
    assert weights.sum()==end-start and (weights>0).all()
    return bins,weights


def calls_at_bins(packed, bins):
    """Little-endian bit lookup without unpacking the whole chromosome."""
    return ((packed[:,bins//8] >> (bins%8)[None,:]) & 1).astype(np.uint8)


def classify_enhancers(enhancers, fractions, anchor_indices, presence_fraction=PRESENCE_FRACTION,
                       anchor_fraction=ANCHOR_FRACTION):
    """Feature-level, label-independent median; called tracks are the breadth unit."""
    assert fractions.shape[1]==len(enhancers) and 0<presence_fraction<=1 and 0<anchor_fraction<=1
    covered=enhancers.mapping_status.eq('covered').to_numpy()
    assert np.isfinite(fractions[:,covered]).all()
    presence=fractions>=presence_fraction
    breadth=presence.sum(0).astype(float); count=presence[anchor_indices].sum(0)
    active=covered & (count>=int(np.ceil(anchor_fraction*len(anchor_indices))))
    median=float(np.median(breadth[active])) if active.any() else np.nan
    out=enhancers.copy(); out['breadth']=np.where(covered,breadth,np.nan)
    out['k562_track_count']=np.where(covered,count,np.nan); out['k562_active']=active
    out['specificity']=np.select([~covered,covered & ~active,active & (breadth<=median),active & (breadth>median)],
                                ['missing','inactive','higher','lower'],default='no_reference')
    out['breadth_median']=median
    out['n_reference_enhancers']=int(active.sum())
    return out,presence.astype(np.uint8)


def annotate_enhancers(q):
    """Only cached DNase calls and four anchor coverage tracks are accessed."""
    tracks=pd.read_csv(DNASE/'tracks.tsv',sep='\t')
    assert np.array_equal(tracks['index'],np.arange(len(tracks))) and len(tracks)==674
    anchor=np.flatnonzero(tracks.description.eq('DNASE:K562'))
    assert anchor.tolist()==[121,122,123,625]
    cfg=json.loads((DNASE/'coverage_manifest.json').read_text()); assert cfg['bin_bp']==BIN_BP
    enh=q[['enhancer_id','chrom','enhancer_start','enhancer_end','enhancer_length']].drop_duplicates('enhancer_id').reset_index(drop=True)
    enh['enhancer_index']=np.arange(len(enh)); enh['mapping_status']='missing_chromosome'
    fractions=np.full((len(tracks),len(enh)),np.nan,dtype=np.float32)
    center_calls=np.full_like(fractions,np.nan)
    signal=np.full((len(anchor),len(enh)),np.nan,dtype=np.float32)
    source_stamps=[]
    for chrom,group in enh.groupby('chrom',sort=False):
        folder=DNASE/chrom
        if chrom not in cfg['chromosomes']:
            continue
        packed_path=folder/'calls.packed.npy'; cov_path=folder/'coverage.npy'
        if not packed_path.exists() or not cov_path.exists():
            raise FileNotFoundError(f'Existing DNase cache incomplete for {chrom}; do not rescan automatically')
        packed=np.load(packed_path,mmap_mode='r'); coverage=np.load(cov_path,mmap_mode='r')
        assert coverage.shape[0]==len(tracks) and packed.shape==(len(tracks),(coverage.shape[1]+7)//8)
        source_stamps.extend([stamp(packed_path),stamp(cov_path)])
        for i,row in group.iterrows():
            start,end=int(row.enhancer_start),int(row.enhancer_end)
            if start<0 or end<=start:
                enh.loc[i,'mapping_status']='invalid_interval';continue
            if end>coverage.shape[1]*BIN_BP:
                enh.loc[i,'mapping_status']='outside_cached_domain';continue
            bins,weights=interval_bins(start,end,coverage.shape[1])
            calls=calls_at_bins(packed,bins)
            fractions[:,i]=(calls @ weights)/(end-start)
            center_calls[:,i]=calls_at_bins(packed,np.array([(start+end)//2//BIN_BP]))[:,0]
            # Cached coverage is a 128-bp SUM; divide by 128 for mean processed signal per bp.
            signal[:,i]=np.asarray(coverage[np.ix_(anchor,bins)],float) @ weights / ((end-start)*BIN_BP)
            enh.loc[i,'mapping_status']='covered'
        progress(chrom,'annotated',len(group),'unique enhancers')
    enh['k562_signal_mean']=signal.mean(0)
    enh,presence=classify_enhancers(enh,fractions,anchor)
    np.save(CACHE/'track_enhancer_fraction.npy',fractions)
    np.save(CACHE/'track_enhancer_presence.npy',presence)
    np.save(CACHE/'track_enhancer_center_calls.npy',center_calls)
    np.save(CACHE/'k562_enhancer_signal.npy',signal)
    tracks.to_csv(CACHE/'tracks.tsv',sep='\t',index=False)
    enh.to_csv(CACHE/'enhancers.tsv.gz',sep='\t',index=False)
    manifest=dict(version=1, bin_bp=BIN_BP, genome='hg38', interval='benchmark chromStart/chromEnd, zero-based half-open',
        presence_fraction=PRESENCE_FRACTION, anchor_fraction=ANCHOR_FRACTION, anchor_indices=anchor.tolist(),
        anchor_needed=int(np.ceil(ANCHOR_FRACTION*len(anchor))), n_tracks=len(tracks),
        median=float(enh.breadth_median.iloc[0]) if enh.k562_active.any() else None,
        reference='unique covered K562-active enhancers in mapped benchmark before context filtering',
        n_reference=int(enh.k562_active.sum()), median_ties='breadth <= median: higher specificity',
        no_call='presence 0 at missing features is a placeholder: mapping_status must be applied',
        continuous_signal='overlap-weighted mean of processed 128-bp sum / 128; not raw read density',
        dnase_files=source_stamps, tracks_sha256=digest(DNASE/'tracks.tsv'),
        coverage_manifest_sha256=digest(DNASE/'coverage_manifest.json'),
        thresholds_sha256=digest(DNASE/'track_thresholds.tsv'),
        source_limitations='Normalized then scaled/clipped/adjusted signal; nominal Poisson FDR is approximate; inherited sparse zero-fill semantics',
        inputs=input_provenance(), code_sha256=digest(Path(__file__)))
    atomic_json(CACHE/'enhancer_manifest.json',manifest)
    return enh


def read_zarr_chunk(path, indices):
    """Strict local Zarr-v2 decode: no missing chunks silently become zeros."""
    path=Path(path); meta=json.loads((path/'.zarray').read_text())
    assert meta['zarr_format']==2 and meta['order']=='C' and not meta.get('filters')
    assert meta.get('dimension_separator','.')=='.'
    assert all(isinstance(i,(int,np.integer)) for i in indices)
    raw=(path/'.'.join(map(str,indices))).read_bytes()
    if meta['compressor'] is not None:raw=numcodecs.get_codec(meta['compressor']).decode(raw)
    return np.frombuffer(raw,dtype=meta['dtype']).reshape(meta['chunks'])


def score_window(window, strand, in_context, pseudocount=PSEUDOCOUNT):
    """Exact EP_linking9 score on a pre-cropped float64 TSS window."""
    assert window.ndim==3 and window.shape[1] in [2,4]
    assert np.isfinite(window).all() and (window>=0).all()
    og=np.where(np.asarray(strand)=='+',0,1) if window.shape[1]==4 else np.zeros(len(window),int)
    alt=og+window.shape[1]//2; rows=np.arange(len(window))
    wt=window[rows,og].sum(1); pert=window[rows,alt].sum(1)
    s=np.log2((pert+pseudocount)/(wt+pseudocount)); s[~in_context]=0
    return s


def build_scores(q):
    n=len(q); signed=np.zeros((n,len(RUNS)),float); contexts=np.zeros_like(signed,bool)
    shuffled=np.full(signed.shape,-1,np.int8); sources=[]; summaries=[]
    for j,(model,arm,base) in enumerate(RUNS):
        bp=MODELS[model]['bin']; hb=int(round(HALF_BP/bp)); path=E2G/base
        if base.endswith('.zarr'):
            meta=json.loads((path/'preds/.zarray').read_text())
            assert meta['shape'][0]==n and meta['chunks']==[1,4,meta['shape'][2]]
            done=read_zarr_chunk(path/'done',[0]).ravel()[:n]; assert done.all(),'Unfinished AlphaGenome rows'
            ic=read_zarr_chunk(path/'in_context',[0]).ravel()[:n].astype(bool)
            sh=read_zarr_chunk(path/'shuffled',[0]).ravel()[:n].astype(bool)
            center=meta['shape'][2]//2
            w=np.empty((n,4,2*hb+1),float)
            for i in range(n):w[i]=read_zarr_chunk(path/'preds',[i,0,0])[0,:,center-hb:center+hb+1]
            sources.append(dict(base=base, metadata=meta, group_stamp=stamp(path),
                                array_metadata={k:digest(path/k/'.zarray') for k in ['preds','done','in_context','shuffled']}))
        else:
            a=np.load(str(path)+'.npy',mmap_mode='r'); assert a.shape[0]==n and a.shape[1] in [2,4]
            center=a.shape[2]//2; assert center>=hb
            w=np.asarray(a[:,:,center-hb:center+hb+1],dtype=float)
            ic=np.load(str(path)+'_in_context.npy').astype(bool)
            shpath=Path(str(path)+'_shuffled.npy')
            sh=np.load(shpath).astype(bool) if shpath.exists() else None
            assert sh is not None or arm=='acc',f'Missing shuffle completion: {base}'
            sources.extend([stamp(str(path)+'.npy'),stamp(str(path)+'_in_context.npy')])
            if sh is not None:sources.append(stamp(shpath))
        assert ic.shape==(n,)
        expected=(q.enhancer_start.to_numpy()>=q.tss.to_numpy()-MODELS[model]['reach']) & (q.enhancer_end.to_numpy()<=q.tss.to_numpy()+MODELS[model]['reach'])
        assert np.array_equal(ic,expected),f'{model}: context geometry / pair ordering differs'
        if sh is not None:
            assert sh.shape==(n,); shuffled[:,j]=sh
        signed[:,j]=score_window(w,q.strand.to_numpy(),ic); contexts[:,j]=ic
        summaries.append(dict(model=model,arm=arm,bin_bp=bp,window_bins=2*hb+1,actual_window_bp=(2*hb+1)*bp,
                              in_context=int(ic.sum()),unshuffled_in_context=int((ic & ~sh).sum()) if sh is not None else None))
        progress(model,arm,'scored',n,'pairs;',int(ic.sum()),'in context')
    for model in MODELS:
        cols=[j for j,r in enumerate(RUNS) if r[0]==model]
        assert np.all(contexts[:,cols]==contexts[:,cols[0],None])
    # Preserve raw context and completion separately. Unknown acc completion is not a failure.
    valid=np.isfinite(signed) & ((shuffled!=0) | np.array([r[1]=='acc' for r in RUNS])[None,:])
    np.savez_compressed(CACHE/'model_scores.npz',signed=signed,context=contexts,shuffled=shuffled,valid=valid)
    pd.DataFrame(summaries).to_csv(CACHE/'score_inventory.tsv',sep='\t',index=False)
    atomic_json(CACHE/'scores_manifest.json',dict(version=1,runs=RUNS,models=MODELS,half_bp=HALF_BP,pseudocount=PSEUDOCOUNT,
        inputs=input_provenance(),sources=sources,scorer='EP_linking9 score_slab; float64 sums; out-of-context placeholder zero',
        code_sha256=digest(Path(__file__))))


def build_cache():
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (CACHE/'complete.json').exists():
            load_cache(); progress('Completed EP10 cache reused; no prediction or DNase array reads'); return
        q=benchmark_pairs(); assert len(q)==10280 and q.label.sum()==463
        q.to_csv(CACHE/'benchmark_pairs.tsv.gz',sep='\t',index=False)
        # Each expensive stage is checkpointed independently.
        if (CACHE/'enhancer_manifest.json').exists():
            em=json.loads((CACHE/'enhancer_manifest.json').read_text())
            assert em['inputs']==input_provenance() and em['presence_fraction']==PRESENCE_FRACTION and em['anchor_fraction']==ANCHOR_FRACTION, 'Stale partial enhancer cache: use a new EP10_CACHE'
        if (CACHE/'scores_manifest.json').exists():
            sm=json.loads((CACHE/'scores_manifest.json').read_text())
            assert sm['inputs']==input_provenance() and sm['runs']==json.loads(json.dumps(RUNS)) and sm['half_bp']==HALF_BP and sm['pseudocount']==PSEUDOCOUNT, 'Stale partial scores: use a new EP10_CACHE'
        if not (CACHE/'enhancer_manifest.json').exists():annotate_enhancers(q)
        if not (CACHE/'scores_manifest.json').exists():build_scores(q)
        atomic_json(CACHE/'complete.json',dict(version=1, inputs=input_provenance(),
                     enhancer_manifest_sha256=digest(CACHE/'enhancer_manifest.json'),
                     scores_manifest_sha256=digest(CACHE/'scores_manifest.json'),
                     product_sha256={name:digest(CACHE/name) for name in ['benchmark_pairs.tsv.gz','enhancers.tsv.gz','model_scores.npz']}))
        load_cache()


def load_cache():
    """Validate ordering/settings and use only EP10 products plus small input hashes."""
    if not (CACHE/'complete.json').exists():raise FileNotFoundError('EP10 cache incomplete; run build stage first')
    complete=json.loads((CACHE/'complete.json').read_text()); assert complete['inputs']==input_provenance(),'Benchmark inputs changed: use a new EP10_CACHE'
    assert all(digest(CACHE/name)==sha for name,sha in complete['product_sha256'].items()), 'Cached product changed'
    em=json.loads((CACHE/'enhancer_manifest.json').read_text()); sm=json.loads((CACHE/'scores_manifest.json').read_text())
    assert digest(CACHE/'enhancer_manifest.json')==complete['enhancer_manifest_sha256']
    assert digest(CACHE/'scores_manifest.json')==complete['scores_manifest_sha256']
    assert em['inputs']==sm['inputs']==complete['inputs']
    assert em['presence_fraction']==PRESENCE_FRACTION and em['anchor_fraction']==ANCHOR_FRACTION
    assert sm['runs']==json.loads(json.dumps(RUNS)) and sm['models']==MODELS
    assert sm['half_bp']==HALF_BP and sm['pseudocount']==PSEUDOCOUNT
    q=pd.read_csv(CACHE/'benchmark_pairs.tsv.gz',sep='\t'); enh=pd.read_csv(CACHE/'enhancers.tsv.gz',sep='\t')
    assert np.array_equal(q.mapped_row_id,np.arange(len(q))) and q.pair_id.is_unique and enh.enhancer_id.is_unique
    data=np.load(CACHE/'model_scores.npz'); arrays={k:data[k] for k in ['signed','context','shuffled','valid']};data.close()
    assert all(x.shape==(len(q),len(RUNS)) for x in arrays.values())
    for name in ['fraction','presence','center_calls']:
        assert np.load(CACHE/f'track_enhancer_{name}.npy',mmap_mode='r').shape==(em['n_tracks'],len(enh))
    joined=q.merge(enh.drop(columns=['chrom','enhancer_start','enhancer_end','enhancer_length']),on='enhancer_id',how='left',sort=False,validate='many_to_one')
    assert len(joined)==len(q) and joined.mapping_status.notna().all()
    assert np.array_equal(joined.mapped_row_id,q.mapped_row_id)
    return joined,enh,arrays


if __name__=='__main__':build_cache()
