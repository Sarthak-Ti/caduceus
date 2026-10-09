"""Build the promoter-level input caches read by comparing_model8 and comparing_model9.

    .../caduceus/.pixi/envs/default/bin/python prepare_model8.py cage
    .../EpiBERT/.pixi/envs/default/bin/python  prepare_model8.py rampage   # needs TensorFlow + pysam
    .../caduceus/.pixi/envs/default/bin/python prepare_model8.py enformer_extra
    .../caduceus/.pixi/envs/default/bin/python prepare_model8.py borzoi

`cage` must run first (it writes promoters.tsv, which the others read). No GPU or
model inference; source data are read-only. Outputs go to comparing_model8_results/
(and its extensions/ subfolder for enformer_extra and borzoi).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'comparing_model8_results'
EXT_OUT = OUT / 'extensions'
DATA = Path('/data1/lesliec/sarthak/data')
EPI = Path('/data1/lesliec/sarthak/EpiBERT')
RP = EPI / 'data/rampage_ladder_1840'
MO = DATA / 'joint_playground/model_out'
BOR = DATA / 'borzoi'
ANNOTATION = DATA / 'DE_danwei/gencode.v49.annotation.gtf.exon_transcript__protein_coding.parquet'
PADS = (256, 512, 1024)


def prepare_promoters():
    regions = pd.read_csv(RP / 'regions.tsv', sep='\t')
    bed = pd.read_csv(DATA / 'DK_zarr/sequences_enformer.bed', sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'split'])
    test = bed[bed.split == 'test'].reset_index(drop=True)
    assert np.array_equal(test.iloc[regions.enformer_row][['chrom','start','end']].values,
                          regions[['chrom','start','end']].values)
    # Independently enforce no EpiBERT input/train overlap and the old CAGE blacklist.
    train = pd.read_csv(EPI / 'data_processing/write_TF_records/sequence_splits/hg38_train_524k.bed',
                        sep='\t', header=None)
    for r in regions.itertuples():
        tr = train[train[0] == r.chrom]
        assert not ((tr[1] < r.center+262144) & (tr[2] > r.center-262144)).any()
        tr = bed[(bed.split == 'train') & (bed.chrom == r.chrom)]
        centers = (tr.start.to_numpy()+tr.end.to_numpy())//2
        overlap = np.maximum(0, np.minimum(centers+262144,r.end)-np.maximum(centers-262144,r.start))
        assert not (overlap > .2*(r.end-r.start)).any()
    # Drop approximate edge grids and identical genomic windows before selecting genes.
    regions = regions[~regions.edge_shifted].drop_duplicates(['chrom','output_start']).copy()
    tx = pd.read_parquet(ANNOTATION)
    tx = tx[tx.feature == 'transcript'].copy()
    tx['length'] = tx.end0-tx.start0
    tx = tx.sort_values(['gene_id','length','transcript_id'], ascending=[True,False,True])
    tx = tx.drop_duplicates('gene_id')  # annotation-only longest genomic-span transcript
    tx['tss'] = np.where(tx.strand == '+', tx.start0, tx.end0-1)
    rows = []
    for t in tx.itertuples():
        candidates = regions[(regions.chrom == t.chrom) &
                             (regions.output_start <= t.tss-max(PADS)) &
                             (regions.output_start+896*128 >= t.tss+max(PADS))]
        if candidates.empty:
            continue
        # Use one region per gene; maximize distance to the output boundary.
        r = candidates.iloc[np.argmin(np.abs(candidates.center.to_numpy()-t.tss))]
        base = dict(gene_id=t.gene_id, gene_name=t.gene_name, transcript_id=t.transcript_id,
                    chrom=t.chrom, strand=t.strand, tss=int(t.tss), enformer_row=int(r.enformer_row),
                    output_start=int(r.output_start), input_start=int(r.center-262144),
                    input_end=int(r.center+262144))
        for pad in PADS:
            # Whole bins with centers in [TSS-pad, TSS+pad): exactly 2*pad/128 bins.
            lo = int(np.ceil((t.tss-pad-r.output_start-64)/128))
            hi = int(np.ceil((t.tss+pad-r.output_start-64)/128))
            assert 0 <= lo < hi <= 896 and hi-lo == 2*pad//128
            rows.append(dict(**base, pad=pad, bin_start=lo, bin_end=hi,
                             genomic_start=int(r.output_start+lo*128),
                             genomic_end=int(r.output_start+hi*128)))
    promoters = pd.DataFrame(rows)
    # Mark overlap with another selected gene's promoter; exclude in a sensitivity check.
    promoters['ambiguous_promoter'] = False
    for (_, _), group in promoters.groupby(['pad','chrom']):
        for i,r in group.iterrows():
            promoters.loc[i,'ambiguous_promoter'] = bool(((group.genomic_start < r.genomic_end) &
                                                        (group.genomic_end > r.genomic_start) &
                                                        (group.gene_id != r.gene_id)).any())
    # Bootstrap connected components of overlapping 524-kb input contexts.
    intervals = promoters[['chrom','enformer_row','input_start','input_end']].drop_duplicates()
    blocks, block, end, chrom = {}, -1, -1, None
    for r in intervals.sort_values(['chrom','input_start']).itertuples():
        if r.chrom != chrom or r.input_start >= end:
            block += 1
            end = r.input_end
        else:
            end = max(end,r.input_end)
        chrom = r.chrom
        blocks[r.enformer_row] = block
    promoters['block'] = promoters.enformer_row.map(blocks)
    text = promoters.to_csv(sep='\t', index=False)
    if not (OUT / 'promoters.tsv').exists() or (OUT / 'promoters.tsv').read_text() != text:
        (OUT / 'promoters.tsv').write_text(text)
    regions.to_csv(OUT / 'retained_regions.tsv', sep='\t', index=False)
    print(f'{len(regions)} exact, unique regions; {promoters.gene_id.nunique()} genes; {block+1} blocks', flush=True)
    return promoters


def aggregate(array, p):
    return np.asarray([np.sum(array[int(r.enformer_row),int(r.bin_start):int(r.bin_end)],axis=0,
                              dtype=np.float64) for r in p.itertuples()])


def cage():
    import numcodecs
    OUT.mkdir(exist_ok=True)
    p = prepare_promoters()
    # Read Zarr v2 chunks directly: installed zarr 3 hangs opening this legacy store.
    store = DATA / 'enformer/data/labels.zarr/test'
    metadata = json.loads((store / '.zarray').read_text())
    assert metadata['shape'] == [1937,896,5313]
    assert metadata['chunks'] == [32,896,5313] and metadata['order'] == 'C'
    assert metadata['filters'] is None and metadata['dtype'] == '<f4'
    codec = numcodecs.get_codec(metadata['compressor'])
    measured = np.zeros((len(p),638),np.float64)
    truth = np.load(MO / 'GM12878_base_targets_conv.npy')
    for first in range(0,1937,32):
        selected = p[(p.enformer_row >= first) & (p.enformer_row < first+32)]
        if selected.empty:
            continue
        buffer = codec.decode((store / f'{first//32}.0.0').read_bytes())
        chunk = np.frombuffer(buffer,dtype=metadata['dtype']).reshape(metadata['chunks'])
        a = chunk[:min(32,1937-first),:,4675:]
        assert np.array_equal(a[:,:,5110-4675],truth[first:first+32])
        for i,r in selected.iterrows():
            measured[i] = a[int(r.enformer_row)-first,int(r.bin_start):int(r.bin_end)].sum(axis=0,dtype=np.float64)
        print(f'CAGE chunk {first//32+1}/61',flush=True)
    predictions = {}
    for name,stem in [('cis-Mamba','GM12878_no_mlm'),('Enformer','enformer_test_track5110')]:
        f = MO / (f'{stem}_predictions.npy' if name == 'cis-Mamba' else f'{stem}.npy')
        a = np.load(f)
        assert a.shape == truth.shape and np.isfinite(a).all() and (a >= 0).all()
        if name == 'cis-Mamba':
            assert np.array_equal(np.load(MO / f'{stem}_targets.npy'),truth)
        predictions[name] = aggregate(a,p)
    # Sum of released Enformer CAGE values over promoter bins; log2(sum+p) in analysis.
    np.savez_compressed(OUT / 'cage_promoter_inputs.npz',measured=measured,**predictions)


def rampage():
    import tensorflow as tf
    import pysam
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scipy.ndimage import convolve1d
    from tensorflow.core.framework import tensor_pb2
    p = pd.read_csv(OUT / 'promoters.tsv',sep='\t')
    regions = pd.read_csv(RP / 'regions.tsv',sep='\t')  # all 1840 for sequence multiset audit
    datasets = pd.read_csv(RP / 'datasets.tsv',sep='\t')
    lookup, expected = defaultdict(list), Counter()
    fasta_path = DATA / 'multispecies/references/human/hg38.fa'
    with pysam.FastaFile(str(fasta_path)) as fasta:
        for r in regions.itertuples():
            seq = fasta.fetch(r.chrom,int(r.rna_start)-2,int(r.rna_start)-2+524292).upper().encode()
            digest = hashlib.sha256(seq).digest()
            lookup[digest].append(r.enformer_row)
            expected[digest] += 1
    selections = {int(row):g for row,g in p.groupby('enformer_row')}
    raw = np.zeros((len(p),50),np.float64)
    native = np.zeros_like(raw)
    kernel = np.exp(-np.arange(-1,2,dtype=np.float32)**2/2)
    kernel /= kernel.sum()
    gm = np.flatnonzero(datasets.dataset_name.str.contains('GM12878'))
    assert len(gm) == 1
    def read_one(i):
        row = datasets.iloc[i]
        seen = Counter()
        x, y = np.full(len(p),np.nan), np.full(len(p),np.nan)
        for rec in tf.compat.v1.io.tf_record_iterator(row.file,tf.io.TFRecordOptions(compression_type='ZLIB')):
            features = tf.train.Example.FromString(rec).features.feature
            digest = hashlib.sha256(features['sequence'].bytes_list.value[0]).digest()
            assert digest in lookup
            seen[digest] += 1
            rna = tf.make_ndarray(tensor_pb2.TensorProto.FromString(features['rna'].bytes_list.value[0])).reshape(-1)
            assert rna.shape == (4096,) and np.isfinite(rna).all() and (rna >= 0).all()
            # Match training_utils_rampage_finetune.py: clip, sqrt, Gaussian(3,1), then crop.
            transformed = convolve1d(np.sqrt(np.clip(rna,0,65500)),kernel,mode='constant',cval=0)
            for enformer_row in lookup[digest]:
                if enformer_row not in selections:
                    continue
                for j,r in selections[enformer_row].iterrows():
                    lo,hi = int(r.bin_start)+1600,int(r.bin_end)+1600
                    a,b = rna[lo:hi].sum(dtype=np.float64),transformed[lo:hi].sum(dtype=np.float64)
                    if np.isfinite(x[j]):
                        assert np.isclose(x[j],a) and np.isclose(y[j],b)
                    x[j],y[j] = a,b
        assert seen == expected and np.isfinite(x).all() and np.isfinite(y).all()
        print(f'Validated RAMPAGE {i+1}/50: {row.dataset_name}',flush=True)
        return i,x,y
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read_one,i) for i in range(50)]
        for future in as_completed(futures):
            i,x,y = future.result()
            raw[:,i],native[:,i] = x,y
    # Independent match to previously validated raw GM12878 cache.
    old = np.load(RP / 'aligned_rampage.npz')
    full = np.full((1937,896),np.nan,np.float32)
    full[old['enformer_rows']] = old['gm12878']
    assert np.allclose(raw[:,gm[0]],aggregate(full,p),atol=1e-5,rtol=1e-6)
    a = np.load(MO / 'epibert_gm12878_rampage_test_outputs.npy')
    assert a.shape == (1937,896) and np.isfinite(a).all() and (a >= 0).all()
    pred = aggregate(a,p)
    # measured_native: sum of sqrt(clip(raw,0,65500)) smoothed with Gaussian(3,1) over promoter bins
    np.savez_compressed(OUT / 'rampage_promoter_inputs.npz',measured_raw=raw,measured_native=native,
                        epibert=pred,gm_index=gm[0],kernel=kernel)


def enformer_extra():
    """Additional Enformer-grid arms: cis-Mamba variants and retrained Enformers."""
    EXT_OUT.mkdir(exist_ok=True,parents=True)
    p=pd.read_csv(OUT/'promoters.tsv',sep='\t')
    ref=np.load(MO/'GM12878_base_targets_conv.npy')
    values={}
    for label,stem in [('cis-Mamba (no pretraining)','nopretrain'),('cis-Mamba (8 CTs)','immune_CAGE')]:
        pf,tf=MO/f'{stem}_predictions.npy',MO/f'{stem}_targets.npy'
        pred,target=np.load(pf,mmap_mode='r'),np.load(tf,mmap_mode='r')
        if pred.ndim==3:
            assert pred.shape[-1]==1 and target.shape[-1]==1
            pred,target=pred[:,:,0],target[:,:,0]
        assert pred.shape==target.shape==ref.shape
        assert np.allclose(target,ref,rtol=1e-5,atol=1e-6)
        assert np.isfinite(pred).all() and (pred>=0).all()
        values[label]=aggregate(pred,p)
    for label,stem in [('Enformer retrained (1 CT, 768ch)','enformer_retrained_gm12878'),
                       ('Enformer retrained (1 CT, 1536ch)','enformer_retrained_n1_1536'),
                       ('Enformer retrained (16 CTs)','enformer_retrained_n16'),
                       ('Enformer retrained (42 CTs)','enformer_retrained_n42')]:
        folder=MO/stem
        pf,tf=folder/f'{stem}_cage_predictions.npy',folder/f'{stem}_cage_targets.npy'
        pred,target=np.load(pf,mmap_mode='r'),np.load(tf,mmap_mode='r')
        if pred.ndim==3:
            tracks=pd.read_csv(folder/f'{stem}_cage_tracks.csv')
            track=tracks[(tracks.cell_type=='GM12878')&(tracks.assay=='cage')]
            assert len(track)==1 and int(track.target_index.iloc[0])==5110
            col=int(track.column.iloc[0]);pred,target=pred[:,:,col],target[:,:,col]
        assert pred.shape==target.shape==ref.shape
        assert np.allclose(target,ref,rtol=1e-5,atol=1e-6)
        assert np.isfinite(pred).all() and (pred>=0).all()
        values[label]=aggregate(pred,p)
    np.savez_compressed(EXT_OUT/'enformer_extra_inputs.npz',**values)
    print('Prepared six additional Enformer-grid arms',flush=True)


def borzoi_promoters():
    bedfile=DATA/'DK_zarr/sequences_borzoi_fold3-4.bed'
    bed=pd.read_csv(bedfile,sep='\t',header=None,names=['chrom','start','end','split','fold','idx'])
    regions=bed[bed.fold=='fold3'].reset_index(drop=True)
    assert len(regions)==6888 and ((regions.end-regions.start)==196608).all()
    regions['enformer_row']=np.arange(len(regions)) # shared helper's row key; Borzoi row, NOT an Enformer row
    regions['output_start']=regions.start
    regions['center']=(regions.start+regions.end)//2
    regions['input_start']=regions.center-262144
    regions['input_end']=regions.center+262144
    train=bed[~bed.fold.isin(['fold3','fold4'])]
    regions['allowed']=True
    regions['strict_no_input_label_overlap']=True
    for chrom,g in train.groupby('chrom'):
        ts,te=g.start.to_numpy(),g.end.to_numpy()
        for i in np.flatnonzero(regions.chrom.to_numpy()==chrom):
            r=regions.iloc[i]
            assert not (np.minimum(te,r.end)-np.maximum(ts,r.start)>0).any(), 'Training labels overlap test labels'
            overlap=np.maximum(0,np.minimum(te,r.input_end)-np.maximum(ts,r.input_start))
            regions.loc[i,'allowed']=not (overlap>.2*196608).any()
            regions.loc[i,'strict_no_input_label_overlap']=not (overlap>0).any()
    retained=regions[regions.allowed].drop_duplicates(['chrom','output_start'])
    tx=pd.read_parquet(ANNOTATION)
    tx=tx[tx.feature=='transcript'].copy()
    tx['length']=tx.end0-tx.start0
    tx=tx.sort_values(['gene_id','length','transcript_id'],ascending=[True,False,True]).drop_duplicates('gene_id')
    tx['tss']=np.where(tx.strand=='+',tx.start0,tx.end0-1)
    rows=[]
    for t in tx.itertuples():
        candidates=retained[(retained.chrom==t.chrom)&(retained.start<=t.tss-max(PADS))&
                            (retained.end>=t.tss+max(PADS))]
        if candidates.empty:continue
        r=candidates.iloc[np.argmin(np.abs(candidates.center.to_numpy()-t.tss))]
        base=dict(gene_id=t.gene_id,gene_name=t.gene_name,transcript_id=t.transcript_id,
                  chrom=t.chrom,strand=t.strand,tss=int(t.tss),enformer_row=int(r.enformer_row),
                  borzoi_row=int(r.enformer_row),output_start=int(r.start),input_start=int(r.input_start),
                  input_end=int(r.input_end),strict_no_input_label_overlap=bool(r.strict_no_input_label_overlap))
        for pad in PADS:
            lo=int(np.ceil((t.tss-pad-r.start-16)/32));hi=int(np.ceil((t.tss+pad-r.start-16)/32))
            assert 0<=lo<hi<=6144 and hi-lo==2*pad//32
            rows.append(dict(**base,pad=pad,bin_start=lo,bin_end=hi,
                             genomic_start=int(r.start+lo*32),genomic_end=int(r.start+hi*32)))
    p=pd.DataFrame(rows)
    p['ambiguous_promoter']=False
    for (_,_,_),g in p.groupby(['pad','chrom','strand']):
        for i,r in g.iterrows():
            p.loc[i,'ambiguous_promoter']=bool(((g.genomic_start<r.genomic_end)&(g.genomic_end>r.genomic_start)&
                                                (g.gene_id!=r.gene_id)).any())
    intervals=p[['chrom','enformer_row','input_start','input_end']].drop_duplicates()
    blocks,block,end,chrom={},-1,-1,None
    for r in intervals.sort_values(['chrom','input_start']).itertuples():
        if r.chrom!=chrom or r.input_start>=end:
            block+=1;end=r.input_end
        else:end=max(end,r.input_end)
        chrom=r.chrom;blocks[r.enformer_row]=block
    p['block']=p.enformer_row.map(blocks)
    p.to_csv(EXT_OUT/'borzoi_promoters.tsv',sep='\t',index=False)
    print(f'Borzoi: {len(retained)}/6888 allowed regions; {p.gene_id.nunique()} genes; {block+1} blocks',flush=True)
    return p


def strand_aggregate(a,p):
    assert a.shape==(6888,6144,2)
    values=[]
    for r in p.itertuples():
        arr=a[int(r.borzoi_row),int(r.bin_start):int(r.bin_end),0 if r.strand=='+' else 1]
        assert np.isfinite(arr).all() and (arr>=0).all()
        values.append(arr.sum(dtype=np.float64))
    return np.asarray(values)


def borzoi():
    """Independent Borzoi-grid benchmark. Seeks within uncompressed Borzoi chunks to read
    promoter slices, avoiding the full 108-GB panel and filesystem mmap stalls."""
    EXT_OUT.mkdir(exist_ok=True,parents=True)
    p=borzoi_promoters()
    tracks=pd.read_csv(BOR/'targets_human.txt',sep='\t').iloc[:1276]
    assert tracks.description.str.startswith('CAGE:').all()
    assert np.array_equal(tracks.strand_pair.to_numpy(),np.arange(1276)^1)
    assert tracks.iloc[870:872].description.str.contains('GM12878').all()
    assert tracks.identifier.iloc[::2].str.endswith('+').all() and tracks.identifier.iloc[1::2].str.endswith('-').all()
    store=BOR/'borzoi_fold3_CAGE.zarr/fold3'
    meta=json.loads((store/'.zarray').read_text())
    assert meta['shape']==[6888,6144,1276] and meta['chunks']==[1,6144,64]
    assert meta['dtype']=='<f2' and meta['compressor'] is None and meta['filters'] is None and meta['order']=='C'
    target=np.load(MO/'borzoi_gm12878_cage_ep4_targets.npy')
    assert target.shape==(6888,1,6144,2)
    target=target[:,0]
    measured=np.zeros((len(p),638),float)
    groups=list(p.groupby('borzoi_row'))
    def read_group(item):
        row,g=item
        records=list(g.itertuples())
        group_low,group_high=int(g.bin_start.min()),int(g.bin_end.max())
        for c in range(20):
            path=store/f'{row}.0.{c}'
            assert path.stat().st_size==6144*64*2
            low,high=group_low,group_high
            if c==870//64:low,high=0,6144
            with path.open('rb') as handle:
                handle.seek(low*64*2)
                chunk=np.frombuffer(handle.read((high-low)*64*2),dtype='<f2').reshape(high-low,64)
            if c==870//64:
                assert np.array_equal(chunk[:,[870%64,871%64]].astype(np.float32),target[row]), 'GM target mismatch'
            n=min(64,1276-c*64)//2
            for r in records:
                i=r.Index
                strand=0 if r.strand=='+' else 1
                a=chunk[int(r.bin_start)-low:int(r.bin_end)-low,strand:2*n:2]
                assert np.isfinite(a).all() and (a>=0).all()
                measured[i,c*32:c*32+n]=a.sum(axis=0,dtype=np.float64)
            del chunk
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k,_ in enumerate(pool.map(read_group,groups)):
            if k%50==0:print(f'Borzoi panel {k+1}/{len(groups)} regions',flush=True)
    assert np.array_equal(measured[:,435],strand_aggregate(target,p))
    specs=[('cis-Mamba (Borzoi grid)',MO/'borzoi_gm12878_cage_ep4_outputs.npy','h1w2'),
           ('Borzoi pretrained',BOR/'model_outputs_fold3_gm12878CAGE.npy','hw2'),
           ('Borzoi LoRA',BOR/'model_outputs/borzoi_finetuned_gm12878_lorav2.npy','hw3'),
           ('Borzoi full FT',BOR/'model_outputs/borzoi_finetuned_gm12878_trunk.npy','hw3'),
           ('Borzoi from scratch',BOR/'model_outputs/borzoi_finetuned_gm12878_scratch.npy','hw3')]
    predictions={}
    for label,path,layout in specs:
        a=np.load(path)
        want=(6888,1,6144,2) if layout=='h1w2' else (6888,6144,3 if layout=='hw3' else 2)
        assert a.shape==want
        a=a[:,0] if layout=='h1w2' else a[:,:,:2]
        predictions[label]=strand_aggregate(a,p)
    # Check pretrained extraction against original full prediction chunks on five rows.
    import numcodecs
    full=BOR/'model_outputs_fold3.zarr/fold3'
    fm=json.loads((full/'.zarray').read_text())
    assert fm['chunks']==[1,6144,7611] and fm['dtype']=='<f2'
    pred=np.load(BOR/'model_outputs_fold3_gm12878CAGE.npy')
    for row in np.unique(np.linspace(0,6887,5).astype(int)):
        file=full/f'{row}.0.0'
        a=np.frombuffer(numcodecs.get_codec(fm['compressor']).decode(file.read_bytes()),dtype='<f2').reshape(6144,7611)
        assert np.array_equal(a[:,[870,871]],pred[row])
    # Deposited native Borzoi CAGE targets (no inverse transform), GM12878 = slot 435 of 638, 32-bp bins.
    np.savez_compressed(EXT_OUT/'borzoi_inputs.npz',measured=measured,**predictions)
    print('Borzoi input preparation complete',flush=True)


if __name__ == '__main__':
    {'cage':cage,'rampage':rampage,'enformer_extra':enformer_extra,'borzoi':borzoi}[sys.argv[1]]()
