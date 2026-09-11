"""Headless export of dsQTL_benchmark8 notebook cache cells; no model inference."""
from pathlib import Path
import json, hashlib, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import poisson
import numcodecs
import torch
import fcntl

ROOT = Path('/data1/lesliec/sarthak/caduceus/code_test')
CACHE = ROOT / 'dsQTL_benchmark8_cache'
SOURCE = Path('/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_reprocessed.zarr')
TARGETS = Path('/data1/lesliec/sarthak/data/DK_zarr/targets_DNase_borzoi.txt')
DSQ = Path('/data1/lesliec/sarthak/data/joint_playground/dsQTL')
BIMS = Path('/data1/deyk/extras/1000G_BIMS_hg38')
BIN = 128
FDR = 0.01
GM_FRACTION = 0.30
MERGED_FRACTION = 0.50  # >= half the constituent bins called in a track
CHROMS = [f'chr{i}' for i in range(1,23)] + ['chrX','chrY']
WORKERS = int(os.environ.get('DSQTL8_WORKERS', '2'))

def progress(*parts):
    message = ' '.join(map(str, parts))
    print(message, flush=True)
    with (CACHE / 'build.log').open('a') as log:
        log.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata(path):
    m = json.loads((Path(path)/'.zarray').read_text())
    assert m['zarr_format'] == 2 and m['order'] == 'C' and not m.get('filters')
    assert m.get('dimension_separator', '.') == '.'
    return m

def read_chunk(path, m, indices):
    """Read a local Zarr-v2 chunk once, without asynchronous Zarr I/O.

    Valid only for the checked C-order, no-filter, dot-separated stores here.
    Absent chunks use the declared Zarr fill value; the manifest inventories them.
    """
    p = Path(path) / '.'.join(map(str, indices))
    if not p.exists():
        # Omitted chunks are legal in sparse Zarr stores (often assembly gaps).
        assert 'fill_value' in m and m['fill_value'] == 0, 'Unexpected sparse fill value'
        return np.zeros(m['chunks'], dtype=m['dtype'])
    raw = p.read_bytes()
    if m['compressor'] is not None:
        raw = numcodecs.get_codec(m['compressor']).decode(raw)
    return np.frombuffer(raw, dtype=m['dtype']).reshape(m['chunks'])

def init_cache():
    CACHE.mkdir(exist_ok=True)
    tracks = pd.read_csv(TARGETS, sep='\t')
    assert np.array_equal(tracks['index'], np.arange(len(tracks)))
    gm = np.flatnonzero(tracks.description.eq('DNASE:GM12878'))
    assert len(gm) > 0
    metas = {ch: metadata(SOURCE/ch) for ch in CHROMS}
    for m in metas.values():
        assert m['shape'][0] == len(tracks) and m['chunks'][0] == len(tracks)
        assert m['dtype'] == '<f2'
    omitted = {ch: [j for j in range((m['shape'][1]+m['chunks'][1]-1)//m['chunks'][1])
                    if not (SOURCE/ch/f'0.{j}').exists()] for ch,m in metas.items()}
    config = dict(version=2, source=str(SOURCE), targets_sha256=digest(TARGETS),
                  bin_bp=BIN, chromosomes=CHROMS, metadata=metas,
                  pooling='sum float32; floor only for approximate Poisson',
                  partial_terminal_bins='excluded', invalid_values='fail',
                  missing_source_chunks='declared zero fill; see omitted_chunks', omitted_chunks=omitted, signal='processed normalized coverage; no inverse clipping')
    p = CACHE/'coverage_manifest.json'
    if p.exists():
        assert json.loads(p.read_text()) == config, 'Cache config mismatch: use a new CACHE directory.'
    else:
        atomic_json(p, config)
    tracks.to_csv(CACHE/'tracks.tsv',sep='\t',index=False)
    return tracks, gm, metas

def pool_chrom(ch, m):
    """One pass per source chunk; carry partial bins across 1-Mb chunk edges."""
    out = CACHE/ch
    out.mkdir(exist_ok=True)
    marker = out/'coverage.done.json'
    if marker.exists():
        return json.loads(marker.read_text())
    t = time.time()
    nt, length = m['shape']; chunk_bp = m['chunks'][1]
    nb = length//BIN
    temp = out/'coverage.tmp.npy'
    pooled = np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(nt,nb))
    # Source is hard-clipped to <=128 per base; retain explicit bounds checks.
    hist = np.zeros((nt, BIN*128+1),dtype=np.int64)
    sums = np.zeros(nt, dtype=np.float64)
    carry = np.empty((nt,0),dtype=np.float32)
    written = 0
    for j,start in enumerate(range(0,length,chunk_bp)):
        x = read_chunk(SOURCE/ch,m,(0,j))[:, :min(chunk_bp,length-start)]
        # Exact CPU float16 -> float32 conversion; torch avoids NumPy's slow half conversion.
        # Copy gives torch a writable buffer; neither the decoded input nor source is modified.
        x = torch.from_numpy(x.copy()).float().numpy()
        # min/max propagate NaN and reject +/-Inf as well as out-of-range values.
        assert x.min() >= 0 and x.max() <= 128, (ch,j,'invalid coverage')
        if carry.shape[1]:
            x = np.concatenate((carry,x),axis=1)
        n = x.shape[1]//BIN
        y = x[:,:n*BIN].reshape(nt,n,BIN).sum(axis=2,dtype=np.float32)
        carry = x[:,n*BIN:].copy()
        pooled[:,written:written+n] = y
        sums += y.sum(axis=1,dtype=np.float64)
        for tr in range(nt):
            hist[tr] += np.bincount(np.floor(y[tr]).astype(np.int64),minlength=hist.shape[1])
        written += n
        if j % 10 == 0:
            progress(f'{ch}: {start/1e6:.0f}/{length/1e6:.0f} Mb, {time.time()-t:.0f}s')
    assert written == nb and np.all(hist.sum(axis=1)==nb)
    pooled.flush(); del pooled
    temp.replace(out/'coverage.npy')
    np.savez(out/'histogram.npz', histogram=hist, sums=sums)
    result = dict(chrom=ch,bins=nb,excluded_terminal_bp=length % BIN,seconds=time.time()-t)
    atomic_json(marker,result)
    progress('Pooled', result)
    return result

def bh_count_threshold(hist, lam, alpha):
    """Exact BH rejection set for discrete Poisson p-values using tied counts.

    At count k, rank is the number of bins with count >= k. The smallest k
    passing p(k)<=alpha*rank/m is the BH count cutoff, including all ties.
    """
    n = hist.sum()
    k = np.flatnonzero(hist)
    tail = np.cumsum(hist[::-1],dtype=np.int64)[::-1]
    p = poisson.sf(k-1,lam)
    ok = p <= alpha*tail[k]/n
    return int(k[ok][0]) if ok.any() else int(len(hist))

def call_peaks(tracks, gm):
    settings = dict(fdr=FDR, gm_fraction=GM_FRACTION, merged_fraction=MERGED_FRACTION,
                    gm_indices=gm.tolist(), median_ties='breadth <= median: high specificity',
                    merged_presence='fraction of constituent bins >= merged_fraction')
    marker = CACHE/'peaks.done.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == settings, 'Peak settings changed: use a new cache or remove peak products first.'
        return
    nt = len(tracks)
    hist = np.zeros((nt,BIN*128+1),dtype=np.int64)
    sums = np.zeros(nt,dtype=np.float64)
    for ch in CHROMS:
        with np.load(CACHE/ch/'histogram.npz') as h:
            hist += h['histogram']; sums += h['sums']
    nb = int(hist[0].sum()); assert np.all(hist.sum(axis=1)==nb)
    lam = sums/nb  # empirical global mean per track, NOT 100M / genome by assumption
    thresholds = np.array([bh_count_threshold(hist[i],lam[i],FDR) for i in range(nt)])
    report = tracks.copy()
    report['lambda_bin'] = lam; report['count_threshold'] = thresholds
    report['n_tested_bins'] = nb
    report['n_peak_bins'] = [hist[i,thresholds[i]:].sum() for i in range(nt)]
    report['p_at_threshold'] = poisson.sf(thresholds-1,lam)
    report.to_csv(CACHE/'track_thresholds.tsv',sep='\t',index=False)
    gm_needed = int(np.ceil(GM_FRACTION*len(gm)))
    bin_tables, merged_tables = [], []
    for ch in CHROMS:
        folder = CACHE/ch
        y = np.load(folder/'coverage.npy',mmap_mode='r')
        n = y.shape[1]
        # Keep all-genome bit calls, and a union-of-peaks matrix in track x peak orientation.
        packed = np.lib.format.open_memmap(folder/'calls.packed.npy',mode='w+',dtype='uint8',shape=(nt,(n+7)//8))
        breadth = np.zeros(n,dtype=np.uint16)
        gm_count = np.zeros(n,dtype=np.uint8)
        for tr in range(nt):
            calls = y[tr] >= thresholds[tr]  # equivalent to floor(y) >= integer threshold
            packed[tr] = np.packbits(calls,bitorder='little')
            breadth += calls
            if tr in gm: gm_count += calls
        packed.flush()
        union = np.flatnonzero(breadth>0)
        np.save(folder/'peak_bin_indices.npy',union.astype(np.int32))
        union_mat = np.lib.format.open_memmap(folder/'tracks_x_peak_bins.npy',mode='w+',dtype='uint8',shape=(nt,len(union)))
        gm_bins = np.flatnonzero(gm_count >= gm_needed)
        bt = pd.DataFrame(dict(chrom=ch,start=gm_bins*BIN,end=(gm_bins+1)*BIN,
                               bin_index=gm_bins,breadth=breadth[gm_bins],gm_tracks=gm_count[gm_bins]))
        bin_tables.append(bt)
        if len(gm_bins):
            starts = np.r_[0,np.flatnonzero(np.diff(gm_bins)>1)+1]
            ends = np.r_[starts[1:],len(gm_bins)]
            lo,hi = gm_bins[starts],gm_bins[ends-1]+1
        else:
            lo=hi=np.array([],dtype=int)
        fraction = np.lib.format.open_memmap(folder/'tracks_x_merged_fraction.npy',mode='w+',dtype='float32',shape=(nt,len(lo)))
        merged = np.lib.format.open_memmap(folder/'tracks_x_merged_presence.npy',mode='w+',dtype='uint8',shape=(nt,len(lo)))
        for tr in range(nt):
            calls = np.unpackbits(packed[tr],bitorder='little',count=n)
            union_mat[tr] = calls[union]
            cs = np.r_[0,np.cumsum(calls,dtype=np.int64)]
            fraction[tr] = (cs[hi]-cs[lo])/(hi-lo)
            merged[tr] = fraction[tr] >= MERGED_FRACTION
        union_mat.flush(); fraction.flush(); merged.flush()
        mt = pd.DataFrame(dict(chrom=ch,start=lo*BIN,end=hi*BIN,merged_index=np.arange(len(lo)),
                               n_bins=hi-lo,breadth=merged.sum(axis=0,dtype=np.uint16)))
        merged_tables.append(mt)
        np.save(folder/'breadth.npy',breadth)
        np.save(folder/'gm_track_count.npy',gm_count)
        progress(ch,'union bins',len(union),'GM bins',len(gm_bins),'merged',len(lo))
        del packed,union_mat,fraction,merged,y
    for kind,tables in [('bins',bin_tables),('merged',merged_tables)]:
        table = pd.concat(tables,ignore_index=True)
        assert len(table)>0, f'No GM peaks ({kind})'
        median = float(table.breadth.median())
        table['specificity'] = np.where(table.breadth<=median,'high_specificity','low_specificity')
        table['median_breadth'] = median
        table['peak_id'] = table.chrom+':'+table.start.astype(str)+'-'+table.end.astype(str)
        table.to_csv(CACHE/f'gm12878_{kind}.tsv.gz',sep='\t',index=False)
    atomic_json(marker,settings)

def build_cache():
    CACHE.mkdir(exist_ok=True)
    # Prevent two notebook/headless processes from writing the same cache concurrently.
    with (CACHE/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        numcodecs.blosc.set_nthreads(1)
        torch.set_num_threads(1)
        tracks,gm,metas = init_cache()
        progress('Build started; workers:', WORKERS)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda ch: pool_chrom(ch,metas[ch]),CHROMS))
        call_peaks(tracks,gm)
        progress('Genome-wide peak cache complete')

if __name__ == '__main__':
    build_cache()
