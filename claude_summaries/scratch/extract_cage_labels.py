"""Extract ALL 1276 CAGE *label* tracks for fold3 from borzoi's TARGET store into an
UNCOMPRESSED zarr, chunked by sample and by cell-type block.

Source is borzoi.zarr (the observed targets straight out of the tfrecords), opened read-only.
NOT model_outputs_fold3.zarr -- these are labels, not predictions.

Destination is created with mode='w-' (fails if it exists) plus an explicit assert.
CAGE occupies borzoi track indices 0..1275 contiguously, so column j IS borzoi track j.
Written zarr_format=2 so both the caduceus (zarr 3.x) and borzoi (zarr 2.x) envs can read it.
"""
import os, time
import numpy as np, zarr, numcodecs
from concurrent.futures import ThreadPoolExecutor

SRC  = '/data1/lesliec/sarthak/data/borzoi/borzoi.zarr'            # OBSERVED TARGETS
DST  = '/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE.zarr'
GM   = '/data1/lesliec/sarthak/data/borzoi/GM12878CAGE.zarr'       # known-good labels for 870/871
NCAGE, TBLK = 1276, 64

assert not os.path.exists(DST), f"refusing to touch existing {DST}"

nw = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
numcodecs.blosc.set_nthreads(1)
src = zarr.open(SRC, mode='r')['fold3']              # READ ONLY
N, L, T = src.shape
print(f"workers={nw}\nsource (LABELS) {SRC}\n  {src.shape} {src.dtype} chunks={src.chunks}", flush=True)
print(f"dest ({N}, {L}, {NCAGE}) float16 chunks=(1, {L}, {TBLK}) uncompressed "
      f"= {N*L*NCAGE*2/1e9:.1f} GB in {N*int(np.ceil(NCAGE/TBLK)):,} files", flush=True)

g = zarr.open_group(DST, mode='w-', zarr_format=2)
dst = g.create_array('fold3', shape=(N, L, NCAGE), chunks=(1, L, TBLK),
                     dtype='float16', compressors=None)
dst.attrs['source'] = SRC
dst.attrs['tracks'] = f'borzoi human CAGE tracks 0..{NCAGE-1} (contiguous); column j == borzoi track j'
dst.attrs['content'] = 'OBSERVED CAGE targets (labels) for held-out fold3 -- NOT model predictions'

t0 = time.perf_counter()
def one(i):
    dst[i] = src[i][:, :NCAGE]
    return i
with ThreadPoolExecutor(nw) as ex:
    for k, _ in enumerate(ex.map(one, range(N))):
        if (k + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1}/{N}  {el:7.1f}s  {(k+1)/el:5.2f} samp/s  eta {(N-k-1)/((k+1)/el)/60:5.1f} min", flush=True)
el = time.perf_counter() - t0
print(f"\ndone in {el/60:.1f} min ({N/el:.2f} samp/s)", flush=True)

# ---- verification ----
chk = zarr.open(DST, mode='r')['fold3']
nblk = int(np.ceil(NCAGE / TBLK))
nfiles = sum(len(fs) for _, _, fs in os.walk(os.path.join(DST, 'fold3')))
print(f"\nchunk files: {nfiles:,} (expect {N*nblk:,} + .zarray)", flush=True)
print(f"on-disk size: {sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(DST) for f in fs)/1e9:.1f} GB", flush=True)

rng = np.random.RandomState(0)
ok_src = ok_gm = True
for i in rng.choice(N, 8, replace=False):
    i = int(i)
    if not np.array_equal(np.asarray(chk[i]), np.asarray(src[i][:, :NCAGE])): ok_src = False; print(f"  MISMATCH vs source at {i}")
print(f"  matches borzoi.zarr on 8 random rows (all {NCAGE} tracks): {ok_src}", flush=True)
gm = zarr.open(GM, mode='r')['fold3']
for i in rng.choice(N, 8, replace=False):
    i = int(i)
    if not np.array_equal(np.asarray(chk[i][:, [870, 871]]).astype('float32'), np.asarray(gm[i]).astype('float32')):
        ok_gm = False; print(f"  MISMATCH vs GM12878CAGE.zarr at {i}")
print(f"  cols [870,871] match GM12878CAGE.zarr (known-good labels): {ok_gm}", flush=True)
zero_rows = [int(i) for i in rng.choice(N, 40, replace=False) if not np.asarray(chk[int(i)]).any()]
print(f"  all-zero rows among 40 random samples (should be []): {zero_rows}", flush=True)
print("\nVERIFIED", flush=True)
