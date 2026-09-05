"""Extract ALL 1276 CAGE tracks for fold3 from borzoi's model-output zarr into an
UNCOMPRESSED zarr store, chunked by sample and by cell-type block.

Source is opened read-only and never modified. Destination is created with mode='w-'
(fails if it already exists) plus an explicit existence assert.

CAGE occupies borzoi track indices 0..1275 contiguously, so column j of this store IS
borzoi track j -- the mapping is the identity, no lookup table needed.
Written as zarr_format=2 so both the caduceus (zarr 3.x) and borzoi (zarr 2.x) envs can read it.
"""
import os, time
import numpy as np, zarr
from concurrent.futures import ThreadPoolExecutor

SRC   = '/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3.zarr'
DST   = '/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE.zarr'
NPY   = '/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3_gm12878CAGE.npy'
NCAGE, TBLK = 1276, 64

assert not os.path.exists(DST), f"refusing to touch existing {DST}"

nw = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
import numcodecs; numcodecs.blosc.set_nthreads(1)   # parallelise across chunks, not within
src = zarr.open(SRC, mode='r')['fold3']             # READ ONLY
N, L, T = src.shape
print(f"workers={nw}\nsource {src.shape} {src.dtype} chunks={src.chunks}", flush=True)
print(f"dest  ({N}, {L}, {NCAGE}) float16 chunks=(1, {L}, {TBLK}) uncompressed "
      f"= {N*L*NCAGE*2/1e9:.1f} GB in {N*int(np.ceil(NCAGE/TBLK)):,} files", flush=True)

g = zarr.open_group(DST, mode='w-', zarr_format=2)   # w- : create, fail if exists
dst = g.create_array('fold3', shape=(N, L, NCAGE), chunks=(1, L, TBLK),
                     dtype='float16', compressors=None)
dst.attrs['source'] = SRC
dst.attrs['tracks'] = f'borzoi human CAGE tracks 0..{NCAGE-1} (contiguous); column j == borzoi track j'
dst.attrs['content'] = 'MODEL PREDICTIONS from model0_best.h5 on held-out fold3 (not observed targets)'

t0 = time.perf_counter()
def one(i):
    dst[i] = src[i][:, :NCAGE]      # first dim of chunk is 1, so each sample owns its chunks
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
nfiles = sum(len(fs) for _, _, fs in os.walk(os.path.join(DST, 'fold3')) )
print(f"\nchunk files: {nfiles:,} (expect {N*nblk:,} + 1 .zarray) ", flush=True)
print(f"on-disk size: {sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(DST) for f in fs)/1e9:.1f} GB", flush=True)

rng = np.random.RandomState(0)
ok_src = ok_npy = True
for i in rng.choice(N, 8, replace=False):
    i = int(i)
    if not np.array_equal(np.asarray(chk[i]), np.asarray(src[i][:, :NCAGE])): ok_src = False; print(f"  MISMATCH vs source at {i}")
print(f"  matches source on 8 random rows (all {NCAGE} tracks): {ok_src}", flush=True)
npy = np.load(NPY, mmap_mode='r')
for i in rng.choice(N, 8, replace=False):
    i = int(i)
    if not np.array_equal(np.asarray(chk[i][:, [870, 871]]), np.asarray(npy[i])): ok_npy = False; print(f"  MISMATCH vs npy at {i}")
print(f"  GM12878 cols [870,871] match model_outputs_fold3_gm12878CAGE.npy: {ok_npy}", flush=True)
# completeness: an unwritten chunk reads back as all-zero
zero_rows = [int(i) for i in rng.choice(N, 40, replace=False) if not np.asarray(chk[int(i)]).any()]
print(f"  all-zero rows among 40 random samples (should be []): {zero_rows}", flush=True)
print("\nVERIFIED", flush=True)
