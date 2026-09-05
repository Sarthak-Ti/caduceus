"""Extract borzoi's GM12878 CAGE tracks (870 +, 871 -) from the fold3 model-output zarr.

Source is opened read-only and never modified. Output is written to a temp file and
atomically renamed, so a crash cannot leave a truncated .npy behind.
"""
import os, time
import numpy as np, zarr, numcodecs
from concurrent.futures import ThreadPoolExecutor

SRC = '/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3.zarr'
DST = '/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3_gm12878CAGE.npy'
OBS = '/data1/lesliec/sarthak/data/borzoi/GM12878CAGE.zarr'
TRACKS = [870, 871]   # CNhs12332+/-  CAGE:B lymphoblastoid cell line: GM12878 ENCODE

assert not os.path.exists(DST), f"refusing to overwrite existing {DST}"

nw = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
numcodecs.blosc.set_nthreads(1)   # parallelise across chunks, not within
print(f"workers={nw}  blosc_nthreads={numcodecs.blosc.get_nthreads()}", flush=True)

z = zarr.open(SRC, mode='r')[f'fold3']          # READ ONLY
N, L, T = z.shape
print(f"source {z.shape} {z.dtype} chunks={z.chunks}", flush=True)
out = np.empty((N, L, len(TRACKS)), dtype=z.dtype)   # float16, ~169 MB

t0 = time.perf_counter()
def one(i):
    out[i] = z[i][:, TRACKS]
    return i
with ThreadPoolExecutor(nw) as ex:
    for k, _ in enumerate(ex.map(one, range(N))):
        if (k + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1}/{N}  {el:7.1f}s  {(k+1)/el:5.2f} samp/s  eta {(N-k-1)/((k+1)/el)/60:5.1f} min", flush=True)
el = time.perf_counter() - t0
print(f"read {N} samples in {el/60:.1f} min ({N/el:.2f} samp/s)", flush=True)

tmp = DST + '.tmp.npy'
np.save(tmp, out)
os.replace(tmp, DST)
print(f"\nwrote {DST}  shape={out.shape} dtype={out.dtype} size={os.path.getsize(DST)/1e6:.0f} MB", flush=True)

print("\nverifying 12 random rows against direct zarr reads...", flush=True)
chk = np.load(DST, mmap_mode='r')
obs = zarr.open(OBS, mode='r')['fold3']
rng = np.random.RandomState(0)
ok = True
for i in rng.choice(N, 12, replace=False):
    ref = np.asarray(z[int(i)][:, TRACKS])
    if not np.array_equal(np.asarray(chk[int(i)]), ref):
        ok = False; print(f"  MISMATCH at row {i}")
print(f"  exact match on all 12 rows: {ok}")
print(f"  saved NaN count: {int(np.isnan(np.asarray(chk, dtype=np.float32)).sum())}")
print(f"  obs targets shape {obs.shape} -> row-aligned with saved array: {obs.shape[:2] == chk.shape[:2]}")
r = [np.corrcoef(np.asarray(chk[int(i)], dtype=np.float32)[:, c],
                 np.asarray(obs[int(i)], dtype=np.float32)[:, c])[0, 1]
     for i in rng.choice(N, 40, replace=False) for c in (0, 1)]
print(f"  mean Pearson vs observed over 40 random rows x 2 strands: {np.nanmean(r):.4f}  (expect ~0.60)")
