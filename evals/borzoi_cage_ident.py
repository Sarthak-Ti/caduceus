#!/usr/bin/env python
"""
Cell-type identifiability: score every model's GM12878 prediction against every other
cell type's observed CAGE, per region, per strand.

This is the second pass over borzoi_fold3_CAGE.zarr. The first (borzoi_cage_avg.py)
cached target-vs-target correlations R(TRUE_GM12878, TRUE_ct); those set the oracle
ceiling but say nothing about the models. Section 16 of comparing_model4 asks the sharper
question -- does a model's GM12878 prediction actually look more like GM12878 than like
some other cell type -- and that needs R(pred, TRUE_ct), which is what this writes.

Reads   borzoi_fold3_CAGE.zarr/fold3 (6888, 6144, 1276) float16   -- 108 GB
        the six model prediction files (see ARMS below)
Writes  /data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_ident.npz

    r_<key>_<strand>    (6888, 638) float32   raw Pearson, pred vs each cell type
    rsm_<key>_<strand>  (6888, 638) float32   smoothed-log Pearson (log1p, sigma=6.0)

Column order follows cols_plus / cols_minus from borzoi_fold3_CAGE_avg.npz, so column
435 is GM12878 itself -- i.e. r_<key>_<strand>[:, 435] IS the model's own accuracy, and
the identifiability margin is that column minus the max over the others.

DNase is dropped on load; the finetuned arms carry it in channel 2 and nothing here
uses it.

    python evals/borzoi_cage_ident.py --limit 32     # smoke test, writes nothing
    sbatch evals/borzoi_cage_ident.sh
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d

CAGE_ZARR = "/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE.zarr"
AVG_NPZ = "/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_avg.npz"
OUT = "/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_ident.npz"
MO = "/data1/lesliec/sarthak/data/borzoi/model_outputs"
JP = "/data1/lesliec/sarthak/data/joint_playground/model_out"

N_CAGE = 1276
GM_PLUS, GM_MINUS = 870, 871
SMOOTH_SIGMA = 6.0
BLOCK = 8

COLS_PLUS = np.arange(0, N_CAGE, 2, dtype=np.int32)
COLS_MINUS = np.arange(1, N_CAGE, 2, dtype=np.int32)
GM_SLOT = int(np.flatnonzero(COLS_PLUS == GM_PLUS)[0])

# key -> (path, how to pull out (plus, minus) for one region). Channel order is
# (+, -, [DNase]) everywhere; the caduceus arrays carry a length-1 axis at position 1.
ARMS = [
    ("borzoi_pre", f"/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3_gm12878CAGE.npy", "hw2"),
    ("borzoi_lora", f"{MO}/borzoi_finetuned_gm12878_lorav2.npy", "hw3"),
    ("borzoi_trunk", f"{MO}/borzoi_finetuned_gm12878_trunk.npy", "hw3"),
    ("borzoi_scratch", f"{MO}/borzoi_finetuned_gm12878_scratch.npy", "hw3"),
    ("cad_ep4", f"{JP}/borzoi_gm12878_cage_ep4_outputs.npy", "h1w2"),
    ("cad_ep10", f"{JP}/borzoi_gm12878_cage_ep10_outputs.npy", "h1w2"),
]

_Z = None
_P = None


def _init():
    global _Z, _P
    _Z = zarr.open(CAGE_ZARR, mode="r")["fold3"]
    _P = {k: (np.load(p, mmap_mode="r"), lay) for k, p, lay in ARMS}


def _slab(key, lo, hi):
    """(n, 6144, 2) float32 for one arm, DNase dropped."""
    arr, lay = _P[key]
    a = np.asarray(arr[lo:hi])
    if lay == "h1w2":
        a = a[:, 0]
    return a[:, :, :2].astype(np.float32)


def _corr_cols(X, y):
    """Pearson of vector y against every column of X. (T, C), (T,) -> (C,)."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / (
            np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()))
    r[~np.isfinite(r)] = np.nan
    return r.astype(np.float32)


def _block(bounds):
    lo, hi = bounds
    raw = np.asarray(_Z[lo:hi])
    n = hi - lo
    slabs = {k: _slab(k, lo, hi) for k, _, _ in ARMS}
    out = {}
    for key, _, _ in ARMS:
        for tag in ("plus", "minus"):
            out[f"r_{key}_{tag}"] = np.empty((n, len(COLS_PLUS)), np.float32)
            out[f"rsm_{key}_{tag}"] = np.empty((n, len(COLS_PLUS)), np.float32)
    for k in range(n):
        a = raw[k].astype(np.float32)
        for si, (tag, cols) in enumerate((("plus", COLS_PLUS), ("minus", COLS_MINUS))):
            X = a[:, cols]
            # smoothing the 638-track panel dominates the cost, so do it once per
            # strand and reuse it for all six arms
            L = gaussian_filter1d(np.log1p(np.clip(X, 0, None)), sigma=SMOOTH_SIGMA,
                                  axis=0, truncate=4.0)
            for key, _, _ in ARMS:
                p = slabs[key][k, :, si]
                out[f"r_{key}_{tag}"][k] = _corr_cols(X, p)
                ps = gaussian_filter1d(np.log1p(np.clip(p, 0, None)),
                                       sigma=SMOOTH_SIGMA, truncate=4.0)
                out[f"rsm_{key}_{tag}"][k] = _corr_cols(L, ps)
    return lo, hi, out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default=OUT)
    p.add_argument("-j", "--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    missing = [p_ for _, p_, _ in ARMS if not os.path.exists(p_)]
    assert not missing, "prediction files not on disk yet:\n  " + "\n  ".join(missing)

    # the column convention must match the first pass, or the two npz files cannot be
    # indexed together
    a = np.load(AVG_NPZ)
    assert np.array_equal(a["cols_plus"], COLS_PLUS), "cols_plus differs from borzoi_cage_avg.npz"
    assert json.loads(str(a["meta"]))["gm_slot"] == GM_SLOT, "gm_slot differs from the first pass"

    z = zarr.open(CAGE_ZARR, mode="r")["fold3"]
    n_all = z.shape[0]
    n = n_all if args.limit is None else min(args.limit, n_all)
    for k, p_, lay in ARMS:
        arr = np.load(p_, mmap_mode="r")
        assert arr.shape[0] == n_all, f"{k}: {arr.shape[0]} regions, expected {n_all}"
        print(f"  {k:<16} {str(arr.shape):<22} {arr.dtype} layout={lay}")
    print(f"fold3 regions {n_all}, processing {n}, workers {args.workers}")

    res = {}
    for key, _, _ in ARMS:
        for tag in ("plus", "minus"):
            res[f"r_{key}_{tag}"] = np.empty((n, len(COLS_PLUS)), np.float32)
            res[f"rsm_{key}_{tag}"] = np.empty((n, len(COLS_PLUS)), np.float32)

    blocks = [(s, min(s + BLOCK, n)) for s in range(0, n, BLOCK)]
    t0, done = time.time(), 0
    with Pool(args.workers, initializer=_init) as pool:
        for lo, hi, out in pool.imap_unordered(_block, blocks):
            for k, v in out.items():
                res[k][lo:hi] = v
            done += hi - lo
            if done % 400 < BLOCK:
                el = time.time() - t0
                print(f"  {done}/{n}  {done / el:.1f}/s  "
                      f"eta {(n - done) / max(done / el, 1e-9) / 60:.0f} min", flush=True)
    print(f"pass complete in {(time.time() - t0) / 60:.1f} min")

    # gate: column GM_SLOT is the model against GM12878 itself, i.e. its own accuracy.
    # It must NOT be 1.0 (that would mean the prediction file is a copy of the target)
    # and it must be the same number the eval script recorded for the arms that have one.
    print(f"\n{'arm':<16}{'own R+ (raw)':>14}{'own R- (raw)':>14}"
          f"{'max other+':>12}{'max other-':>12}{'margin+':>10}{'margin-':>10}")
    for key, _, _ in ARMS:
        row = []
        for tag in ("plus", "minus"):
            r = res[f"r_{key}_{tag}"]
            own = r[:, GM_SLOT]
            assert np.nanmedian(own) < 0.999, (
                f"GATE FAILED: {key} {tag} correlates ~1.0 with the target -- the "
                f"prediction file is the target")
            oth = np.delete(r, GM_SLOT, axis=1)
            row.append((np.nanmedian(own), np.nanmedian(np.nanmax(oth, axis=1))))
        print(f"{key:<16}{row[0][0]:>14.4f}{row[1][0]:>14.4f}"
              f"{row[0][1]:>12.4f}{row[1][1]:>12.4f}"
              f"{row[0][0] - row[0][1]:>10.4f}{row[1][0] - row[1][1]:>10.4f}")

    if args.limit is not None:
        print("--limit set; not writing")
        return

    meta = dict(source=CAGE_ZARR, arms=[k for k, _, _ in ARMS], n_regions=int(n),
                n_cell_types=int(len(COLS_PLUS)), gm_slot=GM_SLOT,
                smooth_sigma=SMOOTH_SIGMA,
                note="column order follows cols_* in borzoi_fold3_CAGE_avg.npz; "
                     "column gm_slot is the arm's own GM12878 accuracy")
    np.savez(args.out, meta=json.dumps(meta), **res)
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
