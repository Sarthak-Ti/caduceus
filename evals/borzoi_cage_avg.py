#!/usr/bin/env python
"""
Precompute the cross-cell-type CAGE average over Borzoi fold3, per strand.

The Enformer notebooks (comparing_model3/4 sections 5-7 and 14-17) are all defined
against AVG_ALL, the mean CAGE track over every cell type. This is the Borzoi analogue.
Everything is kept STRAND-SEPARATE: + and - are treated as independent samples, so no
array in the output mixes them.

Reads   /data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE.zarr/fold3
        (6888, 6144, 1276) float16, uncompressed, chunks (1, 6144, 64) -- 108 GB
Writes  /data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_avg.npz  (~750 MB)

    avg_plus,  avg_minus       (6888, 6144) float32  mean over all 638 same-strand CTs
    avg_plus_loo, avg_minus_loo (6888, 6144) float32  same, GM12878 left out (637 CTs)
    r_ct_plus, r_ct_minus      (6888, 638)  float32  raw Pearson, GM12878 vs each CT
    rsm_ct_plus, rsm_ct_minus  (6888, 638)  float32  smoothed-log Pearson, same
    cols_plus, cols_minus      (638,) int32          borzoi track index of each column

WHY THE LEAVE-ONE-OUT AVERAGE
-----------------------------
comparing_model4 section 14b established that including GM12878 in AVG moves `gap` by
<=0.001, but section 19a point 1 warns that result does NOT transfer to a residual
metric: r_spec = R(pred - AVG, TRUE - AVG) has 1/638 of TRUE subtracted out of its own
residual. Both versions are written so the residual work can use the honest one.

WHY THE PER-CELL-TYPE CORRELATIONS
----------------------------------
Section 16's identifiability margin needs R(TRUE_GM12878, TRUE_other_ct) for every other
cell type. Computing it here costs nothing extra -- the region is already resident -- and
saves a second pass over 108 GB. The full (6888, 638) matrix is stored rather than just
the max, so any comparison panel can be selected later without re-reading.

SIGMA
-----
The Enformer notebooks smooth with sigma=1.5 on 128 bp bins. Borzoi bins are 32 bp, so
the same genomic width is sigma = 1.5 * 128 / 32 = 6.0.

GATES
-----
Nothing here is assumed. Before the pass: targets_human.txt must report tracks 0..1275 as
CAGE with strand_pair mapping evens<->odds, and 870/871 must be the GM12878 pair; a
sample of regions from borzoi_fold3_CAGE.zarr must be bit-identical to GM12878CAGE.zarr
at those tracks. After the pass: column 435 of r_ct_* is GM12878 against itself and must
be exactly 1.0 in every region -- if the strand/column mapping were wrong it would not be.

    python evals/borzoi_cage_avg.py --limit 32     # smoke test, writes nothing
    sbatch evals/borzoi_cage_avg.sh
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter1d

CAGE_ZARR = "/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE.zarr"
GM_ZARR = "/data1/lesliec/sarthak/data/borzoi/GM12878CAGE.zarr"
TARGETS = "/data1/lesliec/sarthak/data/borzoi/targets_human.txt"
OUT = "/data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_avg.npz"

N_CAGE = 1276                  # borzoi tracks 0..1275 are all of CAGE, contiguous
GM_PLUS, GM_MINUS = 870, 871   # CNhs12332+ / CNhs12332-
SMOOTH_SIGMA = 6.0             # 1.5 at 128 bp -> 6.0 at 32 bp
BLOCK = 8

COLS_PLUS = np.arange(0, N_CAGE, 2, dtype=np.int32)
COLS_MINUS = np.arange(1, N_CAGE, 2, dtype=np.int32)
# position of GM12878 inside each same-strand block
GM_SLOT = int(np.flatnonzero(COLS_PLUS == GM_PLUS)[0])

_Z = None


def _init():
    global _Z
    _Z = zarr.open(CAGE_ZARR, mode="r")["fold3"]


def _corr_vs_one(X, j):
    """Pearson of column j against every column of X. (T, C) -> (C,)."""
    Xc = X - X.mean(axis=0)
    ss = np.sqrt((Xc ** 2).sum(axis=0))
    y = Xc[:, j]
    sy = ss[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * y[:, None]).sum(axis=0) / (ss * sy)
    r[~np.isfinite(r)] = np.nan
    return r.astype(np.float32)


def _block(bounds):
    lo, hi = bounds
    raw = np.asarray(_Z[lo:hi])
    n = hi - lo
    out = {
        "avg_plus": np.empty((n, 6144), np.float32),
        "avg_minus": np.empty((n, 6144), np.float32),
        "avg_plus_loo": np.empty((n, 6144), np.float32),
        "avg_minus_loo": np.empty((n, 6144), np.float32),
        "r_ct_plus": np.empty((n, len(COLS_PLUS)), np.float32),
        "r_ct_minus": np.empty((n, len(COLS_MINUS)), np.float32),
        "rsm_ct_plus": np.empty((n, len(COLS_PLUS)), np.float32),
        "rsm_ct_minus": np.empty((n, len(COLS_MINUS)), np.float32),
    }
    for k in range(n):
        a = raw[k].astype(np.float32)
        for tag, cols, gm in (("plus", COLS_PLUS, GM_PLUS), ("minus", COLS_MINUS, GM_MINUS)):
            X = a[:, cols]
            s = X.sum(axis=1)
            out[f"avg_{tag}"][k] = s / len(cols)
            # the leave-one-out mean falls straight out of the total; no second sum
            out[f"avg_{tag}_loo"][k] = (s - a[:, gm]) / (len(cols) - 1)
            out[f"r_ct_{tag}"][k] = _corr_vs_one(X, GM_SLOT)
            L = gaussian_filter1d(np.log1p(np.clip(X, 0, None)), sigma=SMOOTH_SIGMA,
                                  axis=0, truncate=4.0)
            out[f"rsm_ct_{tag}"][k] = _corr_vs_one(L, GM_SLOT)
    return lo, hi, out


def gate_before():
    t = pd.read_csv(TARGETS, sep="\t")
    cage = t.iloc[:N_CAGE]
    assert cage["description"].str.contains("CAGE", case=False).all(), \
        "GATE FAILED: tracks 0..1275 are not all CAGE"
    assert not t.iloc[N_CAGE]["description"].upper().startswith("CAGE"), \
        f"GATE FAILED: track {N_CAGE} is also CAGE, so the CAGE block is longer than assumed"
    sp = cage["strand_pair"].to_numpy()
    assert (sp[COLS_PLUS] == COLS_MINUS).all() and (sp[COLS_MINUS] == COLS_PLUS).all(), \
        "GATE FAILED: strand_pair does not map even<->odd, so parity is not strand"
    for i, sign in ((GM_PLUS, "+"), (GM_MINUS, "-")):
        row = t.iloc[i]
        assert "GM12878" in row["description"], \
            f"GATE FAILED: track {i} is {row['description']!r}, not GM12878"
        assert row["identifier"].endswith(sign), \
            f"GATE FAILED: track {i} identifier {row['identifier']} is not strand {sign}"

    # the identity-mapping claim: column j of this store IS borzoi track j
    z = zarr.open(CAGE_ZARR, mode="r")["fold3"]
    g = zarr.open(GM_ZARR, mode="r")["fold3"]
    nz = z.shape[0]
    probe = [0, 1, 2, nz // 2, nz - 1]
    for i in probe:
        a = np.asarray(z[i][:, [GM_PLUS, GM_MINUS]]).astype(np.float32)
        b = np.asarray(g[i]).astype(np.float32)
        assert np.array_equal(a, b), \
            f"GATE FAILED: region {i} at tracks [{GM_PLUS},{GM_MINUS}] differs from GM12878CAGE.zarr"
    # control: a neighbouring track must NOT match, else the check proves nothing
    off = np.asarray(z[probe[0]][:, [GM_PLUS + 2, GM_MINUS + 2]]).astype(np.float32)
    assert not np.array_equal(off, np.asarray(g[probe[0]]).astype(np.float32)), \
        "GATE FAILED: the neighbouring cell type also matches, so the track mapping is not discriminating"
    print(f"pre-gate passed: CAGE 0..{N_CAGE - 1}, even=+/odd=-, GM12878 at "
          f"{GM_PLUS}/{GM_MINUS} = slot {GM_SLOT}, store agrees with GM12878CAGE.zarr "
          f"on regions {probe} and neighbours differ")
    return z.shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default=OUT)
    p.add_argument("-j", "--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N regions; does not write the npz")
    args = p.parse_args()

    n_all = gate_before()
    n = n_all if args.limit is None else min(args.limit, n_all)
    print(f"fold3 regions {n_all}, processing {n}, workers {args.workers}, block {BLOCK}")

    res = {k: np.empty((n, 6144) if k.startswith("avg") else (n, len(COLS_PLUS)), np.float32)
           for k in ("avg_plus", "avg_minus", "avg_plus_loo", "avg_minus_loo",
                     "r_ct_plus", "r_ct_minus", "rsm_ct_plus", "rsm_ct_minus")}

    blocks = [(s, min(s + BLOCK, n)) for s in range(0, n, BLOCK)]
    t0 = time.time()
    done = 0
    with Pool(args.workers, initializer=_init) as pool:
        for lo, hi, out in pool.imap_unordered(_block, blocks):
            for k, v in out.items():
                res[k][lo:hi] = v
            done += hi - lo
            if done % 400 < BLOCK:
                el = time.time() - t0
                print(f"  {done}/{n} regions  {done / el:.1f}/s  "
                      f"eta {(n - done) / max(done / el, 1e-9) / 60:.0f} min", flush=True)
    print(f"pass complete in {(time.time() - t0) / 60:.1f} min")

    # post-gate: the self-column must be exactly 1.0 everywhere
    for tag in ("plus", "minus"):
        for key in (f"r_ct_{tag}", f"rsm_ct_{tag}"):
            self_r = res[key][:, GM_SLOT]
            bad = int((np.abs(self_r - 1.0) > 1e-4).sum())
            assert bad == 0, (
                f"GATE FAILED: {key}[:, {GM_SLOT}] is GM12878 against itself but departs "
                f"from 1.0 in {bad} regions (min {np.nanmin(self_r):.6f}) -- the column "
                f"mapping is wrong")
        # the loo average must actually differ from the full average
        d = float(np.abs(res[f"avg_{tag}"] - res[f"avg_{tag}_loo"]).max())
        assert d > 0, f"GATE FAILED: avg_{tag}_loo is identical to avg_{tag}"
        print(f"post-gate {tag}: self-correlation exactly 1.0 in all {n} regions; "
              f"max|avg - avg_loo| = {d:.6g}")

    for tag in ("plus", "minus"):
        a = res[f"avg_{tag}"]
        r = res[f"r_ct_{tag}"]
        # slot GM_SLOT is GM12878 vs itself; exclude it from the summary
        oth = np.delete(r, GM_SLOT, axis=1)
        print(f"  {tag}: AVG mean {a.mean():.5f} max {a.max():.3f} | "
              f"R(GM12878, other CT) median {np.nanmedian(oth):.4f}, "
              f"max-over-CT median {np.nanmedian(np.nanmax(oth, axis=1)):.4f}")

    if args.limit is not None:
        print("--limit set; not writing")
        return

    meta = dict(source=CAGE_ZARR, n_regions=int(n), n_bins=6144, bin_bp=32,
                n_cell_types=int(len(COLS_PLUS)), gm_track_plus=GM_PLUS,
                gm_track_minus=GM_MINUS, gm_slot=GM_SLOT, smooth_sigma=SMOOTH_SIGMA,
                note="strand-separate; loo excludes GM12878; r_ct columns follow cols_*")
    np.savez(args.out, cols_plus=COLS_PLUS, cols_minus=COLS_MINUS,
             meta=json.dumps(meta), **res)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
