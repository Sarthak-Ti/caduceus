"""Cache builder for IGVF_meeting_figures2 — every slow read happens here, once.

Nothing in this module draws. It opens the CAGE targets, the prediction arrays,
the cross-cell-type averages and the accessibility tracks, reduces each panel to
the few kilobytes it actually plots, and writes that to `CACHE`. The analysis
module never touches a source store, so restyling a figure is free.

Sizes, to make the point: the Borzoi fold3 grid alone is a 339 MB target plus
seven 339 MB prediction arrays plus a 170 MB DNase track. What figure 2 needs
out of all that is 6,888 x 2 correlations per arm, and what figure 1 needs is
one 6,144-bin slice. The reduction is ~4 orders of magnitude.

Stages, each independently runnable and independently cached:

    enformer   128 bp GM12878 CAGE grid   -> boxplot, generic-vs-accuracy, ladder
    borzoi     32 bp fold3 grid           -> boxplot, ladder, the locus panel
    qtl        eQTL + dsQTL               -> panel D table, dsQTL strata

Run:  python igvf_meeting_figures2_pipeline.py [enformer|borzoi|qtl|all]

Arm identity is resolved through `igvf_arms` at read time, so every array in the
cache is keyed on a canonical arm id and no source-notebook label survives into
the figures.
"""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

sys.path.insert(0, '/data1/lesliec/sarthak/caduceus')
from evals.plotting_utils import GeneAnnotation

import igvf_arms as A

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('IGVF2_CACHE', ROOT / 'IGVF_meeting_figures2_cache'))
CACHE.mkdir(exist_ok=True)

DATA = '/data1/lesliec/sarthak/data'
E_MO = f'{DATA}/joint_playground/model_out'
BORZOI = f'{DATA}/borzoi'
B_MO = f'{BORZOI}/model_outputs'
JP = f'{DATA}/joint_playground/model_out'
QTL = f'{DATA}/joint_playground/eQTL/EPCOTv2_LCLs/'
DSQ = f'{DATA}/joint_playground/dsQTL'
CBP = f'{DATA}/chrombpnet/dsqtls.yoruba.lcls.benchmarking.all.tsv'
DSQTL8_CACHE = ROOT / 'dsQTL_benchmark8_cache'

# --- the example locus for figure 1 -----------------------------------------
# comparing_model5 §9c rank 9: chr8:40,763,508-40,960,116, minus strand, a
# +0.557 gain for cis-Mamba over pretrained Borzoi. Chosen by the presenter.
# Provenance worth keeping straight: §9c ranked loci using the epoch-10 Caduceus
# run, which is no longer a displayed arm anywhere in these figures. The locus is
# therefore rank 9 of a list built with a sibling checkpoint -- selected for
# illustration either way, which is why figure 2 carries the distribution.
LOCUS_REGION, LOCUS_STRAND = 3304, 'minus'

# --- Enformer grid constants (comparing_model4 §8a / comparing_model6 §1) ----
E_BIN, E_T, E_TRACK, E_CAGE_START = 128, 896, 5110, 4675
E_CROP = (131072 - E_T * E_BIN) // 2
E_CTX, E_OVF = 524288, 0.20
E_ACTIVE, E_SILENT, E_TSSPAD = 1.0, 0.2, 500
E_DECIM = 1
E_SMOOTH_SIGMA = 1.5
E_K_MIN = 3                    # §12a: a region needs >= 3 true peak bins to be eligible

# --- Borzoi grid constants (comparing_model5 §1 / comparing_model6 §2) -------
N_REG, N_BIN, BIN_BP = 6888, 6144, 32
SEQ_LEN, TGT_SPAN = 524288, 196608
STRANDS = ('plus', 'minus')
GM_SLOT = 435
B_ACTIVE, B_SILENT, B_TSSPAD = 0.50, 0.05, 500
B_DECIM = 10
B_SMOOTH_SIGMA = 6.0           # 1.5 at 128 bp -> 6.0 at 32 bp

N_BOOT, BOOT_SEED = 200, 0


def progress(*parts):
    message = ' '.join(map(str, parts))
    print(message, flush=True)
    with (CACHE / 'build.log').open('a') as log:
        log.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')


def mark_done(stage, **info):
    (CACHE / f'{stage}.done.json').write_text(json.dumps(
        dict(stage=stage, built=time.strftime('%Y-%m-%d %H:%M:%S'), **info), indent=2))


def is_done(stage):
    return (CACHE / f'{stage}.done.json').exists()


# ============================================================================
# Shared numerics. Ported unchanged from the source notebooks so a cached number
# is the same number those notebooks print.
# ============================================================================
def rowwise_pearson(X, Y):
    """Pearson per row of two (N, T) arrays -> (N,). nan where a row is constant."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (Xc * Yc).sum(axis=1) / np.sqrt((Xc * Xc).sum(axis=1) * (Yc * Yc).sum(axis=1))
    r = np.asarray(r, dtype=np.float32)
    r[~np.isfinite(r)] = np.nan
    return r


def smooth_log(X, sigma):
    """log1p then gaussian smooth along the bin axis. Order matters."""
    return gaussian_filter1d(np.log1p(np.clip(X, 0, None)), sigma=sigma, axis=1, truncate=4.0)


def smoothed_log_pearson(X, Y, sigma):
    return rowwise_pearson(smooth_log(X, sigma), smooth_log(Y, sigma))


# --- the ladder engine, verbatim from comparing_model6 §0 -------------------
def s_auroc(pos, neg):
    """P(a random positive outranks a random negative). Reference implementation."""
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def to_codes(vp, vn):
    """Map both classes onto a shared sorted-value index. One sort per arm-rung."""
    uniq, codes = np.unique(np.concatenate([vp, vn]), return_inverse=True)
    codes = np.asarray(codes).ravel()
    return codes[:len(vp)], codes[len(vp):], len(uniq)


def auroc_codes(cp_codes, cn_codes, n_uniq):
    """Mann-Whitney AUROC from value codes, same 0.5-credit tie handling as s_auroc."""
    cp = np.bincount(cp_codes, minlength=n_uniq).astype(np.float64)
    cn = np.bincount(cn_codes, minlength=n_uniq).astype(np.float64)
    below = np.concatenate([[0.0], np.cumsum(cn)[:-1]])
    return float((cp * below).sum() + 0.5 * float((cp * cn).sum())) / (cp.sum() * cn.sum())


def group_by_region(mask, rid):
    """Bin indices ordered by region, plus per-region slices -- makes the bootstrap cheap."""
    idx = np.flatnonzero(mask)
    r = rid[idx]
    o = np.argsort(r, kind='stable')
    idx, r = idx[o], r[o]
    regs, starts = np.unique(r, return_index=True)
    stops = np.append(starts[1:], len(idx))
    return idx, np.stack([starts, stops], axis=1), regs


def make_rung(name, subtitle, pos_mask, neg_mask, rid):
    pi, ps, pr = group_by_region(pos_mask, rid)
    ni, ns, nr = group_by_region(neg_mask, rid)
    return dict(name=name, subtitle=subtitle, pi=pi, ps=ps, n_preg=len(pr),
                ni=ni, ns=ns, n_nreg=len(nr), n_pos=len(pi), n_neg=len(ni))


def _draw(codes, sl, n_reg, rng):
    pick = rng.integers(0, n_reg, n_reg)
    return np.concatenate([codes[sl[k, 0]:sl[k, 1]] for k in pick])


def score_rung(pv, rung, n_boot=N_BOOT, seed=BOOT_SEED):
    """Point estimate + region-level bootstrap CI for one arm on one rung."""
    vp, vn = pv[rung['pi']], pv[rung['ni']]
    cp, cn, U = to_codes(vp, vn)
    a0 = auroc_codes(cp, cn, U)
    if n_boot == 0:
        return a0, float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bo = np.empty(n_boot)
    for b in range(n_boot):
        bo[b] = auroc_codes(_draw(cp, rung['ps'], rung['n_preg'], rng),
                            _draw(cn, rung['ns'], rung['n_nreg'], rng), U)
    lo, hi = np.percentile(bo, [2.5, 97.5])
    return a0, float(lo), float(hi)


def rungs_to_json(rungs):
    """The plot-facing half of a rung: everything except the index arrays."""
    return [dict(name=r['name'], subtitle=r['subtitle'], n_pos=int(r['n_pos']),
                 n_neg=int(r['n_neg']), n_preg=int(r['n_preg']), n_nreg=int(r['n_nreg']))
            for r in rungs]


def tss_mask(chroms, centers, gene_df, strand=None, pad=500):
    """Boolean (n_region, n_bin): is this bin within `pad` bp of a protein-coding TSS."""
    df = gene_df[gene_df['feature'] == 'transcript']
    if strand is not None:
        df = df[df['strand'] == strand]
    by_chrom = {}
    for ch, g in df.groupby('chrom'):
        p = np.where(g['strand'].to_numpy() == '+', g['start0'].to_numpy(), g['end0'].to_numpy())
        by_chrom[str(ch)] = np.sort(p)
    out = np.zeros(centers.shape, bool)
    for ch in np.unique(chroms):
        arr = by_chrom.get(ch)
        if arr is None or arr.size == 0:
            continue
        rows = np.flatnonzero(chroms == ch)
        c = centers[rows]
        i = np.searchsorted(arr, c)
        lo = arr[np.clip(i - 1, 0, arr.size - 1)]
        hi = arr[np.clip(i, 0, arr.size - 1)]
        out[rows] = np.minimum(np.abs(c - lo), np.abs(c - hi)) <= pad
    return out


# ============================================================================
# STAGE: Enformer grid (128 bp). Figures 2a, 3a and 4.
# Loading and gating follow comparing_model6 §1, which is comparing_model4 §20.
# ============================================================================
def build_enformer():
    progress('[enformer] loading target and nine arms')
    E_TRUE = np.load(f'{E_MO}/GM12878_base_targets_conv.npy').astype(np.float32)

    # source label -> loader. Same gating as comparing_model6 §1: every arm must
    # carry the GM12878 reference target, or it is not scored against it.
    E_SPEC = [
        ('cis-Mamba (1 cell type)', 'mamba', None),
        ('cis-Mamba, no pretraining', 'nopretrain', None),
        ('cis-Mamba (8 cell types)', 'immune', None),
        ('Enformer (pretrained)', 'enformer', None),
        ('AlphaGenome', 'alphagenome', None),
        ('Enformer retrained, 1 CT', 'enformer_retrained_gm12878',
         'enformer_retrained_gm12878'),
        ('Enformer retrained, 1 CT (1536ch)', 'enformer_retrained_n1_1536',
         'enformer_retrained_n1_1536'),
        ('Enformer retrained, 16 CTs', 'enformer_retrained_n16', 'enformer_retrained_n16'),
        ('Enformer retrained, 42 CTs', 'enformer_retrained_n42', 'enformer_retrained_n42'),
    ]
    E_P = {}
    for src_label, key, stem in E_SPEC:
        arm = A.to_id(A.ENFORMER_GRID, src_label)
        if key == 'enformer':
            arr = np.load(f'{E_MO}/enformer_test_track{E_TRACK}.npy').astype(np.float32)
        elif key == 'alphagenome':
            arr = np.asarray(np.load(f'{E_MO}/alphagenome_cage_test_128bp.npy',
                                     mmap_mode='r')[:, :, 156], dtype=np.float32)
        elif key in ('mamba', 'immune', 'nopretrain'):
            f = {'mamba': 'GM12878_no_mlm', 'immune': 'immune_CAGE',
                 'nopretrain': 'nopretrain'}[key]
            _p = np.load(f'{E_MO}/{f}_predictions.npy')
            arr = (_p[:, :, 0] if _p.ndim == 3 else _p).astype(np.float32)
            _t = np.load(f'{E_MO}/{f}_targets.npy')
            _t = _t[:, :, 0] if _t.ndim == 3 else _t
            assert np.allclose(np.asarray(_t, np.float32), E_TRUE, rtol=1e-5, atol=1e-6), \
                f'GATE FAILED [{arm}]: its target is not the GM12878 reference target'
        else:
            d = f'{E_MO}/{key}'
            pr = np.load(f'{d}/{stem}_cage_predictions.npy', mmap_mode='r')
            tg = np.load(f'{d}/{stem}_cage_targets.npy', mmap_mode='r')
            if pr.ndim == 3:
                tr = pd.read_csv(f'{d}/{stem}_cage_tracks.csv')
                row = tr[(tr['cell_type'] == 'GM12878') & (tr['assay'] == 'cage')]
                assert len(row) == 1 and int(row['target_index'].iloc[0]) == E_TRACK
                col = int(row['column'].iloc[0])
                pr, tg = pr[:, :, col], tg[:, :, col]
            assert np.allclose(np.asarray(tg), E_TRUE, rtol=1e-5, atol=1e-6), \
                f'GATE FAILED [{arm}]'
            arr = np.ascontiguousarray(pr, dtype=np.float32)
        assert arr.shape == E_TRUE.shape
        E_P[arm] = arr
    progress(f'[enformer] {len(E_P)} arms loaded, all target-gated')

    # --- leave-GM12878-out cross-cell-type average, and the ATAC witness -----
    _lab = zarr.open(f'{DATA}/enformer/data/labels.zarr', mode='r')['test']
    n_cage = _lab.shape[2] - E_CAGE_START
    _gm = np.asarray(_lab[:, :, E_TRACK], dtype=np.float32)
    assert np.array_equal(_gm, E_TRUE)
    E_AVGALL = np.load(f'{DATA}/enformer/data/avg_cage_test_all.npy')
    E_AVG = np.clip((E_AVGALL * n_cage - _gm) / (n_cage - 1), 0, None)
    E_ATAC = np.load(f'{DATA}/enformer/data/GM12878_ATAC_WW_denoised.npz')['test'][:, :, 0]

    # --- blacklist: test regions whose input context reaches into a train window
    _bed = pd.read_csv(f'{DATA}/DK_zarr/sequences_enformer.bed', sep='\t', header=None,
                       dtype={0: str})
    _sc = _bed.shape[1] - 1
    E_TB = _bed[_bed[_sc] == 'test'].reset_index(drop=True)
    _tr = {}
    for _ch, _g in _bed[_bed[_sc] == 'train'].groupby(0):
        _ce = (_g[1].astype(int).to_numpy() + _g[2].astype(int).to_numpy()) // 2
        _tr[str(_ch)] = (_ce - E_CTX // 2, _ce + E_CTX // 2)
    _bl = np.zeros(len(E_TB), bool)
    for _i in range(len(E_TB)):
        _ch = str(E_TB.iloc[_i, 0])
        _c, _d = int(E_TB.iloc[_i, 1]), int(E_TB.iloc[_i, 2])
        if _ch in _tr:
            _a, _b = _tr[_ch]
            if np.any(np.maximum(0, np.minimum(_b, _d) - np.maximum(_a, _c)) > E_OVF * (_d - _c)):
                _bl[_i] = True
    E_ALLOWED = ~_bl
    progress(f'[enformer] {E_ALLOWED.sum()} / {len(E_TB)} regions allowed')

    # ---------------------------------------------------------------- fig 2a
    # Per-region correlation against the GM12878 target, both metrics.
    # comparing_model4 §9, vectorised (rowwise_pearson over the whole array
    # rather than a per-region python loop -- same arithmetic).
    box = pd.DataFrame({'region': np.arange(len(E_TB)), 'allowed': E_ALLOWED})
    for arm, arr in E_P.items():
        box[f'r_sm_{arm}'] = smoothed_log_pearson(arr, E_TRUE, E_SMOOTH_SIGMA)
        box[f'r_raw_{arm}'] = rowwise_pearson(arr.astype(np.float64),
                                              E_TRUE.astype(np.float64))
    box.to_parquet(CACHE / 'enformer_boxplot.parquet', index=False)
    progress('[enformer] wrote enformer_boxplot.parquet')

    # ----------------------------------------------------------------- fig 4
    # x = how generic the region is, R(TRUE, AVG); y = R(prediction, TRUE).
    # `eligible` is comparing_model4 §12a: allowed regions with >= 3 true peak
    # bins, where a peak bin is TRUE at or above the global 95th percentile.
    THR_T = float(np.percentile(E_TRUE[E_ALLOWED], 95))
    eligible = np.flatnonzero(E_ALLOWED & ((E_TRUE >= THR_T).sum(axis=1) >= E_K_MIN))
    progress(f'[enformer] {len(eligible)} eligible regions '
             f'(>= {E_K_MIN} bins at or above the p95 target threshold {THR_T:.4f})')
    gen = pd.DataFrame({'region': eligible})
    gen['x_sm'] = smoothed_log_pearson(E_TRUE[eligible], E_AVG[eligible], E_SMOOTH_SIGMA)
    gen['x_raw'] = rowwise_pearson(E_TRUE[eligible].astype(np.float64),
                                   E_AVG[eligible].astype(np.float64))
    for arm, arr in E_P.items():
        gen[f'y_sm_{arm}'] = smoothed_log_pearson(arr[eligible], E_TRUE[eligible],
                                                  E_SMOOTH_SIGMA)
        gen[f'y_raw_{arm}'] = rowwise_pearson(arr[eligible].astype(np.float64),
                                              E_TRUE[eligible].astype(np.float64))
    gen.to_parquet(CACHE / 'enformer_generic.parquet', index=False)
    progress('[enformer] wrote enformer_generic.parquet')

    # ---------------------------------------------------------------- fig 3a
    # The stratum ladder. Four nested strata, each adding exactly one term.
    _txs = GeneAnnotation().df
    _ctr = (E_TB.iloc[:, 1].to_numpy().astype(np.int64) + E_CROP)[:, None] \
        + np.arange(E_T)[None, :] * E_BIN + E_BIN // 2
    _chr = E_TB.iloc[:, 0].to_numpy().astype(str)
    E_TSS = tss_mask(_chr, _ctr, _txs, strand=None, pad=E_TSSPAD)

    E_A = np.repeat(E_ALLOWED, E_T)
    E_RID = np.repeat(np.arange(len(E_TB)), E_T)
    _tv, _av, _ac, _ts = E_TRUE.ravel(), E_AVG.ravel(), E_ATAC.ravel(), E_TSS.ravel()
    E_OPEN = float(np.percentile(_ac[E_A], 95))
    _hi_avg = _av >= np.percentile(_av[E_A & _ts], 90)
    _lo_avg = _av < np.percentile(_av[E_A & _ts], 50)
    _keep = np.zeros_like(E_A)
    _keep[::E_DECIM] = True

    R0_POS = E_A & (_tv >= E_ACTIVE) & _keep
    R0_NEG = E_A & (_tv < E_SILENT) & _keep
    R1_POS = E_A & _ts & (_tv >= E_ACTIVE)
    R1_NEG = E_A & _ts & (_tv < E_SILENT)
    R2_POS = R1_POS & _lo_avg
    R2_NEG = R1_NEG & _hi_avg
    R3_POS = R2_POS & (_ac >= E_OPEN)
    R3_NEG = R2_NEG & (_ac < E_OPEN)
    E_RUNGS = [
        make_rung('R0 · all bins', 'any allowed bin', R0_POS, R0_NEG, E_RID),
        make_rung('R1 · promoters', 'protein-coding TSS only', R1_POS, R1_NEG, E_RID),
        make_rung('R2 · + disagreement', 'cross-CT average points the wrong way',
                  R2_POS, R2_NEG, E_RID),
        make_rung('R3 · + accessibility', 'ATAC-confirmed', R3_POS, R3_NEG, E_RID),
    ]

    arms = [(arm, lambda a=arm: E_P[a].ravel()) for arm in E_P]
    arms.append(('null_avg', lambda: E_AVG.ravel()))
    arms.append(('acc_atac', lambda: E_ATAC.ravel()))
    lad = score_ladder_to_frame(arms, E_RUNGS, '[enformer]')
    # ATAC defines the R3 classes, so its 1.000 there restates the definition.
    lad['by_construction'] = (lad.arm == 'acc_atac') & (lad.rung_i == 3)
    assert float(lad[lad.by_construction].auroc.iloc[0]) == 1.0, \
        'ATAC does not perfectly separate the R3 classes -- it should, since it defines them'

    # --- gate: rung 3 must reproduce comparing_model4 §20, rung 2 its §20b ---
    PUB_R3 = {'mamba_1ct': 0.779, 'mamba_nopre': 0.727, 'mamba_8ct': 0.568,
              'enformer_pre': 0.484, 'enformer_n1_1536': 0.425, 'alphagenome': 0.408,
              'enformer_n1': 0.355, 'enformer_n16': 0.319, 'enformer_n42': 0.267,
              'null_avg': 0.000}
    PUB_R2 = {'mamba_1ct': 0.538, 'mamba_nopre': 0.494, 'enformer_pre': 0.355,
              'mamba_8ct': 0.337, 'enformer_n1_1536': 0.295, 'enformer_n1': 0.236,
              'alphagenome': 0.221, 'enformer_n16': 0.209, 'enformer_n42': 0.166}
    for rung_i, pub, src in ((3, PUB_R3, 'comparing_model4 §20'),
                             (2, PUB_R2, 'comparing_model4 §20b')):
        got = lad[lad.rung_i == rung_i].set_index('arm').auroc
        bad = {k: (v, float(got[k])) for k, v in pub.items() if abs(got[k] - v) > 2e-3}
        assert not bad, f'GATE FAILED [{src}]: {bad}'
        progress(f'[enformer] gate: rung {rung_i} reproduces {src} for all {len(pub)} arms')

    lad.to_parquet(CACHE / 'ladder_enformer.parquet', index=False)
    (CACHE / 'rungs_enformer.json').write_text(json.dumps(rungs_to_json(E_RUNGS), indent=2))
    mark_done('enformer', n_regions=int(len(E_TB)), n_allowed=int(E_ALLOWED.sum()),
              n_eligible=int(len(eligible)), arms=sorted(E_P), open_threshold=E_OPEN)
    progress('[enformer] stage complete')


def score_ladder_to_frame(arms, rungs, tag):
    """arms: list of (arm_id, loader). One arm resident at a time."""
    rows = []
    for arm, loader in arms:
        pv = loader()
        for r_i, rung in enumerate(rungs):
            a0, lo, hi = score_rung(pv, rung, seed=BOOT_SEED + 1000 * r_i)
            rows.append(dict(arm=arm, rung=rung['name'], rung_i=r_i,
                             auroc=a0, lo=lo, hi=hi))
        del pv
        progress(f'{tag} scored {arm}')
    return pd.DataFrame(rows)


# ============================================================================
# STAGE: Borzoi fold3 grid (32 bp, strand-separate). Figures 1, 2b and 3b.
# Loading and gating follow comparing_model5 §1 and comparing_model6 §2.
# ============================================================================
def build_borzoi():
    progress('[borzoi] loading target')
    _g = zarr.open(f'{BORZOI}/GM12878CAGE.zarr', mode='r')['fold3']
    assert _g.shape == (N_REG, N_BIN, 2), f'unexpected target shape {_g.shape}'
    _gt = np.asarray(_g[:], dtype=np.float32)
    TRUE = {'plus': np.ascontiguousarray(_gt[:, :, 0]),
            'minus': np.ascontiguousarray(_gt[:, :, 1])}
    B_TV = np.concatenate([TRUE['plus'].ravel(), TRUE['minus'].ravel()])
    del _gt

    # --- cross-cell-type averages, plain and leave-GM12878-out ---------------
    _a = np.load(f'{BORZOI}/borzoi_fold3_CAGE_avg.npz')
    AVG = {s: _a[f'avg_{s}'] for s in STRANDS}
    B_AV = np.concatenate([_a['avg_plus_loo'].ravel(), _a['avg_minus_loo'].ravel()])
    _meta = json.loads(str(_a['meta']))
    assert _meta['gm_slot'] == GM_SLOT
    assert _a['cols_plus'][GM_SLOT] == 870 and _a['cols_minus'][GM_SLOT] == 871
    N_CT = _meta['n_cell_types']
    del _a

    # --- blacklist -----------------------------------------------------------
    bed = pd.read_csv(f'{DATA}/DK_zarr/sequences_borzoi_fold3-4.bed', sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'split', 'fold', 'idx'])
    assert (bed.end - bed.start).nunique() == 1 and (bed.end - bed.start).iloc[0] == TGT_SPAN
    test_bed = bed[bed.fold == 'fold3'].reset_index(drop=True)
    assert len(test_bed) == N_REG, f'{len(test_bed)} fold3 rows, expected {N_REG}'
    train_bed = bed[~bed.fold.isin(['fold3', 'fold4'])]
    PAD = (SEQ_LEN - TGT_SPAN) // 2
    blacklisted = np.zeros(N_REG, dtype=bool)
    label_overlap = np.zeros(N_REG, dtype=bool)
    for chrom, g in train_bed.groupby('chrom'):
        ts, te = g.start.to_numpy(), g.end.to_numpy()
        for i in np.flatnonzero((test_bed.chrom == chrom).to_numpy()):
            c, d = int(test_bed.start[i]), int(test_bed.end[i])
            if np.maximum(0, np.minimum(te, d + PAD) - np.maximum(ts, c - PAD)).max() \
                    > 0.20 * TGT_SPAN:
                blacklisted[i] = True
            if np.maximum(0, np.minimum(te, d) - np.maximum(ts, c)).max() > 0:
                label_overlap[i] = True
    allowed = ~blacklisted
    assert label_overlap.sum() == 0, (
        f'GATE FAILED: {label_overlap.sum()} fold3 TARGET spans overlap a training target span')
    progress(f'[borzoi] {allowed.sum()} / {N_REG} regions allowed, no label leakage')

    # --- DNase, with the row-order and tracking checks -----------------------
    B_DN = np.load(f'{BORZOI}/GM12878DNase.npz')['test'][:, :, 0]
    assert B_DN.shape == (N_REG, N_BIN), f'DNase is {B_DN.shape}, expected {(N_REG, N_BIN)}'
    _wi = pd.read_csv(f'{BORZOI}/GM12878_borzoi_window_index.tsv', sep='\t', dtype={'chrom': str})
    _wt = _wi[_wi.npz_split == 'test'].reset_index(drop=True)
    assert len(_wt) == N_REG \
        and (_wt.start.to_numpy() == test_bed.start.to_numpy()).all() \
        and (_wt.chrom.astype(str).to_numpy() == test_bed.chrom.astype(str).to_numpy()).all(), \
        'GATE FAILED: the DNase npz row order is not the bed fold3 row order'
    _r_ok = pearsonr(np.log1p(B_DN[:1200]).ravel(), np.log1p(TRUE['plus'][:1200]).ravel())[0]
    _perm = np.random.default_rng(0).permutation(1200)
    _r_ctl = pearsonr(np.log1p(B_DN[_perm]).ravel(), np.log1p(TRUE['plus'][:1200]).ravel())[0]
    assert _r_ok > 0.15 and _r_ok > 5 * abs(_r_ctl), 'DNase does not track CAGE'
    progress(f'[borzoi] DNase gate passed: r={_r_ok:+.3f}, shuffled control r={_r_ctl:+.3f}')

    # --- strand-matched protein-coding TSS mask on the 32 bp grid ------------
    _pc = GeneAnnotation(gene_types=('protein_coding',)).df
    _ctr = (test_bed.start.to_numpy().astype(np.int64)[:, None]
            + np.arange(N_BIN)[None, :] * BIN_BP + BIN_BP // 2)
    _chr = test_bed.chrom.to_numpy().astype(str)
    B_TS = np.concatenate([tss_mask(_chr, _ctr, _pc, strand=st, pad=B_TSSPAD).ravel()
                           for st in ('+', '-')])

    # --- the ladder's strata -------------------------------------------------
    B_A = np.concatenate([np.repeat(allowed, N_BIN)] * 2)
    B_RID = np.concatenate([np.repeat(np.arange(N_REG), N_BIN)] * 2)
    B_DNF = np.concatenate([B_DN.ravel()] * 2)          # DNase is unstranded
    B_OPEN = float(np.percentile(B_DNF[B_A], 95))
    _hi_avg = B_AV >= np.percentile(B_AV[B_A & B_TS], 90)
    _lo_avg = B_AV < np.percentile(B_AV[B_A & B_TS], 50)
    _keep = np.zeros_like(B_A)
    _keep[::B_DECIM] = True

    B_RUNGS = [
        make_rung('R0 · all bins', f'any allowed bin, 1-in-{B_DECIM} subsample',
                  B_A & (B_TV >= B_ACTIVE) & _keep, B_A & (B_TV < B_SILENT) & _keep, B_RID),
        make_rung('R1 · promoters', 'protein-coding TSS only',
                  B_A & B_TS & (B_TV >= B_ACTIVE), B_A & B_TS & (B_TV < B_SILENT), B_RID),
        make_rung('R2 · + disagreement', 'cross-CT average points the wrong way',
                  B_A & B_TS & (B_TV >= B_ACTIVE) & _lo_avg,
                  B_A & B_TS & (B_TV < B_SILENT) & _hi_avg, B_RID),
        make_rung('R3 · + accessibility', 'DNase-confirmed',
                  B_A & B_TS & (B_TV >= B_ACTIVE) & _lo_avg & (B_DNF >= B_OPEN),
                  B_A & B_TS & (B_TV < B_SILENT) & _hi_avg & (B_DNF < B_OPEN), B_RID),
    ]

    # --- one pass over the arms: each is loaded once and serves all three -----
    # figures before being released. Peak memory is one arm, not the panel.
    B_ARMS_DEF = [
        ('borzoi_pre', f'{BORZOI}/model_outputs_fold3_gm12878CAGE.npy', 'hw2'),
        ('borzoi_lora', f'{B_MO}/borzoi_finetuned_gm12878_lorav2.npy', 'hw3'),
        ('borzoi_trunk', f'{B_MO}/borzoi_finetuned_gm12878_trunk.npy', 'hw3'),
        ('borzoi_scratch', f'{B_MO}/borzoi_finetuned_gm12878_scratch.npy', 'hw3'),
        ('cad_ep4', f'{JP}/borzoi_gm12878_cage_ep4_outputs.npy', 'h1w2'),
        ('alphagenome', f'{JP}/alphagenome_cage_gm12878_borzoi_fold3_32bp.npy', 'hw2'),
    ]
    missing = [p for _, p, _ in B_ARMS_DEF if not os.path.exists(p)]
    assert not missing, 'prediction files not on disk:\n  ' + '\n  '.join(missing)

    box = pd.DataFrame({
        'region': np.tile(np.arange(N_REG), 2),
        'strand': np.repeat(['plus', 'minus'], N_REG),
        'allowed': np.concatenate([allowed, allowed]),
    })
    locus = {}
    lad_rows = []
    for src_key, path, layout in B_ARMS_DEF:
        arm = A.to_id(A.BORZOI_GRID, src_key)
        a = np.load(path, mmap_mode='r')
        if layout == 'h1w2':                 # caduceus writes (N, 1, T, 2)
            a = a[:, 0]
        p = {'plus': np.ascontiguousarray(a[:, :, 0], dtype=np.float32),
             'minus': np.ascontiguousarray(a[:, :, 1], dtype=np.float32)}
        assert not np.array_equal(p['plus'], TRUE['plus']), f'GATE FAILED: {arm} IS the target'

        box[f'r_sm_{arm}'] = np.concatenate(
            [smoothed_log_pearson(p[s], TRUE[s], B_SMOOTH_SIGMA) for s in STRANDS])
        box[f'r_raw_{arm}'] = np.concatenate(
            [rowwise_pearson(p[s].astype(np.float64), TRUE[s].astype(np.float64))
             for s in STRANDS])
        locus[f'pred_{arm}'] = p[LOCUS_STRAND][LOCUS_REGION].copy()

        pv = np.concatenate([p['plus'].ravel(), p['minus'].ravel()])
        for r_i, rung in enumerate(B_RUNGS):
            a0, lo, hi = score_rung(pv, rung, seed=BOOT_SEED + 1000 * r_i)
            lad_rows.append(dict(arm=arm, rung=rung['name'], rung_i=r_i,
                                 auroc=a0, lo=lo, hi=hi))
        del a, p, pv
        progress(f'[borzoi] scored {arm}')

    for arm, pv in (('null_avg', B_AV), ('acc_dnase', B_DNF)):
        for r_i, rung in enumerate(B_RUNGS):
            a0, lo, hi = score_rung(pv, rung, seed=BOOT_SEED + 1000 * r_i)
            lad_rows.append(dict(arm=arm, rung=rung['name'], rung_i=r_i,
                                 auroc=a0, lo=lo, hi=hi))
        progress(f'[borzoi] scored {arm}')

    box.to_parquet(CACHE / 'borzoi_boxplot.parquet', index=False)
    progress('[borzoi] wrote borzoi_boxplot.parquet')

    lad = pd.DataFrame(lad_rows)
    lad['by_construction'] = (lad.arm == 'acc_dnase') & (lad.rung_i == 3)
    assert float(lad[lad.by_construction].auroc.iloc[0]) == 1.0, \
        'DNase does not perfectly separate the R3 classes -- it should, since it defines them'

    # --- gate: rung 3 must reproduce comparing_model5 §12, rung 2 its §12b ---
    PUB_R3 = {'mamba_1ct_borzoi': 0.626, 'borzoi_trunk': 0.462,
              'borzoi_scratch': 0.436, 'borzoi_lora': 0.395, 'alphagenome': 0.348,
              'borzoi_pre': 0.177, 'null_avg': 0.000}
    PUB_R2 = {'mamba_1ct_borzoi': 0.354, 'borzoi_trunk': 0.300,
              'borzoi_scratch': 0.238, 'borzoi_lora': 0.231, 'alphagenome': 0.179,
              'borzoi_pre': 0.092}
    for rung_i, pub, src in ((3, PUB_R3, 'comparing_model5 §12'),
                             (2, PUB_R2, 'comparing_model5 §12b')):
        got = lad[lad.rung_i == rung_i].set_index('arm').auroc
        bad = {k: (v, float(got[k])) for k, v in pub.items() if abs(got[k] - v) > 2e-3}
        assert not bad, f'GATE FAILED [{src}]: {bad}'
        progress(f'[borzoi] gate: rung {rung_i} reproduces {src} for all {len(pub)} arms')

    lad.to_parquet(CACHE / 'ladder_borzoi.parquet', index=False)
    (CACHE / 'rungs_borzoi.json').write_text(json.dumps(rungs_to_json(B_RUNGS), indent=2))

    # ------------------------------------------------------------------ fig 1
    # Everything the example-locus panel draws, for EVERY arm -- so changing
    # which arms appear is an analysis-side edit, not a rebuild.
    chrom = str(test_bed.chrom[LOCUS_REGION])
    start = int(test_bed.start[LOCUS_REGION])
    end = start + N_BIN * BIN_BP
    locus['true'] = TRUE[LOCUS_STRAND][LOCUS_REGION].copy()
    locus['avg'] = AVG[LOCUS_STRAND][LOCUS_REGION].copy()
    locus['dnase'] = B_DN[LOCUS_REGION].copy()
    np.savez_compressed(CACHE / 'locus.npz', **locus)

    # gene names nearest the strongest observed bins, over every GENCODE biotype
    ann_all = GeneAnnotation(gene_types=None)
    recs = ann_all.query(chrom, start, end, collapse=True)
    peaks = start + (np.argsort(locus['true'])[-5:] + 0.5) * BIN_BP
    scored = sorted((float(np.min(np.abs(peaks - (r['tx_start'] if r['strand'] == '+'
                                                  else r['tx_end'])))), r['gene_name'])
                    for r in recs)
    genes = list(dict.fromkeys(g for _, g in scored))
    (CACHE / 'locus.json').write_text(json.dumps(dict(
        region=LOCUS_REGION, strand=LOCUS_STRAND, chrom=chrom, start=start, end=end,
        bin_bp=BIN_BP, n_bin=N_BIN, n_cell_types=int(N_CT),
        genes=genes, n_genes_in_window=len(recs),
        r_sm={arm[5:]: float(box.loc[(box.region == LOCUS_REGION)
                                     & (box.strand == LOCUS_STRAND), arm].iloc[0])
              for arm in box.columns if arm.startswith('r_sm_')},
    ), indent=2))
    progress(f'[borzoi] wrote locus {chrom}:{start:,}-{end:,} '
             f'({LOCUS_STRAND} strand), {len(genes)} genes in window')

    mark_done('borzoi', n_regions=N_REG, n_allowed=int(allowed.sum()),
              locus_region=LOCUS_REGION, locus_strand=LOCUS_STRAND,
              open_threshold=B_OPEN, n_cell_types=int(N_CT))
    progress('[borzoi] stage complete')


# ============================================================================
# STAGE: the QTL benchmarks. Figures 5 and 6.
# eQTL loading is IGVF_meeting_figures cell 14, unchanged apart from arm ids.
# dsQTL reads the table dsQTL_benchmark8 already built; nothing is recomputed.
# ============================================================================
def abs_logfc(win):
    """|log2 mean(ref) / mean(alt)| over the scoring window; 0 where unscored."""
    r, a = win[:, :, 0], win[:, :, 1]
    unscored = (r.sum(1) == 0) & (a.sum(1) == 0)
    out = np.zeros(len(r))
    v = ~unscored
    out[v] = np.log2(r[v].mean(1) / a[v].mean(1))
    return np.abs(out), ~unscored


def build_qtl():
    import pickle
    progress('[qtl] eQTL: loading nine arms')
    N, CENTER_BP = 1331, 2048
    q = pd.read_csv(QTL + 'LCLs.txt', sep=' ', header=None)
    q.columns = ['label', 'qtl_idx', 'gene_idx', 'chrom', 'gene_start', 'gene_end',
                 'strand', 'qtl_loc', 'ref', 'alt', 'sign_target']
    labels = q['label'].to_numpy()
    genes = sorted(pickle.load(open(QTL + 'genes.pickle', 'rb')).keys())
    t = np.loadtxt(QTL + 'ensemblTSS.txt', dtype='str')
    tss = {t[i, 0]: int(t[i, 1]) for i in range(t.shape[0])}
    dist = np.array([abs(tss[genes[q.iloc[i]['gene_idx']]] - (q.iloc[i]['qtl_loc'] - 1))
                     for i in range(N)])

    EQTL_SPEC = [
        ('cis-Mamba (1 cell type)', 'nomlm_maskonly.npy', 128),
        ('cis-Mamba, no pretraining', 'nopretrain.npy', 128),
        ('cis-Mamba (8 cell types)', 'immune.npy', 128),
        ('Enformer (pretrained)', 'enformer_eqtl.npy', 128),
        ('Borzoi (pretrained)', 'borzoi_eqtl.npy', 32),
        ('Borzoi LoRA (GM12878)', 'borzoi_eqtl_lorav2.npy', 32),
        ('Borzoi full FT (GM12878)', 'borzoi_eqtl_trunk.npy', 32),
        ('Borzoi from scratch (GM12878)', 'borzoi_eqtl_scratch.npy', 32),
        ('AlphaGenome', 'alphagenome_eqtl.zarr', 1),
    ]
    S, SC = {}, {}
    for src_label, f, bp in EQTL_SPEC:
        arm = A.to_id(A.EQTL, src_label)
        if f.endswith('.zarr'):
            full = np.asarray(zarr.open(QTL + f, mode='r')[:], dtype=np.float64)
        else:
            full = np.asarray(np.load(QTL + f, mmap_mode='r'), dtype=np.float64)
        assert full.shape[0] == N and full.shape[2] == 2, (arm, full.shape)
        nb = CENTER_BP // bp
        assert nb * bp == CENTER_BP, f'{arm}: {bp} bp bins do not tile {CENTER_BP} bp'
        cs = full.shape[1] // 2 - nb // 2
        S[arm], SC[arm] = abs_logfc(full[:, cs:cs + nb, :])
    common = np.logical_and.reduce(list(SC.values()))
    assert common.sum() > 0
    progress(f'[qtl] eQTL common support: {common.sum()} / {N} variants scorable by every arm')

    # Distance bins are the panel's x-facets: proximal / flanking / distal.
    TH = [1_000, 10_000, 524_288 // 2]
    BINS = [('<1 kb\n(proximal)', dist < TH[0]),
            ('1-10 kb\n(flanking)', (dist >= TH[0]) & (dist < TH[1])),
            ('10-262 kb\n(distal)', (dist >= TH[1]) & (dist < TH[2]))]
    rows = []
    for arm in S:
        r = {'arm': arm}
        for nm, m in BINS:
            mm = m & common
            r[nm] = average_precision_score(labels[mm], S[arm][mm])
        rows.append(r)
    pd.DataFrame(rows).to_parquet(CACHE / 'eqtl_paneld.parquet', index=False)
    (CACHE / 'eqtl_paneld.json').write_text(json.dumps(dict(
        binnames=[b[0] for b in BINS],
        chance=[float(labels[m & common].mean()) for _, m in BINS],
        n=[int((m & common).sum()) for _, m in BINS],
        n_common=int(common.sum()), n_total=N,
    ), indent=2))
    progress('[qtl] wrote eqtl_paneld.parquet')

    # --- dsQTL: reuse the table dsQTL_benchmark8 already built ---------------
    src = DSQTL8_CACHE / 'stratified_metrics.tsv'
    assert src.exists(), (
        f'{src} is missing -- run the dsQTL_benchmark8 pipeline first; this stage '
        f'deliberately does not recompute it')
    tab = pd.read_csv(src, sep='\t')
    tab['arm'] = [A.to_id(A.DSQTL, m) for m in tab.model]
    tab.to_parquet(CACHE / 'dsqtl_stratified.parquet', index=False)
    med = {}
    for definition in tab.definition.unique():
        peaks = pd.read_csv(DSQTL8_CACHE / f'gm12878_{definition}.tsv.gz', sep='\t',
                            usecols=['median_breadth'])
        med[definition] = int(peaks.median_breadth.iloc[0])
    (CACHE / 'dsqtl_stratified.json').write_text(json.dumps(dict(median_breadth=med), indent=2))
    progress(f'[qtl] wrote dsqtl_stratified.parquet ({len(tab)} rows)')

    mark_done('qtl', eqtl_common=int(common.sum()), dsqtl_rows=int(len(tab)))
    progress('[qtl] stage complete')


STAGES = {'enformer': build_enformer, 'borzoi': build_borzoi, 'qtl': build_qtl}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = list(STAGES) if which == 'all' else [which]
    assert all(s in STAGES for s in todo), f'unknown stage; pick from {list(STAGES)} or all'
    for stage in todo:
        t0 = time.time()
        STAGES[stage]()
        progress(f'[{stage}] {time.time() - t0:.0f}s')
