"""Panel 3's Enformer CAGE row, redone for CD4 T and NK cells.

    cd4_nk_panel3_enformer_cage.png   rung schematic, then rung AUROCs, CD4 row above NK row

Truth is the observed Enformer CAGE track the CD4/NK cis-Mamba runs were fine-tuned
on (outputs/cage_finetuning/20261003_cd4_nk). Rungs are built exactly as
igvf_meeting_figures2_pipeline.build_enformer builds them for GM12878, with the
cell's leave-one-out Enformer CAGE average and, for R3, the DNase track the
cis-Mamba runs take as input (128 bp mean over each output window).

    python make_cd4_nk_panel.py compute   # slow: one pass over enformer.npy and labels.zarr
    python make_cd4_nk_panel.py plot
    python make_cd4_nk_panel.py           # both

`compute` first rebuilds the GM12878 row with the same code and checks it against
cache/panel3_rung_scores.csv before scoring CD4 and NK.
"""
from pathlib import Path
import dataclasses
import json
import sys

ROOT = Path('/data1/lesliec/sarthak/caduceus')
OUT = Path(__file__).resolve().parent
CACHE = OUT / 'cache'
sys.path[:0] = [str(ROOT / 'notebooks/figures'), str(ROOT)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import igvf_arms as A
import igvf_meeting_figures2_pipeline as P
from make_figures import draw_schematic
from evals.plotting_utils import GeneAnnotation

DATA = '/data1/lesliec/sarthak/data'
MO = f'{DATA}/joint_playground/model_out'
RUN = ROOT / 'outputs/cage_finetuning/20261003_cd4_nk'
EVAL = 'test_evaluation_20261012'
DNASE = f'{DATA}/DK_zarr/zarr_arrays/cell_type_arrays'
SHARED = '/data1/lesliec/sarthak/EpiBERT/data/rampage_ladder_1840'
SOURCE_TRACKS = CACHE / 'cd4_nk_source_tracks.npz'

# Enformer target index; AlphaGenome column (= its track_index)
CELLS = {
    'GM12878': dict(track=5110, ag=156),
    'CD4': dict(track=4759, ag=7, dnase='CD4T_DNase.npz'),   # AG: naive CD4 T LQhCAGE
    'NK': dict(track=4764, ag=38, dnase='NK_DNase.npz'),
}
ROW_ORDER = ('CD4', 'NK')


# =============================================================================
# compute
# =============================================================================
def test_regions():
    """Test BED rows, and the 1,840 shared regions panel 3 uses (comparing_model7 §5a:
    Enformer test windows whose 524 kb context avoids EpiBERT's training intervals)."""
    bed = pd.read_csv(f'{DATA}/DK_zarr/sequences_enformer.bed', sep='\t', header=None, dtype={0: str})
    tb = bed[bed[bed.shape[1] - 1] == 'test'].reset_index(drop=True)
    with np.load(f'{SHARED}/aligned_rampage.npz') as z:
        rows = z['enformer_rows'].copy()
    shared = pd.read_csv(f'{SHARED}/regions.tsv', sep='\t')
    assert len(rows) == 1840 and shared.enformer_row.tolist() == rows.tolist()
    assert np.array_equal(tb.iloc[rows, :3].to_numpy(), shared[['chrom', 'start', 'end']].to_numpy())
    return tb, rows, shared


def source_tracks():
    """Observed labels and pretrained Enformer predictions for the three tracks.

    Both sources are (1937, 896, 5313), so one column means reading the whole
    array; this reads each once, in blocks, and keeps the three columns.
    """
    if SOURCE_TRACKS.exists():
        with np.load(SOURCE_TRACKS) as z:
            return {k: z[k] for k in z.files}
    cols = [CELLS[c]['track'] for c in CELLS]
    labels = zarr.open(f'{DATA}/enformer/data/labels.zarr', mode='r')['test']
    enformer = np.load(f'{DATA}/enformer/data/model_out/enformer.npy', mmap_mode='r')
    n = labels.shape[0]
    assert enformer.shape == labels.shape
    lab = np.empty((n, P.E_T, len(cols)), np.float32)
    enf = np.empty((n, P.E_T, len(cols)), np.float32)
    for i in range(0, n, 32):
        lab[i:i+32] = np.asarray(labels[i:i+32])[:, :, cols]
        enf[i:i+32] = np.asarray(enformer[i:i+32])[:, :, cols]
        print(f'  source tracks {min(i+32, n)}/{n}', flush=True)
    out = {}
    for j, cell in enumerate(CELLS):
        out[f'label_{cell}'] = lab[:, :, j]
        out[f'enformer_{cell}'] = enf[:, :, j]
    np.savez(SOURCE_TRACKS, **out)
    return out


def leave_one_out_avg(true):
    """build_enformer's cross-cell-type CAGE average, with this cell's track removed."""
    n_cage = 5313 - P.E_CAGE_START
    avg_all = np.load(f'{DATA}/enformer/data/avg_cage_test_all.npy')
    return np.clip((avg_all * n_cage - true) / (n_cage - 1), 0, None)


def binned_dnase(cell, regions):
    """Measured DNase (the cis-Mamba input) averaged into the 896 output bins."""
    out = np.empty((len(regions), P.E_T), np.float32)
    with np.load(f'{DNASE}/{CELLS[cell]["dnase"]}') as z:
        for ch, rows in regions.groupby('chrom').groups.items():
            track = z[ch][0]
            for i in rows:
                s, e = int(regions.output_start[i]), int(regions.output_end[i])
                assert e - s == P.E_T * P.E_BIN and e <= track.size
                out[i] = track[s:e].astype(np.float32).reshape(P.E_T, P.E_BIN).mean(1)
    assert np.isfinite(out).all(), f'{cell} DNase has non-finite bins'
    return out


def target_gate(name, targets, true):
    t = np.asarray(targets, np.float32)
    t = t[:, :, 0] if t.ndim == 3 else t
    assert np.allclose(t, true, rtol=1e-5, atol=1e-6), f'GATE FAILED [{name}]: target is not this cell\'s truth'


def build_rungs(true, avg, acc, tss, accessibility):
    """The four nested rungs of comparing_model7 §5c (CAGE), on the 1,840 shared regions."""
    rid = np.repeat(np.arange(len(true)), P.E_T)
    tv, av, ac, ts = true.ravel(), avg.ravel(), acc.ravel(), tss.ravel()
    assert all(np.isfinite(v).all() for v in (tv, av, ac))
    open_thr = float(np.percentile(ac, 95))
    median, high = np.percentile(av[ts], [50, 90])
    assert high > median > 0
    hi_avg, lo_avg = av >= high, av < median
    r0_pos, r0_neg = tv >= P.E_ACTIVE, tv < P.E_SILENT
    r1_pos, r1_neg = r0_pos & ts, r0_neg & ts
    r2_pos, r2_neg = r1_pos & lo_avg, r1_neg & hi_avg
    r3_pos, r3_neg = r2_pos & (ac >= open_thr), r2_neg & (ac < open_thr)
    return [
        P.make_rung('R0 · all bins', 'any allowed bin', r0_pos, r0_neg, rid),
        P.make_rung('R1 · promoters', 'protein-coding TSS only', r1_pos, r1_neg, rid),
        P.make_rung('R2 · + disagreement', 'cross-CT average points the wrong way', r2_pos, r2_neg, rid),
        P.make_rung('R3 · + accessibility', f'{accessibility}-confirmed', r3_pos, r3_neg, rid),
    ]


def score(arms, rungs, target, acc_arm):
    rows = []
    for arm, pred in arms.items():
        pv = np.ascontiguousarray(pred, np.float32).ravel()
        for r_i, rung in enumerate(rungs):
            a0, lo, hi = P.score_rung(pv, rung, seed=P.BOOT_SEED + 1000 * r_i)
            rows.append(dict(target=target, arm=arm, rung_i=r_i, rung=rung['name'], auroc=a0, lo=lo, hi=hi))
        print(f'  [{target}] scored {arm}', flush=True)
    df = pd.DataFrame(rows)
    df['by_construction'] = (df.arm == acc_arm) & (df.rung_i == 3)
    assert (df[df.by_construction].auroc == 1.0).all(), f'{acc_arm} should separate R3 perfectly'
    assert (df[(df.arm == 'null_avg') & (df.rung_i >= 2)].auroc == 0.0).all(), 'average should be 0 on R2/R3'
    return df


def compute():
    tb, rows, shared = test_regions()
    centers = shared.output_start.to_numpy()[:, None] + (np.arange(P.E_T) + .5)[None, :] * P.E_BIN
    tss = P.tss_mask(shared.chrom.to_numpy(), centers, GeneAnnotation(cache_dir=f'{DATA}/DE_danwei').df,
                     strand=None, pad=P.E_TSSPAD)

    def sub(arr):
        return np.asarray(arr[rows], np.float32)
    ag_regions = pd.read_csv(f'{MO}/alphagenome_cage_test_regions.csv')
    assert (ag_regions.chrom.to_numpy() == tb[0].to_numpy()).all() \
        and (ag_regions.start.to_numpy() == tb[1].to_numpy()).all(), 'AlphaGenome region order'
    ag_tracks = pd.read_csv(f'{MO}/alphagenome_cage_test_tracks.csv')
    ag = np.load(f'{MO}/alphagenome_cage_test_128bp.npy', mmap_mode='r')
    src = source_tracks()

    def ag_track(cell):
        col = CELLS[cell]['ag']
        assert int(ag_tracks.track_index.iloc[col]) == col
        print(f'  [{cell}] AlphaGenome column {col}: {ag_tracks.biosample_name.iloc[col]} '
              f'({ag_tracks["Assay title"].iloc[col]})', flush=True)
        return np.asarray(ag[:, :, col], np.float32)

    # --- gate: the GM12878 row reproduces the cached panel 3 numbers ---------
    true = src['label_GM12878']
    gm_ref = np.load(f'{MO}/GM12878_base_targets_conv.npy').astype(np.float32)
    assert np.array_equal(true, gm_ref)
    enf_ref = np.load(f'{MO}/enformer_test_track5110.npy').astype(np.float32)
    print(f'  enformer.npy track 5110 vs enformer_test_track5110.npy: max |diff| '
          f'{np.abs(src["enformer_GM12878"] - enf_ref).max():.3g}', flush=True)
    mamba_gm = np.load(f'{MO}/GM12878_no_mlm_predictions.npy')
    target_gate('GM12878 mamba_1ct', np.load(f'{MO}/GM12878_no_mlm_targets.npy'), true)
    atac = np.load(f'{DATA}/enformer/data/GM12878_ATAC_WW_denoised.npz')['test'][:, :, 0]
    avg = leave_one_out_avg(true)
    rungs = build_rungs(sub(true), sub(avg), sub(atac), tss, 'ATAC')
    ref_rungs = json.loads((CACHE / 'panel3_rungs.json').read_text())['enformer_cage']
    for got, ref in zip(P.rungs_to_json(rungs), ref_rungs):
        assert (got['n_pos'], got['n_neg']) == (ref['n_pos'], ref['n_neg']), (got, ref)
    gm = {'null_avg': avg, 'acc_atac': atac, 'mamba_1ct': mamba_gm[..., 0] if mamba_gm.ndim == 3 else mamba_gm,
          'enformer_pre': src['enformer_GM12878'], 'alphagenome': ag_track('GM12878')}
    gm = score({arm: sub(pred) for arm, pred in gm.items()}, rungs, 'GM12878', 'acc_atac')
    ref = pd.read_csv(CACHE / 'panel3_rung_scores.csv')
    ref = ref[ref.target == 'enformer_cage'].set_index(['arm', 'rung_i'])
    merged = gm.set_index(['arm', 'rung_i']).join(ref[['auroc', 'lo', 'hi']], rsuffix='_ref')
    diff = (merged[['auroc', 'lo', 'hi']].to_numpy() - merged[['auroc_ref', 'lo_ref', 'hi_ref']].to_numpy())
    print(merged[['auroc', 'auroc_ref']].round(4).to_string(), flush=True)
    assert np.abs(diff).max() < 1e-6, f'GATE FAILED: GM12878 rebuild differs from panel 3 by {np.abs(diff).max():.3g}'
    print('  gate: GM12878 rebuild reproduces cached panel 3 (AUROC and CI)', flush=True)

    # --- CD4 and NK ----------------------------------------------------------
    scores, rung_json, meta = [], {}, {}
    for cell in ROW_ORDER:
        d = RUN / cell / EVAL
        assert (d / 'SUCCESS').exists()
        regions = pd.read_csv(d / 'regions.tsv', sep='\t')
        assert (regions.region_index.to_numpy() == np.arange(len(tb))).all()
        assert (regions.chrom.to_numpy() == tb[0].to_numpy()).all()
        assert (regions.output_start.to_numpy() == tb[1].to_numpy() + P.E_CROP).all()
        true = src[f'label_{cell}']
        assert np.array_equal(np.load(d / 'expression_targets.npy')[:, 0, :, 0].astype(np.float32), true)
        avg = leave_one_out_avg(true)
        dnase = binned_dnase(cell, regions.iloc[rows].reset_index(drop=True))
        rungs = build_rungs(sub(true), sub(avg), dnase, tss, 'DNase')

        arms = {'null_avg': avg,
                'mamba_cage_ft': np.load(d / 'expression_outputs.npy')[:, 0, :, 0]}
        if cell == 'CD4':
            for arm, stem in (('mamba_8ct', 'immune_CAGE_CD4T'), ('mamba_nobcell', 'nobcell_CAGE_CD4T')):
                target_gate(f'{cell} {arm}', np.load(f'{MO}/{stem}_targets.npy'), true)
                arms[arm] = np.load(f'{MO}/{stem}_predictions.npy')[:, :, 0]
        arms['enformer_pre'] = src[f'enformer_{cell}']
        if cell == 'NK':
            for n in (16, 42):
                stem = f'{MO}/enformer_retrained_n{n}/enformer_retrained_n{n}_cage'
                tracks = pd.read_csv(f'{stem}_tracks.csv')
                row = tracks[tracks.target_index == CELLS[cell]['track']]
                assert len(row) == 1
                col = int(row.column.iloc[0])
                target_gate(f'{cell} enformer_n{n}', np.load(f'{stem}_targets.npy', mmap_mode='r')[:, :, col], true)
                arms[f'enformer_n{n}'] = np.load(f'{stem}_predictions.npy', mmap_mode='r')[:, :, col]
        arms['alphagenome'] = ag_track(cell)
        for arm, pred in arms.items():
            assert pred.shape == true.shape, (cell, arm, pred.shape)
        arms = {arm: sub(pred) for arm, pred in arms.items()}
        arms = {'null_avg': arms.pop('null_avg'), 'acc_dnase': dnase, **arms}

        scores.append(score(arms, rungs, cell, 'acc_dnase'))
        rung_json[cell] = P.rungs_to_json(rungs)
        meta[cell] = dict(track=CELLS[cell]['track'], ag_column=CELLS[cell]['ag'],
                          checkpoint=str(d), n_regions=len(rows))
        print(f'  [{cell}] ' + ' | '.join(f'{r["name"]}: {r["n_pos"]:,}/{r["n_neg"]:,}' for r in rung_json[cell]),
              flush=True)

    pd.concat(scores).to_csv(CACHE / 'cd4_nk_rung_scores.csv', index=False)
    (CACHE / 'cd4_nk_rungs.json').write_text(json.dumps(dict(rungs=rung_json, meta=meta), indent=2))
    print('wrote cache/cd4_nk_rung_scores.csv and cache/cd4_nk_rungs.json', flush=True)


# =============================================================================
# plot
# =============================================================================
TITLES = {'CD4': 'CD4+ T cell · Enformer CAGE ground truth (CNhs11955) · 128 bp',
          'NK': 'NK cell · Enformer CAGE ground truth (CNhs11957) · 128 bp'}
NOTES = {'CD4': '1,840 regions · leave-CD4-out Enformer CAGE average · CD4 T DNase (cis-Mamba input)',
         'NK': '1,840 regions · leave-NK-out Enformer CAGE average · NK DNase (cis-Mamba input)'}
LABELS = {
    'CD4': {'mamba_cage_ft': 'cis-Mamba (CD4 CAGE)', 'acc_dnase': 'CD4 T DNase (measured)',
            'alphagenome': 'AlphaGenome (naive CD4 T, LQhCAGE)'},
    'NK': {'mamba_cage_ft': 'cis-Mamba (NK CAGE)', 'acc_dnase': 'NK DNase (measured)'},
}


def register_arms():
    for arm in (A.Arm('mamba_cage_ft', 'cis-Mamba', 'ours', '#177A3B'),
                A.Arm('mamba_nobcell', 'cis-Mamba (immune, no B cell)', 'ours', '#63BF82')):
        A.BY_ID[arm.id] = arm


def panel():
    scores = pd.read_csv(CACHE / 'cd4_nk_rung_scores.csv')
    rungs = json.loads((CACHE / 'cd4_nk_rungs.json').read_text())['rungs']
    n_arms = {cell: scores[scores.target == cell].arm.nunique() for cell in ROW_ORDER}
    row_heights = {cell: 2.15 + .32*n_arms[cell] for cell in ROW_ORDER}
    height = 4.7 + sum(row_heights.values()) + .85
    fig = plt.figure(figsize=(26, height))
    left, right, wspace = .188, .989, .20
    width = (right-left) / (4 + 3*wspace)
    columns = [(left+i*width*(1+wspace), width) for i in range(4)]
    center = (left+right)/2
    top = height-.2
    draw_schematic(fig, columns, top, height)
    top -= 4.7
    for cell in ROW_ORDER:
        d = scores[scores.target == cell]
        references = ['null_avg', 'acc_dnase']
        models = d[(d.rung_i == 3) & ~d.arm.isin(references)].sort_values('auroc', ascending=False).arm.tolist()
        order = references + models
        labels = []
        for arm in order:
            label = LABELS[cell].get(arm, A.label(arm))
            if arm == 'null_avg':
                label = '[null] cross-cell-type average'
            labels.append(label)
        fig.text(center, (top-.10)/height, TITLES[cell], ha='center', va='top', fontsize=23, weight='bold')
        fig.text(center, (top-.53)/height, NOTES[cell], ha='center', va='top', fontsize=12.5, color='#555555')
        bar_height = .32*len(order)
        for i, (xleft, col_width) in enumerate(columns):
            ax = fig.add_axes([xleft, (top-1.55-bar_height)/height, col_width, bar_height/height])
            table = d[d.rung_i == i].set_index('arm').loc[order]
            y = np.arange(len(order))
            bars = ax.barh(y, table.auroc, color=[A.color(arm) for arm in order], height=.66,
                           xerr=np.vstack([np.maximum(0, table.auroc-table.lo), np.maximum(0, table.hi-table.auroc)]),
                           error_kw=dict(ecolor='#666666', lw=.9, capsize=2), zorder=3)
            for bar, flag in zip(bars, table.by_construction):
                if flag:
                    bar.set_hatch('///')
                    bar.set_edgecolor('white')
            for row_y, row in enumerate(table.itertuples()):
                ax.text(row.hi+.014, row_y, f'{row.auroc:.3f}', va='center', fontsize=12.5, color='#555555')
            if i < 2:
                ax.axvline(.5, ls='--', lw=1, color='#777777', zorder=4)
            ax.set_ylim(len(order)-.5, -.5)
            ax.set_xlim(0, 1.24)
            ax.set_xticks([0, .25, .5, .75, 1.])
            ax.set_yticks(y, labels if i == 0 else ['']*len(order), fontsize=14)
            if i == 0:
                for text, arm in zip(ax.get_yticklabels(), order):
                    if arm.startswith('mamba_'):
                        text.set_weight('bold')
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=12)
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', color='#DDDDDD', lw=.65, ls='--')
            ax.set_axisbelow(True)
            rung = rungs[cell][i]
            subtitle = ['all allowed bins', 'protein-coding TSS ±500 bp',
                        'cross-cell-type average disagrees', 'DNase-confirmed'][i]
            ax.set_title(f'{rung["name"]}\n{subtitle}\n{rung["n_pos"]:,} pos / {rung["n_neg"]:,} neg',
                         fontsize=13.7, pad=11)
            ax.set_xlabel('AUROC', fontsize=12)
        top -= row_heights[cell]
    fig.text(center, .018,
             'Bars: AUROC · whiskers: 95% region-bootstrap CI · hatched DNase R3 = 1 and average R2/R3 = 0 by construction\n'
             'CD4 AlphaGenome uses its naive CD4 T cell LQhCAGE track, a different sample from the CD4+ T cell truth.',
             ha='center', va='center', fontsize=12, color='#555555')
    fig.savefig(OUT / 'cd4_nk_panel3_enformer_cage.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot():
    register_arms()
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#555555', 'text.color': '#222222',
                         'axes.labelcolor': '#333333', 'xtick.color': '#555555'}):
        panel()


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('compute', 'all'):
        compute()
    if stage in ('plot', 'all'):
        plot()
