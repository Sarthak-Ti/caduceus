"""Panel 3's Enformer CAGE row, redone on bins where GM12878 CAGE and RAMPAGE agree.

    panel3_cage_rampage_agreement.png   rung schematic, original Enformer CAGE row, agreement row

A bin can be positive only if both assays call it active, and negative only if both
call it silent. Bins where the two disagree are left out of both classes. Each assay
keeps the thresholds it had in comparing_model7 §5:
    CAGE     active >= 1.0                         silent < 0.2
    RAMPAGE  active >= median nonzero (~0.994)     silent == 0
RAMPAGE is the EpiBERT GM12878 track, already on the 1,840 x 896 Enformer grid
(EpiBERT/data/rampage_ladder_1840/aligned_rampage.npz). The cross-cell-type average (R2)
and ATAC (R3) conditions are not changed.

The cis-Mamba (RAMPAGE) arm is left out of both rows because its predictions are missing.

    python make_agreement_panel.py compute   # rebuilds the original row, checks it against the cache, then scores the agreement row
    python make_agreement_panel.py plot
    python make_agreement_panel.py           # both
"""
from pathlib import Path
import json
import sys

ROOT = Path('/data1/lesliec/sarthak/caduceus')
OUT = Path(__file__).resolve().parent
CACHE = OUT / 'cache'
sys.path[:0] = [str(OUT), str(ROOT / 'notebooks/figures'), str(ROOT)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import igvf_arms as A
import igvf_meeting_figures2_pipeline as P
import make_cd4_nk_panel as C
import make_figures as F
from evals.plotting_utils import GeneAnnotation

DATA, MO, SHARED = C.DATA, C.MO, C.SHARED
ARMS = [arm for arm in F.ARMS['enformer_cage'] if arm != 'mamba_rampage']
RP_ACTIVE_PERCENTILE = 50
ROW_ORDER = ('original', 'agreement')


# =============================================================================
# compute
# =============================================================================
def predictions(true, rows):
    """Every model arm of the Enformer CAGE row, on the 1,840 shared regions."""
    out = {}
    for arm, stem in (('mamba_1ct', 'GM12878_no_mlm'), ('mamba_nopre', 'nopretrain'), ('mamba_8ct', 'immune_CAGE')):
        C.target_gate(arm, np.load(f'{MO}/{stem}_targets.npy'), true)
        pred = np.load(f'{MO}/{stem}_predictions.npy')
        out[arm] = pred[:, :, 0] if pred.ndim == 3 else pred
    # bp-resolution, two strands, over the full 524 kb input: sum the strands, crop the
    # Enformer output window, mean per 128 bp bin (reproduces the cached panel 3 row)
    raw = np.load(f'{MO}/mamba_GM12878_AG_CAGE.npy', mmap_mode='r')
    span = P.E_T * P.E_BIN
    offset = (raw.shape[2] - span) // 2
    ag_mamba = np.zeros((len(true), P.E_T), np.float32)
    for r in rows:
        ag_mamba[r] = np.asarray(raw[r, 0, offset:offset+span]).sum(-1).reshape(P.E_T, P.E_BIN).mean(1)
    out['mamba_ag_cage'] = ag_mamba
    out['enformer_pre'] = np.load(f'{MO}/enformer_test_track{P.E_TRACK}.npy')
    for arm, key in (('enformer_n1', 'enformer_retrained_gm12878'), ('enformer_n1_1536', 'enformer_retrained_n1_1536'),
                     ('enformer_n16', 'enformer_retrained_n16'), ('enformer_n42', 'enformer_retrained_n42')):
        stem = f'{MO}/{key}/{key}_cage'
        pred = np.load(f'{stem}_predictions.npy', mmap_mode='r')
        targets = np.load(f'{stem}_targets.npy', mmap_mode='r')
        if pred.ndim == 3:
            tracks = pd.read_csv(f'{stem}_tracks.csv')
            row = tracks[(tracks.cell_type == 'GM12878') & (tracks.assay == 'cage')]
            assert len(row) == 1 and int(row.target_index.iloc[0]) == P.E_TRACK
            col = int(row.column.iloc[0])
            pred, targets = pred[:, :, col], targets[:, :, col]
        C.target_gate(arm, targets, true)
        out[arm] = pred
    out['alphagenome'] = np.load(f'{MO}/alphagenome_cage_test_128bp.npy', mmap_mode='r')[:, :, C.CELLS['GM12878']['ag']]
    out['epibert'] = np.load(f'{MO}/epibert_gm12878_rampage_test_outputs.npy')
    for arm, pred in out.items():
        assert pred.shape == true.shape, (arm, pred.shape)
    return {arm: np.asarray(pred[rows], np.float32) for arm, pred in out.items()}


def rampage_classes(rows):
    """Per-bin RAMPAGE active / silent masks, with the comparing_model7 §5 thresholds."""
    with np.load(f'{SHARED}/aligned_rampage.npz') as z:
        assert np.array_equal(z['enformer_rows'], rows)
        rampage = z['gm12878'].ravel()
    assert np.isfinite(rampage).all()
    active = float(np.percentile(rampage[rampage > 0], RP_ACTIVE_PERCENTILE))
    ref = json.loads((Path(SHARED) / 'thresholds.json').read_text())
    ref = next(t for t in ref if t['assay'] == 'RAMPAGE')
    assert np.isclose(active, ref['active_threshold']) and ref['silent_rule'] == '==0'
    return rampage >= active, rampage == 0, active


def agreement_rungs(rungs, rp_active, rp_silent, rid):
    """The same rungs with discordant bins removed from both classes."""
    out = []
    for rung in rungs:
        pos = np.zeros(rid.size, bool)
        neg = np.zeros(rid.size, bool)
        pos[rung['pi']] = True
        neg[rung['ni']] = True
        out.append(P.make_rung(rung['name'], rung['subtitle'], pos & rp_active, neg & rp_silent, rid))
    return out


def compute():
    tb, rows, shared = C.test_regions()
    centers = shared.output_start.to_numpy()[:, None] + (np.arange(P.E_T) + .5)[None, :] * P.E_BIN
    tss = P.tss_mask(shared.chrom.to_numpy(), centers, GeneAnnotation(cache_dir=f'{DATA}/DE_danwei').df,
                     strand=None, pad=P.E_TSSPAD)
    true = np.load(f'{MO}/GM12878_base_targets_conv.npy').astype(np.float32)
    atac = np.load(f'{DATA}/enformer/data/GM12878_ATAC_WW_denoised.npz')['test'][:, :, 0]
    avg = C.leave_one_out_avg(true)
    arms = {'null_avg': avg[rows].astype(np.float32), 'acc_atac': atac[rows].astype(np.float32),
            **predictions(true, rows)}
    assert sorted(arms) == sorted(ARMS), sorted(set(arms) ^ set(ARMS))

    # --- gate: the original row reproduces the cached panel 3 numbers --------
    rungs = C.build_rungs(true[rows], avg[rows], atac[rows], tss, 'ATAC')
    original = C.score(arms, rungs, 'original', 'acc_atac')
    ref = pd.read_csv(CACHE / 'panel3_rung_scores.csv')
    ref = ref[(ref.target == 'enformer_cage') & ref.arm.isin(ARMS)].set_index(['arm', 'rung_i'])
    merged = original.set_index(['arm', 'rung_i']).join(ref[['auroc', 'lo', 'hi']], rsuffix='_ref', how='outer')
    diff = merged[['auroc', 'lo', 'hi']].to_numpy() - merged[['auroc_ref', 'lo_ref', 'hi_ref']].to_numpy()
    assert np.isfinite(diff).all() and np.abs(diff).max() < 1e-6, \
        f'GATE FAILED: original row differs from panel 3 by {np.nanmax(np.abs(diff)):.3g}'
    print('  gate: original row reproduces cached panel 3 for all arms (AUROC and CI)', flush=True)

    # --- agreement row -------------------------------------------------------
    rp_active, rp_silent, rp_threshold = rampage_classes(rows)
    rid = np.repeat(np.arange(len(rows)), P.E_T)
    concordant = agreement_rungs(rungs, rp_active, rp_silent, rid)
    agreement = C.score(arms, concordant, 'agreement', 'acc_atac')

    rung_json = {'original': P.rungs_to_json(rungs), 'agreement': P.rungs_to_json(concordant)}
    for name, rs in rung_json.items():
        print(f'  [{name}] ' + ' | '.join(f'{r["name"]}: {r["n_pos"]:,}/{r["n_neg"]:,}' for r in rs), flush=True)
    pd.concat([original, agreement]).to_csv(CACHE / 'agreement_rung_scores.csv', index=False)
    (CACHE / 'agreement_rungs.json').write_text(json.dumps(dict(
        rungs=rung_json, rampage_active_threshold=rp_threshold, rampage_silent_rule='==0',
        cage_active_threshold=P.E_ACTIVE, cage_silent_threshold=P.E_SILENT), indent=2))
    print('wrote cache/agreement_rung_scores.csv and cache/agreement_rungs.json', flush=True)


# =============================================================================
# plot
# =============================================================================
def panel():
    scores = pd.read_csv(CACHE / 'agreement_rung_scores.csv')
    info = json.loads((CACHE / 'agreement_rungs.json').read_text())
    rungs = info['rungs']
    titles = {'original': 'Enformer CAGE ground truth · 128 bp',
              'agreement': 'Enformer CAGE ground truth, RAMPAGE-concordant bins only · 128 bp'}
    notes = {'original': '1,840 regions · Enformer CAGE average · ATAC',
             'agreement': f'1,840 regions · pos: CAGE ≥ {info["cage_active_threshold"]:g} and RAMPAGE ≥ '
                          f'{info["rampage_active_threshold"]:.3f} · neg: CAGE < {info["cage_silent_threshold"]:g} '
                          'and RAMPAGE = 0 · discordant GM12878 bins excluded'}
    row_heights = {name: 2.15 + .32*len(ARMS) for name in ROW_ORDER}
    height = 4.7 + sum(row_heights.values()) + .85
    fig = plt.figure(figsize=(26, height))
    left, right, wspace = .188, .989, .20
    width = (right-left) / (4 + 3*wspace)
    columns = [(left+i*width*(1+wspace), width) for i in range(4)]
    center = (left+right)/2
    top = height-.2
    F.draw_schematic(fig, columns, top, height)
    top -= 4.7
    for name in ROW_ORDER:
        d = scores[scores.target == name]
        references = ['null_avg', 'acc_atac']
        models = d[(d.rung_i == 3) & ~d.arm.isin(references)].sort_values('auroc', ascending=False).arm.tolist()
        order = references + models
        labels = ['[null] cross-cell-type average' if arm == 'null_avg' else A.label(arm) for arm in order]
        fig.text(center, (top-.10)/height, titles[name], ha='center', va='top', fontsize=23, weight='bold')
        fig.text(center, (top-.53)/height, notes[name], ha='center', va='top', fontsize=12.5, color='#555555')
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
                    if arm.startswith('mamba_') or arm == 'epibert':
                        text.set_weight('bold')
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=12)
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', color='#DDDDDD', lw=.65, ls='--')
            ax.set_axisbelow(True)
            rung = rungs[name][i]
            subtitle = ['all allowed bins', 'protein-coding TSS ±500 bp',
                        'cross-cell-type average disagrees', 'ATAC-confirmed'][i]
            ax.set_title(f'{rung["name"]}\n{subtitle}\n{rung["n_pos"]:,} pos / {rung["n_neg"]:,} neg',
                         fontsize=13.7, pad=11)
            ax.set_xlabel('AUROC', fontsize=12)
        top -= row_heights[name]
    fig.text(center, .018,
             'Bars: AUROC · whiskers: 95% region-bootstrap CI · hatched accessibility R3 = 1 and average R2/R3 = 0 by construction\n'
             'RAMPAGE: EpiBERT GM12878 track · the agreement filter applies to GM12878 only; '
             'average and ATAC criteria are unchanged · cis-Mamba (RAMPAGE) omitted',
             ha='center', va='center', fontsize=12, color='#555555')
    fig.savefig(OUT / 'panel3_cage_rampage_agreement.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot():
    F.register_arms()
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
