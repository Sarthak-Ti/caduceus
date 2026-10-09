"""Follow-up panels to IGVF_meeting_figures2, redrawn from the small tables in cache/.

    panel3_combined_four_truths.png             rung AUROCs on four ground truths
    overall_correlations_three_truths_128bp.png mean per-region smoothed log Pearson
    panel4_three_truths_128bp.png               accuracy vs similarity to the average
    fig1b_zmat4_ag_truth_32bp(_zoom).png        ZMAT4 locus against observed AG CAGE

The cache holds scores already computed from the model predictions (1,840 shared
held-out Enformer test regions; Borzoi row reuses the original fold3 benchmark).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import igvf_arms as A
from evals.plotting_utils import GeneAnnotation, Track, plot_tracks, GENCODE_V49

# panel 3
ROW_ORDER = ('enformer_cage', 'borzoi_cage', 'ag_cage', 'rampage_observed')
MAMBA_128 = ['mamba_1ct', 'mamba_nopre', 'mamba_8ct', 'mamba_ag_cage', 'mamba_rampage']
ARMS = {
    'enformer_cage': ['null_avg', 'acc_atac', *MAMBA_128, 'enformer_pre',
                      'enformer_n1', 'enformer_n1_1536', 'enformer_n16', 'enformer_n42',
                      'alphagenome', 'epibert'],
    'borzoi_cage': ['null_avg', 'acc_dnase', 'mamba_1ct_borzoi', 'borzoi_pre',
                    'borzoi_lora', 'borzoi_trunk', 'borzoi_scratch', 'alphagenome'],
    'ag_cage': ['null_avg', 'acc_atac', *MAMBA_128, 'enformer_pre', 'alphagenome', 'epibert'],
    'rampage_observed': ['null_avg', 'acc_atac', *MAMBA_128, 'enformer_pre', 'alphagenome', 'epibert'],
}
# three-truth panels
TARGETS = ('enformer_cage', 'ag_cage', 'rampage_observed')
TITLES = ('Enformer CAGE ground truth', 'AG CAGE ground truth', 'RAMPAGE ground truth')
MATCHED_MAMBA = dict(zip(TARGETS, ('mamba_1ct', 'mamba_ag_cage', 'mamba_rampage')))
# ZMAT4 locus
ZMAT4_ARMS = ['borzoi_lora', 'borzoi_trunk', 'borzoi_scratch', 'mamba_1ct_borzoi',
              'mamba_ag_cage', 'alphagenome']


def register_arms():
    """Arms added after igvf_arms was written, plus their relabels."""
    changes = {
        'enformer_n1': dict(hidden=False),
        'enformer_n1_1536': dict(label='Enformer (retrained, 1 CT, 1536ch)'),
        'mamba_1ct': dict(label='cis-Mamba (Enformer CAGE)'),
        'null_avg': dict(label='[null] Enformer CAGE average'),
    }
    for key, kwargs in changes.items():
        updated = dataclasses.replace(A.BY_ID[key], **kwargs)
        A.BY_ID[key] = updated
        A.ARMS[:] = [updated if a.id == key else a for a in A.ARMS]
    for arm in [A.Arm('mamba_ag_cage', 'cis-Mamba (AG CAGE)', 'ours', '#00796B', frozenset([A.ENFORMER_GRID])),
                A.Arm('mamba_rampage', 'cis-Mamba (RAMPAGE)', 'ours', '#004D2E', frozenset([A.ENFORMER_GRID])),
                A.Arm('epibert', 'EpiBERT (RAMPAGE)', 'epibert', '#7B3294', frozenset([A.ENFORMER_GRID]))]:
        if arm.id not in A.BY_ID:
            A.ARMS.append(arm)
        A.BY_ID[arm.id] = arm


def draw_schematic(fig, columns, top, total_height):
    # the analysis module restyles matplotlib globally on import
    with matplotlib.rc_context():
        from igvf_meeting_figures2_analysis import _simple_tracks, _simple_classes
    x, truth, avg, acc = _simple_tracks()
    classes = _simple_classes(x, truth, avg, acc)
    titles = [('R0 · all bins', 'every bin'), ('R1 · promoters', 'only near a TSS'),
              ('R2 · + disagreement', 'and the average disagrees'),
              ('R3 · + accessibility', 'and accessibility agrees')]
    center = (columns[0][0] + columns[-1][0] + columns[-1][1]) / 2
    fig.text(center, (top-.1)/total_height, 'What each rung selects', ha='center', va='top',
             fontsize=25, weight='bold')
    axes = []
    for i, ((left, width), (pos, neg), (title, subtitle)) in enumerate(zip(columns, classes, titles)):
        ax = fig.add_axes([left, (top-4.25)/total_height, width, 3.25/total_height])
        axes.append(ax)
        ax.fill_between(x, 0, truth, color='#E4E4E9', lw=0)
        ax.plot(x, truth, color='#55555D', lw=1.1)
        ax.axhline(.34, color='#A0A0A8', ls='--', lw=.8)
        ax.fill_between(x, -.36, -.06, color='#F1F1F4', lw=0)
        ax.fill_between(x, -.36, -.06, where=pos, color='#2E7D32', lw=0, interpolate=True)
        ax.fill_between(x, -.36, -.06, where=neg, color='#C62828', lw=0, interpolate=True)
        if i >= 1:
            ax.plot([.15, 9.85], [-.82, -.82], color='#444444', lw=1.3)
            for t in (2., 3.5, 5., 6.5):
                ax.plot([t, t], [-.82, -.52], color='#444444', lw=1.3)
                ax.annotate('', xy=(t+.42, -.52), xytext=(t, -.52),
                            arrowprops=dict(arrowstyle='-|>', color='#444444', lw=1.3))
        if i >= 2:
            ax.plot(x, avg, color='#3F5B8C', lw=1.8, ls=(0, (5, 2.5)))
        if i == 3:
            access = -1.9 + acc/acc.max()*.7
            ax.fill_between(x, -1.9, access, color='#8C564B', alpha=.3, lw=0)
            ax.plot(x, access, color='#8C564B', lw=1.4)
            ax.text(5, -2.15, 'accessibility', ha='center', va='top', fontsize=12, color='#555555')
        ax.set_title(f'{title}\n{subtitle}', fontsize=16, pad=11)
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.35, 3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel('Observed signal', fontsize=17)
    handles = [Patch(color='#2E7D32', label='positive class'), Patch(color='#C62828', label='negative class'),
               Patch(color='#F1F1F4', label='neither'), Line2D([0], [0], color='#55555D', lw=1.1, label='observed'),
               Line2D([0], [0], color='#3F5B8C', lw=1.8, ls=(0, (5, 2.5)), label='cross-cell-type average')]
    fig.legend(handles=handles, ncol=5, frameon=False, fontsize=13.5, loc='center left',
               bbox_to_anchor=(columns[0][0], (top-3.86)/total_height))


def panel3_four_truths():
    scores = pd.read_csv(CACHE / 'panel3_rung_scores.csv')
    rungs = json.loads((CACHE / 'panel3_rungs.json').read_text())
    row_heights = {key: 2.15 + .32*len(ARMS[key]) for key in ROW_ORDER}
    height = 4.7 + sum(row_heights.values()) + .85
    fig = plt.figure(figsize=(26, height))
    left, right, wspace = .188, .989, .20
    width = (right-left) / (4 + 3*wspace)
    columns = [(left+i*width*(1+wspace), width) for i in range(4)]
    center = (left+right)/2
    top = height-.2
    draw_schematic(fig, columns, top, height)
    top -= 4.7
    titles = {'enformer_cage': 'Enformer CAGE ground truth · 128 bp',
              'borzoi_cage': 'Borzoi CAGE ground truth · 32 bp, strand-specific',
              'ag_cage': 'AlphaGenome CAGE ground truth · 128 bp',
              'rampage_observed': 'epiBERT RAMPAGE ground truth · 128 bp'}
    notes = {'enformer_cage': '1,840 regions · Enformer CAGE average · ATAC',
             'borzoi_cage': '6,797 regions · Borzoi CAGE average · DNase · original fold3 benchmark',
             'ag_cage': '1,840 regions · observed AG-training targets · Enformer CAGE average · ATAC',
             'rampage_observed': '1,840 regions · observed RAMPAGE targets · Enformer CAGE average · ATAC'}
    model_labels = {'mamba_1ct_borzoi': 'cis-Mamba (Borzoi CAGE)'}
    for target in ROW_ORDER:
        d = scores[scores.target == target]
        references = [arm for arm in ARMS[target] if arm in ('null_avg', 'acc_atac', 'acc_dnase')]
        models = d[(d.rung_i == 3) & ~d.arm.isin(references)].sort_values('auroc', ascending=False).arm.tolist()
        order = references + models
        labels = []
        for arm in order:
            label = model_labels.get(arm, A.label(arm))
            if arm == 'null_avg':
                label = '[null] cross-cell-type average'
            if arm == 'alphagenome' and target == 'borzoi_cage':
                label = 'AlphaGenome†'
            labels.append(label)
        fig.text(center, (top-.10)/height, titles[target], ha='center', va='top', fontsize=23, weight='bold')
        fig.text(center, (top-.53)/height, notes[target], ha='center', va='top', fontsize=12.5, color='#555555')
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
            rung = rungs[target][i]
            subtitle = ['all allowed bins', 'protein-coding TSS ±500 bp',
                        'cross-cell-type average disagrees',
                        'DNase-confirmed' if target == 'borzoi_cage' else 'ATAC-confirmed'][i]
            if target == 'borzoi_cage' and i == 0:
                subtitle = 'all bins, 1-in-10 subsample'
            ax.set_title(f'{rung["name"]}\n{subtitle}\n{rung["n_pos"]:,} pos / {rung["n_neg"]:,} neg',
                         fontsize=13.7, pad=11)
            ax.set_xlabel('AUROC', fontsize=12)
        top -= row_heights[target]
    fig.text(center, .018,
             'Bars: AUROC · whiskers: 95% region-bootstrap CI · hatched accessibility R3 = 1 and average R2/R3 = 0 by construction\n'
             '† AlphaGenome training overlap with Borzoi fold3 is unverified. epiBERT is omitted from the Borzoi row.',
             ha='center', va='center', fontsize=12, color='#555555')
    fig.savefig(OUT / 'panel3_combined_four_truths.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overall_correlations():
    corr = pd.read_csv(CACHE / 'correlations.csv')
    order = corr[corr.target == 'rampage_observed'].sort_values('mean_sm', ascending=False).arm.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(22, 8.3), sharey=True)
    for ax, target, title in zip(axes, TARGETS, TITLES):
        d = corr[corr.target == target].set_index('arm').loc[order]
        y = np.arange(len(order))
        ax.barh(y, d.mean_sm, color=[A.color(a) for a in order], height=.65,
                xerr=np.vstack([np.maximum(0, d.mean_sm-d.lo_sm), np.maximum(0, d.hi_sm-d.mean_sm)]),
                error_kw=dict(ecolor='#444444', lw=1, capsize=2), zorder=3)
        for i, row in enumerate(d.itertuples()):
            ax.text(row.hi_sm + .009, i, f'{row.mean_sm:.3f}', va='center', fontsize=12)
        ax.set_yticks(y, [A.label(a) for a in order], fontsize=12)
        ax.set_ylim(len(order)-.5, -.5)
        ax.set_xlim(0, .72)
        ax.set_title(f'{title}\n{int(d.n_valid_sm.iloc[0]):,} regions with valid correlations', fontsize=16, pad=15)
        ax.set_xlabel('Mean per-region smoothed log Pearson', fontsize=13)
        ax.grid(axis='x', color='#dddddd', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    bold = {A.label(a) for a in (*MATCHED_MAMBA.values(), 'epibert')}
    for label in axes[0].get_yticklabels():
        if label.get_text() in bold:
            label.set_weight('bold')
    fig.suptitle('GM12878 model correlations at 128 bp', fontsize=23, weight='bold', y=.985)
    fig.tight_layout(rect=(0, .08, 1, .93), w_pad=3)
    fig.text(.57, .033,
             '1,840 evaluation regions · log1p → Gaussian σ = 1.5 bins → per-region Pearson\n'
             'Same valid regions for every model within each column; constant signals omitted · 95% region-bootstrap CIs',
             ha='center', va='center', fontsize=12, color='#555555')
    fig.savefig(OUT / 'overall_correlations_three_truths_128bp.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def panel4_three_truths():
    curves = pd.read_csv(CACHE / 'panel4_running_medians.csv')
    stats = pd.read_csv(CACHE / 'panel4_summary.csv').set_index(['target', 'arm'])
    columns = json.loads((CACHE / 'panel4_columns.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(23, 7.7), sharex=True, sharey=True)
    xmin = min(0., np.floor(curves.x_median.min() * 20) / 20)
    ymin = min(0., np.floor(curves.y_median.min() * 20) / 20)
    for ax, target, title, detail in zip(axes, TARGETS, TITLES, columns):
        handles = []
        for arm in detail['arms']:
            points = curves[(curves.target == target) & (curves.arm == arm)].sort_values('bin_index')
            ax.plot(points.x_median, points.y_median, '-o', color=A.color(arm),
                    lw=2.5, ms=5, markeredgecolor='white', markeredgewidth=.6, zorder=6)
            label = A.label(arm)
            if arm == 'enformer_n1_1536':
                label = 'Enformer (retrained, 1 CT)'
            handles.append(Line2D([0], [0], color=A.color(arm), lw=2.5, marker='o', ms=5,
                                  markeredgecolor='white', markeredgewidth=.6,
                                  label=f'{label}   median {stats.loc[(target, arm), "median_sm"]:.2f}'))
        ax.set_xlim(xmin, 1)
        ax.set_ylim(ymin, 1)
        ax.set_title(f'{title}\n{detail["n_regions"]:,} eligible regions', fontsize=17, pad=16)
        ax.set_xlabel('R(ground truth, Enformer CAGE average)\n→ region more similar to the average', fontsize=13)
        ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=10.5,
                  handlelength=2, labelspacing=.65)
        ax.grid(color='#e1e1e1', lw=.8, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel('R(model prediction, ground truth)', fontsize=15)
    fig.suptitle('Model accuracy vs similarity to the cross-cell-type average', fontsize=23, weight='bold', y=.985)
    fig.tight_layout(rect=(0, .105, 1, .93), w_pad=2.5)
    fig.text(.51, .038,
             '128 bp · smoothed log Pearson · 12 equal-count running medians · original Enformer CAGE average (GM12878 excluded)\n'
             'Target-matched cis-Mamba per column; epiBERT in all three · ≥3 positive bins at/above truth p95 (RAMPAGE p95 = 0)',
             ha='center', va='center', fontsize=11.5, color='#555555')
    fig.savefig(OUT / 'panel4_three_truths_128bp.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def zmat4_locus(arrays, meta, r_sm, windows, annotation, region, stem):
    common = dict(chrom=meta['chrom'], start=meta['start'], bin_size=32)
    tracks = []
    for arm in ZMAT4_ARMS:
        if arm == 'mamba_ag_cage':
            label = 'cis-Mamba (AG CAGE)\n(not held out)'
            color = '#B85B17'
        else:
            label = A.label(arm, A.BORZOI_GRID).replace(' (', '\n(')
            color = A.color(arm)
        tracks.append(Track(arrays[arm], name=label, color=color,
                            reference_exempt=arm != 'mamba_ag_cage', **common))
    tracks.append(Track(arrays['avg'], name=f"AVG\n({meta['n_cell_types']} CTs)",
                        color='#6B7280', style='fill', alpha=0.4,
                        reference_exempt=True, **common))
    truth = Track(arrays['ag_observed'], name='True GM12878\nAG CAGE (−)', color='black',
                  is_truth=True, **common)
    dnase = Track(arrays['dnase'], name='GM12878 DNase\n(unstranded)',
                  color='#8c564b', style='fill', alpha=0.5,
                  reference_exempt=True, **common)
    fig, axes = plot_tracks(
        tracks, ground_truth=truth, bottom_tracks=[dnase],
        genes=[('All genes', annotation)], region=(meta['chrom'], *region),
        collapse_genes=True, gene_track_ratio=1.0, figsize=(17.5, 12.5),
    )
    for i, ax in enumerate(axes):
        label = ax.get_ylabel()
        ax.set_ylabel('')
        ax.text(-0.022, 0.5, label, transform=ax.transAxes, ha='right', va='center',
                fontsize=10.5, weight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if i < len(ZMAT4_ARMS):
            for line in ax.lines:
                if line.get_linestyle() not in ('--', ':'):
                    line.set_linewidth(1.8)
            # Every number refers to the full original panel-1b locus, even in the zoom.
            ax.text(0.012, 0.86, f"r(sm) to AG truth = {r_sm[ZMAT4_ARMS[i]]:.3f}",
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
        for j, w in enumerate(windows):
            ax.axvspan(w['start'], w['end'], color='#D29A22' if j == 0 else '#378561',
                       alpha=0.11, lw=0, zorder=0)
    for text in axes[-1].texts:
        if text.get_text() == 'ZMAT4':
            text.set_fontweight('bold')
            text.set_fontsize(12)
    low, high = axes[-1].get_ylim()
    axes[-1].set_ylim(low, high + 0.8)
    # AlphaGenome API predictions have their own units; the new Mamba prediction
    # and its observed AG target share a scale and the truth-max reference.
    if tuple(region) != (meta['start'], meta['end']):
        centers = meta['start']+(np.arange(meta['n_bin'])+0.5)*32
        visible = (centers >= region[0]) & (centers < region[1])
        values = arrays['alphagenome'][visible]
        axes[ZMAT4_ARMS.index('alphagenome')].set_ylim(min(0, float(values.min())), float(values.max())*1.05)
    fig.suptitle('ZMAT4 locus · observed AlphaGenome CAGE targets · 32 bp bins',
                 fontsize=17, weight='bold', y=0.995)
    fig.text(0.5, 0.968,
             'Gold shading: strongest cross-cell-type-average peak. '
             'Green shading: observed GM12878 peaks in the AG target data (minus strand).',
             ha='center', fontsize=10.5)
    fig.text(0.5, 0.014,
             'AG-CAGE Mamba overlaps its training regions. Mamba and its AG target share a scale; '
             'other tracks use their own units.\n'
             'r(sm): log1p followed by Gaussian smoothing (σ = 192 bp), then Pearson; '
             'scores use the full 196,608 bp locus.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.935, bottom=0.09, hspace=0.26)
    fig.savefig(OUT / f'{stem}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def zmat4_figures():
    info = json.loads((CACHE / 'zmat4.json').read_text())
    meta, windows = info['locus'], info['peak_windows']
    with np.load(CACHE / 'zmat4_tracks_32bp.npz') as z:
        arrays = {k: z[k] for k in z.files}
    annotation = GeneAnnotation(gene_types=None, cache_dir=str(Path(GENCODE_V49).parent))
    zmat4_locus(arrays, meta, info['r_sm_ag_target'], windows, annotation,
                (meta['start'], meta['end']), 'fig1b_zmat4_ag_truth_32bp')
    zoom = (max(meta['start'], int(min(w['peak_center'] for w in windows))-8000),
            min(meta['end'], int(max(w['peak_center'] for w in windows))+8000))
    zmat4_locus(arrays, meta, info['r_sm_ag_target'], windows, annotation, zoom,
                'fig1b_zmat4_ag_truth_32bp_zoom')


def main():
    register_arms()
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#555555', 'text.color': '#222222',
                         'axes.labelcolor': '#333333', 'xtick.color': '#555555'}):
        panel3_four_truths()
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelcolor': '#252525',
                         'axes.titlecolor': '#252525', 'text.color': '#252525'}):
        overall_correlations()
        panel4_three_truths()
    zmat4_figures()


if __name__ == '__main__':
    main()
