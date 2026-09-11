"""Figure functions for IGVF_meeting_figures2 — reads the cache, never a source store.

Every function here loads a few kilobytes of parquet/npz and returns a matplotlib
figure. Nothing opens a zarr, a prediction array or a labels store, so restyling
is instant and a kernel restart costs nothing. If a panel raises because a cache
file is missing, run the matching pipeline stage:

    python igvf_meeting_figures2_pipeline.py enformer   # figures 2a, 3a, 4
    python igvf_meeting_figures2_pipeline.py borzoi     # figures 1, 2b, 3b
    python igvf_meeting_figures2_pipeline.py qtl        # figures 5, 6

Arm labels and colours come from `igvf_arms` and nowhere else, so a rename or a
recolour is one edit there and every panel follows.

The one exception to "no slow reads" is the gene annotation that figure 1's gene
track needs. It is loaded lazily and memoised per process, so the first call to
panel 1 pays for it once and later calls do not.
"""
from pathlib import Path
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import igvf_arms as A
from igvf_arms import ENFORMER_GRID, BORZOI_GRID, EQTL, DSQTL

ROOT = Path(__file__).resolve().parent
# The locus panels draw with evals.plotting_utils, which lives at the repo root.
# This has to happen at import time, not inside the panel: the panel's own
# `from evals...` runs before any helper that could set the path up, so a kernel
# whose cwd is code_test/ would fail on the import rather than on the lookup.
CADUCEUS_ROOT = str(ROOT.parent)
if CADUCEUS_ROOT not in sys.path:
    sys.path.insert(0, CADUCEUS_ROOT)

CACHE = Path(os.environ.get('IGVF2_CACHE', ROOT / 'IGVF_meeting_figures2_cache'))
FIGDIR = Path(os.environ.get('IGVF2_FIGDIR', ROOT / 'igvf_meeting_figs2'))
FIGDIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
    'figure.dpi': 110, 'savefig.dpi': 300, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.9,
    'axes.edgecolor': '#444444', 'text.color': '#222222',
    'axes.labelcolor': '#222222', 'xtick.color': '#444444',
    'ytick.color': '#444444', 'figure.facecolor': 'white',
})
INK, INK2 = '#222222', '#555555'

METRIC_LABEL = {'sm': 'Smoothed-log Pearson', 'raw': 'Pearson'}


def _need(name):
    p = CACHE / name
    assert p.exists(), (
        f'{p} is missing. Build it with:  python igvf_meeting_figures2_pipeline.py '
        f'{"borzoi" if "borzoi" in name or "locus" in name else "enformer" if "enformer" in name else "qtl"}')
    return p


def tidy(ax, grid='y'):
    """Recessive grid behind the marks, no tick spurs on the categorical axis."""
    ax.grid(axis=grid, ls='--', lw=0.55, color='#CCCCCC', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x' if grid == 'y' else 'y', length=0)
    return ax


def save(fig, stem):
    """PNG for slides, PDF for anything that gets scaled."""
    fig.savefig(FIGDIR / f'{stem}.png', bbox_inches='tight')
    fig.savefig(FIGDIR / f'{stem}.pdf', bbox_inches='tight')
    return fig


def _label_color_for(hexcolor):
    """White text on a dark bar, dark ink on a light one -- so a value label drawn
    on top of an arbitrary arm color (igvf_arms.py spans dark green to pale pink)
    stays readable either way."""
    r, g, b = (int(hexcolor.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#FFFFFF' if luminance < 140 else '#1a1a1a'


# ============================================================================
# FIGURE 1 - the example locus. comparing_model5 §9c rank 9, region 3304.
# ============================================================================
_ANN = {}


def _annotation(gene_types):
    """GeneAnnotation, memoised per process. The only slow load left in this module."""
    key = gene_types
    if key not in _ANN:
        from evals.plotting_utils import GeneAnnotation
        _ANN[key] = GeneAnnotation(gene_types=gene_types)
    return _ANN[key]


# The default gene track is all GENCODE biotypes, which already contains the
# protein-coding genes -- a separate coding-only track would be a strict subset and
# earns no space. The coding constant is kept for gene_tracks='both'/'coding',
# where IG/TR biotypes are included because GM12878 is an EBV-transformed B-cell
# line and IGK/IGH/IGL are among its most cell-type-specific loci.
GENE_TYPES_CODING = ('protein_coding', 'IG_V_gene', 'IG_D_gene', 'IG_J_gene', 'IG_C_gene',
                     'TR_V_gene', 'TR_D_gene', 'TR_J_gene', 'TR_C_gene')

FIG1_ARMS = ('borzoi_pre', 'alphagenome')

# One title, used by both locus panels. The poster carries its own heading, so the
# panels do not name which arms they draw -- figure 1 and figure 1b are the same
# locus and read as one picture.
FIG1_TITLE = 'ZMAT4 locus example in Borzoi test set (held out)'


def panel1_locus(arms=FIG1_ARMS, show_dnase=True, gene_tracks='all',
                 title=FIG1_TITLE, stem='fig1_locus', aspect=2.0):
    """Predictions, truth and the cross-cell-type average at one held-out locus.

    `arms` defaults to pretrained Borzoi and AlphaGenome only -- the two models
    the audience is asked to accept before the truth is revealed. Every arm is in
    the cache, so adding the retrained models back is a keyword, not a rebuild.

    The observed GM12878 DNase sits below the truth axis, next to the gene models:
    a different assay in different units, so it is reference_exempt and is not read
    as another prediction.

    AlphaGenome is drawn `reference_exempt`: it predicts in its own units rather
    than Borzoi squashed counts, so the shared truth-max line would compare two
    different scales. Its axis auto-scales instead, and its label carries the
    not-held-out caveat.

    `aspect` is the figure's width:height ratio. Height is set first, from the
    (squished) per-subplot height below; width is then `aspect * height`, so the
    figure lands on the requested ratio regardless of how many arms are drawn.
    """
    from evals.plotting_utils import Track, plot_tracks
    d = np.load(_need('locus.npz'))
    meta = json.loads((CACHE / 'locus.json').read_text())
    chrom, start, end = meta['chrom'], meta['start'], meta['end']
    common = dict(chrom=chrom, start=start, bin_size=meta['bin_bp'])
    sign = '+' if meta['strand'] == 'plus' else '-'

    tracks = [Track(d[f'pred_{a}'], name=A.label(a, BORZOI_GRID).replace(' (', '\n('),
                    color=A.color(a), reference_exempt=(a == 'alphagenome'), **common)
              for a in arms]
    tracks.append(Track(d['avg'], name=f"AVG\n({meta['n_cell_types']} CTs)",
                        color='#6B7280', style='fill', alpha=0.35,
                        reference_exempt=True, **common))
    truth = Track(d['true'], name=f'True CAGE\n({sign})', color='#000000',
                  is_truth=True, **common)

    bottom = []
    if show_dnase:
        bottom.append(Track(d['dnase'], name='GM12878 DNase\n(unstranded)', color='#8c564b',
                            style='fill', alpha=0.55, reference_exempt=True, **common))

    genes = {'both': [('Protein-coding\n+ IG/TR', _annotation(GENE_TYPES_CODING)),
                      ('All genes', _annotation(None))],
             'all': [('All genes', _annotation(None))],
             'coding': [('Protein-coding\n+ IG/TR', _annotation(GENE_TYPES_CODING))]}[gene_tracks]

    # Title is one line: name, provenance, coordinates. The span, the bin count and
    # the gene list all used to hang off it; the gene track and the axis already say
    # those, so they were costing two lines of the figure's most prominent text.
    heading = f"{title}, {chrom}:{start:,}-{end:,} {sign} strand"
    # Per-subplot height squished from 1.55/5.5 down to 1.0/3.0 -- shorter tracks,
    # padding trimmed to match -- then width is derived from `aspect` so the figure
    # hits the requested ratio instead of the old fixed 14" width.
    height = 1.0 * (len(tracks) + 3) + 3.0
    fig, axes = plot_tracks(tracks, genes=genes, region=(chrom, start, end),
                            ground_truth=truth, bottom_tracks=bottom, collapse_genes=True,
                            title=heading, gene_track_ratio=(0.7, 2.0),
                            figsize=(aspect * height, height))
    # plot_tracks draws the y-axis label and the right-hand tick numbers small
    # (9pt / 8pt) to fit a dense multi-panel stack -- bump both up for legibility here.
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, weight='bold')
        ax.tick_params(axis='y', labelsize=11)
    # Prediction lines only (one axis per arm, at the front of `axes`) -- thicker so
    # they read from the back of a room. The dashed truth-max reference line on these
    # same axes is left alone; the '--' filter keeps it from being fattened too.
    for ax in axes[:len(arms)]:
        for line in ax.lines:
            if line.get_linestyle() not in ('--', ':'):
                line.set_linewidth(3.0)
    # AVG and True CAGE (and DNase) are drawn with fill_between, which has no stroke
    # of its own -- unlike the prediction lines above, "thicker" for these means
    # giving the fill an edge, at the same weight, so every track reads equally bold.
    # The edge is forced fully opaque (alpha=1) regardless of the fill's own alpha, or
    # it would trace each thin spike in the same washed-out tone as its interior and
    # add no visible weight.
    for ax in axes:
        for coll in ax.collections:
            edge = coll.get_facecolor()[0].copy()
            edge[3] = 1.0
            coll.set_edgecolor(edge)
            coll.set_linewidth(3.0)
    # ZMAT4 is the locus the whole figure is about; every other gene in the window is
    # context. Find its label text on the gene axis and make it stand out.
    for txt in axes[-1].texts:
        if txt.get_text() == 'ZMAT4':
            txt.set_fontsize(14)
            txt.set_weight('bold')
    # ZMAT4 sits on the gene track's top row, and its label is drawn just above the
    # gene backbone with clip_on=True -- at this bigger size the label's top edge can
    # run past the axis's own ylim and get clipped. No row in this window is packed
    # above it, so the fix is headroom, not a row reassignment: push the y-axis's top
    # limit up, which reads as sliding the whole gene track down away from the edge.
    gy_lo, gy_hi = axes[-1].get_ylim()
    axes[-1].set_ylim(gy_lo, gy_hi + 0.8)
    # plot_tracks already ran tight_layout() once, sized for the small default labels;
    # the bigger, two-line rotated labels above need the axes re-flowed for them, or
    # they run into the row above/below on these squished subplot heights. Re-running
    # it now lets tight_layout measure the new label bboxes and pick its own hspace.
    fig.tight_layout()
    # plot_tracks sets the suptitle at 13pt; re-issue it larger now that it is one line
    # (tight_layout above can also nudge it, so this has to come after).
    fig.suptitle(heading, fontsize=21, weight='bold', y=0.995)
    return save(fig, stem)


# The fix, at the same locus: what adaptation to GM12878 actually buys. Pretrained
# Borzoi and AlphaGenome are deliberately absent -- figure 1 already showed them,
# and the two panels are meant to be read as before and after on one picture.
FIG1B_ARMS = ('borzoi_lora', 'borzoi_trunk', 'borzoi_scratch', 'mamba_1ct_borzoi')


def panel1b_locus_retrained(arms=FIG1B_ARMS, show_dnase=True, gene_tracks='all',
                            title=FIG1_TITLE, aspect=1.75):
    """The figure-1 locus again, with the adapted models instead of the stock ones.

    Same region, same axes, same truth and same cross-cell-type average, so the
    panel can be shown straight after figure 1 and the only thing that changed is
    which models are drawn.
    """
    return panel1_locus(arms=arms, show_dnase=show_dnase, gene_tracks=gene_tracks,
                        title=title, stem='fig1b_locus_retrained', aspect=aspect)


# ============================================================================
# FIGURE 2 - per-region accuracy. The premise: on the standard metric everyone
# looks fine. AlphaGenome is excluded; the no-pretraining ablation is included
# on the Enformer grid, which is the only grid it exists on.
# ============================================================================
def _accuracy_bars(df, arms, metric, title, subtitle, figsize, show_spread=False,
                   labels=None):
    """Mean per-region correlation as ranked horizontal bars, in the eQTL panel's style.

    A bar carries the mean, which is the number the boxplots printed anyway.

    `show_spread=True` adds the interquartile range as a thin segment. Off by
    default: the IQRs of these arms overlap almost entirely, so the whiskers add
    a lot of ink, push the value labels away from the bars, and say only what the
    near-equal bar lengths already say. Worth switching on if anyone asks whether
    the ordering is meaningful -- the answer is that per-region spread dwarfs the
    between-arm differences, which is the point of this figure.
    """
    vals = {a: df[f'r_{metric}_{a}'].to_numpy() for a in arms}
    vals = {a: v[~np.isnan(v)] for a, v in vals.items()}
    order = sorted(arms, key=lambda a: -vals[a].mean())
    means = np.array([vals[a].mean() for a in order])
    q1 = np.array([np.percentile(vals[a], 25) for a in order])
    q3 = np.array([np.percentile(vals[a], 75) for a in order])
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, means, height=0.68, color=[A.color(a) for a in order], zorder=3)
    if show_spread:
        for yi, (lo, hi) in enumerate(zip(q1, q3)):
            ax.plot([lo, hi], [yi, yi], color='#333333', lw=1.1, zorder=4,
                    solid_capstyle='butt')
    # Value label drawn inside the bar, right-aligned near its end, rather than out
    # past it -- frees the strip of axis that used to exist only to hold the label,
    # so the bars can run most of the width even on a squarer, less panoramic figure.
    for yi, v in enumerate(means):
        ax.text(v - 0.012, yi, f'{v:.3f}', va='center', ha='right', fontsize=10,
                weight='bold', color=_label_color_for(A.color(order[yi])), zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([(labels or {}).get(a, A.label(a)) for a in order], fontsize=10)
    ax.invert_yaxis()
    right = max(q3.max(), means.max()) if show_spread else means.max()
    ax.set_xlim(0, min(1.0, right * 1.05))
    ax.set_xlabel(f'{METRIC_LABEL[metric]} correlation'
                  + ('   (bar = mean, line = IQR)' if show_spread else '   (mean)'),
                  fontsize=13)
    tidy(ax, grid='x')
    ax.tick_params(axis='x', labelsize=12)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    # Title and subtitle as figure-level text rather than ax.set_title/ax.text: the
    # axes' own left edge sits well right of the canvas's left edge (the row labels
    # live in that gap), so anchoring to the axes pushed a left-aligned title right
    # of where it visually should start. Figure fraction x=0.02 sits at the true
    # left margin instead, which also buys the now-bigger text more room before it
    # would run into the right edge and force the tight bbox to widen the canvas
    # (see FIG2_FIGSIZE). subplots_adjust reserves the top strip tight_layout no
    # longer knows to leave, since it never saw a title on the axes.
    fig.subplots_adjust(top=0.87)
    fig.text(0.02, 0.985, title, fontsize=15.5, weight='bold', color=INK,
             ha='left', va='top')
    fig.text(0.02, 0.925, subtitle, fontsize=13, color=INK2, ha='left', va='top')
    return fig


# Requested figure ratio, 7.35:4.95 (~1.485:1) -- noticeably squarer than the old
# 9.6:4.0 / 9.6:3.4 panels. Shared by 2a and 2b.
#
# Width is 7.8, not 7.35: the fig-level title text (see below) is drawn at its
# natural size with nothing to wrap or shrink it, and at this fontsize the longest
# title ("...Enformer test set average region correlation") renders ~7.62in wide.
# A canvas narrower than that overflows, and save()'s bbox_inches='tight' then
# expands the saved PNG to include the overflow -- silently stretching the actual
# aspect ratio away from what was requested. 7.8 gives it headroom to fit inside
# the nominal canvas, so the "tight" crop only trims ordinary whitespace and the
# saved ratio actually lands close to what FIG2_FIGSIZE says.
FIG2_FIGSIZE = (7.8, 7.8 / (7.35 / 4.95))


def panel2a_accuracy_enformer(metric='sm', show_spread=False):
    """Enformer grid, 128 bp. Includes the no-pretraining ablation."""
    df = pd.read_parquet(_need('enformer_boxplot.parquet'))
    df = df[df.allowed]
    arms = [a.id for a in A.arms_for(ENFORMER_GRID, exclude=('alphagenome',),
                                     include_reference=False)]
    fig = _accuracy_bars(df, arms, metric,
                         'GM12878 CAGE, Enformer test set average region correlation',
                         f'{len(df):,} allowed regions', FIG2_FIGSIZE, show_spread)
    return save(fig, f'fig2a_accuracy_enformer_{metric}')


def panel2b_accuracy_borzoi(metric='sm', show_spread=False):
    """Borzoi fold3 grid, 32 bp, both strands pooled."""
    df = pd.read_parquet(_need('borzoi_boxplot.parquet'))
    df = df[df.allowed]
    arms = [a.id for a in A.arms_for(BORZOI_GRID, exclude=('alphagenome',),
                                     include_reference=False)]
    n_reg = df.region.nunique()
    # Local to this panel: name the training data on the cis-Mamba arm. Every other
    # panel's cis-Mamba is the Enformer-data model, and the Borzoi arms around it here
    # make "Borzoi" ambiguous unless the arm says which sense is meant. Overridden here
    # rather than in the registry so panels 1b and 3c keep the plain label.
    fig = _accuracy_bars(df, arms, metric,
                         'GM12878 CAGE, Borzoi test set average region correlation',
                         f'{n_reg:,} allowed regions x 2 strands = {len(df):,} samples',
                         FIG2_FIGSIZE, show_spread,
                         labels={'mamba_1ct_borzoi': 'cis-Mamba (1 cell type, Borzoi data)'})
    return save(fig, f'fig2b_accuracy_borzoi_{metric}')


# ============================================================================
# FIGURE 3b / 3c - the stratum ladder. Same ruler, shrinking stratum.
# Ported from comparing_model6 fig_ladder_bars, on canonical arm ids.
# ============================================================================
ANCHOR_LO = 'chance = 0.5   (the cross-CT average is a GOOD predictor here)'
# Rungs 2-3 are scaled between two anchors that are not a chance line, so 0.5 has no
# direct reading on them. It is neither labelled nor drawn there.
ANCHOR_HI = '0 = pure average   ·   1 = fully specific'

# Shared four-column geometry for 3a/3b/3c (all now 24in wide, see their figsize
# calls), so panel3_combined's stack lines rung columns up across all three rows.
# Left previously came from each panel's own tight_layout(), which sizes the left
# margin to that panel's own y-axis content -- "CAGE signal" (3a) needs far less
# room than the longest model tick label (3b/3c), so the panels' columns landed at
# different x positions when stacked. RUNG_COL_LEFT is fixed at (just past) the
# widest of the two auto-computed 3b/3c margins instead, and reused verbatim by 3a
# -- the one visible cost is blank space to the left of 3a's short y-label, where
# 3b/3c actually need that width for their model names. Right and wspace already
# came out the same across all three unprompted (no other panel puts anything to
# the right of the last column or needs a different gap), so pinning them is just
# insurance against that changing if the content ever does.
RUNG_COL_LEFT, RUNG_COL_RIGHT, RUNG_COL_WSPACE = 0.145, 0.9931, 0.20


def _ladder_bars(df, rungs, grid, grid_title, order_by_rung=-1, show_anchor_footnote=True):
    """Four ranked panels, one per rung, rows in a single shared order.

    Reference rows -- the null and the measured accessibility baseline -- are
    pinned to the top of every panel: they are what the model rows are read
    against, not competitors to be ranked among them. A bar flagged
    `by_construction` is hatched, because the accessibility baseline scoring
    1.000 on the rung whose classes it defines is a restatement of the class
    definition and not a result.

    `show_anchor_footnote=False` drops the "rungs 0-1 / rungs 2-3" caption: the
    text is identical on every grid (it explains the rung logic, not this
    grid's numbers), so panel3b turns it off and panel3c is the one panel that
    keeps it -- otherwise the two copies collide when panel3_combined stacks
    3b directly above 3c.
    """
    unknown = sorted(set(df.arm) - set(A.BY_ID))
    assert not unknown, (
        f'{unknown} appear in the cached ladder but are not in igvf_arms.py. The cache '
        f'predates the registry -- rebuild it: python igvf_meeting_figures2_pipeline.py '
        f'{"borzoi" if grid == BORZOI_GRID else "enformer"}')
    # The registry decides what is drawn, exactly as it does for the bar panels, so a
    # `hidden` arm is scored and cached but never reaches a panel.
    df = df[df.arm.isin({a.id for a in A.arms_for(grid)})]
    fam = {a: A.BY_ID[a].family for a in df.arm.unique()}
    d_last = df[df.rung_i == (len(rungs) - 1 if order_by_rung == -1 else order_by_rung)]
    refs = [a for a in d_last.arm if fam[a] == 'reference']
    models = d_last[[fam[a] != 'reference' for a in d_last.arm]] \
        .sort_values('auroc', ascending=False).arm.tolist()
    order = list(dict.fromkeys(refs)) + models

    # Widened from 5.0/rung: the extra width buys real gaps between the rung
    # panels (see the w_pad below) instead of sitting unused as letterbox margin
    # in panel3_combined, which pads whichever of 3a/3b/3c is narrowest up to
    # its own target ratio anyway.
    fig, axes = plt.subplots(1, len(rungs), figsize=(6.0 * len(rungs), 5.9), sharey=True)
    for ax, (r_i, rung) in zip(np.atleast_1d(axes), enumerate(rungs)):
        d = df[df.rung_i == r_i].set_index('arm').loc[order].reset_index()
        bc = d['by_construction'].to_numpy() if 'by_construction' in d \
            else np.zeros(len(d), bool)
        y = np.arange(len(d))
        bars = ax.barh(y, d.auroc, height=0.66, color=[A.color(a) for a in d.arm], zorder=3,
                       xerr=np.vstack([np.nan_to_num(d.auroc - d.lo),
                                       np.nan_to_num(d.hi - d.auroc)]),
                       error_kw=dict(ecolor='#666666', lw=1.0, capsize=2.2))
        for bar, flag in zip(bars, bc):
            if flag:
                bar.set_hatch('///')
                bar.set_edgecolor('white')
        for yi, (v, h) in enumerate(zip(d.auroc, d.hi)):
            ax.text((v if np.isnan(h) else h) + 0.015, yi, f'{v:.3f}', va='center',
                    ha='left', fontsize=14, color=INK2)
        # Only rungs 0-1 are ordinary discrimination, where 0.5 is chance. On rungs
        # 2-3 the scale runs from "pure average" to "fully specific" and 0.5 is not
        # an interpretable point, so no guide is drawn.
        if r_i < 2:
            ax.axvline(0.5, ls='--', lw=1.1, color=INK2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([A.label(a, grid) for a in d.arm], fontsize=16)
        ax.set_ylim(len(order) - 0.5, -0.5)     # not invert_yaxis(): sharey would undo it
        ax.set_xlim(0, 1.30)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        # R2's cached subtitle ("cross-CT average points the wrong way") is display
        # text, not a threshold or a count, so it is safe to override here rather
        # than in the pipeline that wrote the cache.
        subtitle = 'cross-CT average disagrees' if r_i == 2 else rung['subtitle']
        ax.set_title(f"{rung['name']}\n{textwrap.fill(subtitle, 34)}\n"
                     f"{rung['n_pos']:,} pos / {rung['n_neg']:,} neg", fontsize=18, pad=10)
        ax.set_xlabel('AUROC', fontsize=13)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(axis='x', ls='--', lw=0.55, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    # Extra breathing room between the four rung panels -- there is width to spare,
    # especially once this feeds panel3_combined's wider canvas.
    fig.tight_layout(w_pad=4.5)
    # Pin the column geometry to the shared constants (see RUNG_COL_LEFT) rather
    # than leaving it at whatever tight_layout just chose, so 3a's columns land
    # in the same place as these when panel3_combined stacks them.
    fig.subplots_adjust(left=RUNG_COL_LEFT, right=RUNG_COL_RIGHT, wspace=RUNG_COL_WSPACE)
    # Title and footnotes centered on the actual column block, computed from the
    # axes' final positions rather than a literal x=0.5/0.24/0.76 -- those pinned
    # margins mean the columns no longer start near the canvas edge, so centering
    # on the canvas no longer centers on the panels.
    col_l, col_r = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((col_l + col_r) / 2, 1.055, grid_title, ha='center', fontsize=27.5,
             fontweight='bold', color=INK)
    if show_anchor_footnote:
        mid_lo = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        mid_hi = (axes[2].get_position().x0 + axes[-1].get_position().x1) / 2
        fig.text(mid_lo, -0.045, 'rungs 0-1:  ' + ANCHOR_LO, ha='center', fontsize=16,
                 color=INK2)
        fig.text(mid_hi, -0.045, 'rungs 2-3:  ' + ANCHOR_HI, ha='center', fontsize=16,
                 color=INK2)
    return fig


def panel3b_ladder_enformer():
    df = pd.read_parquet(_need('ladder_enformer.parquet'))
    rungs = json.loads((CACHE / 'rungs_enformer.json').read_text())
    # Footnote dropped here -- it's identical on 3c, and repeating it directly
    # underneath immediately above 3c's title is where they'd collide when
    # panel3_combined stacks the two.
    fig = _ladder_bars(df, rungs, ENFORMER_GRID,
                       'Classification of Enformer bins (128 bp)',
                       show_anchor_footnote=False)
    return save(fig, 'fig3b_ladder_enformer')


def panel3c_ladder_borzoi():
    df = pd.read_parquet(_need('ladder_borzoi.parquet'))
    rungs = json.loads((CACHE / 'rungs_borzoi.json').read_text())
    fig = _ladder_bars(df, rungs, BORZOI_GRID,
                       'Classification of Borzoi bins (32 bp)')
    return save(fig, 'fig3c_ladder_borzoi')


def panel3_combined():
    """Figures 3a, 3b and 3c stacked into one tall panel, one on top of the next.

    Composited as images rather than redrawn into a shared axes grid: 3a is a
    schematic on made-up data and 3b/3c are bar-chart grids on two different
    resolutions, so there is no shared x- or y-axis for a redraw to line up
    anyway, and each panel already reads fine at its own native layout. Each
    sub-figure is rendered through its own panel function exactly as it would be
    on its own, then pasted top to bottom at a shared width -- no explicit
    row-to-row alignment.

    Target ratio 15:12.25 (~1.224:1). The natural stack -- same width, heights
    summed -- comes out closer to square than that; rather than crush the
    content vertically to force the wider ratio (shrinking the text all three
    panels just had enlarged) or crop anything, the canvas is padded out
    horizontally with plain white margin until the ratio matches.
    """
    import io
    from PIL import Image
    from matplotlib.transforms import Bbox

    DPI = 300   # matches rcParams['savefig.dpi'] above, so save() below neither
                # upscales nor downscales what gets pasted in -- no resample blur.
    figs = [panel3a_schematic(), panel3b_ladder_enformer(), panel3c_ladder_borzoi()]
    imgs = []
    for f in figs:
        # NOT bbox_inches='tight': that crops each figure to its own content's
        # extent, and 3a's content (one short y-label) sits well inside its
        # pinned-but-otherwise-unneeded left margin while 3b/3c's (long model
        # names) fills theirs almost exactly -- so 'tight' trimmed a different
        # amount of "blank" margin off each one, and the RUNG_COL_LEFT alignment
        # (real, verified on the un-cropped figures) was gone by the time these
        # got pasted. Keep the full canvas width (0 to W) so no figure is cropped
        # horizontally at all -- identical crop (none) for all three -- and only
        # take the tight *vertical* extent, to still capture the title/footnote
        # text that intentionally sits outside the nominal [0,1] figure fraction.
        renderer = f.canvas.get_renderer()
        tbb_in = f.get_tightbbox(renderer)   # already in inches, not display/pixel units
        w_in, h_in = f.get_size_inches()
        full_bbox = Bbox.from_extents(0, tbb_in.y0, w_in, tbb_in.y1)
        buf = io.BytesIO()
        f.savefig(buf, format='png', dpi=DPI, bbox_inches=full_bbox, facecolor='white')
        plt.close(f)
        buf.seek(0)
        imgs.append(Image.open(buf).convert('RGB'))

    # Common width = the widest panel's own width; the other(s) scaled up to it
    # uniformly (same factor on both axes), so nothing gets stretched out of shape.
    common_w = max(im.width for im in imgs)
    scaled = [im.resize((common_w, round(im.height * common_w / im.width)), Image.LANCZOS)
             for im in imgs]
    # A small gap between A/B/C -- otherwise adjacent, since each was already
    # tight-cropped (vertically) to its own content in the export step above.
    gap_px = round(0.15 * DPI)
    content_h = sum(im.height for im in scaled) + gap_px * (len(scaled) - 1)

    # Pad out to the target ratio in whichever direction the natural stack falls
    # short -- width if it's proportionally too tall/narrow, height if it's
    # proportionally too wide -- rather than only ever widening. All three
    # sub-figures are the same width with no cropping now (see the bbox note
    # above), so which way this goes depends on their summed height alone.
    target_ratio = 15 / 12.25
    if common_w / content_h > target_ratio:
        canvas_w, canvas_h = common_w, round(common_w / target_ratio)
    else:
        canvas_w, canvas_h = round(content_h * target_ratio), content_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    y = (canvas_h - content_h) // 2
    for i, im in enumerate(scaled):
        canvas.paste(im, ((canvas_w - im.width) // 2, y))
        y += im.height
        if i < len(scaled) - 1:
            y += gap_px

    fig, ax = plt.subplots(figsize=(canvas_w / DPI, canvas_h / DPI), dpi=DPI)
    ax.imshow(np.asarray(canvas))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return save(fig, 'fig3_combined')


# ============================================================================
# FIGURE 3a - the slide version of the ladder's four strata. One continuous made-up
# track per panel, banded by value: above the active threshold is the positive
# class, below the silent threshold is the negative class, the band between is
# neither. Each panel then greys out whatever its extra criterion removes, so the
# narrowing is the only thing that moves across the row.
#
# No bin counts, no thresholds spelled out, no formulae -- the presenter says those.
# `panel3a_detail()` is the version that does spell them out.
# ============================================================================
SIMPLE_POS = '#2E7D32'     # positive class
SIMPLE_NEG = '#C62828'     # negative class
SIMPLE_MID = '#C4C4CA'     # neither: between the two thresholds
SIMPLE_DROP = 0.13         # alpha for a region this rung has excluded

S_ACTIVE, S_SILENT = 1.0, 0.34
S_TSS = (2.0, 3.5, 5.0, 6.5)     # the peak at 8.0 sits away from any TSS
S_TSSPAD = 0.62


def _simple_tracks(npt=1400):
    """Observed, cross-CT average and accessibility as smooth invented curves.

    Shaped so each rung removes something visible: the 8.0 peak has no TSS, the
    5.0 peak is generic, the 3.5 peak is not accessible, and the 6.5 trough is a
    silent promoter the average expects to be active.
    """
    x = np.linspace(0, 10, npt)
    g = lambda c, h, w: h * np.exp(-0.5 * ((x - c) / w) ** 2)
    true = 0.07 + g(2.0, 2.7, 0.16) + g(3.5, 2.3, 0.15) + g(5.0, 3.1, 0.15) \
        + g(8.0, 1.9, 0.18) + 0.04 * np.sin(3.1 * x)
    avg = 0.06 + g(5.0, 2.4, 0.20) + g(6.5, 1.9, 0.22) + g(8.0, 0.9, 0.25) \
        + 0.03 * np.sin(2.3 * x)
    acc = 0.05 + g(2.0, 1.0, 0.22) + g(5.0, 0.8, 0.22) + g(8.0, 0.7, 0.25)
    return x, np.clip(true, 0, None), np.clip(avg, 0, None), np.clip(acc, 0, None)


def _simple_classes(x, true, avg, acc):
    """(positive, negative) masks per rung -- the same shape of rule as the real ladder.

    Returned per position rather than per value band, because a band painted on
    value alone would colour the base of a tall peak as a negative. At R0 that is
    harmless; from R2 on it would say a peak the rung kept is also a negative.
    """
    near_tss = np.zeros_like(x, bool)
    for t in S_TSS:
        near_tss |= np.abs(x - t) <= S_TSSPAD
    hot, cold = true >= S_ACTIVE, true < S_SILENT
    avg_lo, avg_hi = avg < 0.35, avg >= 0.80
    acc_open = acc >= 0.45
    r0 = (hot, cold)
    r1 = (near_tss & hot, near_tss & cold)
    r2 = (r1[0] & avg_lo, r1[1] & avg_hi)
    r3 = (r2[0] & acc_open, r2[1] & ~acc_open)
    return [r0, r1, r2, r3]


def panel3a_schematic():
    """The four strata as one track per panel, with class shown as a strip beneath it.

    The track itself stays neutral. Colouring the area under the curve competes with
    the average and accessibility lines drawn over it, and it forces a choice between
    banding by value (which paints the base of a tall peak red) and banding by class
    (which leaves a hole under every peak). A strip under the axis has neither
    problem: every position gets the same area regardless of its value, and the
    curves above it stay readable.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    x, true, avg, acc = _simple_tracks()
    classes = _simple_classes(x, true, avg, acc)
    titles = [('R0 · all bins', 'every bin'),
              ('R1 · promoters', 'only near a TSS'),
              ('R2 · + disagreement', 'and the average disagrees'),
              ('R3 · + accessibility', 'and it is also accessible')]
    STRIP_LO, STRIP_HI = -0.36, -0.06
    GENE_Y = -0.82
    ACC_LO, ACC_HI = -1.90, -1.20
    TRACK_FILL, TRACK_LINE = '#E4E4E9', '#55555D'

    fig, axes = plt.subplots(1, 4, figsize=(24.0, 4.9), sharey=True)
    for col, (ax, (pos, neg), (name, sub)) in enumerate(zip(axes, classes, titles)):
        # the observed track, neutral so the overlaid curves stay legible
        ax.fill_between(x, 0, true, color=TRACK_FILL, lw=0, zorder=2)
        ax.plot(x, true, lw=1.1, color=TRACK_LINE, zorder=4)
        ax.axhline(S_SILENT, ls='--', lw=0.8, color='#A0A0A8', zorder=3)

        # the class strip, immediately under the axis
        ax.fill_between(x, STRIP_LO, STRIP_HI, color='#F1F1F4', lw=0, zorder=2)
        ax.fill_between(x, STRIP_LO, STRIP_HI, where=pos, color=SIMPLE_POS, lw=0,
                        interpolate=True, zorder=3)
        ax.fill_between(x, STRIP_LO, STRIP_HI, where=neg, color=SIMPLE_NEG, lw=0,
                        interpolate=True, zorder=3)

        if col >= 1:                                   # the gene, with its TSSs
            ax.plot([0.15, 9.85], [GENE_Y, GENE_Y], lw=1.3, color='#444444',
                    solid_capstyle='butt', zorder=3)
            for t in S_TSS:
                ax.plot([t, t], [GENE_Y, GENE_Y + 0.30], lw=1.3, color='#444444', zorder=4)
                ax.annotate('', xy=(t + 0.42, GENE_Y + 0.30), xytext=(t, GENE_Y + 0.30),
                            zorder=4, arrowprops=dict(arrowstyle='-|>', lw=1.3,
                                                      color='#444444'))
        if col >= 2:                                   # the generic expectation
            # Drawn on R3 as well as R2. R3 is built on R2, so it still requires the
            # average to disagree -- accessibility is an extra term, not a
            # replacement. Showing it only on R2 makes the criterion look dropped.
            ax.plot(x, avg, lw=1.8, ls=(0, (5, 2.5)), color='#3F5B8C', zorder=5)
        if col == 3:                                   # accessibility, its own strip
            a = ACC_LO + (acc / acc.max()) * (ACC_HI - ACC_LO)
            ax.fill_between(x, ACC_LO, a, color=SCHEM_ACC, alpha=0.30, lw=0, zorder=2)
            ax.plot(x, a, lw=1.4, color=SCHEM_ACC, zorder=5)
            # Named directly on its own strip, the way 'class' is on col 0 -- one
            # fewer entry the legend has to carry. Underneath rather than to the
            # side: a caption under the strip it names, not a row label beside it.
            ax.text(5.0, ACC_LO - 0.30, 'accessibility', fontsize=13, ha='center',
                    va='top', color=INK2, clip_on=False)

        ax.set_title(f'{name}\n{sub}', fontsize=17.5, pad=12, color=INK)
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.05, 3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('CAGE signal', fontsize=18, color='black', labelpad=6)

    # Extra breathing room between the four rung panels, same as B/C.
    fig.tight_layout(w_pad=4.5)
    # Pin the column geometry to the same shared constants B/C use, so this
    # panel's columns land in the same place when panel3_combined stacks them.
    fig.subplots_adjust(left=RUNG_COL_LEFT, right=RUNG_COL_RIGHT, wspace=RUNG_COL_WSPACE)
    # Centered on the actual column block (computed from final axes positions),
    # not a literal x=0.5 -- the pinned left margin means the columns start well
    # right of the canvas edge, so canvas-centered text reads as off-center above
    # them.
    col_l, col_r = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((col_l + col_r) / 2, 1.02, 'What each rung selects', ha='center',
             fontsize=29, fontweight='bold', color=INK)

    handles = [Patch(color=SIMPLE_POS, label='positive class'),
               Patch(color=SIMPLE_NEG, label='negative class'),
               Patch(color='#F1F1F4', label='neither'),
               Line2D([0], [0], color=TRACK_LINE, lw=1.1, label='observed'),
               Line2D([0], [0], color='#3F5B8C', lw=1.8, ls=(0, (5, 2.5)),
                      label='cross-cell-type average')]
    # Bottom left, at the same height as the accessibility strip -- convert that
    # strip's data-space y (shared by every axis via sharey) into a figure fraction,
    # computed after tight_layout so it reflects where the axes actually landed,
    # rather than guessing an offset that would drift if the layout ever changes.
    disp_y = axes[0].transData.transform((0, (ACC_LO + ACC_HI) / 2))[1]
    acc_y = fig.transFigure.inverted().transform((0, disp_y))[1]
    # One row, no stacking -- ncol = every handle. Left edge lines up with column
    # 1's own left edge (RUNG_COL_LEFT), not the canvas edge.
    fig.legend(handles=handles, loc='center left', ncol=len(handles), frameon=False,
               fontsize=16, bbox_to_anchor=(RUNG_COL_LEFT, acc_y))
    return save(fig, 'fig3a_schematic')


# ============================================================================
# FIGURE 3a - what the four rungs actually select. A schematic on invented data,
# drawn with the SAME boolean logic as the pipeline builds the real rungs with,
# so the picture cannot drift from what it claims to explain.
#
# Nothing here is cached: it is ~30 numbers and a few hundred drawing calls.
# ============================================================================
SCHEM_POS = '#0F766E'     # in the positive class at this rung
SCHEM_NEG = '#9D174D'     # in the negative class at this rung
SCHEM_OUT = '#DCDCE0'     # in neither class -- excluded, not "negative"
SCHEM_AVG = '#6B7280'     # cross-CT average: a reference, so the reference grey
SCHEM_ACC = '#8c564b'     # accessibility, the same brown the locus panels use

# Invented 32-bin window. Values are chosen so every rung keeps some bins and
# drops others, which is the whole point of the ladder.
SCHEM_TRUE = np.array([0.05, 0.10, 0.03, 0.08, 2.60, 0.06, 0.04, 0.55, 0.09, 0.05,
                       0.07, 3.10, 0.12, 0.04, 0.06, 0.62, 0.03, 0.11, 2.20, 0.08,
                       0.05, 0.48, 0.07, 0.09, 1.90, 0.06, 0.04, 0.10, 0.05, 2.40,
                       0.08, 0.03])
SCHEM_AVG_V = np.array([0.10, 0.12, 0.08, 0.09, 1.40, 0.11, 0.07, 0.30, 0.10, 0.08,
                        1.60, 0.12, 0.09, 0.06, 0.10, 0.28, 0.05, 0.13, 0.14, 0.10,
                        0.08, 0.26, 0.09, 0.11, 1.30, 1.50, 0.07, 0.12, 0.06, 0.90,
                        0.10, 0.05])
SCHEM_ACC_V = np.array([0.10, 0.08, 0.12, 0.15, 0.90, 0.11, 0.09, 0.20, 0.10, 0.07,
                        0.14, 1.10, 0.13, 0.08, 0.10, 0.22, 0.06, 0.12, 0.18, 0.09,
                        0.07, 0.19, 0.11, 0.13, 0.85, 0.95, 0.08, 0.10, 0.09, 0.80,
                        0.12, 0.06])
SCHEM_TSS = [4, 11, 18, 24]        # bin index of each protein-coding TSS
SCHEM_TSSPAD = 1                   # +/- 1 bin stands in for the real +/- 500 bp

# Toy stand-ins for the real cuts. On the Enformer grid these are 1.0 / 0.2 for
# the noise floor, the 50th and 90th percentile of AVG over promoter bins, and
# the 95th percentile of ATAC over allowed bins.
SCHEM_ACTIVE, SCHEM_SILENT = 1.0, 0.2
SCHEM_AVG_LO, SCHEM_AVG_HI = 0.20, 0.80
SCHEM_ACC_OPEN = 0.50


def _schem_rungs():
    """The four rungs on the toy data, expression for expression as the pipeline."""
    tv, av, ac = SCHEM_TRUE, SCHEM_AVG_V, SCHEM_ACC_V
    ts = np.zeros(len(tv), bool)
    for t in SCHEM_TSS:
        ts[max(0, t - SCHEM_TSSPAD):t + SCHEM_TSSPAD + 1] = True
    lo_avg, hi_avg = av < SCHEM_AVG_LO, av >= SCHEM_AVG_HI

    r0p, r0n = tv >= SCHEM_ACTIVE, tv < SCHEM_SILENT
    r1p, r1n = ts & r0p, ts & r0n
    r2p, r2n = r1p & lo_avg, r1n & hi_avg
    r3p, r3n = r2p & (ac >= SCHEM_ACC_OPEN), r2n & (ac < SCHEM_ACC_OPEN)
    return ts, [(r0p, r0n), (r1p, r1n), (r2p, r2n), (r3p, r3n)]


SCHEM_TITLES = [
    ('R0 · all bins', 'any allowed bin',
     'pos:  observed ≥ active\nneg:  observed < silent\nbins in between: neither class'),
    ('R1 · promoters', 'protein-coding TSS only',
     'both classes restricted to bins\nwithin 500 bp of a TSS'),
    ('R2 · + disagreement', 'cross-CT average points the wrong way',
     'pos:  + average in the bottom half\nneg:  + average in the top decile'),
    ('R3 · + accessibility', 'ATAC-confirmed',
     'pos:  + ATAC ≥ p95  (open)\nneg:  + ATAC < p95  (closed)'),
]


def panel3a_detail():
    """What each rung selects, on invented data. Sits above the two real ladders.

    One row is added per panel, top to bottom, mirroring the fact that each rung
    adds exactly one term to the one on its left.

    The `Class` strip is the readout, not the bar colours. Negatives are silent
    bins -- their observed value is ~0.05 against peaks of ~3 -- so on a linear
    axis a coloured bar for a negative is a couple of pixels tall and invisible.
    The strip gives every bin the same area regardless of its value, which is the
    only way both classes are legible in one picture. Pale grey means excluded at
    this rung, which is a different thing from being in the negative class.
    """
    from matplotlib.patches import Patch
    ts, rungs = _schem_rungs()
    n = len(SCHEM_TRUE)
    x = np.arange(n)
    # (row label, first rung at which the row appears). A spacer row carries the
    # per-rung criteria text so it lands at the same height in all four columns.
    rows = [('Observed\n(GM12878)', 0), ('Class', 0), ('', 0), ('Genes', 1),
            ('Cross-CT\naverage', 2), ('ATAC', 3)]
    SPACER = 2

    def thresh(ax, y, text, show, ls='--'):
        ax.axhline(y, ls=ls, lw=0.9, color='#555555', zorder=4)
        if show:
            ax.text(-0.4, y, text, fontsize=6.5, va='bottom', ha='left', color='#555555',
                    zorder=6, bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.85))

    fig, axes = plt.subplots(len(rows), 4, figsize=(20.0, 7.4),
                             gridspec_kw=dict(
                                 height_ratios=[2.0, 0.34, 0.62, 0.42, 1.25, 1.25],
                                 hspace=0.22, wspace=0.10))
    for col, ((pos, neg), (name, sub, crit)) in enumerate(zip(rungs, SCHEM_TITLES)):
        colors = np.where(pos, SCHEM_POS, np.where(neg, SCHEM_NEG, SCHEM_OUT))
        for row, (rname, first_col) in enumerate(rows):
            ax = axes[row, col]
            if row == SPACER:
                ax.set_axis_off()
                ax.text(0.0, 0.92, crit, transform=ax.transAxes, fontsize=8.2, va='top',
                        ha='left', color=INK2, linespacing=1.55)
                continue
            if col < first_col:
                ax.set_visible(False)
                continue
            ax.set_xlim(-0.7, n - 0.3)
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=7, length=2)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if col >= 1:                       # the TSS windows, once R1 imposes them
                for t in SCHEM_TSS:
                    ax.axvspan(t - SCHEM_TSSPAD - 0.5, t + SCHEM_TSSPAD + 0.5,
                               color='#EEF2F7', zorder=0)
            if rname.startswith('Observed'):
                ax.bar(x, SCHEM_TRUE, width=0.72, color=colors, zorder=3)
                thresh(ax, SCHEM_ACTIVE, 'active', col == 0)
                thresh(ax, SCHEM_SILENT, 'silent', col == 0, ls=':')
                ax.set_ylim(0, 3.6)
            elif rname == 'Class':
                ax.bar(x, np.ones(n), width=0.9, color=colors, zorder=3)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                for sp in ('left', 'bottom'):
                    ax.spines[sp].set_visible(False)
            elif rname == 'Genes':
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                for sp in ('left', 'bottom'):
                    ax.spines[sp].set_visible(False)
                ax.axhline(0.35, lw=1.2, color='#444444', zorder=2)
                for t in SCHEM_TSS:
                    ax.plot([t, t], [0.35, 0.92], lw=1.2, color='#444444', zorder=3)
                    ax.plot([t + 0.9], [0.92], marker='>', ms=5.5, color='#444444', zorder=3)
                    ax.plot([t, t + 0.9], [0.92, 0.92], lw=1.2, color='#444444', zorder=3)
            elif rname.startswith('Cross-CT'):
                ax.bar(x, SCHEM_AVG_V, width=0.72, color=SCHEM_AVG, zorder=3)
                thresh(ax, SCHEM_AVG_HI, 'top decile', col == 2)
                thresh(ax, SCHEM_AVG_LO, 'bottom half', col == 2, ls=':')
                ax.set_ylim(0, 2.0)
            else:
                ax.bar(x, SCHEM_ACC_V, width=0.72, color=SCHEM_ACC, zorder=3)
                thresh(ax, SCHEM_ACC_OPEN, 'open (p95)', col == 3)
                ax.set_ylim(0, 1.35)
            if col == first_col:
                ax.set_ylabel(rname, fontsize=8, rotation=0, ha='right', va='center',
                              labelpad=8, color=INK2)
        top = axes[0, col]
        top.set_title(f'{name}\n{sub}\n{int(pos.sum())} pos / {int(neg.sum())} neg',
                      fontsize=10, pad=10)

    fig.text(0.5, 1.04, 'What each rung selects — schematic, invented data',
             ha='center', fontsize=13.5, fontweight='bold', color=INK)
    fig.text(0.5, 1.005,
             'Each panel adds exactly one criterion to the one on its left, and every '
             'panel is a strict subset of it. Read the Class strip: pale grey is '
             'excluded at that rung, which is not the same as being a negative.',
             ha='center', fontsize=10, color=INK2)
    fig.legend(handles=[Patch(color=SCHEM_POS, label='positive class'),
                        Patch(color=SCHEM_NEG, label='negative class'),
                        Patch(color=SCHEM_OUT, label='in neither class at this rung')],
               loc='lower center', ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.04))
    return save(fig, 'fig3a_detail')


# ============================================================================
# FIGURE 4 - accuracy against how generic the region is. comparing_model4 §13.
# Running medians only: the background scatter and the ringed marker region are
# dropped, so the panel carries the trend and nothing else.
# ============================================================================
FIG4_ARMS = ('enformer_pre', 'enformer_n1_1536', 'mamba_1ct', 'alphagenome')


def _running_median(x, y, nbins=12):
    ok = ~np.isnan(x) & ~np.isnan(y)
    xs, ys = x[ok], y[ok]
    edges = np.unique(np.nanpercentile(xs, np.linspace(0, 100, nbins + 1)))
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xs >= a) & (xs <= b if b == edges[-1] else xs < b)
        if m.sum() >= 5:
            cx.append(np.median(xs[m]))
            cy.append(np.median(ys[m]))
    return np.array(cx), np.array(cy)


def panel4_generic(metric='sm', arms=FIG4_ARMS, scatter=False, nbins=12):
    """y = R(prediction, TRUE GM12878) against x = R(TRUE, cross-CT average).

    x rising means the region's GM12878 signal looks more like the average of
    every other cell type -- more generic. A model that is really predicting
    cell-type-specific signal should not need the region to be generic.
    """
    df = pd.read_parquet(_need('enformer_generic.parquet'))
    x = df[f'x_{metric}'].to_numpy()

    fig, ax = plt.subplots(figsize=(15.0, 5.076))
    handles = []
    for a in arms:
        yy = df[f'y_{metric}_{a}'].to_numpy()
        if scatter:
            ax.scatter(x, yy, s=4, alpha=0.13, color=A.color(a), edgecolors='none')
        cx, cy = _running_median(x, yy, nbins)
        ax.plot(cx, cy, '-o', color=A.color(a), lw=2.5, ms=5, markeredgecolor='white',
                markeredgewidth=0.6, zorder=6)
        handles.append(Line2D([0], [0], color=A.color(a), lw=2.5, marker='o', ms=5,
                              markeredgecolor='white', markeredgewidth=0.6,
                              label=f'{A.label(a, ENFORMER_GRID)}   '
                                    f'median {np.nanmedian(yy):.2f}'))
    ax.set_xlim(0, 1)
    ax.set_xlabel('R(TRUE GM12878, cross-CT average)   →  region more generic',
                  fontsize=18)
    ax.set_ylabel('R(prediction, TRUE GM12878)', fontsize=18)
    ax.set_title(f'Accuracy vs how generic the region is — {METRIC_LABEL[metric]}\n'
                 f'{len(df):,} regions, running medians over {nbins} equal-count bins',
                 fontsize=20, pad=14, loc='center')
    ax.legend(handles=handles, fontsize=14, loc='upper left')
    tidy(ax, grid='both')
    fig.tight_layout()
    return save(fig, f'fig4_generic_{metric}')


# ============================================================================
# FIGURE 5 - dsQTL by peak specificity. Table built by dsQTL_benchmark8;
# this only recolours and relabels it.
# ============================================================================
def panel5_dsqtl(definition='merged', xmin=0.30):
    """Two strata, same variants, same metric: low- vs high-specificity peaks."""
    tab = pd.read_parquet(_need('dsqtl_stratified.parquet'))
    meta = json.loads((CACHE / 'dsqtl_stratified.json').read_text())
    median = meta['median_breadth'][definition]
    arms = [a.id for a in A.arms_for(DSQTL)]
    sub = tab[tab.stratum.isin(['low_specificity', 'high_specificity'])]
    assert sub.AP.min() >= xmin, 'A score falls below the displayed axis; lower xmin.'
    xmax = np.ceil((sub.AP.max() + 0.05) * 20) / 20

    y = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 9.11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.265, right=0.97, top=0.79, bottom=0.14, wspace=0.12)
    for ax, group, ttl, rel in zip(axes, ['low_specificity', 'high_specificity'],
                                   ['Low specificity', 'High specificity'], ['>', '≤']):
        d = tab[(tab.definition == definition) & (tab.stratum == group)]
        assert not d.arm.duplicated().any()
        d = d.set_index('arm').loc[arms]
        assert d.AP.notna().all()
        ax.barh(y, d.AP.to_numpy() - xmin, left=xmin, height=0.60,
                color=[A.color(a) for a in arms], edgecolor='none', zorder=3)
        for row, v in enumerate(d.AP):
            ax.text(v + 0.008, row, f'{v:.3f}', va='center', ha='left', fontsize=13,
                    color='#333333')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(y) - 0.4, -0.8)
        ax.set_yticks(y, [A.label(a, DSQTL) for a in arms])
        ax.tick_params(axis='y', length=0, pad=12, labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.grid(axis='x', color='#E9E9E9', linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.set_title(ttl, loc='left', fontsize=19, weight='semibold', pad=42)
        ax.text(0, 1.025, f'{rel}{median} tracks  ·  n = {int(d.n.iloc[0]):,}  ·  '
                          f'{int(d.positives.iloc[0])} dsQTLs',
                transform=ax.transAxes, ha='left', va='bottom', color='#666666', fontsize=13)
        ax.set_xlabel('AUPRC (average precision)', fontsize=15, labelpad=10)
    fig.text(0.265, 0.925, 'dsQTL prediction by peak specificity', fontsize=22,
             weight='semibold', color=INK, ha='left')
    return save(fig, f'fig5_dsqtl_{definition}')


# ============================================================================
# FIGURE 6 - LCL eQTL variant effect by distance to TSS. Poster Panel D.
# ============================================================================
def panel6_eqtl():
    """AUPRC in three distance bins, common support only.

    Accessibility is held fixed between ref and alt, so whatever separates the
    arms here is the sequence channel. The distal facet is the one that carries
    the specificity argument.
    """
    df = pd.read_parquet(_need('eqtl_paneld.parquet'))
    meta = json.loads((CACHE / 'eqtl_paneld.json').read_text())
    binnames, chance, ns = meta['binnames'], meta['chance'], meta['n']

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.973))
    for bi, (ax, nm) in enumerate(zip(axes, binnames)):
        sub = df.sort_values(nm, ascending=False)
        y = np.arange(len(sub))
        ax.barh(y, sub[nm], height=0.68, color=[A.color(a) for a in sub.arm], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([A.label(a, EQTL) for a in sub.arm], fontsize=15.5)
        ax.invert_yaxis()
        for yi, v in enumerate(sub[nm]):
            ax.text(v + 0.008, yi, f'{v:.3f}', va='center', ha='left', fontsize=12.5, color=INK2)
        ax.axvline(chance[bi], ls=':', lw=1.4, color=INK2, zorder=4)
        ax.text(chance[bi], 0.01, f' chance {chance[bi]:.2f}', fontsize=9, color=INK2,
                ha='left', va='bottom', transform=ax.get_xaxis_transform())
        ax.set_xlim(0, min(1.0, sub[nm].max() * 1.30))
        ax.set_title(f"{nm.replace(chr(10), ', ')}, n={ns[bi]}", fontsize=15)
        ax.tick_params(axis='x', labelsize=13)
        tidy(ax, grid='x')
        ax.spines['left'].set_visible(False)
    axes[1].set_xlabel('AUPRC  (average precision, |logFC|)', fontsize=16)
    fig.text(0.5, 1.015, 'LCL eQTL variant effect, by distance to TSS', ha='center',
             fontsize=23, fontweight='bold', color=INK)
    fig.tight_layout()
    return save(fig, 'fig6_eqtl')


PANELS = [panel1_locus, panel1b_locus_retrained, panel2a_accuracy_enformer,
          panel2b_accuracy_borzoi, panel3a_schematic, panel3a_detail,
          panel3b_ladder_enformer, panel3c_ladder_borzoi, panel4_generic,
          panel5_dsqtl, panel6_eqtl]
