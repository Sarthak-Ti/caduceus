"""General-purpose genomic track plotting utility.

Plot many signal tracks together (multiple models + ground truth + arbitrary extra
tracks) with an optional IGV/UCSC-style gene track underneath, all aligned on a shared
genomic-coordinate x-axis.

The utility is deliberately decoupled from the eval classes: it consumes generic
``Track`` objects (a 1-D numpy array plus display/coordinate metadata). You extract the
arrays from your eval objects yourself and wrap them in ``Track``s. This keeps it usable
for any signal, not just model predictions.

--------------------------------------------------------------------------------
Pairing with ``evals/evals_utils_joint.py`` (documentation only -- not imported here)
--------------------------------------------------------------------------------
The ``Evals`` class returns per-region tensors. To turn one region/channel into tracks::

    out = evals(idx)                      # idx = ct * num_regions + region
    pred  = out[1].squeeze(0).detach().cpu().float().numpy()   # (L, C) predicted (acc head)
    truth = out[4].squeeze(0).detach().cpu().float().numpy()   # (L, C) ground truth (expr)
    ch = 0                                                      # channel / cell type column

    chrom, start, end = evals.dataset.sequences.iloc[region][:3]
    bin_size = evals.dataset.pool                              # bp per output bin
    # predicted window is the center of the bed region; crop_output bins removed per side:
    crop_bp = evals.dataset.crop_output * bin_size
    pred_start = int(start) + crop_bp

    tracks = [
        Track(pred[:, ch],  name="Mamba",  color="#2ca02c",
              chrom=chrom, start=pred_start, bin_size=bin_size),
    ]
    truth_track = Track(truth[:, ch], name="True", chrom=chrom, start=pred_start,
                        bin_size=bin_size, is_truth=True)
    plot_tracks(tracks, genes=ann, ground_truth=truth_track,
                region=(chrom, pred_start, pred_start + len(pred) * bin_size))
"""

import os
import re
from dataclasses import dataclass

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Default GENCODE annotation used when a gene track is requested without an explicit path.
GENCODE_V49 = "/data1/lesliec/sarthak/data/DE_danwei/gencode.v49.annotation.gtf"

# Consistent palette carried over from code_test/comparing_model.ipynb so figures made
# with this utility match the existing notebook figures.
DEFAULT_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#17becf"]
TRUTH_COLOR = "gray"

# GTF columns (standard 9-column spec).
_GTF_COLS = ["chrom", "source", "feature", "start", "end",
             "score", "strand", "frame", "attribute"]


# --------------------------------------------------------------------------------------
# GTF parsing helpers (same idiom as /data1/lesliec/sarthak/data/DE_danwei/build_rna_info.py)
# --------------------------------------------------------------------------------------
def _get_attr(attr_str, key):
    """Pull a quoted GTF attribute value, e.g. gene_name "FOO" -> "FOO"."""
    m = re.search(fr'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None


def unversion(gene_id):
    """ENSG00000123456.7 -> ENSG00000123456."""
    return gene_id.split(".")[0] if gene_id else gene_id


@dataclass
class Track:
    """A single plottable signal.

    values     : 1-D signal array, shape (n_bins,).
    name       : label shown on the axis.
    color      : matplotlib color; None -> assigned from DEFAULT_COLORS.
    style      : "line" (predictions) or "fill" (filled area, e.g. ground truth).
    alpha      : opacity.
    ylim       : explicit (lo, hi) y-limits; None -> auto.
    chrom/start/bin_size : optional coordinate metadata. When present on all tracks
                 (or a `region` is passed to plot_tracks) the x-axis is drawn in genomic
                 coordinates. `start` is the 0-based genomic coord of the FIRST bin;
                 `bin_size` is bp per bin. Missing on any track -> bin-index fallback.
    is_truth   : mark this track as ground truth (styled by the ground-truth helper).
    overlay_group : tracks sharing the same non-None id are drawn on one shared axis.
    reference_exempt : skip the ground-truth dashed max-reference line AND the truth-max
                 y-limit anchoring on this track's axis. Use for tracks on a different
                 assay/scale from the ground truth (e.g. an accessibility track).
    """
    values: np.ndarray
    name: str = ""
    color: str = None
    style: str = "line"
    alpha: float = 0.85
    ylim: tuple = None
    chrom: str = None
    start: int = None
    bin_size: int = None
    is_truth: bool = False
    overlay_group: int = None
    reference_exempt: bool = False

    def __post_init__(self):
        self.values = np.asarray(self.values).squeeze()
        if self.values.ndim != 1:
            raise ValueError(
                f"Track '{self.name}' values must be 1-D after squeeze, "
                f"got shape {self.values.shape}. Select a single channel first."
            )

    @property
    def has_coords(self):
        return self.start is not None and self.bin_size is not None

    def x_coords(self):
        """x position (bin centers in genomic coords) if coords set, else bin indices."""
        n = len(self.values)
        if self.has_coords:
            return self.start + (np.arange(n) + 0.5) * self.bin_size
        return np.arange(n)


class GeneAnnotation:
    """Parse a GENCODE GTF into a per-transcript/exon table and query it by region.

    The full GTF is large (~3 GB), so on first use the needed feature rows are parsed
    (in chunks) and written to a compact parquet cache next to the GTF. Subsequent
    instantiations read the parquet directly. The parsed frame is kept on the instance,
    so repeated ``query`` calls in a session do not re-read anything.
    """

    def __init__(self, gtf_path=GENCODE_V49, gene_types=("protein_coding",),
                 feature_types=("exon", "transcript"), cache_parquet=True,
                 cache_dir=None):
        self.gtf_path = gtf_path
        self.gene_types = tuple(gene_types) if gene_types else None
        self.feature_types = tuple(feature_types)

        cache_path = self._cache_path(cache_dir)
        if cache_parquet and cache_path and os.path.exists(cache_path):
            self.df = pd.read_parquet(cache_path)
        else:
            self.df = self._parse_gtf()
            if cache_parquet and cache_path:
                try:
                    self.df.to_parquet(cache_path, index=False)
                except OSError:
                    pass  # cache is best-effort; parsing still succeeded

    def _cache_path(self, cache_dir):
        tag = "_".join(self.feature_types)
        if self.gene_types:
            tag += "__" + "_".join(self.gene_types)
        base = os.path.basename(self.gtf_path) + f".{tag}.parquet"
        if cache_dir:
            return os.path.join(cache_dir, base)
        gtf_dir = os.path.dirname(self.gtf_path)
        if os.access(gtf_dir, os.W_OK):
            return os.path.join(gtf_dir, base)
        return None  # directory not writable and no cache_dir given -> skip caching

    def _parse_gtf(self):
        attr_keys = ["gene_id", "transcript_id", "gene_name",
                     "gene_type", "transcript_type"]
        keep = []
        reader = pd.read_csv(
            self.gtf_path, sep="\t", comment="#", header=None,
            names=_GTF_COLS, dtype={"chrom": str}, chunksize=1_000_000,
        )
        for chunk in reader:
            chunk = chunk[chunk["feature"].isin(self.feature_types)]
            if chunk.empty:
                continue
            for key in attr_keys:
                chunk[key] = chunk["attribute"].apply(lambda s, k=key: _get_attr(s, k))
            if self.gene_types:
                chunk = chunk[chunk["gene_type"].isin(self.gene_types)]
            if chunk.empty:
                continue
            # GTF is 1-based fully-closed [start, end] -> 0-based half-open [start-1, end).
            chunk = chunk.assign(start0=chunk["start"].astype(int) - 1,
                                 end0=chunk["end"].astype(int))
            keep.append(chunk[["chrom", "feature", "start0", "end0", "strand",
                               "gene_id", "transcript_id", "gene_name"]])
        if not keep:
            return pd.DataFrame(columns=["chrom", "feature", "start0", "end0", "strand",
                                         "gene_id", "transcript_id", "gene_name"])
        return pd.concat(keep, ignore_index=True)

    def query(self, chrom, start, end, collapse=False):
        """Return transcript records overlapping [start, end) on chrom.

        Each record: {gene_name, gene_id, transcript_id, strand, tx_start, tx_end,
                      exons: [(s, e), ...]}.
        collapse=True merges exons across a gene's transcripts into one record per gene.
        """
        sub = self.df[(self.df["chrom"] == chrom)
                      & (self.df["start0"] < end)
                      & (self.df["end0"] > start)]
        if sub.empty:
            return []

        exons = sub[sub["feature"] == "exon"]
        tx = sub[sub["feature"] == "transcript"]

        # Per-gene splice-variability (alt polyA / TSS / internal splice sites). Computed
        # once per gene and attached to every record of that gene; drawn as ticks under the
        # gene track. Most meaningful on the collapsed (one-model-per-gene) view.
        sv_by_gene = {gid: _splice_variability(gg, gg["strand"].iloc[0])
                      for gid, gg in sub.groupby("gene_id")}

        if collapse:
            records = []
            for gid, g in sub.groupby("gene_id"):
                g_ex = g[g["feature"] == "exon"]
                ex = _merge_intervals(list(zip(g_ex["start0"], g_ex["end0"])))
                if ex:
                    tx_start, tx_end = min(s for s, _ in ex), max(e for _, e in ex)
                else:
                    tx_start, tx_end = int(g["start0"].min()), int(g["end0"].max())
                records.append({
                    "gene_name": g["gene_name"].iloc[0], "gene_id": gid,
                    "transcript_id": None, "strand": g["strand"].iloc[0],
                    "tx_start": tx_start, "tx_end": tx_end, "exons": ex,
                    "splice_variability": sv_by_gene.get(gid),
                })
            return sorted(records, key=lambda r: r["tx_start"])

        records = []
        for tid, t_ex in exons.groupby("transcript_id"):
            ex = sorted(zip(t_ex["start0"].astype(int), t_ex["end0"].astype(int)))
            row = tx[tx["transcript_id"] == tid]
            if not row.empty:
                tx_start, tx_end = int(row["start0"].iloc[0]), int(row["end0"].iloc[0])
            else:
                tx_start, tx_end = ex[0][0], ex[-1][1]
            records.append({
                "gene_name": t_ex["gene_name"].iloc[0],
                "gene_id": t_ex["gene_id"].iloc[0],
                "transcript_id": tid, "strand": t_ex["strand"].iloc[0],
                "tx_start": tx_start, "tx_end": tx_end, "exons": ex,
                "splice_variability": sv_by_gene.get(t_ex["gene_id"].iloc[0]),
            })
        # transcripts with no exon rows in-window (e.g. a single long intron spans it)
        for tid, row in tx.groupby("transcript_id"):
            if tid in exons["transcript_id"].values:
                continue
            records.append({
                "gene_name": row["gene_name"].iloc[0],
                "gene_id": row["gene_id"].iloc[0],
                "transcript_id": tid, "strand": row["strand"].iloc[0],
                "tx_start": int(row["start0"].iloc[0]), "tx_end": int(row["end0"].iloc[0]),
                "exons": [],
                "splice_variability": sv_by_gene.get(row["gene_id"].iloc[0]),
            })
        return sorted(records, key=lambda r: r["tx_start"])


def _merge_intervals(intervals):
    """Merge overlapping (start, end) intervals into a sorted, disjoint list."""
    if not intervals:
        return []
    intervals = sorted((int(s), int(e)) for s, e in intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# --------------------------------------------------------------------------------------
# Splice-variability from the annotation (where isoforms disagree about boundaries)
# --------------------------------------------------------------------------------------
# The GTF is a catalog of a gene's annotated transcripts, so it does NOT tell you what
# spliced in *your* sample -- only where splicing *can* differ between isoforms. We derive
# three kinds of variable boundary from the per-transcript exon coordinates:
#   polya  : distinct transcript 3' ends  (alternative polyadenylation / cleavage sites)
#   tss    : distinct transcript 5' ends  (alternative promoters / first exons)
#   splice : distinct internal exon boundaries (donor/acceptor -- cassette exons, alt 5'/3'
#            splice sites all show up here)
# Each returned site is (coord, support_frac) where support_frac = fraction of the gene's
# transcripts using that (clustered) site, so a dominant site scores ~1 and a rare isoform
# scores near 0. Nearby sites are clustered (default 25 bp) so a tight spray of near-
# identical ends reads as one tick rather than a blob.
def _cluster_sites(coords, n_tx, cluster_bp=25):
    """Cluster 1-D coords within `cluster_bp`; return [(rep_coord, support_frac), ...].

    rep_coord is the most-used coordinate in each cluster; support_frac is the cluster's
    total transcript count over `n_tx`.
    """
    if not coords or n_tx <= 0:
        return []
    cnt = Counter(int(c) for c in coords)
    uniq = sorted(cnt)
    clusters = [[uniq[0]]]
    for c in uniq[1:]:
        if c - clusters[-1][-1] <= cluster_bp:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    out = []
    for cl in clusters:
        rep = max(cl, key=lambda c: (cnt[c], -c))     # dominant coord (ties -> smaller)
        out.append((rep, sum(cnt[c] for c in cl) / n_tx))
    return sorted(out)


def _splice_variability(g, strand, cluster_bp=25):
    """Per-gene splice-variability sites from a sub-frame `g` (one gene's GTF rows)."""
    ex_rows = g[g["feature"] == "exon"]
    tx_rows = g[g["feature"] == "transcript"]
    chains = {tid: sorted(zip(t["start0"].astype(int), t["end0"].astype(int)))
              for tid, t in ex_rows.groupby("transcript_id")}
    tx_bounds = {tid: (int(t["start0"].iloc[0]), int(t["end0"].iloc[0]))
                 for tid, t in tx_rows.groupby("transcript_id")}
    all_tids = set(chains) | set(tx_bounds)
    n_tx = len(all_tids)
    if n_tx == 0:
        return {"polya": [], "tss": [], "splice": [], "n_tx": 0}

    lefts, rights, donors, acceptors = [], [], [], []
    for tid in all_tids:
        ex = chains.get(tid)
        if ex:
            for i, (s, e) in enumerate(ex):
                if i > 0:
                    acceptors.append(s)            # internal exon start
                if i < len(ex) - 1:
                    donors.append(e)               # internal exon end
            ts, te = ex[0][0], ex[-1][1]
        else:
            ts = te = None
        if tid in tx_bounds:                       # transcript row = authoritative bounds
            ts, te = tx_bounds[tid]
        lefts.append(ts)
        rights.append(te)

    # 3' end (polyA) is the high coord on +, low coord on -; 5' end (TSS) is the other.
    tes = rights if strand == "+" else lefts
    tss = lefts if strand == "+" else rights
    return {
        "polya":  _cluster_sites(tes, n_tx, cluster_bp),
        "tss":    _cluster_sites(tss, n_tx, cluster_bp),
        "splice": _cluster_sites(donors + acceptors, n_tx, cluster_bp),
        "n_tx": n_tx,
    }


# --------------------------------------------------------------------------------------
# Gene-track rendering
# --------------------------------------------------------------------------------------
def _pack_rows(records, xlim, gap_frac=0.02, label_char_frac=0.0):
    """Greedy interval packing: assign each record to the lowest row whose last used
    x-end is left of this record's start (plus a small gap), so labels/boxes don't
    collide. Returns a list of row indices parallel to `records`.

    label_char_frac : width of one label character as a fraction of the x-span. When
        non-zero, a record occupies max(tx_end, tx_start + label width) instead of just
        tx_end. Small features whose names are far wider than they are -- miRNAs and
        snoRNAs are ~100 bp with a 10-character name -- otherwise pack onto one row and
        render as a pile of overlapping text.
    """
    span = max(1.0, xlim[1] - xlim[0])
    gap = span * gap_frac
    row_last_end = []
    rows = []
    for r in records:
        extent = r["tx_end"]
        if label_char_frac:
            name = r.get("gene_name") or r.get("gene_id") or ""
            extent = max(extent, r["tx_start"] + len(name) * label_char_frac * span)
        placed = None
        for i, last_end in enumerate(row_last_end):
            if r["tx_start"] > last_end + gap:
                placed = i
                break
        if placed is None:
            placed = len(row_last_end)
            row_last_end.append(extent)
        else:
            row_last_end[placed] = extent
        rows.append(placed)
    return rows


# Splice-variability tick styling: (color, legend label). polyA is the headline mark.
_TICK_STYLE = {
    "polya":  ("#e6550d", "alt polyA / 3′ end"),
    "tss":    ("#31a354", "alt TSS / 5′ end"),
    "splice": ("#9e9e9e", "splice site"),
}
_TICK_ORDER = ["splice", "tss", "polya"]   # draw order: subtle -> headline (last = on top)


def _draw_splice_ticks(ax, records, xlim, exon_height, rows, n_rows, show):
    """Thin vertical ticks just below each gene model marking annotated splice variability
    (alt polyA / TSS / internal splice sites). Tick length & opacity scale with the
    fraction of the gene's transcripts using that site, so dominant sites read boldest."""
    seen_types = set()
    for r, row in zip(records, rows):
        sv = r.get("splice_variability")
        if not sv:
            continue
        y = n_rows - 1 - row
        top = y - exon_height / 2 - 0.05          # sit just under the exon boxes
        for kind in _TICK_ORDER:
            if kind not in show:
                continue
            color = _TICK_STYLE[kind][0]
            for coord, support in sv.get(kind, []):
                if not (xlim[0] <= coord <= xlim[1]):
                    continue
                # support can exceed 1 for clustered splice sites (a cluster sums nearby
                # boundaries); clamp for the length/opacity scaling.
                w = min(1.0, support)
                length = 0.10 + 0.14 * w           # dominant site -> longer tick
                ax.plot([coord, coord], [top - length, top], color=color, lw=1.1,
                        alpha=0.4 + 0.6 * w, solid_capstyle="butt", zorder=5)
                seen_types.add(kind)
    if seen_types:
        handles = [Line2D([0], [0], color=_TICK_STYLE[k][0], lw=2)
                   for k in _TICK_ORDER if k in seen_types]
        labels = [_TICK_STYLE[k][1] for k in _TICK_ORDER if k in seen_types]
        ax.legend(handles, labels, fontsize=6, loc="lower right",
                  frameon=False, ncol=len(handles), handlelength=1.2,
                  columnspacing=1.0, borderaxespad=0.1)


def _draw_gene_track(ax, records, xlim, exon_height=0.34, label=True, n_arrows=12,
                     splice_marks=None, ylabel="Genes", label_char_frac=0.0045):
    """Draw IGV/UCSC-style transcript models: intron backbone lines, thick exon boxes,
    strand chevrons along the backbone, and gene-name labels.

    splice_marks : None (off), "polya" (alt 3'-end ticks only), or "all" (polyA + TSS +
                   internal splice-site ticks). True is an alias for "polya".
    ylabel       : y-axis label for this gene axis. Set it when several gene tracks are
                   stacked so each names the annotation subset it is drawing.
    """
    ax.set_xlim(xlim)
    if not records:
        ax.text(0.5, 0.5, "no genes in window", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_yticks([])
        ax.set_ylabel(ylabel, fontsize=9, weight="bold")
        return

    rows = _pack_rows(records, xlim, label_char_frac=label_char_frac if label else 0.0)
    n_rows = max(rows) + 1
    span = xlim[1] - xlim[0]

    for r, row in zip(records, rows):
        y = n_rows - 1 - row  # top row = first-placed, count downward
        color = "#2c3e50" if r["strand"] == "+" else "#7f3f3f"
        vis_start, vis_end = max(r["tx_start"], xlim[0]), min(r["tx_end"], xlim[1])

        # intron backbone
        ax.plot([vis_start, vis_end], [y, y], color=color, lw=1.0, zorder=1)

        # strand chevrons spaced along the visible backbone
        if vis_end > vis_start:
            step = span / n_arrows
            marker = ">" if r["strand"] == "+" else "<"
            xs = np.arange(vis_start + step * 0.5, vis_end, step) if step > 0 else []
            if len(xs):
                ax.plot(xs, np.full_like(xs, y, dtype=float), marker=marker,
                        linestyle="none", markersize=4, color=color, zorder=2)

        # exon boxes
        for es, ee in r["exons"]:
            es_c, ee_c = max(es, xlim[0]), min(ee, xlim[1])
            if ee_c <= es_c:
                continue
            ax.add_patch(Rectangle((es_c, y - exon_height / 2), ee_c - es_c, exon_height,
                                   facecolor=color, edgecolor="none", zorder=3))

        if label:
            lx = min(max(r["tx_start"], xlim[0]), xlim[1])
            ax.text(lx, y + exon_height / 2 + 0.08, r["gene_name"] or r["gene_id"],
                    fontsize=7, style="italic", color=color, va="bottom", ha="left",
                    clip_on=True, zorder=4)

    # optional splice-variability ticks under the models
    if splice_marks:
        mode = "polya" if splice_marks is True else str(splice_marks).lower()
        show = {"polya"} if mode == "polya" else set(_TICK_STYLE) if mode == "all" else set()
        if show:
            _draw_splice_ticks(ax, records, xlim, exon_height, rows, n_rows, show)

    ax.set_ylim(-0.75, n_rows - 0.2)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=9, weight="bold")


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def _coerce_track(t, idx):
    """Turn a Track / (name, array) / bare-array into a Track with a default color."""
    if isinstance(t, Track):
        track = t
    elif isinstance(t, tuple) and len(t) == 2:
        track = Track(values=t[1], name=str(t[0]))
    else:
        track = Track(values=t, name=f"track {idx}")
    if track.color is None and not track.is_truth:
        track.color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    if track.color is None and track.is_truth:
        track.color = TRUTH_COLOR
    return track


def _normalize_genes(genes):
    """Normalise the `genes` argument of `plot_tracks` to a list of (label, source).

    Accepts, and keeps backwards compatibility with, a single GeneAnnotation or GTF path.
    Also accepts a sequence of them, a sequence of (label, source) pairs, or a dict
    {label: source} -- each entry becomes its own gene axis, drawn in the order given.
    Returns [] for None.
    """
    if genes is None:
        return []
    if isinstance(genes, dict):
        return [(str(k), v) for k, v in genes.items()]
    if isinstance(genes, (GeneAnnotation, str)):
        return [("Genes", genes)]
    if isinstance(genes, (list, tuple)):
        # a bare (label, source) pair, or a sequence of entries
        if (len(genes) == 2 and isinstance(genes[0], str)
                and isinstance(genes[1], (GeneAnnotation, str))):
            return [(genes[0], genes[1])]
        out = []
        for g in genes:
            if isinstance(g, (list, tuple)) and len(g) == 2 and isinstance(g[0], str):
                out.append((g[0], g[1]))
            else:
                out.append(("Genes", g))
        return out
    return [("Genes", genes)]


def plot_tracks(tracks, genes=None, region=None, ground_truth=None, overlay=False,
                figsize=None, title=None, sharex=True, gene_track_ratio=0.6,
                truth_reference=True, collapse_genes=False, splice_marks=None,
                bottom_tracks=None, show=False):
    """Plot a stack of signal tracks with an optional gene track underneath.

    tracks        : list of Track (or (name, array) tuples, or bare arrays).
    genes         : GeneAnnotation, a GTF path string, or None. When set, a gene track
                    is drawn at the bottom (needs a genomic region -- see `region`).
                    Several annotation subsets can be stacked as separate gene axes by
                    passing a list of (label, GeneAnnotation) pairs or a {label: ann}
                    dict, e.g. [("Protein-coding", ann_pc), ("All genes", ann_all)].
    region        : (chrom, start, end) in genomic coords. Sets the x-limits and the gene
                    query window. If None, inferred from the first coord-bearing track.
    ground_truth  : optional Track treated as truth: drawn as a gray fill on its own axis
                    and (if truth_reference) a dashed line at its max is added to every
                    prediction axis, with prediction y-limits anchored to that max.
    bottom_tracks : optional list of Track drawn BELOW the ground-truth axis and above
                    the gene tracks. `tracks` all sit above the truth axis, so this is
                    the only way to put a track under it -- meant for a second assay
                    (accessibility, say) that belongs next to the gene models rather
                    than among the predictions. Mark such a track reference_exempt if it
                    is not in the target's units, or the truth-max line will compare two
                    different scales.
    overlay       : True -> all signal tracks share one axis. Otherwise one axis each,
                    except tracks sharing an `overlay_group` id, which share an axis.
    gene_track_ratio : gene-axis height relative to one signal axis. A sequence gives a
                    per-gene-axis height, e.g. (0.6, 1.6) to make a dense second track
                    taller than a sparse first one.
    truth_reference  : draw the dashed truth-max reference line on prediction axes.
    collapse_genes   : merge exons across each gene's transcripts into one model per
                    gene (declutters dense loci); default False = full transcript models.
    splice_marks  : draw GTF-derived splice-variability ticks under the gene track.
                    None (off), "polya" (alt polyA / 3'-end sites only), or "all" (polyA +
                    alt TSS + internal splice sites). Tick length/opacity scale with the
                    fraction of the gene's transcripts using each (clustered) site. These
                    are annotation possibilities, not observed splicing. Best with
                    collapse_genes=True (one model per gene).
    show          : call plt.show() before returning.

    Returns (fig, axes).
    """
    gene_specs = _normalize_genes(genes)
    tracks = [_coerce_track(t, i) for i, t in enumerate(tracks)]
    bottom_tracks = [_coerce_track(t, len(tracks) + i)
                     for i, t in enumerate(bottom_tracks or [])]
    if ground_truth is not None:
        gt = ground_truth if isinstance(ground_truth, Track) else Track(ground_truth,
                                                                        name="True")
        gt.is_truth = True
        gt.style = "fill" if gt.style == "line" else gt.style
        if gt.color is None:
            gt.color = TRUTH_COLOR

    # ---- decide x-axis mode --------------------------------------------------------
    all_signal = tracks + ([gt] if ground_truth is not None else []) + bottom_tracks
    genomic = region is not None or (len(all_signal) > 0
                                     and all(t.has_coords for t in all_signal))
    if gene_specs and not genomic:
        print("[plot_tracks] gene track requested but no genomic coordinates available; "
              "pass `region=(chrom,start,end)` or set chrom/start/bin_size on tracks. "
              "Skipping gene track.")
        gene_specs = []

    # ---- resolve region / xlim -----------------------------------------------------
    chrom = None
    if region is not None:
        chrom, xlo, xhi = region[0], float(region[1]), float(region[2])
    elif genomic:
        coord_tracks = [t for t in all_signal if t.has_coords]
        chrom = coord_tracks[0].chrom
        xs = [t.x_coords() for t in coord_tracks]
        xlo = min(x[0] for x in xs) - coord_tracks[0].bin_size / 2
        xhi = max(x[-1] for x in xs) + coord_tracks[0].bin_size / 2
    else:
        xlo = 0
        xhi = max((len(t.values) for t in all_signal), default=1) - 1
    xlim = (xlo, xhi)

    # ---- group tracks into axes ----------------------------------------------------
    # groups: list of lists of Track. Ground truth gets its own axis at the bottom of
    # the signal stack (above the gene track).
    if overlay:
        groups = [list(tracks)]
    else:
        groups, seen = [], {}
        for t in tracks:
            if t.overlay_group is not None:
                if t.overlay_group not in seen:
                    seen[t.overlay_group] = len(groups)
                    groups.append([])
                groups[seen[t.overlay_group]].append(t)
            else:
                groups.append([t])
    if ground_truth is not None:
        groups.append([gt])
    # below the truth axis, so the stack reads predictions -> truth -> other assays -> genes
    groups.extend([t] for t in bottom_tracks)

    n_signal_axes = len(groups)
    n_gene_axes = len(gene_specs)
    n_axes = n_signal_axes + n_gene_axes
    if np.isscalar(gene_track_ratio):
        gene_ratios = [float(gene_track_ratio)] * n_gene_axes
    else:
        gr = list(gene_track_ratio)
        gene_ratios = [float(gr[j]) if j < len(gr) else float(gr[-1])
                       for j in range(n_gene_axes)]
    height_ratios = [1.0] * n_signal_axes + gene_ratios

    if figsize is None:
        figsize = (13, 1.6 * n_signal_axes + 1.8 * sum(gene_ratios) + 0.6)
    fig, axes = plt.subplots(n_axes, 1, figsize=figsize, sharex=sharex,
                             gridspec_kw={"height_ratios": height_ratios},
                             squeeze=False)
    axes = axes[:, 0]

    truth_max = float(np.max(gt.values)) if ground_truth is not None else None

    # ---- draw signal axes ----------------------------------------------------------
    for ax, group in zip(axes[:n_signal_axes], groups):
        is_truth_axis = any(t.is_truth for t in group)
        exempt = any(t.reference_exempt for t in group)
        for t in group:
            x = t.x_coords()
            if t.style == "fill":
                ax.fill_between(x, t.values, color=t.color, alpha=t.alpha, label=t.name)
            else:
                ax.plot(x, t.values, color=t.color, alpha=t.alpha, label=t.name, lw=1.1)

        # dashed truth-max reference on prediction axes
        if truth_max is not None and truth_reference and not is_truth_axis and not exempt:
            ax.axhline(truth_max, color="black", linestyle="--", alpha=0.6, lw=1.0)

        # y-limits
        explicit = next((t.ylim for t in group if t.ylim is not None), None)
        if explicit is not None:
            ax.set_ylim(*explicit)
        else:
            gmax = max(float(np.max(t.values)) for t in group)
            if truth_max is not None and not is_truth_axis and not exempt:
                gmax = max(gmax, truth_max)
            gmin = min(0.0, min(float(np.min(t.values)) for t in group))
            ax.set_ylim(gmin, gmax * 1.05 if gmax > 0 else 1.0)

        # labels: right-side y-label = track name(s), notebook style
        ax.set_ylabel(" / ".join(t.name for t in group if t.name),
                      fontsize=9, weight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.yaxis.tick_right()
        if len(group) > 1:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_xlim(xlim)

    # ---- gene tracks -----------------------------------------------------------------
    # one axis per entry, in the order given, each labelled with its annotation subset
    for j, (glabel, gsrc) in enumerate(gene_specs):
        ann = gsrc if isinstance(gsrc, GeneAnnotation) else GeneAnnotation(gsrc)
        records = ann.query(chrom, int(xlim[0]), int(xlim[1]), collapse=collapse_genes)
        _draw_gene_track(axes[n_signal_axes + j], records, xlim,
                         splice_marks=splice_marks, ylabel=glabel)

    # ---- shared x-axis formatting --------------------------------------------------
    bottom = axes[-1]
    if genomic:
        bottom.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v/1e6:.3f}"))
        bottom.set_xlabel(
            f"{chrom}:{int(xlim[0]):,}-{int(xlim[1]):,}  (Mb)", fontsize=9)
    else:
        bottom.set_xlabel("Bin index", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, weight="bold", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    if show:
        plt.show()
    return fig, axes


def tracks_from_arrays(arrays, names=None, colors=None, styles=None,
                       chrom=None, start=None, bin_size=None):
    """Quick builder: a list of same-region 1-D arrays -> list[Track].

    names/colors/styles are optional per-array lists. chrom/start/bin_size (if given)
    are applied to every track so they share a genomic x-axis.
    """
    n = len(arrays)
    names = names or [f"track {i}" for i in range(n)]
    colors = colors or [None] * n
    styles = styles or ["line"] * n
    return [Track(values=a, name=nm, color=c, style=s,
                  chrom=chrom, start=start, bin_size=bin_size)
            for a, nm, c, s in zip(arrays, names, colors, styles)]
