"""CAGE promoter-expression cache for the eQTL specificity benchmark.

Reads Borzoi's stored training labels (`borzoi.zarr`), which are measured
coverage in genomic spans, and extracts CAGE signal at GENCODE TSSs.  No model
inference happens here.

Geometry, all verified against the stores rather than assumed:

  * `sequences_borzoi.bed` rows are the measured spans, in fold order.  Each is
    exactly 196,608 bp = 6144 bins x 32 bp, so the BED interval IS the measured
    interval; unlike the Enformer store there is no input-vs-output crop to undo.
  * `borzoi.zarr/fold{k}` has shape (n_k, 6144, 7611) with chunks (1, 6144, 7611),
    i.e. one span per chunk.  A single locus therefore costs one ~55 MB chunk.
  * Span starts are NOT aligned to a global 32 bp grid (starts mod 32 are spread
    over all residues), so two spans covering the same base pair carry different
    bin boundaries.  Every extraction records which span it came from and the
    actual bp extent summed, and `check_span_agreement` quantifies the disagreement.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, os, re, time, fcntl
import numpy as np
import pandas as pd
import numcodecs
import torch

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('EQTL7_CACHE', str(ROOT / 'eQTL_benchmark7_cache')))
QTL = Path('/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs')
BORZOI = Path('/data1/lesliec/sarthak/data/borzoi/borzoi.zarr')
SPANS = Path('/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi.bed')
SPANS_XCHECK = Path('/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi_fold3-4.bed')
TARGETS = Path('/data1/lesliec/sarthak/data/DK_zarr/targets_borzoi.txt')
GENCODE = Path('/data1/lesliec/sarthak/data/gpn/gencode.v47.basic.annotation.gtf')

BIN = 32                      # bp per stored Borzoi bin
SPAN_BINS = 6144
SPAN_BP = SPAN_BINS * BIN     # 196,608
N_TARGETS = 7611
FOLDS = [f'fold{i}' for i in range(8)]

# Promoter window, strand-aware and half-open: [tss - UPSTREAM, tss + DOWNSTREAM)
# on '+', mirrored on '-'.  400 bp spans 13-14 of the 32 bp bins depending on
# where the TSS falls in its span's grid.
UPSTREAM, DOWNSTREAM = 300, 100
# Raw per-bin window cached for benchmark genes only, so the pooling rule above
# can be changed later without re-reading the zarr.  +/- 512 bp = 32 bins.
RAW_HALF_BP = 512
# A 1,024 bp window straddles 33 bins whenever it does not start on a bin boundary,
# which is the usual case: span starts are not aligned to a common 32 bp grid.
RAW_BINS = 2 * RAW_HALF_BP // BIN + 1

# Background population for every threshold.  Benchmark genes are extracted
# regardless of biotype but do not enter the threshold reference unless they are
# protein-coding.
BACKGROUND_BIOTYPE = 'protein_coding'
# Descriptions matched here are B-lineage samples.  They stay in the primary
# breadth count and are additionally reported excluded, because a truly
# B-cell-restricted gene otherwise reads as "broadly expressed".
LYMPHOID = re.compile(r'GM12878|lymphoblastoid|B lymphoblast|CD19|B cell', re.I)

WORKERS = int(os.environ.get('EQTL7_WORKERS', '4'))


def progress(*parts):
    message = ' '.join(map(str, parts))
    print(message, flush=True)
    CACHE.mkdir(exist_ok=True)
    with (CACHE / 'build.log').open('a') as log:
        log.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    p = Path(path)
    return dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)


def metadata(path):
    m = json.loads((Path(path) / '.zarray').read_text())
    assert m['zarr_format'] == 2 and m['order'] == 'C' and not m.get('filters')
    assert m.get('dimension_separator', '.') == '.'
    return m


def read_chunk(path, m, indices, allow_missing=False):
    """Read one local Zarr-v2 chunk directly; valid for the checked C-order,
    unfiltered, dot-separated stores here.

    `borzoi.zarr` has a chunk file for every row, so a miss here is a bug -- a float
    index alone builds a path like "1316.0.0.0" -- and fabricating the fill value
    would hide it as a silently all-zero result.  Callers that really do read a
    sparse store must opt in.
    """
    assert all(float(i).is_integer() for i in indices), f'non-integer chunk index {indices}'
    p = Path(path) / '.'.join(str(int(i)) for i in indices)
    if not p.exists():
        assert allow_missing and m.get('fill_value') == 0, f'missing chunk {p}'
        return np.zeros(m['chunks'], dtype=m['dtype'])
    raw = p.read_bytes()
    if m['compressor'] is not None:
        raw = numcodecs.get_codec(m['compressor']).decode(raw)
    return np.frombuffer(raw, dtype=m['dtype']).reshape(m['chunks'])


# --------------------------------------------------------------------------
# Track, span and TSS tables
# --------------------------------------------------------------------------
def cage_tracks():
    """The 1,276 CAGE columns = 638 biosamples x 2 strands.

    Borzoi's CAGE panel is the same 638 FANTOM/ENCODE biosamples as Enformer's,
    split by strand; `strand_pair` names the partner column.  GM12878 is a single
    biosample (columns 870/871), so a GM12878 call rests on one sample, unlike the
    two independent DNase tracks the dsQTL analysis could require 30% support from.
    """
    t = pd.read_csv(TARGETS, sep='\t', index_col=0)
    assert len(t) == N_TARGETS and np.array_equal(t.index.to_numpy(), np.arange(N_TARGETS))
    cage = t[t.description.str.startswith('CAGE:')].copy()
    cage['target_index'] = cage.index.to_numpy()
    cage['strand'] = cage.identifier.str[-1]
    assert cage.strand.isin(['+', '-']).all(), 'CAGE identifiers must carry a strand suffix'
    # A biosample is one description; the two strands share it and pair with each other.
    assert cage.strand_pair.isin(cage.target_index).all()
    pair = cage.set_index('target_index').strand_pair
    assert (pair.reindex(pair.to_numpy()).to_numpy() == pair.index.to_numpy()).all(), 'strand_pair must be an involution'
    assert (cage.groupby('description').size() == 2).all(), 'expected exactly two strands per biosample'
    cage['biosample'] = cage.description
    cage['is_gm12878'] = cage.description.str.contains('GM12878', regex=False)
    cage['is_lymphoid'] = cage.description.str.contains(LYMPHOID)
    assert cage.is_gm12878.sum() == 2, cage.is_gm12878.sum()
    return cage.reset_index(drop=True)


def span_table():
    """Measured spans in zarr row order, cross-checked against the fold3-4 BED.

    Row order within a fold is that fold's order in `sequences_borzoi.bed`.
    `sequences_borzoi_fold3-4.bed` independently records (fold, index) per span,
    so it is used to verify the derived mapping rather than to define it.
    """
    s = pd.read_csv(SPANS, sep='\t', header=None, names=['chrom', 'start', 'end', 'fold'])
    assert (s.end - s.start == SPAN_BP).all(), 'spans must be exactly the measured width'
    s['row'] = s.groupby('fold').cumcount()
    for fold in FOLDS:
        m = metadata(BORZOI / fold)
        assert m['shape'] == [int((s.fold == fold).sum()), SPAN_BINS, N_TARGETS], (fold, m['shape'])
        assert m['chunks'] == [1, SPAN_BINS, N_TARGETS] and m['dtype'] == '<f2'
    x = pd.read_csv(SPANS_XCHECK, sep='\t', header=None,
                    names=['chrom', 'start', 'end', 'split', 'fold', 'index'])
    assert len(x) == len(s)
    assert np.array_equal(x[['chrom', 'start', 'end']].to_numpy(),
                          s[['chrom', 'start', 'end']].to_numpy())
    assert (x.fold.to_numpy() == s.fold.to_numpy()).all()
    # The recorded index is within-fold, not global; fail loudly if that changes.
    assert (x.groupby('fold')['index'].apply(lambda g: np.array_equal(g.to_numpy(), np.arange(len(g))))).all(), \
        'fold3-4 BED index is not a within-fold arange; re-derive the zarr row mapping'
    assert (x['index'].to_numpy() == s.row.to_numpy()).all()
    # Recorded for the manifest: unaligned starts mean overlapping spans disagree at bin level.
    s.attrs['grid_offsets'] = int(pd.unique(s.start % BIN).size)
    return s


def tss_table(benchmark_genes):
    """Unique (gene, TSS) pairs from GENCODE v47 basic transcripts.

    GTF is 1-based inclusive, so a '+' transcript's TSS is `start - 1` and a '-'
    transcript's is `end - 1` in 0-based coordinates.  Gene IDs are stripped of
    their version so they join to `genes.pickle` / `ensemblTSS.txt`.
    """
    rows = []
    with GENCODE.open() as f:
        for line in f:
            if line[0] == '#':
                continue
            p = line.split('\t')
            if p[2] != 'transcript':
                continue
            gene = re.search(r'gene_id "([^".]+)', p[8]).group(1)
            biotype = re.search(r'gene_type "([^"]+)"', p[8]).group(1)
            tss = int(p[3]) - 1 if p[6] == '+' else int(p[4]) - 1
            rows.append((gene, p[0], p[6], tss, biotype))
    t = pd.DataFrame(rows, columns=['gene_id', 'chrom', 'strand', 'tss0', 'gene_type'])
    # A gene on two contigs or two strands would break the per-gene window merge.
    assert t.groupby('gene_id')[['chrom', 'strand']].nunique().le(1).all().all()
    t = t.drop_duplicates(['gene_id', 'tss0']).sort_values(['gene_id', 'tss0'], ignore_index=True)
    t['is_benchmark'] = t.gene_id.isin(benchmark_genes)
    t['is_background'] = t.gene_type.eq(BACKGROUND_BIOTYPE)
    t = t[t.is_benchmark | t.is_background].reset_index(drop=True)
    t['tss_id'] = t.gene_id + ':' + t.tss0.astype(str)
    return t


def canonical_chromosome(value):
    """Normalize the benchmark's PLINK names, including legacy cached chr23/24."""
    name = str(value).removeprefix('chr')
    return 'chr' + {'23': 'X', '24': 'Y', '25': 'X', '26': 'M'}.get(name, name)


def benchmark_gene_table():
    """Benchmark gene identity and the poster's own TSS, kept for cross-checking.

    `ensemblTSS.txt` carries no chromosome or strand, so GENCODE supplies both.
    The poster's distance calculation is left untouched and keeps using this file.
    """
    import pickle
    q = pd.read_csv(QTL / 'LCLs.txt', sep=' ', header=None,
                    names=['label', 'qtl_idx', 'gene_idx', 'chrom', 'gene_start',
                           'gene_end', 'strand', 'qtl_loc', 'ref', 'alt', 'sign_target'])
    genes = sorted(pickle.load((QTL / 'genes.pickle').open('rb')).keys())
    q['gene_id'] = [genes[i] for i in q.gene_idx]
    # The benchmark uses PLINK chromosome coding, where X is 23 and Y is 24.
    q['chrom_name'] = q.chrom.map(canonical_chromosome)
    e = np.loadtxt(QTL / 'ensemblTSS.txt', dtype='str')
    poster_tss = dict(zip(e[:, 0], e[:, 1].astype(np.int64)))
    g = q.drop_duplicates('gene_id')[['gene_id', 'chrom_name']].reset_index(drop=True)
    g['poster_tss'] = [poster_tss.get(x, -1) for x in g.gene_id]
    return q, g


# --------------------------------------------------------------------------
# Window construction and span assignment
# --------------------------------------------------------------------------
def promoter_windows(tss):
    """Strand-aware promoter interval per TSS, half-open and 0-based."""
    up = np.where(tss.strand.to_numpy() == '+', UPSTREAM, DOWNSTREAM)
    down = np.where(tss.strand.to_numpy() == '+', DOWNSTREAM, UPSTREAM)
    return tss.tss0.to_numpy() - up, tss.tss0.to_numpy() + down


def merge_gene_windows(tss):
    """Per-gene union of promoter windows.

    The gene score sums over its TSSs, so overlapping promoter windows -- common
    for genes with clustered alternative first exons -- would otherwise be counted
    twice.  Merging first makes the sum a sum over distinct base pairs.  A merged
    interval wider than one span cannot be read from a single chunk; those genes
    fall back to their unmerged windows, which are then non-overlapping by
    construction because only distant TSSs survive the split.

    Windows are merged when they come within one BIN of each other, not merely when
    they overlap.  Sums are taken over whole 32 bp bins, so two units separated by
    1-31 bp would still share an edge bin and double-count it -- bounded at one bin,
    but real: 272 of 20,229 genes hit this at a 0 bp merge threshold.  Requiring a
    >= BIN gap makes a shared bin impossible whatever the span's grid offset.
    """
    lo, hi = promoter_windows(tss)
    d = pd.DataFrame(dict(gene_id=tss.gene_id.to_numpy(), chrom=tss.chrom.to_numpy(),
                          lo=lo, hi=hi)).sort_values(['gene_id', 'lo'], kind='stable')
    units = []
    for gene, g in d.groupby('gene_id', sort=False):
        chrom = g.chrom.iloc[0]
        cur_lo, cur_hi = None, None
        for a, b in zip(g.lo.to_numpy(), g.hi.to_numpy()):
            if cur_lo is None:
                cur_lo, cur_hi = a, b
            elif a < cur_hi + BIN and b - cur_lo <= SPAN_BP:
                cur_hi = max(cur_hi, b)
            else:
                units.append((gene, chrom, cur_lo, cur_hi))
                cur_lo, cur_hi = a, b
        units.append((gene, chrom, cur_lo, cur_hi))
    u = pd.DataFrame(units, columns=['gene_id', 'chrom', 'lo', 'hi'])
    u['unit_bp'] = u.hi - u.lo
    return u.reset_index(drop=True)


def raw_windows(tss):
    """TSS-centred raw-bin window for benchmark genes, anchored on the TSS itself."""
    b = tss[tss.is_benchmark].copy()
    b['lo'] = b.tss0 - RAW_HALF_BP
    b['hi'] = b.tss0 + RAW_HALF_BP
    return b.reset_index(drop=True)


def assign_spans(windows, spans):
    """Assign each window to one covering span, minimising the number of chunks read.

    Every span is the same width, so the spans covering [lo, hi) are exactly those
    whose start lies in [hi - SPAN_BP, lo] -- a contiguous slice of the sorted
    starts.  Picking spans to read is therefore the classic problem of stabbing a
    set of intervals with the fewest points, which the right-endpoint greedy solves
    optimally: sort windows by the largest admissible span index, and whenever a
    window is not yet stabbed take that largest index.  Optimal and deterministic,
    and it does not look at labels, gene identity or model scores.

    A window with no covering span is `uncovered` -- missing, not zero.
    """
    windows = windows.reset_index(drop=True)
    out = np.full(len(windows), -1, dtype=np.int64)
    for chrom, w in windows.groupby('chrom', sort=False):
        s = spans[spans.chrom == chrom]
        if s.empty:
            continue
        order = np.argsort(s.start.to_numpy(), kind='stable')
        starts = s.start.to_numpy()[order]
        ids = s.index.to_numpy()[order]
        lo, hi = w.lo.to_numpy(), w.hi.to_numpy()
        a = np.searchsorted(starts, hi - SPAN_BP, side='left')
        b = np.searchsorted(starts, lo, side='right')
        ok = a < b
        idx = np.flatnonzero(ok)
        chosen = []
        for j in idx[np.argsort(b[idx], kind='stable')]:
            if not chosen or chosen[-1] < a[j]:
                chosen.append(b[j] - 1)
        chosen = np.array(chosen, dtype=np.int64)
        # Each window takes the first chosen span at or after its lower bound.
        pos = np.searchsorted(chosen, a[idx], side='left')
        assert (pos < len(chosen)).all() and (chosen[pos] < b[idx]).all()
        out[w.index.to_numpy()[idx]] = ids[chosen[pos]]
    return out


def attach_span(windows, spans):
    """Add the chosen span, its bin range and the bp actually summed.

    Bins are half-open and clipped to the span; a window fully inside a span always
    lands on whole bins, so `covered_bp` exceeds the requested width by the two
    partial edge bins.  Recording it keeps the grid-offset effect visible instead of
    hiding it inside a single number.
    """
    w = windows.reset_index(drop=True)
    w['span'] = assign_spans(w, spans)
    w['covered'] = w.span >= 0
    ss = np.where(w.covered, spans.start.reindex(w.span).to_numpy(), 0).astype(np.int64)
    w['bin_lo'] = np.where(w.covered, (w.lo.to_numpy() - ss) // BIN, -1).astype(np.int64)
    w['bin_hi'] = np.where(w.covered, -(-(w.hi.to_numpy() - ss) // BIN), -1).astype(np.int64)
    w['n_bins'] = np.where(w.covered, w.bin_hi - w.bin_lo, 0).astype(np.int64)
    w['covered_bp'] = (w.n_bins * BIN).astype(np.int64)
    w['fold'] = np.where(w.covered, spans.fold.reindex(w.span).to_numpy(), '')
    # Uncovered rows must not drag this to float: a float row number builds a chunk
    # path like "1316.0.0.0", which does not exist and would read as silent zeros.
    w['span_row'] = np.where(w.covered, spans.row.reindex(w.span).to_numpy(), -1).astype(np.int64)
    ins = w.covered.to_numpy()
    assert (w.bin_lo.to_numpy()[ins] >= 0).all() and (w.bin_hi.to_numpy()[ins] <= SPAN_BINS).all()
    return w


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _read_span(fold_meta, fold, row, bins):
    """Return the CAGE columns for the requested bin rows of one span.

    Slicing the f16 chunk before converting keeps the conversion at a few hundred
    kB instead of the 187 MB a whole-chunk float32 cast would cost; torch does the
    exact f16->f32 conversion far faster than NumPy's.
    """
    chunk = read_chunk(BORZOI / fold, fold_meta[fold], (row, 0, 0))[0]
    part = chunk[bins]
    return torch.from_numpy(np.array(part)).float().numpy()


def extract(units, raw, cage, fold_meta):
    """One chunk read per chosen span; every window on that span is served from it."""
    ci = cage.target_index.to_numpy()
    sums = np.zeros((len(units), len(cage)), dtype=np.float32)
    raw_out = np.zeros((len(raw), RAW_BINS, len(cage)), dtype=np.float16)
    jobs = {}
    # Index by position in the FULL frame: uncovered rows keep their slot and stay zero,
    # which is why the coverage flag, not the value, is what marks a window as missing.
    for i in np.flatnonzero(units.covered.to_numpy()):
        r = units.iloc[i]
        jobs.setdefault((r.fold, r.span_row), ([], []))[0].append((int(i), int(r.bin_lo), int(r.bin_hi)))
    for i in np.flatnonzero(raw.covered.to_numpy()):
        r = raw.iloc[i]
        jobs.setdefault((r.fold, r.span_row), ([], []))[1].append((int(i), int(r.bin_lo), int(r.bin_hi)))
    keys = sorted(jobs)
    done = [0]
    t0 = time.time()

    def run(key):
        fold, row = key
        want, want_raw = jobs[key]
        lo = min([b for _, b, _ in want + want_raw])
        hi = max([b for _, _, b in want + want_raw])
        block = _read_span(fold_meta, fold, row, slice(lo, hi))[:, ci]
        for i, a, b in want:
            sums[i] = block[a - lo:b - lo].sum(axis=0)
        for i, a, b in want_raw:
            raw_out[i, :b - a] = block[a - lo:b - lo].astype(np.float16)
        done[0] += 1
        if done[0] % 200 == 0:
            progress(f'spans {done[0]}/{len(keys)}  {time.time() - t0:.0f}s')

    numcodecs.blosc.set_nthreads(1)
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run, keys))
    progress(f'extracted {len(keys)} spans in {time.time() - t0:.0f}s')
    return sums, raw_out


def check_span_agreement(units, spans, cage, fold_meta, n=200, seed=0):
    """Re-read a sample of windows from a *different* covering span.

    Span bin grids are offset from each other, so the two reads sum slightly
    different base pairs and are not expected to be identical.  This measures how
    large that disagreement actually is, rather than assuming it away.
    """
    rng = np.random.default_rng(seed)
    have = units[units.covered]
    pick = have.iloc[rng.choice(len(have), min(n, len(have)), replace=False)]
    rows = []
    for r in pick.itertuples():
        s = spans[(spans.chrom == r.chrom) & (spans.start <= r.lo) &
                  (spans.start + SPAN_BP >= r.hi) & (spans.index != r.span)]
        if s.empty:
            continue
        alt = s.iloc[len(s) // 2]
        a_lo = (r.lo - alt.start) // BIN
        a_hi = -(-(r.hi - alt.start) // BIN)
        alt_v = _read_span(fold_meta, alt.fold, alt.row, slice(a_lo, a_hi))[:, cage.target_index.to_numpy()].sum(axis=0)
        own = _read_span(fold_meta, r.fold, r.span_row, slice(r.bin_lo, r.bin_hi))[:, cage.target_index.to_numpy()].sum(axis=0)
        d = np.abs(alt_v - own).sum() / max(own.sum(), 1e-9)
        rows.append(dict(gene_id=r.gene_id, chrom=r.chrom, lo=r.lo, hi=r.hi,
                         span=r.span, alt_span=int(alt.name), own_bp=int(r.covered_bp),
                         alt_bp=int((a_hi - a_lo) * BIN), relative_abs_difference=float(d)))
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / 'span_agreement.tsv', sep='\t', index=False)
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def validate_expression_cache(validate_sources=False):
    """Validate completed products without rereading source coverage.

    The existing version-1 cache retains BOTH strands, so it can supply the new
    sense-strand analysis without another span scan. Source validation is optional
    for cache-only analysis and enabled when the builder is invoked.
    """
    marker = CACHE / 'expression_manifest.json'
    config = json.loads(marker.read_text())
    expected = dict(bin_bp=BIN, span_bp=SPAN_BP, raw_window_bp=RAW_HALF_BP,
                    raw_bins=RAW_BINS, merge_gap_bp=BIN,
                    background_biotype=BACKGROUND_BIOTYPE)
    assert all(config.get(k) == v for k, v in expected.items()), 'Extraction settings changed: use a new EQTL7_CACHE directory.'
    assert config['promoter_window'] == dict(
        upstream=UPSTREAM, downstream=DOWNSTREAM, strand_aware=True,
        pooling='sum of all bins overlapping the window')
    required = ['genes.tsv.gz', 'biosamples.tsv', 'cage_tracks.tsv', 'tss.tsv.gz',
                'gene_windows.tsv.gz', 'raw_bins_index.tsv.gz', 'benchmark_genes.tsv',
                'gene_cage_columns.npy', 'gene_cage_biosamples.npy', 'raw_bins.npy']
    missing = [f for f in required if not (CACHE / f).is_file()]
    assert not missing, f'Incomplete expression cache: {missing}'
    genes = pd.read_csv(CACHE / 'genes.tsv.gz', sep='\t')
    tracks = pd.read_csv(CACHE / 'cage_tracks.tsv', sep='\t')
    bios = pd.read_csv(CACHE / 'biosamples.tsv', sep='\t')
    raw = pd.read_csv(CACHE / 'raw_bins_index.tsv.gz', sep='\t')
    assert not genes.gene_id.duplicated().any()
    assert not bios.biosample.duplicated().any()
    assert genes.strand.isin(['+', '-']).all()
    assert genes.mapping_status.isin(['covered', 'partial', 'uncovered']).all()
    for filename, shape in [
        ('gene_cage_columns.npy', (len(genes), len(tracks))),
        ('gene_cage_biosamples.npy', (len(genes), len(bios))),
        ('raw_bins.npy', (len(raw), RAW_BINS, len(tracks))),
    ]:
        assert np.load(CACHE / filename, mmap_mode='r').shape == shape, filename
    if validate_sources:
        assert config['source'] == str(BORZOI)
        assert config['spans'] == stamp(SPANS) and config['gencode'] == stamp(GENCODE)
        assert config['targets_sha256'] == digest(TARGETS)
        _, current_bench = benchmark_gene_table()
        saved = pd.read_csv(CACHE / 'benchmark_genes.tsv', sep='\t')
        # This table is metadata only; extraction uses GENCODE chromosome names.
        # Accept the earlier cache's chr23 spelling after the user's PLINK fix.
        saved['chrom_name'] = saved.chrom_name.map(canonical_chromosome)
        pd.testing.assert_frame_equal(saved, current_bench, check_dtype=False)
    return genes


def build_expression_cache():
    CACHE.mkdir(exist_ok=True)
    with (CACHE / 'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (CACHE / 'expression_manifest.json').exists():
            genes = validate_expression_cache(validate_sources=True)
            progress('Reusing completed expression cache; no span reads.')
            return genes
        cage = cage_tracks()
        spans = span_table()
        q, bench = benchmark_gene_table()
        tss = tss_table(set(bench.gene_id))
        fold_meta = {f: metadata(BORZOI / f) for f in FOLDS}

        config = dict(
            version=1, source=str(BORZOI), spans=stamp(SPANS), targets_sha256=digest(TARGETS),
            gencode=stamp(GENCODE), bin_bp=BIN, span_bp=SPAN_BP,
            promoter_window=dict(upstream=UPSTREAM, downstream=DOWNSTREAM, strand_aware=True,
                                 pooling='sum of all bins overlapping the window'),
            raw_window_bp=RAW_HALF_BP, raw_bins=RAW_BINS,
            multi_tss='sum over per-gene merged promoter windows',
            merge_gap_bp=BIN,  # units closer than one bin are merged, so none can share a bin
            background_biotype=BACKGROUND_BIOTYPE,
            n_cage_columns=len(cage), n_biosamples=int(cage.description.nunique()),
            gm12878_columns=cage.target_index[cage.is_gm12878].tolist(),
            lymphoid_biosamples=sorted(cage.description[cage.is_lymphoid].unique().tolist()),
            distinct_span_grid_offsets=spans.attrs['grid_offsets'],
            span_assignment='minimum stabbing set over covering spans; deterministic, label-independent',
            uncovered='missing, never zero-filled')
        marker = CACHE / 'expression_manifest.json'
        if marker.exists():
            assert json.loads(marker.read_text()) == config, 'Cache config mismatch: use a new CACHE directory.'

        units = attach_span(merge_gene_windows(tss), spans)
        raw = attach_span(raw_windows(tss), spans)
        progress(f'{len(tss)} TSSs, {len(units)} gene windows '
                 f'({int(units.covered.sum())} covered), {len(raw)} raw windows '
                 f'({int(raw.covered.sum())} covered)')
        progress(f'distinct spans to read: '
                 f'{len(set(map(tuple, units[units.covered][["fold", "span_row"]].to_numpy())) | set(map(tuple, raw[raw.covered][["fold", "span_row"]].to_numpy())))}')

        sums, raw_out = extract(units, raw, cage, fold_meta)

        # Gene x CAGE column, summing a gene's disjoint merged windows.
        genes = pd.Index(sorted(tss.gene_id.unique()))
        gene_cage = np.zeros((len(genes), len(cage)), dtype=np.float32)
        pos = genes.get_indexer(units.gene_id)
        np.add.at(gene_cage, pos[units.covered.to_numpy()], sums[units.covered.to_numpy()])

        # Strand pairs collapse to the 638 biosamples the specificity is defined over.
        order = cage.sort_values(['description', 'strand']).index.to_numpy()
        bios = cage.description.to_numpy()[order]
        labels, inv = np.unique(bios, return_inverse=True)
        gene_bios = np.zeros((len(genes), len(labels)), dtype=np.float32)
        np.add.at(gene_bios.T, inv, gene_cage[:, order].T)
        assert np.isclose(gene_bios.sum(), gene_cage.sum(), rtol=1e-4), 'strand collapse lost signal'

        cov = units.groupby('gene_id').covered.agg(['sum', 'size'])
        gene_table = pd.DataFrame(dict(gene_id=genes))
        gene_table = gene_table.merge(tss.drop_duplicates('gene_id')[['gene_id', 'chrom', 'strand', 'gene_type', 'is_benchmark', 'is_background']], on='gene_id')
        gene_table['n_tss'] = tss.groupby('gene_id').size().reindex(genes).to_numpy()
        gene_table['n_windows'] = cov['size'].reindex(genes).to_numpy()
        gene_table['n_windows_covered'] = cov['sum'].reindex(genes).to_numpy()
        gene_table['mapping_status'] = np.where(
            gene_table.n_windows_covered == 0, 'uncovered',
            np.where(gene_table.n_windows_covered < gene_table.n_windows, 'partial', 'covered'))

        np.save(CACHE / 'gene_cage_columns.npy', gene_cage)
        np.save(CACHE / 'gene_cage_biosamples.npy', gene_bios)
        np.save(CACHE / 'raw_bins.npy', raw_out)
        pd.Series(labels, name='biosample').to_csv(CACHE / 'biosamples.tsv', sep='\t', index=False)
        cage.to_csv(CACHE / 'cage_tracks.tsv', sep='\t', index=False)
        tss.to_csv(CACHE / 'tss.tsv.gz', sep='\t', index=False)
        units.to_csv(CACHE / 'gene_windows.tsv.gz', sep='\t', index=False)
        raw.drop(columns=['is_background']).to_csv(CACHE / 'raw_bins_index.tsv.gz', sep='\t', index=False)
        gene_table.to_csv(CACHE / 'genes.tsv.gz', sep='\t', index=False)
        bench.to_csv(CACHE / 'benchmark_genes.tsv', sep='\t', index=False)
        atomic_json(marker, config)
        progress('Expression cache complete:', gene_table.mapping_status.value_counts().to_dict())
        return gene_table


if __name__ == '__main__':
    build_expression_cache()
