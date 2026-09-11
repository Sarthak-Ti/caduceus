"""eQTL benchmark scored, annotated by target-gene CAGE specificity, and stratified.

Scoring reproduces `IGVF_meeting_figures.ipynb::load_eqtl` exactly -- same nine arms,
same central 2,048 bp crop, same |log2 mean(ref)/mean(alt)|, same
"unscored iff both ref and alt sum to zero" rule -- but keeps the original
benchmark row indices, per-arm scorable masks and file provenance, which the
collapsed `.upset_cache.npz` in eQTL_benchmark6 does not.
"""
from eqtl_benchmark7_pipeline import *
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_PAIRS = 1331
CENTER_BP = 2048

ARMS = [
    ('cis-Mamba (1 cell type)',       'ours',        'nomlm_maskonly.npy',      128),
    ('cis-Mamba, no pretraining',     'ours',        'nopretrain.npy',          128),
    ('cis-Mamba (8 cell types)',      'ours',        'immune.npy',              128),
    ('Enformer (pretrained)',         'enformer',    'enformer_eqtl.npy',       128),
    ('Borzoi (pretrained)',           'borzoi',      'borzoi_eqtl.npy',          32),
    ('Borzoi LoRA (GM12878)',         'borzoi',      'borzoi_eqtl_lorav2.npy',   32),
    ('Borzoi full FT (GM12878)',      'borzoi',      'borzoi_eqtl_trunk.npy',    32),
    ('Borzoi from scratch (GM12878)', 'borzoi',      'borzoi_eqtl_scratch.npy',  32),
    ('AlphaGenome',                   'alphagenome', 'alphagenome_eqtl.zarr',     1),
]
NAMES = [a[0] for a in ARMS]
FAMILY = {'ours': '#2878B5', 'enformer': '#DD8452', 'borzoi': '#55A868', 'alphagenome': '#8172B3'}

# Keep the current 20-CPM activity gate as an analysis choice, not a calibrated
# expression-detection threshold. Primary inference uses continuous others signal;
# display groups split others signal at the active-background median.
CUTOFF_CPM = 20.0
GM_HIGH_CPM = CUTOFF_CPM
GM_LOW_CPM = CUTOFF_CPM / 5
OTHERS_LOW_PERCENTILE = 50.0
OTHERS_HIGH_PERCENTILE = 90.0
STRAND_MODE = 'sense'

DISTANCE_BINS = [('<1 kb', 0, 1_000), ('1-10 kb', 1_000, 10_000),
                 ('10-262 kb', 10_000, 524_288 // 2)]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def read_predictions(name, bp):
    """Load one arm's (pairs, bins, ref/alt) array, cropped to the central 2,048 bp."""
    path = QTL / name
    nb = CENTER_BP // bp
    assert nb * bp == CENTER_BP, f'{name}: {bp} bp bins do not tile {CENTER_BP} bp'
    if name.endswith('.zarr'):
        m = metadata(path)
        assert m['chunks'] == [1, m['shape'][1], 2], 'Expected one full profile per chunk'
        assert m['shape'][1] >= nb, 'Prediction profile is shorter than the scoring crop'
        assert m['shape'][0] == N_PAIRS and m['shape'][2] == 2 and m['chunks'][0] == 1
        cs = m['shape'][1] // 2 - nb // 2
        out = np.empty((N_PAIRS, nb, 2), dtype=np.float64)
        for j in range(N_PAIRS):
            out[j] = read_chunk(path, m, (j, 0, 0))[0, cs:cs + nb]
        return out
    a = np.load(path, mmap_mode='r')
    assert a.ndim == 3 and a.shape[0] == N_PAIRS and a.shape[2] == 2, (name, a.shape)
    assert a.shape[1] >= nb, 'Prediction profile is shorter than the scoring crop'
    cs = a.shape[1] // 2 - nb // 2
    return np.asarray(a[:, cs:cs + nb, :], dtype=np.float64)


def abs_logfc(win):
    """The poster's scorer: |log2 mean(ref)/mean(alt)|, zero where both sums vanish.

    An arm is unscored only when ref AND alt are both empty over the window.  A
    one-sided zero would make the ratio infinite, so it is counted separately and
    excluded from the strict mask rather than silently passed through.
    """
    r, a = win[:, :, 0], win[:, :, 1]
    unscored = (r.sum(1) == 0) & (a.sum(1) == 0)
    out = np.zeros(len(r))
    v = ~unscored
    with np.errstate(divide='ignore', invalid='ignore'):
        out[v] = np.log2(r[v].mean(1) / a[v].mean(1))
    strict = (r.mean(1) > 0) & (a.mean(1) > 0) & np.isfinite(out)
    return np.abs(out), ~unscored, strict


def score_pairs(validate_sources=False):
    """Use cached scores by default; source validation is an explicit option.

    Cached mode checks the model definition and benchmark row hash without
    opening prediction arrays or source-Zarr metadata. Prediction sources are
    assumed immutable unless validate_sources=True or a new cache is requested.
    """
    path = CACHE / 'model_scores.npz'
    if path.exists() and not validate_sources:
        saved = json.loads((CACHE / 'scores_manifest.json').read_text())
        assert saved['arms'] == json.loads(json.dumps(ARMS))
        assert saved['center_bp'] == CENTER_BP
        assert saved['benchmark_sha256'] == digest(QTL / 'LCLs.txt')
        with np.load(path) as d:
            arrays = tuple(d[k] for k in ['scores', 'scorable', 'strict'])
        assert all(x.shape == (N_PAIRS, len(ARMS)) for x in arrays)
        return arrays
    provenance = json.loads(json.dumps(dict(
        version=1, arms=ARMS, center_bp=CENTER_BP,
        scorer='abs(log2(mean(ref)/mean(alt))) over the central crop',
        unscored='both ref and alt window sums are zero (poster rule)',
        files=[stamp(QTL / f) for _, _, f, _ in ARMS if not f.endswith('.zarr')] +
              [dict(path=str(QTL / f), zarray=metadata(QTL / f)) for _, _, f, _ in ARMS if f.endswith('.zarr')],
        benchmark_sha256=digest(QTL / 'LCLs.txt'))))
    CACHE.mkdir(exist_ok=True)
    if path.exists():
        assert json.loads((CACHE / 'scores_manifest.json').read_text()) == provenance
        with np.load(path) as d:
            return d['scores'], d['scorable'], d['strict']
    scores = np.full((N_PAIRS, len(ARMS)), np.nan)
    scorable = np.zeros_like(scores, dtype=bool)
    strict = np.zeros_like(scores, dtype=bool)
    for i, (name, family, f, bp) in enumerate(ARMS):
        win = read_predictions(f, bp)
        assert np.isfinite(win).all() and (win >= 0).all(), name
        scores[:, i], scorable[:, i], strict[:, i] = abs_logfc(win)
        progress(f'{name}: scorable {int(scorable[:, i].sum())}, strict {int(strict[:, i].sum())}')
    np.savez_compressed(path, scores=scores, scorable=scorable, strict=strict)
    atomic_json(CACHE / 'scores_manifest.json', provenance)
    return scores, scorable, strict


def pair_table():
    """One row per variant-gene pair, keeping the original benchmark row index.

    Repeated variants and repeated genes keep separate pair IDs; distance is the
    poster's own `abs(ensembl TSS - (qtl_loc - 1))`, left untouched so the
    distance-stratified panel stays comparable.
    """
    q, bench = benchmark_gene_table()
    assert len(q) == N_PAIRS, f'{len(q)} benchmark rows, expected {N_PAIRS}'
    assert set(q.label.unique()) == {0, 1}, f'unexpected labels {sorted(q.label.unique())}'
    q = q.reset_index().rename(columns={'index': 'pair_row'})
    q['pair_id'] = q.chrom_name + ':' + q.qtl_loc.astype(str) + '|' + q.gene_id
    poster_tss = bench.set_index('gene_id').poster_tss
    q['distance'] = np.abs(poster_tss.reindex(q.gene_id).to_numpy() - (q.qtl_loc.to_numpy() - 1))
    q['distance_bin'] = 'beyond'
    for nm, lo, hi in DISTANCE_BINS:
        q.loc[(q.distance >= lo) & (q.distance < hi), 'distance_bin'] = nm
    return q


# --------------------------------------------------------------------------
# Expression specificity
# --------------------------------------------------------------------------
def sense_expression_matrix(genes, bios, cage):
    """Derive gene x biosample matching-strand CAGE from the existing column cache.

    This reads no source Zarr. Source column positions, not target IDs, index the
    cached matrix. The combined-strand cache is retained as a sensitivity readout.
    """
    path = CACHE / 'gene_cage_sense_biosamples.npy'
    manifest = CACHE / 'sense_expression_manifest.json'
    config = dict(version=1, pooling='matching annotated gene strand; per-gene promoter union',
                  columns=stamp(CACHE / 'gene_cage_columns.npy'),
                  genes_sha256=digest(CACHE / 'genes.tsv.gz'),
                  tracks_sha256=digest(CACHE / 'cage_tracks.tsv'),
                  biosamples_sha256=digest(CACHE / 'biosamples.tsv'))
    if path.exists() and manifest.exists() and json.loads(manifest.read_text()) == config:
        x = np.load(path, mmap_mode='r')
        assert x.shape == (len(genes), len(bios))
        return x
    columns = np.load(CACHE / 'gene_cage_columns.npy', mmap_mode='r')
    assert columns.shape == (len(genes), len(cage))
    assert genes.strand.isin(['+', '-']).all()
    lookup = cage.reset_index(names='column').pivot(index='description', columns='strand', values='column')
    lookup = lookup.reindex(bios)
    assert lookup[['+', '-']].notna().all().all(), 'Each biosample must have one column per strand'
    x = np.empty((len(genes), len(bios)), dtype=np.float32)
    for strand in ['+', '-']:
        rows = np.flatnonzero(genes.strand.eq(strand))
        indices = lookup[strand].to_numpy(dtype=np.int64)
        x[rows] = columns[np.ix_(rows, indices)]
    assert np.isfinite(x).all() and (x >= 0).all()
    tmp = path.with_name(path.stem + '.tmp.npy')
    np.save(tmp, x); tmp.replace(path)
    atomic_json(manifest, config)
    return x


def expression_matrix(strand_mode=STRAND_MODE):
    """Processed promoter signal per million over covered protein-coding genes.

    Normalization is recomputed independently for each readout/biosample. This is
    relative processed CAGE signal, not raw library CPM or whole-gene RNA abundance.
    Non-covered genes return NaN and cannot become low-expression controls.
    """
    genes = validate_expression_cache()
    bios = pd.read_csv(CACHE / 'biosamples.tsv', sep='\t').biosample.to_numpy()
    cage = pd.read_csv(CACHE / 'cage_tracks.tsv', sep='\t')
    if strand_mode == 'sense':
        x = sense_expression_matrix(genes, bios, cage)
    elif strand_mode == 'combined':
        x = np.load(CACHE / 'gene_cage_biosamples.npy', mmap_mode='r')
    else:
        raise ValueError('strand_mode must be sense or combined')
    assert x.shape == (len(genes), len(bios))
    assert np.isfinite(x).all() and (x >= 0).all()
    background = genes.is_background.to_numpy(dtype=bool) & genes.mapping_status.eq('covered').to_numpy()
    assert background.any(), 'No covered background genes'
    total = x[background].sum(axis=0, dtype=np.float64)
    assert (total > 0).all(), 'A biosample with no promoter signal cannot be normalized'
    cpm = np.asarray(x, dtype=np.float64) / total * 1e6
    cpm[~genes.mapping_status.eq('covered').to_numpy()] = np.nan
    return genes, bios, cpm, background


def classify_expression(genes, cpm, is_gm, lymphoid, *, high_cpm=GM_HIGH_CPM,
                        low_cpm=GM_LOW_CPM, low_percentile=OTHERS_LOW_PERCENTILE,
                        high_percentile=OTHERS_HIGH_PERCENTILE, exclude_lymphoid=False):
    """Pure classification, independent of benchmark labels and model predictions.

    Display groups split others mean at the GM-active background median.
    Strict directional and ratio groups remain separate secondary annotations.
    """
    assert 0 <= low_cpm < high_cpm
    assert 0 <= low_percentile < high_percentile <= 100
    is_gm, lymphoid = np.asarray(is_gm, bool), np.asarray(lymphoid, bool)
    assert cpm.shape == (len(genes), len(is_gm)) and lymphoid.shape == is_gm.shape
    assert is_gm.sum() == 1, 'Expected one GM12878 biosample'
    others = ~is_gm & (~lymphoid if exclude_lymphoid else True)
    assert others.any(), 'No comparison biosamples remain'
    covered = genes.mapping_status.eq('covered').to_numpy()
    background = covered & genes.is_background.to_numpy(dtype=bool)
    assert background.any() and np.isfinite(cpm[covered]).all() and (cpm[covered] >= 0).all()
    gm = cpm[:, is_gm].ravel().copy()
    avg = cpm[:, others].mean(axis=1)
    lo, hi = np.percentile(avg[background], [low_percentile, high_percentile])
    gm_high, gm_low = gm >= high_cpm, gm < low_cpm
    others_low, others_high = avg < lo, avg >= hi
    group = np.select([~covered,
                       gm_high & others_low, gm_low & others_high,
                       gm_high & others_high, gm_low & others_low],
                      ['missing', 'gm_high_others_low', 'gm_low_others_high',
                       'gm_high_others_high', 'low_in_both'], default='intermediate')
    # Retain the user's breadth/tau diagnostics under unambiguous names.
    breadth = (cpm[:, others] >= high_cpm).sum(axis=1).astype(float)
    reference = background & gm_high
    median = float(np.median(breadth[reference])) if reference.any() else np.nan
    panel = cpm[:, is_gm | others]
    peak = panel.max(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        tau = np.where(peak[:, 0] > 0,
                       (1 - panel / np.maximum(peak, 1e-12)).sum(axis=1) / (panel.shape[1] - 1), np.nan)
        contrast = np.log2((gm + 0.1) / (avg + 0.1))
    gm[~covered] = np.nan; avg[~covered] = np.nan
    breadth[~covered] = np.nan; tau[~covered] = np.nan; contrast[~covered] = np.nan
    out = genes.copy()
    out['gm12878_cpm'] = gm
    out['others_mean_cpm'] = avg
    out['gm12878_expressed'] = covered & gm_high
    out['directional_group'] = group
    ratio_median = float(np.median(contrast[reference])) if reference.any() else np.nan
    out['expression_group'] = np.select(
        [~covered, covered & gm_high & (contrast > ratio_median),
         covered & gm_high & (contrast <= ratio_median), covered & ~gm_high],
        ['missing', 'gm_expressed_more_specific', 'gm_expressed_broad', 'gm_not_expressed'],
        default='no_active_reference')
    others_median = float(np.median(avg[reference])) if reference.any() else np.nan
    out['specificity_group'] = np.select(
        [~covered, covered & gm_high & (avg < others_median),
         covered & gm_high & (avg >= others_median), covered & ~gm_high],
        ['missing', 'gm_active_others_lower', 'gm_active_others_higher', 'gm_not_expressed'],
        default='no_active_reference')
    out['others_median_gm_active_background'] = others_median
    out['specificity_ratio_median'] = ratio_median
    out['n_specificity_reference_genes'] = int(reference.sum())
    out['ratio_pseudocount_cpm'] = 0.1
    out['breadth'] = breadth
    out['tau'] = tau
    out['log2_gm_over_others'] = contrast
    out['n_other_biosamples'] = int(others.sum())
    out['gm_high_cutoff_cpm'] = high_cpm
    out['gm_low_cutoff_cpm'] = low_cpm
    out['others_low_cutoff_cpm'] = lo
    out['others_high_cutoff_cpm'] = hi
    out['others_low_percentile'] = low_percentile
    out['others_high_percentile'] = high_percentile
    out['n_background_genes'] = int(background.sum())
    out['median_breadth_gm_expressed'] = median
    out['median_breadth_all'] = float(np.median(breadth[background]))
    out['breadth_group'] = np.select(
        [~covered, covered & gm_high & (breadth <= median), covered & gm_high & (breadth > median)],
        ['missing', 'narrower_breadth_GM_high', 'broader_breadth_GM_high'], default='GM_not_high')
    out['cutoff_cpm'] = high_cpm
    out['lymphoid_excluded'] = exclude_lymphoid
    return out


def classify_genes(cutoff=CUTOFF_CPM, exclude_lymphoid=False, *, low_cpm=None,
                   low_percentile=OTHERS_LOW_PERCENTILE, high_percentile=OTHERS_HIGH_PERCENTILE,
                   strand_mode=STRAND_MODE):
    genes, bios, cpm, _ = expression_matrix(strand_mode)
    cage = pd.read_csv(CACHE / 'cage_tracks.tsv', sep='\t')
    is_gm = np.isin(bios, cage.description[cage.is_gm12878].unique())
    lymphoid = np.isin(bios, cage.description[cage.is_lymphoid].unique())
    out = classify_expression(genes, cpm, is_gm, lymphoid, high_cpm=cutoff,
                             low_cpm=cutoff / 5 if low_cpm is None else low_cpm,
                             low_percentile=low_percentile, high_percentile=high_percentile,
                             exclude_lymphoid=exclude_lymphoid)
    out['strand_mode'] = strand_mode
    return out


def sweep_cutoff(cutoffs=(10., 20., 50., 100.)):
    """Predeclared sensitivity settings; counts only, never select cutoffs by AP."""
    rows = []
    for mode in ['sense', 'combined']:
        for c in cutoffs:
            for excl in [False, True]:
                g = classify_genes(c, excl, strand_mode=mode)
                counts = g[g.is_benchmark].specificity_group.value_counts().to_dict()
                counts['gm_low_others_high'] = int((g.is_benchmark & g.directional_group.eq('gm_low_others_high')).sum())
                rows.append(dict(high_cpm=c, low_cpm=c / 5, strand_mode=mode,
                                 lymphoid_excluded=excl,
                                 others_low_cpm=float(g.others_low_cutoff_cpm.iloc[0]),
                                 others_high_cpm=float(g.others_high_cutoff_cpm.iloc[0]), **counts))
    out = pd.DataFrame(rows)
    for group in ['gm_active_others_lower', 'gm_active_others_higher', 'gm_not_expressed',
                  'gm_low_others_high', 'no_active_reference', 'missing']:
        if group not in out: out[group] = 0
        out[group] = out[group].fillna(0).astype(int)
    out.to_csv(CACHE / 'cutoff_sweep.tsv', sep='\t', index=False)
    return out


def annotate_pairs(cutoff=CUTOFF_CPM, exclude_lymphoid=False, **kwargs):
    """Join onto original benchmark rows; never substitute missing genes with zeros."""
    q = pair_table()
    g = classify_genes(cutoff, exclude_lymphoid, **kwargs)
    v = q.merge(g.rename(columns={'chrom': 'gene_chrom', 'strand': 'gene_strand',
                                  'mapping_status': 'gene_mapping_status'}),
                on='gene_id', how='left', validate='many_to_one', sort=False)
    assert len(v) == N_PAIRS and (v.pair_row.to_numpy() == np.arange(N_PAIRS)).all()
    v['gene_mapping_status'] = v.gene_mapping_status.fillna('absent_from_gencode')
    v['expression_group'] = v.expression_group.fillna('missing')
    v['directional_group'] = v.directional_group.fillna('missing')
    v['specificity_group'] = v.specificity_group.fillna('missing')
    known = v.gene_chrom.notna()
    assert (v.gene_chrom[known] == v.chrom_name[known]).all(), 'Gene chromosome mismatch'
    return v


def save_expression_annotations(genes):
    genes.to_csv(CACHE / 'gene_expression_specificity.tsv.gz', sep='\t', index=False)
    settings = ['others_median_gm_active_background', 'specificity_ratio_median', 'n_specificity_reference_genes', 'ratio_pseudocount_cpm', 'gm_high_cutoff_cpm', 'gm_low_cutoff_cpm', 'others_low_cutoff_cpm',
                'others_high_cutoff_cpm', 'others_low_percentile', 'others_high_percentile',
                'n_background_genes', 'n_other_biosamples', 'strand_mode', 'lymphoid_excluded']
    config = dict(version=4, method='Primary: continuous specificity adjusted for GM expression and distance; display: others mean split at active-background median',
                  primary_groups=['gm_active_others_lower', 'gm_active_others_higher'],
                  primary_group_column='specificity_group', ratio_group_column='expression_group',
                  extreme_group_column='directional_group',
                  thresholds=json.loads(genes[settings].iloc[0].to_json()),
                  background='display median: covered GM-active protein-coding genes; extreme percentiles: all covered protein-coding genes',
                  expression_cache_sha256=digest(CACHE / 'expression_manifest.json'),
                  sense_cache_sha256=digest(CACHE / 'sense_expression_manifest.json')
                  if (CACHE / 'sense_expression_manifest.json').exists() else None,
                  analysis_code_sha256=digest(Path(__file__)))
    atomic_json(CACHE / 'classification_manifest.json', config)
    return config


# --------------------------------------------------------------------------
# Stratified evaluation
# --------------------------------------------------------------------------
def metric_values(y, s):
    if len(y) == 0 or np.unique(y).size < 2:
        return np.nan, np.nan
    return average_precision_score(y, s), roc_auc_score(y, s)


SECONDARY_GROUPS = ['gm_high_others_low', 'gm_high_others_high', 'gm_low_others_high', 'low_in_both', 'intermediate']
GROUPS = ['gm_active_others_lower', 'gm_active_others_higher', 'gm_expressed_more_specific', 'gm_expressed_broad', 'gm_not_expressed', 'no_active_reference', 'all_rows', 'all_common', 'annotated', 'gm_high_others_low',
          'gm_high_others_high', 'gm_low_others_high', 'low_in_both', 'intermediate', 'missing']


def group_mask(v, common, group):
    if group == 'all_rows':
        return np.ones(len(v), bool)
    if group == 'all_common':
        return common.copy()
    if group == 'annotated':
        return common & v.expression_group.ne('missing').to_numpy()
    if group in ['gm_active_others_lower', 'gm_active_others_higher']:
        return common & v.specificity_group.eq(group).to_numpy() if 'specificity_group' in v else np.zeros(len(v), bool)
    column = 'directional_group' if group in SECONDARY_GROUPS and 'directional_group' in v else 'expression_group'
    return common & v[column].eq(group).to_numpy()


def bootstrap_ap(y, s_by_arm, clusters, n_boot, rng):
    """Cluster bootstrap over target genes, with the SAME draws for every arm.

    Pairs sharing a target gene are resampled together: they share the gene's
    expression annotation, often overlap in LD, and are not independent
    observations.  Shared draws make the per-arm columns directly differenceable.
    """
    unique, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(unique))]
    boot = np.full((n_boot, s_by_arm.shape[1]), np.nan)
    if len(unique) < 2 or np.unique(y).size < 2:
        return boot, len(unique)
    for b in range(n_boot):
        take = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        yy = y[take]
        if np.unique(yy).size < 2:
            continue
        for a in range(s_by_arm.shape[1]):
            boot[b, a] = average_precision_score(yy, s_by_arm[take, a])
    return boot, len(unique)


def stratified_metrics(v, scores, scorable, n_boot=400, seed=42, by_distance=False):
    """AP/AUROC per expression group, evaluated on identical rows across arms."""
    common = scorable.all(axis=1)
    y = v.label.to_numpy()
    assert np.isfinite(scores).all(), 'Nonfinite scores: resolve/report strict-support differences before evaluation'
    assert scores.shape == scorable.shape == (len(v), len(ARMS))
    rng = np.random.default_rng(seed)
    rows, diffs, draws = [], [], {}
    strata = [(g, None) for g in GROUPS]
    if by_distance:
        strata += [(g, d) for g in GROUPS for d, _, _ in DISTANCE_BINS]
    for group, dist in strata:
        mask = group_mask(v, common, group)
        if dist is not None:
            mask = mask & v.distance_bin.eq(dist).to_numpy()
        idx = np.flatnonzero(mask)
        boot, n_clusters = bootstrap_ap(y[idx], scores[idx], v.gene_id.to_numpy()[idx], n_boot, rng)
        observed = np.array([metric_values(y[idx], scores[idx, a])[0] for a in range(len(ARMS))])
        draws[(group, dist)] = dict(indices=idx, bootstrap=boot, observed=observed,
                                    genes=np.unique(v.gene_id.to_numpy()[idx]))
        prevalence = float(y[idx].mean()) if len(idx) else np.nan
        for a, (name, family, _, _) in enumerate(ARMS):
            ap, roc = metric_values(y[idx], scores[idx, a])
            good = np.isfinite(boot[:, a])
            ci = np.percentile(boot[good, a], [2.5, 97.5]) if good.any() else [np.nan, np.nan]
            rows.append(dict(stratum=group, distance_bin=dist or 'all', model=name, family=family,
                             n=len(idx), positives=int(y[idx].sum()),
                             negatives=int(len(idx) - y[idx].sum()),
                             n_genes=n_clusters,
                             n_variants=int(pd.unique(
                                 v.chrom_name.to_numpy()[idx] + ':' + v.qtl_loc.to_numpy()[idx].astype(str)).size)
                             if len(idx) else 0,
                             prevalence=prevalence, AP=ap, AUROC=roc,
                             AP_minus_prevalence=ap - prevalence,
                             AP_ci_low=ci[0], AP_ci_high=ci[1],
                             median_distance=float(np.median(v.distance.to_numpy()[idx])) if len(idx) else np.nan,
                             valid_bootstraps=int(good.sum())))
            # Paired difference against the reference arm, on the shared draws.
            ok = np.isfinite(boot[:, a]) & np.isfinite(boot[:, 0])
            d = boot[ok, 0] - boot[ok, a]
            ci = np.percentile(d, [2.5, 97.5]) if len(d) else [np.nan, np.nan]
            diffs.append(dict(stratum=group, distance_bin=dist or 'all', reference=NAMES[0],
                              model=name, delta_AP=metric_values(y[idx], scores[idx, 0])[0] - ap,
                              ci_low=ci[0], ci_high=ci[1], valid_bootstraps=len(d)))
    table = pd.DataFrame(rows)
    table.to_csv(CACHE / 'stratified_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(diffs).to_csv(CACHE / 'paired_model_differences.tsv', sep='\t', index=False)
    return table, draws


def interaction_test(draws, group_a, group_b, reference=0):
    """Difference-of-differences: does an arm's edge depend on the stratum?

    AP moves with prevalence and with the difficulty of the candidate set, so two
    different AP values across strata are not by themselves a model x specificity
    interaction.  What is testable is whether the GAP between two arms differs
    between strata, which is what this estimates:

        (AP_ref - AP_model)|group_a  -  (AP_ref - AP_model)|group_b

    The point estimate is the observed-data AP contrast, not the bootstrap mean.
    Both strata are resampled independently but with the same draws across arms,
    so the arm difference within a stratum is paired; the two strata are disjoint
    gene sets (asserted). LD/shared variants across genes are not modeled; these
    are descriptive contrasts and do not remove prevalence or cohort confounding.
    """
    aa, ab = draws[(group_a, None)], draws[(group_b, None)]
    assert not np.intersect1d(aa['indices'], ab['indices']).size, 'Interaction strata must be disjoint'
    assert not np.intersect1d(aa['genes'], ab['genes']).size, 'Independent draws require disjoint gene clusters'
    ba, bb = aa['bootstrap'], ab['bootstrap']
    assert ba.shape == bb.shape and 0 <= reference < ba.shape[1]
    observed = (aa['observed'][reference] - aa['observed']) - (ab['observed'][reference] - ab['observed'])
    rows = []
    for a, name in enumerate(NAMES):
        da = ba[:, reference] - ba[:, a]
        db = bb[:, reference] - bb[:, a]
        ok = np.isfinite(da) & np.isfinite(db)
        d = da[ok] - db[ok]
        ci = np.percentile(d, [2.5, 97.5]) if len(d) else [np.nan, np.nan]
        rows.append(dict(group_a=group_a, group_b=group_b, reference=NAMES[reference], model=name,
                         difference_of_differences=float(observed[a]),
                         bootstrap_mean=float(np.mean(d)) if len(d) else np.nan,
                         ci_low=ci[0], ci_high=ci[1], valid_bootstraps=int(len(d)),
                         crosses_zero=bool(ci[0] <= 0 <= ci[1]) if len(d) else None,
                         CI_available=bool(len(d))))
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / f'interaction_{group_a}_vs_{group_b}.tsv', sep='\t', index=False)
    return out


def matched_sensitivity(v, scores, scorable, groups=None, caliper=0.5, seed=7):
    """Repeat the secondary extreme contrast with genes matched on GM expression.

    Match nearest unmatched genes on log2 GM CPM within a fixed caliper.
    Expression imbalance is a possible confound, not proof of its causal effect.
    This greedy match is a sensitivity check and need not maximize sample size.

    Matching is on GENES, not variant-gene pairs: the confound is a property of the
    gene, and matching pairs would let one gene be drawn several times.  It uses no
    labels and no model scores.
    """
    groups = list(EXTREME if groups is None else groups)
    common = scorable.all(axis=1)
    rng = np.random.default_rng(seed)
    genes = (v[common & v.directional_group.isin(groups).to_numpy()]
             .drop_duplicates('gene_id')[['gene_id', 'directional_group', 'gm12878_cpm']])
    genes = genes[np.isfinite(genes.gm12878_cpm.to_numpy())]
    key = np.log2(genes.gm12878_cpm.to_numpy() + 0.1)
    side = {g: genes.index[genes.directional_group.eq(g)].to_numpy() for g in groups}
    small, large = sorted(groups, key=lambda g: len(side[g]))
    pos = pd.Series(key, index=genes.index)
    pool = list(side[large])
    matches = []
    # Deterministic given the seed; order is shuffled so ties do not follow gene order.
    for i in rng.permutation(side[small]):
        if not pool:
            break
        d = np.abs(pos[pool].to_numpy() - pos[i])
        j = int(np.argmin(d))
        if d[j] <= caliper:
            matches.append((i, pool.pop(j)))
    if not matches:
        progress('matched_sensitivity: no gene pairs within the caliper')
        return pd.DataFrame()
    pd.DataFrame([dict(gene_a=genes.loc[i, 'gene_id'], gene_b=genes.loc[j, 'gene_id'],
                       log2_gap=float(abs(pos[i]-pos[j])), seed=seed, caliper=caliper)
                  for i,j in matches]).to_csv(CACHE / f'matched_extremes_genes_caliper_{caliper:g}_seed_{seed}.tsv', sep='\t', index=False)
    keep_genes = set(genes.gene_id[[a for a, _ in matches]]) | set(genes.gene_id[[b for _, b in matches]])
    sel = common & v.gene_id.isin(keep_genes).to_numpy() & v.directional_group.isin(groups).to_numpy()
    y = v.label.to_numpy()
    rows = []
    for g in groups:
        mask = sel & v.directional_group.eq(g).to_numpy()
        idx = np.flatnonzero(mask)
        for a, name in enumerate(NAMES):
            ap, roc = metric_values(y[idx], scores[idx, a])
            rows.append(dict(stratum=g, model=name, n=len(idx), positives=int(y[idx].sum()),
                             prevalence=float(y[idx].mean()) if len(idx) else np.nan,
                             n_genes=int(v.gene_id[mask].nunique()),
                             median_gene_gm_cpm=float(v.loc[mask].drop_duplicates('gene_id').gm12878_cpm.median()),
                             median_gm_cpm=float(np.nanmedian(v.gm12878_cpm.to_numpy()[idx])) if len(idx) else np.nan,
                             AP=ap, AUROC=roc))
    out = pd.DataFrame(rows)
    out['matched_gene_pairs'] = len(matches)
    out['caliper_log2'] = caliper
    out.to_csv(CACHE / f'matched_extremes_caliper_{caliper:g}_seed_{seed}.tsv', sep='\t', index=False)
    wide = out.pivot(index='model', columns='stratum', values='AP').reindex(NAMES)
    wide['delta'] = wide[groups[0]] - wide[groups[1]]
    progress(f'matched sensitivity: {len(matches)} gene pairs; '
             f'median GM CPM ' + ' vs '.join(f'{out[out.stratum==g].median_gm_cpm.iloc[0]:.1f}' for g in groups) +
             f'; arms favouring {groups[0]}: {int((wide.delta > 0).sum())}/{len(NAMES)}')
    return out


def export_pairs(v, scores, scorable, strict):
    out = v.copy()
    out['common_support'] = scorable.all(axis=1)
    out['strict_support'] = strict.all(axis=1)
    for a, name in enumerate(NAMES):
        out[name] = np.where(scorable[:, a], scores[:, a], np.nan)
        out[name + ' (scorable)'] = scorable[:, a]
    out.to_csv(CACHE / 'pair_scores_and_specificity.tsv.gz', sep='\t', index=False)
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
# The median display split uses all eligible genes; extremes remain secondary.
PRESENT = ['gm_active_others_lower', 'gm_active_others_higher']
EXTREME = ['gm_high_others_low', 'gm_high_others_high']
FOCUSED = [(6, 4), (5, 4), (7, 4)]  # full FT, LoRA, scratch vs pretrained Borzoi
PRETTY = {'gm_active_others_lower': 'Higher specificity (GM active, lower others)',
          'gm_active_others_higher': 'Lower specificity (GM active, higher others)', 'gm_expressed_more_specific': 'Higher GM/others ratio (secondary)',
          'gm_expressed_broad': 'Lower GM/others ratio (secondary)',
          'gm_not_expressed': 'Below GM12878 activity cutoff',
          'no_active_reference': 'No active background reference', 'gm_high_others_low': 'Higher specificity\n(GM12878 active, others low)',
          'gm_low_others_high': 'GM12878 low / others high',
          'gm_high_others_high': 'Lower specificity\n(GM12878 active, others high)',
          'low_in_both': 'Low in GM12878 and others',
          'intermediate': 'Intermediate / outside the contrast',
          'missing': 'Missing or incomplete CAGE coverage',
          'all_common': 'All (common support)', 'annotated': 'All annotated', 'all_rows': 'All rows'}
MODEL_COLORS = ['#2878B5', '#69A6D2', '#164F7A', '#DD8452', '#218C74', '#56AB91',
                '#126553', '#86B99D', '#7853A6']


def plot_diagnostic(table, groups=PRESENT, name='relative_specificity'):
    """AP/AUROC and gene-bootstrap CIs; undefined metrics stay visibly undefined."""
    figures = CACHE / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    colors = ['#2878B5', '#DD8452', '#55A868', '#C44E52']
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, metric in zip(axes, ['AP', 'AUROC']):
        for off, group, color in zip(np.linspace(-.25, .25, len(groups)), groups, colors):
            d = table[(table.stratum == group) & (table.distance_bin == 'all')].set_index('model').loc[NAMES]
            label = f'{PRETTY[group]} (n={int(d.n.iloc[0])}, pos={int(d.positives.iloc[0])}, genes={int(d.n_genes.iloc[0])})'
            ax.barh(np.arange(len(d)) + off, d[metric], height=.48 / len(groups), color=color, label=label)
            if metric == 'AP':
                for j, r in enumerate(d.itertuples()):
                    if np.isfinite(r.AP_ci_low): ax.plot([r.AP_ci_low, r.AP_ci_high], [j + off] * 2, color='#333', lw=1)
                if np.isfinite(d.prevalence.iloc[0]): ax.axvline(d.prevalence.iloc[0], color=color, ls=':', lw=1)
        ax.set_xlabel('Average precision (dotted: prevalence)' if metric == 'AP' else 'AUROC')
        ax.set_xlim(0, 1); ax.spines[['top', 'right']].set_visible(False)
    axes[1].axvline(.5, color='gray', ls=':')
    axes[0].set_yticks(np.arange(len(NAMES)), NAMES); axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('eQTL prediction: target-gene relative specificity\n'
                 'Intervals resample genes; undefined metrics have no bar')
    fig.tight_layout(); fig.savefig(figures / f'{name}_diagnostic.png', dpi=180, bbox_inches='tight')
    return fig


def plot_presentation(table, groups=PRESENT, name='relative_specificity', xmin=None):
    """Lower specificity left, higher specificity right; shared scales."""
    figures = CACHE / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    order = list(groups)[::-1]
    selected = table[table.stratum.isin(order) & (table.distance_bin == 'all')]
    finite = selected.AP[np.isfinite(selected.AP)]
    lo = float(xmin) if xmin is not None and len(finite) and finite.min() >= xmin else 0.0
    hi = min(1.0, float(finite.max()) + .10) if len(finite) else 1.0
    fig, axes = plt.subplots(1, len(order), figsize=(7.6 * len(order), 6.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, order):
        d = table[(table.stratum == group) & (table.distance_bin == 'all')].set_index('model').loc[NAMES]
        ax.barh(np.arange(len(d)), d.AP - lo, left=lo, height=.62, color=MODEL_COLORS, zorder=3)
        for j, val in enumerate(d.AP):
            if np.isfinite(val): ax.text(val + .007, j, f'{val:.3f}', va='center', fontsize=9, color='#444')
        if not np.isfinite(d.AP).any():
            ax.text(.5, .5, 'AP not estimable\nNeeds both positive and negative pairs',
                    transform=ax.transAxes, ha='center', va='center', color='#555')
        ax.set_xlim(lo, hi)
        ax.set_title(f'{PRETTY[group]}\nn={int(d.n.iloc[0])}, positives={int(d.positives.iloc[0])}, genes={int(d.n_genes.iloc[0])}', fontsize=11)
        ax.set_xlabel('AUPRC (average precision)'); ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0); ax.grid(axis='x', color='#E7E7E7', zorder=0)
    axes[0].set_yticks(np.arange(len(NAMES)), NAMES, fontsize=9.5); axes[0].invert_yaxis()
    fig.suptitle('eQTL prediction by target-gene expression', fontsize=15)
    fig.text(.5, .01, f'Both axes start at {lo:.2f}. Point estimates; uncertainty omitted.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95])
    for ext in ['png', 'pdf']: fig.savefig(figures / f'{name}_presentation.{ext}', dpi=180, bbox_inches='tight')
    return fig


def continuous_design(scores, gm, others, distance):
    """Label-independent transforms, fixed over the common eligible cohort.

    Scores use standardized midranks so coefficient units are comparable across
    arms. Covariates are standardized log signals; specificity = -log2(others+.1).
    """
    from scipy.stats import rankdata
    def z(x):
        x = np.asarray(x, float)
        assert np.isfinite(x).all() and x.std() > 1e-10, 'Constant/nonfinite predictor'
        return (x - x.mean()) / x.std()
    q, e, d = z(-np.log2(others + .1)), z(np.log2(gm + .1)), z(np.log1p(distance))
    designs = []
    for score in scores.T:
        s = z(rankdata(score, method='average'))
        designs.append(np.column_stack([np.ones(len(s)), s, q, e, d, s*q, s*e, s*d]))
    return designs


def fit_cluster_logit(y, x, clusters):
    """Binomial GLM and gene-cluster influence contributions (CR1 correction).

    Sharing cluster ordering across models preserves covariance for paired
    coefficient contrasts. Checked against statsmodels cov_cluster in tests.
    """
    import statsmodels.api as sm
    import warnings
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    assert np.unique(y).size == 2 and np.linalg.matrix_rank(x) == x.shape[1]
    ids, inv = np.unique(clusters, return_inverse=True)
    n, k = x.shape; ng = len(ids)
    assert ng > 1 and n > k
    with warnings.catch_warnings():
        warnings.simplefilter('error', PerfectSeparationWarning)
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-9)
    assert fit.converged and np.isfinite(fit.params).all(), 'GLM did not converge'
    mu = fit.fittedvalues
    bread = np.linalg.inv((x.T * (mu * (1-mu))) @ x)
    cluster_scores = np.zeros((ng, k))
    np.add.at(cluster_scores, inv, x * (y-mu)[:, None])
    influence = cluster_scores @ bread
    influence *= np.sqrt(ng / (ng-1) * (n-1) / (n-k))
    return fit.params, influence, ids


def continuous_analysis(v, scores, scorable):
    """Conditional score-specificity association; not adjusted AP or causal effect."""
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    mask = (scorable.all(1) & v.gene_mapping_status.eq('covered').to_numpy()
            & v.gm12878_cpm.ge(GM_HIGH_CPM).to_numpy())
    idx = np.flatnonzero(mask); sub = v.iloc[idx]
    y = sub.label.to_numpy(); clusters = sub.gene_id.to_numpy()
    assert np.isfinite(scores[idx]).all() and np.unique(y).size == 2
    xlist = continuous_design(scores[idx], sub.gm12878_cpm.to_numpy(),
                              sub.others_mean_cpm.to_numpy(), sub.distance.to_numpy())
    terms = ['intercept', 'score_rank_z', 'specificity_z', 'gm_log_z', 'distance_log_z',
             'score_x_specificity', 'score_x_gm', 'score_x_distance']
    rows, fitted = [], []
    for name, x in zip(NAMES, xlist):
        beta, influence, ids = fit_cluster_logit(y, x, clusters)
        se = np.sqrt((influence**2).sum(0)); fitted.append((beta, influence))
        for j, term in enumerate(terms):
            rows.append(dict(model=name, term=term, coefficient=beta[j], se=se[j],
                             ci_low=beta[j]-1.96*se[j], ci_high=beta[j]+1.96*se[j],
                             p_value=2*norm.sf(abs(beta[j]/se[j])), n=len(idx),
                             positives=int(y.sum()), n_genes=len(ids), converged=True))
    coefficients = pd.DataFrame(rows)
    comparisons = []
    for a, b in FOCUSED:
        delta = fitted[a][0][5] - fitted[b][0][5]
        # SAME genes and score standardization cohort: covariance is not discarded.
        se = np.linalg.norm(fitted[a][1][:, 5] - fitted[b][1][:, 5])
        p = 2*norm.sf(abs(delta/se)) if se > 0 else (1. if delta == 0 else np.nan)
        comparisons.append(dict(model=NAMES[a], reference=NAMES[b], delta_interaction=delta,
                                se=se, ci_low=delta-1.96*se, ci_high=delta+1.96*se, p_value=p))
    comparisons = pd.DataFrame(comparisons)
    comparisons['p_holm_three_contrasts'] = multipletests(comparisons.p_value, method='holm')[1]
    coefficients.to_csv(CACHE / 'continuous_coefficients.tsv', sep='\t', index=False)
    comparisons.to_csv(CACHE / 'continuous_focused_comparisons.tsv', sep='\t', index=False)
    sub[['pair_row', 'gene_id', 'label', 'gm12878_cpm', 'others_mean_cpm', 'distance']].to_csv(
        CACHE / 'continuous_cohort.tsv', sep='\t', index=False)
    atomic_json(CACHE / 'continuous_manifest.json', dict(
        version=1, n=len(idx), positives=int(y.sum()), n_genes=len(ids),
        gm_active_cpm=GM_HIGH_CPM, specificity='z(-log2(others_mean_CPM+0.1))',
        score='z(midranks), separately per arm on the same eligible pair cohort',
        gm='z(log2(GM_CPM+0.1))', distance='z(log1p(distance_bp))', terms=terms,
        inference='binomial GLM; gene-cluster CR1 sandwich; normal 95% intervals; paired cluster influences',
        comparison_units='difference of standardized score-by-specificity log-odds coefficients; not AP',
        focused=[dict(model=NAMES[a], reference=NAMES[b]) for a,b in FOCUSED],
        limitations='Exploratory after prior threshold inspection; parametric association, not causal or held-out performance; cross-gene LD not modeled',
        code_sha256=digest(Path(__file__)), classification_sha256=digest(CACHE / 'classification_manifest.json')))
    return coefficients, comparisons


def focused_ap_comparisons(draws, groups=None):
    """Selected paired AP gaps and their difference between display groups."""
    groups = PRESENT if groups is None else groups
    aa, bb = [draws[(g, None)] for g in groups]
    assert not np.intersect1d(aa['genes'], bb['genes']).size
    rows = []
    for a,b in FOCUSED:
        obs = []; boot = []
        for g, d in zip(groups, [aa, bb]):
            obs.append(d['observed'][a]-d['observed'][b])
            boot.append(d['bootstrap'][:, a]-d['bootstrap'][:, b])
        for label, point, samples in zip([*groups, 'higher_minus_lower'], [*obs, obs[0]-obs[1]], [*boot, boot[0]-boot[1]]):
            good = samples[np.isfinite(samples)]
            lo, hi = np.percentile(good, [2.5,97.5]) if len(good) else (np.nan,np.nan)
            rows.append(dict(model=NAMES[a], reference=NAMES[b], contrast=label,
                             delta_AP=point, ci_low=lo, ci_high=hi, valid_bootstraps=len(good)))
    out = pd.DataFrame(rows); out.to_csv(CACHE / 'focused_AP_comparisons.tsv', sep='\t', index=False)
    return out


def plot_continuous(coefficients, comparisons):
    """Forest plot of conditional interaction estimates, in log-odds units."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios':[1.1,1]})
    data = coefficients[coefficients.term.eq('score_x_specificity')].set_index('model').loc[NAMES]
    for ax, d, value, labels in [(axes[0], data, 'coefficient', NAMES),
                               (axes[1], comparisons, 'delta_interaction', ['Full FT − pretrained', 'LoRA − pretrained', 'Scratch − pretrained'])]:
        for i, row in enumerate(d.itertuples()):
            ax.plot([row.ci_low, row.ci_high], [i,i], color='#555', lw=1.5)
            ax.scatter(getattr(row, value), i, color=MODEL_COLORS[i] if ax is axes[0] else '#218C74', s=35)
        ax.set_yticks(range(len(d)), labels); ax.invert_yaxis(); ax.axvline(0,color='#888',ls='--')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Score × specificity, adjusted for GM expression and distance')
    axes[0].set_xlabel('Interaction coefficient (standardized predictors; log odds)')
    axes[1].set_title('Focused Borzoi comparisons (paired by gene)')
    axes[1].set_xlabel('Difference in interaction coefficients')
    fig.suptitle('Continuous specificity analysis — all eligible pairs')
    fig.text(.5,.01,'95% gene-cluster intervals. Conditional associations, not AUPRC or causal effects.',ha='center')
    fig.tight_layout(rect=[0,.04,1,.95])
    for ext in ['png','pdf']: fig.savefig(CACHE / 'figures' / f'continuous_specificity.{ext}',dpi=180,bbox_inches='tight')
    return fig


def main():
    scores, scorable, strict = score_pairs()
    progress(f'Common support {int(scorable.all(axis=1).sum())} / {N_PAIRS}; strict {int(strict.all(axis=1).sum())}')
    assert np.array_equal(scorable.all(axis=1), strict.all(axis=1)), 'Poster and finite-score cohorts differ; investigate'
    g = classify_genes(); save_expression_annotations(g)
    v = annotate_pairs()
    table, draws = stratified_metrics(v, scores, scorable, by_distance=True)
    interaction_test(draws, *PRESENT)
    focused_ap_comparisons(draws)
    coefficients, comparisons = continuous_analysis(v, scores, scorable)
    plot_continuous(coefficients, comparisons)
    export_pairs(v, scores, scorable, strict); sweep_cutoff()
    plot_diagnostic(table); plot_presentation(table, xmin=.30)
    progress(table[table.distance_bin == 'all'][['stratum', 'n', 'positives', 'n_genes']].drop_duplicates().to_string(index=False))


if __name__ == '__main__':
    main()
