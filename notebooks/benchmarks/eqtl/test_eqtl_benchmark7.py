"""Checks for the eQTL specificity pipeline. Uses synthetic stores, not real data."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import numcodecs
import pytest

import eqtl_benchmark7_pipeline as p


def make_spans(rows):
    s = pd.DataFrame(rows, columns=['chrom', 'start', 'fold', 'row'])
    s['end'] = s.start + p.SPAN_BP
    return s[['chrom', 'start', 'end', 'fold', 'row']]


def test_promoter_window_is_strand_aware():
    tss = pd.DataFrame(dict(gene_id=['a', 'b'], chrom=['chr1'] * 2, strand=['+', '-'],
                            tss0=[10_000, 10_000], gene_type=['protein_coding'] * 2,
                            is_benchmark=[False] * 2, is_background=[True] * 2))
    lo, hi = p.promoter_windows(tss)
    # '+' reaches 300 bp upstream (lower coordinate); '-' reaches 300 bp upstream (higher).
    assert (lo[0], hi[0]) == (10_000 - 300, 10_000 + 100)
    assert (lo[1], hi[1]) == (10_000 - 100, 10_000 + 300)


def test_merge_windows_deduplicates_overlaps_but_keeps_distant_tss():
    tss = pd.DataFrame(dict(gene_id=['g', 'g', 'g'], chrom=['chr1'] * 3, strand=['+'] * 3,
                            tss0=[10_000, 10_150, 900_000], gene_type=['protein_coding'] * 3,
                            is_benchmark=[False] * 3, is_background=[True] * 3))
    u = p.merge_gene_windows(tss)
    assert len(u) == 2
    # The two nearby promoters overlap, so their union is counted once, not twice.
    assert (u.lo.iloc[0], u.hi.iloc[0]) == (10_000 - 300, 10_150 + 100)
    assert u.unit_bp.iloc[0] < 400 * 2
    assert (u.lo.iloc[1], u.hi.iloc[1]) == (900_000 - 300, 900_000 + 100)


def test_merge_never_produces_a_unit_wider_than_a_span():
    n = 40
    tss = pd.DataFrame(dict(gene_id=['g'] * n, chrom=['chr1'] * n, strand=['+'] * n,
                            tss0=np.arange(n) * 10_000, gene_type=['protein_coding'] * n,
                            is_benchmark=[False] * n, is_background=[True] * n))
    u = p.merge_gene_windows(tss)
    assert (u.unit_bp <= p.SPAN_BP).all()


def test_assign_spans_is_minimal_and_only_uses_covering_spans():
    spans = make_spans([('chr1', 0, 'fold0', 0), ('chr1', 50_000, 'fold0', 1),
                        ('chr1', 100_000, 'fold1', 0), ('chr2', 0, 'fold0', 2)])
    w = pd.DataFrame(dict(chrom=['chr1', 'chr1', 'chr1', 'chr3'],
                          lo=[1_000, 120_000, 150_000, 10],
                          hi=[1_400, 120_400, 150_400, 410]))
    got = p.assign_spans(w, spans)
    for i in range(3):
        s = spans.loc[got[i]]
        assert s.start <= w.lo[i] and w.hi[i] <= s.end
    # One span covers windows 1 and 2, so the optimal stabbing set reuses it.
    assert got[1] == got[2]
    assert got[3] == -1, 'a window on a contig with no span must stay uncovered'


def test_assign_spans_minimises_reads_on_a_dense_chromosome():
    spans = make_spans([('chr1', i * 20_000, 'fold0', i) for i in range(60)])
    w = pd.DataFrame(dict(chrom=['chr1'] * 50, lo=np.arange(50) * 20_000 + 5_000))
    w['hi'] = w.lo + 400
    got = p.assign_spans(w, spans)
    assert (got >= 0).all()
    # 50 windows spread over 1 Mb fit inside far fewer than 50 of the 196 kb spans.
    assert len(np.unique(got)) <= 8


def test_attach_span_bin_arithmetic_covers_the_request():
    # Deliberately unaligned start: real span starts are spread over all residues mod 32.
    spans = make_spans([('chr1', 1_000_005, 'fold0', 7)])
    w = pd.DataFrame(dict(chrom=['chr1'], lo=[1_010_000], hi=[1_010_400]))
    a = p.attach_span(w, spans)
    r = a.iloc[0]
    assert r.covered and r.fold == 'fold0' and r.span_row == 7
    lo_bp = spans.start[0] + r.bin_lo * p.BIN
    hi_bp = spans.start[0] + r.bin_hi * p.BIN
    assert lo_bp <= r.lo and hi_bp >= r.hi, 'bins must cover the whole requested window'
    assert hi_bp - lo_bp - (r.hi - r.lo) < 2 * p.BIN, 'at most two partial edge bins'
    assert r.covered_bp == r.bin_hi * p.BIN - r.bin_lo * p.BIN


def write_zarr(path, array, chunks):
    path.mkdir(parents=True, exist_ok=True)
    codec = numcodecs.Blosc(cname='zlib', clevel=1, shuffle=2)
    meta = dict(zarr_format=2, order='C', filters=None, fill_value=0.0,
                shape=list(array.shape), chunks=list(chunks), dtype=array.dtype.str,
                compressor=codec.get_config())
    (path / '.zarray').write_text(__import__('json').dumps(meta))
    for i in range(array.shape[0]):
        block = np.ascontiguousarray(array[i:i + 1])
        (path / f'{i}.0.0').write_bytes(codec.encode(block))
    return meta


def test_extract_reads_the_right_bins_and_columns():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        original = (p.BORZOI, p.CACHE, p.SPAN_BINS, p.N_TARGETS, p.WORKERS)
        try:
            p.CACHE = d / 'cache'
            p.CACHE.mkdir()
            p.SPAN_BINS, p.N_TARGETS, p.WORKERS = 64, 6, 1
            p.SPAN_BP = p.SPAN_BINS * p.BIN
            p.BORZOI = d / 'borzoi.zarr'
            # Value encodes (bin, target) so a misread bin or column is detectable.
            arr = np.arange(2 * p.SPAN_BINS * p.N_TARGETS, dtype=np.float16
                            ).reshape(2, p.SPAN_BINS, p.N_TARGETS) % 2048
            meta = write_zarr(p.BORZOI / 'fold0', arr, (1, p.SPAN_BINS, p.N_TARGETS))
            cage = pd.DataFrame(dict(target_index=[1, 4], description=['CAGE:x +', 'CAGE:x -']))
            spans = make_spans([('chr1', 0, 'fold0', 0), ('chr1', 320, 'fold0', 1)])
            units = p.attach_span(pd.DataFrame(dict(chrom=['chr1'], lo=[64], hi=[192])), spans)
            units['gene_id'] = 'g'
            raw = p.attach_span(pd.DataFrame(dict(chrom=['chr1'], lo=[0], hi=[64])), spans)
            sums, raw_out = p.extract(units, raw, cage, {'fold0': meta})
            span0 = arr[units.span.iloc[0]]
            expected = span0[units.bin_lo.iloc[0]:units.bin_hi.iloc[0]][:, [1, 4]].astype(np.float32).sum(axis=0)
            np.testing.assert_allclose(sums[0], expected, rtol=1e-6)
            r = raw.iloc[0]
            np.testing.assert_array_equal(
                raw_out[0][:r.bin_hi - r.bin_lo],
                arr[r.span][r.bin_lo:r.bin_hi][:, [1, 4]])
        finally:
            p.BORZOI, p.CACHE, p.SPAN_BINS, p.N_TARGETS, p.WORKERS = original
            p.SPAN_BP = p.SPAN_BINS * p.BIN


def test_uncovered_windows_are_missing_not_zero():
    spans = make_spans([('chr1', 0, 'fold0', 0)])
    w = pd.DataFrame(dict(chrom=['chr1'], lo=[10_000_000], hi=[10_000_400]))
    a = p.attach_span(w, spans)
    assert not a.covered.iloc[0] and a.bin_lo.iloc[0] == -1 and a.covered_bp.iloc[0] == 0


# --------------------------------------------------------------------------
import eqtl_benchmark7_analysis as an


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    # Synthetic tests must never overwrite real benchmark tables or logs.
    monkeypatch.setattr(an, 'CACHE', tmp_path)
    monkeypatch.setattr(p, 'CACHE', tmp_path)



def test_abs_logfc_matches_the_poster_rule():
    win = np.zeros((4, 8, 2))
    win[0, :, 0] = 4.0; win[0, :, 1] = 1.0     # ref 4x alt -> |log2 4| = 2
    win[1, :, :] = 0.0                          # both empty -> unscored, score 0
    win[2, :, 0] = 1.0                          # alt empty -> scored by the poster rule
    win[3, :, 0] = 2.0; win[3, :, 1] = 2.0     # no effect -> exactly 0, and kept
    s, scorable, strict = an.abs_logfc(win)
    assert s[0] == pytest.approx(2.0)
    assert not scorable[1] and s[1] == 0.0
    # The poster marks an arm unscored only when BOTH sides vanish, so a one-sided
    # zero stays "scorable" with an infinite ratio; the strict mask separates it.
    assert scorable[2] and not strict[2]
    assert s[3] == 0.0 and scorable[3] and strict[3], 'true zero effects must be retained'


def test_bootstrap_uses_identical_draws_for_every_arm():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(30, int), np.zeros(30, int)]
    s = rng.normal(size=(60, 3))
    s[:, 1] = s[:, 0]                            # arm 1 duplicates arm 0
    clusters = np.repeat(np.arange(20), 3)
    boot, n = an.bootstrap_ap(y, s, clusters, 40, np.random.default_rng(1))
    assert n == 20
    ok = np.isfinite(boot[:, 0])
    np.testing.assert_allclose(boot[ok, 0], boot[ok, 1])
    assert not np.allclose(boot[ok, 0], boot[ok, 2])


def test_bootstrap_resamples_whole_genes():
    # Every pair of a gene must move together, so a gene-constant label can never
    # be split across a resample.
    y = np.repeat([1, 0], 20)
    s = np.repeat(np.arange(40), 1).reshape(-1, 1).astype(float)
    clusters = np.repeat(np.arange(8), 5)
    boot, n = an.bootstrap_ap(y, s, clusters, 25, np.random.default_rng(3))
    assert n == 8 and np.isfinite(boot[:, 0]).any()


def test_interaction_uses_observed_estimate_not_bootstrap_mean():
    a_ap = np.full(len(an.NAMES), .5); a_ap[0], a_ap[1] = .6, .4
    b_ap = np.full(len(an.NAMES), .5); b_ap[0], b_ap[1] = .6, .5
    observed_a = a_ap.copy(); observed_a[0] = .65
    draws = {('a', None): dict(indices=np.arange(4), genes=['g1'], observed=observed_a,
                              bootstrap=np.tile(a_ap, (50, 1))),
             ('b', None): dict(indices=np.arange(4, 8), genes=['g2'], observed=b_ap,
                              bootstrap=np.tile(b_ap, (50, 1)))}
    out = an.interaction_test(draws, 'a', 'b').set_index('model')
    assert out.difference_of_differences.iloc[1] == pytest.approx(.15)
    assert out.bootstrap_mean.iloc[1] == pytest.approx(.1)
    assert out.difference_of_differences.iloc[0] == pytest.approx(0.)
    assert out.CI_available.all()


def test_group_mask_partitions_the_common_support():
    v = pd.DataFrame(dict(expression_group=['gm_high_others_low', 'gm_high_others_high',
                                            'gm_low_others_high', 'low_in_both', 'missing']))
    common = np.array([True, True, True, True, False])
    parts = [an.group_mask(v, common, g) for g in
             ['gm_high_others_low', 'gm_high_others_high', 'gm_low_others_high',
              'low_in_both', 'missing']]
    total = np.sum(parts, axis=0)
    assert (total <= 1).all(), 'expression groups must be mutually exclusive'
    np.testing.assert_array_equal(np.sum(parts, axis=0).astype(bool), common)


def test_span_row_stays_integral_when_some_windows_are_uncovered():
    """Regression: a NaN from an uncovered row once promoted span_row to float, and
    the float built a chunk path like "1316.0.0.0" that silently read as zeros."""
    spans = make_spans([('chr1', 0, 'fold0', 1316)])
    w = pd.DataFrame(dict(chrom=['chr1', 'chr2'], lo=[1_000, 1_000], hi=[1_400, 1_400]))
    a = p.attach_span(w, spans)
    assert a.span_row.dtype.kind == 'i', f'span_row must stay integral, got {a.span_row.dtype}'
    assert a.span_row.iloc[0] == 1316 and a.span_row.iloc[1] == -1
    assert '.'.join(str(int(i)) for i in (a.span_row.iloc[0], 0, 0)) == '1316.0.0'


def test_read_chunk_refuses_to_fabricate_zeros():
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / 'z.zarr'
        arr = np.ones((2, 4, 3), dtype=np.float16)
        meta = write_zarr(store, arr, (1, 4, 3))
        np.testing.assert_array_equal(p.read_chunk(store, meta, (0, 0, 0))[0], arr[0])
        # A float index is the bug signature and must fail loudly.
        with pytest.raises(AssertionError, match='non-integer'):
            p.read_chunk(store, meta, (0.5, 0, 0))
        # An absent chunk is a bug in this store, not sparsity, unless opted into.
        with pytest.raises(AssertionError, match='missing chunk'):
            p.read_chunk(store, meta, (5, 0, 0))
        assert not p.read_chunk(store, meta, (5, 0, 0), allow_missing=True).any()


def test_units_of_a_gene_can_never_share_a_bin():
    """Sums are over whole 32 bp bins, so units 1-31 bp apart would double-count an
    edge bin. Merging at a BIN-width gap makes that impossible at any grid offset."""
    tss = pd.DataFrame(dict(gene_id=['g'] * 3, chrom=['chr1'] * 3, strand=['+'] * 3,
                            # second window starts 10 bp after the first ends: no overlap,
                            # but the same 32 bp bin covers both edges.
                            tss0=[10_000, 10_410, 900_000], gene_type=['protein_coding'] * 3,
                            is_benchmark=[False] * 3, is_background=[True] * 3))
    u = p.merge_gene_windows(tss).sort_values('lo').reset_index(drop=True)
    gaps = u.lo.to_numpy()[1:] - u.hi.to_numpy()[:-1]
    assert (gaps >= p.BIN).all(), f'units {gaps} bp apart can share a bin'
    for offset in range(p.BIN):     # every possible span grid alignment
        spans = make_spans([('chr1', 900_000 - p.SPAN_BP // 2 + offset, 'fold0', 0)])
        a = p.attach_span(u, spans)
        cov = a[a.covered]
        occupied = [set(range(r.bin_lo, r.bin_hi)) for r in cov.itertuples()]
        for i in range(len(occupied)):
            for j in range(i + 1, len(occupied)):
                assert not (occupied[i] & occupied[j]), f'shared bin at offset {offset}'


def test_benchmark_chromosomes_use_plink_coding():
    """The benchmark codes X as 23, so a naive 'chr'+str() yields chr23 and every
    X-linked gene fails the GENCODE consistency check."""
    q, bench = p.benchmark_gene_table()
    assert set(bench.chrom_name) <= {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY', 'chrM'}, \
        sorted(set(bench.chrom_name) - {f'chr{i}' for i in range(1, 23)})
    assert (q.chrom == 23).any() and (q.chrom_name[q.chrom == 23] == 'chrX').all()


def directional_fixture():
    # Reference quantiles depend only on the 20 background genes, not these targets.
    bg_other = np.arange(20) * 10.
    background = np.column_stack([np.full(20, 50.), bg_other, bg_other, bg_other])
    targets = np.array([
        [100, 0, 0, 0],       # GM high / others low
        [0, 600, 0, 0],       # GM low / others high despite small binary breadth
        [100, 200, 200, 200], # high in both, never "others low"
        [0, 0, 0, 0],         # low in both
        [50, 50, 50, 50],     # intermediate
        [np.nan]*4,           # missing
        [100, 0, 0, 0],       # partial coverage is missing, even with high values
    ])
    g = pd.DataFrame(dict(gene_id=[f'g{i}' for i in range(27)],
                          is_background=[True]*20+[False]*7,
                          is_benchmark=[False]*20+[True]*7,
                          mapping_status=['covered']*25+['uncovered', 'partial']))
    return g, np.vstack([background, targets])


def test_primary_groups_use_directional_mean_not_breadth():
    genes, x = directional_fixture()
    # Cutoffs are passed explicitly: the fixture values encode boundary behaviour at
    # 100/20, so this must assert the classification logic, not the production policy
    # constant, which is free to move (see test_production_cutoffs).
    result = an.classify_expression(genes, x, [True, False, False, False], [True, False, False, False],
                                    high_cpm=100., low_cpm=20.)
    assert result.directional_group.iloc[20:].tolist() == [
        'gm_high_others_low', 'gm_low_others_high', 'gm_high_others_high',
        'low_in_both', 'intermediate', 'missing', 'missing']
    assert result.others_low_cutoff_cpm.iloc[0] == pytest.approx(95.)
    assert result.others_high_cutoff_cpm.iloc[0] == pytest.approx(171.)
    assert result.breadth.iloc[21] == 1  # inverse class does not require ubiquitous activity
    assert result[['gm12878_cpm', 'others_mean_cpm', 'breadth', 'tau']].iloc[25:].isna().all().all()
    assert an.PRESENT == ['gm_active_others_lower', 'gm_active_others_higher']


def test_threshold_reference_includes_GM_low_background_and_excludes_benchmark_only():
    genes, x = directional_fixture()
    gm = [True, False, False, False]; lymphoid = [True, False, False, False]
    first = an.classify_expression(genes, x, gm, lymphoid)
    x[:20, 0] = 0  # Every reference gene is now GM-low; quantiles must still work.
    x[20:25, 1:] = 1e9
    second = an.classify_expression(genes, x, gm, lymphoid)
    assert first.others_low_cutoff_cpm.equals(second.others_low_cutoff_cpm)
    assert first.others_high_cutoff_cpm.equals(second.others_high_cutoff_cpm)
    assert second.median_breadth_gm_expressed.isna().all()  # diagnostics cannot block primary classification


def test_directional_boundaries_and_lymphoid_exclusion():
    genes, x = directional_fixture()
    x[20] = [100, 95, 95, 95]  # Exactly median is NOT below the low cutoff.
    x[21] = [20, 200, 200, 200]  # Exactly low cutoff is NOT GM-low.
    cuts = dict(high_cpm=100., low_cpm=20.)   # the values these boundaries are written against
    r = an.classify_expression(genes, x, [1, 0, 0, 0], [1, 1, 0, 0], **cuts)
    assert r.directional_group.iloc[20] == 'intermediate'
    assert r.directional_group.iloc[21] == 'intermediate'
    r = an.classify_expression(genes, x, [1, 0, 0, 0], [1, 1, 0, 0], exclude_lymphoid=True, **cuts)
    assert r.n_other_biosamples.eq(2).all()
    with pytest.raises(AssertionError, match='No comparison'):
        an.classify_expression(genes, x, [1, 0, 0, 0], [1, 1, 1, 1], exclude_lymphoid=True, **cuts)


def test_sense_matrix_matches_gene_strand_and_metadata_order(tmp_path):
    genes = pd.DataFrame(dict(gene_id=['gplus', 'gminus'], strand=['+', '-']))
    # Intentionally scrambled biosample/strand column order.
    cage = pd.DataFrame(dict(description=['B', 'A', 'A', 'B'], strand=['-', '+', '-', '+']))
    bios = np.array(['A', 'B'])
    genes.to_csv(tmp_path/'genes.tsv.gz', sep='\t', index=False)
    cage.to_csv(tmp_path/'cage_tracks.tsv', sep='\t', index=False)
    pd.DataFrame({'biosample': bios}).to_csv(tmp_path/'biosamples.tsv', sep='\t', index=False)
    np.save(tmp_path/'gene_cage_columns.npy', np.array([[20, 2, 200, 3], [5, 100, 7, 150]], dtype=np.float32))
    expected = np.array([[2, 3], [7, 5]])
    np.testing.assert_array_equal(an.sense_expression_matrix(genes, bios, cage), expected)
    np.testing.assert_array_equal(an.sense_expression_matrix(genes, bios, cage), expected)  # cache reuse


def test_expression_normalization_uses_covered_background_only(tmp_path, monkeypatch):
    genes = pd.DataFrame(dict(gene_id=['g1', 'g2', 'target', 'missing'],
                              strand=['+']*4, is_background=[1, 1, 0, 1],
                              mapping_status=['covered']*3+['uncovered']))
    bios = pd.DataFrame({'biosample': ['GM12878', 'other']})
    bios.to_csv(tmp_path/'biosamples.tsv', sep='\t', index=False)
    pd.DataFrame({'description': []}).to_csv(tmp_path/'cage_tracks.tsv', sep='\t', index=False)
    x = np.array([[1., 2.], [3., 6.], [20., 40.], [1000., 2000.]])
    monkeypatch.setattr(an, 'validate_expression_cache', lambda: genes)
    monkeypatch.setattr(an, 'sense_expression_matrix', lambda *args: x)
    _, _, cpm, bg = an.expression_matrix()
    np.testing.assert_allclose(cpm[0], [250000., 250000.])
    assert np.isnan(cpm[3]).all()
    np.testing.assert_array_equal(bg, [True, True, False, False])


def test_completed_builder_never_rescans(tmp_path, monkeypatch):
    (tmp_path/'expression_manifest.json').write_text('{}')
    genes = pd.DataFrame({'gene_id': ['g']})
    validated = []
    def validate(validate_sources):
        validated.append(validate_sources); return genes
    def no_scan(*args, **kwargs):
        raise AssertionError('Completed cache unexpectedly rescanned source')
    monkeypatch.setattr(p, 'validate_expression_cache', validate)
    monkeypatch.setattr(p, 'cage_tracks', no_scan)
    monkeypatch.setattr(p, 'extract', no_scan)
    assert p.build_expression_cache() is genes
    assert validated == [True]


def test_empty_inverse_group_has_no_fabricated_metric_or_interval(tmp_path):
    n = 6
    v = pd.DataFrame(dict(label=[1, 0, 1, 0, 1, 0], gene_id=[f'g{i}' for i in range(n)],
                          chrom_name=['chr1']*n, qtl_loc=np.arange(n), distance=np.arange(n),
                          expression_group=['gm_high_others_low']*n))
    scores = np.tile(np.linspace(.1, .9, n)[:, None], (1, len(an.ARMS)))
    table, draws = an.stratified_metrics(v, scores, np.ones_like(scores, bool), n_boot=10)
    inverse = table[table.stratum == 'gm_low_others_high']
    assert inverse.n.eq(0).all() and inverse.AP.isna().all()
    interaction = an.interaction_test(draws, *an.PRESENT)
    assert interaction.difference_of_differences.isna().all()
    assert not interaction.CI_available.any() and interaction.crosses_zero.isna().all()
    fig = an.plot_presentation(table, xmin=.3)
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    an.plt.close(fig)


def test_notebook_cells_have_valid_source_and_directional_contrast():
    import nbformat, ast
    nb = nbformat.read(Path(__file__).with_name('eQTL_benchmark7.ipynb'), as_version=4)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    for c in cells:
        ast.parse(c.source)
    assert all('\n' in c.source for c in cells)
    assert any('an.interaction_test(draws, *an.PRESENT)' in c.source for c in cells)


def test_runner_rejects_invalid_stage_without_work():
    import run_eqtl_benchmark7 as runner
    with pytest.raises(SystemExit) as exc:
        runner.main(['invalid-stage'])
    assert exc.value.code == 2


def test_plink_chromosome_fix_also_accepts_legacy_cached_names():
    assert p.canonical_chromosome(23) == p.canonical_chromosome('chr23') == 'chrX'
    assert p.canonical_chromosome(24) == p.canonical_chromosome('chrY') == 'chrY'
    assert p.canonical_chromosome(1) == p.canonical_chromosome('chr1') == 'chr1'


def test_cached_scores_never_open_prediction_zarr_metadata(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(an, 'QTL', tmp_path)
    monkeypatch.setattr(an, 'N_PAIRS', 3)
    (tmp_path/'LCLs.txt').write_text('synthetic benchmark identity')
    settings = dict(arms=json.loads(json.dumps(an.ARMS)), center_bp=an.CENTER_BP,
                    benchmark_sha256=an.digest(tmp_path/'LCLs.txt'))
    (tmp_path/'scores_manifest.json').write_text(json.dumps(settings))
    scores = np.zeros((3, len(an.ARMS)))
    np.savez(tmp_path/'model_scores.npz', scores=scores, scorable=np.ones_like(scores, bool), strict=np.ones_like(scores, bool))
    def no_source(*args):
        raise AssertionError('Cache-only scoring opened original prediction metadata')
    monkeypatch.setattr(an, 'metadata', no_source)
    np.testing.assert_array_equal(an.score_pairs()[0], scores)


def test_expression_validator_rejects_incomplete_products(tmp_path):
    import json
    config = dict(bin_bp=p.BIN, span_bp=p.SPAN_BP, raw_window_bp=p.RAW_HALF_BP,
                  raw_bins=p.RAW_BINS, merge_gap_bp=p.BIN, background_biotype=p.BACKGROUND_BIOTYPE,
                  promoter_window=dict(upstream=p.UPSTREAM, downstream=p.DOWNSTREAM,
                                       strand_aware=True, pooling='sum of all bins overlapping the window'))
    (tmp_path/'expression_manifest.json').write_text(json.dumps(config))
    with pytest.raises(AssertionError, match='Incomplete expression cache'):
        p.validate_expression_cache()


def test_relative_split_uses_active_background_and_keeps_ties_together():
    g = pd.DataFrame(dict(gene_id=list('abcdefg'), is_background=[True]*4+[False]*3,
                          is_benchmark=[False]*4+[True]*3, mapping_status=['covered']*6+['partial']))
    # Only the first three reference genes are GM-active. Median ratio is zero.
    x = np.array([[100, 50], [100, 100], [100, 200], [0, 1e6],
                  [100, 90], [100, 100], [100, 0]], float)
    r = an.classify_expression(g, x, [1, 0], [1, 0])
    assert r.specificity_ratio_median.eq(0).all()
    assert r.n_specificity_reference_genes.eq(3).all()
    assert r.expression_group.tolist() == ['gm_expressed_more_specific', 'gm_expressed_broad',
        'gm_expressed_broad', 'gm_not_expressed', 'gm_expressed_more_specific', 'gm_expressed_broad', 'missing']
    # Benchmark-only changes cannot move the reference threshold.
    x[4:6, 1] = 1e9
    altered = an.classify_expression(g, x, [1, 0], [1, 0])
    assert altered.specificity_ratio_median.equals(r.specificity_ratio_median)
    # No GM-active background: do not silently invent a split.
    x[:4, 0] = 0
    empty = an.classify_expression(g, x, [1, 0], [1, 0])
    assert empty.specificity_ratio_median.isna().all()
    assert empty.expression_group.iloc[4:6].eq('no_active_reference').all()


def test_secondary_group_masks_do_not_replace_primary_groups():
    v = pd.DataFrame(dict(expression_group=['gm_expressed_more_specific', 'gm_expressed_broad', 'gm_not_expressed'],
                          directional_group=['gm_high_others_low', 'gm_high_others_high', 'gm_low_others_high']))
    common = np.array([True, False, True])
    assert an.group_mask(v, common, 'gm_expressed_more_specific').tolist() == [True, False, False]
    assert an.group_mask(v, common, 'gm_expressed_broad').tolist() == [False]*3
    assert an.group_mask(v, common, 'gm_low_others_high').tolist() == [False, False, True]


def test_production_cutoffs_keep_their_fivefold_gap():
    """The policy constants themselves, kept separate from the boundary-logic tests.

    20 CPM gates GM activity: every benchmark target had measurable expression
    variation in LCLs, so a gate that excludes most of them is measuring the wrong
    thing. The low gate stays a fifth of it, preserving the dead zone that keeps
    marginal genes out of both directional groups.
    """
    assert an.GM_HIGH_CPM == 20.0 and an.GM_LOW_CPM == 4.0
    assert an.GM_LOW_CPM * 5 == an.GM_HIGH_CPM
    assert an.CUTOFF_CPM == an.GM_HIGH_CPM
    assert an.GM_HIGH_CPM in an.sweep_cutoff.__defaults__[0], 'sweep must bracket the primary'


def test_primary_display_uses_others_median_and_keeps_extremes_secondary():
    assert an.PRESENT == ['gm_active_others_lower', 'gm_active_others_higher']
    assert an.EXTREME == ['gm_high_others_low', 'gm_high_others_high']
    assert an.OTHERS_LOW_PERCENTILE == 50 and an.OTHERS_HIGH_PERCENTILE == 90


def test_matched_sensitivity_balances_gm_expression_and_uses_no_labels():
    rng = np.random.default_rng(0)
    n = 120
    gm = np.r_[rng.uniform(20, 60, n // 2), rng.uniform(60, 400, n // 2)]
    v = pd.DataFrame(dict(
        gene_id=[f'g{i}' for i in range(n)],
        directional_group=['gm_high_others_low'] * (n // 2) + ['gm_high_others_high'] * (n // 2),
        gm12878_cpm=gm, label=rng.integers(0, 2, n)))
    scores = rng.normal(size=(n, len(an.NAMES)))
    scorable = np.ones((n, len(an.NAMES)), bool)
    out = an.matched_sensitivity(v, scores, scorable, caliper=0.5)
    assert not out.empty
    med = out.drop_duplicates('stratum').set_index('stratum').median_gm_cpm
    before = abs(np.median(gm[:n // 2]) - np.median(gm[n // 2:]))
    after = abs(med['gm_high_others_low'] - med['gm_high_others_high'])
    assert after < before, f'matching must reduce the GM imbalance ({after} vs {before})'
    assert after <= 2 ** 0.5 * med.min(), 'matched medians should sit within the caliper'
    # Shuffling the labels must not change which genes were matched.
    shuffled = v.assign(label=rng.permutation(v.label.to_numpy()))
    again = an.matched_sensitivity(shuffled, scores, scorable, caliper=0.5)
    assert again.matched_gene_pairs.iloc[0] == out.matched_gene_pairs.iloc[0]
    assert (again.n.to_numpy() == out.n.to_numpy()).all()


def test_others_median_split_preserves_all_active_and_ignores_benchmark_targets():
    g = pd.DataFrame(dict(gene_id=list('abcdef'), is_background=[True]*3+[False]*3,
                          mapping_status=['covered']*5+['partial']))
    x = np.array([[20, 10], [40, 50], [0, 1e6], [20, 29], [20, 30], [100, 0]], float)
    r = an.classify_expression(g, x, [1,0], [1,0])
    assert r.others_median_gm_active_background.eq(30).all()
    assert r.specificity_group.tolist() == ['gm_active_others_lower', 'gm_active_others_higher',
                                          'gm_not_expressed', 'gm_active_others_lower',
                                          'gm_active_others_higher', 'missing']
    x[3:5,1] = 1e8
    assert an.classify_expression(g, x, [1,0], [1,0]).others_median_gm_active_background.eq(30).all()


def test_continuous_cluster_covariance_matches_statsmodels_and_recovers_interaction():
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    from scipy.special import expit
    rng = np.random.default_rng(119); ng = 900; n = ng*2
    gm = np.repeat(np.exp(rng.normal(4, .6, ng)),2)
    others = np.repeat(np.exp(rng.normal(3, .9, ng)),2)
    distance = np.exp(rng.normal(8, 1, n)); score = rng.normal(size=(n,1))
    x = an.continuous_design(score, gm, others, distance)[0]
    y = rng.binomial(1, expit(x @ np.array([-1.7,.8,.1,.2,-.3,1.2,.1,.1])))
    clusters = np.repeat(np.arange(ng),2)
    beta, influence, ids = an.fit_cluster_logit(y, x, clusters)
    fit = sm.GLM(y,x,family=sm.families.Binomial()).fit(tol=1e-9)
    np.testing.assert_allclose(influence.T @ influence, cov_cluster(fit,clusters), rtol=1e-6, atol=1e-9)
    assert beta[5] - 1.96*np.linalg.norm(influence[:,5]) > 0
    assert len(ids)==ng
    # Monotone rescaling of model scores leaves midrank design unchanged.
    np.testing.assert_allclose(x, an.continuous_design(np.exp(score),gm,others,distance)[0])


def test_continuous_cohort_and_paired_identical_model_contrast(tmp_path):
    rng = np.random.default_rng(16); n=700
    v=pd.DataFrame(dict(pair_row=np.arange(n),gene_id=[f'g{i//2}' for i in range(n)],
                        gene_mapping_status=['covered']*(n-1)+['uncovered'],
                        gm12878_cpm=rng.uniform(25,150,n),others_mean_cpm=rng.uniform(1,200,n),
                        distance=rng.uniform(0,1e5,n),label=rng.binomial(1,.25,n)))
    scores=np.tile(rng.normal(size=(n,1)),(1,len(an.NAMES))); scorable=np.ones_like(scores,bool)
    scorable[0,0]=False;v.loc[1,'gm12878_cpm']=0
    (tmp_path/'classification_manifest.json').write_text('{}')
    coefficients, comparisons=an.continuous_analysis(v,scores,scorable)
    assert coefficients.n.eq(n-3).all()
    np.testing.assert_allclose(comparisons.delta_interaction,0,atol=1e-12)
    np.testing.assert_allclose(comparisons.se,0,atol=1e-12)
    assert comparisons.p_holm_three_contrasts.eq(1).all()
