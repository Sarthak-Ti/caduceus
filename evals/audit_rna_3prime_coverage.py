"""Measure strand-matched terminal coverage before choosing an E2G readout.

Reads the unsquashed bigWigs used to construct the Poisson training targets.
Signal means summed base coverage, not a count of distinct molecules. Gene bodies
may overlap, so pooled sums are gene-associated coverage, not a library total.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


ROOT = Path('/data1/lesliec/sarthak')
RNA = ROOT / 'data/DE_danwei/K562_gex'
INFO = ROOT / 'data/DE_danwei/k562_sc_rna_info_cellranger.json'
CRISPR = Path('/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv')
BOUNDS = Path('/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed')
WINDOWS = (1000, 5000, 10000, 20000, 50000, 100000)


def terminal_mass(values, strand, width):
    """values is a genomic-order gene-body vector; 3' is left on minus."""
    return float(values[-width:].sum(dtype=np.float64) if strand == '+'
                 else values[:width].sum(dtype=np.float64))


def main(outdir):
    outdir.mkdir(parents=True, exist_ok=False)
    info = json.loads(INFO.read_text())
    bounds = pd.read_csv(BOUNDS, sep='\t')
    names = {n: g for n, g in bounds.set_index('name').Ensembl_ID.to_dict().items()
             if g in info}
    pairs = pd.read_csv(CRISPR, sep='\t')
    pairs['ensg_id'] = pairs.measuredGeneSymbol.map(names)
    pairs = pairs.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)
    e2g_genes = set(pairs.ensg_id)
    context_genes = set()
    for r in pairs.itertuples():
        g = info[r.ensg_id]
        offset = 163840 if g['strand'] == '+' else 524288 - 1 - 163840
        start = g['tss'] - offset
        if r.chrom == g['chrom'] and r.chromStart >= start and r.chromEnd <= start + 524288:
            context_genes.add(r.ensg_id)
    # Validation genes offer an independent description of the training target.
    genes = e2g_genes | {g for g, x in info.items() if x['split'] == 'val'}
    tracks = {'+': pyBigWig.open(str(RNA / 'K562.rnastrand_reverse.bw')),
              '-': pyBigWig.open(str(RNA / 'K562.rnastrand_forward.bw'))}
    records = []
    for i, gene in enumerate(sorted(genes, key=lambda g: (info[g]['chrom'], info[g]['gene_start']))):
        g = info[gene]
        chrom, start, end, strand = g['chrom'], g['gene_start'], g['gene_end'], g['strand']
        assert 0 <= start < end <= tracks[strand].chroms(chrom), (gene, g)
        raw = tracks[strand].values(chrom, start, end, numpy=True)
        assert not np.isinf(raw).any() and not (raw < 0).any(), gene
        raw = np.nan_to_num(raw)
        # Match bigwig_to_npz.py followed by make_fixed_npz.py, including the cast.
        native = np.sqrt(1 + raw).astype(np.float16).astype(np.float32) - 1
        offset = 163840 if strand == '+' else 524288 - 1 - 163840
        model_start = g['tss'] - offset
        r = dict(ensg_id=gene, chrom=chrom, gene_start=start, gene_end=end,
                 strand=strand, gene_length=end-start, split=g['split'],
                 in_e2g=gene in e2g_genes, in_context_e2g=gene in context_genes,
                 body_in_model_context=start >= model_start and end <= model_start + 524288)
        for scale, v in [('raw', raw), ('native', native)]:
            total = float(v.sum(dtype=np.float64))
            r[f'{scale}_body_sum'] = total
            for width in WINDOWS:
                mass = terminal_mass(v, strand, width)
                r[f'{scale}_terminal_{width}_sum'] = mass
                r[f'{scale}_terminal_{width}_fraction'] = mass / total if total else np.nan
            if total:
                from_end = v[::-1] if strand == '+' else v
                cumulative = np.cumsum(from_end, dtype=np.float64)
                for fraction in (.5, .8, .9, .95):
                    r[f'{scale}_bp_for_{int(100*fraction)}pct'] = int(np.searchsorted(cumulative, fraction*total)) + 1
        records.append(r)
        if (i + 1) % 500 == 0:
            print(f'Coverage: {i+1}/{len(genes)} genes', flush=True)
    for track in tracks.values():
        track.close()
    table = pd.DataFrame(records)
    table.to_csv(outdir / 'per_gene_coverage.tsv', sep='\t', index=False)
    summaries = []
    for cohort, subset in [('e2g_in_context', table.in_context_e2g),
                           ('e2g_all', table.in_e2g), ('validation', table.split.eq('val'))]:
        for group, condition in [('all_nonzero', table.raw_body_sum.gt(0)),
                                 ('longer_than_10kb', table.raw_body_sum.gt(0) & table.gene_length.gt(10000)),
                                 ('longer_than_50kb', table.raw_body_sum.gt(0) & table.gene_length.gt(50000))]:
            d = table[subset & condition]
            for scale in ('raw', 'native'):
                for width in WINDOWS:
                    fractions = d[f'{scale}_terminal_{width}_fraction']
                    summaries.append(dict(cohort=cohort, gene_group=group, scale=scale,
                        terminal_bp=width, n_genes=len(d), median=fractions.median(),
                        p10=fractions.quantile(.1), p25=fractions.quantile(.25),
                        mean=fractions.mean(), fraction_genes_at_least_80pct=(fractions >= .8).mean(),
                        fraction_genes_at_least_90pct=(fractions >= .9).mean(),
                        pooled_coverage_fraction=d[f'{scale}_terminal_{width}_sum'].sum()/d[f'{scale}_body_sum'].sum()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / 'coverage_summary.tsv', sep='\t', index=False)
    provenance = dict(gene_info=str(INFO), raw_plus=str(RNA / 'K562.rnastrand_reverse.bw'),
        raw_minus=str(RNA / 'K562.rnastrand_forward.bw'), n_genes=len(table),
        e2g_in_context_genes=len(context_genes),
        zero_signal_e2g_genes=int((table.in_context_e2g & table.raw_body_sum.eq(0)).sum()),
        truncated_body_e2g_genes=int((table.in_context_e2g & ~table.body_in_model_context).sum()),
        description='Exact base coverage in annotation-defined gene bodies and terminal genomic windows; strand matched. No enhancer outcomes used to choose windows.')
    (outdir / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2), flush=True)
    print(summary[summary.cohort.eq('e2g_in_context') & summary.gene_group.ne('longer_than_50kb')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args().output_dir)
