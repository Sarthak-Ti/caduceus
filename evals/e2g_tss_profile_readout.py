"""Score Poisson RNA profiles after accessibility reduction at CRISPR elements.

Primary readout: sum of the strand-matched gene-body profile in its trained
sqrt(1+coverage)-1 scale. Also retain terminal 10/50/100 kb and TSS controls,
per-base inverse-transformed coverage estimates, and the independent count head.
The inverse of a conditional mean is not an unbiased estimate of raw coverage.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
LEGACY_CKPT = REPO / 'outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt'
E2G = ROOT / 'data/joint_playground/e2g'
LEGACY = E2G / 'k562_tss_sc_rna_poisson_ep35.npy'
CRISPR = Path('/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv')
BOUNDS = Path('/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed')
DISTANCES = E2G / 'ENCODE-rE2G_extended_CrossValidated_K562_EPCrisprBenchmark_ensemble_data_GRCh38.predictions.tsv.gz'
REGIONS = ('body', 'three_prime_10kb', 'three_prime_50kb', 'three_prime_100kb', 'tss_4kb')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def map_interval(start, end, window_start, length, flip):
    """Map a genomic half-open interval, retaining full-coverage information."""
    lo, hi = int(start) - window_start, int(end) - window_start
    full = 0 <= lo < hi <= length
    if flip:
        lo, hi = length - hi, length - lo
    return max(0, min(length, lo)), max(0, min(length, hi)), full


def gene_regions(gene):
    start, end = int(gene['gene_start']), int(gene['gene_end'])
    result = {'body': (start, end)}
    for width in (10000, 50000, 100000):
        result[f'three_prime_{width//1000}kb'] = (
            (max(start, end-width), end) if gene['strand'] == '+'
            else (start, min(end, start+width)))
    result['tss_4kb'] = (gene['tss']-2000, gene['tss']+2000)
    return result


def reduce_profile(profile, intervals):
    """Return two (batch, regions) arrays; invert each base BEFORE summing."""
    values = np.asarray(profile, dtype=np.float64)
    assert values.ndim == 2 and np.isfinite(values).all() and (values >= 0).all()
    native = np.full((len(values), len(REGIONS)), np.nan)
    reconstructed = native.copy()
    for i, name in enumerate(REGIONS):
        lo, hi = intervals[name]
        if hi > lo:
            segment = values[:, lo:hi]
            native[:, i] = segment.sum(axis=1)
            reconstructed[:, i] = (segment * (segment + 2)).sum(axis=1)
    return native, reconstructed


def load_pairs(checkpoint, legacy_parity=False):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if legacy_parity:
        assert checkpoint == LEGACY_CKPT.resolve(), 'Legacy parity check is specific to epoch 35'
    config_path = checkpoint.parent.parent / '.hydra/config.yaml'
    config = yaml.safe_load(config_path.read_text())
    d, head = config['dataset'], config['decoder']
    assert head['_name_'] == 'tss_profile' and head['profile_softplus']
    assert head['n_tracks'] == 1 and head['profile_region'] == 'all'
    assert config['task']['loss']['_name_'] == 'tss_profile_poisson_loss'
    assert d['expression_stranded'] and d['rc_strand']
    assert d['mlm'] == d['acc_mlm'] == 0
    assert d.get('pool', 1) == 1 and d.get('crop_output', 0) == 0
    assert Path(d['expression_data_path']).name == 'K562_rnaseq_stranded_fixed.npz'
    info_path = Path(d['tss_json_file'])
    info = json.loads(info_path.read_text())
    tss = pd.read_csv(BOUNDS, sep='\t')
    names = {n: g for n, g in tss.set_index('name').Ensembl_ID.to_dict().items() if g in info}
    pairs = pd.read_csv(CRISPR, sep='\t')
    pairs['ensg_id'] = pairs.measuredGeneSymbol.map(names)
    pairs = pairs.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)
    pairs['benchmark_index'] = np.arange(len(pairs))
    pairs['pair_id'] = (pairs.chrom + ':' + pairs.chromStart.astype(str) + '-'
                        + pairs.chromEnd.astype(str) + '_' + pairs.measuredGeneSymbol)
    assert pairs.pair_id.is_unique
    distances = pd.read_csv(DISTANCES, sep='\t')
    distances['pair_id'] = (distances.chr + ':' + distances.start.astype(str) + '-'
                           + distances.end.astype(str) + '_' + distances.TargetGene)
    assert distances.pair_id.is_unique
    pairs['distance_to_tss'] = pairs.pair_id.map(distances.set_index('pair_id').DistanceToTSS)
    geometry = []
    length = d['length']
    tss_pos = d.get('upstream')
    if tss_pos is None:
        tss_pos = length // 2
    for r in pairs.itertuples():
        g = info[r.ensg_id]
        flip = g['strand'] == '-'
        start = g['tss'] - (length-1-tss_pos if flip else tss_pos)
        lo, hi, full = map_interval(r.chromStart, r.chromEnd, start, length, flip)
        row = dict(window_start=start, flip=flip, gene_strand=g['strand'],
                   expression_split=g['split'], gene_length=g['gene_end']-g['gene_start'],
                   element_lo=lo, element_hi=hi,
                   element_in_context=full and r.chrom == g['chrom'])
        for name, (a, b) in gene_regions(g).items():
            lo, hi, full = map_interval(a, b, start, length, flip)
            row.update({f'{name}_lo': lo, f'{name}_hi': hi, f'{name}_full': full})
        geometry.append(row)
    pairs = pd.concat([pairs, pd.DataFrame(geometry)], axis=1)
    previous = None
    if legacy_parity:
        previous = np.load(LEGACY)
        previous_context = np.load(LEGACY.with_name(LEGACY.stem+'_in_context.npy'))
        assert previous.shape == (len(pairs), 2)
        np.testing.assert_array_equal(pairs.element_in_context.to_numpy(), previous_context)
    return pairs, config, previous, info_path


def selected_pairs(pairs, max_pairs=None):
    """Keep the full benchmark by default; deterministically sample for short checks."""
    active = pairs.element_in_context.to_numpy(bool, copy=True)
    if max_pairs is not None:
        assert max_pairs > 0
        ids = np.flatnonzero(active)
        if len(ids) > max_pairs:
            ids = np.random.default_rng(2222).choice(ids, size=max_pairs, replace=False)
            active[:] = False
            active[ids] = True
    return active


def save_state(outdir, native, reconstructed, counts, done):
    with (outdir / 'predictions.tmp.npz').open('wb') as handle:
        np.savez(handle, native=native, reconstructed=reconstructed, counts=counts, done=done)
    (outdir / 'predictions.tmp.npz').replace(outdir / 'predictions.npz')


def metrics(pairs, scores, subset):
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    from scipy.stats import spearmanr
    rows = []
    d = pairs[subset]
    y = d.Regulated.to_numpy(int)
    for name, delta in scores.items():
        s = np.asarray(delta)[subset]
        if not np.isfinite(s).all() or len(np.unique(y)) < 2:
            continue
        for direction, rank in [('decrease', -s), ('absolute', np.abs(s))]:
            if name == 'distance' and direction == 'absolute':
                continue
            precision, recall, _ = precision_recall_curve(y, rank)
            r = dict(readout=name, direction=direction, n_pairs=len(y), n_positive=int(y.sum()),
                     prevalence=float(y.mean()), auprc=float(auc(recall, precision)),
                     average_precision=float(average_precision_score(y, rank)))
            if name != 'distance':
                r['positive_fraction_decreased'] = float((s[y == 1] < 0).mean())
                r['effect_spearman'] = float(spearmanr(s, d.EffectSize.to_numpy()).statistic)
            rows.append(r)
    return rows


def summarize(outdir, pairs, native, reconstructed, counts, done, pseudocount, bootstrap, protocol):
    from sklearn.metrics import precision_recall_curve, auc
    scores = {'scalar_count': (counts[:, 1]-counts[:, 0])/np.log(2),
              'distance': pairs.distance_to_tss.to_numpy(float)}
    for space, values in [('native', native), ('reconstructed', reconstructed)]:
        for i, name in enumerate(REGIONS):
            scores[f'{name}_{space}'] = np.log2((values[:, 1, i]+pseudocount) /
                                                (values[:, 0, i]+pseudocount))
    table = pairs.copy()
    table['evaluated'] = done
    for allele, ai in [('before', 0), ('after', 1)]:
        table[f'count_{allele}'] = counts[:, ai]
        for space, values in [('native', native), ('reconstructed', reconstructed)]:
            for ri, name in enumerate(REGIONS):
                table[f'{name}_{space}_{allele}'] = values[:, ai, ri]
    for name, value in scores.items():
        if name != 'distance':
            table[f'delta_{name}'] = value
    table.to_csv(outdir/'scores.tsv', sep='\t', index=False)
    common = done & pairs.element_in_context.to_numpy() & pairs.body_full.to_numpy()
    distance = pairs.distance_to_tss.to_numpy()
    subsets = {'common_full_gene': common,
               '0-10kb': common & (distance <= 10000),
               '10-100kb': common & (distance > 10000) & (distance <= 100000),
               '>100kb': common & (distance > 100000)}
    for split in ('train', 'val', 'test'):
        subsets[f'expression_{split}'] = common & pairs.expression_split.eq(split).to_numpy()
    results = []
    for name, subset in subsets.items():
        results.extend(dict(subset=name, **row) for row in metrics(pairs, scores, subset))
    report = pd.DataFrame(results, columns=['subset', 'readout', 'direction', 'n_pairs',
        'n_positive', 'prevalence', 'auprc', 'average_precision',
        'positive_fraction_decreased', 'effect_spearman'])
    report.to_csv(outdir/'metrics.tsv', sep='\t', index=False)
    # Paired gene-cluster bootstrap on the predeclared primary support and score.
    ids = np.flatnonzero(common)
    gene_ids, group = np.unique(pairs.loc[common, 'ensg_id'], return_inverse=True)
    rng = np.random.default_rng(2222)
    y = pairs.loc[common, 'Regulated'].to_numpy(int)
    methods = ['body_native', 'body_reconstructed', 'scalar_count', 'distance']
    boots = []
    for _ in range(bootstrap if len(np.unique(y)) >= 2 else 0):
        weight = np.bincount(rng.integers(len(gene_ids), size=len(gene_ids)), minlength=len(gene_ids))[group]
        if np.unique(y[weight > 0]).size < 2:
            continue
        row = []
        for name in methods:
            precision, recall, _ = precision_recall_curve(y, -scores[name][ids], sample_weight=weight)
            row.append(float(auc(recall, precision)))
        boots.append(row)
    boots = np.asarray(boots)
    ci = {name: [float(x) for x in np.quantile(boots[:, i], [.025, .975])]
          for i, name in enumerate(methods)} if len(boots) else {}
    delta_ci = {f'body_native_minus_{name}': [float(x) for x in np.quantile(boots[:, 0]-boots[:, i], [.025, .975])]
                for i, name in enumerate(methods[1:], 1)} if len(boots) else {}
    summary = dict(evaluated_pairs=int(done.sum()), primary_pairs=int(common.sum()),
                   primary_positives=int(y.sum()), primary_genes=len(gene_ids),
                   excluded_truncated_gene_pairs=int((done & ~pairs.body_full.to_numpy()).sum()),
                   bootstrap_replicates=len(boots), auprc_95pct_ci=ci, paired_delta_95pct_ci=delta_ci)
    atomic_json(outdir/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    display = ['body_native', 'body_reconstructed', 'three_prime_10kb_native',
               'three_prime_50kb_native', 'scalar_count', 'distance']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if len(np.unique(y)) >= 2:
        for name in display:
            precision, recall, _ = precision_recall_curve(y, -scores[name][common])
            axes[0].plot(recall, precision, label=f'{name} ({auc(recall, precision):.3f})')
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(.5, .5, 'Both label classes are needed for a PR curve',
                     ha='center', va='center', transform=axes[0].transAxes)
    if len(y):
        axes[0].axhline(y.mean(), color='gray', ls=':')
    axes[0].set(xlabel='Recall', ylabel='Precision', title=f'Predicted decrease; {len(y)} pairs, {y.sum()} positives')
    bins = ['0-10kb', '10-100kb', '>100kb']
    for i, name in enumerate(display):
        sub = report[report.readout.eq(name) & report.direction.eq('decrease')].set_index('subset')
        axes[1].bar(np.arange(3)+(i-2.5)*.13, sub.auprc.reindex(bins).to_numpy(dtype=float), .13, label=name)
    for i, name in enumerate(bins):
        axes[1].hlines(pairs.loc[subsets[name], 'Regulated'].mean(), i-.45, i+.45, color='gray', ls=':')
    axes[1].set(xticks=np.arange(3), xticklabels=bins, ylabel='AUPRC', title='Distance strata, identical pairs')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir/f'precision_recall.{ext}', dpi=180)
    plt.close(fig)
    headline = report[report.subset.eq('common_full_gene') & report.direction.eq('decrease')]
    lines = ['# Poisson profile enhancer–gene evaluation', '',
        f"Checkpoint: `{protocol['checkpoint']}`. Accessibility divided by {protocol['scale_factor']:g} over each element plus {protocol['flank']} bp per side.", '',
        f"Pair limit: {protocol['max_pairs'] if protocol['max_pairs'] is not None else 'none (full benchmark)'}. Unselected or out-of-context pairs have evaluated=false and missing scores.", '',
        f'Primary score: `log2((sum(after profile) + {pseudocount:g})/(sum(before profile) + {pseudocount:g}))` over the full strand-matched gene body. Rank enhancer links by its negative.', '',
        'The native profile is on the sqrt(1+coverage)-1 training scale. Reconstructed coverage uses the per-base inverse p*(p+2) BEFORE summing; it is an approximate coverage estimate. Scalar count differences are divided by ln(2), yielding changes in log2(1+CPM).', '',
        f'Common support: {len(y)} pairs, {y.sum()} positives, {len(gene_ids)} genes. Incomplete gene bodies are excluded from every primary comparison. Split labels reproduce the expression dataset labels.', '',
        '| Readout | Signed AUPRC | Average precision |', '|---|---:|---:|']
    lines.extend(f'| {r.readout} | {r.auprc:.4f} | {r.average_precision:.4f} |' for r in headline.itertuples())
    lines += ['', 'See `metrics.tsv` for absolute scores, distance strata and expression splits; `summary.json` for paired gene-bootstrap uncertainty; `scores.tsv` for every pair, sum, delta and context flag.', '',
              'The primary readout is the full gene body. The terminal controls were saved from the same forward passes.']
    (outdir/'README.md').write_text('\n'.join(lines)+'\n')
    print(headline[['readout', 'auprc', 'average_precision']].to_string(index=False), flush=True)
    return summary


def run(args):
    args.ckpt_path = args.ckpt_path.resolve(strict=True)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/'.evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pairs, config, previous, info_path = load_pairs(args.ckpt_path, args.legacy_parity)
        sources = [args.ckpt_path, args.ckpt_path.parent.parent/'.hydra/config.yaml',
                   info_path, CRISPR, BOUNDS, DISTANCES, Path(__file__),
                   REPO/'evals/e2g_tss_profile.py', REPO/'src/tasks/decoders.py',
                   REPO/'src/dataloaders/datasets/tss_dataset.py']
        if previous is not None:
            sources.extend([LEGACY, LEGACY.with_name(LEGACY.stem+'_in_context.npy')])
        protocol = dict(checkpoint=str(args.ckpt_path), scale_factor=args.scale_factor,
            flank=args.flank, pseudocount=args.pseudocount, regions=list(REGIONS),
            primary='body_native', batch_size=args.batch_size, inference_precision='float32',
            legacy_parity=args.legacy_parity, max_pairs=args.max_pairs,
            pair_sampling_seed=2222, bootstrap=args.bootstrap,
            target_transform='sqrt(1+coverage)-1',
            source_sha256={str(p): sha256(p) for p in sources})
        path = outdir/'protocol.json'
        if path.exists():
            assert json.loads(path.read_text()) == protocol, 'Protocol/source changed; use a new output directory'
        else:
            atomic_json(path, protocol)
            pairs.to_csv(outdir/'pairs.tsv', sep='\t', index=False)
        active = selected_pairs(pairs, args.max_pairs)
        print(f'{len(pairs)} pairs, {pairs.element_in_context.sum()} in context, '
              f'{active.sum()} selected, {pairs.loc[active].ensg_id.nunique()} genes; '
              f'{(active & pairs.body_full.to_numpy()).sum()} pairs with full gene body.', flush=True)
        if args.prepare_only:
            return
        state_path = outdir/'predictions.npz'
        if state_path.exists():
            with np.load(state_path) as state:
                native, reconstructed, counts, done = [state[k] for k in ('native', 'reconstructed', 'counts', 'done')]
        else:
            native = np.full((len(pairs), 2, len(REGIONS)), np.nan)
            reconstructed = native.copy()
            counts = np.full((len(pairs), 2), np.nan)
            done = np.zeros(len(pairs), dtype=bool)
        status = dict(status='starting', slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                      required_pairs=int(active.sum()), completed_pairs=int(done.sum()))
        atomic_json(outdir/'status.json', status)
        started = time.time()
        try:
            if not np.all(done[active]):
                import torch
                assert torch.cuda.is_available(), 'Run on a GPU allocation'
                torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 4))))
                sys.path.insert(0, str(REPO))
                sys.path.insert(0, str(REPO/'evals'))
                from e2g_tss_profile import Evals
                model = Evals(str(args.ckpt_path), load_data=True, expression_data_path=None)
                assert model.dataset.pool == 1 and model.dataset.crop_output == 0
                gene_index = {g: i for i, g in enumerate(model.dataset.genes)}
                status['status'] = 'running'
                atomic_json(outdir/'status.json', status)
                n_genes = 0
                for gene, group in pairs.loc[active & ~done].groupby('ensg_id', sort=False):
                    ((seq, acc), (_, _, _, tss_mask, gene_mask, _, _)) = model.dataset[gene_index[gene]]
                    first = group.iloc[0]
                    intervals = {name: (int(first[f'{name}_lo']), int(first[f'{name}_hi'])) for name in REGIONS}
                    length = model.dataset.length
                    # Reference shares a batch with the first perturbation. Cache it for this gene.
                    row_ids = [None] + list(group.index)
                    reference = None
                    for start in range(0, len(row_ids), args.batch_size):
                        ids = row_ids[start:start+args.batch_size]
                        keep = len(ids)
                        padded = ids + [ids[-1]]*(args.batch_size-keep)
                        seq_batch = seq.unsqueeze(0).repeat(args.batch_size, 1, 1)
                        acc_batch = acc.unsqueeze(0).repeat(args.batch_size, 1, 1)
                        for j, row_id in enumerate(padded):
                            if row_id is not None:
                                r = pairs.loc[row_id]
                                lo, hi = max(0, int(r.element_lo)-args.flank), min(length, int(r.element_hi)+args.flank)
                                acc_batch[j, 0, lo:hi] /= args.scale_factor
                        profile, scalar = model(data=((seq_batch, acc_batch), (None, None, None,
                            tss_mask.unsqueeze(0).repeat(args.batch_size, 1),
                            gene_mask.unsqueeze(0).repeat(args.batch_size, 1), None, None)), return_profile=True)
                        profile = profile.float().cpu().numpy()[:keep]
                        scalar = scalar.float().cpu().numpy().reshape(-1)[:keep]
                        assert profile.shape == (keep, 1, length) and np.isfinite(scalar).all()
                        n, r = reduce_profile(profile[:, 0], intervals)
                        for j, row_id in enumerate(ids):
                            if row_id is None:
                                if previous is not None:
                                    np.testing.assert_allclose(scalar[j], previous[group.index, 0], rtol=2e-5, atol=2e-5,
                                                               err_msg=f'{gene}: reference scalar differs from legacy')
                                reference = (n[j], r[j], scalar[j])
                            else:
                                if previous is not None:
                                    np.testing.assert_allclose(scalar[j], previous[row_id, 1], rtol=2e-5, atol=2e-5,
                                                               err_msg=f'pair {row_id}: perturbed scalar differs from legacy')
                                native[row_id] = [reference[0], n[j]]
                                reconstructed[row_id] = [reference[1], r[j]]
                                counts[row_id] = [reference[2], scalar[j]]
                                done[row_id] = True
                        del profile, scalar, seq_batch, acc_batch
                    n_genes += 1
                    if n_genes <= 2 or n_genes % 10 == 0:
                        save_state(outdir, native, reconstructed, counts, done)
                        status.update(completed_pairs=int(done.sum()), elapsed_seconds=round(time.time()-started, 1))
                        if previous is not None:
                            status['max_scalar_parity_error'] = float(np.max(np.abs(counts[done]-previous[done])))
                        atomic_json(outdir/'status.json', status)
                        print(json.dumps(status), flush=True)
            save_state(outdir, native, reconstructed, counts, done)
            summary = summarize(outdir, pairs, native, reconstructed, counts, done,
                                args.pseudocount, args.bootstrap, protocol)
            status.update(status='completed', completed_pairs=int(done.sum()), summary=summary,
                          elapsed_seconds=round(time.time()-started, 1))
            atomic_json(outdir/'status.json', status)
        except BaseException as error:
            save_state(outdir, native, reconstructed, counts, done)
            status.update(status='failed', completed_pairs=int(done.sum()), error=str(error))
            atomic_json(outdir/'status.json', status)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt-path', type=Path, required=True,
                        help='TSS Poisson profile/count checkpoint with its saved .hydra/config.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scale-factor', type=float, default=100)
    parser.add_argument('--flank', type=int, default=100)
    parser.add_argument('--pseudocount', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--legacy-parity', action='store_true',
                        help='Compare the original epoch-35 checkpoint against its cached scalar predictions')
    parser.add_argument('--max-pairs', type=int,
                        help='Evaluate a deterministic sample of at most this many in-context pairs (smoke tests)')
    args = parser.parse_args()
    assert args.scale_factor >= 1 and args.flank >= 0 and args.pseudocount > 0
    assert args.batch_size >= 2 and args.bootstrap > 0
    assert args.max_pairs is None or args.max_pairs > 0
    run(args)
