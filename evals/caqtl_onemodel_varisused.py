"""Score the European ChromBPNet var.isused SNPs using the dsQTL REF/ALT method.

Coordinates are hg38, 1-based in the source table. Output channels are REF, ALT;
rows retain the source table's var.isused order. The default readout is 500 bp
inside a 1,000 bp accessibility mask, with observed accessibility in the flanks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


OUTPUT_DIR = Path('/data1/lesliec/sarthak/data/joint_playground/caQTL')
VARIANTS = OUTPUT_DIR / 'caqtls.eu.lcls.benchmarking.all.tsv'
DATA_PATH = '/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz'


def load_variants(path):
    """Select exactly var.isused; validate rather than silently drop any rows."""
    qtls = pd.read_csv(path, sep='\t')
    required = {'var.chr', 'var.pos_hg38', 'var.allele1', 'var.allele2',
                'var.isused', 'obs.pval'}
    missing = required - set(qtls.columns)
    if missing:
        raise ValueError(f'Missing benchmark columns: {sorted(missing)}')
    used = qtls['var.isused'].astype(str).str.lower()
    if not used.isin(['true', 'false']).all():
        raise ValueError('var.isused must contain only True/False values')
    qtls = qtls.loc[used.eq('true')].copy()
    qtls.insert(0, 'source_row', qtls.index.to_numpy())
    qtls.reset_index(drop=True, inplace=True)
    qtls.insert(0, 'benchmark_row', np.arange(len(qtls)))
    if qtls.empty:
        raise ValueError('The benchmark has no var.isused variants')
    for column in ['var.allele1', 'var.allele2']:
        if not qtls[column].isin(list('ACGT')).all():
            raise ValueError(f'{column} must contain single A/C/G/T alleles')
    if qtls['var.allele1'].eq(qtls['var.allele2']).any():
        raise ValueError('Reference and alternate alleles must differ')
    pos = pd.to_numeric(qtls['var.pos_hg38'], errors='raise')
    if not (np.isfinite(pos) & (pos >= 1) & (pos == np.floor(pos))).all():
        raise ValueError('var.pos_hg38 must contain positive 1-based integer positions')
    qtls['var.pos_hg38'] = pos.astype(np.int64)
    key = ['var.chr', 'var.pos_hg38', 'var.allele1', 'var.allele2']
    if qtls.duplicated(key).any():
        raise ValueError('Duplicate variants in the var.isused benchmark')
    pval = pd.to_numeric(qtls['obs.pval'], errors='raise')
    if not (pval.between(0, 1) & ((pval < 1e-6) | (pval > 1e-3))).all():
        raise ValueError('Unexpected p-value in var.isused: expected <1e-6 or >1e-3')
    qtls['label'] = (pval < 1e-6).astype(np.int8)
    return qtls


def score_variant(evals, row, mask_size, out_size, pool, ctt_val):
    """Return (output bins, 2) predictions with identical REF/ALT context."""
    length = evals.dataset.length
    half = length // 2
    pos0 = int(row['var.pos_hg38']) - 1
    chrom = row['var.chr']
    ref, alt = row['var.allele1'], row['var.allele2']
    start = pos0 - half
    idx = evals.dataset.expand_seqs(chrom, start, start + length)
    _, originals = evals.dataset[idx]
    seq, acc = originals[:2]
    if seq.ndim != 2 or acc.ndim != 2 or seq.shape[0] != length or acc.shape[0] != length:
        raise ValueError('Expected unpooled sequence/accessibility inputs at the checkpoint context length')
    expected = np.zeros(5, dtype=np.float32)  # A, C, G, T, N
    expected['ACGT'.index(ref)] = 1
    observed = seq[half, :5].float().cpu().numpy()
    if not np.array_equal(observed, expected):
        raise ValueError(f'hg38 REF mismatch at {chrom}:{pos0 + 1}: '
                         f'expected allele1={ref}, observed one-hot={observed.tolist()}')

    alt_seq = seq.clone()
    alt_seq[half, :5] = 0
    alt_seq[half, 'ACGT'.index(alt)] = 1
    mask_start = half - mask_size // 2
    out_start = half - out_size // 2
    profiles = []
    for allele_seq in (seq, alt_seq):
        out = evals.mask(mask_start, mask_start + mask_size,
                         data=(None, None, allele_seq, acc),
                         mask_accessibility=True, ctt_val=ctt_val)
        if out[1].ndim != 3 or out[1].shape[1] != length:
            raise ValueError('Expected base-resolution accessibility predictions over the input context')
        pred = out[1][0, out_start:out_start + out_size, 0].float().cpu().numpy()
        if pool > 1:
            pred = pred.reshape(-1, pool).mean(axis=1)
        if not np.isfinite(pred).all() or (pred < 0).any() or not (pred > 0).any():
            raise ValueError(f'Invalid accessibility predictions at {chrom}:{pos0 + 1}')
        profiles.append(pred)
        del out
    return np.stack(profiles, axis=-1)


def main(args):
    if not 0 < args.out_size <= args.mask_size:
        raise ValueError('Require 0 < out_size <= mask_size so the readout is fully masked')
    if args.pool < 1 or args.out_size % args.pool:
        raise ValueError('pool must be positive and divide out_size')
    qtls = load_variants(args.variants)
    total = len(qtls)
    end = total if args.end is None else args.end
    if not 0 <= args.start < end <= total:
        raise ValueError(f'Require 0 <= start < end <= {total}')
    qtls = qtls.iloc[args.start:end].copy()
    qtls.insert(0, 'prediction_row', np.arange(len(qtls)))
    output = Path(args.output)
    if not output.is_absolute():
        output = OUTPUT_DIR / output
    if output.suffix != '.npy':
        raise ValueError('output must end in .npy')
    scores_path = output.with_suffix('.scores.tsv.gz')
    run_path = output.with_suffix('.run.json')
    partial_path = output.with_suffix('.partial.npy')
    shape = (len(qtls), args.out_size // args.pool, 2)
    print(f'European hg38 var.isused: {total:,} total; selected [{args.start}:{end}]\n'
          f'Positives: {int(qtls.label.sum()):,}; controls: {int((qtls.label == 0).sum()):,}\n'
          f'Mask: {args.mask_size} bp; readout: {args.out_size} bp; pool: {args.pool}\n'
          f'Predictions: {output}, shape={shape}, channels=[REF, ALT]\n'
          f'Row identities and logFC scores: {scores_path}', flush=True)
    if args.dry_run:
        return
    if not args.ckpt_path:
        raise ValueError('--ckpt_path is required for inference')
    for path in (output, scores_path, run_path, partial_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; choose a new --output')

    # Keep --help/--dry_run independent of model imports and GPU initialization.
    from evals_utils_joint import Evals

    evals = Evals(args.ckpt_path, load_data=args.load_data, data_idxs=args.data_idxs,
                  data_path=args.data_path, encoder_numcelltypes=args.encoder_numcelltypes)
    evals.dataset.return_celltype_idx_og = False
    if evals.dataset.length is None or args.mask_size > evals.dataset.length:
        raise ValueError('mask_size must fit within the checkpoint context length')

    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted job leaves a clearly named partial file, with NaNs for
    # unscored rows. Only a complete run publishes the final .npy filename.
    predictions = np.lib.format.open_memmap(partial_path, mode='w+', dtype=np.float32, shape=shape)
    predictions[:] = np.nan
    for i, (_, row) in enumerate(tqdm(qtls.iterrows(), total=len(qtls), disable=args.quiet)):
        predictions[i] = score_variant(evals, row, args.mask_size, args.out_size,
                                        args.pool, args.ctt_val)
        if (i + 1) % 100 == 0:
            predictions.flush()
    predictions.flush()
    means = predictions.mean(axis=1, dtype=np.float64)
    qtls['pred.ref_mean'] = means[:, 0]
    qtls['pred.alt_mean'] = means[:, 1]
    qtls['pred.log2_alt_over_ref'] = np.log2(means[:, 1] / means[:, 0])
    qtls['pred.abs_log2fc'] = np.abs(qtls['pred.log2_alt_over_ref'])
    qtls.to_csv(scores_path, sep='\t', index=False, compression='gzip')
    metadata = {
        'arguments': vars(args),
        'variants_sha256': hashlib.sha256(Path(args.variants).read_bytes()).hexdigest(),
        'genome_build': 'hg38', 'source_position_base': 1,
        'context_length': evals.dataset.length,
        'selection': 'var.isused == True; no additional variant filtering',
        'benchmark_rows': [args.start, end], 'shape': list(shape),
        'dtype': 'float32', 'channels': ['REF (var.allele1)', 'ALT (var.allele2)'],
        'score': 'log2(mean(ALT)/mean(REF)); no pseudocount; absolute value for classification',
        'signed_score_orientation': 'ALT minus REF; not harmonized to obs.beta',
        'accessibility_context': 'same observed flanks for both alleles; central window masked',
        'predictions': str(output), 'scores': str(scores_path), 'status': 'complete',
    }
    del predictions
    partial_path.replace(output)
    run_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {len(qtls):,} paired variant predictions and scores.', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt_path', help='Checkpoint to score; required unless --dry_run')
    parser.add_argument('-o', '--output', default='caqtl_varisused.npy',
                        help='Absolute .npy path, or filename relative to the caQTL data directory')
    parser.add_argument('--variants', default=str(VARIANTS))
    parser.add_argument('--mask_size', type=int, default=1000)
    parser.add_argument('--out_size', type=int, default=500)
    parser.add_argument('--pool', type=int, default=1)
    parser.add_argument('--data_path', default=DATA_PATH)
    parser.add_argument('--data_idxs', nargs='+', type=int, default=None)
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--ctt_val', type=int, default=6)
    parser.add_argument('--encoder_numcelltypes', type=int, default=None)
    parser.add_argument('--start', type=int, default=0, help='First row of the var.isused subset (0-based)')
    parser.add_argument('--end', type=int, default=None, help='Exclusive end row; default is all variants')
    parser.add_argument('--dry_run', action='store_true', help='Validate selection/options without loading models or writing files')
    parser.add_argument('--quiet', action='store_true', help='Disable the per-variant progress bar')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
