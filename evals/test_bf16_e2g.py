#!/usr/bin/env python3
"""
Does running E2G under bf16 autocast change the scored quantity?

Background: the e2g_* scripts that inline their own Evals class run the forward in fp32, while
evals_utils_joint.Evals (which only e2g_striped* imports) wraps it in autocast(bfloat16). Training
is precision=bf16, so bf16 eval actually reproduces the training forward and fp32 is the outlier --
but E2G scores a DIFFERENCE between two nearly identical predictions, which is the case most
sensitive to a reduced mantissa. This script measures that before anything is switched over.

Method: for the first --n_elements CRISPR pairs, build the exact (before, after) batch that
e2g_tss_profile_dn.py builds, then run it twice through the SAME Evals object -- once as-is
(fp32) and once inside torch.autocast(bfloat16). autocast is a context manager, so this needs no
change to e2g_tss_profile.py; the inner ops pick it up from the enclosing scope.

What matters is not whether the raw predictions agree but whether delta = after - before agrees,
since delta is what downstream AUC is computed on. The headline number is
max|delta_fp32 - delta_bf16| expressed as a fraction of std(delta_fp32): the rounding noise
measured against the spread of the actual signal.

Reads only. Writes nothing except an optional --save_npz.
"""
print('bf16 vs fp32 E2G comparison', flush=True)

import numpy as np
import pandas as pd
import json
import sys
import os
sys.path.append('/data1/lesliec/sarthak/caduceus/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr, spearmanr

#same import route e2g_tss_profile_dn.py uses, so the checkpoint loading is identical
from e2g_tss_profile import Evals


def run_batch(evals, sb, ab, tb, gb, use_bf16):
    """One forward pass, optionally under bf16 autocast. Returns a flat float64 numpy array."""
    if use_bf16:
        with torch.autocast(device_type=evals.device.type, dtype=torch.bfloat16):
            out = evals(data=((sb, ab), (None, None, None, tb, gb, None, None)))
    else:
        out = evals(data=((sb, ab), (None, None, None, tb, gb, None, None)))
    return out.float().cpu().numpy().reshape(-1).astype(np.float64)


def main(args):
    TSS_bounds_file = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
    tss = pd.read_csv(TSS_bounds_file, sep="\t")

    path = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    gs_df = pd.read_csv(path, sep="\t")

    #expression_data_path=None for the same reason e2g_tss_profile_dn.py does it: only the count
    #head is scored, and the RNA npz is large
    evals_kwargs = dict(load_data=args.load_data, expression_data_path=None)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path
    evals = Evals(args.ckpt_path, **evals_kwargs)

    tss_dict = evals.dataset.tss_dict
    gene_to_idx = {gene: i for i, gene in enumerate(evals.dataset.genes)}

    name_to_ensg = tss.set_index('name')['Ensembl_ID'].to_dict()
    name_to_ensg = {name: ensg for name, ensg in name_to_ensg.items() if ensg in tss_dict}
    gs_df['ensg_id'] = gs_df['measuredGeneSymbol'].map(name_to_ensg)
    gs_df = gs_df.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)

    SEQ_LEN = evals.dataset.length
    TSS_FINAL_POS = evals.dataset.upstream if evals.dataset.upstream is not None else SEQ_LEN // 2
    print(f'window: length={SEQ_LEN}, TSS at index {TSS_FINAL_POS}, rc_strand={evals.dataset.rc_strand}')
    print(f'{len(gs_df)} CRISPR pairs available; testing the first {args.n_elements} in-context ones')

    #only in-context elements are informative here: out-of-context rows are unperturbed, so their
    #delta is exactly 0 in both precisions and would flatter the comparison
    rows32, rows16, kept = [], [], []

    #--rows_npy restricts to specific gs_df row indices. Without it the first N in-context rows are
    #used, and those are overwhelmingly non-regulatory: their true effect is ~0, so their sign is
    #noise by construction and any sign-agreement number computed over them is meaningless. Pass the
    #rows this model actually responded to (e.g. top true positives by |delta|) to measure the
    #precision question on the detections E2G is actually scored on.
    want = None
    if args.rows_npy is not None:
        want = [int(r) for r in np.load(args.rows_npy)]
        print(f'restricting to {len(want)} caller-supplied rows from {args.rows_npy}')

    iterator = gs_df.loc[want].iterrows() if want is not None else gs_df.iterrows()
    total = len(want) if want is not None else args.n_elements

    pbar = tqdm(total=total)
    for i, row in iterator:
        if want is None and len(kept) >= args.n_elements:
            break

        chrom, start, end = row['chrom'], row['chromStart'], row['chromEnd']
        ensgid = row['ensg_id']
        gene_info = tss_dict[ensgid]
        temp_tss = gene_info['tss']

        flip = evals.dataset.rc_strand and gene_info['strand'] == '-'
        tss_pre_pos = TSS_FINAL_POS if not flip else (SEQ_LEN - 1 - TSS_FINAL_POS)
        window_start = temp_tss - tss_pre_pos

        lo = start - window_start
        hi = end - window_start
        if not ((lo >= 0) and (hi <= SEQ_LEN)):
            continue
        if flip:
            lo, hi = SEQ_LEN - hi, SEQ_LEN - lo

        idx = gene_to_idx[ensgid]
        ((s, a), (su, au, counts, tss_mask, gene_mask, strand, expression)) = evals.dataset[idx]

        #n_alt = 1: row 0 is the unperturbed reference, row 1 is the accessibility-knockdown
        s = s.unsqueeze(0).repeat(2, 1, 1)
        a = a.unsqueeze(0).repeat(2, 1, 1)
        tss_mask = tss_mask.unsqueeze(0).repeat(2, 1) if tss_mask is not None else None
        gene_mask = gene_mask.unsqueeze(0).repeat(2, 1) if gene_mask is not None else None

        startmask = max(0, lo - args.dist_additional_mask)
        endmask = min(SEQ_LEN, hi + args.dist_additional_mask)
        a[1:, 0, startmask:endmask] = a[1:, 0, startmask:endmask] / args.scale_factor

        #identical inputs down both paths, so any difference is precision alone -- UNLESS the model
        #itself is nondeterministic, which --control checks by running fp32 twice instead. Without
        #that control a nonzero result is ambiguous: it could be precision or just kernel jitter.
        p32 = run_batch(evals, s, a, tss_mask, gene_mask, use_bf16=False)
        p16 = run_batch(evals, s, a, tss_mask, gene_mask, use_bf16=not args.control)

        rows32.append(p32)
        rows16.append(p16)
        kept.append(i)
        pbar.update(1)
    pbar.close()

    if len(kept) < 2:
        print(f'Only {len(kept)} in-context elements found -- nothing to compare.')
        return

    p32 = np.stack(rows32)   # (n, 2) -> col 0 before, col 1 after
    p16 = np.stack(rows16)
    d32 = p32[:, 1] - p32[:, 0]   # the scored quantity, a log-space difference
    d16 = p16[:, 1] - p16[:, 0]

    absdiff = np.abs(d32 - d16)
    spread = d32.std()

    print()
    print('=' * 68)
    print('MODE: ' + ('CONTROL (fp32 vs fp32 -- measures nondeterminism)' if args.control
                      else 'fp32 vs bf16'))
    print(f'n elements compared              : {len(kept)}')
    print()
    print('-- raw predictions (before/after, log scale) --')
    print(f'  max |fp32 - bf16|              : {np.abs(p32 - p16).max():.6g}')
    print(f'  mean |fp32 - bf16|             : {np.abs(p32 - p16).mean():.6g}')
    print()
    print('-- delta = after - before  (what downstream AUC actually scores) --')
    print(f'  std(delta_fp32)                : {spread:.6g}   <- the signal')
    print(f'  max |delta_fp32 - delta_bf16|  : {absdiff.max():.6g}   <- the noise')
    print(f'  mean|delta_fp32 - delta_bf16|  : {absdiff.mean():.6g}')
    if spread > 0:
        print(f'  max noise / signal spread      : {absdiff.max() / spread:.4%}')
    print(f'  pearson  r(delta32, delta16)   : {pearsonr(d32, d16)[0]:.6f}')
    print(f'  spearman r(delta32, delta16)   : {spearmanr(d32, d16).correlation:.6f}')
    agree = (np.sign(d32) == np.sign(d16)).mean()
    print(f'  sign agreement                 : {agree:.2%}')
    print('=' * 68)
    print()
    print('Interpretation: "max noise / signal spread" is the number that decides this. Well under')
    print('1% with spearman ~1.0 and full sign agreement means bf16 is safe for E2G and the eight')
    print('inlined Evals copies can be switched over. A few percent, or any sign flips, means leave')
    print('them fp32 and instead make e2g_striped* match, so the two families stay comparable.')

    if args.save_npz:
        np.savez(args.save_npz, p32=p32, p16=p16, d32=d32, d16=d16, rows=np.array(kept))
        print(f'\nsaved raw values to {args.save_npz}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare fp32 vs bf16 autocast on the E2G scored delta')
    parser.add_argument('--ckpt_path', type=str,
                        default='/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt',
                        help='Checkpoint to test (default: the e2g_tss_profile_dn.py default)')
    parser.add_argument('--n_elements', type=int, default=50,
                        help='Number of in-context CRISPR pairs to compare')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by (matches e2g_tss_profile_dn.py)')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp masked each side of the element')
    parser.add_argument('--load_data', action='store_true',
                        help='Load accessibility into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override the accessibility path from the checkpoint config')
    parser.add_argument('--rows_npy', type=str, default=None,
                        help='npy of gs_df row indices to test. Use this to target elements the model '
                             'actually responded to; the default first-N-in-context sample is mostly '
                             'non-regulatory, where the true effect is ~0 and sign agreement is noise.')
    parser.add_argument('--control', action='store_true',
                        help='CONTROL: run fp32 twice instead of fp32-vs-bf16. Any nonzero result here '
                             'is model nondeterminism, not precision, and sets the noise floor that the '
                             'real comparison has to beat to mean anything.')
    parser.add_argument('--save_npz', type=str, default=None,
                        help='Optional path to save the raw fp32/bf16 values for inspection')
    args = parser.parse_args()
    print(args)
    main(args)
