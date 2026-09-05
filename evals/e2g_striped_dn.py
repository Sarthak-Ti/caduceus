#!/usr/bin/env python3
"""
E2G (enhancer-to-gene) evaluation for the striped model with cell type token.

Variant of e2g_striped.py that can also dinucleotide shuffle the element instead of, or as well as,
scaling down its accessibility. See --perturbation.

For each enhancer-gene pair, runs the model on a before (unperturbed) sequence and one or more
after (perturbed) sequences and saves a (n_elements, n_rows, out_bins) array for downstream
AUC evaluation.

Output layout (rows axis): n_blocks blocks of 2 strand rows, block 0 is the unperturbed 'before',
blocks 1.. are the perturbed 'after' (a single mean block unless --save_all_shuffles).
  default      → 4 rows per element: [before+, before-, after+, after-]
  --save_all_shuffles → 2*(1+n_shuffles) rows: [before+, before-, shuf0+, shuf0-, shuf1+, ...]
"""
print('E2G striped evaluation (dinucleotide shuffle)', flush=True)
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')

import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
import argparse
import os

from evals.evals_utils_joint import Evals
from evals.utils.dinuc_shuffle import dinucleotide_shuffle

E2G_DEFAULT_DIR = '/data1/lesliec/sarthak/data/joint_playground/e2g/'

#the striped accessibility head is stranded: channel 0 = plus, channel 1 = minus
N_STRANDS = 2


def main(args):
    TSS_bounds_file = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
    tss = pd.read_csv(TSS_bounds_file, sep="\t")

    path = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    gs_df = pd.read_csv(path, sep="\t")

    json_path = '/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json'
    with open(json_path, 'r') as f:
        tss_dict = json.load(f)

    name_to_ensg = tss.set_index('name')['Ensembl_ID'].to_dict()
    name_to_ensg = {name: ensg for name, ensg in name_to_ensg.items() if ensg in tss_dict}
    gs_df['ensg_id'] = gs_df['measuredGeneSymbol'].map(name_to_ensg)
    gs_df = gs_df.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)

    #row 0 of each batch is the unperturbed reference, rows 1..n_alt are the perturbed versions
    n_alt = args.n_shuffles if args.perturbation in ('shuffle', 'both_perturbations') else 1

    #by default collapse the shuffles to their mean, so the output keeps the original row layout.
    #--save_all_shuffles keeps every individual shuffle instead
    average_shuffles = args.perturbation in ('shuffle', 'both_perturbations') and not args.save_all_shuffles
    n_blocks = 2 if average_shuffles else 1 + n_alt
    n_rows = n_blocks * N_STRANDS

    evals = Evals(
        args.ckpt_path,
        load_data=args.load_data,
        data_path=args.data_path,
        data_idxs=None,  # use all
        encoder_numcelltypes=args.encoder_numcelltypes,
    )
    evals.skip_softplus = not args.softplus
    evals.dataset.return_celltype_idx_og = False  # use args.ctt_val, not dataset token

    length = evals.dataset.length
    half = length // 2

    in_context = np.ones(len(gs_df), dtype=bool)
    shuffled = np.zeros(len(gs_df), dtype=bool)  # whether the element actually got dinucleotide shuffled
    #lazy alloc after first forward pass to infer out_bins
    outputs = None
    #spread across the shuffles, otherwise averaging throws that information away
    shuffle_std = None

    for i, row in tqdm(gs_df.iterrows(), total=len(gs_df)):
        chrom  = row['chrom']
        start  = row['chromStart']
        end    = row['chromEnd']
        ensgid = row['ensg_id']
        temp_tss = tss_dict[ensgid]['tss']

        if args.center_tss:
            in_context[i] = (start >= temp_tss - half) and (end <= temp_tss + half)
            center = temp_tss
        else:
            center = (start + end) // 2

        idx = evals.dataset.expand_seqs(chrom, center, center)
        outputs1, outputs2 = evals.dataset[idx]
        s  = outputs1[0]
        a  = outputs1[1]
        su = outputs2[0]
        au = outputs2[1]

        s = s.unsqueeze(0).repeat(1 + n_alt, 1, 1)  # (1+n_alt, C, L): row 0=before, rows 1..=after
        a = a.unsqueeze(0).repeat(1 + n_alt, 1, 1)

        if in_context[i]:
            # Locate the element within the window first, then derive each perturbation region from it
            if args.center_tss:
                seq_start = center - half
                e_start = start - seq_start
                e_end   = end   - seq_start
            else:
                elen    = end - start
                e_start = half - elen // 2
                e_end   = e_start + elen

            # divide the primary accessibility track for the after sequences
            if args.perturbation in ('acc', 'both_perturbations'):
                startmask = max(0, e_start - args.dist_additional_mask)
                endmask   = min(length, e_end + args.dist_additional_mask)
                a[1:, 0, startmask:endmask] = a[1:, 0, startmask:endmask] / args.scale_factor

            # Dinucleotide shuffle the element itself, leaving the accessibility track intact
            if args.perturbation in ('shuffle', 'both_perturbations'):
                sh_start = max(0, e_start - args.dist_additional_shuffle)
                sh_end   = min(length, e_end + args.dist_additional_shuffle)
                if sh_end - sh_start >= args.min_shuffle_len:
                    try:
                        # shuffle from row 0, which is never perturbed. seed is keyed to the row index so
                        # the shuffles are reproducible regardless of batching or where you restart
                        shuf = dinucleotide_shuffle(s[0, :, sh_start:sh_end], n_shuffles=n_alt,
                                                    random_state=args.seed + i)
                        for k in range(n_alt):
                            s[1 + k, :, sh_start:sh_end] = shuf[k]
                        shuffled[i] = True
                    except ValueError:
                        # every shuffle came back identical (low complexity element), leave it unshuffled
                        pass

        # chunk the forward pass, a full (1+n_alt) batch will not fit for large --n_shuffles
        outs = []
        for j in range(0, 1 + n_alt, args.batch_size):
            sb = s[j:j + args.batch_size]
            ab = a[j:j + args.batch_size]
            out = evals(data=((sb, ab), (su, au)), ctt_val=args.ctt_val)
            # out[1] is acc: (batch, out_bins, n_channels)
            pred = out[1].float().cpu().numpy()

            if args.pool > 1:
                b, _, c = pred.shape
                n_pooled = pred.shape[1] // args.pool
                pred = pred[:, :n_pooled * args.pool, :].reshape(b, n_pooled, args.pool, c).mean(axis=2)

            outs.append(pred)
        preds = np.concatenate(outs, axis=0)  # (1+n_alt, out_bins, n_channels)
        assert preds.shape[-1] >= N_STRANDS, f'model gave {preds.shape[-1]} strand channels, need {N_STRANDS}'

        # (1+n_alt, n_strands, out_bins), so block b strand c lands on row b*N_STRANDS + c
        blocks = preds[:, :, :N_STRANDS].transpose(0, 2, 1)
        out_bins = blocks.shape[-1]

        if outputs is None:
            outputs = np.zeros((len(gs_df), n_rows, out_bins), dtype=np.float32)
            if average_shuffles:
                shuffle_std = np.zeros((len(gs_df), N_STRANDS, out_bins), dtype=np.float32)

        if average_shuffles:
            outputs[i] = np.concatenate([blocks[0], blocks[1:].mean(axis=0)], axis=0)
            shuffle_std[i] = blocks[1:].std(axis=0)
        else:
            outputs[i] = blocks.reshape(n_rows, out_bins)

    if outputs is None:
        print('No elements processed.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output + '.npy')
    np.save(out_path, outputs)
    print(f'Saved E2G striped results {outputs.shape} to {out_path}')

    #saved unconditionally now, with shuffling you need to know which rows were actually perturbed
    ic_path = os.path.join(args.output_dir, args.output + '_in_context.npy')
    np.save(ic_path, in_context)
    print(f'Saved in_context flags to {ic_path}')

    if args.perturbation in ('shuffle', 'both_perturbations'):
        sh_path = os.path.join(args.output_dir, args.output + '_shuffled.npy')
        np.save(sh_path, shuffled)
        print(f'Saved shuffled flags to {sh_path} ({shuffled.sum()}/{len(shuffled)} elements shuffled)')

    if average_shuffles:
        std_path = os.path.join(args.output_dir, args.output + '_shuffle_std.npy')
        np.save(std_path, shuffle_std)
        print(f'Saved per-shuffle std to {std_path}')

    #sidecar so the npy is self describing
    meta_path = os.path.join(args.output_dir, args.output + '_args.json')
    with open(meta_path, 'w') as f:
        json.dump({**vars(args), 'n_alt': n_alt, 'n_strands': N_STRANDS, 'n_blocks': n_blocks,
                   'averaged': average_shuffles, 'shape': list(outputs.shape)}, f, indent=2)
    print(f'Saved run args to {meta_path}')
    print('E2G striped run complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run E2G evaluation (striped model with cell type token, dinucleotide shuffle)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='k562_e2g_striped',
                        help='Output filename stem (no extension)')
    parser.add_argument('--output_dir', type=str, default=E2G_DEFAULT_DIR,
                        help='Directory to save output .npy files')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by in the masked region')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp to mask on each side of the element')
    parser.add_argument('--center_tss', action='store_true',
                        help='Center sequence window at TSS instead of enhancer midpoint. '
                             'Enhancers outside the window are predicted without modulation (in_context=0).')
    parser.add_argument('--perturbation', type=str, default='acc',
                        choices=['acc', 'shuffle', 'both_perturbations'],
                        help='acc: scale down accessibility over the element (original behaviour). '
                             'shuffle: dinucleotide shuffle the element, accessibility left intact. '
                             'both_perturbations: apply both to every alternate block.')
    parser.add_argument('--n_shuffles', type=int, default=5,
                        help='Number of dinucleotide shuffles per element (ignored for --perturbation acc)')
    parser.add_argument('--save_all_shuffles', action='store_true',
                        help='Save every individual shuffle instead of their mean. Default is to average '
                             'the shuffles, keeping the original 4 row layout.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed; element i uses seed + i, so shuffles are reproducible per row')
    parser.add_argument('--dist_additional_shuffle', type=int, default=0,
                        help='Extra bp to shuffle on each side of the element. Kept separate from '
                             '--dist_additional_mask so the sequence and accessibility regions can differ.')
    parser.add_argument('--min_shuffle_len', type=int, default=10,
                        help='Skip shuffling elements shorter than this many bp')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of sequences per forward pass')
    parser.add_argument('--load_data', action='store_true',
                        help='Load accessibility data into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to single-cell-type accessibility data (overrides dataset config); '
                             'required because expand_seqs only supports celltypes=1')
    parser.add_argument('--ctt_val', type=int, default=None,
                        help='Cell type token value to pass to the encoder (None = use dataset token)')
    parser.add_argument('--encoder_numcelltypes', type=int, default=None,
                        help='Override number of cell types in the encoder')
    parser.add_argument('--pool', type=int, default=1,
                        help='Average-pool output bins by this factor (default: 1 = no pooling). '
                             'E.g. --pool 32 gives 32 bp resolution if model bins are 1 bp.')
    parser.add_argument('--softplus', action='store_true',
                        help='Apply softplus to model output (default: off)')
    args = parser.parse_args()

    print(args)
    print(f'Running E2G striped ({args.perturbation}), saving to {os.path.join(args.output_dir, args.output)}')
    print(f'Loading checkpoint from {args.ckpt_path}')
    main(args)
