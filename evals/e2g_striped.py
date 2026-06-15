#!/usr/bin/env python3
"""
E2G (enhancer-to-gene) evaluation for the striped model with cell type token.

For each enhancer-gene pair, runs the model on a before (unperturbed) and after
(accessibility-masked at the enhancer) sequence and saves a
(n_elements, 4, out_bins) array for downstream AUC evaluation.

Output layout (rows axis):
  row 0: before, + strand
  row 1: before, - strand
  row 2: after  (enhancer masked), + strand
  row 3: after  (enhancer masked), - strand
"""
print('E2G striped evaluation', flush=True)
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

E2G_DEFAULT_DIR = '/data1/lesliec/sarthak/data/joint_playground/e2g/'


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
    outputs = None  # lazy alloc after first forward pass to infer out_bins and n_channels

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

        s = s.unsqueeze(0).repeat(2, 1, 1)  # (2, C, L): batch 0=before, 1=after
        a = a.unsqueeze(0).repeat(2, 1, 1)

        if in_context[i]:
            seq_start = center - half
            if args.center_tss:
                startmask = (start - seq_start) - args.dist_additional_mask
                endmask   = (end   - seq_start) + args.dist_additional_mask
            else:
                mask_len  = end - start + args.dist_additional_mask * 2
                startmask = half - mask_len // 2
                endmask   = half + mask_len // 2
            startmask = max(0, startmask)
            endmask   = min(length, endmask)
            # divide the primary accessibility track for the after sequence
            a[1, 0, startmask:endmask] = a[1, 0, startmask:endmask] / args.scale_factor

        out = evals(data=((s, a), (su, au)), ctt_val=args.ctt_val)
        # out[1] is acc: (batch=2, out_bins, n_channels)
        pred = out[1].float().cpu().numpy()

        if args.pool > 1:
            n_pooled = pred.shape[1] // args.pool
            pred = pred[:, :n_pooled * args.pool, :].reshape(2, n_pooled, args.pool, 2).mean(axis=2)

        # pred shape: (2, out_bins, 2): batch x bins x strand (0=plus, 1=minus)
        # output rows: 0=before+, 1=before-, 2=after+, 3=after-
        if outputs is None:
            outputs = np.zeros((len(gs_df), 4, pred.shape[1]), dtype=np.float32)

        outputs[i, 0, :] = pred[0, :, 0]  # before +
        outputs[i, 1, :] = pred[0, :, 1]  # before -
        outputs[i, 2, :] = pred[1, :, 0]  # after  +
        outputs[i, 3, :] = pred[1, :, 1]  # after  -

    if outputs is None:
        print('No elements processed.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output + '.npy')
    np.save(out_path, outputs)
    print(f'Saved E2G striped results {outputs.shape} to {out_path}')
    if args.center_tss:
        ic_path = os.path.join(args.output_dir, args.output + '_in_context.npy')
        np.save(ic_path, in_context)
        print(f'Saved in_context flags to {ic_path}')
    print('E2G striped run complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run E2G evaluation (striped model with cell type token)')
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
    print(f'Running E2G striped, saving to {os.path.join(args.output_dir, args.output)}')
    print(f'Loading checkpoint from {args.ckpt_path}')
    main(args)