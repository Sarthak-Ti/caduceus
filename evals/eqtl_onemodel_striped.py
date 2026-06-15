#!/usr/bin/env python3
"""
Evaluate eQTL predictions for the striped models.

For each eQTL, runs the model on a ref and alt sequence centered at the gene TSS
and saves a (num_qtls, out_bins, 2) array for downstream AUC evaluation.

Usage:
    python eqtl_onemodel_striped.py --ckpt_path <path> [options]
"""
print('eQTL on one model (striped)', flush=True)
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

from evals.evals_utils_joint import Evals

EQTL_BASE_DIR = '/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/'

MAPPING = {
    'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
    'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
    'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
    'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
    'N': torch.tensor([0, 0, 0, 0], dtype=torch.float32),
}
ONEHOT_MAPPING = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def main(args):
    evals = Evals(
        args.ckpt_path,
        load_data=args.load_data,
        data_idxs=args.data_idxs,
        data_path=args.data_path,
        encoder_numcelltypes=args.encoder_numcelltypes,
    )
    evals.skip_softplus = not args.softplus  # default: skip softplus; pass --softplus to enable
    evals.dataset.return_celltype_idx_og = False  # use hard-coded ctt_val, not dataset token

    qtls = pd.read_csv(EQTL_BASE_DIR + 'LCLs.txt', sep=' ', header=None)
    qtls.columns = ['label', 'qtl_idx', 'gene_idx', 'chrom', 'gene_start', 'gene_end',
                    'strand', 'qtl_loc', 'ref', 'alt', 'sign_target']

    with open(EQTL_BASE_DIR + 'genes.pickle', 'rb') as f:
        gene_annotation = pickle.load(f)
    ordered_genes = sorted(list(gene_annotation.keys()))
    tmpgeneTSS = np.loadtxt(EQTL_BASE_DIR + 'ensemblTSS.txt', dtype='str')
    geneTSS_dic = {tmpgeneTSS[i, 0]: int(tmpgeneTSS[i, 1]) for i in range(tmpgeneTSS.shape[0])}

    length = evals.dataset.length
    half = length // 2

    # allocate after first pass to get out_bins from the model output
    output_array = None

    for i in tqdm(range(qtls.shape[0])):
        temp = qtls.iloc[i]
        chrom = 'chrX' if temp['chrom'] == 23 else 'chr' + str(temp['chrom'])
        pos = temp['qtl_loc'] - 1  # convert to zero-based
        gene_idx = temp['gene_idx']
        tss_loc = geneTSS_dic[ordered_genes[gene_idx]]

        start = tss_loc - half
        end = tss_loc + half

        eQTL_pos = pos - start
        if not (0 <= eQTL_pos < length):
            continue

        idx = evals.dataset.expand_seqs(chrom, start, end)
        outputs1, outputs2 = evals.dataset[idx]
        s = outputs1[0]
        a = outputs1[1]
        su, au = outputs2[0], outputs2[1]

        s = s.unsqueeze(0).repeat(2, 1, 1)  # (2, C, L): ref and alt
        a = a.unsqueeze(0).repeat(2, 1, 1)

        current_nuc = ONEHOT_MAPPING[int(np.argmax(s[0, :, eQTL_pos].cpu().numpy()))]
        assert current_nuc == temp['ref'], (
            f'current nuc {current_nuc} does not match ref {temp["ref"]} for {temp["label"]}'
        )

        s[1, :4, eQTL_pos] = MAPPING[temp['alt']]
        data = ((s, a), (su, au))

        out = evals(data=data, ctt_val=args.ctt_val)
        # out[1] is acc (shape: batch, out_bins, C); channel 0=plus strand, 1=minus strand
        strand_idx = 0 if temp['strand'] == 1 else 1
        pred = out[1][:, :, strand_idx].float().cpu().numpy()  # (2, out_bins)

        if args.pool > 1:
            out_bins_pooled = pred.shape[1] // args.pool
            pred = pred[:, :out_bins_pooled * args.pool].reshape(2, out_bins_pooled, args.pool).mean(axis=2)

        if output_array is None:
            output_array = np.zeros((qtls.shape[0], pred.shape[1], 2))

        output_array[i, :, 0] = pred[0]  # ref
        output_array[i, :, 1] = pred[1]  # alt

    if output_array is None:
        print('No valid eQTLs processed.')
        return

    np.save(args.output, output_array)
    if args.verbose:
        print(f'Saved eQTL results ({output_array.shape}) to {args.output}')
        print('eQTL run complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eQTL on one joint model')
    parser.add_argument('--ckpt_path', type=str,
                        default='/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt',
                        help='Path to the checkpoint')
    parser.add_argument('-o', '--output', type=str,
                        default=EQTL_BASE_DIR + 'output_test.npy',
                        help='Output .npy file path')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--data_idxs', nargs='+', type=int, default=None,
                        help='Dataset indices (cell type subset); default uses all')
    parser.add_argument('--load_data', action='store_true',
                        help='Load data into memory from checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz',
                        help='Path to accessibility data')
    parser.add_argument('--ctt_val', type=int, default=None,
                        help='Cell type token value to use (None = use dataset token)')
    parser.add_argument('--pool', type=int, default=1,
                        help='Number of output bins to average-pool (default: 1 = no pooling)')
    parser.add_argument('--encoder_numcelltypes', type=int, default=None,
                        help='Override number of cell types for the encoder')
    parser.add_argument('--softplus', action='store_true',
                        help='Apply softplus to model output (default: off)')
    args = parser.parse_args()

    print(args)
    if args.verbose:
        print(f'Running eQTL on model, saving to {args.output}')
        print(f'Loading checkpoint from {args.ckpt_path}')

    main(args)