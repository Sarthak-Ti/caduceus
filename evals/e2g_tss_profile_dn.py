#e2g for TSS models whose decoder returns TWO outputs (TSSProfileDecoder: profile + counts),
#and for the single-output TSSDecoder models as well -- Evals builds the decoder from the
#config's _name_, so the same script covers both rounds.
#
#Variant of e2g_tss_profile.py that can also dinucleotide shuffle the element instead of, or as
#well as, scaling down its accessibility. See --perturbation.
#
#Output layout (columns axis): column 0 is the unperturbed 'before', columns 1.. are the
#perturbed 'after' (a single mean column unless --save_all_shuffles).
#  default             -> (n_pairs, 2)             [before, after]
#  --save_all_shuffles -> (n_pairs, 1 + n_shuffles) [before, shuf0, shuf1, ...]
#
#NOTE: the count head is a plain regression head, so these predictions live on the natural-log
#target scale and CAN BE NEGATIVE -- score the perturbation as a log-space difference
#(col 1 - col 0), not a ratio. Averaging the shuffle columns therefore takes a GEOMETRIC mean in
#expression space; --save_all_shuffles keeps the individual columns so that choice can be made
#downstream instead (the output is one scalar per row, so keeping them all costs ~1MB).
print('E2G for centering at TSS model (profile decoder, dinucleotide shuffle)', flush=True)

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

#reuse the checkpoint loading / state-dict surgery from the unperturbed E2G script so the two
#stay in sync -- same pattern eval_tss.py uses to borrow Evals from e2g_tss.py
from e2g_tss_profile import Evals
from evals.utils.dinuc_shuffle import dinucleotide_shuffle

#channels 0-4 of the sequence are the A,C,G,T,N one-hot; channel 5 is the appended gene-body or
#TSS-neighborhood mask (append_gene_mask / append_tss_mask). Only the one-hot part may be
#shuffled: dinucleotide_shuffle returns a pure one-hot, so handing it all 6 channels would zero
#the mask channel wherever the element overlaps it. Restricting to [:N_ONEHOT] leaves the RNG
#stream untouched (an all-zero channel contributes no transitions and draws no random numbers),
#so this is bit-identical to the 6-channel shuffle the striped/enformer scripts do.
N_ONEHOT = 5


def main(args):
    TSS_bounds_file = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
    tss = pd.read_csv(TSS_bounds_file, sep="\t")

    path = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    gs_df = pd.read_csv(path, sep="\t")

    #the profile models train with an RNA-seq track (expression_data_path), but only the count
    #head is scored here, so switch that source off: it is a 12G npz that --load_data would pull
    #into RAM on top of the accessibility npz, and a 2MB per-bp slice read for every gene
    evals_kwargs = dict(load_data=args.load_data, expression_data_path=None)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path

    evals = Evals(args.ckpt_path, **evals_kwargs)
    # look_tss_len = evals.cfg['dataset']['tss_distance']
    print(f'decoder: {type(evals.decoder).__name__} '
          f'count_region={getattr(evals.decoder, "count_region", None)} '
          f'profile_region={getattr(evals.decoder, "profile_region", None)} '
          f'pool_region={getattr(evals.decoder, "pool_region", None)}')

    #gene metadata comes from the checkpoint's own tss_json_file (via the dataset) rather than
    #a hardcoded path, so the same script works for models trained on different gene sets
    #(e.g. bulk RNA vs single-cell). split=None in Evals means all genes are present.
    tss_dict = evals.dataset.tss_dict
    gene_to_idx = {gene: i for i, gene in enumerate(evals.dataset.genes)}

    name_to_ensg = tss.set_index('name')['Ensembl_ID'].to_dict()
    name_to_ensg = {name: ensg for name, ensg in name_to_ensg.items() if ensg in tss_dict} #remove extra values
    gs_df['ensg_id'] = gs_df['measuredGeneSymbol'].map(name_to_ensg)
    gs_df = gs_df.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)
    print(f'{len(gs_df)} CRISPR pairs with a gene in this model\'s tss_json_file')

    #row 0 of each batch is the unperturbed reference, rows 1..n_alt are the perturbed versions
    n_alt = args.n_shuffles if args.perturbation in ('shuffle', 'both_perturbations') else 1

    #by default collapse the shuffles to their mean, so the output keeps the original 2 column
    #layout. --save_all_shuffles keeps every individual shuffle instead
    average_shuffles = args.perturbation in ('shuffle', 'both_perturbations') and not args.save_all_shuffles
    n_cols = 2 if average_shuffles else 1 + n_alt
    print(f'perturbation={args.perturbation}  n_alt={n_alt}  averaged={average_shuffles}  '
          f'output columns={n_cols}')

    #window geometry read off the dataset so it matches TSSDataset.__getitem__ exactly:
    #  upstream=None -> TSS centered at length//2 (TSS-centered models)
    #  upstream set  -> TSS placed `upstream` bp from the left edge of the final (post-RC)
    #                   sequence, so the window is ASYMMETRIC about the TSS (Decima-style)
    #evaluating=True in Evals means shift_sequences is disabled, so there is no random shift.
    SEQ_LEN = evals.dataset.length
    TSS_FINAL_POS = evals.dataset.upstream if evals.dataset.upstream is not None else SEQ_LEN // 2
    print(f'window: length={SEQ_LEN}, TSS at index {TSS_FINAL_POS}, rc_strand={evals.dataset.rc_strand}')

    outputs = np.zeros((len(gs_df), n_cols)) #num_samples x (1 unperturbed + perturbed) values
    in_context = np.ones(len(gs_df), dtype=bool)
    shuffled = np.zeros(len(gs_df), dtype=bool) #whether the element actually got dinucleotide shuffled
    #spread across the shuffles, otherwise averaging throws that information away
    shuffle_std = np.zeros(len(gs_df)) if average_shuffles else None

    for i, row in tqdm(gs_df.iterrows(), total=len(gs_df)):
        chrom = row['chrom']
        start = row['chromStart']
        end = row['chromEnd']
        ensgid = row['ensg_id']
        gene_info = tss_dict[ensgid]
        temp_tss = gene_info['tss']

        #minus-strand genes are reverse complemented by the dataset when rc_strand is set, so
        #the element coordinates have to be mapped through the same flip
        flip = evals.dataset.rc_strand and gene_info['strand'] == '-'
        tss_pre_pos = TSS_FINAL_POS if not flip else (SEQ_LEN - 1 - TSS_FINAL_POS)
        window_start = temp_tss - tss_pre_pos #genomic coord at pre-flip sequence index 0

        #pre-flip element bounds; check the element falls inside the window before flipping
        lo = start - window_start
        hi = end - window_start
        in_context[i] = (lo >= 0) and (hi <= SEQ_LEN)
        if flip:
            lo, hi = SEQ_LEN - hi, SEQ_LEN - lo

        #get idx based on the protein
        idx = gene_to_idx[ensgid]

        ((s,a),(su,au,counts,tss_mask,gene_mask,strand,expression)) = evals.dataset[idx]
        s = s.unsqueeze(0).repeat(1 + n_alt, 1, 1) #now is (1+n_alt) x 6 x 524288
        a = a.unsqueeze(0).repeat(1 + n_alt, 1, 1) #now is (1+n_alt) x 2 x 524288
        tss_mask = tss_mask.unsqueeze(0).repeat(1 + n_alt, 1) if tss_mask is not None else None #now is (1+n_alt) x 524288
        gene_mask = gene_mask.unsqueeze(0).repeat(1 + n_alt, 1) if gene_mask is not None else None #now is (1+n_alt) x 524288

        if in_context[i]:
            #alter the accessibility in the region around the element
            if args.perturbation in ('acc', 'both_perturbations'):
                startmask = max(0, lo - args.dist_additional_mask)
                endmask   = min(SEQ_LEN, hi + args.dist_additional_mask)

                a[1:,0,startmask:endmask] = a[1:,0,startmask:endmask] / args.scale_factor

            #dinucleotide shuffle the element itself, leaving the accessibility track intact
            if args.perturbation in ('shuffle', 'both_perturbations'):
                sh_start = max(0, lo - args.dist_additional_shuffle)
                sh_end   = min(SEQ_LEN, hi + args.dist_additional_shuffle)
                if sh_end - sh_start >= args.min_shuffle_len:
                    try:
                        #shuffle from row 0, which is never perturbed, and only over the one-hot
                        #channels so the appended gene/TSS mask in channel 5 survives intact.
                        #seed is keyed to the row index so the shuffles are reproducible
                        #regardless of batching or where you restart
                        shuf = dinucleotide_shuffle(s[0,:N_ONEHOT,sh_start:sh_end], n_shuffles=n_alt,
                                                    random_state=args.seed + i)
                        for k in range(n_alt):
                            s[1 + k,:N_ONEHOT,sh_start:sh_end] = shuf[k]
                        shuffled[i] = True
                    except ValueError:
                        #every shuffle came back identical (low complexity element), leave it unshuffled
                        pass

        #chunk the forward pass, a full (1+n_alt) batch at 524kb will not fit for large --n_shuffles
        preds = []
        for j in range(0, 1 + n_alt, args.batch_size):
            sb = s[j:j + args.batch_size]
            ab = a[j:j + args.batch_size]
            tb = tss_mask[j:j + args.batch_size] if tss_mask is not None else None
            gb = gene_mask[j:j + args.batch_size] if gene_mask is not None else None
            #counts is (batch, n_tracks) = (batch, 1) here, so reshape(-1) gives one value per row
            out = evals(data=((sb,ab), (None, None, None, tb, gb, None, None)))
            preds.append(out.float().cpu().numpy().reshape(-1))
        preds = np.concatenate(preds, axis=0) #(1+n_alt,): row 0 unperturbed, rows 1.. perturbed

        if average_shuffles:
            outputs[i, 0] = preds[0]
            outputs[i, 1] = preds[1:].mean()
            shuffle_std[i] = preds[1:].std()
        else:
            outputs[i] = preds

    out_path = os.path.join(args.output_dir, args.output + '.npy')
    np.save(out_path, outputs)
    print(f'saved E2G results {outputs.shape} to {out_path}')
    ic_path = os.path.join(args.output_dir, args.output + '_in_context.npy')
    np.save(ic_path, in_context)
    print(f'saved in_context flags ({in_context.sum()}/{len(in_context)} in window) to {ic_path}')

    if args.perturbation in ('shuffle', 'both_perturbations'):
        sh_path = os.path.join(args.output_dir, args.output + '_shuffled.npy')
        np.save(sh_path, shuffled)
        print(f'saved shuffled flags to {sh_path} ({shuffled.sum()}/{len(shuffled)} elements shuffled)')

    if average_shuffles:
        std_path = os.path.join(args.output_dir, args.output + '_shuffle_std.npy')
        np.save(std_path, shuffle_std)
        print(f'saved per-shuffle std to {std_path}')

    #sidecar so the npy is self describing
    meta_path = os.path.join(args.output_dir, args.output + '_args.json')
    with open(meta_path, 'w') as f:
        json.dump({**vars(args), 'n_alt': n_alt, 'n_cols': n_cols,
                   'averaged': average_shuffles, 'shape': list(outputs.shape)}, f, indent=2)
    print(f'saved run args to {meta_path}')
    print('E2G run complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run E2G evaluation for a TSS model with dinucleotide shuffling')
    parser.add_argument('--ckpt_path', type=str,
                        default='/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt',
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='k562_tss_sc_rna_poisson_ep35_dinuc',
                        help='Output filename (no extension)')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/lesliec/sarthak/data/joint_playground/e2g/',
                        help='Directory to save output .npy file')
    parser.add_argument('--perturbation', type=str, default='acc',
                        choices=['acc', 'shuffle', 'both_perturbations'],
                        help='acc: scale down accessibility over the element (original behaviour). '
                             'shuffle: dinucleotide shuffle the element, accessibility left intact. '
                             'both_perturbations: apply both to every alternate row.')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by in the masked region')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp to mask on each side of the element')
    parser.add_argument('--n_shuffles', type=int, default=5,
                        help='Number of dinucleotide shuffles per element (ignored for --perturbation acc)')
    parser.add_argument('--save_all_shuffles', action='store_true',
                        help='Save every individual shuffle instead of their mean. Default is to average '
                             'the shuffles, keeping the original 2 column layout. Cheap here (one scalar '
                             'per row), and avoids baking in a geometric mean on the log-scale count head.')
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
                        help='Load data into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to accessibility data (overrides dataset config)')
    args = parser.parse_args()

    print(args)
    print(f'Running E2G ({args.perturbation}) on model and saving results to {args.output}')
    print(f'Loading checkpoint from {args.ckpt_path}')

    main(args)
