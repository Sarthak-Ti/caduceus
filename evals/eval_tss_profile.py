#generic scalar-expression evaluation for TSS models whose decoder returns TWO outputs
#(TSSProfileDecoder: a per-bp profile and a scalar count). Only the COUNT head is scored here --
#it is the same log-scale quantity a single-output TSSDecoder predicts, so these numbers are
#directly comparable to eval_tss.py's. The profile head is not evaluated.
#no accessibility ablation, no extra masking -- just the dataset's standard evaluation path
#(evaluating=True, so no random shift and no rc_aug, but rc_strand still reverse complements
#minus-strand genes exactly as it does during training).
print('TSS expression eval (no perturbation, profile decoder)', flush=True)

import numpy as np
import sys
import os
sys.path.append('/data1/lesliec/sarthak/caduceus/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr, spearmanr

#reuse the checkpoint loading / state-dict surgery from the profile E2G script so both stay in
#sync -- that Evals builds the decoder from the config's _name_ and returns the count head
from e2g_tss_profile import Evals


def main(args):
    #only the count head is scored, so switch off the RNA-seq source the profile head trains on:
    #it is a 12G npz and the dataset would read a 2MB per-bp slice for every gene for nothing
    evals_kwargs = dict(load_data=args.load_data, split=args.split, expression_data_path=None)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path
    if args.tss_json_file is not None:
        evals_kwargs['tss_json_file'] = args.tss_json_file

    evals = Evals(args.ckpt_path, **evals_kwargs)
    ds = evals.dataset

    #report the window geometry the checkpoint was trained with so the log is self documenting
    tss_final_pos = ds.upstream if ds.upstream is not None else ds.length // 2
    print(f'split={args.split}  n_genes={len(ds)}')
    print(f'window: length={ds.length}, TSS at index {tss_final_pos}, rc_strand={ds.rc_strand}, '
          f'append_gene_mask={ds.append_gene_mask}, append_tss_mask={ds.append_tss_mask}')
    #pool_region/bp_predictor belong to TSSDecoder, count_region/profile_region to
    #TSSProfileDecoder -- print all four so either decoder logs its actual readout region
    print(f'decoder: {type(evals.decoder).__name__} '
          f'count_region={getattr(evals.decoder, "count_region", None)} '
          f'profile_region={getattr(evals.decoder, "profile_region", None)} '
          f'pool_region={getattr(evals.decoder, "pool_region", None)} '
          f'bp_predictor={getattr(evals.decoder, "bp_predictor", None)}')

    genes = list(ds.genes)
    if args.limit is not None:
        #debug path: only the first N genes of the split
        n = min(args.limit, len(genes))
        genes = genes[:n]
        ds = torch.utils.data.Subset(ds, range(n))

    #gene metadata in prediction row order, so downstream analysis can join across models
    #(which may be trained on different gene jsons) and stratify without reopening the json
    info = evals.dataset.tss_dict
    chroms = np.array([info[g]['chrom'] for g in genes])
    tss_pos = np.array([info[g]['tss'] for g in genes])
    gene_starts = np.array([-1 if info[g].get('gene_start') is None else info[g]['gene_start'] for g in genes])
    gene_ends = np.array([-1 if info[g].get('gene_end') is None else info[g]['gene_end'] for g in genes])

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=False, drop_last=False,
    )

    preds, targets, strands = [], [], []
    for batch in tqdm(loader, total=len(loader)):
        (x, acc), aux = batch
        #aux = (seq_unmask, acc_umask, counts, tss_mask, gene_mask, strand, expression);
        #Evals.__call__ passes both masks to the decoder, which uses whichever it needs, and
        #returns just the count head -- (batch, n_tracks) -- for the two-output decoder
        out = evals(data=((x, acc), tuple(aux)))
        preds.append(out.float().cpu().numpy())
        targets.append(aux[2].float().numpy())
        strands.append(aux[5].numpy())

    preds = np.concatenate(preds, axis=0)          # (n_genes, n_tracks)
    targets = np.concatenate(targets, axis=0)      # (n_genes,)
    strands = np.concatenate(strands, axis=0)      # (n_genes,)
    assert len(preds) == len(genes), f'{len(preds)} predictions for {len(genes)} genes'

    #metrics on the same quantities the count loss compares during training: counts vs y[2]
    p = preds.squeeze(-1) if preds.ndim > 1 and preds.shape[1] == 1 else preds[:, 0]
    mse = float(np.mean((p - targets) ** 2))
    pear = pearsonr(p, targets)
    spear = spearmanr(p, targets)
    print(f'\nn={len(p)}  mse={mse:.4f}  pearson_r={pear[0]:.4f}  spearman_rho={spear[0]:.4f}')
    print(f'pred  mean={p.mean():.3f} std={p.std():.3f} min={p.min():.3f} max={p.max():.3f}')
    print(f'target mean={targets.mean():.3f} std={targets.std():.3f} min={targets.min():.3f} max={targets.max():.3f}')
    for s, name in ((1, '+'), (-1, '-')):
        m = strands == s
        if m.sum() > 2:
            print(f"  strand {name}: n={int(m.sum())} pearson_r={pearsonr(p[m], targets[m])[0]:.4f} "
                  f"mse={float(np.mean((p[m] - targets[m]) ** 2)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output + '.npz')
    np.savez(
        out_path,
        genes=np.array(genes), preds=preds, targets=targets, strands=strands,
        chroms=chroms, tss=tss_pos, gene_starts=gene_starts, gene_ends=gene_ends,
        split=args.split, ckpt_path=args.ckpt_path,
        tss_json_file=evals.dataset_args.get('tss_json_file', ''),
    )
    print(f'saved predictions to {out_path}')
    print('eval complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scalar expression eval for a two-output (profile+count) TSS model')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='tss_profile_eval',
                        help='Output filename (no extension)')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/lesliec/sarthak/data/joint_playground/tss_eval/',
                        help='Directory to save the output .npz file')
    parser.add_argument('--split', type=str, default='test',
                        help="Split to evaluate ('test', 'val', 'train'); None-like splits not supported")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation (training used 1 per GPU at 524kb; raise if it fits)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers (the dataset reopens the npz per item, so >0 helps)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Debug: only evaluate the first N genes of the split')
    parser.add_argument('--load_data', action='store_true',
                        help='Load data into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to accessibility data (overrides dataset config)')
    parser.add_argument('--tss_json_file', type=str, default=None,
                        help='Path to the gene JSON (overrides dataset config)')
    args = parser.parse_args()

    print(args)
    print(f'Loading checkpoint from {args.ckpt_path}')

    main(args)
