#!/usr/bin/env python3
"""
Evaluate expression predictions from a joint accessibility-expression model.

Single pass over the dataset. Results are preallocated into
(num_regions, num_celltypes, L, C) arrays using the known dataset layout:
    dataset index i  ->  ct = i // num_regions,  region = i % num_regions

Usage:
    python eval_joint_expression.py --ckpt_path <path> [options]

Key options:
    --pool P        spatially pool P consecutive bins before computing correlations
    --skip_softplus disable softplus on model output
    --split         dataset split to evaluate (default: test)
    --out_name NAME save per-sample Pearson/Spearman arrays to
                    /data1/lesliec/sarthak/data/joint_playground/model_out/NAME_{pearson,spearman}.npy
                    shape: (num_regions, num_celltypes, num_strands)
    --save_outputs  additionally save the model predictions as NAME_outputs.npy
    --save_targets  additionally save the targets as NAME_targets.npy
                    both shape: (num_regions, num_celltypes, L, C), post-pooling
    --float16       store the saved outputs/targets as float16 instead of float32
"""
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

from evals.evals_utils_joint import Evals

OUT_DIR = '/data1/lesliec/sarthak/data/joint_playground/model_out'


def pool_spatial(arr, pool_size):
    """Mean-pool arr along the length axis (axis 0) by pool_size. Truncates tail to fit."""
    if pool_size == 1:
        return arr
    L = (arr.shape[0] // pool_size) * pool_size
    return arr[:L].reshape(-1, pool_size, arr.shape[1]).mean(axis=1)


def evaluate_all(evals, pool, num_workers=4, save_outputs=False, save_targets=False, dtype=np.float32):
    """
    Single pass over the full dataset. Computes per-region correlations on the fly
    to avoid storing the full (num_regions, num_celltypes, L, C) arrays.
    Dataset layout: index i -> ct = i // num_regions, region = i % num_regions
    Returns pearson_arr, spearman_arr each of shape (num_regions, num_celltypes, C),
    then outputs_arr and targets_arr, each of shape (num_regions, num_celltypes, L, C)
    if the corresponding flag is set and None otherwise. Those two are preallocated to
    NaN, so any (region, celltype) slot that never got written stays NaN and is visible.
    Values are stored post-pooling, so they line up index-for-index with the correlations.
    """
    num_regions = len(evals.dataset.sequences)
    num_celltypes = evals.dataset.celltypes
    loader = DataLoader(evals.dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    pearson_arr = None
    spearman_arr = None
    outputs_arr = None
    targets_arr = None

    for i, data in enumerate(tqdm(loader, desc='evaluating', leave=False)):
        out = evals(data=data)
        if len(out) < 5:
            continue

        pred = out[1].squeeze(0).detach().cpu().float().numpy()  # (L, C)
        tgt = out[4].squeeze(0).detach().cpu().float().numpy()   # (L, C)

        if pred.ndim == 1:
            pred = pred[:, None]
        if tgt.ndim == 1:
            tgt = tgt[:, None]

        if pool > 1:
            pred = pool_spatial(pred, pool)
            tgt = pool_spatial(tgt, pool)

        if pearson_arr is None: #create after knowing some shapes
            C = pred.shape[1]
            pearson_arr = np.full((num_regions, num_celltypes, C), np.nan, dtype=np.float32)
            spearman_arr = np.full_like(pearson_arr, np.nan)
            L = pred.shape[0]
            if save_outputs:
                outputs_arr = np.full((num_regions, num_celltypes, L, C), np.nan, dtype=dtype)
            if save_targets:
                targets_arr = np.full((num_regions, num_celltypes, L, C), np.nan, dtype=dtype)

        ct = i // num_regions
        if len(data[0]) > 2: #dataset returns the cell type index only if return_celltype_idx_og is set
            assert ct == data[0][2].item(), f"Expected cell type {ct} from index {i}, but got {data[0][2].item()}"
        else:
            assert ct == 0, f"Dataset returns no cell type index, so expected a single cell type, but index {i} implies cell type {ct}"
        region = i % num_regions
        for c in range(pred.shape[1]):
            pearson_arr[region, ct, c] = pearsonr(pred[:, c], tgt[:, c])[0]
            spearman_arr[region, ct, c] = spearmanr(pred[:, c], tgt[:, c])[0]
        if outputs_arr is not None:
            outputs_arr[region, ct] = pred
        if targets_arr is not None:
            targets_arr[region, ct] = tgt

    return pearson_arr, spearman_arr, outputs_arr, targets_arr


def main():
    parser = argparse.ArgumentParser(description='Evaluate expression predictions')
    parser.add_argument('--ckpt_path', help='Path to model checkpoint')
    parser.add_argument('--split', default='test',
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--pool', type=int, default=1,
                        help='Spatial pooling factor: average this many consecutive bins '
                             'before computing correlations (default: 1 = no pooling)')
    parser.add_argument('--skip_softplus', action='store_true',
                        help='Disable softplus activation on model output')
    parser.add_argument('--device', default=None,
                        help='Torch device string, e.g. "cuda:0" (default: auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes for prefetching (default: 4)')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Additionally save the model predictions as <out_name>_outputs.npy '
                             '(num_regions, num_celltypes, L, C), post-pooling')
    parser.add_argument('--save_targets', action='store_true',
                        help='Additionally save the targets as <out_name>_targets.npy, same shape')
    parser.add_argument('--float16', action='store_true',
                        help='Store saved outputs/targets as float16 instead of float32 (halves file size)')
    parser.add_argument('--out_name', default=None,
                        help='If set, save per-sample correlation arrays as '
                             f'{OUT_DIR}/<out_name>_pearson.npy and _spearman.npy  '
                             'Shape: (num_regions, num_celltypes, num_strands)')
    args = parser.parse_args()
    if (args.save_outputs or args.save_targets) and not args.out_name:
        parser.error('--save_outputs/--save_targets require --out_name, otherwise the arrays are discarded')
    raw_dtype = np.float16 if args.float16 else np.float32

    evals = Evals(args.ckpt_path, split=args.split, device=args.device)
    if args.skip_softplus:
        evals.skip_softplus = True

    num_regions = len(evals.dataset.sequences)
    num_celltypes = evals.dataset.celltypes

    print(f"Checkpoint   : {args.ckpt_path}")
    print(f"Split        : {args.split}  ({len(evals.dataset)} examples)")
    print(f"Regions      : {num_regions}")
    print(f"Cell types   : {num_celltypes}")
    print(f"Pool         : {args.pool} bins")
    print(f"skip_softplus: {evals.skip_softplus}")
    print(f"save_outputs : {args.save_outputs}   save_targets: {args.save_targets}   dtype: {np.dtype(raw_dtype).name}")

    pearson_arr, spearman_arr, outputs_arr, targets_arr = evaluate_all(
        evals, pool=args.pool, num_workers=args.num_workers,
        save_outputs=args.save_outputs, save_targets=args.save_targets, dtype=raw_dtype)
    # shape: (num_regions, num_celltypes, num_strands)

    print(f"\nMean Pearson : {np.nanmean(pearson_arr):.4f}")
    print(f"Mean Spearman: {np.nanmean(spearman_arr):.4f}")

    if args.out_name:
        os.makedirs(OUT_DIR, exist_ok=True)
        pearson_path = os.path.join(OUT_DIR, f'{args.out_name}_pearson.npy')
        spearman_path = os.path.join(OUT_DIR, f'{args.out_name}_spearman.npy')
        np.save(pearson_path, pearson_arr)
        np.save(spearman_path, spearman_arr)
        print(f"Saved Pearson  {pearson_arr.shape} → {pearson_path}")
        print(f"Saved Spearman {spearman_arr.shape} → {spearman_path}")
        for arr, tag in ((outputs_arr, 'outputs'), (targets_arr, 'targets')):
            if arr is None:
                continue
            path = os.path.join(OUT_DIR, f'{args.out_name}_{tag}.npy')
            np.save(path, arr)
            print(f"Saved {tag:8s} {arr.shape} {arr.dtype} ({arr.nbytes / 1e6:.0f} MB) → {path}")
            unfilled = int(np.isnan(arr[:, :, 0, 0]).sum())
            if unfilled:
                print(f"  WARNING: {unfilled} (region, celltype) slots never written, still NaN")


if __name__ == '__main__':
    main()