# ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt'
print('caQTL on one model', flush=True)
# import sys
# sys.path.append('/data1/lesliec/sarthak/caduceus/')
from evals_utils_joint import Evals
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import zarr

def main(args):
    ckpt_path = args.ckpt_path
    evals = Evals(ckpt_path,load_data=args.load_data, data_idxs=args.data_idxs, data_path=args.data_path)
    zarr_open = zarr.open(args.output, mode='r+')

    mapping = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'N': torch.tensor([0, 0, 0, 0], dtype=torch.float32),
    }

    onehot_mapping = {0:'A',1:'C',2:'G',3:'T'}

    qtls = pd.read_csv('/data1/lesliec/sarthak/data/joint_playground/caQTL/caqtls.eu.lcls.benchmarking.all.tsv', sep='\t')
    qtls = qtls[qtls['var.isused']]

    length = 524288
    si = 524288//2 - int(args.mask_size/2)
    se = 524288//2 + int(args.mask_size/2)
    if args.end > qtls.shape[0]:
        args.end = qtls.shape[0]

    for i in tqdm(range(args.start,args.end)):
        temp = qtls.iloc[i]
        chrom = temp['var.chr']
        pos = temp['var.pos_hg38']-1
        ref_allele = temp['var.allele1']
        alt_allele = temp['var.allele2']
        
        start = pos - length//2
        end = pos + length//2
        
        idx = evals.dataset.expand_seqs(chrom,start,end)
        # data = evals.dataset[idx]
        
        ((s,a),(su,au)) = evals.dataset[idx]
        data = (None,None,su,au) #can be s and a or None, it isn't used by the mask function
        out = evals.mask(si,se, data=data, mask_accessibility=True)
        
        current_nuc = out[2][524288//2].cpu().numpy()
        current_nuc = np.argmax(current_nuc)
        current_nuc = onehot_mapping[current_nuc]
        
        assert current_nuc == ref_allele, f"Current nucleotide {current_nuc} does not match reference allele {ref_allele}"
        
        data[2][524288//2,:4] = mapping[alt_allele]
        
        out2 = evals.mask(si,se, data=data, mask_accessibility=True)
        
        pred1 = out[1][0, 524288//2-250:524288//2+250, 0].cpu().numpy()
        pred2 = out2[1][0, 524288//2-250:524288//2+250, 0].cpu().numpy()
        preds_combined = np.stack((pred1, pred2), axis=1)
        zarr_open[i] = preds_combined
        # zarr_open[i,:,0] = pred1
        # zarr_open[i,:,1] = pred2
        

    # np.save(f'/data1/lesliec/sarthak/data/joint_playground/dsQTL/{args.output}', output_array)
    if args.verbose:
        print(f'saved caQTL results to {args.output}')
        # print(f'loaded checkpoint from {args.ckpt_path}')
        print('caQTL run complete')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run caQTL on one model')
    parser.add_argument('--ckpt_path', type=str, default='/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt', help='Path to the checkpoint')
    parser.add_argument('-o', '--output', type=str, default='/data1/lesliec/sarthak/data/joint_playground/dsQTL/basic_dsqtl.npy', help='Output file path')
    parser.add_argument('--mask_size', type=int, default=500, help='Size of the mask')
    parser.add_argument('-v', '--verbose', action='store_false', help='Disable verbose output')
    parser.add_argument('--data_idxs', nargs='+',type=int,help='List of mask sizes', default=None) #intentionally passes None as a default, tells it to not find any data idxs
    parser.add_argument('--load_data', action='store_true', help='Load data from the checkpoint')
    parser.add_argument('--data_path', type=str, default='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz', help='Path to the data')
    parser.add_argument('--start', type=int, default=0, help='Start index for the data')
    parser.add_argument('--end', type=int, default=None, help='End index for the data')
    args = parser.parse_args()
    
    print(args)
    if args.verbose:
        # print(f'args: {args}')
        print(f'running caQTL on model and saving results to {args.output}')
        print(f'loading checkpoint from {args.ckpt_path}')
        print(f'mask size: {args.mask_size}')
    
    main(args)
    # ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/last.ckpt' #for the generalizing model