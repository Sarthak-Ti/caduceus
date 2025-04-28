#this script runs dsQTL on one model, but does it more efficiently using a data loader
print('dsQTL on one model', flush=True)
# import sys
# sys.path.append('/data1/lesliec/sarthak/caduceus/')
from evals_utils_joint import Evals
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

def main(args):
    ckpt_path = args.ckpt_path
    evals = Evals(ckpt_path,load_data=args.load_data,data_idxs=args.data_idxs)

    mapping = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'N': torch.tensor([0, 0, 0, 0], dtype=torch.float32),
    }

    onehot_mapping = {0:'A',1:'C',2:'G',3:'T'}

    qtls = pd.read_csv('/data1/lesliec/sarthak/data/joint_playground/dsQTL/filtered_dsqtls.txt', sep='\t')

    bims = {}
    for i in range(1,23):
        bim_path = f'/data1/deyk/extras/1000G_BIMS_hg38/1000G.EUR.QC.{i}.bim'
        bim = pd.read_csv(bim_path, sep='\t', header=None)
        bims[i] = bim


    bim_pos = {
        chrom: dict(zip(bims[chrom][1].values, bims[chrom][3].index))
        for chrom in bims
    }

    length = 524288
    si = 524288//2 - int(args.mask_size/2)
    se = 524288//2 + int(args.mask_size/2)

    output_array = np.zeros((qtls.shape[0],500,2))

    for i in tqdm(range(qtls.shape[0])):
        rsid = qtls.iloc[i]['SNPname2']
        chrom = qtls.iloc[i]['chrom_hg19']
        bimrow = bim_pos[int(chrom[3:])][rsid]
        bimval = bims[int(chrom[3:])].iloc[bimrow]
        pos = bimval[3]-1
        start = pos - length//2
        end = pos + length//2
        
        idx = evals.dataset.expand_seqs(chrom,start,end)
        # data = evals.dataset[idx]
        
    #now we will create a dataloader, remove the original elements of the array in the dataset, so just iterate over these elements. Load them
    #apply the alternate seq to them. And then we're good!! Idk how to do this tho, sounds complicated lol!
        
        # ((s,a),(su,au)) = evals.dataset[idx]
        # data = (None,None,su,au) #can be s and a or None, it isn't used by the mask function
        # out = evals.mask(si,se, data=data, mask_accessibility=True)
        
        # current_nuc = out[2][524288//2].cpu().numpy()
        # current_nuc = np.argmax(current_nuc)
        # current_nuc = onehot_mapping[current_nuc]
        
        # if current_nuc == bimval[4]:
        #     alt_key = 5
        # elif current_nuc == bimval[5]:
        #     alt_key = 4
        # else:
        #     raise ValueError("Neither of the alleles match the current nucleotide")
        
        # data[2][524288//2,:4] = mapping[bimval[alt_key]]
        
        # out2 = evals.mask(si,se, data=data, mask_accessibility=True)
        
        # pred1 = out[1][0, 524288//2-250:524288//2+250, 0].cpu().numpy()
        # pred2 = out2[1][0, 524288//2-250:524288//2+250, 0].cpu().numpy()
        # output_array[i,:,0] = pred1
        # output_array[i,:,1] = pred2
        

    np.save(f'/data1/lesliec/sarthak/data/joint_playground/dsQTL/{args.output}', output_array)
    if args.verbose:
        print(f'saved dsQTL results to {args.output}')
        print(f'loaded checkpoint from {args.ckpt_path}')
        print('dsQTL run complete')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run dsQTL on one model')
    parser.add_argument('--ckpt_path', type=str, default='/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt', help='Path to the checkpoint')
    parser.add_argument('-o', '--output', type=str, default='/data1/lesliec/sarthak/data/joint_playground/dsQTL/basic_dsqtl.npy', help='Output file path')
    parser.add_argument('--mask_size', type=int, default=500, help='Size of the mask')
    parser.add_argument('-v', '--verbose', action='store_false', help='Disable verbose output')
    parser.add_argument('--data_idxs', nargs='+',type=int,help='List of mask sizes', default=None)
    parser.add_argument('--load_data', action='store_true', help='Load data from the checkpoint')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    args = parser.parse_args()
    
    print(args)
    if args.verbose:
        # print(f'args: {args}')
        print(f'running dsQTL on model and saving results to {args.output}')
        print(f'loading checkpoint from {args.ckpt_path}')
        print(f'mask size: {args.mask_size}')
    
    main(args)
    # ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/last.ckpt' #for the generalizing model