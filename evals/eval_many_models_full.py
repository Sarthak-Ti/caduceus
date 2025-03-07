#this is a way to evaluate some models that are all trained at 128 bp resolution, it wouldn't be hard to train it further by doing some more stuff or making this more modular

import pandas as pd
import sys
import evals_utils_enformer as e
import torch
import os
import zarr
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np

#we want  numbers 25-31
labels_list = [25,26,27,28,29,30,31]

labels_df = pd.read_csv('/data1/lesliec/sarthak/caduceus/outputs/labels.csv')
labels_df = labels_df[labels_df['ID'].isin(labels_list)]
out_path = '/data1/lesliec/sarthak/data/borzoi/model_outputs/'

for i in range(len(labels_df)):
    output_dir = labels_df.iloc[i][' output dir'].strip()
    wandb_group = labels_df.iloc[i][' wandb group'].strip()
    wandb_name = labels_df.iloc[i][' wandb name'].strip()
    print(output_dir, wandb_group, wandb_name)
    
    #first load in everything
    
    ckpt_path = f'/data1/lesliec/sarthak/caduceus/{output_dir}/checkpoints/last.ckpt'
    if wandb_group == 'gpnmsa':
        dataset_class = 'GPNMSA'
    else:
        dataset_class = 'Enformer'
    evals = e.Evals(ckpt_path, dataset_class=dataset_class)
    
    evals.evaluate_zarr(f'{out_path}/{wandb_name}.zarr',num_workers=8)
    
    evals.correlate(f'{out_path}/{wandb_name}.zarr',axis=0) #because it's 896 x num_targets, want targets to stay but the 896 is what gets correlated