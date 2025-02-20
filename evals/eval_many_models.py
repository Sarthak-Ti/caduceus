#we have several models to evaluate for the project, let's do it and save out the results
#model outputs will be saved in a different folder
'''
12, enformer, -1, outputs/2025-01-25/00-16-16-651049, enformer, Enformer_cnn_1bp, 
13, enformer, -1, outputs/2025-01-25/12-09-36-269879, enformer, Enformer_cnn_8bp, 
14, enformer, -1, outputs/2025-01-25/14-01-50-384463, enformer, Enformer_cnn_32bp, 
15, enformer, -1, outputs/2025-01-25/14-02-50-070142, enformer, Enformer_cnn_128bp, 
21, gpnmsa, -1, outputs/2025-02-05/12-33-11-305656, gpnmsa, gpnmsa_basic_cnn_2, 
22, enformer, -1, outputs/2025-02-05/12-33-11-311476, enformer, Enformer_cnn_128bp_524k_2, 
23, enformer, -1, outputs/2025-02-05/12-33-11-311519, enformer, Enformer_cnn_128bp_393k_2, 
24, gpnmsa, -1, outputs/2025-02-05/12-33-11-311701, gpnmsa, gpnmsa_basic_ohe_2, 

'''

import pandas as pd
import sys
import evals_utils_enformer as e
import torch
import os
import zarr
from numcodecs import Blosc
from tqdm import tqdm

compression = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)

labels_list = [12,13,14,15,21,22,23,24]

labels_df = pd.read_csv('/data1/lesliec/sarthak/caduceus/outputs/labels.csv')
labels_df = labels_df[labels_df['ID'].isin(labels_list)]
out_path = '/data1/lesliec/sarthak/data/borzoi/model_outputs/'

for i in range(len(labels_df)):
    output_dir = labels_df.iloc[i][' output dir'].strip()
    wandb_group = labels_df.iloc[i][' wandb group'].strip()
    wandb_name = labels_df.iloc[i][' wandb name'].strip()
    print(output_dir, wandb_group, wandb_name)
    
    #first check to see if the output tensor already is saved, if so, skip
    if os.path.exists(f'{out_path}/{wandb_name}.zarr'):
        print('Already exists, skipping')
        continue    
    
    ckpt_path = f'/data1/lesliec/sarthak/caduceus/{output_dir}/checkpoints/last.ckpt'
    if wandb_group == 'gpnmsa':
        dataset_class = 'GPNMSA'
    else:
        dataset_class = 'Enformer'
    evals = e.Evals(ckpt_path, dataset_class=dataset_class)
    out = evals.dataset[0]
    print(out[0].shape, out[1].shape)
    out = evals(0)
    print(out.shape)
    # outputs = evals.evaluate(batch_size=2)
    # print(outputs.shape)
    #now we have to create the zarr file
    root = zarr.open(f'{out_path}/{wandb_name}.zarr', mode='w', zarr_format=2)
    root.create_array(
        'evals',
        shape=(len(evals.dataset), out.shape[1], out.shape[2]), #first dim is batch dim
        chunks=(1, out.shape[1], out.shape[2]),
        dtype='f2',  # Float16
        compressors=compression)
    print('full array shape:',root['evals'].shape)
    
    with torch.no_grad():
        for j in tqdm(range(len(evals.dataset))):
            out = evals(j)
            # print('out_shape:',out.shape)
            root['evals'][j] = out.detach().cpu().squeeze().numpy()

    #and now we save it out
    # torch.save(outputs, f'{out_path}/{wandb_name}.pt')