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
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np

# compression = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE) #we will not be doing compression

labels_list = [12,13,14,15,21,22,23,24]

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
    root = zarr.open(f'{out_path}/{wandb_name}.zarr', mode='r+')
    print('evals array shape:',root['evals'].shape)

    try:
        root.create_array('corrs', shape=(len(evals.dataset), 674), chunks=(1, 674), dtype='float32')
    except zarr.errors.ContainsArrayError:
        print("corr Array already exists")
        assert root['corrs'].shape == (len(evals.dataset), 674), f"Shape mismatch: {root['corrs'].shape} vs {(len(evals.dataset), 674)}"
    
    for j in tqdm(range(len(evals.dataset))):
        # root['corrs'][j] = out.detach().cpu().squeeze().numpy()
        #now we loop through the outputs
        data,label = evals.dataset[j]
        out = root['evals'][j]
        # print(label.shape, out.shape)
        # break
        label = label.numpy() #just because it's now a numpy array we can calculate the correlation of
        corrs = np.zeros((label.shape[1]))
        for k in range(label.shape[1]):
            corr = spearmanr(label[:,k], out[:,k])
            corrs[k] = corr.correlation if not np.isnan(corr.correlation) else 0.0 #basically sets it to 0 if nan!  
        root['corrs'][j] = corrs

    #and now we save it out
    # torch.save(outputs, f'{out_path}/{wandb_name}.pt')