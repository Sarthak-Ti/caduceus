#now we test ISM batched across all cell types with the GPU
from shap_utils import ShapUtils
ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-09/17-38-16-568113/checkpoints/last.ckpt' #this is again the 10% 100 epochs one
util = ShapUtils('DNase', ckpt_path, percentage_background = 1/30000)
#we don't actually care about the percentage background, we do our own thing, can make a new utils, but works for now

#use data loaders
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
#wrap it all in a for loop for many cCREs and wrap that in eval
dataloader = DataLoader(util.dataset, batch_size = 161, shuffle = False)
iterator = iter(dataloader)
for ccre in tqdm(range(100)): # just do for 1000 of them
    with torch.no_grad():
        batch = next(iterator)
        # batch = next(iter(dataloader)) #such a minor bug but causing all the issues
        # print(batch)
        # continue
        #now let's get the batch
        #let's see how long it takes on the GPU
        backbone = util.backbone.cuda().eval() #cuda and eval together
        decoder = util.decoder.cuda().eval()
        a = batch[0].cuda()
        # a = a.cuda()
        b,_ = backbone(a)
        out_gt = decoder(b)
        # print(out_gt.shape) #much faster!!
        out_gt_np = out_gt.detach().cpu().numpy().reshape(1, 1, 161)
        # print(out_gt_np.shape)
        
        ISM_results = np.ones((4,1023,161))*out_gt_np
        seq = batch[0][0] #just take a single sample!
        token_list = [7,8,9,10] #tokenize this list first
        #for each one there's all 4 tokens, and 1023 positions, will find a way to aggregate across celltypes later
        for idx, nucleotide in enumerate(seq): #the 8th index is the first one we want
            if idx < 7:
                continue
            temp_token_list = token_list.copy() #we make a copy
            if nucleotide not in temp_token_list:
                continue #basically skips this and none of it is updated, because is a weird tooken
            temp_token_list.remove(nucleotide)
            temp_token_list = np.array(temp_token_list)
            results_list = []
            for j in temp_token_list:
                temp_seq = batch[0].clone() #batch was never on the gpu
                temp_seq[:,idx] = j
                a,_ = backbone(temp_seq.cuda())
                out = decoder(a)
                results_list.append(out.detach().cpu().numpy())
            ISM_results[temp_token_list-7,idx,:] = np.array(results_list).squeeze()
        #relatively slow on the CPU
        #and now we subtract the out_gt from it
        ISM_results_normalized = ISM_results - out_gt_np #is mutated - reference
        np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_{ccre}.npy', ISM_results_normalized)
        # break
#if this works, then we know that we can just run it!! 
