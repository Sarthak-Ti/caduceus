#this will be doing ISM with my improved method, based on the odl script, and manually verified the results. We'll do this for the two Dnase base models
#the multitasking model can wait for now

from ism_utils import ISMUtils
import numpy as np

#test it on various ccREs and compare it to the saved values
#first for the non all cell type
multitasking_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-23/09-35-33-196632/checkpoints/381-val_loss=3.57483.ckpt'
cts_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-29/15-45-02-282170/checkpoints/last.ckpt'
cts_path2 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-09/17-38-16-568113/checkpoints/last.ckpt' #this is again the 10% 100 epochs one
ctst_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-23/09-35-11-173861/checkpoints/last.ckpt'

#let's try the ctst path
util_cts = ISMUtils('DNase',cts_path, cfg = 'DNase_full.yaml')
#now we can do the ism
for i in range(56,100):
    ism = util_cts.calculate_ISM(i, cuda = True)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_fulltrain{i}.npy', ism)

# util_cts2 = ISMUtils('DNase',cts_path2) #the old 10% 100 epoch model that also used the old embeddings
del util_cts

util_ctst = ISMUtils('DNase_ctst',ctst_path)

for i in range(100):
    ism = util_ctst.calculate_ISM(i, cuda = True)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_ctst{i}.npy', ism)