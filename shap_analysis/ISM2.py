#this will be doing ISM with my improved method, based on the odl script, and manually verified the results. We'll do this for the two Dnase base models
#the multitasking model can wait for now

from ism_utils import ISMUtils
import numpy as np
#TODO if name ==main, then make this thing a fucntion, so not such terrible code! Input is like the seed and all that stuff
#make sure seed is still first, and then also add a if that file exists, continue

#test it on various ccREs and compare it to the saved values
#first for the non all cell type
multitasking_path1 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-27/18-39-11-031863/checkpoints/last.ckpt' #the newly trained one 300 epoch, lower loss
multitasking_path2 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-23/09-35-33-196632/checkpoints/381-val_loss=3.57483.ckpt' #the original one we were testing
# multitasking_path2 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/3-25/15-42-38-865149/checkpoints/last.ckpt'
# ctst_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-23/09-35-11-173861/checkpoints/last.ckpt'

#let's try the ctst path
# util_cts = ISMUtils('DNase',cts_path, cfg = 'DNase_full.yaml')
# #now we can do the ism
# for i in range(56,100):
#     ism = util_cts.calculate_ISM(i, cuda = True)
#     np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_fulltrain{i}.npy', ism)

# util_cts2 = ISMUtils('DNase',cts_path2) #the old 10% 100 epoch model that also used the old embeddings
# del util_cts

util_multitasking = ISMUtils('DNase_allcelltypes', multitasking_path1, classification=True)
util_multitasking2 = ISMUtils('DNase_allcelltypes', multitasking_path2, classification=False)
train_idx = np.load('/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/most_variable_train.npy')
# total = len(util_multitasking.dataset)
rng = np.random.default_rng(seed=420)
for idx in train_idx:
    ism = util_multitasking.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/train/highvar_class_{idx}', ism)
    ism = util_multitasking2.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/train/highvar_reg_{idx}', ism)
    #now generate a random number between 1 and 800000
    while True:
        random_number = rng.integers(low=1, high=800001)
        if random_number not in train_idx:
            break
    ism = util_multitasking.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/train/lowvar_class_{random_number}', ism)
    ism = util_multitasking2.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/train/lowvar_reg_{random_number}', ism)

val_idx = np.load('/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/most_variable_val.npy')
for idx in val_idx:
    ism = util_multitasking.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/val/highvar_class_{idx}', ism)
    ism = util_multitasking2.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/val/highvar_reg_{idx}', ism)
    #now generate a random number between 1 and 800000
    while True:
        random_number = rng.integers(low=1, high=100001)
        if random_number not in val_idx:
            break
    ism = util_multitasking.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/val/lowvar_{random_number}', ism)
    ism = util_multitasking2.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/val/lowvar_reg_{random_number}', ism)



test_idx = np.load('/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/most_variable_test.npy')
for idx in test_idx:
    ism = util_multitasking.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/test/highvar_{idx}', ism)
    ism = util_multitasking2.calculate_ISM(idx,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/test/highvar_reg_{idx}', ism)
    #now generate a random number between 1 and 800000
    while True:
        random_number = rng.integers(low=1, high=100001)
        if random_number not in test_idx:
            break
    ism = util_multitasking.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/test/lowvar_{random_number}', ism)
    ism = util_multitasking2.calculate_ISM(random_number,cuda=True,progress_bar=False)
    np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/test/lowvar_reg_{random_number}', ism)