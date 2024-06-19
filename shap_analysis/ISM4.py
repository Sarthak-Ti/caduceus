#this will be doing ISM with my improved method, based on the odl script, and manually verified the results.
#smartly calculates ism values and saves it u sing efficient np.float16

from ism_utils import ISMUtils
import numpy as np
import os

def ism(name, path, split, filename, classification):
    utils = ISMUtils(name, path, classification=classification, split=split)
    idxs = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}_idx.npy') #note, we have since removed a few indicies
    #but they are still in here, the ones are 752763...
    #remove that value from idxs
    # idxs = np.delete(idxs, 752763)
    if split=='train':
        idxs = idxs[idxs!=752763]
    for i, idx in enumerate(idxs):
        #check if the file is already there, if so skip
        if os.path.exists(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}/{filename}_{idx}_reg.npy'):
            continue
        ism_vals = utils.calculate_ISM(idx,cuda=True,progress_bar=False)
        if classification:
            regout,classout = ism_vals
            classout16 = classout.astype(np.float16)
        else:
            regout = ism_vals
        regout16 = regout.astype(np.float16)
        if np.isinf(regout16).any():
            print('ERRRORRRRRRRRRRR OVERFLOWWWWWWWWWWWW')
        #now we actually save it out
        np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}/{filename}_{idx}_reg.npy', regout16)
        if classification:
            np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}/{filename}_{idx}_class.npy', classout16)
            # np.savez_compressed(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}/{filename}_{idx}.npz', reg=regout16, classout=classout16)

def main():
    multitasking_path1 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-27/18-39-11-031863/checkpoints/25-val_loss=0.52186.ckpt' #the 25 epoch one
    ctst_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-25/15-41-13-286486/checkpoints/last.ckpt'
    multitasking_path2 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-27/18-39-11-031863/checkpoints/last.ckpt' #the last epoch one
    splits = ['train', 'val', 'test']
    # rng = np.random.default_rng(seed=420)
    for split in splits:
        ism('DNase_allcelltypes', multitasking_path1, split, 'multitasking_25epoch', True)
        ism('DNase_ctst', ctst_path, split, 'ctst_bestepoch', True)
        # ism('DNase_allcelltypes', multitasking_path2, split, 'multitasking_bestepoch', True)


if __name__ == '__main__':
    main()