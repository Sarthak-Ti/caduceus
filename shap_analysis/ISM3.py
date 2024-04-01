#this will be doing ISM with my improved method, based on the odl script, and manually verified the results. We'll do this for the two Dnase base models
#the multitasking model can wait for now

from ism_utils import ISMUtils
import numpy as np
import os

def ism(name, path, split, filename, classification, rng, random_number_list=None):
    utils = ISMUtils(name, path, classification=classification)
    idxs = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/most_variable_{split}.npy')
    high = len(utils.dataset.array)
    if random_number_list == None:
        random_number_list = []
    for i, idx in enumerate(idxs):
        #check if the file is already there, if so skip
        if os.path.exists(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/{split}/highvar_{filename}_{idx}.npy'):
            continue
        ism = utils.calculate_ISM(idx,cuda=True,progress_bar=False)
        np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/{split}/highvar_{filename}_{idx}', ism)
        #only do this part if random_number_list is None
        while True:
            random_number = rng.integers(low=1, high=high)
            if random_number not in idxs:
                random_number_list.append(random_number)
                break
        ism = utils.calculate_ISM(random_number,cuda=True,progress_bar=False)
        np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs/{split}/lowvar_{filename}_{random_number}', ism)
        return random_number_list

def main():
    multitasking_path1 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-27/18-39-11-031863/checkpoints/last.ckpt' #the newly trained one 300 epoch, lower loss
    multitasking_path2 = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-23/09-35-33-196632/checkpoints/381-val_loss=3.57483.ckpt' #the original one we were testing
    splits = ['train', 'val', 'test']
    rng = np.random.default_rng(seed=420)
    for split in splits:
        ism('DNase_allcelltypes', multitasking_path1, split, 'highvar_class', True, rng)
        ism('DNase_allcelltypes', multitasking_path2, split, 'highvar_reg', False, rng)


if __name__ == '__main__':
    main()