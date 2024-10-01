from evals_utils_enformer import Evals, pearsonr2
import numpy as np

# cage = Evals('/data/leslie/sarthak/caduceus/outputs/2024-09-10/16-15-22-499398/checkpoints/15-val_loss=0.52479.ckpt')
# allout = cage.evaluate(4)
# print('evaluted cage model')

split='test'
labels = np.load(f'/data/leslie/sarthak/data/enformer/data/{split}_label.npy')
labels = labels.transpose(0, 2, 1)

#now do pearson correlation
# from tqdm import tqdm
# corrs = np.zeros((allout.shape[0], allout.shape[2]))
# for i in tqdm(range(allout.shape[0])):
#     corrs[i,:] = pearsonr2(allout[i].T, labels[i].T)
#no need for that at all, all you need to do is this

# allout = np.load('/data/leslie/sarthak/data/enformer/data/model_out/enformer.npy')
# allout = allout.transpose(0, 2, 1)
# corrs = pearsonr2(allout, labels)
# np.save('/data/leslie/sarthak/data/enformer/data/model_out/mambacage_epoch15.npy_corrs.npy', corrs)
# print('saved cage model')
# np.save('/data/leslie/sarthak/data/enformer/data/model_out/mambacage.npy',allout)

#and now for the other model
gradbatch = Evals('/data/leslie/sarthak/caduceus/outputs/2024-09-27/13-42-15-772960/checkpoints/10-val_loss=0.23360.ckpt')
allout = gradbatch.evaluate(4)
print('evaluated gradbatch model')
labels = labels[:,:allout.shape[2],:] #limit to non cage data
allout = allout.transpose(0, 2, 1)
corrs = pearsonr2(allout, labels)

np.save('/data/leslie/sarthak/data/enformer/data/model_out/enformer_kmer_mamba_DNase_1_epoch35.npy_corrs.npy', corrs)