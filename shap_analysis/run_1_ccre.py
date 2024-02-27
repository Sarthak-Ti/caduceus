#the point of this is to train the explainer and then run it on 5 ccres for all the cell types to analyze!

#we test the code to see if it runs, then run it on more of th edata

import shap
from shap_utils import ShapUtils
import numpy as np
from tqdm import tqdm
all_ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-17/09-34-21-368888/checkpoints/last.ckpt'
ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-09/17-38-16-568113/checkpoints/last.ckpt'

util = ShapUtils('DNase', ckpt_path, percentage_background = 1/50000)
util.load_from_indices()
ccres = util.background_embed.shape[0]/util.dataset.cell_types
print('explaining')
explainer = shap.DeepExplainer(util.model, util.background_embed)
print(ccres)
print(idx:=util.test_input_indices) #the 5 indices 

#now we analyze each of the 5 and all 161 cell types
output_ccres = []
for i in idx[:1]:
    for j in tqdm(range(161)):
        a,_ = util.dataset[i*161+j]
        a_embed = util.backbone.backbone.embeddings.word_embeddings(a.unsqueeze(0))
        # shap_values = explainer.shap_values(util.test_input[i:i+1])
        #these shap values will be 
        shap_values = explainer.shap_values(a_embed)
        output_ccres.append(shap_values)
#and then we concatenate them
output_ccres = np.concatenate(output_ccres, axis = 0)
#now we can save out this numpy array
np.save(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/shap_values_{ccres}ccre_DNase.npy', output_ccres)
