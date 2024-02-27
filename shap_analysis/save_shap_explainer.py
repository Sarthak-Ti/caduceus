#we will run and save out the shap values for the allcelltypes (multitasking) and normal (cell type specific) models

import shap
from shap_utils import ShapUtils
import pickle
all_ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-17/09-34-21-368888/checkpoints/last.ckpt'
ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-09/17-38-16-568113/checkpoints/last.ckpt'

util = ShapUtils('DNase', ckpt_path, percentage_background = 1/5000)
util.load_from_indices()
ccres = util.background_embed.shape[0]/util.dataset.cell_types
# print('doing explainer')
explainer = shap.DeepExplainer(util.model, util.background_embed)
with open(f'/data/leslie/sarthak/data/shap_explainer_{ccres}ccre_DNase.pkl', 'wb') as f:
    pickle.dump(explainer, f)
# print('explainer done')

del util


util = ShapUtils('DNase_allcelltypes', all_ckpt_path, percentage_background = 1/5000)
util.load_from_indices()

ccres = util.background_embed.shape[0]

explainer = shap.DeepExplainer(util.model, util.background_embed)
import pickle
with open(f'/data/leslie/sarthak/data/shap_explainer_{ccres}ccre_DNase_allcelltypes.pkl', 'wb') as f:
    pickle.dump(explainer, f)