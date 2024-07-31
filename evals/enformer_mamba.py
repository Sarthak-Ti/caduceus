#all of this seems good, let's combine it together again

import sys
sys.path.append('/data/leslie/sarthak/caduceus/')

import torch
ckpt_path = '/data/leslie/sarthak/caduceus/outputs/2024-07-14/21-18-35-955024/checkpoints/05-val_loss=0.70020.ckpt'
#remember this mamba model has not been pretrained!

from caduceus.configuration_caduceus import CaduceusConfig
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import EnformerDecoder
from src.dataloaders.datasets.enformer_dataset import EnformerDataset

dataset = EnformerDataset('test', 131_072, rc_aug = False, load_into_memory=True)

import yaml
cfg = '/data/leslie/sarthak/caduceus/outputs/2024-07-11/10-46-06-793653/config.json'
cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
# train_cfg = cfg['train']  # grab section `train` section of config
# model_cfg = cfg['model_config']  # grab the `model` section of config
# print(cfg)
state_dict = torch.load(ckpt_path, map_location='cpu')
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    state_dict["state_dict"], "model."
)
model_state_dict = state_dict["state_dict"]
# model_state_dict.keys()

config = CaduceusConfig(**cfg['model']['config'])
config.norm_epsilon = float(config.norm_epsilon)
# config.complement_map = complement_map
#added complement map to automatically be that default one!
backbone = DNAEmbeddingModelCaduceus(config)
cfg['decoder'].pop('_name_')
cfg['decoder']['d_output'] = dataset.d_output
decoder = EnformerDecoder(**cfg['decoder'])
#now we load in the weights
for key in list(model_state_dict.keys()):
    if "torchmetrics" in key:
        model_state_dict.pop(key)
# the state_dict keys slightly mismatch from Lightning..., so we fix it here
decoder_state_dict = {}
for key in list(model_state_dict.keys()):
    if "decoder" in key or "output_transform" in key:
        decoder_state_dict[key[10:]] = model_state_dict.pop(key)
decoder.load_state_dict(decoder_state_dict, strict=True)
backbone.load_state_dict(model_state_dict, strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = decoder.to(device).eval()
backbone = backbone.to(device).eval()


import numpy as np
def pearsonr2(x, y):
    # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
    # because they want to compute everything pairwise between rows of a
    # stacked x and y.
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = np.sum((x - xm) * (y - ym), axis=-1)/(x.shape[-1]-1)
    sx = np.std(x, ddof=1, axis=-1)
    sy = np.std(y, ddof=1, axis=-1)
    rho = cov/(sx * sy)

    return rho


from tqdm import tqdm
corrs = np.zeros((len(dataset), dataset.d_output))
for i in tqdm(range(len(dataset)), total=len(dataset)):
    x,y = dataset[i]
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        out1,_ = backbone(x)
        out2 = decoder(out1)
        out2 = out2.cpu().detach().squeeze(0).numpy()
    corrs[i,:] = pearsonr2(out2.T, y.numpy().T)

#now save it out
np.save('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_mamba_corrs.npy', corrs)