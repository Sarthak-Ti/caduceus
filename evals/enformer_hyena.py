import sys
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
import src.dataloaders.datasets.enformer_dataset as enformer_dataset

#these are the only things we modify
dataset = enformer_dataset.EnformerDataset('test', 160_000, rc_aug = False, load_into_memory=True)
#now do for k562
ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-07-15/20-32-21-218315/checkpoints/10-val_loss=0.65673.ckpt'
convolutions = False
# dataset = enformer_dataset.EnformerDataset('test', 160_000, rc_aug = False, load_into_memory=False)

#now we can load in the model and evaluate the data
import yaml
cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/enformer.yaml'
cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
train_cfg = cfg['train']  # grab section `train` section of config
model_cfg = cfg['model_config']  # grab the `model` section of config
# d_output = train_cfg['d_output']  #TODO make it so that we just adjust this with self.classificaiton, no need for evals
# print(d_output, model_cfg, train_cfg)
d_output = dataset.d_output
from src.models.sequence.dna_embedding import DNAEmbeddingModel
from src.tasks.decoders import EnformerDecoder
import torch
backbone = DNAEmbeddingModel(**model_cfg)
decoder = EnformerDecoder(model_cfg['d_model'], l_output=0, mode='pool', d_output=d_output, convolutions=convolutions)
state_dict = torch.load(ckpt_path, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    state_dict["state_dict"], "model."
)
model_state_dict = state_dict["state_dict"]
model_state_dict.keys()

# need to remove torchmetrics. to remove keys, need to convert to list first
for key in list(model_state_dict.keys()):
    if "torchmetrics" in key:
        model_state_dict.pop(key)
# the state_dict keys slightly mismatch from Lightning..., so we fix it here
decoder_state_dict = {}
for key in list(model_state_dict.keys()):
    if "decoder" in key or "output_transform" in key:
        decoder_state_dict[key[10:]] = model_state_dict.pop(key)
# decoder_state_dict['output_transform.weight'] = model_state_dict.pop('decoder.0.output_transform.weight')
# decoder_state_dict['output_transform.bias'] = model_state_dict.pop('decoder.0.output_transform.bias')
# decoder_state_dict['output_transform_profile.weight'] = model_state_dict.pop('decoder.0.output_transform_profile.weight')
# decoder_state_dict['output_transform_profile.bias'] = model_state_dict.pop('decoder.0.output_transform_profile.bias')
decoder.load_state_dict(decoder_state_dict, strict=True)
backbone.load_state_dict(model_state_dict, strict=True)
decoder = decoder.to(device).eval()
backbone = backbone.to(device).eval()
#we will now make a script that goes through and finds how correlated it is
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
    corrs[i,:] = pearsonr2(np.exp(out2).T, y.numpy().T)

#now save it out
np.save('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_hyena_corrs_predictall_exp.npy', corrs)