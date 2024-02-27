#the key thing is that we need to make sure we specify yes to filtering the cell types when calling the dataloader

#memory issues, let's restart the kernel and try again
#it will probably be best to make the yaml the original 16 in size

import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.dataloaders.datasets.DNase_dataset import DNaseDataset
from src.tasks.decoders import SequenceDecoder
import pytorch_lightning as pl


# sys.path.append(os.environ.get("SAFARI_PATH", "."))

# from src.models.sequence.long_conv_lm import ConvLMHeadModel
from src.models.sequence.dna_embedding import DNAEmbeddingModel
# from transformers import AutoTokenizer, GPT2LMHeadModel
# from spacy.lang.en.stop_words import STOP_WORDS
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import torch.nn.functional as F

# d_output = 161

tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
ccre = DNaseDataset(max_length = 1024, split = 'test', tokenizer=tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes.yaml'
cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)

train_cfg = cfg['train']  # grab section `train` section of config
model_cfg = cfg['model_config']  # grab the `model` section of config

d_output = train_cfg['d_output'] 

backbone = DNAEmbeddingModel(**model_cfg)

decoder = SequenceDecoder(model_cfg['d_model'], d_output=d_output, l_output=0, mode='pool')

ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-12/11-21-32-769024/checkpoints/last.ckpt'
state_dict = torch.load(ckpt_path, map_location='cpu')  # has both backbone and decoder
        
# loads model from ddp by removing prexix to single if necessary
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    state_dict["state_dict"], "model."
)

model_state_dict = state_dict["state_dict"]

# need to remove torchmetrics. to remove keys, need to convert to list first
for key in list(model_state_dict.keys()):
    if "torchmetrics" in key:
        model_state_dict.pop(key)

# the state_dict keys slightly mismatch from Lightning..., so we fix it here
decoder_state_dict = {}
decoder_state_dict['output_transform.weight'] = model_state_dict.pop('decoder.0.output_transform.weight')
decoder_state_dict['output_transform.bias'] = model_state_dict.pop('decoder.0.output_transform.bias')

#now adjust the backbone
# embedding = torch.nn.Embedding(20, 128)
# backbone.backbone.embeddings.word_embeddings = embedding #again a hack

# now actually load the state dict to the decoder and backbone separately
decoder.load_state_dict(decoder_state_dict, strict=True)
backbone.load_state_dict(model_state_dict, strict=True)

decoder = decoder.to(device)
backbone = backbone.to(device)
#now dataset class
from src.dataloaders.datasets.DNase_allcelltypes import DNaseAllCellTypeDataset
DNase_all = DNaseAllCellTypeDataset(max_length = 1024, split = 'test', tokenizer=tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter=True)

#let's do 5 batches of 4096
from torch.utils.data import DataLoader
predict_list = []
target_list = []
#output shape will be N x 161
targets = torch.zeros((len(DNase_all), 161))
predicts = torch.zeros((len(DNase_all), 161))
DNase = DataLoader(DNase_all, batch_size=4096, shuffle=False) #results are identical even if you shuffle, obviously since it's just the mean
with torch.no_grad():
    idx = 0
    for i, batch in tqdm(enumerate(DNase), total = len(DNase)):
        seq, target = batch
        seq = seq.to(device)
        b_size = seq.shape[0]
        # target = target.to(device)
        y_hat, _ = backbone(seq)
        y_hat = decoder(y_hat)
        # print(y_hat)
        # print(target)
        targets[idx:b_size+idx,:] = target
        predicts[idx:b_size+idx,:] = y_hat.detach().cpu()
        idx += b_size

#now save it out
torch.save(targets, '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-12/11-21-32-769024/checkpoints/targets.pt')
torch.save(predicts, '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-12/11-21-32-769024/checkpoints/predicts.pt')