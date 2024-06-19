#this is the script to evaluate the model on the DNase data
ckpt_path = '/lila/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-07/22-40-54-392625/checkpoints/val/mse.ckpt'

#memory issues, let's restart the kernel and try again
#it will probably be best to make the yaml the original 16 in size

import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 
os.chdir('/data/leslie/sarthak/hyena/hyena-dna/')
sys.path.append(os.getcwd())
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

d_output = 1

tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
ccre = DNaseDataset(max_length = 1024, split = 'test', tokenizer=tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase.yaml'
cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)

train_cfg = cfg['train']  # grab section `train` section of config
model_cfg = cfg['model_config']  # grab the `model` section of config

d_output = train_cfg['d_output'] 

backbone = DNAEmbeddingModel(**model_cfg)

decoder = SequenceDecoder(model_cfg['d_model'], d_output=d_output, l_output=0, mode='pool')

# ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-07/09-47-18-698056/checkpoints/val/mse.ckpt'
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
embedding = torch.nn.Embedding(20, 128)
backbone.backbone.embeddings.word_embeddings = embedding #again a hack

# now actually load the state dict to the decoder and backbone separately
decoder.load_state_dict(decoder_state_dict, strict=True)
backbone.load_state_dict(model_state_dict, strict=True)

decoder = decoder.to(device)
backbone = backbone.to(device)
#find the amount of free memory
# torch.cuda.memory_allocated()
#let's do 5 batches of 4096
from torch.utils.data import DataLoader
predict_list = []
target_list = []
# decoder = decoder.to(device)
# backbone = backbone.to(device)
ccre_loader = DataLoader(ccre, batch_size=2048, shuffle=False) #results are identical even if you shuffle, obviously since it's just the mean
#make a numpy array of the predictions and targets
predict_array = torch.zeros((len(ccre), 1))
target_array = torch.zeros((len(ccre), 1))
start_idx = 0
with torch.no_grad():
    for i, batch in tqdm(enumerate(ccre_loader), total = len(ccre_loader)):
        # b_size = batch[0].shape[0]
        seq, target = batch
        b_size = seq.shape[0]
        seq = seq.to(device)
        target = target.to(device)
        y_hat, _ = backbone(seq)
        y_hat = decoder(y_hat)
        # print(y_hat)
        # print(target)
        # predict_list.extend(y_hat.detach().cpu().numpy())
        # target_list.extend(target.detach().cpu().numpy())
        # if i == 5:
        #     break
        
        #add to numpy array
        predict_array[start_idx:start_idx+b_size] = y_hat.detach().cpu()
        target_array[start_idx:start_idx+b_size] = target.detach().cpu()
        
        start_idx += b_size
        
        if i == 100:
            break
        

#save the torch tensors
torch.save(predict_array, '/data/leslie/sarthak/data/predict_array.pt')
torch.save(target_array, '/data/leslie/sarthak/data/target_array.pt')

#and calculate correlation
from sklearn.metrics import r2_score
r2_score(target, predict)
a = r2_score(target_array[:start_idx], predict_array[:start_idx])
print(a)
print('done!')