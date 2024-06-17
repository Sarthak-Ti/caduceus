#this file contains an evaluation helper class which will contain the data loader and the model, and enables quick evaluation of the model on the test set

import torch
# import argparse
# import os
import sys
import yaml 
from tqdm import tqdm
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.tasks.decoders import SequenceDecoder
import pytorch_lightning as pl
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.models.sequence.dna_embedding import DNAEmbeddingModel
from torch.utils.data import DataLoader
from src.dataloaders.datasets.ccre_dataset import CcreDataset
from src.models.sequence.long_conv_lm import ConvLMHeadModel

class Evals():
    def __init__(self, model_type, ckpt_path, filter=True, cfg = None, classification=True, split='test', single_cell_type=None, bias=None):
        #model type is like ccre, DNase, DNase_ctst etc.
        #ckpt_path is the path to the model checkpoint
        #filter is a boolean which is true if you want to filter the dataset to only include the cell types present in the training set
        #cfg is the path to the config file, if it's not provided, it will be inferred from the model type, but if you need a specific config, this is good to provide
        #classification is a boolean which is true if you want to evaluate a model that has a classification output
        type_list = ['ccre', 'DNase_ctst', 'DNase_allcelltypes', 'DNase', 'Profile']
        if model_type not in type_list:
            raise ValueError('Model type not recognized')
        self.model_type = model_type   
        self.split = split
        self.single_cell_type = single_cell_type
        self.filter = filter
        self.classification = classification
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bias = bias
        self.cfg = cfg #this is for if we want to define a separate one
        self.tokenizer = self.setup_tokenizer()
        self.setup_model()
        self.dataset = self.setup_dataset()
        
        
    def setup_tokenizer(self):
        model_type = self.model_type
        acgtn_list = ['ccre', 'DNase_ctst', 'DNase_allcelltypes', 'Profile']
        extra_list = ['DNase']
        if model_type in acgtn_list:
            tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            return tokenizer
        else:
            tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            return tokenizer
        
    def setup_model(self):
        model_type = self.model_type
        if model_type == 'ccre':
            cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/ccre.yaml'
            self.model = HG38Encoder(cfg, self.ckpt_path, 1024).eval()
        elif model_type == 'DNase_ctst':
            if self.classification:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_ctst_classification.yaml'
            else:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_ctst.yaml'
            self.regression_head(cfg, adjust_embedding=False) #new 171 dimensional vocab and lm head size
        elif model_type == 'DNase_allcelltypes':
            if self.classification:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes_classification.yaml'
            else:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes.yaml'
            self.regression_head(cfg, adjust_embedding=False) #original 16 dimensional vocab and lm head size
        elif model_type == 'DNase':
            if self.classification:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_classification.yaml'
            else:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase.yaml'
            self.regression_head(cfg, adjust_embedding=True) #20 dimensional input but 16 output lm size
        elif model_type == 'Profile':
            cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/profile.yaml'
            self.model = HG38Encoder(cfg, self.ckpt_path, 1024).eval()
    
    def setup_dataset(self):
        model_type = self.model_type
        if model_type == 'ccre':
            dataset = CcreDataset(max_length = 1024, split = self.split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True')
            return dataset
        elif model_type == 'DNase':
            from src.dataloaders.datasets.DNase_dataset import DNaseDataset
            dataset = DNaseDataset(max_length = 1024, split = self.split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter = self.filter, classification=self.classification)
            return dataset
        elif model_type == 'DNase_ctst':
            from src.dataloaders.datasets.DNase_ctst_dataset import DNaseCtstDataset
            dataset = DNaseCtstDataset(max_length = 1024, split = self.split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter = self.filter, classification=self.classification, single_cell_type=self.single_cell_type)
            return dataset
        elif model_type == 'DNase_allcelltypes':
            from src.dataloaders.datasets.DNase_allcelltypes import DNaseAllCellTypeDataset
            dataset = DNaseAllCellTypeDataset(max_length = 1024, split = self.split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter = self.filter, classification=self.classification)
            return dataset
        
    def regression_head(self, cfg, adjust_embedding=False):
        if self.cfg is None:
            cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
        else:
            cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/' + self.cfg
            cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
        train_cfg = cfg['train']  # grab section `train` section of config
        model_cfg = cfg['model_config']  # grab the `model` section of config
        d_output = train_cfg['d_output']  #TODO make it so that we just adjust this with self.classificaiton, no need for evals
        backbone = DNAEmbeddingModel(**model_cfg)
        decoder = SequenceDecoder(model_cfg['d_model'], d_output=d_output, l_output=0, mode='pool')
        state_dict = torch.load(self.ckpt_path, map_location='cpu')  # has both backbone and decoder
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
        if adjust_embedding:
            embedding = torch.nn.Embedding(20, 128)
            backbone.backbone.embeddings.word_embeddings = embedding
        # now actually load the state dict to the decoder and backbone separately
        decoder.load_state_dict(decoder_state_dict, strict=True)
        backbone.load_state_dict(model_state_dict, strict=True)
        self.decoder = decoder.to(self.device).eval()
        self.backbone = backbone.to(self.device).eval()
#let's work on making that evaluation utils
    def evaluate(self, batch_size=4096, num_workers=4, stop = None):
        if self.model_type == 'ccre':
            targets = torch.zeros((len(self.dataset), 1023))
            predicts = torch.zeros((len(self.dataset), 1023))
            ccre = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #results are identical even if you shuffle, obviously since it's just the mean
            with torch.no_grad():
                idx = 0
                for i, batch in tqdm(enumerate(ccre), total = len(ccre)):
                    seq, target = batch
                    seq = seq.unsqueeze(0)
                    seq = seq.to(self.device)
                    b_size = seq.shape[0]
                    y_hat = self.model.encode(seq)
                    y_hat = y_hat[0].logits[0,:,:]
                    targets[idx:b_size+idx,:] = target
                    predicts[idx:b_size+idx,:] = y_hat.detach().cpu()
                    idx += b_size
                    if i == stop:
                        break
        elif self.model_type == 'DNase' or self.model_type == 'DNase_ctst': #this does the flatten then reshape
            targets_flat = torch.zeros(len(self.dataset)) #because output is just a single value, it's so obvious it's kind of stupid
            predicts_flat = torch.zeros(len(self.dataset))
            targets_class_flat = torch.zeros(len(self.dataset))
            predicts_class_flat = torch.zeros(len(self.dataset))
            DNase = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #results are identical even if you shuffle, obviously since it's just the mean
            with torch.no_grad():
                idx = 0
                for i, batch in tqdm(enumerate(DNase), total = len(DNase)):
                    seq, target = batch
                    seq = seq.to(self.device)
                    b_size = seq.shape[0]
                    y_hat, _ = self.backbone(seq)
                    y_hat = self.decoder(y_hat)
                    if self.classification:
                        targets_flat[idx:b_size+idx] = target[1].squeeze()
                        targets_class_flat[idx:b_size+idx] = target[0].squeeze()
                        predicts_flat[idx:b_size+idx] = y_hat[:,1].detach().cpu().squeeze() #note the .squeeze not necessary
                        predicts_class_flat[idx:b_size+idx] = y_hat[:,0].detach().cpu().squeeze()
                    else:
                        targets_flat[idx:b_size+idx] = target.squeeze()
                        predicts_flat[idx:b_size+idx] = y_hat.detach().cpu().squeeze()
                    idx += b_size
                    if i == stop:
                        break
                    # if i == 5:
                    #     break
            if self.single_cell_type is not None:
                targets = targets_flat
                predicts = predicts_flat
                targets2 = targets_class_flat
                predicts2 = predicts_class_flat
            else:
                targets = targets_flat.reshape(-1, self.dataset.cell_types)
                predicts = predicts_flat.reshape(-1, self.dataset.cell_types)
                targets2 = targets_class_flat.reshape(-1, self.dataset.cell_types)
                predicts2 = predicts_class_flat.reshape(-1, self.dataset.cell_types)
            if self.classification:
                targets = (targets2, targets)
                predicts = (predicts2, predicts)
            return targets, predicts
        elif self.model_type == 'DNase_allcelltypes':
            targets = torch.zeros((len(self.dataset), 161))
            predicts = torch.zeros((len(self.dataset), 161))
            targets_class = torch.zeros((len(self.dataset), 161))
            predicts_class = torch.zeros((len(self.dataset), 161))
            DNase = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #results are identical even if you shuffle, obviously since it's just the mean
            with torch.no_grad():
                idx = 0
                for i, batch in tqdm(enumerate(DNase), total = len(DNase)):
                    seq, target = batch
                    seq = seq.to(self.device)
                    b_size = seq.shape[0]
                    # target = target.to(device)
                    y_hat, _ = self.backbone(seq)
                    y_hat = self.decoder(y_hat)
                    # print(y_hat)
                    # print(target)
                    if self.classification:
                        targets_class[idx:b_size+idx,:] = target[0]
                        predicts_class[idx:b_size+idx,:] = y_hat[:,:161].detach().cpu()
                        targets[idx:b_size+idx,:] = target[1]
                        predicts[idx:b_size+idx,:] = y_hat[:,161:].detach().cpu()
                    else:
                        targets[idx:b_size+idx,:] = target
                        predicts[idx:b_size+idx,:] = y_hat.detach().cpu()
                    idx += b_size
                    if i == stop:
                        break
            if self.classification:
                targets = (targets_class, targets)
                predicts = (predicts_class, predicts)
            return targets, predicts
        
class HG38Encoder:
    "Encoder inference for HG38 sequences"
    def __init__(self, model_cfg, ckpt_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.model, self.tokenizer = self.load_model(model_cfg, ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def encode(self, seqs):
        results = []
        logits = self.model(seqs)[0]

        # Using head, so just have logits
        results.append(logits)
        # outputs.append(output)
        return results
        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = ConvLMHeadModel(**config['model_config'])
        
        state_dict = torch.load(ckpt_path, map_location='cuda:0')

        # loads model from ddp by removing prexix to single if necessary
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )

        model_state_dict = state_dict["state_dict"]

        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)

        model.load_state_dict(state_dict["state_dict"])

        return model