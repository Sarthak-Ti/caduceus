#we have joint model, let's see how it evaluates expression
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
# print(sys.path)
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import EnformerDecoder
from src.tasks.encoders import JointCNN
# from src.tasks.encoders import EnformerEncoder
from caduceus.configuration_caduceus import CaduceusConfig
import torch
import numpy as np
from src.dataloaders.datasets.general_dataset import GeneralDataset
import yaml
from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import argparse
import itertools
import inspect
import zarr
from numcodecs import Blosc
from scipy.stats import spearmanr, pearsonr

#set it so only device 3 is seen
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

try:
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except ValueError as e:
    if "Resolver already registered" in str(e):
            print("Resolver already exists, skipping registration.")
            
#edit evals class
class Evals():
    def __init__(self,
                 ckpt_path,
                 dataset=None,
                 split = 'test',
                 device = None,
                 load_data=False,
                 ) -> None:
        
        #now load the cfg from the checkpoint path
        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if device is not None:
            #if we are given a device, we will use that device
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        #now set up dataset
        if dataset is None:
            dataset_args = self.cfg['dataset']
            sig = inspect.signature(GeneralDataset.__init__)
            sig = {k: v for k, v in sig.parameters.items() if k != 'self'}
            to_remove = []
            for k, v in dataset_args.items():
                if k not in sig:
                    # del dataset_args[k]
                    to_remove.append(k)
            for k in to_remove:
                del dataset_args[k]
            dataset_args['split'] = split
            dataset_args['evaluating'] = True #this tells it to not do things like random shifting and rc aug, still does random masking tho, can get og sequence easily
            dataset_args['load_in'] = load_data
            # dataset_args['rc_aug'] = False #we don't want to do rc aug in our evaluation class!!!
            self.dataset_args = dataset_args
            # self.dataset_args['rc_aug'] = False #we don't want to do rc aug in our evaluation class!!!
            self.dataset = GeneralDataset(**dataset_args)
            
            # self.kmer_len = dataset_args['kmer_len']
            # self.dataset = enformer_dataset.EnformerDataset(split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
            #                                                 return_CAGE=dataset_args['return_CAGE'], cell_type=dataset_args.get('cell_type', None),
            #                                                 kmer_len=dataset_args['kmer_len']) #could use dataloader instead, but again kinda complex
        else:
            self.dataset = dataset
         
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
        for key in list(model_state_dict.keys()):
            if "decoder" in key:
                decoder_state_dict[key[10:]] = model_state_dict.pop(key)
        encoder_state_dict = {}
        for key in list(model_state_dict.keys()):
            if "encoder" in key:
                encoder_state_dict[key[10:]] = model_state_dict.pop(key)
        
        cfg['model']['config'].pop('_target_')
        # cfg['model']['config']['complement_map'] = self.dataset.tokenizer.complement_map
        caduceus_cfg = CaduceusConfig(**cfg['model']['config'])
        
        self.backbone = DNAEmbeddingModelCaduceus(config=caduceus_cfg)
        self.backbone.load_state_dict(model_state_dict, strict=True)
        
        #remove self.cfg['decoder']['_name_']
        del self.cfg['decoder']['_name_']
        self.cfg['decoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.decoder = EnformerDecoder(**self.cfg['decoder']) #could do with instantiating, but that is rather complex
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        
        del self.cfg['encoder']['_name_']
        self.cfg['encoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.encoder = JointCNN(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        
        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()
        
    def __call__(self, idx=None, data=None):
        #now evaluate the model on one example
        if data is None:
            (seq,acc),(seq_unmask,acc_unmask,exp) = self.dataset[idx]
            
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
        else:
            (x,y),(seq_unmask,acc_unmask,exp) = data

            if x.dim() == 2:
                x = x.unsqueeze(0) #add batch dim
                y = y.unsqueeze(0) #add batch dim
        
        x,y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            x1 = self.encoder(x,y)
            x1,_ = self.backbone(x1)
            x1 = self.decoder(x1)
        
        return x1, exp
    

# ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/17-59-55-471925/checkpoints/04-val_loss=-0.44593.ckpt'
# evals = Evals(ckpt_path, load_data=True)

# output_array = np.zeros((len(evals.dataset), 896))
# target_array = np.zeros((len(evals.dataset), 896))
# for i in tqdm(range(len(evals.dataset))):
#     out = evals(i)
#     output = out[0][0,:,0].cpu().numpy()
#     target = out[1][:,0]
#     output_array[i] = output
#     target_array[i] = target

# #now save them
# np.save('/data1/lesliec/sarthak/data/joint_playground/model_out/GM12878_base_predictions.npy', output_array)
# np.save('/data1/lesliec/sarthak/data/joint_playground/model_out/GM12878_base_targets.npy', target_array)

# #now let's do the next one
# ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/18-07-46-083163/checkpoints/03-val_loss=-0.48683.ckpt'
# evals = Evals(ckpt_path, load_data=True)

# output_array = np.zeros((len(evals.dataset), 896))
# target_array = np.zeros((len(evals.dataset), 896))
# for i in tqdm(range(len(evals.dataset))):
#     out = evals(i)
#     output = out[0][0,:,0].cpu().numpy()
#     target = out[1][:,0]
#     output_array[i] = output
#     target_array[i] = target
    
# #now save them
# np.save('/data1/lesliec/sarthak/data/joint_playground/model_out/GM12878_base_predictions_conv.npy', output_array)
# np.save('/data1/lesliec/sarthak/data/joint_playground/model_out/GM12878_base_targets_conv.npy', target_array)


ckpt_paths = [
    '/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/17-59-55-471925/checkpoints/07-val_loss=-0.47649.ckpt',
    '/data1/lesliec/sarthak/caduceus/outputs/2025-04-14/18-29-44-495215/checkpoints/06-val_loss=-0.45494.ckpt',
    '/data1/lesliec/sarthak/caduceus/outputs/2025-04-14/18-36-09-021037/checkpoints/13-val_loss=-0.36632.ckpt',
]

model_names = ['GM12878_base_more_train', 'GM12878_no_mlm', 'GM12878_no_finetune']

for ckpt_path, model_name in zip(ckpt_paths, model_names):
    evals = Evals(ckpt_path, load_data=True)

    output_array = np.zeros((len(evals.dataset), 896))
    target_array = np.zeros((len(evals.dataset), 896))
    for i in tqdm(range(len(evals.dataset))):
        out = evals(i)
        output = out[0][0,:,0].cpu().numpy()
        target = out[1][:,0]
        output_array[i] = output
        target_array[i] = target

    #now save them
    np.save(f'/data1/lesliec/sarthak/data/joint_playground/model_out/{model_name}_predictions.npy', output_array)
    np.save(f'/data1/lesliec/sarthak/data/joint_playground/model_out/{model_name}_targets.npy', target_array)