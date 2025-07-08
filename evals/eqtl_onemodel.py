# ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt'
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' #set this to the GPU you want to use, or leave it empty to use all GPUs
print('eQTL on one model', flush=True)
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
from src.dataloaders.datasets.general_dataset import GeneralDataset
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import EnformerDecoder
from src.tasks.encoders import JointCNN
from caduceus.configuration_caduceus import CaduceusConfig
import yaml
from omegaconf import OmegaConf
import os
import itertools
import inspect
import zarr
from numcodecs import Blosc
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import pickle

try:
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except ValueError as e:
    if "Resolver already registered" in str(e):
            print("Resolver already exists, skipping registration.")

class Evals():
    def __init__(self,
                 ckpt_path,
                 dataset=None,
                 split = 'test',
                 device = None,
                 load_data=False,
                 **dataset_overrides #Don't pass None into overrides unless you intentionally want it to be None! Pass in items only that you need
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
            assert dataset_args['mlm'] == 0 and dataset_args['acc_mlm'] == 0, "MLM and acc_mlm should be 0 for the training"
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
            
            for k, v in dataset_overrides.items():
                if k in sig:
                    dataset_args[k] = v
                    print(f"Overriding {k} with {v}")
                else:
                    print(f"Warning: {k} not in dataset args, skipping")
            
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
            (seq,acc),(seq_unmask,acc_unmask) = self.dataset[idx]
            
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
        else:
            (x,y),(seq_unmask,acc_unmask) = data

            if x.dim() == 2:
                x = x.unsqueeze(0) #add batch dim
                y = y.unsqueeze(0) #add batch dim
        
        x,y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            x1 = self.encoder(x,y)
            x1,_ = self.backbone(x1)
            x1 = self.decoder(x1)
        
        return x1

def main(args):
    ckpt_path = args.ckpt_path
    evals = Evals(ckpt_path,load_data=args.load_data, data_idxs=args.data_idxs, data_path=args.data_path, additional_data=None, additional_data_idxs=None) #we don't need it to load expression data, force it not to

    mapping = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'N': torch.tensor([0, 0, 0, 0], dtype=torch.float32),
    }
    
    #should just use the actual data ideally but manually loading it in is easier, doesn't work if you use the zarr file

    onehot_mapping = {0:'A',1:'C',2:'G',3:'T'}

    qtls = pd.read_csv('/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/LCLs.txt', sep=' ', header=None)
    qtls.columns = ['label', 'qtl_idx', 'gene_idx', 'chrom', 'gene_start', 'gene_end', 'strand', 'qtl_loc', 'ref', 'alt', 'sign_target']

    length = 524288

    output_array = np.zeros((qtls.shape[0],896,2))
    
    base_dir = '/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/'
    with open(base_dir+'genes.pickle', 'rb') as f:
        gene_annotation = pickle.load(f)
    ordered_genes = sorted(list(gene_annotation.keys()))
    tmpgeneTSS = np.loadtxt(base_dir+'ensemblTSS.txt', dtype='str')
    geneTSS_dic = {tmpgeneTSS[i, 0]: int(tmpgeneTSS[i, 1]) for i in range(tmpgeneTSS.shape[0])}
    

    for i in tqdm(range(qtls.shape[0])):
        temp = qtls.iloc[i]
        chrom   = 'chrX' if temp['chrom']==23 else 'chr'+str(temp['chrom'])
        pos     = temp['qtl_loc'] - 1  # Convert to zero-based index
        gene_idx= temp['gene_idx']
        tss_loc = geneTSS_dic[ordered_genes[gene_idx]]
        
        start = tss_loc - length//2
        end = tss_loc + length//2

        #let's get eqtl position
        eQTL_pos = pos - start
        if 0 <= eQTL_pos < length:
            2
        else:
            continue
        
        idx = evals.dataset.expand_seqs(chrom,start,end)
        # data = evals.dataset[idx]
        
        ((s,a),(su,au)) = evals.dataset[idx]
        # data = ((s,a), (None, None)) #we don't need theunmasked as we will just put it into the model normally
        # data = (None,None,su,au) #can be s and a or None, it isn't used by the mask function
        #now we add a batch dimension to s and a, and double the values
        s = s.unsqueeze(0).repeat(2, 1, 1) #now is 1 x 6 x 524288
        a = a.unsqueeze(0).repeat(2, 1, 1)
        #now we can edit the input sequence
        current_nuc = s[0, :, eQTL_pos].cpu().numpy()
        current_nuc = np.argmax(current_nuc)
        current_nuc = onehot_mapping[current_nuc]

        assert current_nuc == temp['ref'], f'current nuc {current_nuc} does not match ref {temp["ref"]} for {temp["label"]}'

        s[1, :4, eQTL_pos] = mapping[temp['alt']]
        data = ((s,a),(None,None))
        
        out = evals(data = data)
        
        output_array[i,:,:] = out[:,:,0].cpu().numpy().T
        

    np.save(f'/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/{args.output}', output_array)
    if args.verbose:
        print(f'saved dsQTL results to {args.output}')
        # print(f'loaded checkpoint from {args.ckpt_path}')
        print('dsQTL run complete')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run dsQTL on one model')
    parser.add_argument('--ckpt_path', type=str, default='/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt', help='Path to the checkpoint')
    parser.add_argument('-o', '--output', type=str, default='output_test.npy', help='Output file path')
    parser.add_argument('-v', '--verbose', action='store_false', help='Disable verbose output')
    parser.add_argument('--data_idxs', nargs='+',type=int,help='List of mask sizes', default=None) #intentionally passes None as a default, tells it to not find any data idxs
    parser.add_argument('--load_data', action='store_true', help='Load data from the checkpoint')
    parser.add_argument('--data_path', type=str, default='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz', help='Path to the data')
    args = parser.parse_args()
    
    print(args)
    if args.verbose:
        # print(f'args: {args}')
        print(f'running eQTL on model and saving results to {args.output}')
        print(f'loading checkpoint from {args.ckpt_path}')
    
    main(args)
    # ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/last.ckpt' #for the generalizing model