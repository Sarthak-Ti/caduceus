# ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt'
print('dsQTL on one model', flush=True)
#note that this is a legacy version of the dsQTL code that uses cell type token but the old approach which I am not using anymore
# import sys
# sys.path.append('/data1/lesliec/sarthak/caduceus/')
# from evals_utils_joint import Evals
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
# print(sys.path)
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import JointMaskingDecoder
from src.tasks.encoders import JointCNNWithCTT
# from src.tasks.encoders import EnformerEncoder
from caduceus.configuration_caduceus import CaduceusConfig
from src.dataloaders.datasets.general_dataset import GeneralDataset
import yaml
from omegaconf import OmegaConf
import os
# import seaborn as sns
import itertools
import inspect
import zarr
from numcodecs import Blosc
from scipy.stats import spearmanr, pearsonr

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
                #  data_idxs=None, #the actual value in the full self.dataset.data (so for GM12878 it's 12 or 69). Lets you access a new celltype or just subset to a smaller set of celltypes
                #  sequences_bed_file=None,
                 ) -> None:
        #TODO make it so that we can take in arbitrary dataset information in like a dict and adds options to dataset
        
        #now load the cfg from the checkpoint path
        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        
        if self.cfg['train'].get('custom_metric', None) == 'ce_loss_mask_acc': #makes sure we never do softplus in the loss if it's a categorical model!
            self.skip_softplus=True
        else:
            self.skip_softplus=False
        
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
            
            for k, v in dataset_overrides.items():
                if k in sig:
                    dataset_args[k] = v
                    print(f"Overriding {k} with {v}")
                else:
                    print(f"Warning: {k} not in dataset args, skipping")
            
            # if data_idxs is not None:
            #     dataset_args['data_idxs'] = data_idxs
            # if sequences_bed_file is not None:
            #     dataset_args['sequences_bed_file'] = sequences_bed_file
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
        self.decoder = JointMaskingDecoder(**self.cfg['decoder']) #could do with instantiating, but that is rather complex
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        
        del self.cfg['encoder']['_name_']
        self.cfg['encoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.cfg['encoder']['celltypes'] = 7 #hardcoding for now because it becomes 1 when we use the npz file
        self.encoder = JointCNNWithCTT(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        
        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()
        
    def __call__(self, idx=None, data=None, softplus=True, og=False):
        #now evaluate the model on one example
        if data is None:
            (seq,acc,ctt),(seq_unmask,acc_unmask) = self.dataset[idx]
            
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
            ctt = ctt.unsqueeze(0)
        else:
            x,y,ctt,seq_unmask,acc_unmask = data

            if x.dim() == 2:
                x = x.unsqueeze(0) #add batch dim
                y = y.unsqueeze(0) #add batch dim
                ctt = ctt.unsqueeze(0)
        
        x,y = x.to(self.device), y.to(self.device)
        ctt = ctt.to(self.device)
        
        with torch.no_grad():
            # print(x.shape, y.shape, ctt.shape)
            # print(ctt) #is indeed 6
            x1 = self.encoder(x,y,ctt)
            # print(torch.allclose(x1[0,:,0], self.encoder.ctt_embedding(ctt))) #is indeed true!!
            x1,_ = self.backbone(x1)
            x1 = self.decoder(x1)
            seq,acc = x1

            if softplus and not self.skip_softplus:
                acc = torch.nn.functional.softplus(x1[1])
        
        return seq, acc, seq_unmask, acc_unmask
    
    def freeze(self):
        '''freezes the model, so that it doesn't update the weights during training'''
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def mask(self, start=None, stop=None, idx=None, data=None, run=True,
             mask_accessibility=False, mask_sequence=False,
             randomize_sequence=False, randomize_accessibility=False,
             start_acc=None, stop_acc=None, ctt_val=None
             ):
        '''given an index for the dataset, or just the data, it can mask the data that you want, then runs it through the model, lots of options like to mask accessibility or sequence etc.
        If you don't mask anything, then it's like seeing how confident the model is. But it can mask out regions and see that
        
        Args:
            start (int): the start index to mask, if None requires you don't randomize or mask anything
            stop (int): the end index to mask, if None requires you don't randomize or mask anything
            idx (int, optional): the index of the data in the dataset to use. If None, data must be provided. Defaults to None.
            data (tuple, optional): the data to use, in the form of (x,y,seq_unmask,acc_unmask). If None, idx must be provided. Defaults to None.
            run (bool, optional): whether to run the model on the masked data. If False, it will just return the masked data. Defaults to True.
            mask_accessibility (bool, optional): whether to mask the accessibility values. Defaults to False.
            mask_sequence (bool, optional): whether to mask the sequence values. Defaults to False.
            randomize_sequence (bool, optional): whether to randomize the sequence values instead of masking them. Defaults to False.
            randomize_accessibility (bool, optional): whether to randomize the accessibility values instead of masking them. Defaults to False.
            start_acc (int, optional): the start index to randomize the accessibility values. If None, defaults to start
            stop_acc (int, optional): the end index to randomize the accessibility values. If None, defaults to stop
            
        Returns:
            seq_out (torch.Tensor): the output sequence after running the model, if run is True.
            acc_out (torch.Tensor): the output accessibility after running the model, if run is True.
            seq_unmask_out (torch.Tensor): the original unmasked sequence, if run is True.
            acc_unmask_out (torch.Tensor): the original unmasked accessibility, if run is True.
            x (torch.Tensor): the input sequence after masking, if run is True.
            y (torch.Tensor): the input accessibility after masking, if run is True.
        '''
        if data is not None:
            (x,y,seq_unmask,acc_unmask) = data
        elif idx is not None:
            (x,y),(seq_unmask,acc_unmask) = self.dataset[idx]
        else:
            raise ValueError("Must provide either idx or data")
        
        #now mask the data
        seq = seq_unmask.clone().transpose(1,0) #now is N x length
        acc = acc_unmask.clone().transpose(1,0)
        acc[-1] = 0
        seq[-1] = 0 #zero out the mask so it runs unmasked
        
        if start_acc is None:
            start_acc = start
            stop_acc = stop
        
        if randomize_accessibility:
            random_start = np.random.randint(0, acc.shape[1] - (stop_acc-start_acc)-1) #doesn't include the end point
            acc[0,start:stop] = acc[0,random_start:random_start+(stop_acc-start_acc)]
        
        if mask_accessibility:
            acc[-1, start_acc:stop_acc] = 1 #tells the model it's masked
            acc[0, start_acc:stop_acc] = 0 #zero out the original accessibility values
            
        if randomize_sequence:
            random_indices = torch.randint(0, 4, size=(stop-start,))
            seq[:4,start:stop] = torch.nn.functional.one_hot(random_indices, num_classes=4).transpose(1,0)
            
        if mask_sequence:
            seq[-1,start:stop] = 1
            seq[:-1,start:stop] = 0 #zero out the original sequence values
        
        #now we manually define ctt
        ctt = torch.tensor([ctt_val], dtype=torch.long)
        
        if run:
            x = seq.unsqueeze(0).to(self.device)
            y = acc.unsqueeze(0).to(self.device)
            # ctt = ctt.unsqueeze(0).to(self.device)
            data = (x,y,ctt,seq_unmask,acc_unmask)
            seq_out, acc_out, seq_unmask_out, acc_unmask_out = self(data=data)
        
            return seq_out, acc_out, seq_unmask_out, acc_unmask_out, x, y #it returns the values that were used as inputs, so yyou can see the modified inputs as well as th eoriginal data
        
        else:
            #just return the masked data
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
            return None, None, seq_unmask, acc_unmask, x, y

def main(args):
    ckpt_path = args.ckpt_path
    evals = Evals(ckpt_path,load_data=args.load_data, data_idxs=args.data_idxs, data_path=args.data_path)

    mapping = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'N': torch.tensor([0, 0, 0, 0], dtype=torch.float32),
    }

    onehot_mapping = {0:'A',1:'C',2:'G',3:'T'}

    qtls = pd.read_csv('/data1/lesliec/sarthak/data/joint_playground/dsQTL/filtered_dsqtls.txt', sep='\t')

    bims = {}
    for i in range(1,23):
        bim_path = f'/data1/deyk/extras/1000G_BIMS_hg38/1000G.EUR.QC.{i}.bim'
        bim = pd.read_csv(bim_path, sep='\t', header=None)
        bims[i] = bim


    bim_pos = {
        chrom: dict(zip(bims[chrom][1].values, bims[chrom][3].index))
        for chrom in bims
    }

    length = 524288
    si = 524288//2 - int(args.mask_size/2)
    se = 524288//2 + int(args.mask_size/2)

    output_array = np.zeros((qtls.shape[0],args.out_size//args.pool,2))

    for i in tqdm(range(qtls.shape[0])):
        rsid = qtls.iloc[i]['SNPname2']
        chrom = qtls.iloc[i]['chrom_hg19']
        bimrow = bim_pos[int(chrom[3:])][rsid]
        bimval = bims[int(chrom[3:])].iloc[bimrow]
        pos = bimval[3]-1
        start = pos - length//2
        end = pos + length//2
        
        idx = evals.dataset.expand_seqs(chrom,start,end)
        # data = evals.dataset[idx]
        
        ((s,a,ctt),(su,au)) = evals.dataset[idx]
        data = (None,None,su,au) #can be s and a or None, it isn't used by the mask function. We are not putting ctt in because we want to override it!
        out = evals.mask(si,se, data=data, mask_accessibility=True, ctt_val=args.ctt_val)
        
        current_nuc = out[2][524288//2].cpu().numpy()
        current_nuc = np.argmax(current_nuc)
        current_nuc = onehot_mapping[current_nuc]
        
        if current_nuc == bimval[4]:
            alt_key = 5
        elif current_nuc == bimval[5]:
            alt_key = 4
        else:
            raise ValueError("Neither of the alleles match the current nucleotide")
        
        data[2][524288//2,:4] = mapping[bimval[alt_key]]
        
        out2 = evals.mask(si,se, data=data, mask_accessibility=True, ctt_val=args.ctt_val)
        
        oi = 524288//2 - args.out_size//2
        oe = 524288//2 + args.out_size//2
        
        pred1 = out[1][0, oi:oe, 0].cpu().numpy()
        pred2 = out2[1][0, oi:oe, 0].cpu().numpy()
        
        if args.pool > 1:
            pred1 = np.mean(pred1.reshape(-1, args.pool), axis=1)
            pred2 = np.mean(pred2.reshape(-1, args.pool), axis=1)
        
        output_array[i,:,0] = pred1
        output_array[i,:,1] = pred2
        

    np.save(f'/data1/lesliec/sarthak/data/joint_playground/dsQTL/{args.output}', output_array)
    if args.verbose:
        print(f'saved dsQTL results to {args.output}')
        # print(f'loaded checkpoint from {args.ckpt_path}')
        print('dsQTL run complete')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run dsQTL on one model')
    parser.add_argument('--ckpt_path', type=str, default='/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/08-val_loss=0.00000.ckpt', help='Path to the checkpoint')
    parser.add_argument('-o', '--output', type=str, default='/data1/lesliec/sarthak/data/joint_playground/dsQTL/basic_dsqtl.npy', help='Output file path')
    parser.add_argument('--mask_size', type=int, default=500, help='Size of the mask')
    parser.add_argument('-v', '--verbose', action='store_false', help='Disable verbose output')
    parser.add_argument('--data_idxs', nargs='+',type=int,help='List of indices to be used to grab data', default=None) #intentionally passes None as a default, tells it to not find any data idxs
    parser.add_argument('--load_data', action='store_true', help='Load data from the checkpoint')
    parser.add_argument('--pool', type=int, default=1, help='Amount of bins to pool')
    parser.add_argument('--out_size', type=int, default=500, help='Output size')
    parser.add_argument('--data_path', type=str, default='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz', help='Path to the data')
    parser.add_argument('--ctt_val', type=int, help='Cell type token value to use for masking')
    args = parser.parse_args()
    
    print(args)
    if args.verbose:
        # print(f'args: {args}')
        print(f'running dsQTL on model and saving results to {args.output}')
        print(f'loading checkpoint from {args.ckpt_path}')
        print(f'mask size: {args.mask_size}')
    
    main(args)
    # ckpt_path = '/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/last.ckpt' #for the generalizing model