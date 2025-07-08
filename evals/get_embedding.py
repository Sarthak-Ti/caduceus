import h5py
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import JointMaskingDecoder
from src.tasks.encoders import JointCNN
from caduceus.configuration_caduceus import CaduceusConfig
import torch
import numpy as np
from src.dataloaders.datasets.general_dataset import GeneralDataset
import yaml
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import argparse
import itertools
import inspect
import zarr
import time

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
        self.encoder = JointCNN(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        
        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()
        
    def __call__(self, idx=None, data=None, softplus=True, og=False, embed=False):
        #now evaluate the model on one example
        if data is None:
            (seq,acc),(seq_unmask,acc_unmask) = self.dataset[idx]
            
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
        else:
            x,y,seq_unmask,acc_unmask = data

            if x.dim() == 2:
                x = x.unsqueeze(0) #add batch dim
                y = y.unsqueeze(0) #add batch dim
        
        x,y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            x1 = self.encoder(x,y)
            x1,_ = self.backbone(x1)
            if embed:
                return x1, y
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


def main(args):
    start_time = time.time()
    
    data_file = h5py.File('/data1/lesliec/sarthak/data/joint_playground/koo_benchmark/gLM_finetune_weights/data/Processed_lentiMPRA_K562.h5')
    train_seq = data_file['x_train']
    # y_train = data_file['y_train'][:]
    valid_seq = data_file['x_valid']
    # y_valid = data_file['y_valid'][:]
    test_seq = data_file['x_test']
    # y_test = np.squeeze(data_file['y_test'][:])

    ckpt_path = args.ckpt_path
    evals = Evals(ckpt_path, device=0)

    N = 524288
    k = 230

    start = (N - k) // 2
    end   = start + k

    # total_num = y_train.shape[0] + y_valid.shape[0] + y_test.shape[0]
    
    zarr_open = zarr.open(args.zarr_path, mode='r+')
    total_num = zarr_open.shape[0]
    full_output = zarr_open[args.start:args.end].sum((1,2)) #tells us if there's anything

    full_seqs = torch.zeros((total_num, 230, 4), dtype=torch.float32)
    full_seqs[:train_seq.shape[0],:,:] = torch.tensor(np.array(train_seq)) #was giving me this warning for some reason
    full_seqs[train_seq.shape[0]:train_seq.shape[0]+valid_seq.shape[0],:,:] = torch.tensor(np.array(valid_seq))
    full_seqs[train_seq.shape[0]+valid_seq.shape[0]:,:] = torch.tensor(np.array(test_seq))

    # full_seqs = full_seqs[args.start:args.end] #subset to what we need #don't do this, doing based on global index!
    full_seqs = full_seqs.transpose(2,1).to(evals.device) #shape (total_num, 4, 230)

    batch_size = args.batch_size
    for i in tqdm(range(args.start, args.end, batch_size)):
        
        #get the batch
        if i + batch_size > args.end:
            batch_size = args.end - i
            print('batch size changed to', batch_size)

        if full_output[i-args.start:i+batch_size-args.start].sum() != 0:
            print('skipping', i)
            continue
        
        full_seq = torch.zeros((batch_size, 6, 524288), dtype=torch.float32).to(evals.device)
        full_seq[:, 4, :] = 1 #by default set all to N
        full_seq[:, :4, start:end] = full_seqs[i:i+batch_size] #assign seq
        full_seq[:, 4, start:end] = 0 #Get rid of the N

        full_acc = torch.zeros((batch_size, 2, 524288), dtype=torch.float32).to(evals.device)
        full_acc[:, 1, :] = 1 #by default set all to mask token and input no accessibility

        #now let's input into the model
        full_input = (full_seq, full_acc, None, None)
        out = evals(data=full_input, embed=True)
        embed = out[0][:,start:end,:]
        
        #save the output
        zarr_open[i:i+batch_size] = embed.cpu().numpy()

        #get the time
        elapsed_time = (time.time() - start_time) / 3600
        if elapsed_time > args.total_time - 0.1: #if within an hour of total time, break
            print(f"Elapsed time: {elapsed_time} hours, breaking")
            break
    
    print(f"Elapsed time: {(time.time() - start_time) / 3600} hours, index: {i+4} of {args.end}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get embeddings for the lentiMPRA K562 dataset')
    parser.add_argument('--ckpt_path', type=str, default='/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/13-44-58-301569/checkpoints/last.ckpt', help='Path to the checkpoint file')
    parser.add_argument('--total_time', type=int, default=2, help='Total time to run the script')
    parser.add_argument('--zarr_path', type=str, default='/data1/lesliec/sarthak/data/joint_playground/koo_benchmark/embeddings_lentiMPRA_K562.zarr', help='Path to the zarr file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the model')
    parser.add_argument('--start', type=int, default=0, help='Start index for the model')
    parser.add_argument('--end', type=int, default=393328, help='End index for the model')
    parser.add_argument('--load_in', action='store_true', help='Load in the data')
    
    args = parser.parse_args()
    args.end = min(args.end, 393328)
    
    
    
    print(args)
    main(args)