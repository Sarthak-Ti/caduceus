#this file contains an evaluation helper class which will contain the data loader and the model, and enables quick evaluation of the model on the test set
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
# print(sys.path)
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import EnformerDecoder
# from src.tasks.encoders import EnformerEncoder
from caduceus.configuration_caduceus import CaduceusConfig
import torch
import numpy as np
from src.dataloaders.datasets.enformer_dataset import EnformerDataset
from src.dataloaders.datasets.GPNMSA_dataset import GPNMSADataset
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

try:
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except ValueError as e:
    if "Resolver already registered" in str(e):
            print("Resolver already exists, skipping registration.")
    

class IdentityNet(torch.nn.Module):
    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        return x, None

class Evals():
    def __init__(self,
                 ckpt_path,
                 model_type = 'Enformer',
                 dataset=None,
                 split = 'test',
                 dataset_class = 'Enformer'
                 ) -> None:
        
        #first define which dataset we will use
        if dataset_class == 'Enformer':
            self.DatasetClass = EnformerDataset
        elif dataset_class == 'GPNMSA':
            self.DatasetClass = GPNMSADataset
        else:
            raise ValueError('Dataset class not recognized')
        
        #now load the cfg from the checkpoint path
        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.split = split

        #now set up dataset
        if dataset is None:
            dataset_args = self.cfg['dataset']
            sig = inspect.signature(self.DatasetClass.__init__)
            sig = {k: v for k, v in sig.parameters.items() if k != 'self'}
            to_remove = []
            for k, v in dataset_args.items():
                if k not in sig:
                    # del dataset_args[k]
                    to_remove.append(k)
            for k in to_remove:
                del dataset_args[k]
            dataset_args['split'] = split
            dataset_args['rc_aug'] = False #we don't want to do rc aug in our evaluation class!!!
            self.dataset_args = dataset_args
            # self.dataset_args['rc_aug'] = False #we don't want to do rc aug in our evaluation class!!!
            self.dataset = self.DatasetClass(**dataset_args)
            
            # self.kmer_len = dataset_args['kmer_len']
            # self.dataset = enformer_dataset.EnformerDataset(split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
            #                                                 return_CAGE=dataset_args['return_CAGE'], cell_type=dataset_args.get('cell_type', None),
            #                                                 kmer_len=dataset_args['kmer_len']) #could use dataloader instead, but again kinda complex
        else:
            self.dataset = dataset
         
        #check if self.dataset.d_output exists
        if hasattr(self.dataset, 'd_output'):
            self.cfg['decoder']['d_output'] = self.dataset.d_output
            # print(self.dataset.d_output)
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
        cfg['model']['config']['complement_map'] = self.dataset.tokenizer.complement_map
        caduceus_cfg = CaduceusConfig(**cfg['model']['config'])
        
        self.backbone = DNAEmbeddingModelCaduceus(config=caduceus_cfg)
        self.backbone.load_state_dict(model_state_dict, strict=True)
        
        #remove self.cfg['decoder']['_name_']
        del self.cfg['decoder']['_name_']
        self.decoder = EnformerDecoder(**self.cfg['decoder']) #could do with instantiating, but that is rather complex
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        
        if encoder_state_dict: #if it's emtpy, means no encoder, so just use identity! This should never be true for caduceus
            raise NotImplementedError('Encoder not implemented for Caduceus')
            # del self.cfg['encoder']['_name_']
            # self.encoder = EnformerEncoder(**self.cfg['encoder'])
            # self.encoder.load_state_dict(encoder_state_dict, strict=True)
            # self.skip_embedding = True
        else:
            self.encoder = IdentityNet()
            self.skip_embedding = False
        
        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()
        
    def __call__(self, idx=None, data=None):
        #now evaluate the model on one example
        if data is None:
            data = self.dataset[idx][0]
            data = data.unsqueeze(0)
        data = data.to(self.device)
        if data.dim() == 1: #if it's one hot, need to have unsqueezed it before
            data = data.unsqueeze(0)
        with torch.no_grad():
            x,_ = self.encoder(data)
            x,_ = self.backbone(x)
            x = self.decoder(x)
        return x
    
    def evaluate(self, batch_size=8):
        #now evaluate the model on the entire dataset
        # dataset_args = self.cfg['dataset'] #get the dataset args
        dataset_args = self.dataset_args
        dataset_args['return_target'] = False
        dataset_args['rc_aug'] = False
        dataset = self.DatasetClass(**dataset_args)
        # dataset = enformer_dataset.EnformerDataset(self.split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
        #                                                     return_CAGE=dataset_args['return_CAGE'], cell_type=dataset_args.get('cell_type', None),
        #                                                     kmer_len=dataset_args['kmer_len'], return_target=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = []
        for data in tqdm(loader, total=len(loader)):
            # data = data[0]
            x = self(data=data)
            results.append(x.cpu().numpy().astype(np.float16))
        return np.concatenate(results, axis=0)
    
    def evaluate_zarr(self, zarr_name, batch_size=8, overwrite = False):
        # raise NotImplementedError('Zarr evaluation not implemented yet. will do what evaluation does but save it to zarr by making new one and saving it in evals')
        #this will evaluate it and then saves it to a zarr file
        compression = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
        dataset_args = self.dataset_args.copy()
        dataset_args['return_target'] = False
        dataset = self.DatasetClass(**dataset_args)
        data,label = self.dataset[0] #here we need the label so use self.dataset which will return the label!
        
        #now doesn't load it but just creates zarr file and saves it there
        if overwrite:
            mode = 'w'
            print('overwriting zarr file')
        else:
            mode = 'r+'
        root = zarr.open(zarr_name, mode=mode)
        
        try:
            root.create_array('evals', shape=(len(self.dataset), *label.shape), chunks=(1, *label.shape), dtype='f2', compressors=compression)
        except zarr.errors.ContainsArrayError:
            print("eval Array already exists")
            assert root['evals'].shape == (len(self.dataset), *label.shape), f"Shape mismatch: {root['evals'].shape} vs {(len(self.dataset), *label.shape)}"
        
        #now make the data loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        #now loop through the data and save it to the zarr file
        batch_idx = 0
        with torch.no_grad():
            for data in tqdm(loader, total=len(loader)):
                x = self(data=data)
                b_size = x.shape[0]
                root['evals'][batch_idx:batch_idx+b_size] = x.cpu().numpy().astype(np.float16)
                batch_idx += b_size
        
        print('Finished saving to zarr file!')
            
    
    def correlate(self, zarr_name, axis, corr_type='spearman', pool_size=None, original_pool=1, pool_type='mean'):
        '''
        this function takes in a zarr file that is already had its outputs saved, and it correlates it, it can also handle pooling.
        You input an axis to correlate, the zarr file name, and the correlation type.
        Also if you want to do pooling, this way you can pool the output manually and compare it with the pooled output!
        Note that previously we used the other dataset, but here we can just manually pool it
        
        Args:
            zarr_name (str): the name of the zarr file to save the correlations to
            axis (int): the axis to correlate
            corr_type (str): the type of correlation to use, either 'spearman' or 'pearson'
            pool_size (int): the size of the pooling to do, if None, then no pooling is done
            original_pool (int): the original pooling size, if None, then no pooling is done
            pool_type (str): the type of pooling to do, either 'mean' or 'max'
        '''
        #this does spearman correlation and saves it, doesn't do pooling or anything specific? we can add that functionality later!
        

        #so we need the dataset, but will use self.dataset since we do need the labels
        #now we load the zarr file
        root = zarr.open(zarr_name, mode='r+')
        #now make sure evals exists
        assert 'evals' in root, f"evals not found in {zarr_name}"
        
        if pool_type != 'mean':
            raise NotImplementedError('Pooling type not implemented yet')
        
        #now we will try to save out the data in a correlation array
        if pool_size is None:
            #then the thing we make is just called coors
            zarr_name = 'corrs'
            print('correlating model as is')
            pool = 1
        else:
            pool = pool_size // original_pool
            zarr_name = f'corrs_{pool_size}'
            print(f'correlating model with pooling of {pool_size//original_pool}, original is {original_pool}, but pooling to be {pool_size}')
            
        
        if corr_type == 'spearman':
            corr_fn = spearmanr
        elif corr_type == 'pearson':
            corr_fn = pearsonr
        else:
            raise ValueError('Correlation type not recognized')
        
        #now try to create this array in zarr, no compression of course
        data,label = self.dataset[0] #here we need the label so use self.dataset which will return the label!
        label_shape = list(label.shape)
        #and pop out the axis
        label_shape.pop(axis)
        assert len(label_shape) == 1, f'Label shape is {label.shape}, should be 1d only, don\'t have loops for more dimensions, shape is {label_shape}'
        try:
            root.create_array(zarr_name, shape=(len(self.dataset), *label_shape), chunks=(1, *label_shape), dtype='float32')
        except zarr.errors.ContainsArrayError:
            print("corr array already exists, overwriting")
            assert root[zarr_name].shape == (len(self.dataset), *label_shape), f"Shape mismatch: {root[zarr_name].shape} vs {(len(self.dataset), *label_shape)}"
            
        
        #now we need to loop through the data and save it to the zarr file
        for i in tqdm(range(len(self.dataset))):
            data,label = self.dataset[i]
            out = root['evals'][i]
            label = label.numpy()
            corrs = np.zeros((label_shape))

            if pool > 1:
                #pool the data based on the 
                out = out.reshape(out.shape[0] // pool, pool, out.shape[1]).mean(axis=1)
                label = label.reshape(label.shape[0] // pool, pool, label.shape[1]).mean(axis=1)

            for j in range(len(corrs)):
                corr = corr_fn(label[:,j], out[:,j])
                corrs[j] = corr.correlation if not np.isnan(corr.correlation) else 0.0 #basically sets it to 0 if nan!
            
            root[zarr_name][i] = corrs
                

            
    
    def plot_track(self, idx, track=121):
        '''
        given an index, plots one track and compares it to the real results
        '''
        #now plot the track
        seq, label = self.dataset[idx]
        seq = seq.unsqueeze(0)
        # data = data.cpu().numpy()
        x = self(data = seq).cpu().squeeze().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(x[:, track], label='Predicted Coverage', color='b')
        ax1.set_title('Predicted Coverage')
        ax1.legend()
        
        ax2.plot(label[:, track], label='Actual Coverage', color='r')
        ax2.set_title('Actual Coverage')
        ax2.legend()
        
        ax2.set_xlabel('Position')
        fig.suptitle(f'{self.model_type} Model Coverage Comparison')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        return fig, (ax1, ax2) 
    
    def test(self):
        print('test worked')
    
    # def ism_init(self):
    #     numbers = (7, 8, 9, 10, 11)
    #     length = self.kmer_len
    #     combinations = itertools.product(numbers, repeat=length)
    #     self.combination_dict = {comb: idx for idx, comb in enumerate(combinations)}
    #     self.reverse_dict = {v: k for k, v in self.combination_dict.items()}
        
    # def ism(self, idx, pos):
    #     #we simply return the output of the model but we slightly modify the value
    #     true_out = self(idx)

        
    #     return true_out

        


def pearsonr2(x, y):
    # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
    # because they want to compute everything pairwise between rows of a
    # stacked x and y.
    #so essentially input x and y and returns correlations of each row in x with the corresponding row in y
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = np.sum((x - xm) * (y - ym), axis=-1)/(x.shape[-1]-1)
    sx = np.std(x, ddof=1, axis=-1)
    sy = np.std(y, ddof=1, axis=-1)
    rho = cov/(sx * sy)
    #now return the correlation
    return rho
        
        
def main(path, name, split='test', save_model_out=False):
    evals = Evals(path)
    print('model loaded, now evaluating')
    allout = evals.evaluate(4)
    split='test'
    labels = np.load(f'/data/leslie/sarthak/data/enformer/data/{split}_label.npy')
    allout = allout.transpose(0, 2, 1)
    labels = labels.transpose(0, 2, 1)
    if save_model_out:
        np.save(f'/data/leslie/sarthak/data/enformer/data/model_out/{name}.npy', allout)
    #if we are not using cage have to cut off labels
    if allout.shape[1] != labels.shape[1]:
        labels = labels[:, :allout.shape[1], :]
    corrs = pearsonr2(allout, labels)
    np.save(f'/data/leslie/sarthak/data/enformer/data/model_out/{name}.npy_corrs.npy', corrs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model and save correlation results.")
    
    # Adding arguments
    parser.add_argument('--path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--name', type=str, required=True, help='Name for the output file')
    parser.add_argument('--split', type=str, default='test', help='Split type (default: test)')
    parser.add_argument('--save_model_out', action='store_true', help='Save model output')
    args = parser.parse_args()
    
    # Calling main with arguments
    main(args.path, args.name, args.split)