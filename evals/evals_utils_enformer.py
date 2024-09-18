#this file contains an evaluation helper class which will contain the data loader and the model, and enables quick evaluation of the model on the test set
import sys
sys.path.append('/data/leslie/sarthak/caduceus/')
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import EnformerDecoder
# from src.tasks.encoders import EnformerEncoder
from caduceus.configuration_caduceus import CaduceusConfig
import torch
import numpy as np
import src.dataloaders.datasets.enformer_dataset as enformer_dataset
import yaml
from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import argparse

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
                 ) -> None:
        
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
            self.dataset = enformer_dataset.EnformerDataset(split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
                                                            return_CAGE=dataset_args['return_CAGE'], cell_type=dataset_args.get('cell_type', None),
                                                            kmer_len=dataset_args['kmer_len']) #could use dataloader instead, but again kinda complex
        else:
            self.dataset = dataset
         
        self.cfg['decoder']['d_output'] = self.dataset.d_output
        print(self.dataset.d_output)
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
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        with torch.no_grad():
            x,_ = self.encoder(data)
            x,_ = self.backbone(x)
            x = self.decoder(x)
        return x
    
    def evaluate(self, batch_size=8):
        #now evaluate the model on the entire dataset
        dataset_args = self.cfg['dataset'] #get the dataset args
        dataset = enformer_dataset.EnformerDataset(self.split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
                                                            return_CAGE=dataset_args['return_CAGE'], cell_type=dataset_args.get('cell_type', None),
                                                            kmer_len=dataset_args['kmer_len'], return_target=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = []
        for data in tqdm(loader, total=len(loader)):
            # data = data[0]
            x = self(data=data)
            results.append(x.cpu().numpy().astype(np.float16))
        return np.concatenate(results, axis=0)
    
    def plot_track(self, idx, track=121):
        '''
        given an index, plots one track and compares it to the real results
        '''
        #now plot the track
        seq, label = self.dataset[idx]
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
        
        
def main(path, name, split='test'):
    evals = Evals(path)
    print('model loaded, now evaluating')
    allout = evals.evaluate(4)
    split='test'
    labels = np.load(f'/data/leslie/sarthak/data/enformer/data/{split}_label.npy')
    allout = allout.transpose(0, 2, 1)
    labels = labels.transpose(0, 2, 1)
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
    args = parser.parse_args()
    
    # Calling main with arguments
    main(args.path, args.name, args.split)