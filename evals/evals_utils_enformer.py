#this file contains an evaluation helper class which will contain the data loader and the model, and enables quick evaluation of the model on the test set
import sys
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.models.sequence.dna_embedding import DNAEmbeddingModel
from src.tasks.decoders import EnformerDecoder
from src.tasks.encoders import EnformerEncoder
import torch
import numpy as np
import src.dataloaders.datasets.enformer_dataset as enformer_dataset
import yaml
import os

class IdentityNet(torch.nn.Module):
    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        return x, None

class Evals():
    def __init__(self,
                 ckpt_path,
                 dataset=None,
                 split = 'test',
                 ) -> None:
        
        #now load the cfg from the checkpoint path
        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        self.cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #now set up dataset
        if dataset is None:
            dataset_args = self.cfg['dataset']
            self.dataset = enformer_dataset.EnformerDataset(split, dataset_args['max_length'], rc_aug = dataset_args['rc_aug'],
                                                            return_CAGE=dataset_args['return_cage'], cell_type=dataset_args.get('cell_type', None),
                                                            kmer_len=dataset_args['kmer_len'])
        else:
            self.dataset = dataset
         
        self.cfg['decoder']['d_output'] = self.dataset.d_output     
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )
        model_state_dict = state_dict["state_dict"]
        for key in list(model_state_dict.keys()):
            if 'output_transform' in key:
                print(key)
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
        
        self.backbone = DNAEmbeddingModel(**self.cfg['model'])
        self.backbone.load_state_dict(model_state_dict, strict=True)
        
        self.decoder = EnformerDecoder(**self.cfg['decoder'])
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        
        if encoder_state_dict: #if it's emtpy, means no encoder, so just use identity!
            self.encoder = EnformerEncoder(**self.cfg['encoder'])
            self.encoder.load_state_dict(encoder_state_dict, strict=True)
        else:
            self.encoder = IdentityNet()
        
        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()
        

        
        
        
def main():
    evals = Evals('/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-08-03/10-17-19-541733/checkpoints/00-val_loss=0.68478.ckpt')

if __name__ == '__main__':
    main()