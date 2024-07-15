
import torch
from random import randrange, random, sample
import numpy as np
import sys
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import h5py
import pandas as pd


"""

Loads a dataset for profile prediction from Enformer data
First needs to have been converted to the numpy array and hdf5 file

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

class EnformerDataset():
    def __init__(
        self,
        split,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        use_padding = True,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        uppercase = True,
        d_output = None,
        data_path = None, #the path for the numpy arrays of the data, should contain chroomosome and summit locations
        return_CAGE = False, #doesn't return cage data
        load_into_memory = False, #if you have the RAM, load it all into memory
        cell_type = None, #whether to just do one specific cell type
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer #defined the proper one in the data loader
        self.d_output = d_output
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval
        self.rc_aug = rc_aug
        self.uppercase = uppercase
        self.return_CAGE = return_CAGE


        if data_path is None:
            data_path=f'/data/leslie/sarthak/data/enformer/data/{split}_seq.npz'
        seq_data = np.load(data_path)
        self.seq = np.array(seq_data['sequence_array'])
        self.seq_rc = np.array(seq_data['sequence_array_rc'])
        #close the file
        seq_data.close()

        #now we need to load the labels
        
        if load_into_memory:
            with h5py.File(data_path.replace('_seq.npz', '_label.h5'),'r') as f:
                self.labels = f['labels'][:]
        else:
            self.labels = h5py.File(data_path.replace('_seq.npz', '_label.h5'),'r')['labels']
            
        self.keep = None
        if self.d_output is None and self.return_CAGE:
            self.d_output = self.labels.shape[-1]
        else:
            self.d_output = 4675
        
        if cell_type is not None:
            targets = '/data/leslie/sarthak/data/enformer/data/human/targets.txt'
            targets = pd.read_csv(targets, sep='\t')
            #nah let's just do it properly, we'll have overlap, but it's fine!
            #get the indices to keep
            self.keep = targets[targets['description'].str.endswith('K562', na=False)]['index'].to_numpy()
            if not self.return_CAGE:
                self.keep = self.keep[self.keep < 4675]
            self.d_output = len(self.keep)
            
            #here we could find a way to actually implement it and find all the k562 ones, but in my case we want just a few
            #so we can manually define it for k562
            # if cell_type != 'K562':
            #     raise ValueError('Cell type not implemented')
            # indices_keep = np.array(121, 
            

        
    def __len__(self):
        return self.seq.shape[0]

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        if idx < 0:
            idx = len(self) + idx # negative indexing

        #simply access the sequences and labels
        #first determine if rc
        if self.rc_aug and coin_flip():
            seq = self.seq_rc[idx]
            flip = True
        else:
            seq = self.seq[idx]
            flip = False
        if len(seq) < self.max_length:
            #pad with 11s on both sides
            pad_left = (self.max_length - len(seq)) // 2
            pad_right = self.max_length - len(seq) - pad_left
            seq = np.concatenate([np.ones(pad_left)*11, seq, np.ones(pad_right)*11])
        
        #and gather the data
        targets = self.labels[idx]
        
        seq = torch.LongTensor(seq)
        # print(counts)
        targets = torch.FloatTensor(targets)
        if not self.return_CAGE:
            targets = targets[:, :4675]
        if flip: #this flips each column independently
            targets = targets.flip(dims=[0])
        
        if self.keep is not None:
            targets = targets[:, self.keep]

        return seq, targets #coutns is literally just the sum of the cts + 1 then logged

'''
Can run in the terminal using these commands
cd /data/leslie/sarthak/hyena/hyena-dna/
python
import src.dataloaders.datasets.enformer_dataset as enformer_dataset
dataset = enformer_dataset.EnformerDataset('train', 160_000, rc_aug = True, cell_type = 'K562')
out = dataset[0]
out[0] #the input data tokenized
'''

