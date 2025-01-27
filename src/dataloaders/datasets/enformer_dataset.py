
import torch
from random import random
import numpy as np
# import sys
# sys.path.append(os.path.join(base_path,'/hyena/hyena-dna/')
# from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
# import h5py
import pandas as pd
import zarr
import json
import os

"""

Loads a dataset for profile prediction from Enformer data
updated class, uses bed file to load larger length sequence and zarr file labels
Significantly more efficient and faster

"""

base_path = os.path.abspath(__file__).split('sarthak')[0]+'sarthak/'
print('base_path:',base_path)


chrom_info= {'chr1': [10000, 248946422],
 'chr2': [10000, 242183529],
 'chr3': [10000, 198235559],
 'chr4': [10000, 190204555],
 'chr5': [10000, 181478259],
 'chr6': [60000, 170745979],
 'chr7': [10000, 159335973],
 'chr8': [60000, 145078636],
 'chr9': [10000, 138334717],
 'chr10': [10000, 133787422],
 'chr11': [60000, 135076622],
 'chr12': [10000, 133265309],
 'chr13': [16000000, 114354328],
 'chr14': [16022637, 106883718],
 'chr15': [17000000, 101981189],
 'chr16': [10000, 90228345],
 'chr17': [60000, 83247441],
 'chr18': [10000, 80263285],
 'chr19': [60000, 58607616],
 'chr20': [60000, 64334167],
 'chr21': [5010000, 46699983],
 'chr22': [10510000, 50808468],
 'chrX': [10000, 156030895],
 'chrY': [2781479, 56887902],}

class Tokenizer():
    #a super basic class that literally just stores the complement map
    def __init__(self, kmer_len):
        if kmer_len is None:
            self.complement_map = {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 10,
                "8": 9,
                "9": 8,
                "10": 7,
                "11": 11,
                "12": 12,
                "13": 13,
                "14": 14,
                "15": 15
            }
        else:
            with open(os.path.join(base_path,f'data/enformer/data/complement_map_{kmer_len}mer.json'), 'r') as f:
                self.complement_map = json.load(f)

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
        load_into_memory = None, #if you have the RAM, load it all into memory
        cell_type = None, #whether to just do one specific cell type
        kmer_len = None,
        return_target = True,
        one_hot = False,
        pool = 1,
        pool_type = 'mean',
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
        self.return_target = return_target
        self.one_hot = one_hot
        self.pool = pool
        self.pool_type = pool_type
        
        if self.rc_aug:
            raise NotImplementedError('rc_aug not implemented with this, but would be easy following profile dataset')
        
        genome_np = os.path.join(base_path,'data/chrombpnet_test/hg38_tokenized.npz')
        if kmer_len is not None:
            genome_np = os.path.join(base_path,f'data/chrombpnet_test/hg38_tokenized_kmer_{kmer_len}.npz')
            print(f'Using kmer genome with length {kmer_len}')
             #RC will be implemented by loading the json file and then mapping every element to the reverse complement and reversing order
            with open(os.path.join(base_path,f'data/enformer/data/complement_map_{kmer_len}mer.json'), 'r') as f:
                complement_map = json.load(f)
            max_key = int(list(complement_map.keys())[-1])
        else:
            genome_np = os.path.join(base_path,'data/chrombpnet_test/hg38_tokenized.npz')
            complement_map = {"7": 10, "8": 9, "9": 8, "10": 7, "11": 11}
            max_key = 11
        
        #load in the tokenized genome
        with np.load(genome_np) as data:
            self.genome = {key: np.array(data[key]) for key in data}

        #create the complement array to do RC augmentation
        self.complement_array = np.zeros(max_key + 1, dtype=int)
        for k, v in complement_map.items():
            self.complement_array[int(k)] = v
            

        if split == 'val':
            split = 'valid'
        seqs = pd.read_csv(os.path.join(base_path,'data/enformer/data/human/sequences.bed'), sep='\t', header=None)
        self.seqs_bed = seqs[seqs[3] == split]
        # self.seq = np.zeros((len(self.seqs_bed), max_length), dtype=self.genome['chr1'].dtype) #note with 16 bit it takes a lo;t of space, can migrate to not preallocating, just get when we need it
        self.length = 131072 #the length of the sequences form enformer
        self.seqs_np = self.seqs_bed.to_numpy()
        # for i in range(self.seqs_np.shape[0]):
        #     row = self.seqs_np[i]
        #     chrom = row[0]
        #     start = row[1]
        #     end = row[2]
        #     diff = (self.max_length - length)//2 #note this works even if we want a shorter length!
        #     start = start - diff
        #     end = end + diff
        #     leftpad = np.zeros(0)
        #     rightpad = np.zeros(0)
        #     if start < 0:
        #         leftpad = np.ones(-start)*11
        #         start = 0
        #     chromlen = chrom_info[chrom][1]
        #     if end > chromlen:
        #         rightpad = np.ones(end-chromlen)*11
        #         end = chromlen
        #     seq = np.concatenate([leftpad, self.genome[chrom][start:end], rightpad])
        #     self.seq[i] = seq
        
        #this makes it so we have loaded the whole sequence information into memory
        
        #now we have to load the other data from the zarr file
        if split == 'valid':
            split = 'val'
        if data_path is None:
            data_path = os.path.join(base_path,f'data/enformer/data/{split}_label.zarr')
            self.labels = zarr.open(data_path, mode='r')['labels']
        else:
            self.labels = zarr.open(data_path, mode='r')[split] #we made it more effective, so not with this ['labels']
        
        self.keep = None
        if self.return_CAGE:
            self.d_output = self.labels.shape[-1]
        else:
            self.d_output = 4675 #the non cage data!
        
        if cell_type=='DNase':
            self.d_output = 674
            self.keep = np.array([i for i in range(0, 674)])
        elif isinstance(cell_type,str):
            targets = os.path.join(base_path,'data/enformer/data/human/targets.txt')
            targets = pd.read_csv(targets, sep='\t')
            #nah let's just do it properly, we'll have overlap, but it's fine!
            #get the indices to keep
            self.keep = targets[targets['description'].str.endswith(cell_type, na=False)]['index'].to_numpy()
            if not self.return_CAGE:
                self.keep = self.keep[self.keep < 4675]
                assert(len(self.keep) > 0)
            self.d_output = len(self.keep)
        elif isinstance(cell_type, list):
            self.keep = np.array(cell_type)
            self.d_output = len(self.keep)
        elif isinstance(cell_type, int):
            # self.keep = np.array([cell_type])
            self.labels = np.array(self.labels[:, :, cell_type:cell_type+1])
            # self.labels = self.labels[:, :, cell_type:cell_type+1]
            self.keep = None #we just limited the data and loaded it into memory
            self.d_output=1
        elif cell_type is not None:
            raise ValueError('Cell type not implemented')
        if self.keep is not None:
            self.d_output = len(self.keep)
        
            #here we could find a way to actually implement it and find all the k562 ones, but in my case we want just a few
            #so we can manually define it for k562
            # if cell_type != 'K562':
            #     raise ValueError('Cell type not implemented')
            # indices_keep = np.array(121, 
        self.tokenizer = Tokenizer(kmer_len)
            

        
    def __len__(self):
        return self.seqs_np.shape[0]

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        if idx < 0:
            idx = len(self) + idx # negative indexing

        row = self.seqs_np[idx]
        chrom = row[0]
        start = row[1]
        end = row[2]
        diff = (self.max_length - self.length)//2 #note this works even if we want a shorter length!
        start = start - diff
        end = end + diff
        leftpad = np.zeros(0)
        rightpad = np.zeros(0)
        if start < 0:
            leftpad = np.ones(-start)*11
            start = 0
        chromlen = chrom_info[chrom][1]
        if end > chromlen:
            rightpad = np.ones(end-chromlen)*11
            end = chromlen
        seq = np.concatenate([leftpad, self.genome[chrom][start:end], rightpad])
        
        if self.rc_aug and coin_flip():
            seq = self.complement_array[seq[::-1]]
            flip = True
        else:
            flip = False
        if not self.return_target:
            return torch.LongTensor(seq)
        
        #and gather the data
        targets = self.labels[idx]
        seq = torch.LongTensor(seq)
        # print(counts)
        targets = torch.FloatTensor(targets)
        if not self.return_CAGE and len(self.labels.shape) == 3: #otherwise we already filtered it
            targets = targets[:, :4675]
        if flip: #this flips each column independently
            targets = targets.flip(dims=[0])
        
        if self.keep is not None:
            targets = targets[:, self.keep]
            
        if self.pool > 1:
            #first reshape
            if targets.shape[0] % self.pool != 0:
                raise ValueError('Pool size must divide sequence length')
            
            targets = targets.view(targets.size(0) // self.pool, self.pool, targets.size(1))
            if self.pool_type != 'mean':
                raise NotImplementedError('Only mean pooling implemented')
            targets = targets.mean(dim=1)
            
        
        if self.one_hot:
            x = seq
            x_onehot = torch.nn.functional.one_hot((x-7)%4, num_classes=4).float().transpose(1, 0) #need to make sure it is the right order, so now is shape 4xseq_len
            if 11 in x:
                x_onehot[:, x == 11] = 0
            seq = x_onehot

        return seq, targets #seq is size seq_len, targets is 896xnum_targets

'''
Can run in the terminal using these commands
cd /data/leslie/sarthak/hyena/hyena-dna/
python
import src.dataloaders.datasets.enformer_dataset as enformer_dataset
dataset = enformer_dataset.EnformerDataset('test', 196608, return_CAGE=True, one_hot=True)
out = dataset[0]
out[0] #the input data tokenized

#if you want to do it for a different dataset (my less processed one)
dataset = enformer_dataset.EnformerDataset('train', 196608, cell_type='DNase', one_hot=True, data_path='/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr', pool = 64)
zarr_open = zarr.open('/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr', mode='r')
'''

