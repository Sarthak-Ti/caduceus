
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
import pyBigWig

"""

Loads a dataset using the already aligned data from GPN MSA. HEavily based on enformer dataset
default allows loading the enformer like data at high resolution
uses bed file to load larger length sequence and zarr file labels
It's also been modified to work in parallel by opening the zarr store in the __getitem__ function

"""

base_path = os.path.abspath(__file__).split('sarthak')[0]+'sarthak/'
# print('base_path:',base_path)


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

class GPNMSADataset():
    def __init__(
        self,
        split,
        max_length,
        pad_max_length=None,
        rc_aug=False,
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
        msa_path = None,
        pad_one_hot = 512,
        phastcons = False,
        phylop = False,
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.return_CAGE = return_CAGE
        self.return_target = return_target
        self.one_hot = one_hot
        self.pool = pool
        self.pool_type = pool_type
        self.pad_one_hot = pad_one_hot
        self.phastcons = phastcons
        self.phylop = phylop

        
        # genome_np = os.path.join(base_path,'data/chrombpnet_test/hg38_tokenized.npz')
        # if kmer_len is not None:
        #     genome_np = os.path.join(base_path,f'data/chrombpnet_test/hg38_tokenized_kmer_{kmer_len}.npz')
        #     print(f'Using kmer genome with length {kmer_len}')
        #      #RC will be implemented by loading the json file and then mapping every element to the reverse complement and reversing order
        #     with open(os.path.join(base_path,f'data/enformer/data/complement_map_{kmer_len}mer.json'), 'r') as f:
        #         complement_map = json.load(f)
        #     max_key = int(list(complement_map.keys())[-1])
        # else:
        #     genome_np = os.path.join(base_path,'data/chrombpnet_test/hg38_tokenized.npz')
        #     complement_map = {"7": 10, "8": 9, "9": 8, "10": 7, "11": 11}
        #     max_key = 11
        
        #load in the tokenized genome
        # with np.load(genome_np) as data:
        #     self.genome = {key: np.array(data[key]) for key in data}
        
        if msa_path is None:
            self.msa_path = '/data1/lesliec/sarthak/data/gpn/99.zarr'
        else:
            self.msa_path = msa_path
        # self.msa = zarr.open(msa_path, mode='r')
        if self.phastcons:
            self.phastcons_path = '/data1/lesliec/sarthak/data/gpn/hg38.phastCons100way.bw'
        if self.phylop:
            self.phylop_path = '/data1/lesliec/sarthak/data/gpn/hg38.phyloP100way.bw'
        
        #also initialize the way we convert to tokenized data
        self.key = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': 4, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
        lookup = np.zeros(256, dtype=np.int8)
        lookup[:] = -1  # Default value for unmapped characters
        for char, value in self.key.items():
            lookup[ord(char)] = value
        
        self.lookup = lookup
        
        if self.rc_aug: #the rc aug is the same idea but flips what the bases correspond to. Could have flipped ohe, but weird with the N...
            # raise NotImplementedError('rc_aug not implemented with this, but would be easy by following what they did with the key... wait that\'s actually smart is just use the other key')
            self.rc_key = {'A': 3, 'C': 2, 'G': 1, 'T': 0, 'N': 4, '-': 4, 'a': 3, 'c': 2, 'g': 1, 't': 0, 'n': 4}
            lookup = np.zeros(256, dtype=np.int8)
            lookup[:] = -1  # Default value for unmapped characters
            for char, value in self.rc_key.items():
                lookup[ord(char)] = value
            self.rc_lookup = lookup

        #create the complement array to do RC augmentation
        # self.complement_array = np.zeros(max_key + 1, dtype=int)
        # for k, v in complement_map.items():
        #     self.complement_array[int(k)] = v
        
        
            

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
        self.split = split    
        
        if data_path is None:
            self.data_path = '/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr'
            # self.labels = zarr.open(data_path, mode='r')[split]
        else:
            self.data_path = data_path
            # self.labels = zarr.open(data_path, mode='r')[split] #we made it more effective, so not with this ['labels']
        
        self.keep = None
        if self.return_CAGE:
            self.d_output = 5313
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
            raise NotImplementedError('Have to implement this, can\'t open self.labels, need self.keep approach')
            # self.keep = np.array([cell_type])
            # self.labels = np.array(self.labels[:, :, cell_type:cell_type+1])
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
        
        msa = zarr.open(self.msa_path, mode='r')

        row = self.seqs_np[idx]
        chrom = row[0]
        start = row[1]
        end = row[2]
        diff = (self.max_length - self.length)//2 #note this works even if we want a shorter length!
        start = start - diff
        end = end + diff
        
        #now deal with padding if we are outside of the genome (roughtly 10 values, so we could ignore them or just set as n?)
        leftpad = np.zeros((0,100))
        rightpad = np.zeros((0,100))
        if start < 0:
            leftpad = np.ones((-start,100))*4 #pad with 4 as 4 corresponds to N
            start = 0
        chromlen = chrom_info[chrom][1]
        if end > chromlen:
            rightpad = np.ones((end-chromlen,100))*4
            end = chromlen
        
        
        if self.rc_aug and coin_flip():
            temp_seq = msa[chrom[3:]][start:end]
            arr_as_int = np.frombuffer(temp_seq.tobytes(), dtype=np.uint8)
            mapped_arr = np.take(self.rc_lookup, arr_as_int)
            seq = mapped_arr.reshape(temp_seq.shape[0], temp_seq.shape[1])
            seq = np.concatenate([leftpad, seq, rightpad])
            seq = torch.LongTensor(seq)

            #and flip seqs along the axis 0
            seq = seq.flip(0)
            flip = True
        else:
            #now we get our data and convert it!
            temp_seq = msa[chrom[3:]][start:end]
            arr_as_int = np.frombuffer(temp_seq.tobytes(), dtype=np.uint8)
            mapped_arr = np.take(self.lookup, arr_as_int)
            seq = mapped_arr.reshape(temp_seq.shape[0], temp_seq.shape[1])
            seq = np.concatenate([leftpad, seq, rightpad])

            seq = torch.LongTensor(seq)
            flip = False

        if self.one_hot:
            x = seq
            x_onehot = torch.nn.functional.one_hot(x, num_classes=5).float().reshape(x.shape[0],-1).transpose(1, 0) #5 classes because N or - is its own class, ACGTN in that order!
            #also reshape to concatenate all the data in that dimension
            # x_onehot = torch.nn.functional.one_hot((x-7)%4, num_classes=4).float().transpose(1, 0) #need to make sure it is the right order, so now is shape 5xseq_len
            #and we have to stack it!
            
            if self.pad_one_hot is not None:
                tempseq = torch.zeros(self.pad_one_hot, x_onehot.shape[1])
                tempseq[:x_onehot.shape[0], :] = x_onehot
                x_onehot = tempseq

            seq = x_onehot
        else:
            raise NotImplementedError('Only one hot implemented, not sure how to do it without one hot, cannot do phastcons like this either')
        
        #now we append the phastcons labels
        
        if not self.return_target:
            return seq
        
        # return 0
        #and gather the data
        labels = zarr.open(self.data_path, mode='r')[self.split]
        targets = labels[idx]
        # seq = torch.LongTensor(seq)
        # print(counts)
        targets = torch.FloatTensor(targets)
        if not self.return_CAGE and len(labels.shape) == 3: #otherwise we already filtered it
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
            
        
        

        return seq, targets #seq is size seq_len, targets is 896xnum_targets

'''
Can run in the terminal using these commands
cd /data/leslie/sarthak/hyena/hyena-dna/
python
import src.dataloaders.datasets.GPNMSA_dataset as d
dataset = d.GPNMSADataset('test', 196608, return_CAGE=True, one_hot=True)
out = dataset[0]
out[0] #the input data tokenized

#if you want to do it for a different dataset (my less processed one)
dataset = d.GPNMSADataset('train', 196608, cell_type='DNase', one_hot=True, data_path='/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr', pool = 64)
zarr_open = zarr.open('/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr', mode='r')
'''

