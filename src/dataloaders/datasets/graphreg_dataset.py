
import torch
from random import random
import numpy as np
# import sys
# sys.path.append(os.path.join(base_path,'/hyena/hyena-dna/')
# from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
# import h5py
# import pandas as pd
# import zarr
import json
import os

"""

Loads the dataset used by graphreg, utilizes numpy files that are obtained from the tfr record files
The translation si found in load_data.ipynb in GraphReg
Based on the enformer dataset class
KMer is not currently implemented but can be, need to convert seq.py to seq_6.py or something, we've done this before
In this first step, we don't need to stich anything together, only need that for the later steps!
The details for which data we use is found in Seq-CNN_e2e.py which shows how we separate epigenomic data from Y data and also how to sum Y data together
It's length 180k because we have 60k binned, and we have 3 epigenomic marks

For next dataset, can load the model and run it through as part of the dataset maybe?? idk maybe too intensive, so let's not... can saave outputs!

"""

base_path = os.path.abspath(__file__).split('sarthak')[0]+'sarthak/'
# print('base_path:',base_path)

class Tokenizer():
    #a super basic class that literally just stores the complement map
    def __init__(self, vocab_size = 16, kmer_len=None):
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
            }
            if vocab_size > 12:
                for i in range(12,vocab_size):
                    self.complement_map[str(i)] = i
        else:
            with open(os.path.join(base_path,f'data/enformer/data/complement_map_{kmer_len}mer.json'), 'r') as f:
                self.complement_map = json.load(f)

#now split the chromosomes into train val and test splits
splits_dict = {
    'train': [1,2,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22],
    'val': [3,13],
    'test': [4,14]
}


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

class GraphRegDataset():
    def __init__(
        self,
        split,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        rc_aug=False,
        replace_N_token=False,  # replace N token with pad token
        data_path = None, #the path for the numpy arrays of the data, should contain chroomosome and summit locations
        kmer_len = None,
        cell_type = 'GM12878',
        has_TSS = False, #this will see if we have to have the TSS data to keep it as an example
        remove_repeats = False, #checks if there are repeats in the data, which many cuz take 6 million, shift right 2 million, then do again, so 4 million overlap...
        clean_data = True,
        vocab_size = 16,
        one_hot = False
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer #defined the proper one in the data loader
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.rc_aug = rc_aug
        self.clean_data = clean_data
        self.one_hot = one_hot

        # if self.max_length != 6_000_000:
        #     raise NotImplementedError('max_length is only working on 6_000_000, could allow shorter sequences...')
        
        # if self.rc_aug or self.add_eos or self.replace_N_token:
        #     raise NotImplementedError('not implemented with this, but rc_aug would be easy following profile dataset')
        
        if data_path is None:
            data_path = os.path.join(base_path,'data/GraphReg/torchdata/')
        
        chr_list = np.load(data_path+'chr_list.npy')
        
        #load the sequences
        if kmer_len is None:
            self.seqs = np.load(data_path+'seq.npy')
            complement_map = {"7": 10, "8": 9, "9": 8, "10": 7, "11": 11}
            max_key = 11
        else:
            # raise NotImplementedError('kmer not implemented, need to kmerize the sequences')
            self.seqs = np.load(data_path+f'seq_{kmer_len}mer.npy')
            print(f'Using kmer genome with length {kmer_len}')
            #RC will be implemented by loading the json file and then mapping every element to the reverse complement and reversing order
            with open(os.path.join(base_path,f'data/enformer/data/complement_map_{kmer_len}mer.json'), 'r') as f:
                complement_map = json.load(f)
            max_key = int(list(complement_map.keys())[-1])
        self.max_key = max_key
        
        #create the complement array to do RC augmentation
        self.complement_array = np.zeros(max_key + 1, dtype=int)
        for k, v in complement_map.items():
            self.complement_array[int(k)] = v

        # self.labels = zarr.open(data_path, mode='r')['labels']
        self.tss = np.load(data_path+'tss.npy')
        self.labels = np.load(data_path+f'epi_{cell_type}.npy')
        #and now find the splits
        splits = splits_dict[split]
        mask = np.isin(chr_list, splits)
        self.seqs = self.seqs[mask]
        self.labels = self.labels[mask]
        self.tss = self.tss[mask]
        # print('Loaded data, shape:',self.seqs.shape, self.labels.shape, self.tss.shape) #for test Loaded data, shape: (150, 6000000) (150, 180000) (150, 1200)

        if remove_repeats:
            self.seqs = self.seqs[:,2_000_000:4_000_000] #only the middle 2 million
            # self.labels = self.labels[:,20_000:40_000] #only the middle 2 million but binned to 100bp
            #oh it's not already in that third dimension, so need to reshape it to work with the data...
            #when we do the reshape, will make it so that we take every third element for each epigenomic mark
            #it's apparently identical if we just do this
            self.labels = self.labels[:,20_000*3:40_000*3] #then it still works with the later reshape, this seems to actually work, but have a hard time reasoning through it!

        if has_TSS:
            summed = self.tss[:,400:800].sum(1)
            mask = summed > 0
            self.seqs = self.seqs[mask]
            self.labels = self.labels[mask]
            self.tss = self.tss[mask]
        

        
        #now we also define this chunking, basically break it up into chunks
        #we divide 6 million by max_length to get the number of chunks
        current_len = self.seqs.shape[1]
        assert current_len % self.max_length == 0, 'max_length must divide 6_000_000, or 2_000_000 if no repeats'
        self.num_chunks = current_len // self.max_length
        self.seqs = self.seqs.reshape(self.num_chunks*self.seqs.shape[0], self.seqs.shape[1]//self.num_chunks) #for the sequence, second part is just max_length
        self.labels = self.labels.reshape(self.num_chunks*self.labels.shape[0], self.labels.shape[1]//self.num_chunks//3, 3) #for the 3 epigenomic marks, and it's along the length

        if self.clean_data: #do it after the chunking, else still examples with 0!
            #find which elements have more than 50% of the data as N, and remove those points
            self.num_original = self.seqs.shape[0]
            Ns = self.seqs == max_key #11 for nucleotide, but for kmer is different, the last term which is max key
            Ns = Ns.sum(axis=1)
            mask = Ns < self.seqs.shape[1]//2 #it has to have over 50% of the data as not N
            self.seqs = self.seqs[mask]
            self.labels = self.labels[mask]
            self.num_cleaned = self.seqs.shape[0] - self.num_original
        
        #this approach makes it so that we take every third element and that is what corresponds to the epigenomic marks
        #based on graph reg code, we should get elements 0,3,6,... to len of seq*3 as labels[0,:,0]. I think ithis is likely correct! corresponds to what the graph reg data does?
        #can test it with arange and reshape it into matrix you wish, then each element of arange corresponds to the storage of the tensor
        #now get the labels separately
        self.epi_marks = ['h3k4me3', 'h3k27ac', 'dnase']

        #no need for this, we can simply send the whole label over
        # self.y = {}
        # for i, epi in enumerate(self.epi_marks):
        #     self.y[epi] = self.labels[:,:,i]
        
        self.tokenizer = Tokenizer(vocab_size, kmer_len)
        

        
    def __len__(self):
        return self.seqs.shape[0]

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        if idx < 0:
            idx = len(self) + idx # negative indexing

        #simply access the element
        seq = self.seqs[idx]
        label = self.labels[idx]
               
        if self.rc_aug and coin_flip():
            seq = self.complement_array[seq[::-1]]
            flip = True
        else:
            flip = False
        
        seq = torch.LongTensor(seq)
        # print(counts)
        targets = torch.FloatTensor(label)

        if flip: #this flips each column independently
            targets = targets.flip(dims=[0])
            
        if self.one_hot:
            onehot = torch.nn.functional.one_hot((seq-7)%4, num_classes=4)
            if (seq==11).any():
                onehot[seq==11,0] = 0
            seq = onehot.transpose(1,0).float() #transpose to make it the right shape, which is 4 x len

        return seq, targets #seq is 

'''
Can run in the terminal using these commands
cd /data/leslie/sarthak/caduceus/
python
import src.dataloaders.datasets.graphreg_dataset as d
dataset = d.GraphRegDataset('test', 100_000, clean_data=True)
out = dataset[0]
out[0] #the input data tokenized
'''

