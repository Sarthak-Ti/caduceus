
import pandas as pd
import torch
from random import randrange, random
import numpy as np
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

"""

Dataset for sampling arbitrary intervals from the human genome but from cCRE regions
Adapted from teh hg38_dataset.py class

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp



class DNaseDataset():
    def __init__(
        self,
        split,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        use_padding = True,
        # return_seq_indices=False,
        # shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        uppercase = True,
        cell_types = 1205, #no longer needed for me
        d_output = None,
        filter = False,
        classification = False,
    ):
        # print(filter)
        # import sys
        # sys.exit()
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer #defined the proper one in the data loader
        # self.tokenizer = CharacterTokenizer(
        #     characters=['A', 'C', 'G', 'T', 'N', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        #     model_max_length=1026,  # add 2 since default adds eos/eos tokens, crop later
        #     add_special_tokens=False,
        # )
        self.cell_types = cell_types
        self.d_output = d_output
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval
        self.rc_aug = rc_aug
        self.uppercase = uppercase
        self.filter = filter
        self.classification = classification
        # self.cell_types = cell_types

        #we load in based on the split
        data_path = f'/data/leslie/sarthak/data/{split}.csv'
        #load in csv
        df_raw = pd.read_csv(data_path) #note this works because we have a header in the csv
        #now only take the column titled sequence
        # self.df = df_raw[['sequence']]
        #turn to numpy array
        self.array = df_raw.to_numpy() #this is the array which contains the identifiers and the sequences
        
        #and load in the cell type information
        self.cell_type_array = pd.read_csv('/data/leslie/sarthak/data/cCRE_celltype_matrices/cell_type_permutations.csv', header=None).to_numpy() #the index, token permutations, and celltype identifier 
        
        ccre_df = pd.read_csv('/data/leslie/sarthak/data/cCRE_celltype_matrices/cCRE_identities.csv', header=None)
        self.cCRE_identities = ccre_df.to_numpy() #contains the cCRE identifiers in order for the matrix

        cCRE_dict = {}
        for i in range(self.cCRE_identities.shape[0]):
            # print(DNase.cCRE_identities[i][0])
            cCRE_dict[self.cCRE_identities[i][0]] = i
        self.cCRE_dict = cCRE_dict
        
        self.cell_dnase_levels = np.load('/data/leslie/sarthak/data/cCRE_celltype_matrices/cell_type_levels.npy')
        #numpy array which shows the levels of the DNase for each cell type cCRE combination
        
        if self.filter:
            #finally we consider that we filtered the cell types, so we load in that file
            #this is an ugly workaround, basically we use the hash table to find the index in the large array
            self.filtered_cell_types = np.load('/data/leslie/sarthak/data/cCRE_celltype_matrices/cell_type_f_indices.npy')
            # self.filtered_indices = {}
            # for i in range(self.filtered_cell_types.shape[0]):
            #     self.filtered_indices[i] = int(self.filtered_cell_types[i])
            #that is incredibly inefficient, here's the better way
            self.cell_dnase_levels = self.cell_dnase_levels[:,self.filtered_cell_types]
        self.cell_types = self.cell_dnase_levels.shape[1]
        
        #and print the tokenizer vocab
        # print(self.tokenizer._vocab_str_to_int)
        
        
    def __len__(self):
        return len(self.array) * self.cell_types

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        if idx < 0:
            idx = len(self) + idx # negative indexing
        
        #we first find the specific cell type
        celltype_idx = idx%self.cell_types #it's the remainder
        #and now we index into the filtered indices with this if we will be filtering
        # if self.filter:
        #     celltype_idx = self.filtered_indices[celltype_idx] #this gives us what the actual index in the large array is
        #the above is for if we used the old hash table lookup method, it's much more efficient to just have cut down the columns in the first place
        
        #here we load in the celltype tokens
        #then after self.rc_aug we append the tokens to the left
        #then finally we also need to load in the target which is based on the cCRE and the id, which we did load in
        
        cell_seq = self.cell_type_array[celltype_idx, 1] #the unique permutation
        
        #for the sequence index, we find the number of times it goes into that number of cell types
        seq_idx = int(idx/self.cell_types) #int will round down, equivalent to a//b for positive numbers
        
        seq = self.array[seq_idx,2]
        # cCRE_id = self.array[seq_idx][0]
        # interval_length = len(seq)
        if self.uppercase:
            seq = seq.upper()
            
        #now check seqlen and make sure that it is not more than 1018 so have space for our cell type ones
        seqlen = len(seq)
        if seqlen> 1018:
            total_to_remove = seqlen - 1018
            # Characters to remove from the left side
            # If total_to_remove is odd, an extra character is removed from the left
            left_to_remove = total_to_remove // 2 + total_to_remove % 2
            
            # Characters to remove from the right side
            right_to_remove = total_to_remove // 2
            
            # Slicing the sequence to achieve the desired length
            # left_to_remove is subtracted from the start index and right_to_remove from the end index
            seq = seq[left_to_remove:seqlen - right_to_remove]


        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)
            
        #let the tokenizer pad things
        
        #now we need to add our cell type tokens
        #we will add them to the left
        seq = cell_seq + seq

        if self.tokenizer_name == 'char': #will stick with this for sure
            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        # print(seq)
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        #now we need to load in the target
        # col = self.cell_type_array[celltype_idx,0] #this index should be completely identical to the celltype_idx
        col = celltype_idx #should test this first!
        cCRE_id = self.array[seq_idx][0] #get the id from the array
        #nowo find the corresponding index for the row
        # row = np.where(self.cCRE_identities == cCRE_id)[0][0] #finds what the cell type matrix row has that cCRE id
        #use the much faster version by doing
        row = self.cCRE_dict[cCRE_id]
        target = self.cell_dnase_levels[row,col]
        
        data = seq[:-1].clone()  # don't need the eos
        # target = torch.FloatTensor([target]) #double causes an error

        if self.classification:
            out_class = 1
            if target == -10:
                out_class = 0 #0 means it's closed, 1 means it's open
            target = torch.FloatTensor([target])
            class_target = torch.LongTensor([out_class])
            return data, (class_target, target)
            
        else:
            target = torch.FloatTensor([target]) #double causes an error
            return data, target