

"""

This is a general dataset which uses splits based on a sequences.txt file like enformer. It will be general and can use zarr parallel to load fast, or it can simply load it in if it's a smaller part of the dataset
The datasets are chunked in a way that loading multiple cell types won't be too hard! We can apply processing of like clipping and scaling and the unmappable regions
Loading in makes it very slow initially, so is highly discouraged, but can do arbitrary cell types if so, and will be the same speed

If you have presaved it to a zarr array in this folder, then it will simply access that. So cell type could be like GM12878_DNase or K562_ChIP_TF or whatever, and it willl access only that dat aloaded very efficiently since we can just load it into memory


"""

import numpy as np
import zarr
import os
import torch
import pandas as pd
from random import random

# splits_dict = {
#     'train': [1,2,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22],
#     'val': [3,13],
#     'test': [4,14]
# }

def process_data(seq, blacklist, unmap, scale, clip, clip_soft, blacklist_pct=0.5, umap_clip=0.5):
    '''This function will process the data, it does the basenji processing where you do blacklist clipping
    Then skip pooling (can do it later, only might affect unmappable clipping, but if doing mean pooling won't affect clipsoft and clip),
    Then scale and clip soft then clip, 
    then extereme clip,
    then change unmappable regions
    
    Args:
        seq: the sequence to process
        blacklist: the already loaded values form the blacklist npz file that was loaded
        unmappable_npz: the already loaded values from the unmappable regions npz file that was loaded
        scale: the scale to apply (from targets.txt)
        clip: the clip to apply (from targets.txt)
        clip_soft: the soft clip to apply (from targets.txt)
        blacklist_pct: the percentage to clip the blacklist to
        
    
    '''
    #first find blacklist values and if they overlap, clip them to baseline
    if blacklist.sum() > 0:
        baseline_cov = np.percentile(seq, 100*blacklist_pct, axis=1)
        seq_blacklist = seq[:,blacklist==1]
        seq_blacklist = np.minimum(seq_blacklist, baseline_cov[:,None])
        seq[:,blacklist==1] = seq_blacklist
    
    #now scale and clip soft then clip
    seq = seq * scale
    clip_mask = (seq > 32)
    if clip_mask.sum() > 0:
        seq[clip_mask] = np.sqrt(seq[clip_mask] - clip_soft+1) + clip_soft-1
        seq = np.clip(seq,0,clip)
    
    #now extreme clip, could be extreme values like lots of 0.0002 but then one that is 13. Not good potentially
    extreme_clip = np.percentile(seq, 100*0.9999, axis=1)
    seq = np.minimum(seq, extreme_clip[:,None])

    #and finally, use unmappable regions
    if unmap.sum() > 0:
        seq_target_null = np.percentile(seq, q=[100*umap_clip], axis=1)
        seq_unmap = seq[:,unmap==1]
        seq_unmap = np.minimum(seq_unmap, seq_target_null[0,:,None])
        seq[:,unmap==1] = seq_unmap

    return seq

def mask_seq(seq, mask_pct=0.15, replace_with_N=True, span=1, stype='category', weights=None):
    """This function will mask the sequence data, it does the BERT masking where you do 80% truly masked, 10% random, 10% unchanged
    Note that for random replacement, it cannot be the N token, sicne it's very rare anyways, always random nucleotide!
    Args:
        seq: the sequence to mask, this is a tensor of shape (length, N) if categorical, or (length,1) if continuous, N is the number of classes (5 for ohe nucleotide data)
        mask_pct: the percentage of the sequence to mask, default is 0.15 or 15%
        replace_with_N: whether to allow random replacement of values with N (for one hot encoded data). Keep True for other categorical data
        span: the size of the span to mask, default is 1, so it masks every element independently, but can be larger to mask chunks of size span
        stype: the type of sequence, 'category' for categorical like ohe nucleotide data and 'continuous' for continuous like raw accessibility data, default is 'category'
        weights: the weights to use for weighting regions like peaks. default is None, must be a tensor of shape (length,) to weight the peaks higher for masking. can be the same as the seq itself
    Returns:
        seq: the masked sequence, this is a tensor of shape (length, N+1) if categorical or (length, 2) if continuous, where the last column is the mask track (only tells if masked, some are random or unchanged)
        seq_unmask: the unmasked sequence, this is a tensor of shape (length, N+1) if categorical or (length, 2), where the last column is the mask track (tells all elements that have been changed or goign to evaluate)
    """
    # print(seq.shape)
    if len(seq.shape) > 1:
        if replace_with_N:
            random_max = 5
        else:
            random_max = 4
        
    
    num_elements = seq.shape[0]//span #chunks into chunks of size span
    
    if len(seq.shape) == 1: #if it's just a 1D tensor, we need to add a dimension for the mask track
        seq = seq.unsqueeze(1) #so now it's shape length x 1, so we can concatenate other things
    
    # Create a probability vector (one per token) and sample which tokens to mask
    probability_matrix = torch.full((num_elements,), mask_pct) #size of length, defines for each element if we mask it or not
    
    #we can also weight peak regions more
    if weights is not None:
        weights = weights.squeeze()
        assert weights.ndim == 1, f"weights must be a 1D tensor, got {weights.shape}"

        #Trim weights to match the required size
        if weights.shape[0] % span != 0:
            #remove values until it is the right size
            weights = weights[:num_elements*span]
        
        #compute mean over spans
        weights = weights.view(num_elements, span).mean(1) #average the weights over the span, so now it's size length
        
        #normalize weights to have range 0.5 to 1.5
        weights = torch.log(weights + 1) #log transform to reduce the scale of values
        weights = (weights - weights.min()) / (weights.max() - weights.min()) + .5 #normalize to have different range, downweights small values, upweights large ones to almost 3x

        #scale probability matrix by weights
        probability_matrix = probability_matrix * weights #so now we have a weighted probability matrix, so we can mask more in peak regions
        #and scale up probability matrix so that the mean is mask_pct
        probability_matrix = probability_matrix / probability_matrix.mean() * mask_pct #so now we have a weighted probability matrix, so we can mask more in peak regions
        
        #clip to make sure between 0 and 1
        probability_matrix = torch.clamp(probability_matrix, min=0, max=1) #clip to make sure between 0 and 1     
    
    masked_indices = torch.bernoulli(probability_matrix).bool() #finds which indices to mask, so shape is length, and is True or False for each index

    # Get positions that were chosen to be masked
    all_mask_positions = torch.nonzero(masked_indices).squeeze()*span #squeeze to remove the extra dimension, and multiply by span to get the actual positions in the original sequence
    num_masked = all_mask_positions.numel()
    
    # Determine counts for the three groups: 80% truly masked, 10% random, 10% unchanged
    num_mask = int(0.8 * num_masked)
    num_random = int(0.1 * num_masked)
    # To avoid rounding issues, let the remaining be unchanged
    # num_unchanged = num_masked - num_mask - num_random
    
    # Shuffle the masked positions to randomly assign each to a category
    permuted = all_mask_positions[torch.randperm(num_masked)]
    mask_positions = permuted[:num_mask]  # 80%: replace with mask token
    random_positions = permuted[num_mask:num_mask+num_random]  # 10%: random token
    unchanged_positions = permuted[num_mask+num_random:]  # 10%: leave as is

    if span > 1:
        masked_indices = masked_indices.repeat_interleave(span) #so now we have a mask for every element in the span, so size length again
        #and append zeros until the size of seq
        extra = seq.shape[0] % span
        if extra > 0:
            masked_indices = torch.cat([masked_indices, torch.zeros(extra, dtype=torch.bool)]) #so now we have a mask for every element in the span, so size length again
        
        #now for each of the positions, we need to expand and then make masking apply per index
        mask_positions = (mask_positions.unsqueeze(1) + torch.arange(span)).flatten()
        random_positions = (random_positions.unsqueeze(1) + torch.arange(span)).flatten()
        unchanged_positions = (unchanged_positions.unsqueeze(1) + torch.arange(span)).flatten()
        #and now they are grouped and we can just deal with them

    # Append the mask track to the sequence, resulting in a tensor of shape [seq_len, 6], or [seq_len, 2] if acc data
    # where the last column is the mask track
    seq_unmask = torch.cat([seq, masked_indices.unsqueeze(1).float()], dim=1) #so now seq_unmask is shape length x 6, where 6 is the 5 one hot classes and the mask
    
    seq_masked = seq_unmask.clone()  # Create a copy to modify, note that the mask track should be 0 for ones where it's not masked but is random or unchanged
    seq_masked[mask_positions, :-1] = 0  # Set to zero for every class but the last (tells it it's masked)
    
    if stype == 'category':
        # print(f'random_positions shape: {random_positions.shape}, seq shape: {seq.shape}')
        if replace_with_N:
            random_max = seq.shape[1]
        else:
            random_max = seq.shape[1] - 1
        random_tokens = torch.randint(0, random_max, (random_positions.numel()//span,)) #generate random values for each position
        random_one_hot = torch.zeros((random_positions.numel()//span, seq.shape[1])) #one hot encode them
        random_one_hot.scatter_(1, random_tokens.unsqueeze(1), 1.0)
        #now repeat with the span
        random_one_hot = random_one_hot.repeat_interleave(span, dim=0) #so now we have a one hot for each position in the span
        seq_masked[random_positions, :seq.shape[1]] = random_one_hot #assign them to the set positions
        
    elif stype == 'continuous':
        #for accessibility, we will select random values from somewhere else in the sequence and then slightly shift and noise them
        #get a random value between 0 and len(seq)-span
        rand_start = torch.randint(0, seq.shape[0]-span, (random_positions.numel()//span,)) #definitely divisble by span since it was extended by size span
        rand_idx = (rand_start.unsqueeze(1) + torch.arange(span)) #so now we have a random index for each of the random positions, and we can just select from there
        rand_vals = seq.squeeze(1)[rand_idx] #get the values from the sequence at those random positions, so now we have a random value for each of the random positions
        #and we can add some noise to it, so we can just add a small random value to it. Noise will be values between -0.1 and 0.1
        rand_vals_mean = rand_vals.mean(1, keepdim=True) #get the mean of the random values for each position, keeps the dim so we can broadcast it
        noise = torch.randn(rand_vals.shape) * rand_vals_mean * 0.1 #gaussian noise with std of 0.1 times the mean of the random values, so we can add some larger nosie to larger values
        rand_vals = torch.clamp((rand_vals + noise).flatten(), min = 0) #make sure values are at least 0, else obvious there's noise in the region
        #and now set the values
        seq_masked[random_positions, 0] = rand_vals #set the values to the random values with noise, so now we have a random value for each of the random positions
    
    else:
        raise ValueError("stype must be either 'category' or 'continuous'")
    
    #and remove the masked value, it doesn't know it's masked
    seq_masked[random_positions, -1] = 0
    
    #and we remove the mask token from the unchanged value
    seq_masked[unchanged_positions, -1] = 0
    # seq = seq_masked #now we have the masked sequence, so we can use this for the rest of the processing
    
    return seq_masked, seq_unmask #return the masked sequence and the unmasked sequence, so we can use it for the rest of the processing
    
"""another idea is to basically define a list of numbers of random size, from negative binomial mean 500. Varying length for a lot of them. poisson if want more tight
Then you assign a random order, then the last 15% are defined as masked. You mask those, then for the genome you can have a list of 0 or 1 for every element, but it's in spans of whatever was defined by the negative binomial
This can be done at the initialization of the dataloader, and this resets each epoch"""

def open_data(data_path, load_in=False):
    if data_path.endswith('.zarr'):
        data = zarr.open(data_path, mode='r')
        if load_in:
            data = {key: np.array(data[key]) for key in data}
    else:
        #load in the data from np array
        if load_in:
            with np.load(data_path) as data:
                data = {key: np.array(data[key]) for key in data}
        else:
            data = np.load(data_path)
    return data
    
    

def coin_flip():
    return random() > 0.5

class GeneralDataset():
    def __init__(
        self,
        split: str,
        data_path: str,
        length: int = None,
        genome_seq_file: str = '/data1/lesliec/sarthak/data/chrombpnet_test/hg38_tokenized.npz',
        preprocess: bool = False,
        unmappable_file: str = '/data1/lesliec/sarthak/data/borzoi/umap.npz',
        blacklist_file: str = '/data1/lesliec/sarthak/data/borzoi/blacklist.npz',
        sequences_bed_file: str = '/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed',
        shift_sequences: int = 0, #whether to add some noise to the sequences and how much maxx
        load_in: bool = False,
        targets_txt_file: str = '/data1/lesliec/sarthak/data/borzoi/targets.txt',
        one_hot: bool = True,
        pool: int = 1,
        pool_type: str = 'mean',
        return_target: bool = True,
        rc_aug: bool = False,
        crop_output: int = 0,
        mlm: int = None, #masked language modeling
        acc_mlm: int = None, #mlm on accessibility
        acc_type: str = 'continuous', #type of accessibility, either 'category' or 'continuous'. Category is for one hot encoded data, continuous is for raw accessibility data
        acc_mask_size: int = 500,
        pair_mask: bool = False,
        replace_with_N: bool = False, #whether to allow random replacement of values with N
        acc_threshold: float = 1, #this is the threshold for accessibility, so anything above this is accessible
        weight_peaks: bool = False, #whether to weight the peaks more for masking, default is False. Just uses the accessibility values themselves
        evaluating: bool = False, #if evaluating, removes all augmentations!
    ):
        """
        General dataset class, relies on zarr or np array data which is in chromosome format, and a sequences bed file
        
        Args:
            split (str): the split to use, train, val, test
            genome_seq_file (str): the path to the genome sequence file
            preprocess (bool): whether to preprocess the data or not, like deal with unmappable values and things like that
            unmappable_file (str): the path to the unmappable regions file
            blacklist_file (str): the path to the blacklist file
            data_path (str): the path to the data file, can be npz or zarr, but must be in format of chromosome dict and then N x chrom_len where N is a number at least 1
            load_in (bool): whether to load in the data or not, this is slow and should be avoided if possible, but if it's not zarr only load in is possible
            sequences_bed_file (str): the path to the sequences bed file, this is used to filter the data to the split
            length (int): the length of the sequences to use, if None, will use the full length of the sequences in the bed file
            crop_output (int): the amount to crop the output by, crops both sides by that amount before pooling, if 0, will not crop
            mlm (int): the amount of masked language modeling to do, if 0, will not do any mlm, if > 0, will randomly mask that percent of the sequence
            acc_mask (int): the amount of accessibility masking to do, if 0, will not do any masking, if > 0, will randomly mask that percent of the sequence
            acc_mask_size (int): the size of the accessibility mask, like how large the continuous chunks should be so it doesn't just interpolate
            pair_mask (bool): whether to do pair masking or not, if True, will mask accessibility and nucleotides at the same location

        """
        #so we will create a general dataset that can easily access the data
        
        self.evaluating = evaluating #if evaluating, we will not do any augmentations or random shifts
        if self.evaluating:
            #no augmentations or random shifts
            rc_aug = False
            shift_sequences = 0
        
        self.split = split
        self.genome_seq_file = genome_seq_file
        self.preprocess = preprocess
        self.unmappable_file = unmappable_file
        self.blacklist_file = blacklist_file
        self.data_path = data_path
        self.pool = pool
        self.pool_type = pool_type
        self.length = length
        self.rc_aug = rc_aug
        self.shift_sequences = shift_sequences
        self.return_target = return_target
        self.one_hot = one_hot
        self.crop_output = crop_output
        self.mlm = mlm
        self.acc_mlm = acc_mlm
        self.acc_mask_size = acc_mask_size
        self.pair_mask = pair_mask
        self.replace_with_N = replace_with_N
        self.load_in = load_in
        self.acc_type = acc_type #this is the type of accessibility, either 'category' or 'continuous', default is 'category'
        self.acc_threshold = acc_threshold #this is the threshold for accessibility, so anything above this is accessible
        self.weight_peaks = weight_peaks #whether to weight the peaks more for masking, default is False. Just uses the accessibility values themselves
        # self.complement_array = np.array([3, 2, 1, 0, 11]) #this is the complement array for the rc augmentation, so A->T, C->G, G->C, T->A, 11 stays as 11

        #and access the genome seq file
        self.genome = open_data(genome_seq_file, load_in)
        
        if self.preprocess:
            self.blacklist = open_data(blacklist_file, load_in)
            self.unmappable = open_data(unmappable_file, load_in)
        
        self.data = open_data(data_path, load_in)
        
        sequences = pd.read_csv(sequences_bed_file, sep='\t', header=None)
        #filter to the split
        if self.split in sequences[3].values:
            self.sequences = sequences[sequences[3] == self.split].reset_index(drop=True)
        elif self.split == 'val':
            # Handle the case when self.split is not found, because we use val but enformer uses valid
            self.split = 'valid'
            self.sequences = sequences[sequences[3] == self.split].reset_index(drop=True)
        else:
            raise ValueError(f"Split {self.split} not found in sequences bed file")

        #and for rc aug
        max_key = 11
        self.complement_array = np.zeros(max_key + 1, dtype=int)
        complement_map = {"7": 10, "8": 9, "9": 8, "10": 7, "11": 11}
        for k, v in complement_map.items():
            self.complement_array[int(k)] = v
            
        # if self.acc_type == 'category':
            #we have to define a threshold for accessible or not
            #let's just start by setting it to the 50th percentile of chromosome 1
            # self.acc_threshold = np.percentile(self.data['chr1'], 50, axis=1) #this is the threshold for accessibility, so anything above this is accessible
            # self.acc_threshold
            #we will just usse a manual threshold for now, but other approaches of course exxist!!

        #now we will find some way to deal with or find specific things with the target txt file, but for now we can ignore this
        #TODO : find a way to deal with the targets txt file, this is used to find the scale and clip values for the data, but we can ignore it for now
        
    def __len__(self):
        #return the length of the dataset, this is the number of sequences in the split
        return len(self.sequences)
    
    def __getitem__(self, index):
        """Get the item at the index, this will return the sequence and the data for that sequence, and also do any processing if needed
        Args:
            index (int): the index of the item to get
        Returns:
            seq (torch.Tensor): the sequence for that index, this is a tensor of shape (N, length) if one_hot is True, else (length,), N is 5 normally, 6 if masking
            targets (torch.Tensor): the targets for that index, this is a tensor of shape (N, length) where N is the number of targets (minimum 1). Can be 2N or more if masking and how many bins you ahve
            seq_unmask (torch.Tensor): the unmasked sequence for that index, this is a tensor shape (length, N) if you do masking, else it's torch.empty(0). but is the correct values prior to masking. Also specifies if it's been masked/altered
            acc_umask (torch.Tensor): the unmasked accessibility for that index, this is a tensor shape (length,N) if you do masking, else it's torch.empty(0). Also specifies if it's been masked/altered
        
        """
        if not self.load_in: #each worker gets its own so modifying self is fine
            self.genome = open_data(self.genome_seq_file, load_in=False)
            self.data = open_data(self.data_path, load_in=False)
            if self.preprocess:
                self.blacklist = open_data(self.blacklist_file, load_in=False)
                self.unmappable = open_data(self.unmappable_file, load_in=False)
        
        seq_unmask = torch.empty(0)
        acc_umask = torch.empty(0)
        
        chrom, start, end, split = self.sequences.iloc[index]
        if self.shift_sequences > 0:
            shift = np.random.randint(-self.shift_sequences, self.shift_sequences+1)
            start = start+shift
            end = end+shift
        
        #initialize padding if needed
        leftpad = np.zeros(0)
        rightpad = np.zeros(0)
        diff = self.length - (end - start)
        if self.length is not None: #if we have a length and the length is greater than the sequence, we need to pad it
            start = start - diff // 2
            end = end + diff // 2
            if start < 0:
                leftpad = np.ones(-start)*11
                start = 0
            chromlen = self.genome[chrom].shape[0]
            if end > chromlen:
                rightpad = np.ones(end-chromlen)*11
                end = chromlen
            seq = np.concatenate([leftpad.astype(np.int8), self.genome[chrom][start:end], rightpad.astype(np.int8)]) #pad with 11s if needed, but make sure they're int too
                
                
        else: #just grab the sequence and data directly
            seq = self.genome[chrom][start:end]
        
        if self.rc_aug and coin_flip():
            seq = self.complement_array[seq[::-1]]
            flip = True
            self.last_flip = True
        else:
            flip = False
            self.last_flip = False
        
        seq = torch.LongTensor(seq)
        
        if self.one_hot:
            x = seq
            # x_onehot = torch.nn.functional.one_hot((x-7)%4, num_classes=4).float().transpose(1, 0) #need to make sure it is the right order, so now is shape 4xseq_len
            # if 11 in x:
            #     x_onehot[:, x == 11] = 0 #makes it so that all 11s are a row of all 0s
            x_onehot = torch.nn.functional.one_hot(x-7, num_classes=5).float() #N is its own class, also no transpose, so shape is seq_lenx5
            seq = x_onehot
        
        #now do the masking, just get the indices for length, can apply along length dimension
        if self.mlm is not None:
            if not self.one_hot:
                raise ValueError("MLM only works with one hot encoding for now, but can easily be generalized to this")

            seq, seq_unmask = mask_seq(seq, mask_pct=self.mlm, replace_with_N=self.replace_with_N) #this will mask the data and return the unmasked data as well, so we can use it for the rest of the processing
            # seq_unmask = seq_unmask.transpose(1, 0) #transpose it to be 6 x length, so we can use it for the rest of the processing
        
        seq = seq.transpose(1, 0) #transpose it to be 6 x length, so we can use it for the rest of the processing
        
        if not self.return_target:
            return seq, seq_unmask
        
    
        #and get the data
        data = np.concatenate([leftpad[None]*0, self.data[chrom][:,start:end], rightpad[None]*0], axis=1) #multiply by 0 to set it as 0 since it's not tokenized
        #so padsd if needed, and then pads 0s
        #broadcast along the dimension
        #now is shape N x length, where N is the number of targets
        data = data.transpose(1, 0) #now is shape length x N, so we can do the pooling and stuff
        
        if self.preprocess:
            #now we will process the data
            raise NotImplementedError("Processing of data is not implemented yet, below code is incorrect or uses fake functions")
            # blacklist = np.concatenate([leftpad*0, self.blacklist[chrom][start:end], rightpad*0])
            # unmap = np.concatenate([leftpad*0, self.unmappable[chrom][start:end], rightpad*0])
            # scale, clip, clip_soft = self.get_target_params(chrom)
            # data = process_data(data, self.blacklist[chrom], self.unmappable[chrom], scale, clip, clip_soft)
        
        targets = torch.FloatTensor(data)
        if flip:
            targets = targets.flip(dims=[0]) #flip the targets if we flipped the seq
        
        if self.crop_output > 0:
            #crop the output if needed
            targets = targets[self.crop_output:-self.crop_output]
            #do cropping here and after because otherwise it becomes a pain, but faster to not load in all these values and crop, but padding is easier
            
        if self.pool > 1:
            #first reshape
            if targets.shape[0] % self.pool != 0:
                raise ValueError('Pool size must divide sequence length')
            
            targets = targets.view(targets.size(0) // self.pool, self.pool, targets.size(1))
            if self.pool_type != 'mean':
                raise NotImplementedError('Only mean pooling implemented')
            targets = targets.mean(dim=1)
            
        if self.acc_mlm is not None:
            #now we will do the accessibility masking, this is done on the targets, so we can mask the targets and the sequence at the same time
            if targets.shape[1] > 1:
                raise NotImplementedError("I don't know yet how to broadcast the masking effectively, can just do a for loop but maybe some more efficient way?")
            else:
                targets = targets.squeeze(1)
            
            if self.weight_peaks:
                weights = targets #this will be used to weight the peaks more for masking, so we can use the accessibility values themselves
            else:
                weights = None

            if self.pair_mask:
                raise NotImplementedError("Pair masking is not implemented yet")
            else:
                if self.acc_type == 'category':
                    #we have to bin the data
                    targets = (targets > self.acc_threshold).long() #this will bin the data, so anything above the threshold is 1, else 0
                    #and ohe it
                    targets = torch.nn.functional.one_hot(targets, num_classes=2).float() #now input is length x 2
                
                targets, acc_umask = mask_seq(targets, mask_pct=self.acc_mlm, span=self.acc_mask_size, stype=self.acc_type, weights=weights) #mask the acc data
                # acc_umask = acc_umask.transpose(1, 0) #transpose it to be 2 x length, so we can use it for the rest of the processing
        
        #now transpose the targets
        targets = targets.transpose(1, 0)
        
        return (seq, targets), (seq_unmask, acc_umask) #return the sequence, targets, unmasked sequence and unmasked accessibility if we do masking, else empty tensors
    
    def expand_seqs(self,chr,start,stop):
        """This function will expand the sequences to include a new sequence centered around whatever you want
        Args:
            chr (str): the chromosome to expand
            start (int): the start position to expand to
            stop (int): the stop position to expand to
        Returns:
            seq (np.array): the expanded sequence, this is a numpy array of shape (length,) where length is the length of the sequence
        """
        #expand the sequences to the start and stop positions
        
        new_row = pd.DataFrame([[ chr, start, stop, self.split ]], columns=self.sequences.columns)
        self.sequences = pd.concat([self.sequences, new_row], ignore_index=True)
        idx = self.sequences.index[-1] #get the index of the new row
        return idx
        
        
        
        

if __name__ == "__main__":
    #example usage
    dataset = GeneralDataset(
        split='train',
        preprocess=False,
        data_path='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz',
        sequences_bed_file='/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed',
        length=None
    )
    # print(dataset.sequences)
    seq,label = dataset[0]
#here's some oexample things you can use
'''
from src.dataloaders.datasets.general_dataset import GeneralDataset
dataset = GeneralDataset(
    split='train',
    preprocess=False,
    data_path='/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz',
    sequences_bed_file='/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed',
    length=524288
)
'''