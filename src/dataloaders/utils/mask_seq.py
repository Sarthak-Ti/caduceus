import torch

def mask_seq(seq, mask_pct=0.15, replace_with_N=True, span=1, stype='category', weights=None, upweight_fraction=0.95, mask_only=False, mask_tie=1, independent_tracks=False, max_scale=3, weight_floor=0.1, log_weights=False, binary_score_threshold=None, neg_maskrate=None, max_neg_to_pos_ratio=0.5, minimum_neg_masks = 30, return_probability_matrix=False):
    """This function will mask the sequence data, it does the BERT masking where you do 80% truly masked, 10% random, 10% unchanged
    Note that for random replacement, it cannot be the N token, sicne it's very rare anyways, always random nucleotide!
    Args:
        seq: the sequence to mask, this is a tensor of shape (length, N) if categorical, or (length,1) if continuous, N is the number of classes (5 for ohe nucleotide data)
        mask_pct: the percentage of the sequence to mask, default is 0.15 or 15%
        replace_with_N: whether to allow random replacement of values with N (for one hot encoded data). Keep True for other categorical data
        span: the size of the span to mask, default is 1, so it masks every element independently, but can be larger to mask chunks of size span
        stype: the type of sequence, 'category' for categorical like ohe nucleotide data and 'continuous' for continuous like raw accessibility data, default is 'category'
        weights: the weights to use for weighting regions like peaks. default is None, must be a tensor of shape (length,) to weight the peaks higher for masking. can be the same as the seq itself, and can be a binary tensor
        weight_min: the minimum weight to use for weighting, i.e what a value of 0 weight does to the probability, default is 0.3
        weight_max: the maximum weight to use for weighting, i.e what a value of 1 weight does to the probability, default is 3
        mask_only: whether to do the 10% unchanged and 10% random replacement, default is False. If True, will only do the 100% truly masked and leave the rest unchanged
        mask_tie: how much masking is tied across categories. 1 means fully tied, so all tracks are masked the same, 0 means fully indepdendent masking across categories. If true, returns slightlly different values
        independent_tracks: whether to mask independent tracks separately, only used if stype is 'category' and mask_tie is 0. If True, masks each track independently, if False, masks all tracks the same
        max_scale: the maximum scaling factor for continuous weights, default is 3, so prob of 0.15 can at most be 0.45
        binary_score_threshold: the threshold to use for binary weights, only used if weights are provided. If provided, weights above this threshold are considered positive class, below are negative class
        neg_maskrate: the negative class mask rate, only used if weights are provided and binary weights are used. If None, will default to 1/10 of the positive class mask rate
        max_neg_to_pos_ratio: the maximum percentage of negative elements to mask relatived to positive. So 0.5 means can mask 1 negative for every 2 positives. Only used if weights are provided and binary weights are used
    Returns:
        seq: the masked sequence, this is a tensor of shape (length, N+1) or N*2 if mask_tie is less than 1 if categorical or (length, 2) if continuous, where the last column is the mask track (only tells if masked, some are random or unchanged)
        seq_unmask: the unmasked sequence, this is a tensor of shape (length, N+1) or N*2 if categorical or (length, 2), where the last column is the mask track (tells all elements that have been changed or goign to evaluate)
    """
    
    if len(seq.shape) == 1: #if it's just a 1D tensor, we need to add a dimension for the mask track
        seq = seq.unsqueeze(1) #so now it's shape length x 1, so we can concatenate other things
    
    extra_append = 0
    if seq.shape[0]%span != 0:
        #we append on values
        remainder = seq.shape[0] % span
        extra_append = span - remainder
        seq = torch.cat([seq, torch.zeros((extra_append,seq.shape[1]), dtype=torch.float)]) #so now we can have a mask for every element in the span, so size length again
    num_elements = seq.shape[0]//span #chunks into chunks of size span

    # Create a probability vector (one per token) and sample which tokens to mask
    probability_matrix = torch.full((num_elements,), mask_pct, dtype=torch.float64) #size of length, defines for each element if we mask it or not
    
    #we can also weight some regions like peaks more
    if weights is not None:
        assert mask_tie == 1, "Weighting with weights is only supported for mask_tie=1 currently"
        weights = weights.squeeze()
        assert weights.ndim == 1, f"weights must be a 1D tensor, got {weights.shape}"
        
        if binary_score_threshold is not None:
            weights = (weights >= binary_score_threshold).to(torch.bool) #convert to binary weights based on threshold
        
        #we have 2 modes for the weights, can be binary values which can be applied to peaks and nonpeaks, or continuous values between 0 and 1
        #check if binary based on the stored datatype or if it only has 0/1
        isbinary = weights.dtype == torch.bool or (weights.unique().numel() == 2 and torch.allclose(weights.unique(), torch.tensor([0., 1.])))
        #now turn weights to float for processing
        weights = weights.to(torch.float)

        #append on weights to match the required size
        if weights.shape[0] % span != 0:
            #remove values until it is the right size
            # weights = weights[:num_elements*span]
            weights = torch.cat([weights, torch.zeros((extra_append,), dtype=weights.dtype)]) #so now we can have a mask for every element in the span, so size length again
            # print(f'adjustting weights to {weights.shape}')
        
        #compute mean over spans
        weights = weights.view(num_elements, span).mean(1) #average the weights over the span, so now it's size length

        if isbinary: #basically do class specific bernoulli masking, default to background is 1/1 probability, but can be lowered further
            num_pos = (weights >= 0.5).sum().item()
            #manually set the positive class to the masking rate, set the negative class to at most neg_maskrate if provided, else is 1/10 the positive class as the maximum
            if neg_maskrate is None:
                neg_maskrate = mask_pct / 10
        
            num_masked_pos = int(mask_pct * num_pos)
            # num_masked_neg = int(neg_maskrate * (num_elements - num_pos)) #theoretical number of negatives to mask, but we need to find the actual rate such that 
            neg_maskrate_adjusted = min(neg_maskrate, (max_neg_to_pos_ratio * num_masked_pos) / max(1, (num_elements - num_pos))) #adjusted negative mask rate based on the maximum allowed
            neg_maskrate_adjusted = max(neg_maskrate_adjusted, minimum_neg_masks / max(1, (num_elements - num_pos))) #ensure about at least minimum negative masks
            # print(neg_maskrate_adjusted)
            # print( torch.where(weights >= 0.5, torch.full_like(weights, mask_pct), torch.full_like(weights, neg_maskrate_adjusted)))
            probability_matrix = torch.where(weights >= 0.5, torch.full_like(weights, mask_pct), torch.full_like(weights, neg_maskrate_adjusted))


        else:
            if log_weights:
                weights = torch.log(weights + 1) #log transform to reduce the scale of values
            #scale weights to be between 0 and 1
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6) #normalize to have different range, downweights small values, upweights large ones to 1
            # If floor is 0.1, the lowest signal has 10% the probability of the highest signal.
            # This prevents the "Zero Probability" trap.
            floor = weight_floor
            weights = floor + (1 - floor) * weights
            #scale probability matrix by weights
            probability_matrix = probability_matrix * weights #so now we have a weighted probability matrix, so we can mask more in peak regions
            #and scale up probability matrix so that the mean is mask_pct
            upscale = min(mask_pct / (probability_matrix.mean()+1e-6), max_scale)
            probability_matrix = probability_matrix * upscale
            # probability_matrix = probability_matrix / probability_matrix.mean() * mask_pct #so now we have a weighted probability matrix, so we can mask more in peak regions

        
        #clip to make sure between 0 and 1
        probability_matrix = torch.clamp(probability_matrix, min=0, max=1) #clip to make sure between 0 and 1   
    
    masked_indices = torch.bernoulli(probability_matrix.float()).bool() #finds which indices to mask, so shape is length, and is True or False for each index
    #if nothing is masked, then just return it with nothing masked, 1 element also just ignor eit, too few
    if masked_indices.sum() == 0:
        if mask_tie < 1 or independent_tracks:
            seq_unmask = torch.zeros((seq.shape[0], seq.shape[1]*2), dtype=torch.float)
            seq_unmask[:, :seq.shape[1]] = seq
        else:
            seq_unmask = torch.cat([seq, torch.zeros((seq.shape[0], 1)).float()], dim=1)
        #in case we had extra elements appended, don't make shape 524500, keep 524288
        seq_unmask = seq_unmask[:-extra_append] if extra_append > 0 else seq_unmask
        seq_masked = seq_unmask.clone()
        if return_probability_matrix:
            return seq_masked, seq_unmask, probability_matrix
        return seq_masked, seq_unmask

    if mask_tie < 1 or independent_tracks: #have to implement own logic and loop, this is for the dependent tracks, otherwise unclear how to implement it, can make it like 0.999? should actually work for mask only true if mask tie is 1, just means only 1 mask channel tho? so it won't work
        assert mask_only, "Not implemented for mask_only = False yet"
        all_masked_indices = masked_indices.unsqueeze(1).repeat(1, seq.shape[1])
        # print(all_masked_indices.shape) #shape is length//span x num_categories
        #now we find which ones will change based on mask_tie and only changing the masks to be unmasked
        change = masked_indices & (torch.rand(masked_indices.shape) < (1 - mask_tie))
        # print(change.shape, change) #shape is length//span
        flip = change.unsqueeze(1) & (torch.rand_like(all_masked_indices.float()) < 0.10)
        all_masked_indices = all_masked_indices & (~flip)
        #and expand to full size
        all_expanded_masked_indices = all_masked_indices.repeat_interleave(span, dim=0)
        # if extra > 0:
        #     all_expanded_masked_indices = torch.cat([all_expanded_masked_indices, torch.zeros((extra, seq.shape[1]), dtype=torch.bool)]) #so now we have a mask for every element in the span, so size length again
        
        #now we can create the masked and unmasked sequences
        seq_unmask = torch.zeros((seq.shape[0], seq.shape[1]*2), dtype=torch.float)
        seq_unmask[:, :seq.shape[1]] = seq
        seq_unmask[:, seq.shape[1]:] = all_expanded_masked_indices.float()
        seq_masked = seq_unmask.clone()

        seq_masked[:, :seq.shape[1]] = seq_masked[:, :seq.shape[1]] * (~all_expanded_masked_indices).float()
        if span > 1 and extra_append > 0:
            seq_masked = seq_masked[:-extra_append]
            seq_unmask = seq_unmask[:-extra_append]
        
        if return_probability_matrix:
            return seq_masked, seq_unmask, probability_matrix
        return seq_masked, seq_unmask
    

    # Get positions that were chosen to be masked
    # print(f'masked_indices:{masked_indices}, and sum: {masked_indices.sum()}')
    all_mask_positions = torch.nonzero(masked_indices).squeeze()*span #squeeze to remove the extra dimension, and multiply by span to get the actual positions in the original sequence
    #and unsqueeze if the shape is 0D
    if all_mask_positions.ndim == 0:
        all_mask_positions = all_mask_positions.unsqueeze(0)
    # print(f'all_mask_positions.shape: {all_mask_positions.shape}')
    num_masked = all_mask_positions.numel()
    
    # Determine counts for the three groups: 80% truly masked, 10% random, 10% unchanged
    num_mask = int(0.8 * num_masked)
    num_random = int(0.1 * num_masked)
    # To avoid rounding issues, let the remaining be unchanged
    # num_unchanged = num_masked - num_mask - num_random
    
    # Shuffle the masked positions to randomly assign each to a category
    # print(num_masked)
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
    
    if mask_only:
        #now forcibly mask the rest
        seq_masked[unchanged_positions, :-1] = 0  # Set to zero for every class but the last (tells it it's masked)
        seq_masked[random_positions, :-1] = 0  # Set to zero for every class but the last (tells it it's masked)
        if span > 1 and extra_append > 0:
            seq_masked = seq_masked[:-extra_append]
            seq_unmask = seq_unmask[:-extra_append]
        # print(seq_masked.shape)
        if return_probability_matrix:
            return seq_masked, seq_unmask, probability_matrix #return the masked sequence and the unmasked sequence, so we can use it for the rest of the processing
        
        return seq_masked, seq_unmask #return the masked sequence and the unmasked sequence, so we can use it for the rest of the processing
    
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

    if span > 1 and extra_append > 0:
        seq_masked = seq_masked[:-extra_append]
        seq_unmask = seq_unmask[:-extra_append]
    
    if return_probability_matrix:
        return seq_masked, seq_unmask, probability_matrix #return the masked sequence and the unmasked sequence, so we can use it for the rest of the processing

    return seq_masked, seq_unmask #return the masked sequence and the unmasked sequence, so we can use it for the rest of the processing

"""another idea is to basically define a list of numbers of random size, from negative binomial mean 500. Varying length for a lot of them. poisson if want more tight
Then you assign a random order, then the last 15% are defined as masked. You mask those, then for the genome you can have a list of 0 or 1 for every element, but it's in spans of whatever was defined by the negative binomial
This can be done at the initialization of the dataloader, and this resets each epoch"""