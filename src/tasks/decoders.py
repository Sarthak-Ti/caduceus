"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import math

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)

import inspect
from einops.layers.torch import Rearrange
from src.utils.enformer_pytorch import exponential_linspace_int, MaxPool, AttentionPool, ConvBlock, Residual, GELU, TargetLengthCrop
from src.utils.UNET import ConvBlock2, UpResBlock

# def trace_call():
#     stack = inspect.stack()
#     # Adjust the index to 2 to get the caller of the function that called trace_call
#     #let's print a few of the values
#     for i in range(1,len(stack)-1):
#         caller_frame = stack[i]  # This goes two levels up in the call stack
#         frame_info = inspect.getframeinfo(caller_frame[0])

#         print(f"Called from {frame_info.filename} at line {frame_info.lineno} in function {frame_info.function}")


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths
        
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
            
        # print(f"SequenceDecoder: mode={mode}, l_output={l_output}, use_lengths={use_lengths}, d_output={d_output}, d_model={d_model}")
        # import sys
        # sys.exit()
        #wondering why this is the casea, but d_output is none for some reason with my regression thing, think I need to define the task

    def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1) #was -2
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            def restrict(x_seq):
                """Pool sequence over a certain range"""
                L = x_seq.size(1)
                s = x_seq.sum(dim=1, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x_seq[..., -(l_output - 1):, ...].flip(1), dim=1)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(1)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x_seq.dtype, device=x_seq.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            # TODO use same restrict function as pool case
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x.squeeze())
            x_rc = self.output_transform(x_rc.squeeze())
            x = (x + x_rc) / 2
        else:
            x = self.output_transform(x)
            # print('output of forward from decoders.py', x.shape)
            # print(x) #was 1024 x 128 because the decoder was identity and using dna embedding model
            # #but when we added in the experiment yaml about d_output being 1, now we see it is 1024 x 1 as we hoped! 
            # import sys
            # sys.exit()

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out


class TokenDecoder(Decoder):
    """Decoder for token level classification"""
    def __init__(
        self, d_model, d_output=3
    ):
        super().__init__()

        self.output_transform = nn.Linear(d_model, d_output)

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        x = self.output_transform(x)
        return x

class ProfileDecoder(Decoder):
    '''Decoder for profile task and also coutns task'''
    def __init__(self, d_model = 128, d_output = 1, l_output = 0, mode='pool', use_lengths=False, linear_profile=True,):
        super().__init__()
        self.output_transform_counts = nn.Linear(d_model, d_output)
        if linear_profile:
            self.output_transform_profile = nn.Linear(d_model, 1) #maps the last dimension to the counts, so each sequence value gets its own
        else:
            self.output_transform_profile = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),  # Adding a non-linearity is a common practice, though it's optional
                nn.Linear(64, 1)
            )
        if l_output == 0:
            l_output = 1
            self.squeeze = True
        else:
            self.squeeze = False
        self.l_output = l_output
        self.mode = mode
        self.use_lengths = use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
        x_profile = self.output_transform_profile(x)
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x):
                return x[..., -l_output:, :]
        elif self.mode == "first":
            def restrict(x):
                return x[..., :l_output, :]
        elif self.mode == "pool": #what we used for the poster, finds moving average cumsum, if l_out is 0 becomes 1 which is average
            #unsure if mask is used
            if mask is None:
                def restrict(x):
                    return (torch.cumsum(x, dim=-2) / torch.arange(1, 1 + x.size(-2), device=x.device, dtype=x.dtype).unsqueeze(-1))[..., -l_output:, :]           
            else:
                # sum masks
                mask_sums = torch.sum(mask, dim=-1).squeeze() - 1  # for 0 indexing

                # convert mask_sums to dtype int
                mask_sums = mask_sums.type(torch.int64)

                def restrict(x):
                    return (torch.cumsum(x, dim=-2) / torch.arange(1, 1 + x.size(-2), device=x.device, dtype=x.dtype).unsqueeze(-1))[torch.arange(x.size(0)), mask_sums, :].unsqueeze(1)  # need to keep original shape

        elif self.mode == "sum":
            def restrict(x):
                return torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            def restrict(x):
                return x[..., :max(lengths), :]
        elif self.mode == 'mean': #useless, l_output = 0 turns to l_output = 1 which means pool is the average...
            def restrict(x):
                return torch.mean(x, dim=-2).unsqueeze(1)
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'mean' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform_counts(x)

        return (x_profile,x)
    

class EnformerDecoder(Decoder):
    '''Decoder for profile task and also coutns task'''
    def __init__(self, d_model = 128, d_output = 4675, l_output = 0, mode='pool', use_lengths=False, convolutions=False,
                 yshape=114688, bin_size=128, downsampled=False, dropout_rate=0.1,
                 num_downsamples=7, dim_divisible_by=128, kmer_len=None, conjoin_train=True, conjoin_test=False,
                 mouse = False, d_output_mouse = None, d_output2=None, log=False):
        super().__init__()
        # if d_output is None:
        #     d_output = yshape//bin_size
        #yshape is the size in nucleotides of the region you are looking at, independent of how downsampled it is, that's adjusted for later
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.kmer_len = kmer_len
        self.convolutions = convolutions
        
        if downsampled: #a number like 2 tells it how much it is already downsampled, so then only average pool with size 64
            bin_size = bin_size//downsampled #if it's downsampled to size 1, then it doesn't need pooling, but kernel size of 1 means it is just nn.Identity
            self.downsampled = downsampled
        else:
            self.downsampled = 1
        
        self.yshape = yshape//self.downsampled #the point of this is downsampling makes it so that each input represents multiple nucleotides, so when we account for that in the linear layers have to look at the central amount that is relative to the original size
        # if convolutions:
        #     if downsampled:
        #         raise NotImplementedError("Downsampled convolutional EnformerDecoder not implemented")
        #     self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=12)
        #     self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=6)
        #     self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3)
        #     self.pool1 = nn.MaxPool1d(kernel_size=8)
        #     self.pool2 = nn.MaxPool1d(kernel_size=6)
        #     self.pool3 = nn.MaxPool1d(kernel_size=3)
        if convolutions:
            # half_dim = d_model//2
            twice_dim = d_model*2
            # filter_list = exponential_linspace_int(half_dim, d_model, num = (num_downsamples - 1), divisible_by = dim_divisible_by)
            # filter_list = [half_dim, *filter_list] #lasta element of filter list is just d_model
            self.final_pointwise = nn.Sequential(
                Rearrange('b n d -> b d n'),
                ConvBlock(d_model, twice_dim, 1), #just CNN of size 1 and lots of output channels
                Rearrange('b d n -> b n d'),
                nn.Dropout(dropout_rate / 2),
                GELU()
            )
            # self.crop_final = TargetLengthCrop(self.yshape)
            d_model = twice_dim
        
        self.output_transform = nn.Linear(d_model, d_output)
        for p in self.output_transform.parameters():
            p._no_weight_decay = True
        if mouse:
            self.output_transform_mouse = nn.Linear(d_model, d_output_mouse)
            for p in self.output_transform_mouse.parameters():
                p._no_weight_decay = True
        self.pool = nn.AvgPool1d(kernel_size=bin_size) #default stride is kernel size

        if l_output == 0:
            l_output = 1
            self.squeeze = True
        else:
            self.squeeze = False
        self.l_output = l_output
        self.mode = mode
        self.use_lengths = use_lengths
        self.bin_size = bin_size
        self.softplus = nn.Softplus()
        self.log = log

    def forward(self, x, state=None, lengths=None, l_output=None, mask=None, mouse=False, intermediates=None):
        """
        Forward pass for the EnformerDecoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, seq_len, d_model, 2) if conjoining it.
            state (optional): Not used.
            lengths (optional): Not used.
            l_output (optional): Not used.
            mask (optional): Not used.
            mouse (bool mask): size of batch_size, tells you per batch .
            intermediates (optional): Useless, need to allow it to be passed in

        Returns:
            Tensor: Output tensor of shape (batch_size, num_bins, d_output).
        """
        #the first option is we pool it based on the 128 bp and just do some sort of average pool
        #the second option is we do a convolution and then pool it        
            
        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.evaluate_forward(x.squeeze(-1), mouse=mouse)
            x_rc = self.evaluate_forward(x_rc.squeeze(-1), mouse=mouse)
            x = (x + x_rc) / 2 #this is how we keep it rc, otherwise it would separate, unsure why the hiddne states differ, thought those would be identical tho
        else:
            x = self.evaluate_forward(x, mouse=mouse)

        return x

    def evaluate_forward(self, x, mouse=False):
        if self.convolutions is False:
            #we first take just the middle elements of the sequence
            # x = x[:,int(x.shape[1]/2)-int(self.yshape/2):int(x.shape[1]/2)+int(self.yshape/2),:]
            # if self.kmer_len is not None:
            #     x = x[:,self.kmer_len:] #because we cropped that many elements from the right already when we did kmer encoding
            #this approach for kmer_len is no longer used, now tokenized the whole genome instead
            startidx = x.shape[1]//2 - self.yshape//2
            endidx = startidx + self.yshape
            
            x = x[:,startidx:endidx,:] #cuts it to size yshape which is the middle 896, not the full 1536
            x_permute = x.permute(0,2,1)
            # x_pooled = F.avg_pool1d(x_permute, kernel_size=self.bin_size, stride=self.bin_size)
            x_pooled = self.pool(x_permute)
            x = x_pooled.permute(0,2,1)
        else:
            #enformer style convolutions
            # print(x.shape)
            x = self.final_pointwise(x)
            # if self.kmer_len is not None:
            #     x = x[:,self.kmer_len:] #because we cropped that many elements from the right already when we did kmer encoding
            #again this feature is deprecated, no longer used
            # x = self.crop_final(x) #we just implement it ourself
            
            startidx = x.shape[1]//2 - self.yshape//2
            endidx = startidx + self.yshape
            
            x = x[:,startidx:endidx,:] #cuts it to size yshape which is the middle 896, not the full 1536
            x_permute = x.permute(0,2,1)
            # x_pooled = F.avg_pool1d(x_permute, kernel_size=self.bin_size, stride=self.bin_size)
            x_pooled = self.pool(x_permute)
            x = x_pooled.permute(0,2,1)
            
            #my implementation of a separate approach
            # #now we need to do the convolution and then either pol it or figure something else out
            # x = x.permute(0, 2, 1)
            
            # x = self.conv1(x)
            # x = F.relu(x)
            # x = self.pool1(x)
            
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.pool2(x)

            # x = self.conv3(x)
            # x = F.relu(x)
            # x = self.pool3(x)

            # #with our max pooling, the data is already quite small
            # #so just crop it to be the right shape now
            # binlen = self.yshape // self.bin_size
            # startidx = x.shape[2]//2 - binlen//2
            # endidx = startidx + binlen
            # x = x[:,:,startidx:endidx]
            
            # # Apply pooling
            # # x = self.pool(x)
            # # Apply the final linear layer
            # # x now has shape (batch_size, 256, num_bins)
            # x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_bins, 256) for the linear layer
            
        if mouse:
            # x_mouse = self.output_transform_mouse(x)
            raise NotImplementedError("Mouse output not implemented, unsure how to mask it")
        
        x = self.output_transform(x)

        #now apply masking
            
        #softplus activation
        x = self.softplus(x)
        if self.log:
            self.metrics = {'decoder/rate_min': x.detach().min().float()}


        return x

class GraphRegDecoder(Decoder):
    '''Decoder for Graph Reg profiles'''
    def __init__(self, d_model = 128, l_output = 0, mode='pool', convolutions=False,
                 bin_size=100, dropout_rate=0.1,kmer_len=None, conjoin_train=True, conjoin_test=False):
        super().__init__()
        # if d_output is None:
        #     d_output = yshape//bin_size
        #yshape is the size in nucleotides of the region you are looking at, independent of how downsampled it is, that's adjusted for later
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.kmer_len = kmer_len
        if self.kmer_len:
            raise NotImplementedError("Kmer len not implemented")
        self.convolutions = convolutions
        self.bin_size = bin_size
        self.pool = nn.AvgPool1d(kernel_size=bin_size) #default stride is kernel size
        self.output_transform = nn.Linear(d_model, 3)
        
    def forward(self, x, state=None, lengths=None, l_output=None, mask=None, mouse=False):
        """
        Forward pass for the EnformerDecoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, seq_len, d_model, 2) if conjoining it.
            state (optional): Not used.
            lengths (optional): Not used.
            l_output (optional): Not used.
            mask (optional): Not used.
            mouse (bool mask): size of batch_size, tells you per batch .

        Returns:
            Tensor: Output tensor of shape (batch_size, num_bins, d_output).
        """
        #the first option is we pool it based on the 128 bp and just do some sort of average pool
        #the second option is we do a convolution and then pool it        
            
        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.evaluate_forward(x.squeeze(-1))
            x_rc = self.evaluate_forward(x_rc.squeeze(-1))
            x = (x + x_rc) / 2 #this is how we keep it rc, otherwise it would separate, unsure why the hiddne states differ, thought those would be identical tho
        else:
            x = self.evaluate_forward(x)

        return x
    
    def evaluate_forward(self, x, mouse=False):
        if self.convolutions is False:
            x_permute = x.permute(0,2,1) #now is baatch x d model x seqlen
            x_pooled = self.pool(x_permute) #now is batch x d model x num bins
            x = x_pooled.permute(0,2,1) #now is batch x num bins x d model
        else:
            raise NotImplementedError("Convolutional GraphRegDecoder not implemented")
        
        x = self.output_transform(x) #now should be batch x num bins x 3

        #exponential activation
        x = torch.exp(x)
        return x

# class JointMaskingDecoder(Decoder):
#     """Decoder for joint masking of sequence and accessibility, so runs 2 separate output heads
#     d_output1 is sequence, so generally 4 for DNA (maybe 5 if want to do N as well)
#     d_output2 is accessibility, so 1 if only 1 track since we'll then do softplus or sigmoid in loss
#     Key issue is that this does not apply softplus or sigmoid, so need to do that in eval class
#     """
#     def __init__(self, d_model, d_output1=5, d_output2=1):
#         super().__init__()
#         print(f"JointMaskingDecoder: d_model={d_model}, d_output1={d_output1}, d_output2={d_output2}")

#         self.decoder1 = nn.Linear(d_model, d_output1)
#         self.decoder2 = nn.Linear(d_model, d_output2)

#     def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
#         '''
#         x: (n_batch, l_seq, d_model)
#         Returns: (n_batch, l_seq, d_output1), (n_batch, l_seq, d_output2)
#         '''
#         x1 = self.decoder1(x)
#         x2 = self.decoder2(x)
#         return x1, x2
    
class JointMaskingDecoder(Decoder):
    """Decoder for joint masking of sequence and accessibility, so runs 2 separate output heads after upsampling
    d_output1 is sequence, so generally 4 for DNA (maybe 5 if want to do N as well)
    d_output2 is accessibility, so 1 if only 1 track since we'll then do softplus or sigmoid in loss
    upsample is the factor by which to upsample the input sequence
    finetune is whether we are finetuning, if so only output the value of the decoder2
    Key issue is that this does not apply softplus or sigmoid, so need to do that in eval class
    """
    def __init__(self, d_model, d_output1=5, d_output2=1, upsample=1, finetune=False, dpout=0.0):
        super().__init__()
        print(f"JointMaskingDecoder: d_model={d_model}, d_output1={d_output1}, d_output2={d_output2}, upsample={upsample}")
        if dpout > 0:
            self.dropout = nn.Dropout(dpout)
        else:
            self.dropout = nn.Identity()

        self.upsample = upsample
        self.finetune = finetune

        if self.upsample > 1:
            self.n_ups = int(math.log2(upsample))
            if self.n_ups == 1:
                grow_channels = 0 #doesn't change because only did a pooling operation
            else:
                d_model = d_model // 2
                grow_channels = d_model // (self.n_ups-1)  # how much to shrink channels at each upsample step
            # Channel schedule:           bottleneck → ... → final width
            # e.g. d_model=256, up=4  ==> [256, 256, 192]
            # in_channels = [d_model + i * grow_channels for i in range(self.n_ups)]
            # out_channels = [d_model + (i + 1) * grow_channels for i in range(self.n_ups)]
            ch = [d_model + i * grow_channels for i in range(self.n_ups)]
            ch.append(ch[-1]) #the first one is a repeat
            ch = ch[::-1]  # reverse to go from bottleneck to final width
            # print(ch)
            
            assert upsample & (upsample - 1) == 0, "upsample must be a power of 2"
            self.up_blocks = nn.ModuleList()

            for i in range(self.n_ups):
                self.up_blocks.append(UpResBlock(ch[i], ch[i+1]))
                # if i == 0:
                #     self.up_blocks.append(UpResBlock(ch[i], 0))
                # else:                
                #     self.up_blocks.append(UpResBlock(ch[i], grow_channels))
            
            # self.final_conv = ConvBlock2(ch[-2], ch[-1]) #do a final convolution to reduce dimensionality
        else:
            self.n_ups = 0
        
        if not self.finetune:
            self.decoder1 = nn.Linear(d_model, d_output1)
            for p in self.decoder1.parameters():
                p._no_weight_decay = True
        self.decoder2 = nn.Linear(d_model, d_output2)
        for p in self.decoder2.parameters():
            p._no_weight_decay = True
            

    def forward(self, x, intermediates=None, state=None, lengths=None, l_output=None, mask=None):
        '''
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_seq, d_output1), (n_batch, l_seq, d_output2)
        '''
        
        if self.upsample > 1:
            assert intermediates is not None, "intermediates must be provided for upsampling"
            x = x.permute(0, 2, 1)  # (n_batch, d_model, l_seq)
            for i in range(self.n_ups): #+ 1 to include the last upsample block
                bin_size = 2 ** (self.n_ups - 1 - i)     # e.g. 4,2
                skip = intermediates[f'bin_size_{bin_size}'] # (initially were stored as the wrong shape)
                x = self.up_blocks[i](x, skip)
                
            # x = self.final_conv(x)  # perform a final convolution to reduce dimensionality
            x = x.permute(0, 2, 1)  # (n_batch, l_seq, d_model)
            # print(x.shape)
        
        x = self.dropout(x)
        x2 = self.decoder2(x)
        if self.finetune:
            #do softplus on x2, because this poisson loss expects it to be softplussed
            x2 = F.softplus(x2)
            return x2
        x1 = self.decoder1(x)
        return x1, x2
        

class TSSDecoder(Decoder):
    """Decoder that predicts a single expression value from masked base-pair embeddings.

    Two modes:
    - bp_predictor=False (default): averages embeddings over the mask region, then maps
      the averaged embedding to a scalar via an MLP or linear head. The head is a PLAIN
      regression head -- no output activation and no log -- because the target
      (dataset counts, y[2]) is already log1p/log2 transformed, so the prediction lives
      directly on the target's log scale and is trained under MSE (`mse_tss`).
    - bp_predictor=True: projects each position independently (d_model -> 1), applies the
      mask, sums the per-position predictions, then applies Softplus. This is an additive
      model where each base pair contributes independently to the total expression. Here
      the sum is genuinely in LINEAR count space, so log2(sum + 1) is the correct
      transform onto the target scale and is kept.

    pool_region controls which positions the readout pools over:
    - 'tss'  (default): pool over the mask passed at call time (the TSS-neighborhood mask),
      preserving the original TSS-region-averaging behavior.
    - 'all'            : ignore the passed mask and pool over the ENTIRE window (Decima
      style), relying on the gene mask appended as an input channel to select the gene.
    """
    def __init__(self, d_model, d_output=1, hidden_dim=128, dropout=0, simple=False, bp_predictor=False, pool_region='tss'):
        super().__init__()

        assert pool_region in ('tss', 'all'), f"pool_region must be 'tss' or 'all', got {pool_region}"
        self.pool_region = pool_region
        self.bp_predictor = bp_predictor

        if bp_predictor:
            # Per-position linear: d_model -> 1, then sum over masked positions.
            # bias=False: a bias would be summed n_masked times, scaling with mask
            # length rather than sequence content — not useful or well-defined.
            self.bp_linear = nn.Linear(d_model, 1, bias=False)
            # Softplus applied per-position before summing, so each bp contributes
            # a non-negative count. This matches the additive counts interpretation.
            self.softplus = nn.Softplus()
        elif simple:
            # Bare linear readout, unconstrained output: the target is already on a log
            # scale, so no Softplus. (A Softplus here would additionally have made small
            # negative predictions for zero-expression genes impossible while killing
            # their gradient, for no benefit under MSE.)
            self.output_transform = nn.Linear(d_model, d_output)
        else:
            # Upgraded to an MLP for non-linear capacity
            # No final activation -- matches TSSProfileDecoder.count_head, which regresses
            # the same log-scale target under MSE.
            self.output_transform = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, d_output),
            )

    def forward(self, x, mask, state=None, **kwargs):
        """
        x:    (batch, length, d_model) - per-base-pair embeddings
        mask: (batch, length)          - float or bool; 1/True marks positions to include

        Returns:
            (batch, 1) - scalar prediction per sequence, on the target's log scale
        """
        # pool_region='all' ignores the supplied mask and pools over every position
        # (Decima-style mean pool); the gene mask lives in the input channels instead.
        if self.pool_region == 'all':
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)

        mask = mask.unsqueeze(-1).float()  # (batch, length, 1

        if self.bp_predictor:
            # Each position predicts a non-negative contribution, then sum over mask
            per_bp = self.softplus(self.bp_linear(x))  # (batch, length, 1), >= 0
            linear_sum = (per_bp * mask).sum(dim=1)    # (batch, 1)
            return torch.log2(linear_sum + 1)          # Explicit log transformation

        # Averaging mode: pool over masked positions then apply head
        avg_embedding = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)
        # Returned as-is: the target is already log-transformed, so a second log here
        # would force the head to emit 2**target in linear space.
        return self.output_transform(avg_embedding)  # (batch, d_output)


class TSSProfileDecoder(Decoder):
    """Decoder that jointly predicts a per-position RNA-seq profile and a scalar count
    from base-pair embeddings (chrombpnet / Decima style).

    Two heads share the trunk embeddings:
    - profile head: a per-position projection (d_model -> n_tracks) producing a shape
      logit at every position. The downstream multinomial loss softmaxes these over the
      length axis, so this head only learns the RELATIVE shape of the coverage profile.
    - count head: mean-pools the embeddings over the readout region, then maps the pooled
      vector to a scalar via an MLP. Trained with MSE against the precomputed log(1+CPM)
      count (dataset outputs2[2]).

    profile_region ('all' | 'gene') controls the profile head's support:
    - 'all'  (default): the profile spans every position (Decima-style; the gene mask is
      only an input-channel hint).
    - 'gene'          : the profile logits outside the gene body are pushed very negative
      so they drop out of the softmax / multinomial. Requires `gene_mask` at call time.

    profile_softplus (bool) controls what the profile head emits:
    - False (default): raw, unactivated logits. This is what `cbpnet_multinomial_nll`
      wants -- it log_softmaxes them over the length axis, so only relative shape matters.
    - True           : Softplus is applied, so the head emits a non-negative RATE per
      position rather than a logit. This makes the head numerically identical to the
      region-based 3' model's readout (EnformerDecoder applies Softplus unconditionally,
      see `EnformerDecoder.forward`), so `tss_profile_poisson_loss(profile_softplus=True)`
      runs the exact same `F.poisson_nll_loss(..., log_input=False, eps=1e-6)` computation
      as `poisson_loss_nll_nan` does there. Prefer this over letting the Poisson loss
      exponentiate raw logits: exp() is the canonical log link but has multiplicative
      gradients that overflow readily under bf16, whereas Softplus is bounded and is the
      parameterization that actually trained on this target.
      The multinomial term stays available under this flag -- the loss takes
      log(rate + eps) as the logits, and log_softmax of that just normalizes the rate.

    count_region ('all' | 'gene' | 'tss') controls the region the count head mean-pools
    over, independently of the profile head. Defaults to `profile_region` so existing
    configs are unchanged.
    - 'all' : pool over the entire window.
    - 'gene': pool over the gene body. Requires `gene_mask` at call time.
    - 'tss' : pool over the TSS-neighborhood `mask` passed at call time. This makes the
      count head exactly equivalent to TSSDecoder(pool_region='tss') -- same head, same
      pooling, same region. Not offered for the profile head, where restricting the
      softmax to the TSS neighborhood would not be a coverage profile.

    n_tracks > 1 is not supported by the count head: the count target is one scalar per gene.

    Returns a (profile, counts) tuple. PassthroughSequential leaves this tuple intact
    (its last element is a tensor, not a kwargs dict), so the task receives it as `outs`
    and the combined loss (`tss_profile_loss`) unpacks it.
    """
    def __init__(self, d_model, n_tracks=1, hidden_dim=128, dropout=0, profile_region='all',
                 count_region=None, profile_softplus=False):
        super().__init__()

        assert profile_region in ('all', 'gene'), \
            f"profile_region must be 'all' or 'gene', got {profile_region}"
        self.profile_region = profile_region
        self.profile_softplus = profile_softplus
        # None means "follow the profile head", preserving the old single-flag behavior.
        count_region = profile_region if count_region is None else count_region
        assert count_region in ('all', 'gene', 'tss'), \
            f"count_region must be 'all', 'gene' or 'tss', got {count_region}"
        self.count_region = count_region
        self.n_tracks = n_tracks

        # Per-position shape head: independent projection at every base pair.
        self.profile_head = nn.Linear(d_model, n_tracks)

        # Count head: pooled embedding -> scalar count(s). MLP for non-linear capacity.
        # No final activation: the target is log(1+CPM), trained directly under MSE.
        self.count_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_tracks),
        )

    def forward(self, x, mask=None, gene_mask=None, state=None, **kwargs):
        """
        x:         (batch, length, d_model) - per-base-pair embeddings
        mask:      (batch, length)          - TSS-neighborhood mask, required if count_region=='tss'
        gene_mask: (batch, length)          - gene-body mask, required if either region is 'gene'

        Returns:
            profile: (batch, n_tracks, length) - per-position shape logits, or a non-negative
                                                 per-position rate if profile_softplus (expression layout)
            counts:  (batch, n_tracks)         - scalar count prediction per track
        """
        # Region the count head pools over, chosen independently of the profile head.
        # .float() everywhere so the pooled mean is computed in fp32 and does not depend
        # on the autocast dtype -- matches TSSDecoder.
        if self.count_region == 'gene':
            assert gene_mask is not None, "count_region='gene' requires gene_mask"
            count_region = gene_mask.float()
        elif self.count_region == 'tss':
            assert mask is not None, "count_region='tss' requires mask"
            count_region = mask.float()
        else:
            count_region = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=torch.float)

        # Count head: masked mean-pool over the region, then MLP.
        region_e = count_region.unsqueeze(-1)  # (batch, length, 1)
        pooled = (x * region_e).sum(dim=1) / region_e.sum(dim=1).clamp(min=1)  # (batch, d_model)
        counts = self.count_head(pooled)  # (batch, n_tracks)

        # Profile head: per-position logits, transposed to the (batch, n_tracks, length)
        # layout the expression targets use so the multinomial softmax runs over length.
        profile = self.profile_head(x)          # (batch, length, n_tracks)
        profile = profile.transpose(1, 2)       # (batch, n_tracks, length)
        if self.profile_region == 'gene':
            assert gene_mask is not None, "profile_region='gene' requires gene_mask"
            # Suppress out-of-gene positions under the softmax. A large finite negative
            # (not -inf) keeps the log-softmax finite, so the multinomial's 0*logp terms
            # for zeroed-out positions stay well-defined rather than becoming NaN.
            profile = profile.masked_fill(gene_mask.unsqueeze(1) == 0, -1e9)  # (batch, 1, length) broadcast

        if self.profile_softplus:
            # Applied AFTER the gene masking above so out-of-gene positions become a rate of
            # exactly softplus(-1e9) = 0, i.e. "no coverage predicted here", which is what a
            # rate-space readout should say. Doing it before would let the masked_fill write
            # a negative value into a tensor that is supposed to be non-negative.
            # float() first: softplus/exp in bf16 saturates, and the Poisson NLL downstream
            # reads this as an actual rate, so the precision matters more than the memory.
            profile = F.softplus(profile.float())

        return profile, counts


class NDDecoder(Decoder):
    """Decoder for single target (e.g. classification or regression)"""
    def __init__(
        self, d_model, d_output=None, mode="pool"
    ):
        super().__init__()

        assert mode in ["pool", "full"]
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.mode = mode

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.mode == 'pool':
            x = reduce(x, 'b ... h -> b h', 'mean')
        x = self.output_transform(x)
        return x

class StateDecoder(Decoder):
    """Use the output state to decode (useful for stateful models such as RNNs or perhaps Transformer-XL if it gets implemented"""

    def __init__(self, d_model, state_to_tensor, d_output):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)
        self.state_transform = state_to_tensor

    def forward(self, x, state=None):
        return self.output_transform(self.state_transform(state))


class RetrievalHead(nn.Module):
    def __init__(self, d_input, d_model, n_classes, nli=True, activation="relu"):
        super().__init__()
        self.nli = nli

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError

        if (
            self.nli
        ):  # Architecture from https://github.com/mlpen/Nystromformer/blob/6539b895fa5f798ea0509d19f336d4be787b5708/reorganized_code/LRA/model_wrapper.py#L74
            self.classifier = nn.Sequential(
                nn.Linear(4 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, n_classes),
            )
        else:  # Head from https://github.com/google-research/long-range-arena/blob/ad0ff01a5b3492ade621553a1caae383b347e0c1/lra_benchmarks/models/layers/common_layers.py#L232
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, d_model // 2),
                activation_fn,
                nn.Linear(d_model // 2, n_classes),
            )

    def forward(self, x):
        """
        x: (2*batch, dim)
        """
        outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (n_batch, d_input)
        if self.nli:
            features = torch.cat(
                [outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1
            )  # (batch, dim)
        else:
            features = torch.cat([outs0, outs1], dim=-1)  # (batch, dim)
        logits = self.classifier(features)
        return logits


class RetrievalDecoder(Decoder):
    """Combines the standard FeatureDecoder to extract a feature before passing through the RetrievalHead"""

    def __init__(
        self,
        d_input,
        n_classes,
        d_model=None,
        nli=True,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__()
        if d_model is None:
            d_model = d_input
        self.feature = SequenceDecoder(
            d_input, d_output=None, l_output=0, *args, **kwargs
        )
        self.retrieval = RetrievalHead(
            d_input, d_model, n_classes, nli=nli, activation=activation
        )

    def forward(self, x, state=None, **kwargs):
        x = self.feature(x, state=state, **kwargs)
        x = self.retrieval(x)
        return x

class PackedDecoder(Decoder):
    def forward(self, x, state=None):
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "nd": NDDecoder,
    "retrieval": RetrievalDecoder,
    "state": StateDecoder,
    "pack": PackedDecoder,
    "token": TokenDecoder,
    "profile": ProfileDecoder,
    'enformer': EnformerDecoder,
    'graphreg': GraphRegDecoder,
    'jointmask': JointMaskingDecoder,
    'tss': TSSDecoder,
    'tss_profile': TSSProfileDecoder,
}
model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
    "jointmask": ["d_model"],
    "tss": ["d_model"],
    "tss_profile": ["d_model"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
        decoder = {"_name_": name}
    else:
        name = decoder["_name_"]
        decoder = decoder.copy()

    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_attrs_list = model_attrs.get(name, [])
    model_values = utils.config.extract_attrs_from_obj(model, *model_attrs_list)

    # Decoders whose model_attrs key is 'd_model' use keyword instantiation so we can
    # apply the same config-first / d_in / d_model priority as the encoder.
    # Decoders with other attr names (e.g. 'state': ['d_state', 'state_to_tensor'])
    # keep the legacy positional path because the attr names don't match constructor params.
    if model_attrs_list == ['d_model']:
        if model and 'd_model' not in decoder:
            d_in = getattr(model, 'd_in', None)
            resolved = d_in if d_in is not None else (model_values[0] if model_values else None)
            if resolved is not None:
                decoder['d_model'] = resolved
        obj = utils.instantiate(registry, decoder, *dataset_args)
    else:
        # Legacy positional approach; apply d_in resolution for dimension-typed attrs
        resolved_model_args = []
        for attr, value in zip(model_attrs_list, model_values):
            if attr in ('d_model', 'd_output') and model:
                d_in = getattr(model, 'd_in', None)
                resolved_model_args.append(d_in if d_in is not None else value)
            else:
                resolved_model_args.append(value)
        obj = utils.instantiate(registry, decoder, *resolved_model_args, *dataset_args)

    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
