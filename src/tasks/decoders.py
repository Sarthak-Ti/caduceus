"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)

import inspect
from einops.layers.torch import Rearrange
from src.utils.enformer_pytorch import exponential_linspace_int, MaxPool, AttentionPool, ConvBlock, Residual, GELU, TargetLengthCrop

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
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool": #what we used for the poster, finds moving average cumsum, if l_out is 0 becomes 1 which is average
            #unsure if mask is used
            if mask is None:
                restrict = lambda x: (
                    torch.cumsum(x, dim=-2)
                    / torch.arange(
                        1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                    ).unsqueeze(-1)
                )[..., -l_output:, :]           
            else:
                # sum masks
                mask_sums = torch.sum(mask, dim=-1).squeeze() - 1  # for 0 indexing

                # convert mask_sums to dtype int
                mask_sums = mask_sums.type(torch.int64)

                restrict = lambda x: (
                    torch.cumsum(x, dim=-2)
                    / torch.arange(
                        1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                    ).unsqueeze(-1)
                )[torch.arange(x.size(0)), mask_sums, :].unsqueeze(1)  # need to keep original shape

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        elif self.mode == 'mean': #useless, l_output = 0 turns to l_output = 1 which means pool is the average...
            restrict = lambda x: torch.mean(x, dim=-2).unsqueeze(1)
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
                 num_downsamples=7, dim_divisible_by=128, kmer_len=None, conjoin_train=True, conjoin_test=False):
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
            half_dim = d_model//2
            twice_dim = d_model*2
            filter_list = exponential_linspace_int(half_dim, d_model, num = (num_downsamples - 1), divisible_by = dim_divisible_by)
            filter_list = [half_dim, *filter_list]
            self.final_pointwise = nn.Sequential(
                Rearrange('b n d -> b d n'),
                ConvBlock(filter_list[-1], twice_dim, 1),
                Rearrange('b d n -> b n d'),
                nn.Dropout(dropout_rate / 2),
                GELU()
            )
            self.crop_final = TargetLengthCrop(self.yshape)
            d_model = twice_dim
        
        self.output_transform = nn.Linear(d_model, d_output)
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

    def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
        """
        Forward pass for the EnformerDecoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, seq_len, d_model, 2) if conjoining it.
            state (optional): Not used.
            lengths (optional): Not used.
            l_output (optional): Not used.
            mask (optional): Not used.

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

    def evaluate_forward(self, x):
        if self.convolutions is False:
            #we first take just the middle elements of the sequence
            # x = x[:,int(x.shape[1]/2)-int(self.yshape/2):int(x.shape[1]/2)+int(self.yshape/2),:]
            # if self.kmer_len is not None:
            #     x = x[:,self.kmer_len:] #because we cropped that many elements from the right already when we did kmer encoding
            #this approach for kmer_len is no longer used, now tokenized the whole genome instead
            startidx = x.shape[1]//2 - self.yshape//2
            endidx = startidx + self.yshape
            
            x = x[:,startidx:endidx,:]
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
            x = self.crop_final(x)
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
            
        x = self.output_transform(x)

        #softplus activation
        x = self.softplus(x)

        return x
        

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
}
model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
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
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
