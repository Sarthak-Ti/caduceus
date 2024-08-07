import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
from src.utils.enformer_pytorch import exponential_linspace_int, MaxPool, AttentionPool, ConvBlock, Residual, NoPool


class Encoder(nn.Module):
    """Encoder abstraction

    Accepts a tensor and optional kwargs. Other than the main tensor, all other arguments should be kwargs.
    Returns a tensor and optional kwargs.
    Encoders are combined via U.PassthroughSequential which passes these kwargs through in a pipeline. The resulting
    kwargs are accumulated and passed into the model backbone.
    """

    def forward(self, x, **kwargs):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return x, {}




class OneHotEncoder(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        assert n_tokens <= d_model
        self.d_model = d_model

    def forward(self, x):
        return F.one_hot(x.squeeze(-1), self.d_model).float()


class EnformerEncoder(Encoder):
    def __init__(self, d_input=None, d_model=256, filter_sizes=None, flat=False,
                num_downsamples = 7,    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
                dim_divisible_by = 128,
                pool_type = 'max',
                conv_tower = False,
                **kwargs,
             ):
        super().__init__()
        
        self.dim = d_model
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.pool_type = pool_type
        self.use_conv_tower = conv_tower
        if not self.use_conv_tower:
            self.dim = 2*self.dim #basically if we only use the stem, it outputs the model at half the dim, so we fix that!
        
        if self.pool_type == 'max':
            Pool = MaxPool
        elif self.pool_type == 'attention':
            Pool = AttentionPool
        elif self.pool_type == 'none':
            Pool = NoPool
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")
        # Pool = MaxPool if self.pool_type == 'max' else AttentionPool
        half_dim = self.dim // 2
        twice_dim = self.dim * 2
        
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            Pool(half_dim, pool_size = 2)
        )

        filter_list = exponential_linspace_int(half_dim, self.dim, num = (self.num_downsamples - 1), divisible_by = self.dim_divisible_by) #with default options, get [128,128,128,256,256,256]
        filter_list = [half_dim, *filter_list] #appends a 128 in front if default options
        #it's a list of number of filters which tells you how many channels you have, length should be constant

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                Pool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        #first we one hot encode
        x_onehot = torch.nn.functional.one_hot((x-7)%4, num_classes=4).float().transpose(1, 2) #need to make sure it is the right order
        if 11 in x:
            indices = torch.where(x == 11)
            for idx in range(len(indices[0])):
                x_onehot[indices[0][idx], 0, indices[1][idx]] = 0 #modify 0 because if it's N that's 11 which means after %4 it is 0, so that was orignally 1000 set it to 0000
        x = self.stem(x_onehot)
        if self.use_conv_tower:
            x = self.conv_tower(x)
        return x.transpose(1,2), True

# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "onehot": OneHotEncoder,
    "enformer": EnformerEncoder,
}
dataset_attrs = {
    "linear": ["d_input"],  # TODO make this d_data?
    "onehot": ["n_tokens"],
}
model_attrs = {
    "linear": ["d_model"],
    "onehot": ["d_model"],
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return None
    if isinstance(encoder, str):
        name = encoder
    else:
        name = encoder["_name_"]

    # Extract dataset/model arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))

    # Instantiate encoder
    obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
    return obj


def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
