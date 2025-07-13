import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
from src.utils.enformer_pytorch import exponential_linspace_int, MaxPool, AttentionPool, ConvBlock, Residual, NoPool
from src.utils.UNET import ConvBlock2, DownsampleBlock


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
        # twice_dim = self.dim * 2
        
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

class CNNEmbedding(Encoder):
    def __init__(self, d_input, d_model, flat=False, **kwargs):
        super().__init__()
        #just have to appply a super simple conv that goes to model size
        self.conv = nn.Conv1d(d_input, d_model, 1)
                
class JointCNN(Encoder):
    """
    A CNN-based encoder that takes in OHE and accessibility data.

    If `joint` is True, the inputs are concatenated along the channel dimension and passed through a single CNN.
    If `joint` is False, each input is processed by its own CNN, and their outputs are concatenated.

    Args:
        d_input1 (int): The number of input channels for sequence.
        d_input2 (int): The number of input channels for accessibility.
        d_model (int): The desired output dimension of the model.
        joint (bool): Whether to process the inputs jointly or separately. Default is False.
        kernel_size (int): The size of the convolutional kernel. Default is 15.
    """
    def __init__(self, d_model, d_input1=6, d_input2=2, joint=False, kernel_size=15, combine=True, acc_type='continuous', downsample=1, pool_type='max', **kwargs):
        super().__init__()
        print(f"JointMaskingEncoder: d_model={d_model}, d_input1={d_input1}, d_input2={d_input2}, joint={joint}, kernel_size={kernel_size}, combine={combine}, acc_type={acc_type}")
        # print(kwargs)
        self.joint = joint
        self.combine = combine
        self.downsample = downsample
        
        if downsample > 1:
            #we allow the max dimension to be smaller
            self.n_pools = int(math.log2(downsample))
            if self.n_pools == 1:
                grow_channels = 0 #only do pooling, we have cnn embedding of model
            else:
                d_model = d_model // 2
                grow_channels = d_model // (self.n_pools-1)
            # out_dim = [d_model + i * d_model // self.n_pools for i in range(self.n_pools + 1)]
        
        # print(d_input1, d_input2)
        # print('dmodel', d_model)
        # print('acc_type', acc_type)
        if acc_type == 'continuous':
            d_input2 = 2
        elif acc_type == 'category':
            d_input2 = 3

        if joint:
            # Single CNN for joint processing
            self.conv = nn.Sequential(
                nn.Conv1d(d_input1+d_input2, d_model, kernel_size, padding='same'),
                nn.ReLU()
            )
        else:
            # Separate CNNs for each input
            self.conv1 = nn.Sequential(
                nn.Conv1d(d_input1, d_model // 2, kernel_size, padding='same'),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(d_input2, d_model // 2, kernel_size, padding='same'),
                nn.ReLU()
            )
            if combine:
                self.out = nn.Linear(d_model, d_model)
        
        #now handle downsamples and define the convolution and pooling layers
        if self.downsample != 1:
            assert self.downsample > 0 and (self.downsample & (self.downsample - 1)) == 0, "downsample must be a power of 2"
            # now do the downsampling
            #make a list that goes form 2**0 to 2**(n_pools-1) of the number of channels to grow
            # self.bin_sizes = [2 ** i for i in range(1,self.n_pools)]
            # layers = [ConvBlock2(d_model, d_model+d_diff)]
            
            # self.pool_layers = nn.ModuleList()
            # self.conv_layers = nn.ModuleList()
            self.down_blocks = nn.ModuleList()
            for i in range(self.n_pools - 1):
                in_dim = d_model + i * grow_channels
                self.down_blocks.append(
                    DownsampleBlock(in_dim, grow_channels, pool_type=pool_type)
                )
            # for i in range(self.n_pools):
            #     in_dim = d_model + i * d_diff
            #     out_dim = d_model + (i + 1) * d_diff
            #     self.conv_layers.append(ConvBlock2(in_dim, out_dim))
            #     self.pool_layers.append(pool_cls(kernel_size=2, stride=2))
        else:
            self.n_pools = 0
                
    def forward(self, x1, x2):
        """
        Forward pass for the JointCNN.

        Args:
            x1 (torch.Tensor): The first input tensor of shape (batch_size, d_input1, seq_len).
            x2 (torch.Tensor): The second input tensor of shape (batch_size, d_input2, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, d_model, seq_len).
            If downsample > 1, seq_len is divided by downsample.
            torch.Tensor: A dictionary containing intermediate outputs at different bin sizes.
            The keys are "bin_size_2", "bin_size_4", etc., corresponding to the downsampled outputs. each will be of shape (batch_size, d_model + i * grow_channels, seq_len // (2 ** (i + 1)))
        """
        if self.joint:
            # Concatenate inputs along the channel dimension and process jointly
            x = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
            x = self.conv(x)
        else:
            # Process inputs separately and concatenate their outputs
            x1_out = self.conv1(x1)
            x2_out = self.conv2(x2)
            x = torch.cat([x1_out, x2_out], dim=1)  # Concatenate along channel dimension
            if self.combine:
                x = self.out(x.transpose(1,2)).transpose(1,2) #applies linear layer along embedding dimension, across batch and length is independent
        
        intermediates = {}
        if self.downsample != 1:
            intermediates["bin_size_1"] = x  # Store the initial output before convs
            # print(x.shape)
            x = nn.MaxPool1d(kernel_size=2, stride=2)(x)  # Initial downsample
            # print(x.shape)
            for i, block in enumerate(self.down_blocks):
                x, intermediate = block(x)
                bin_size = 2 ** (i + 1)
                intermediates[f"bin_size_{bin_size}"] = intermediate

        return x, intermediates


class JointCNNWithCTT(Encoder):
    """
    A CNN-based encoder that takes in OHE and accessibility data and also uses a cell type token
    Replaces nucleotide 1 (accessibility and sequence) with a cell type token embedding

    If `joint` is True, the inputs are concatenated along the channel dimension and passed through a single CNN.
    If `joint` is False, each input is processed by its own CNN, and their outputs are concatenated.

    Args:
        d_input1 (int): The number of input channels for sequence.
        d_input2 (int): The number of input channels for accessibility.
        d_model (int): The desired output dimension of the model.
        joint (bool): Whether to process the inputs jointly or separately. Default is False.
        kernel_size (int): The size of the convolutional kernel. Default is 15.
    """
    def __init__(self, d_model, celltypes, d_input1=6, d_input2=2, joint=False, kernel_size=15, combine=True, acc_type='continuous', **kwargs):
        super().__init__()
        print(f"JointMaskingEncoder: d_model={d_model}, celltypes={celltypes}, d_input1={d_input1}, d_input2={d_input2}, joint={joint}, kernel_size={kernel_size}, combine={combine}, acc_type={acc_type}")
        # print(kwargs)
        self.joint = joint
        self.combine = combine
        # print(d_input1, d_input2)
        # print('dmodel', d_model)
        # print('acc_type', acc_type)
        if acc_type == 'continuous':
            d_input2 = 2
        elif acc_type == 'category':
            d_input2 = 3

        if joint:
            # Single CNN for joint processing
            self.conv = nn.Sequential(
                nn.Conv1d(d_input1+d_input2, d_model, kernel_size, padding='same'),
                nn.ReLU()
            )
        else:
            # Separate CNNs for each input
            self.conv1 = nn.Sequential(
                nn.Conv1d(d_input1, d_model // 2, kernel_size, padding='same'),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(d_input2, d_model // 2, kernel_size, padding='same'),
                nn.ReLU()
            )
            if combine:
                self.out = nn.Linear(d_model, d_model)
        
        # now we get an embedding for the cell type token
        self.ctt_embedding = nn.Embedding(celltypes, d_model)

    def forward(self, x1, x2, token):
        """
        Forward pass for the JointCNNWithCTT.

        Args:
            x1 (torch.Tensor): The first input tensor of shape (batch_size, d_input1, seq_len).
            x2 (torch.Tensor): The second input tensor of shape (batch_size, d_input2, seq_len).
            token (torch.Tensor): The cell type token of shape (batch_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, d_model, seq_len).
        """
        
        #we first remove the first nucleotide
        x1 = x1[:,:,1:] #remove the first nucleotide
        x2 = x2[:,:,1:] #remove the first nucleotide
        
        if self.joint:
            # Concatenate inputs along the channel dimension and process jointly
            x = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
            x = self.conv(x)
        else:
            # Process inputs separately and concatenate their outputs
            x1_out = self.conv1(x1)
            x2_out = self.conv2(x2)
            x = torch.cat([x1_out, x2_out], dim=1)  # Concatenate along channel dimension
            if self.combine:
                x = self.out(x.transpose(1,2)).transpose(1,2) #applies linear layer along embedding dimension, across batch and length is independent
        
        #and now we add the cell type token
        ctt = self.ctt_embedding(token).unsqueeze(2) #shape batch x d_model x 1
        #now append to x
        x = torch.cat([ctt, x], dim=2) #concatenate along the length dimension

        return x

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
    "cnn": CNNEmbedding,
    "jointcnn": JointCNN,
    "jointcnn_ctt": JointCNNWithCTT,
}
dataset_attrs = {
    "linear": ["d_input"],  # TODO make this d_data?
    "onehot": ["n_tokens"],
    "jointcnn": ["acc_type"],
    "jointcnn_ctt": ["acc_type", "celltypes"],
}
model_attrs = {
    "linear": ["d_model"],
    "onehot": ["d_model"],
    "jointcnn": ["d_model"],
    "jointcnn_ctt": ["d_model"],
}


# def _instantiate(encoder, dataset=None, model=None):
#     """Instantiate a single encoder"""
#     if encoder is None:
#         return None
#     if isinstance(encoder, str):
#         name = encoder
#     else:
#         name = encoder["_name_"]

#     # Extract dataset/model arguments from attribute names
#     dataset_args = utils.config.extract_attrs_from_obj(
#         dataset, *dataset_attrs.get(name, [])
#     )
#     model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
#     print('model_args:',model_args)
#     # print('registry:',registry)
#     # Instantiate encoder
#     obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
#     return obj

def _instantiate(encoder, dataset=None, model=None):
    if encoder is None:
        return None
    if isinstance(encoder, str):
        name = encoder
        encoder = {"_name_": name}
    else:
        name = encoder["_name_"]
        encoder = encoder.copy()  # Make a copy to avoid modifying the original

    # Extract dataset/model arguments from attribute names
    dataset_attrs_list = dataset_attrs.get(name, [])
    model_attrs_list = model_attrs.get(name, [])
    
    # Get dataset attributes and add them as keyword arguments to encoder
    if dataset and dataset_attrs_list:
        print('dataset:', dataset)
        print('dataset_attrs_list:', dataset_attrs_list)
        dataset_values = utils.config.extract_attrs_from_obj(dataset, *dataset_attrs_list)
        print('dataset_args:', dataset_values)
        for attr, value in zip(dataset_attrs_list, dataset_values):
            if value is not None:
                encoder[attr] = value
    
    # Get model attributes and add them as keyword arguments to encoder
    if model and model_attrs_list:
        model_values = utils.config.extract_attrs_from_obj(model, *model_attrs_list)
        print('model_args:', model_values)
        for attr, value in zip(model_attrs_list, model_values):
            if value is not None:
                encoder[attr] = value
    
    # Instantiate the encoder using only keyword arguments
    obj = utils.instantiate(registry, encoder)
    
    return obj

def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
