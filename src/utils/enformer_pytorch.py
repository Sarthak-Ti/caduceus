import math
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.distributed as dist


def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class MaxPool(torch.nn.Module):
    def __init__(self, dim = None, pool_size=2, padding='same'): #note dim is taken as a useless argument since only used for compatibility with AttentionPool
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding

    def forward(self, x):
        if self.padding == 'same':
            pad_total = self.pool_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
            #let's raise an error now to make sure this isn't used because it isn't implemented
            # raise NotImplementedError("Need to test this, particularly to see what the shape of x is. X should be b, d, n. Test this but probably fine?")
        return F.max_pool1d(x, self.pool_size, stride=self.pool_size)
    
class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False) #basically computes the attention logits, since 1x1 kernel
        #does it independently per channel when had option to share weights in enformer implementation

        nn.init.dirac_(self.to_attn_logits.weight) #initializes as dirac delta which is like identity under conv?

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2) #scales the weights by 2

    def forward(self, x):
        '''
        x: (b, d, n) which is batch x dim x seq_len (dim is same as channels)
        '''
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0 #pads if it is not divisible

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x) #rearranges from (b, d, n) to (b, d, n // pool_size, pool_size). This is to let it pool along len dimension
        logits = self.to_attn_logits(x) #applies that 1x1 kenrel over the whole thing

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        #If padding was added, the logits corresponding to the padded values are set to a very large negative
        #number to effectively mask them out during the softmax operation.

        attn = logits.softmax(dim = -1) #applies softmax over the whole thing

        return (x * attn).sum(dim = -1) #applies the attention weights to the pooled values and sums over last dim to get pool

class NoPool(nn.Module):
    def __init__(self, dim = None, pool_size = 2, padding = 'same'):
        super().__init__()

    def forward(self, x):
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

def ConvBlock(dim, dim_out = None, kernel_size = 1, is_distributed = None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed = is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]