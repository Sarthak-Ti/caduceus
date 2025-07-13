import torch
from torch import nn
import torch.nn.functional as F
import math


class ConvBlock2(nn.Module):
    """
    Vanilla version of your pseudocode:
        x = BatchNorm1d(x)
        x = GeLU(x)
        if width == 1:
            x = Linear(out_channels)(x)
        else:
            x = Conv1d(out_channels, width)(x)

    Expected input shape: (batch, channels, length)
    """
    def __init__(self, in_channels: int, out_channels: int, width: int = 5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()
        self.width = width

        if width == 1:
            self.proj = nn.Linear(in_channels, out_channels, bias=True)
        else:
            # padding = (width - 1) // 2
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=width, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the ConvBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length).
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, length)
        '''
        x = self.norm(x)
        x = self.act(x)

        if self.width == 1:
            x = x.permute(0, 2, 1)
            x = self.proj(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.proj(x)

        return x


class DownsampleBlock(nn.Module):
    """
    ConvBlock2 → (skip-padded residual) → ConvBlock2 → residual → 2x Pool.
    Output channels = in_channels + grow_channels.
    Output length   = input_len // 2.
    """
    def __init__(self,
                 in_channels: int,
                 grow_channels: int,
                 width: int = 5,
                 pool_type: str = "max"):
        super().__init__()

        out_channels = in_channels + grow_channels
        pool_cls = nn.MaxPool1d if pool_type == "max" else nn.AvgPool1d

        self.conv1 = ConvBlock2(in_channels, out_channels, width)
        self.conv2 = ConvBlock2(out_channels, out_channels, width)
        self.pool  = pool_cls(kernel_size=2, stride=2)
        self.grow_channels = grow_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- first residual -----
        y = self.conv1(x)                              # (N, C+g, L)
        pad = torch.zeros(x.size(0),
                          self.grow_channels,
                          x.size(2),
                          dtype=x.dtype,
                          device=x.device)
        y = y + torch.cat([x, pad], dim=1)             # pad skip on channel axis

        # ----- second residual -----
        z = self.conv2(y)                              # (N, C+g, L)
        z = z + y

        return self.pool(z), z                            # (N, C+g, L/2)


class UpResBlock(nn.Module):
    """
    Residual up-sampling block that
        • projects the low-res tensor to a *smaller* channel width
        • nearest-neighbour upsamples (x2)
        • adds a 1x1 projection of the skip tensor
        • finishes with a residual conv
    """
    def __init__(self,
                 in_channels:   int,   # channels of low-res input
                #  skip_channels: int,   # channels coming from encoder. Commented out because identical to out_channels
                 out_channels:  int,   # desired (smaller) channels after block
                # grow_channels: int,  # how many channels to grow by, 0 for the first layer
                 residual_scale: float = 0.9):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

        # out_channels = in_channels - grow_channels  # output channels after upsample

        # 1) project low-res features down to out_channels
        self.proj_in = ConvBlock2(in_channels,  out_channels)      # width 5 by default
        # 2) project skip tensor to out_channels (1×1 conv)
        self.skip_proj = ConvBlock2(out_channels, out_channels, width=1)
        # 3) extra conv for more capacity
        self.final = ConvBlock2(out_channels, out_channels)

    def forward(self, x, skip):
        """
        x     : (B, C, L)    ← low-res input
        skip  : (B, C_skip, 2L) ← skip connection from encoder (higher resolution)

        Returns: (B, C_skip, 2L)
        """
        # print(x.shape, skip.shape)
        C_skip = skip.size(1)
        # 1. Project + residual
        out = self.proj_in(x)
        # print(out.shape, x.shape, C_skip)
        out = out + x[:, :C_skip]  # channel-wise residual (crop if needed)

        # 2. Upsample (nearest-neighbor doubling)
        out = F.interpolate(out, scale_factor=2, mode="nearest") * self.residual_scale

        # 3. Add projection of skip connection (width-1 conv)
        out = out + self.skip_proj(skip)

        # 4. Final conv + residual
        out = out + self.final(out)

        return out



'''example how to use this encoder:
sample_ratio = 2
d_model = 512
from src.tasks.encoders import *
a = JointCNN(d_model=d_model, joint=False, downsample=sample_ratio)
test_tensor1 = torch.randn(2, 6, 1000)  # Example input for sequence
test_tensor2 = torch.randn(2, 2, 1000)  # Example input for accessibility
output, intermediates = a(test_tensor1, test_tensor2)
print('Output shape:', output.shape)  # Should be (2, 256, 125)
for key in intermediates.keys():
   print(key, intermediates[key].shape)

output = output.permute(0, 2, 1)  # (batch, length, dim)
#then the model happens, but outputs it the same shape

from src.tasks.decoders import JointMaskingDecoder
decoder = JointMaskingDecoder(d_model=d_model, d_output1=5, d_output2=1, upsample=sample_ratio)
out1, out2 = decoder(output, intermediates=intermediates)
print('Decoder output shapes:', out1.shape, out2.shape)
'''