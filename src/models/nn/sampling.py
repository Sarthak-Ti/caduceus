"""U-Net style downsampling and upsampling stacks.

Implements pseudocode-like behavior:
- Downsampling by MaxPool1d(stride=2)
- Upsampling by Repeat (repeat_interleave along sequence axis)
- Residual channel growth/shrinkage via pad/crop on channel axis
- Optional start channel scaling and linear growth toward final d_model

Norm options (norm_type):
  'layer'     : LayerNorm per token — batch-size independent, simpler
  'rms_batch' : RMSBatchNorm — BatchNorm without mean subtraction, EMA variance
                at inference (decay 0.9). Matches paper pseudocode.

Conv options (use_weight_std):
  False : plain nn.Conv1d
  True  : StandardizedConv1d — scaled weight standardization (Brock et al.)
          Matches paper pseudocode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _residual_pad_or_crop(x, target_channels):
    """Match channel dim by right-padding zeros or cropping tail channels."""
    c = x.shape[-1]
    if c == target_channels:
        return x
    if c > target_channels:
        return x[:, :, :target_channels]
    pad = target_channels - c
    return torch.cat([x, x.new_zeros(x.shape[0], x.shape[1], pad)], dim=-1)


class RMSBatchNorm1d(nn.Module):
    """BatchNorm with learned scale and offset but without shifting by the sample mean.

    Computes per-channel centered variance E[(x-μ)²] over (B, L) during training,
    maintains an EMA (decay=0.9) of that variance, and uses the EMA at eval time.
    Normalizes as x / sqrt(var + eps) — no mean subtraction.
    Input shape: (B, L, C) — channels last.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum  # weight on new batch value; decay = 1 - momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x: (B, L, C)
        if self.training:
            mean = x.mean(dim=[0, 1])                    # (C,)
            var = (x - mean).pow(2).mean(dim=[0, 1])     # (C,) — centered variance
            with torch.no_grad():
                self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            var = self.running_var
        x = x / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class StandardizedConv1d(nn.Conv1d):
    """Conv1d with scaled weight standardization (Brock et al., NFNet).

    Re-parameterizes weights as (W - mean(W)) / std(W) per output channel,
    scaled by a learnable per-output-channel gain. Matches paper pseudocode.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1))

    def forward(self, x):
        w = self.weight  # (out_channels, in_channels, kernel_size)
        mean = w.mean(dim=[1, 2], keepdim=True)
        std = w.std(dim=[1, 2], keepdim=True)
        w_hat = (w - mean) / (std + 1e-5) * self.gain
        return F.conv1d(x, w_hat, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvBlock(nn.Module):
    """norm -> GELU -> (Conv1d same-padding or Linear for width=1).

    Args:
        norm_type: 'layer' for LayerNorm, 'rms_batch' for RMSBatchNorm1d.
        use_weight_std: if True, use StandardizedConv1d instead of nn.Conv1d.
                        Only applies to width > 1 (width=1 always uses Linear).
    """

    def __init__(self, in_channels, out_channels, width=5, norm_type='layer', use_weight_std=False):
        super().__init__()
        # when B × L is large (e.g. small models, long sequences, or large batches) 
        if norm_type == 'rms':
            self.norm = RMSBatchNorm1d(in_channels)
        else:
            self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()
        self.width = width
        if width == 1: #a width of 1 is a pointwise conv, which is equivalent to a linear layer on channels
            self.op = nn.Linear(in_channels, out_channels)
        else:
            conv_cls = StandardizedConv1d if use_weight_std else nn.Conv1d
            self.op = conv_cls(in_channels, out_channels, kernel_size=width, padding=width // 2)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        if self.width == 1:
            return self.op(x)
        x = x.transpose(1, 2)
        x = self.op(x)
        x = x.transpose(1, 2)
        return x


class DownsampleLayer(nn.Module):
    """Single encoder stage block + max-pooling.

    out = conv_block(x, channels_out) + pad(x)
    out = out + conv_block(out, channels_out)
    Then max-pools sequence length by 2.
    """

    def __init__(self, in_channels, out_channels, conv_width=5, norm_type='layer', use_weight_std=False,
                 use_checkpoint=False):
        super().__init__()
        self.downres_increase = ConvBlock(in_channels, out_channels, width=conv_width,
                                          norm_type=norm_type, use_weight_std=use_weight_std)
        self.downres_refine = ConvBlock(out_channels, out_channels, width=conv_width,
                                        norm_type=norm_type, use_weight_std=use_weight_std)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.use_checkpoint = use_checkpoint

    def _forward_body(self, x):
        out = self.downres_increase(x)
        out = out + _residual_pad_or_crop(x, out.shape[-1])
        return out + self.downres_refine(out)

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint(self._forward_body, x, use_reentrant=False)
        else:
            out = self._forward_body(x)
        pooled = self.pool(out.transpose(1, 2)).transpose(1, 2)
        return pooled, out


class UpsampleLayer(nn.Module):
    """Single decoder stage block with repeat-based upsampling and skip fusion."""

    def __init__(self, in_channels, out_channels, conv_width=5, residual_scale_init=0.9,
                 norm_type='layer', use_weight_std=False, use_checkpoint=False):
        super().__init__()
        self.reduce_main = ConvBlock(in_channels, out_channels, width=conv_width,
                                     norm_type=norm_type, use_weight_std=use_weight_std)
        self.skip_project = ConvBlock(out_channels, out_channels, width=1,
                                      norm_type=norm_type, use_weight_std=use_weight_std)
        self.refine = ConvBlock(out_channels, out_channels, width=conv_width,
                                norm_type=norm_type, use_weight_std=use_weight_std)
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale_init], dtype=torch.float32))
        self.use_checkpoint = use_checkpoint

    def _forward_body(self, x, skip):
        # Reduce channels and apply crop-based residual.
        out = self.reduce_main(x) + _residual_pad_or_crop(x, skip.shape[-1])
        # Repeat-based upsampling with learnable per-level scale.
        out = out.repeat_interleave(2, dim=1) * self.residual_scale
        # Process skip with width-1 conv block before fusion.
        out = out + self.skip_project(skip)
        # Final residual conv block.
        return out + self.refine(out)

    def forward(self, x, skip):
        if self.use_checkpoint:
            return checkpoint(self._forward_body, x, skip, use_reentrant=False)
        return self._forward_body(x, skip)


class DownsampleStack(nn.Module):
    """Stack reducing sequence length by `factor` (power of 2) via max-pooling.

    Stores intermediate pre-pooling activations for UNET skip connections.
    Assumes dna_embedder already run, so starts with a pool and storage
    If transformer downsampling, no need to reembed
    """

    def __init__(
        self,
        d_model,
        factor,
        kernel_size=5,
        channel_scale=2,
        start_channels=None,
        grow_channels=None,
        norm_type='layer',
        use_weight_std=False,
        use_checkpoint=False,
    ):
        super().__init__()
        assert factor >= 1 and (factor & (factor - 1)) == 0, \
            f"factor must be a power of 2, got {factor}"
        self.n_layers = int(math.log2(factor))
        self.final_channels = d_model

        if self.n_layers == 0:
            self.stage_channels = []
            self.layers = nn.ModuleList()
            return

        if start_channels is None:
            scale = max(1, int(channel_scale))
            start_channels = max(1, d_model // scale)
        self.start_channels = int(start_channels)

        if self.n_layers == 1:
            stage_channels = [self.start_channels]
        else:
            if grow_channels is None:
                grow_channels = max(1, (self.final_channels - self.start_channels) // (self.n_layers - 1))
            stage_channels = [self.start_channels]
            for _ in range(1, self.n_layers):
                stage_channels.append(min(self.final_channels, stage_channels[-1] + int(grow_channels)))
            stage_channels[-1] = self.final_channels

        self.stage_channels = stage_channels
        self.layers = nn.ModuleList()
        in_ch = stage_channels[0]
        for out_ch in self.stage_channels[1:]:
            self.layers.append(
                DownsampleLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    conv_width=kernel_size,
                    norm_type=norm_type,
                    use_weight_std=use_weight_std,
                    use_checkpoint=use_checkpoint,
                )
            )
            in_ch = out_ch

    def forward(self, x):
        # x: (B, L, d_start) -> (B, L // factor, d_model), list of pre-pool intermediates
        if self.n_layers == 0:
            return x, []

        intermediates = []
        intermediates.append(x)  # include input as first intermediate for skip connection
        #now do a pool
        x = F.max_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        
        for layer in self.layers:
            x, skip = layer(x)
            intermediates.append(skip)
        return x, intermediates


class UpsampleStack(nn.Module):
    """Stack increasing sequence length by `factor` (power of 2) via repeat upsampling.

    Uses processed UNET skip connections from matching DownsampleStack.
    """

    def __init__(
        self,
        d_model,
        factor,
        kernel_size=5,
        channel_scale=2,
        start_channels=None,
        grow_channels=None,
        residual_scale_init=0.9,
        norm_type='layer',
        use_weight_std=False,
        use_checkpoint=False,
    ):
        super().__init__()
        assert factor >= 1 and (factor & (factor - 1)) == 0, \
            f"factor must be a power of 2, got {factor}"
        self.n_layers = int(math.log2(factor))
        self.final_channels = d_model

        if self.n_layers == 0:
            self.stage_channels = []
            self.layers = nn.ModuleList()
            return

        if start_channels is None:
            scale = max(1, int(channel_scale))
            start_channels = max(1, d_model // scale)
        start_channels = int(start_channels)

        if self.n_layers == 1:
            stage_channels = [start_channels]
        else:
            if grow_channels is None:
                grow_channels = max(1, (self.final_channels - start_channels) // (self.n_layers - 1))
            stage_channels = [start_channels]
            for _ in range(1, self.n_layers):
                stage_channels.append(min(self.final_channels, stage_channels[-1] + int(grow_channels)))
            stage_channels[-1] = self.final_channels

        self.stage_channels = stage_channels
        decode_channels = list(reversed(self.stage_channels))

        self.layers = nn.ModuleList()
        in_ch = decode_channels[0]
        for out_ch in decode_channels:
            self.layers.append(
                UpsampleLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    conv_width=kernel_size,
                    residual_scale_init=residual_scale_init,
                    norm_type=norm_type,
                    use_weight_std=use_weight_std,
                    use_checkpoint=use_checkpoint,
                )
            )
            in_ch = out_ch

    def forward(self, x, intermediates):
        # x: (B, L, d_model) -> (B, L * factor, d_model)
        # intermediates: list from DownsampleStack (reverse order pairing)
        if self.n_layers == 0:
            return x

        for i, layer in enumerate(self.layers):
            skip = intermediates[-(i + 1)]  # pair with matching scale (reverse order)
            x = layer(x, skip)

        return x