"""Full blocks wrapping Hydra and MHAGate with pre-norm residuals and MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.nn.hydra import Hydra
from src.models.nn.MHA import MHAGate

from flash_attn.modules.mlp import FusedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm


class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, activation='gelu', dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


def _build_norm(d_model, norm_type='ln'):
    if norm_type == 'ln':
        return nn.LayerNorm(d_model)
    elif norm_type == 'rms':
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class HydraBlock(nn.Module):
    """Hydra (bidirectional SSM) wrapped with pre-norm, residual connections, and optional MLP.

    Pattern with MLP:    x -> norm -> Hydra -> dropout -> + residual -> norm -> MLP -> dropout -> + residual
    Pattern without MLP: x -> norm -> Hydra -> dropout -> + residual
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=64,
        ngroups=1,
        expansion_factor=4,
        norm='ln',
        dropout=0.0,
        mlp_activation='gelu',
        mlp_dropout=0.0,
        fused_mlp=False,
        use_mlp=False,
        use_checkpoint=False,
        residual_in_fp32=False,
        **hydra_kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.residual_in_fp32 = residual_in_fp32
        self.use_mlp = use_mlp
        self.mixer_norm = _build_norm(d_model, norm)
        self.mixer = Hydra(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            **hydra_kwargs,
        )
        self.mixer_dropout = nn.Dropout(dropout)

        if use_mlp:
            self.mlp_norm = _build_norm(d_model, norm)
            if fused_mlp:
                self.mlp = FusedMLP(in_features=d_model, hidden_features=d_model * expansion_factor)
            else:
                self.mlp = MLP(d_model, d_model * expansion_factor, mlp_activation, mlp_dropout)
            self.mlp_dropout = nn.Dropout(dropout)

    def _forward(self, x):
        dtype = x.dtype
        residual = x.float() if self.residual_in_fp32 else x

        h = self.mixer_dropout(self.mixer(self.mixer_norm(residual.to(dtype))))
        residual = residual + (h.float() if self.residual_in_fp32 else h)

        if self.use_mlp:
            h = self.mlp_dropout(self.mlp(self.mlp_norm(residual.to(dtype))))
            residual = residual + (h.float() if self.residual_in_fp32 else h)

        return residual.to(dtype) if self.residual_in_fp32 else residual

    def _forward_with_residual(self, residual):
        # Cross-block fp32 residual path for mamba_only stacks.
        # residual: (B, L, d_model) fp32 running sum across blocks.
        dtype = self.mixer_norm.weight.dtype
        h = self.mixer_dropout(self.mixer(self.mixer_norm(residual.to(dtype))))
        residual = residual + h.float()
        if self.use_mlp:
            h = self.mlp_dropout(self.mlp(self.mlp_norm(residual.to(dtype))))
            residual = residual + h.float()
        return residual

    def forward_with_residual(self, residual):
        if self.use_checkpoint:
            return grad_checkpoint(self._forward_with_residual, residual, use_reentrant=False)
        return self._forward_with_residual(residual)

    def forward(self, x):
        # x: (B, L, d_model)
        if self.use_checkpoint:
            return grad_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class TransformerBlock(nn.Module):
    """MHAGate wrapped with pre-norm, residual connections, and MLP.

    Pattern: x -> norm -> MHAGate -> dropout -> + residual -> norm -> MLP -> dropout -> + residual
    """

    def __init__(
        self,
        d_model,
        head_dim=64,
        expansion_factor=4,
        norm='ln',
        dropout=0.0,
        use_rope=True,
        use_flash_attn=True,
        use_gating=True,
        mlp_activation='gelu',
        mlp_dropout=0.0,
        attention_dropout=0.0,
        fused_mlp=False,
        use_checkpoint=False,
        residual_in_fp32=False,
        **mha_kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.residual_in_fp32 = residual_in_fp32
        self.attn_norm = _build_norm(d_model, norm)
        self.attn = MHAGate(
            dim=d_model,
            head_dim=head_dim,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            use_gating=use_gating,
            dropout=attention_dropout,
            **mha_kwargs,
        )
        self.post_attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = _build_norm(d_model, norm)
        if fused_mlp:
            self.mlp = FusedMLP(in_features=d_model, hidden_features=d_model * expansion_factor)
        else:
            self.mlp = MLP(d_model, d_model * expansion_factor, mlp_activation, mlp_dropout)
        self.post_mlp_dropout = nn.Dropout(dropout)

    def _forward(self, x):
        # old code (no residual_in_fp32):
        # h = x + self.post_attn_dropout(self.attn(self.attn_norm(x)))
        # out = h + self.post_mlp_dropout(self.mlp(self.mlp_norm(h)))
        # return out
        dtype = x.dtype
        residual = x.float() if self.residual_in_fp32 else x

        h = self.post_attn_dropout(self.attn(self.attn_norm(residual.to(dtype))))
        residual = residual + (h.float() if self.residual_in_fp32 else h)

        h = self.post_mlp_dropout(self.mlp(self.mlp_norm(residual.to(dtype))))
        residual = residual + (h.float() if self.residual_in_fp32 else h)

        return residual.to(dtype) if self.residual_in_fp32 else residual

    def _forward_with_residual(self, residual):
        # Cross-block fp32 residual path for transformer_only stacks.
        # residual: (B, L, d_model) fp32 running sum across blocks.
        dtype = self.attn_norm.weight.dtype
        h = self.post_attn_dropout(self.attn(self.attn_norm(residual.to(dtype))))
        residual = residual + h.float()
        h = self.post_mlp_dropout(self.mlp(self.mlp_norm(residual.to(dtype))))
        residual = residual + h.float()
        return residual

    def forward_with_residual(self, residual):
        if self.use_checkpoint:
            return grad_checkpoint(self._forward_with_residual, residual, use_reentrant=False)
        return self._forward_with_residual(residual)

    def forward(self, x):
        # x: (B, L, d_model)
        if self.use_checkpoint:
            return grad_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
