"""Striped Mamba backbone with U-Net style up/downsampling.

Supports three modes:
- 'striped': Alternates HydraBlock (at global-pooled resolution) and TransformerBlock
             (at further transformer-pooled resolution) with UNET skip connections.
- 'mamba_only': Only HydraBlocks at global-pooled resolution.
- 'transformer_only': Only TransformerBlocks at global-pooled resolution.
"""

import math

import torch
import torch.nn as nn

from src.models.nn.blocks import HydraBlock, TransformerBlock, _build_norm
from src.models.nn.sampling import DownsampleStack, UpsampleStack, ConvBlock


class StripedMambaBackbone(nn.Module):

    def __init__(
        self,
        d_model,
        n_blocks,
        d_in=None,
        global_pooling=1,
        transformer_pooling=1,
        mode='striped',
        ssm_per_transformer=1,
        # Hydra config
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=256,
        use_mem_eff_path=True,
        dt_limit_lower=0.0,
        dt_limit_upper=float("inf"),
        a_log_clamp_min=None,
        # Transformer config
        head_dim=64,
        use_rope=True,
        use_flash_attn=True,
        use_gating=True,
        attention_dropout=0.0,
        use_enformer_bias=False,
        pos_emb_dim=32,
        # Shared config
        expansion_factor=4,
        norm='rms',
        dropout=0.0,
        mlp_activation='gelu',
        mlp_dropout=0.0,
        fused_mlp=False,
        hydra_use_mlp=False,
        checkpoint_blocks=False,
        residual_in_fp32=False,
        rescale_prenorm_residual=True,
        zero_linear_biases=True,
        # Sampling config
        downsample_kernel_size=5,
        global_sampling_channel_scale=2,
        global_sampling_start_channels=None,
        global_sampling_grow_channels=None,
        transformer_sampling_channel_scale=1,
        transformer_sampling_start_channels=None,
        transformer_sampling_grow_channels=None,
        upsample_residual_scale_init=0.9,
        sampling_norm_type='rms',
        sampling_use_weight_std=True,
        sampling_checkpoint=False,
        monitor_a_log=False,
    ):
        super().__init__()
        self.monitor_a_log = monitor_a_log

        assert mode in ('striped', 'mamba_only', 'transformer_only'), \
            f"mode must be 'striped', 'mamba_only', or 'transformer_only', got '{mode}'"
        assert global_pooling >= 1 and (global_pooling & (global_pooling - 1)) == 0, \
            f"global_pooling must be a power of 2, got {global_pooling}"
        assert transformer_pooling >= 1 and (transformer_pooling & (transformer_pooling - 1)) == 0, \
            f"transformer_pooling must be a power of 2, got {transformer_pooling}"

        if d_in is None:
            d_in = d_model
            if global_pooling > 1:
                print("Warning: d_in not specified, defaulting to d_model. If using global pooling, consider setting d_in to a smaller value to save compute.")
        else:
            assert global_pooling > 1, "d_in can only be specified if global_pooling > 1 since it needs to somehow increase resolution"
        
        if global_sampling_start_channels is None:
            global_sampling_start_channels = d_in

        self.mode = mode
        self.n_blocks = n_blocks
        self.ssm_per_transformer = ssm_per_transformer
        self.global_pooling = global_pooling
        self.transformer_pooling = transformer_pooling
        self.residual_in_fp32 = residual_in_fp32
        
        #first run one conv block
        self.conv_block = ConvBlock(d_in, d_in, width=5, norm_type=sampling_norm_type, use_weight_std=sampling_use_weight_std)

        # --- Global downsample / upsample ---
        if global_pooling > 1:
            self.global_down = DownsampleStack(
                d_model,
                global_pooling,
                kernel_size=downsample_kernel_size,
                channel_scale=global_sampling_channel_scale,
                start_channels=global_sampling_start_channels,
                grow_channels=global_sampling_grow_channels,
                norm_type=sampling_norm_type,
                use_weight_std=sampling_use_weight_std,
                use_checkpoint=sampling_checkpoint,
            )
            self.global_up = UpsampleStack(
                d_model,
                global_pooling,
                kernel_size=downsample_kernel_size,
                channel_scale=global_sampling_channel_scale,
                start_channels=global_sampling_start_channels,
                grow_channels=global_sampling_grow_channels,
                residual_scale_init=upsample_residual_scale_init,
                norm_type=sampling_norm_type,
                use_weight_std=sampling_use_weight_std,
                use_checkpoint=sampling_checkpoint,
            )
        else:
            self.global_down = None
            self.global_up = None

        # --- Per-block modules ---
        self.hydra_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.trans_down = nn.ModuleList()
        self.trans_up = nn.ModuleList()

        hydra_kwargs = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, ngroups=ngroups, expansion_factor=expansion_factor,
            norm=norm, dropout=dropout, mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout, fused_mlp=fused_mlp, use_mlp=hydra_use_mlp,
            use_checkpoint=checkpoint_blocks, residual_in_fp32=residual_in_fp32,
            chunk_size=chunk_size, use_mem_eff_path=use_mem_eff_path,
            dt_limit=(dt_limit_lower, dt_limit_upper),
            a_log_clamp_min=a_log_clamp_min,
        )
        transformer_kwargs = dict(
            d_model=d_model, head_dim=head_dim, expansion_factor=expansion_factor,
            norm=norm, dropout=dropout, use_rope=use_rope,
            use_flash_attn=use_flash_attn, use_gating=use_gating,
            mlp_activation=mlp_activation, mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout, fused_mlp=fused_mlp,
            use_checkpoint=checkpoint_blocks, residual_in_fp32=residual_in_fp32,
            use_enformer_bias=use_enformer_bias, pos_emb_dim=pos_emb_dim,
        )

        for _ in range(n_blocks):
            if mode == 'striped':
                for _ in range(ssm_per_transformer):
                    self.hydra_blocks.append(HydraBlock(**hydra_kwargs))
            elif mode == 'mamba_only':
                self.hydra_blocks.append(HydraBlock(**hydra_kwargs))

            if mode in ('striped', 'transformer_only'):
                self.transformer_blocks.append(TransformerBlock(**transformer_kwargs))

            if mode == 'striped' and transformer_pooling > 1:
                self.trans_down.append(
                    DownsampleStack(
                        d_model,
                        transformer_pooling,
                        kernel_size=downsample_kernel_size,
                        channel_scale=transformer_sampling_channel_scale,
                        start_channels=transformer_sampling_start_channels,
                        grow_channels=transformer_sampling_grow_channels,
                        norm_type=sampling_norm_type,
                        use_weight_std=sampling_use_weight_std,
                        use_checkpoint=sampling_checkpoint,
                    )
                )
                self.trans_up.append(
                    UpsampleStack(
                        d_model,
                        transformer_pooling,
                        kernel_size=downsample_kernel_size,
                        channel_scale=transformer_sampling_channel_scale,
                        start_channels=transformer_sampling_start_channels,
                        grow_channels=transformer_sampling_grow_channels,
                        residual_scale_init=upsample_residual_scale_init,
                        norm_type=sampling_norm_type,
                        use_weight_std=sampling_use_weight_std,
                        use_checkpoint=sampling_checkpoint,
                    )
                )

        d_out = global_sampling_start_channels if global_pooling > 1 else d_model
        self.final_norm = _build_norm(d_out, norm)
        

        if zero_linear_biases:
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

        if rescale_prenorm_residual:
            if mode == 'mamba_only':
                n_residuals = n_blocks * (2 if hydra_use_mlp else 1)
            elif mode == 'transformer_only':
                n_residuals = n_blocks * 2
            else:  # striped
                n_residuals = n_blocks * ((2 if hydra_use_mlp else 1) * ssm_per_transformer + 2)

            for name, p in self.named_parameters():
                if name.endswith('out_proj.weight') or name.endswith('fc2.weight'):
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals)

    @property
    def metrics(self):
        if not self.monitor_a_log or not self.hydra_blocks:
            return {}
        a_log_min = min(b.mixer.A_log.min().item() for b in self.hydra_blocks)
        return {"a_log_min": a_log_min}

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model) from encoder

        Returns:
            (B, L, d_model) final embedding
        """
        #first conv block
        x = self.conv_block(x) + x

        # 1) Global downsample
        global_intermediates = None
        if self.global_down is not None:
            x, global_intermediates = self.global_down(x)

        # 2) Striped blocks
        # For mamba_only with residual_in_fp32, thread a single fp32 residual across
        # all blocks instead of casting back to bf16 at each block boundary. Not possible striped because CNN breaks it
        cross_block_residual = None
        if self.mode in ('mamba_only', 'transformer_only') and self.residual_in_fp32:
            cross_block_residual = x.float()

        for i in range(self.n_blocks):
            if self.mode == 'striped':
                # Hydra at global-pooled resolution
                for j in range(self.ssm_per_transformer):
                    x = self.hydra_blocks[i * self.ssm_per_transformer + j](x)
                # Downsample for transformer
                if self.transformer_pooling > 1:
                    x, trans_intermediates = self.trans_down[i](x)
                # Transformer at further compressed resolution
                x = self.transformer_blocks[i](x)
                # Upsample back with skip connections
                if self.transformer_pooling > 1:
                    x = self.trans_up[i](x, trans_intermediates)

            elif self.mode == 'mamba_only':
                if self.residual_in_fp32:
                    cross_block_residual = self.hydra_blocks[i].forward_with_residual(cross_block_residual)
                else:
                    x = self.hydra_blocks[i](x)

            elif self.mode == 'transformer_only':
                if self.residual_in_fp32:
                    cross_block_residual = self.transformer_blocks[i].forward_with_residual(cross_block_residual)
                else:
                    x = self.transformer_blocks[i](x)

        # Flush cross-block residual back to model dtype before upsample / norm.
        if cross_block_residual is not None:
            x = cross_block_residual.to(x.dtype)

        # 3) Global upsample
        if self.global_up is not None:
            x = self.global_up(x, global_intermediates)

        x = self.final_norm(x)
        return x