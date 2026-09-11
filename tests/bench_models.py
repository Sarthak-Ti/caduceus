"""Model builders for the Mamba/Caduceus benchmarks.

Builds the *real* production modules (not re-implementations) so the numbers are
faithful to slurm_scripts/train_joint_gm12878_mlmweight.sh with dataset.mlm=0.

Precision: Lightning's `trainer.precision=bf16` is aliased to "bf16-mixed"
(lightning_fabric/plugins/precision/precision.py:24), i.e. fp32 parameters run under
`torch.autocast("cuda", torch.bfloat16)`. The harness reproduces that exactly rather
than casting the model to bf16, which would be a different (and faster) regime.
"""

import torch
import torch.nn as nn

from tests.bench_utils import PROD, prod_ssm_cfg

from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import CaduceusMixerModel, BiMambaWrapper
from src.tasks.encoders import JointCNN
from src.tasks.decoders import JointMaskingDecoder
from src.tasks.metrics import joint_loss


def make_caduceus_config(d_model=None, n_layer=None, bidirectional=True,
                         bidirectional_strategy="add", bidirectional_weight_tie=True,
                         rcps=False, **ssm_overrides):
    ssm_cfg = prod_ssm_cfg()
    ssm_cfg.update(ssm_overrides)
    return CaduceusConfig(
        d_model=d_model or PROD["d_model"],
        n_layer=n_layer or PROD["n_layer"],
        vocab_size=1,
        pad_vocab_size_multiple=1,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=False,
        norm_epsilon=1e-5,
        initializer_cfg={"initializer_range": 0.02, "rescale_prenorm_residual": True,
                         "n_residuals_per_layer": 1},
        bidirectional=bidirectional,
        bidirectional_strategy=bidirectional_strategy,
        bidirectional_weight_tie=bidirectional_weight_tie,
        rcps=rcps,
        skip_embedding=True,
    )


def build_backbone(config=None, device="cuda"):
    """CaduceusMixerModel only. Input is fed via `inputs_embeds` (B, L, d_model),
    matching what the JointCNN encoder + skip_embedding Transpose hands it."""
    config = config or make_caduceus_config()
    model = CaduceusMixerModel(config, device=device, dtype=torch.float32)
    return model.to(device)


def build_bimamba_layer(d_model=None, device="cuda", **kw):
    d_model = d_model or PROD["d_model"]
    ssm_cfg = prod_ssm_cfg()
    ssm_cfg.update(kw.pop("ssm_overrides", {}))
    return BiMambaWrapper(
        d_model=d_model,
        bidirectional=kw.pop("bidirectional", True),
        bidirectional_strategy=kw.pop("bidirectional_strategy", "add"),
        bidirectional_weight_tie=kw.pop("bidirectional_weight_tie", True),
        **ssm_cfg,
    ).to(device)


class JointStep(nn.Module):
    """encoder -> backbone -> decoder -> joint_loss, i.e. one full training step
    of the production model with dataset.mlm=0 / task2=reg / count_weight=16."""

    def __init__(self, config=None, device="cuda", count_weight=16):
        super().__init__()
        self.config = config or make_caduceus_config()
        self.count_weight = count_weight
        # encoder.joint=true, encoder.norm=true -> InstanceNorm1d, activation default ReLU
        self.encoder = JointCNN(
            d_model=self.config.d_model,
            d_input1=PROD["d_input1"],
            d_input2=PROD["d_input2"],
            joint=True,
            kernel_size=PROD["encoder_kernel_size"],
            norm=True,
            acc_type="continuous",
        )
        self.backbone = CaduceusMixerModel(self.config, device=device, dtype=torch.float32)
        self.decoder = JointMaskingDecoder(
            d_model=self.config.d_model,
            d_output1=PROD["d_output1"],
            d_output2=PROD["d_output2"],
        )
        self.to(device)

    def forward(self, x1, x2, y_seq, y_acc):
        emb, _ = self.encoder(x1, x2)          # (B, d_model, L)
        # Pass (B, d_model, L) straight in as input_ids: with skip_embedding=True the
        # backbone's CaduceusEmbeddings is Transpose(0, 2, 1), which is what produces the
        # (B, L, d_model) the blocks expect. Permuting here first would transpose twice.
        hidden, _ = self.backbone(emb)
        seq_logits, acc_pred = self.decoder(hidden)
        return joint_loss((seq_logits, acc_pred), (y_seq, y_acc),
                          task1="mlm", task2="reg", count_weight=self.count_weight)


def make_backbone_input(seqlen, batch_size=1, d_model=None, device="cuda", requires_grad=True):
    d_model = d_model or PROD["d_model"]
    x = torch.randn(batch_size, seqlen, d_model, device=device, dtype=torch.float32)
    x.requires_grad_(requires_grad)
    return x


def make_joint_batch(seqlen, batch_size=1, device="cuda", acc_mask_frac=0.05, seq_mask_frac=0.0):
    """Synthetic batch with the shapes the real dataloader produces.

    x1: (B, d_input1=6, L)  one-hot sequence + mask channels
    x2: (B, d_input2=2, L)  accessibility value + mask channel
    y_seq: (B, L, d_output1=5)  one-hot target + trailing mask channel -> 6 channels
    y_acc: (B, L, 2)  target + mask channel

    dataset.mlm=0 means no sequence positions are masked, so ce_loss_mask_seq sees an
    empty mask; acc_mlm=0.25 with acc_mask_size=500 masks a contiguous-ish fraction of
    accessibility positions. `acc_mask_frac` approximates that.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    bases = torch.randint(0, 4, (batch_size, seqlen), generator=g)
    x1 = torch.zeros(batch_size, PROD["d_input1"], seqlen)
    x1.scatter_(1, bases.unsqueeze(1), 1.0)
    x2 = torch.zeros(batch_size, PROD["d_input2"], seqlen)
    x2[:, 0] = torch.rand(batch_size, seqlen, generator=g) * 3.0

    acc_mask = (torch.rand(batch_size, seqlen, generator=g) < acc_mask_frac).float()
    x2[:, 1] = acc_mask
    y_acc = torch.stack([torch.rand(batch_size, seqlen, generator=g) * 3.0, acc_mask], dim=-1)

    # seq target: d_output1=5 classes + 1 trailing mask channel
    y_seq = torch.zeros(batch_size, seqlen, PROD["d_output1"] + 1)
    y_seq.scatter_(2, bases.unsqueeze(-1), 1.0)
    if seq_mask_frac > 0:
        y_seq[..., -1] = (torch.rand(batch_size, seqlen, generator=g) < seq_mask_frac).float()

    to = lambda t: t.to(device, non_blocking=True)
    return to(x1), to(x2), to(y_seq), to(y_acc)


def count_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:      # bidirectional_weight_tie shares in_proj / out_proj
            continue
        seen.add(id(p))
        total += p.numel()
    return total
