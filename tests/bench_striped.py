"""Benchmark the StripedMambaBackbone in all three modes, against pure Caduceus.

Answers four questions:
  1. Where does time go in each mode (`mamba_only`, `striped`, `transformer_only`)?
  2. Is Hydra (bidirectional Mamba-2 SSD) actually a better implementation of
     bidirectionality than Caduceus' two-Mamba `BiMambaWrapper`?
  3. What does the currently-dormant `global_pooling` U-Net actually buy?
  4. Where is the O(L^2) attention wall, i.e. which modes are even usable at L=524288?

Config mirrors slurm_scripts/train_joint_gm12878_striped.sh:
    d_model=256 n_blocks=16 expand=2 d_state=64 d_conv=5 headdim=64 ngroups=1
    chunk_size=256 use_mem_eff_path=true norm=rms residual_in_fp32=false
with `global_pooling` and `checkpoint_blocks` at their defaults (1 / False), i.e. the
U-Net down/upsampling and block checkpointing that the code supports are *off* in the
production striped config.

    pixi run python -m tests.bench_striped --part blocks
    pixi run python -m tests.bench_striped --part models
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (PROD, env_info, fmt_results_table, measure,
                               print_header, reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

# from configs/model/striped_hydra.yaml + the training script's overrides
STRIPED = dict(
    d_model=256, n_blocks=16, d_state=64, d_conv=5, expand=2, headdim=64, ngroups=1,
    chunk_size=256, use_mem_eff_path=True, norm="rms", residual_in_fp32=False,
    rescale_prenorm_residual=True,
)


def build_striped(mode="mamba_only", n_blocks=None, d_model=None, global_pooling=1,
                  transformer_pooling=1, d_in=None, checkpoint_blocks=False,
                  device="cuda"):
    from src.models.sequence.striped_backbone import StripedMambaBackbone
    cfg = dict(STRIPED)
    cfg.update(mode=mode, global_pooling=global_pooling,
               transformer_pooling=transformer_pooling,
               checkpoint_blocks=checkpoint_blocks)
    if n_blocks is not None:
        cfg["n_blocks"] = n_blocks
    if d_model is not None:
        cfg["d_model"] = d_model
    if d_in is not None:
        cfg["d_in"] = d_in
    return StripedMambaBackbone(**cfg).to(device)


def n_params(m):
    seen, tot = set(), 0
    for p in m.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            tot += p.numel()
    return tot


# ---------------------------------------------------------------------------
# Part 1: per-block scaling. This is what locates the attention wall and shows
# whether one Hydra block really beats one Caduceus BiMamba layer.
# ---------------------------------------------------------------------------
def bench_blocks(seqlens, batch_size=1, iters=3, warmup=1, d_model=256):
    from src.models.nn.blocks import HydraBlock, TransformerBlock
    from tests.bench_models import build_bimamba_layer

    results = []
    print_header(f"PER-BLOCK SCALING  B={batch_size}  d_model={d_model}")

    builders = {
        # one Caduceus layer = two Mamba-1 modules (fwd + rev), d_state=16
        "caduceus_BiMambaWrapper": lambda: build_bimamba_layer(d_model=d_model),
        # one Hydra block = bidirectional Mamba-2 SSD in a single scan, d_state=64
        "striped_HydraBlock": lambda: HydraBlock(
            d_model=d_model, d_state=STRIPED["d_state"], d_conv=STRIPED["d_conv"],
            expand=STRIPED["expand"], headdim=STRIPED["headdim"],
            ngroups=STRIPED["ngroups"], norm="rms",
            chunk_size=STRIPED["chunk_size"],
            use_mem_eff_path=STRIPED["use_mem_eff_path"]).cuda(),
        "striped_TransformerBlock": lambda: TransformerBlock(
            d_model=d_model, head_dim=64, norm="rms", use_rope=True,
            use_flash_attn=True, use_gating=True).cuda(),
    }

    # the U-Net sampling stacks, which are what `global_pooling` adds at full resolution
    from src.models.nn.sampling import DownsampleStack, UpsampleStack

    class _SampleRoundTrip(torch.nn.Module):
        """down(L -> L/f) then up(L/f -> L) with skips, i.e. everything global_pooling
        costs on top of running the blocks at the reduced length."""
        def __init__(self, d_model, factor):
            super().__init__()
            # start_channels defaults to d_model // channel_scale, which would expect a
            # narrower input than d_model; pin it so this is a true round-trip on d_model
            self.down = DownsampleStack(d_model, factor, kernel_size=5, norm_type="rms",
                                        start_channels=d_model)
            self.up = UpsampleStack(d_model, factor, kernel_size=5, norm_type="rms",
                                    start_channels=d_model)

        def forward(self, x):
            x, inter = self.down(x)
            return self.up(x, inter)

    for f in (2, 4, 8):
        builders[f"striped_unet_roundtrip_x{f}"] = (
            lambda f=f: _SampleRoundTrip(d_model, f).cuda())

    for name, build in builders.items():
        try:
            mod = build()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: build failed -- {type(e).__name__}: {e}")
            continue
        print(f"\n  {name}  ({n_params(mod):,} params)")
        for L in seqlens:
            # attention is O(L^2); skip the cells that would take minutes
            if "Transformer" in name and L * batch_size > 262144:
                print(f"    L={L:>7}  skipped (O(L^2): would take minutes)")
                continue
            x = torch.randn(batch_size, L, d_model, device="cuda", requires_grad=True)

            def fwd():
                with AUTOCAST():
                    return mod(x)

            r = measure(name, fwd, seqlen=L, batch_size=batch_size,
                        variant="block", warmup=warmup, iters=iters)
            results.append(r)
            if r.oom:
                print(f"    L={L:>7}  OOM")
            elif r.error:
                print(f"    L={L:>7}  ERROR {r.error[:60]}")
            else:
                print(f"    L={L:>7}  fwd {r.fwd_ms:8.2f}  bwd {r.bwd_ms:8.2f}  "
                      f"step {r.step_ms:8.2f} ms  {r.peak_alloc_gb:6.2f} GB")
            del x
            reset_memory()
        del mod
        reset_memory()
    return results


# ---------------------------------------------------------------------------
# Part 2: full backbones, head to head
# ---------------------------------------------------------------------------
def bench_models(seqlens, configs, batch_size=1, iters=3, warmup=1):
    results = []
    print_header("FULL BACKBONE HEAD-TO-HEAD")
    for label, spec in configs:
        try:
            if spec["kind"] == "caduceus":
                from tests.bench_models import build_backbone, make_caduceus_config
                cfg = make_caduceus_config(n_layer=spec.get("n_blocks", 16))
                cfg.gradient_checkpointing = spec.get("checkpoint", False)
                model = build_backbone(cfg)
                d_model = cfg.d_model
                call = lambda x: model(None, inputs_embeds=x)[0]
            else:
                model = build_striped(**{k: v for k, v in spec.items() if k != "kind"})
                d_model = spec.get("d_in") or STRIPED["d_model"]
                call = lambda x: model(x)
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: build failed -- {type(e).__name__}: {e}")
            continue

        print(f"\n  {label}  ({n_params(model):,} params)")
        for L in seqlens:
            x = torch.randn(batch_size, L, d_model, device="cuda", requires_grad=True)

            def fwd():
                with AUTOCAST():
                    return call(x)

            r = measure(label, fwd, seqlen=L, batch_size=batch_size,
                        variant=label, warmup=warmup, iters=iters,
                        extra={"params": n_params(model)})
            results.append(r)
            if r.oom:
                print(f"    L={L:>7}  OOM")
            elif r.error:
                print(f"    L={L:>7}  ERROR {r.error[:70]}")
            else:
                print(f"    L={L:>7}  fwd {r.fwd_ms:9.2f}  bwd {r.bwd_ms:9.2f}  "
                      f"step {r.step_ms:9.2f} ms  {r.tokens_per_s_step:>11,.0f} tok/s  "
                      f"{r.peak_alloc_gb:6.2f} GB")
            del x
            reset_memory()
        del model
        reset_memory()
    return results


def model_configs(which="core"):
    """Named backbone configurations to compare."""
    core = [
        ("caduceus_bimamba", dict(kind="caduceus")),
        ("striped_mamba_only_gp1", dict(kind="striped", mode="mamba_only")),
        ("striped_mamba_only_gp1_ckpt",
         dict(kind="striped", mode="mamba_only", checkpoint_blocks=True)),
    ]
    pooling = [
        ("striped_mamba_only_gp2", dict(kind="striped", mode="mamba_only", global_pooling=2)),
        ("striped_mamba_only_gp4", dict(kind="striped", mode="mamba_only", global_pooling=4)),
        ("striped_mamba_only_gp8", dict(kind="striped", mode="mamba_only", global_pooling=8)),
        # the intended U-Net shape: cheap channels at full resolution, d_model after pooling
        ("striped_mamba_only_gp8_din64",
         dict(kind="striped", mode="mamba_only", global_pooling=8, d_in=64)),
    ]
    hybrid = [
        # attention needs aggressive pooling to be affordable at this length.
        # NOTE on fairness: mode='striped' builds n_blocks Hydra *and* n_blocks
        # Transformer blocks, so at n_blocks=16 it is 32 blocks / ~81M params against
        # mamba_only's 16 blocks / ~7.9M. `_nb8` halves it to roughly match depth.
        ("striped_gp8_tp8", dict(kind="striped", mode="striped",
                                 global_pooling=8, transformer_pooling=8)),
        ("striped_gp8_tp8_nb8", dict(kind="striped", mode="striped", n_blocks=8,
                                     global_pooling=8, transformer_pooling=8)),
        ("striped_gp8_tp16", dict(kind="striped", mode="striped",
                                  global_pooling=8, transformer_pooling=16)),
        ("transformer_only_gp16", dict(kind="striped", mode="transformer_only",
                                       global_pooling=16)),
        ("transformer_only_gp32", dict(kind="striped", mode="transformer_only",
                                       global_pooling=32)),
    ]
    return {"core": core, "pooling": pooling, "hybrid": hybrid,
            "all": core + pooling + hybrid}[which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="models",
                    choices=["blocks", "models", "all"])
    ap.add_argument("--configs", default="core",
                    choices=["core", "pooling", "hybrid", "all"])
    ap.add_argument("--seqlens", type=int, nargs="*", default=[524288])
    ap.add_argument("--block-seqlens", type=int, nargs="*",
                    default=[16384, 65536, 131072, 262144, 524288])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="tests/results/striped.json")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")

    results = []
    if args.part in ("blocks", "all"):
        results += bench_blocks(args.block_seqlens, args.batch_size, args.iters, args.warmup)
    if args.part in ("models", "all"):
        results += bench_models(args.seqlens, model_configs(args.configs),
                                args.batch_size, args.iters, args.warmup)

    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"part": args.part, "striped_cfg": STRIPED})


if __name__ == "__main__":
    main()
