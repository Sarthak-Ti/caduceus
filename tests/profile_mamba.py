"""Kernel-level profiling of the Caduceus backbone / full training step.

Produces three things:
  1. a self-CUDA-time table over kernels (where the GPU actually spends time)
  2. a `record_function`-annotated breakdown by architectural component
  3. optionally a chrome trace for manual inspection

The `--check-launch-bound` pass re-times the backbone with a much shorter sequence to
separate fixed per-launch/Python overhead from length-proportional GPU work.

Usage
-----
    pixi run python -m tests.profile_mamba --target backbone --seqlen 131072
"""

import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import PROD, cuda_sync, env_info, print_header, reset_memory

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def annotate_backbone(model):
    """Wrap each BiMambaWrapper sub-part in record_function so the profiler can
    attribute time to fwd-scan / rev-scan / flips rather than one opaque blob."""
    from caduceus.modeling_caduceus import BiMambaWrapper

    for module in model.modules():
        if not isinstance(module, BiMambaWrapper):
            continue
        if getattr(module, "_annotated", False):
            continue
        module._annotated = True
        orig_fwd = module.forward

        def make(mod, orig):
            def fwd(hidden_states, inference_params=None):
                with record_function("bimamba/forward_dir"):
                    out = mod.mamba_fwd(hidden_states, inference_params=inference_params)
                if not mod.bidirectional:
                    return out
                with record_function("bimamba/flip_in"):
                    h_rev = hidden_states.flip(dims=(1,))
                with record_function("bimamba/reverse_dir"):
                    out_rev = mod.mamba_rev(h_rev, inference_params=inference_params)
                with record_function("bimamba/flip_out"):
                    out_rev = out_rev.flip(dims=(1,))
                with record_function("bimamba/combine"):
                    if mod.bidirectional_strategy == "add":
                        return out + out_rev
                    return out * out_rev
            return fwd

        module.forward = make(module, orig_fwd)
    return model


def annotate_striped(model):
    """Attribute time to Hydra blocks vs Transformer blocks vs the U-Net sampling
    stacks, which is the split that decides which striped mode is affordable."""
    from src.models.nn.blocks import HydraBlock, TransformerBlock
    from src.models.nn.sampling import DownsampleStack, UpsampleStack

    kinds = ((HydraBlock, "striped/hydra_block"),
             (TransformerBlock, "striped/transformer_block"),
             (DownsampleStack, "striped/downsample"),
             (UpsampleStack, "striped/upsample"))
    for mod in model.modules():
        for cls, label in kinds:
            if isinstance(mod, cls) and not getattr(mod, "_annotated", False):
                mod._annotated = True
                orig = mod.forward

                def make(orig=orig, label=label):
                    def fwd(*a, **kw):
                        with record_function(label):
                            return orig(*a, **kw)
                    return fwd

                mod.forward = make()
    return model


def build_target(target, seqlen, batch_size, n_layer=None, striped_kwargs=None):
    from tests.bench_models import (
        JointStep, build_backbone, make_caduceus_config, make_joint_batch,
    )
    cfg = make_caduceus_config(n_layer=n_layer)
    if target == "backbone":
        model = annotate_backbone(build_backbone(cfg))
        x = torch.randn(batch_size, seqlen, cfg.d_model, device="cuda", requires_grad=True)

        def step():
            with AUTOCAST():
                with record_function("backbone/forward"):
                    out, _ = model(None, inputs_embeds=x)
                loss = out.float().sum()
            with record_function("backbone/backward"):
                loss.backward()
        return model, step

    if target.startswith("striped"):
        from tests.bench_striped import build_striped
        model = annotate_striped(build_striped(**(striped_kwargs or {})))
        x = torch.randn(batch_size, seqlen,
                        (striped_kwargs or {}).get("d_in") or 256,
                        device="cuda", requires_grad=True)

        def step():
            with AUTOCAST():
                with record_function("striped/forward"):
                    out = model(x)
                loss = out.float().sum()
            with record_function("striped/backward"):
                loss.backward()
        return model, step

    model = JointStep(cfg)
    annotate_backbone(model.backbone)
    x1, x2, y_seq, y_acc = make_joint_batch(seqlen, batch_size)

    def step():
        with AUTOCAST():
            with record_function("full/forward"):
                loss = model(x1, x2, y_seq, y_acc)
        with record_function("full/backward"):
            loss.backward()
    return model, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="backbone",
                    choices=["backbone", "full", "striped"])
    ap.add_argument("--mode", default="mamba_only",
                    choices=["mamba_only", "striped", "transformer_only"],
                    help="striped target only")
    ap.add_argument("--global-pooling", type=int, default=1)
    ap.add_argument("--transformer-pooling", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=131072)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--row-limit", type=int, default=40)
    ap.add_argument("--trace", default=None, help="path to write a chrome trace")
    ap.add_argument("--check-launch-bound", action="store_true",
                    help="also profile at L=4096 to expose fixed per-launch overhead")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")

    def profile_at(seqlen, label):
        model, step = build_target(
            args.target, seqlen, args.batch_size, args.n_layer,
            striped_kwargs=dict(mode=args.mode, global_pooling=args.global_pooling,
                                transformer_pooling=args.transformer_pooling))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(args.warmup):
            opt.zero_grad(set_to_none=True)
            step()
        cuda_sync()
        reset_memory()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as prof:
            for _ in range(args.iters):
                opt.zero_grad(set_to_none=True)
                step()
            cuda_sync()

        print_header(f"{label}  target={args.target}  L={seqlen}  B={args.batch_size}  "
                     f"({args.iters} iters)")
        print("\n--- top kernels by self CUDA time ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                        row_limit=args.row_limit))
        print("\n--- component breakdown (record_function scopes) ---")
        keys = prof.key_averages()
        rows = [e for e in keys
                if e.key.startswith(("bimamba/", "backbone/", "full/", "striped/"))]
        total = sum(e.cuda_time_total for e in rows if e.key.endswith("forward")) or 1
        print(f"{'scope':<28} {'cuda ms/iter':>14} {'cpu ms/iter':>14} {'calls':>8}")
        for e in sorted(rows, key=lambda x: -x.cuda_time_total):
            print(f"{e.key:<28} {e.cuda_time_total/1000/args.iters:>14.2f} "
                  f"{e.cpu_time_total/1000/args.iters:>14.2f} {e.count:>8}")

        print("\n--- CPU-side launch overhead ---")
        cpu_total = sum(e.self_cpu_time_total for e in keys)
        cuda_total = sum(e.self_cuda_time_total for e in keys)
        print(f"  total self CPU time : {cpu_total/1000/args.iters:10.2f} ms/iter")
        print(f"  total self CUDA time: {cuda_total/1000/args.iters:10.2f} ms/iter")
        print(f"  CPU/CUDA ratio      : {cpu_total/max(cuda_total,1):10.3f}  "
              f"(>1 means the GPU is starved by Python/launch overhead)")

        if args.trace:
            path = args.trace if seqlen == args.seqlen else args.trace.replace(".json", f".L{seqlen}.json")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            prof.export_chrome_trace(path)
            print(f"\n[profile] trace -> {path}")
        del model, opt
        reset_memory()

    profile_at(args.seqlen, "PROFILE")
    if args.check_launch_bound:
        profile_at(4096, "PROFILE (short sequence: exposes fixed overhead)")


if __name__ == "__main__":
    main()
