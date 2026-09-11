"""`RMSBatchNorm1d` is 25% of a ConvBlock, and ConvBlocks are ~all of the U-Net stacks.

Component timing (A100, L=262144, C=256) put the norm at 9.12 ms of a 36.92 ms
ConvBlock(w5) -- more than the convolution itself (5.91 ms). The reason is in
`src/models/nn/sampling.py`:

    mean = x.mean(dim=[0, 1])                     # full-size pass
    var  = (x - mean).pow(2).mean(dim=[0, 1])     # materializes TWO full-size temporaries
    x    = x / (var + eps).sqrt()                 # full-size pass
    return x * self.weight + self.bias            # two more full-size passes

Three rewrites, all computing the same function:

  V1  torch.var(..., unbiased=False) -- one fused reduction, no full-size temporaries.
  V2  fold the normalization, weight and bias into one affine: precompute
      scale = weight / sqrt(var + eps) as a (C,) vector, then a single fused
      multiply-add over x instead of divide -> multiply -> add.
  V3  both.

Equivalence is asserted against the current implementation before timing.

    pixi run python -m tests.bench_norm_opt --seqlen 262144
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def current(x, weight, bias, eps):
    mean = x.mean(dim=[0, 1])
    var = (x - mean).pow(2).mean(dim=[0, 1])
    x = x / (var + eps).sqrt()
    return x * weight + bias


def v1_fused_var(x, weight, bias, eps):
    # torch.var with unbiased=False is exactly E[(x-mu)^2], in one fused reduction
    var = torch.var(x, dim=[0, 1], unbiased=False)
    x = x / (var + eps).sqrt()
    return x * weight + bias


def v2_fused_affine(x, weight, bias, eps):
    mean = x.mean(dim=[0, 1])
    var = (x - mean).pow(2).mean(dim=[0, 1])
    scale = weight / (var + eps).sqrt()           # (C,), computed on a tiny tensor
    return torch.addcmul(bias, x, scale)          # one fused multiply-add over x


def v3_both(x, weight, bias, eps):
    var = torch.var(x, dim=[0, 1], unbiased=False)
    scale = weight / (var + eps).sqrt()
    return torch.addcmul(bias, x, scale)


VARIANTS = {
    "current (5 passes, 2 temporaries)": current,
    "V1 torch.var fused reduction": v1_fused_var,
    "V2 folded affine (addcmul)": v2_fused_affine,
    "V3 both": v3_both,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="tests/results/norm_opt.json")
    args = ap.parse_args()

    B, L, C = args.batch_size, args.seqlen, args.channels
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"RMSBatchNorm1d REWRITES  B={B}  L={L}  C={C}")

    weight = torch.ones(C, device="cuda", requires_grad=True)
    bias = torch.zeros(C, device="cuda", requires_grad=True)
    eps = 1e-5

    # --- equivalence first, in fp32 so the comparison is not swamped by bf16 noise ---
    xs = torch.randn(B, min(L, 16384), C, device="cuda")
    ref = current(xs, weight, bias, eps).detach()
    for name, fn in VARIANTS.items():
        got = fn(xs, weight, bias, eps).detach()
        err = ((got - ref).norm() / ref.norm()).item()
        print(f"  {name:<38} rel_err vs current = {err:.3e}")
        assert err < 1e-5, f"{name} is NOT equivalent (rel_err={err:.3e})"
    print("  all rewrites equivalent\n")
    del xs, ref
    reset_memory()

    results = []
    x = torch.randn(B, L, C, device="cuda", requires_grad=True)
    for name, fn in VARIANTS.items():
        def fwd(fn=fn):
            with AUTOCAST():
                return fn(x, weight, bias, eps)
        r = measure(name, fwd, seqlen=L, batch_size=B, variant="norm",
                    warmup=args.warmup, iters=args.iters)
        results.append(r)
        if r.oom or r.error:
            print(f"  {name:<38} {'OOM' if r.oom else r.error[:55]}")
        else:
            print(f"  {name:<38} fwd {r.fwd_ms:7.2f}  bwd {r.bwd_ms:7.2f}  "
                  f"step {r.step_ms:7.2f} ms  {r.peak_alloc_gb:5.2f} GB")
        reset_memory()

    print_header("SUMMARY")
    print(fmt_results_table(results))
    ok = [r for r in results if not r.oom and not r.error]
    if ok:
        base = ok[0]
        print(f"\nrelative to current ({base.step_ms:.2f} ms, {base.peak_alloc_gb:.2f} GB):")
        for r in ok[1:]:
            print(f"  {r.name:<38} {base.step_ms / r.step_ms:5.2f}x faster, "
                  f"{r.peak_alloc_gb:.2f} GB")
    save_results(results, args.out, meta={"probe": "norm_opt"})


if __name__ == "__main__":
    main()
