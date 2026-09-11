"""Whole-model effect of the RMSBatchNorm1d rewrite, pooled and non-pooled.

The rewrite measures 1.31-1.50x on the U-Net sampling stacks in isolation. But
`RMSBatchNorm1d` is only used inside `ConvBlock`, and `ConvBlock` appears in:

  * `StripedMambaBackbone.conv_block`  -- one instance, always present
  * `DownsampleLayer` / `UpsampleLayer` -- only when global_pooling > 1

so at global_pooling=1 there is a single ConvBlock against 16 Hydra blocks, and the
rewrite should do almost nothing. This measures both regimes on the full backbone at the
production shape rather than assuming.

    pixi run python -m tests.bench_norm_patch_model --seqlens 524288
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def patched_norm_forward(self, x):
    """RMSBatchNorm1d.forward with torch.var's fused reduction replacing the
    materialized (x - mean).pow(2), and the normalization folded with weight/bias into a
    single addcmul. Equivalent to rel_err <= 8.1e-08."""
    if self.training:
        var = torch.var(x, dim=[0, 1], unbiased=False)
        with torch.no_grad():
            self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
    else:
        var = self.running_var
    scale = self.weight / (var + self.eps).sqrt()
    return torch.addcmul(self.bias, x, scale)


CONFIGS = [
    ("mamba_only gp=1  [current striped config]", dict(mode="mamba_only")),
    ("mamba_only gp=8 d_in=64  [recommended]",
     dict(mode="mamba_only", global_pooling=8, d_in=64)),
    ("mamba_only gp=8", dict(mode="mamba_only", global_pooling=8)),
]


def main():
    from src.models.nn.sampling import RMSBatchNorm1d
    from tests.bench_striped import build_striped, n_params

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="*", default=[524288])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="tests/results/norm_patch_model.json")
    args = ap.parse_args()

    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")

    results, pairs = [], {}
    for label, spec in CONFIGS:
        d_model = spec.get("d_in") or 256
        for patched in (False, True):
            orig = RMSBatchNorm1d.forward
            if patched:
                RMSBatchNorm1d.forward = patched_norm_forward
            try:
                model = build_striped(**spec)
                n_cb = sum(1 for m in model.modules() if isinstance(m, RMSBatchNorm1d))
                tag = "norm patch" if patched else "current   "
                print_header(f"{label}   [{tag}]   {n_params(model):,} params, "
                             f"{n_cb} RMSBatchNorm1d instances")
                for L in args.seqlens:
                    x = torch.randn(args.batch_size, L, d_model, device="cuda",
                                    requires_grad=True)

                    def fwd():
                        with AUTOCAST():
                            return model(x)

                    r = measure(f"{label} [{tag}]", fwd, seqlen=L,
                                batch_size=args.batch_size, variant=tag.strip(),
                                warmup=args.warmup, iters=args.iters,
                                extra={"n_rmsbatchnorm": n_cb})
                    results.append(r)
                    pairs.setdefault((label, L), {})[patched] = r
                    if r.oom:
                        print(f"    L={L:>7}  OOM")
                    elif r.error:
                        print(f"    L={L:>7}  ERROR {r.error[:60]}")
                    else:
                        print(f"    L={L:>7}  fwd {r.fwd_ms:9.2f}  bwd {r.bwd_ms:9.2f}  "
                              f"step {r.step_ms:9.2f} ms  {r.tokens_per_s_step:>11,.0f} tok/s"
                              f"  {r.peak_alloc_gb:6.2f} GB")
                    del x
                    reset_memory()
                del model
                reset_memory()
            finally:
                RMSBatchNorm1d.forward = orig

    print_header("WHOLE-MODEL EFFECT OF THE NORM REWRITE")
    print(f"{'config':<46} {'L':>8} {'before':>11} {'after':>11} {'speedup':>9} {'memory':>18}")
    for (label, L), d in pairs.items():
        a, b = d.get(False), d.get(True)
        if not a or not b or a.oom or b.oom or a.error or b.error:
            print(f"{label:<46} {L:>8}   incomplete")
            continue
        print(f"{label:<46} {L:>8} {a.step_ms:>9.1f}ms {b.step_ms:>9.1f}ms "
              f"{a.step_ms / b.step_ms:>8.3f}x {a.peak_alloc_gb:>7.2f} -> {b.peak_alloc_gb:.2f} GB")

    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"probe": "norm_patch_whole_model"})


if __name__ == "__main__":
    main()
