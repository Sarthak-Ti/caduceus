"""Concrete optimizations for the U-Net upsample stack.

Once `global_pooling` is on, `upsample` is 50% of the striped forward and 2.7x the cost
of `downsample` (profiler, gp8/tp8, L=262144). The asymmetry is structural:

    DownsampleLayer:  2 x ConvBlock(width 5) at length L, then MaxPool -> L/2
    UpsampleLayer:    1 x ConvBlock(width 5) at L
                      repeat_interleave(2, dim=1)          -> length 2L
                      * residual_scale                      (elementwise at 2L)
                      + ConvBlock(width 1)(skip)            (at 2L)
                      + ConvBlock(width 5)                  (at 2L)

so the upsample side runs two conv blocks at the *doubled* resolution plus a gather,
where the downsample side runs both of its convs before pooling.

Three candidate fixes, all exactly equivalent (this file measures them; correctness is
asserted inline):

  A. `repeat_interleave(2, dim=1)` -> `unsqueeze/expand/reshape`.
     repeat_interleave dispatches a gather kernel whose backward is a scatter-add;
     expand+reshape is one contiguous copy whose backward is a plain sum reduction.
  B. Apply `residual_scale` *before* the upsample, so the elementwise pass runs on half
     the elements. A scalar multiply commutes with repetition.
  C. Both.

    pixi run python -m tests.bench_upsample_opt --seqlen 262144
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def up_repeat_interleave(x, scale):
    return x.repeat_interleave(2, dim=1) * scale


def up_expand(x, scale):
    b, l, c = x.shape
    return x.unsqueeze(2).expand(b, l, 2, c).reshape(b, 2 * l, c) * scale


def up_expand_prescale(x, scale):
    b, l, c = x.shape
    x = x * scale                      # half the elements
    return x.unsqueeze(2).expand(b, l, 2, c).reshape(b, 2 * l, c)


def up_repeat_prescale(x, scale):
    return (x * scale).repeat_interleave(2, dim=1)


VARIANTS = {
    "A0 repeat_interleave + post-scale [current]": up_repeat_interleave,
    "A1 expand/reshape + post-scale": up_expand,
    "B  repeat_interleave + pre-scale": up_repeat_prescale,
    "C  expand/reshape + pre-scale": up_expand_prescale,
}



# ---------------------------------------------------------------------------
# The op-level number above is only actionable if we know what fraction of a real
# UpsampleLayer/UpsampleStack the repeat+scale actually is. This measures the real
# modules, patched vs unpatched, plus a breakdown of one layer's four components.
# ---------------------------------------------------------------------------
def _patched_forward_body(self, x, skip):
    """UpsampleLayer._forward_body with the scale applied before the repeat.

    Bitwise identical: each output element is the single product x[i] * residual_scale
    either way, just computed before duplication rather than after."""
    from src.models.nn.sampling import _residual_pad_or_crop
    out = self.reduce_main(x) + _residual_pad_or_crop(x, skip.shape[-1])
    out = (out * self.residual_scale).repeat_interleave(2, dim=1)
    out = out + self.skip_project(skip)
    return out + self.refine(out)


def _ap(fn):
    with AUTOCAST():
        return fn()


def bench_real_modules(seqlen, channels, factor, batch_size, iters, warmup):
    from src.models.nn.sampling import UpsampleStack, DownsampleStack, UpsampleLayer

    results = []
    print_header(f"REAL U-NET MODULES  B={batch_size}  L={seqlen}  C={channels}  factor={factor}")

    down = DownsampleStack(channels, factor, kernel_size=5, norm_type="rms",
                           start_channels=channels).cuda()
    up = UpsampleStack(channels, factor, kernel_size=5, norm_type="rms",
                       start_channels=channels).cuda()
    x = torch.randn(batch_size, seqlen, channels, device="cuda", requires_grad=True)
    with torch.no_grad(), AUTOCAST():
        pooled, inter = down(x)
    pooled = pooled.detach().requires_grad_(True)

    def run(name, fn):
        r = measure(name, fn, seqlen=seqlen, batch_size=batch_size, variant="unet",
                    warmup=warmup, iters=iters)
        results.append(r)
        if r.oom or r.error:
            print(f"  {name:<44} {'OOM' if r.oom else r.error[:60]}")
        else:
            print(f"  {name:<44} fwd {r.fwd_ms:7.2f}  bwd {r.bwd_ms:7.2f}  "
                  f"step {r.step_ms:7.2f} ms  {r.peak_alloc_gb:5.2f} GB")
        reset_memory()
        return r

    run("DownsampleStack [reference]", lambda: _ap(lambda: down(x)[0]))
    base = run("UpsampleStack  [current]", lambda: _ap(lambda: up(pooled, inter)))

    orig = UpsampleLayer._forward_body
    UpsampleLayer._forward_body = _patched_forward_body
    try:
        patched = run("UpsampleStack  [pre-scale patch]",
                      lambda: _ap(lambda: up(pooled, inter)))
    finally:
        UpsampleLayer._forward_body = orig

    if not (base.oom or patched.oom or base.error or patched.error):
        print("")
        print(f"  pre-scale patch on the REAL stack: "
              f"{base.step_ms / patched.step_ms:.3f}x "
              f"({base.step_ms:.2f} -> {patched.step_ms:.2f} ms, "
              f"{base.peak_alloc_gb:.2f} -> {patched.peak_alloc_gb:.2f} GB)")

    # --- component breakdown of the top (most expensive) upsample layer ---
    layer = up.layers[-1]
    lo = seqlen // 2
    xl = torch.randn(batch_size, lo, channels, device="cuda", requires_grad=True)
    sk = torch.randn(batch_size, seqlen, channels, device="cuda", requires_grad=True)
    scale = layer.residual_scale
    print("")
    print("  top UpsampleLayer components:")
    run("  reduce_main   ConvBlock(w5) @ L/2", lambda: _ap(lambda: layer.reduce_main(xl)))
    run("  repeat+scale  (the op above)",
        lambda: _ap(lambda: xl.repeat_interleave(2, dim=1) * scale))
    run("  skip_project  ConvBlock(w1) @ L", lambda: _ap(lambda: layer.skip_project(sk)))
    run("  refine        ConvBlock(w5) @ L", lambda: _ap(lambda: layer.refine(sk)))
    del down, up, x, pooled, inter, xl, sk
    reset_memory()
    return results



def _patched_norm_forward(self, x):
    """RMSBatchNorm1d.forward with the two rewrites measured in tests/bench_norm_opt.py:
    torch.var's fused reduction instead of materializing (x - mean).pow(2), and the
    normalization folded together with weight/bias into a single addcmul.
    Measured 2.18x on the norm in isolation (9.11 -> 4.17 ms)."""
    if self.training:
        var = torch.var(x, dim=[0, 1], unbiased=False)
        with torch.no_grad():
            self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
    else:
        var = self.running_var
    scale = self.weight / (var + self.eps).sqrt()
    return torch.addcmul(self.bias, x, scale)


def bench_norm_patch_end_to_end(seqlen, channels, factor, batch_size, iters, warmup):
    """The norm rewrite is 2.18x on the norm alone. What is it worth on the real stacks?"""
    from src.models.nn.sampling import (UpsampleStack, DownsampleStack, UpsampleLayer,
                                        RMSBatchNorm1d)

    results = []
    print_header(f"NORM REWRITE, END TO END  L={seqlen}  C={channels}  factor={factor}")

    down = DownsampleStack(channels, factor, kernel_size=5, norm_type="rms",
                           start_channels=channels).cuda()
    up = UpsampleStack(channels, factor, kernel_size=5, norm_type="rms",
                       start_channels=channels).cuda()
    x = torch.randn(batch_size, seqlen, channels, device="cuda", requires_grad=True)
    with torch.no_grad(), AUTOCAST():
        pooled, inter = down(x)
    pooled = pooled.detach().requires_grad_(True)

    def run(name, fn):
        r = measure(name, fn, seqlen=seqlen, batch_size=batch_size, variant="normpatch",
                    warmup=warmup, iters=iters)
        results.append(r)
        if r.oom or r.error:
            print(f"  {name:<44} {'OOM' if r.oom else r.error[:55]}")
        else:
            print(f"  {name:<44} fwd {r.fwd_ms:7.2f}  bwd {r.bwd_ms:7.2f}  "
                  f"step {r.step_ms:7.2f} ms  {r.peak_alloc_gb:5.2f} GB")
        reset_memory()
        return r

    d0 = run("DownsampleStack [current]", lambda: _ap(lambda: down(x)[0]))
    u0 = run("UpsampleStack   [current]", lambda: _ap(lambda: up(pooled, inter)))

    orig_norm = RMSBatchNorm1d.forward
    orig_up = UpsampleLayer._forward_body
    RMSBatchNorm1d.forward = _patched_norm_forward
    try:
        d1 = run("DownsampleStack [norm patch]", lambda: _ap(lambda: down(x)[0]))
        u1 = run("UpsampleStack   [norm patch]", lambda: _ap(lambda: up(pooled, inter)))
        UpsampleLayer._forward_body = _patched_forward_body
        u2 = run("UpsampleStack   [norm + pre-scale]", lambda: _ap(lambda: up(pooled, inter)))
    finally:
        RMSBatchNorm1d.forward = orig_norm
        UpsampleLayer._forward_body = orig_up

    print("")
    for label, a, b in (("DownsampleStack, norm patch", d0, d1),
                        ("UpsampleStack,   norm patch", u0, u1),
                        ("UpsampleStack,   norm + pre-scale", u0, u2)):
        if not (a.oom or b.oom or a.error or b.error):
            print(f"  {label:<38} {a.step_ms / b.step_ms:5.3f}x  "
                  f"({a.step_ms:.2f} -> {b.step_ms:.2f} ms, "
                  f"{a.peak_alloc_gb:.2f} -> {b.peak_alloc_gb:.2f} GB)")
    del down, up, x, pooled, inter
    reset_memory()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144,
                    help="length BEFORE upsampling; output is 2x this")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--real-seqlen", type=int, default=262144)
    ap.add_argument("--out", default="tests/results/upsample_opt.json")
    args = ap.parse_args()

    B, L, C = args.batch_size, args.seqlen, args.channels
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"UPSAMPLE OP  B={B}  L={L} -> {2*L}  C={C}")

    scale = torch.tensor([0.9], device="cuda", requires_grad=True)

    # --- correctness first: every variant must match the current implementation ---
    xc = torch.randn(B, min(L, 8192), C, device="cuda")
    ref = up_repeat_interleave(xc, scale).detach()
    for name, fn in VARIANTS.items():
        got = fn(xc, scale).detach()
        err = ((got - ref).norm() / ref.norm()).item()
        assert err < 1e-6, f"{name} is NOT equivalent (rel_err={err:.3e})"
    print("  all variants numerically identical to the current implementation (rel_err < 1e-6)\n")
    del xc, ref
    reset_memory()

    results = []
    x = torch.randn(B, L, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for name, fn in VARIANTS.items():
        def fwd(fn=fn):
            with AUTOCAST():
                return fn(x, scale)
        r = measure(name, fwd, seqlen=L, batch_size=B, variant="upsample",
                    warmup=args.warmup, iters=args.iters)
        results.append(r)
        if r.oom:
            print(f"  {name:<46} OOM")
        elif r.error:
            print(f"  {name:<46} ERROR {r.error[:60]}")
        else:
            print(f"  {name:<46} fwd {r.fwd_ms:7.2f}  bwd {r.bwd_ms:7.2f}  "
                  f"step {r.step_ms:7.2f} ms  {r.peak_alloc_gb:5.2f} GB")
        reset_memory()

    results += bench_real_modules(args.real_seqlen, C, args.factor,
                                  B, args.iters, args.warmup)
    results += bench_norm_patch_end_to_end(args.real_seqlen, C, args.factor,
                                           B, args.iters, args.warmup)

    print_header("SUMMARY")
    print(fmt_results_table(results))
    ok = [r for r in results if not r.oom and not r.error][:4]
    if ok:
        base = ok[0]
        print(f"\nrelative to {base.name} ({base.step_ms:.2f} ms):")
        for r in ok[1:]:
            print(f"  {r.name:<46} {base.step_ms / r.step_ms:5.2f}x faster")
    save_results(results, args.out, meta={"probe": "upsample_op"})


if __name__ == "__main__":
    main()
