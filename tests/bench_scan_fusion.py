"""Does fusing the two bidirectional directions into one kernel actually help?

The full fused rewrite in caduceus/bimamba_opt.py measured *slower* than baseline, but
that experiment confounded three things: (1) fusing the kernels, (2) losing
mamba_inner_fn's checkpoint_lvl=1 recompute, and (3) extra concat/flip traffic. This
isolates (1) at the production shape, and separately measures whether taking `z` out of
the scan kernel (possible because flip(silu(z_rev)) == silu(z), so the gate factors out
of the bidirectional sum) makes the kernel cheaper.

    pixi run python -m tests.bench_scan_fusion --seqlen 262144
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (PROD, env_info, fmt_results_table, measure,
                               print_header, reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def main():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="tests/results/scan_fusion.json")
    args = ap.parse_args()

    dev = "cuda"
    B, L = args.batch_size, args.seqlen
    d = PROD["expand"] * PROD["d_model"]      # 512 per direction
    N = PROD["d_state"]
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"SCAN / CONV FUSION PROBE  B={B}  L={L}  d_inner/dir={d}  d_state={N}")

    bf = torch.bfloat16
    results = []

    def run(name, fn, extra=None):
        r = measure(name, fn, seqlen=L, batch_size=B, warmup=args.warmup,
                    iters=args.iters, extra=extra)
        results.append(r)
        if r.oom:
            print(f"  {name:<38} OOM")
        elif r.error:
            print(f"  {name:<38} ERROR {r.error[:70]}")
        else:
            print(f"  {name:<38} fwd {r.fwd_ms:8.2f}  bwd {r.bwd_ms:8.2f}  "
                  f"step {r.fwd_ms + r.bwd_ms:8.2f} ms  peak {r.peak_alloc_gb:6.2f} GB")
        reset_memory()

    # --- selective scan: 2 x (d channels, 1 group) vs 1 x (2d channels, 2 groups) ---
    def mk(dim, groups):
        return dict(
            u=torch.randn(B, dim, L, device=dev, dtype=bf, requires_grad=True),
            delta=torch.randn(B, dim, L, device=dev, dtype=bf, requires_grad=True),
            A=(-torch.exp(torch.randn(dim, N, device=dev))).requires_grad_(True),
            Bm=torch.randn(B, groups, N, L, device=dev, dtype=bf, requires_grad=True),
            Cm=torch.randn(B, groups, N, L, device=dev, dtype=bf, requires_grad=True),
            D=torch.rand(dim, device=dev, requires_grad=True),
            z=torch.randn(B, dim, L, device=dev, dtype=bf, requires_grad=True),
            db=torch.randn(dim, device=dev, requires_grad=True),
        )

    a, b = mk(d, 1), mk(d, 1)
    fused = mk(2 * d, 2)

    def scan(p, use_z=True):
        return selective_scan_fn(p["u"], p["delta"], p["A"], p["Bm"], p["Cm"], p["D"],
                                 z=p["z"] if use_z else None, delta_bias=p["db"],
                                 delta_softplus=True)

    run("scan: 2 x (512ch, 1 group)  [baseline]",
        lambda: scan(a).float().sum() + scan(b).float().sum())
    run("scan: 1 x (1024ch, 2 groups) [fused]",
        lambda: scan(fused).float().sum())
    run("scan: 2 x (512ch) no z",
        lambda: scan(a, False).float().sum() + scan(b, False).float().sum())
    run("scan: 1 x (1024ch, 2g) no z",
        lambda: scan(fused, False).float().sum())

    # --- depthwise causal conv: 2 x 512ch vs 1 x 1024ch ---
    def mkconv(dim):
        return (torch.randn(B, dim, L, device=dev, dtype=bf, requires_grad=True),
                torch.randn(dim, PROD["d_conv"], device=dev, requires_grad=True),
                torch.randn(dim, device=dev, requires_grad=True))

    xa, wa, ba = mkconv(d)
    xb, wb, bb = mkconv(d)
    xf, wf, bff = mkconv(2 * d)
    cc = lambda x, w, bi: causal_conv1d_fn(x=x, weight=w, bias=bi, activation="silu")
    run("conv: 2 x 512ch  [baseline]",
        lambda: cc(xa, wa, ba).float().sum() + cc(xb, wb, bb).float().sum())
    run("conv: 1 x 1024ch [fused]", lambda: cc(xf, wf, bff).float().sum())

    # --- the copies the fused path must pay to build the 1024-channel buffers ---
    h = torch.randn(B, L, PROD["d_model"], device=dev, dtype=bf, requires_grad=True)
    xz = torch.randn(B, 2 * d, L, device=dev, dtype=bf, requires_grad=True)
    x_only = torch.randn(B, d, L, device=dev, dtype=bf, requires_grad=True)
    run("copy: flip h (B,L,256) [baseline pays]", lambda: h.flip(dims=(1,)) * 1.0)
    run("copy: cat[x, flip(x)] (->1024ch)",
        lambda: torch.cat([x_only, x_only.flip(-1)], dim=1) * 1.0)
    run("copy: flip xz (B,1024,L)", lambda: xz.flip(-1) * 1.0)

    # --- the tied GEMMs the fusion would remove ---
    in_w = torch.randn(2 * d, PROD["d_model"], device=dev, requires_grad=True)
    out_w = torch.randn(PROD["d_model"], d, device=dev, requires_grad=True)
    y = torch.randn(B, L, d, device=dev, dtype=bf, requires_grad=True)

    def in_proj_once():
        with AUTOCAST():
            return (in_w @ h.reshape(-1, PROD["d_model"]).t()).float().sum()

    def in_proj_twice():
        with AUTOCAST():
            hh = h.reshape(-1, PROD["d_model"]).t()
            return (in_w @ hh).float().sum() + (in_w @ hh).float().sum()

    run("in_proj x1 (fused)", in_proj_once)
    run("in_proj x2 (baseline)", in_proj_twice)
    run("out_proj x1 (fused)", lambda: _ap(F.linear, y, out_w))
    run("out_proj x2 (baseline)",
        lambda: _ap(F.linear, y, out_w) + _ap(F.linear, y, out_w))

    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"probe": "scan_fusion"})


def _ap(fn, *a):
    with AUTOCAST():
        return fn(*a).float().sum()


if __name__ == "__main__":
    main()
