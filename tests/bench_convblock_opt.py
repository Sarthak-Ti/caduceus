"""Attack the `refine` ConvBlock, which is 58% of the top UpsampleLayer.

Component timing (A100, L=262144, C=256) put `refine` -- a ConvBlock(width 5) at full
resolution -- at 36.83 ms of the layer's 63.89 ms, with a backward 5.05x its forward
(30.74 vs 6.09 ms). That ratio is much worse than its siblings (2.8-3.4x), which suggests
the convolution backward is running well off peak rather than the block being inherently
expensive.

This prices four things, cheapest-to-adopt first:

  1. `torch.backends.cudnn.benchmark = True` -- free, global, no modelling change.
     Shapes here are static, so cuDNN's algorithm search should pay for itself.
  2. The `transpose -> conv -> transpose` in `ConvBlock.forward`: how much is layout?
  3. Where the block's time goes: norm vs GELU vs conv.
  4. `width 5 -> 3` -- a modelling change, priced for comparison.

    pixi run python -m tests.bench_convblock_opt --seqlen 262144
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def main():
    from src.models.nn.sampling import ConvBlock

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="tests/results/convblock_opt.json")
    args = ap.parse_args()

    B, L, C = args.batch_size, args.seqlen, args.channels
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")

    results = []
    x_blc = torch.randn(B, L, C, device="cuda", requires_grad=True)
    x_bcl = torch.randn(B, C, L, device="cuda", requires_grad=True)

    def run(name, fn, note=""):
        r = measure(name, fn, seqlen=L, batch_size=B, variant="convblock",
                    warmup=args.warmup, iters=args.iters)
        results.append(r)
        if r.oom or r.error:
            print(f"  {name:<48} {'OOM' if r.oom else r.error[:55]}")
        else:
            print(f"  {name:<48} fwd {r.fwd_ms:7.2f}  bwd {r.bwd_ms:7.2f}  "
                  f"step {r.step_ms:7.2f} ms  {r.peak_alloc_gb:5.2f} GB {note}")
        reset_memory()
        return r

    for bench_mode in (False, True):
        torch.backends.cudnn.benchmark = bench_mode
        tag = f"[cudnn.benchmark={bench_mode}]"
        print_header(f"CONVBLOCK  B={B}  L={L}  C={C}   cudnn.benchmark={bench_mode}")

        blk5 = ConvBlock(C, C, width=5, norm_type="rms").cuda()
        run(f"ConvBlock(w5) full {tag}", lambda: _ap(lambda: blk5(x_blc)))

        # component split
        run(f"  norm only {tag}", lambda: _ap(lambda: blk5.norm(x_blc)))
        run(f"  norm+GELU only {tag}", lambda: _ap(lambda: blk5.act(blk5.norm(x_blc))))
        run(f"  transpose->conv->transpose {tag}",
            lambda: _ap(lambda: blk5.op(x_blc.transpose(1, 2)).transpose(1, 2)))
        run(f"  conv only, already (B,C,L) {tag}",
            lambda: _ap(lambda: blk5.op(x_bcl)), note="<- layout win if much cheaper")

        blk3 = ConvBlock(C, C, width=3, norm_type="rms").cuda()
        run(f"ConvBlock(w3) full {tag}", lambda: _ap(lambda: blk3(x_blc)),
            note="<- modelling change")
        blk1 = ConvBlock(C, C, width=1, norm_type="rms").cuda()
        run(f"ConvBlock(w1) full (Linear, no transpose) {tag}",
            lambda: _ap(lambda: blk1(x_blc)))
        del blk5, blk3, blk1
        reset_memory()

    torch.backends.cudnn.benchmark = False
    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"probe": "convblock_opt"})


def _ap(fn):
    with AUTOCAST():
        return fn()


if __name__ == "__main__":
    main()
