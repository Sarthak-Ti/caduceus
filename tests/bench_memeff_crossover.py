"""At what sequence length does Hydra's `use_mem_eff_path` stop paying off?

At L=262144 it is 15% slower and uses 33% more memory than the unfused path, despite the
name. The forward does fuse and does win (44.8 vs 49.3 ms); the hand-written backward
gives it back (122.4 vs 95.7 ms) and it saves `scan` -- the (2B, L, d_inner) doubled-batch
scan output -- for backward, which the unfused path lets autograd free.

That pattern suggests it was tuned for LM-scale contexts where the saved-tensor overhead
is negligible and the forward fusion dominates. This finds the crossover, which decides
whether the advice is "set it false" or "set it false for long sequences".

    pixi run python -m tests.bench_memeff_crossover
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def main():
    from src.models.nn.hydra import Hydra

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="*",
                    default=[2048, 8192, 32768, 65536, 131072, 262144])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="tests/results/memeff_crossover.json")
    args = ap.parse_args()

    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"use_mem_eff_path CROSSOVER  B={args.batch_size}  d_model={args.d_model}")

    results, pairs = [], {}
    for mem_eff in (True, False):
        mod = Hydra(d_model=args.d_model, d_state=64, d_conv=5, expand=2, headdim=64,
                    ngroups=1, chunk_size=256, use_mem_eff_path=mem_eff).cuda()
        print(f"\n  use_mem_eff_path={mem_eff}")
        for L in args.seqlens:
            x = torch.randn(args.batch_size, L, args.d_model, device="cuda",
                            requires_grad=True)

            def fwd():
                with AUTOCAST():
                    return mod(x)

            r = measure(f"Hydra mem_eff={mem_eff}", fwd, seqlen=L,
                        batch_size=args.batch_size, variant=str(mem_eff),
                        warmup=args.warmup, iters=args.iters)
            results.append(r)
            pairs.setdefault(L, {})[mem_eff] = r
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

    print_header("mem_eff=True RELATIVE TO mem_eff=False  (>1 means the flag helps)")
    print(f"{'L':>9} {'True step':>11} {'False step':>11} {'speed ratio':>12} "
          f"{'True GB':>9} {'False GB':>9} {'mem ratio':>10}")
    for L in sorted(pairs):
        t, f = pairs[L].get(True), pairs[L].get(False)
        if not t or not f or t.oom or f.oom or t.error or f.error:
            continue
        print(f"{L:>9} {t.step_ms:>10.2f}m {f.step_ms:>10.2f}m "
              f"{f.step_ms / t.step_ms:>11.3f}x {t.peak_alloc_gb:>8.2f} "
              f"{f.peak_alloc_gb:>8.2f} {f.peak_alloc_gb / t.peak_alloc_gb:>9.3f}x")

    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"probe": "memeff_crossover"})


if __name__ == "__main__":
    main()
