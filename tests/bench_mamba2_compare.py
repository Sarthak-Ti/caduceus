"""Mamba-1 selective scan vs Mamba-2 (SSD) at matched width.

Half the training step is `selective_scan`, a sequential recurrence that runs on CUDA
cores. Mamba-2's chunked SSD formulation is matmul-based and uses tensor cores instead.
`mamba_ssm` 2.1.0 already ships `Mamba2`, and configs/model/striped_hydra.yaml already
runs it -- so the question for the pure model is simply how much faster the kernel is at
the same d_model, which this measures rather than assumes.

This is an architecture comparison, not a drop-in swap: Mamba-2 has different parameters
(no checkpoint compatibility) and different accuracy behaviour. It exists to put a number
on the recommendation.

    pixi run python -m tests.bench_mamba2_compare --seqlen 262144
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (PROD, env_info, fmt_results_table, measure,
                               print_header, reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def main():
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=PROD["d_model"])
    ap.add_argument("--out", default="tests/results/mamba2_compare.json")
    args = ap.parse_args()

    dev = "cuda"
    B, L, D = args.batch_size, args.seqlen, args.d_model
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"MAMBA-1 vs MAMBA-2 (SSD)   B={B}  L={L}  d_model={D}")

    results = []
    x = torch.randn(B, L, D, device=dev, requires_grad=True)

    def run(name, mod, extra=None):
        def fwd():
            with AUTOCAST():
                return mod(x)
        r = measure(name, fwd, seqlen=L, batch_size=B, warmup=args.warmup,
                    iters=args.iters, extra=extra)
        results.append(r)
        if r.oom:
            print(f"  {name:<44} OOM")
        elif r.error:
            print(f"  {name:<44} ERROR {r.error[:70]}")
        else:
            print(f"  {name:<44} fwd {r.fwd_ms:8.2f}  bwd {r.bwd_ms:8.2f}  "
                  f"step {r.step_ms:8.2f} ms  peak {r.peak_alloc_gb:6.2f} GB")
        reset_memory()
        return r

    # --- Mamba-1, exactly the production ssm_cfg (one direction) ---
    m1 = Mamba(d_model=D, d_state=PROD["d_state"], d_conv=PROD["d_conv"],
               expand=PROD["expand"], use_fast_path=True).to(dev)
    run("Mamba-1 (d_state=16, expand=2)", m1,
        extra={"params": sum(p.numel() for p in m1.parameters())})
    del m1
    reset_memory()

    # --- Mamba-2 at matched d_model/expand, across the head/state settings that matter ---
    for d_state, headdim in ((16, 64), (64, 64), (128, 64)):
        d_inner = PROD["expand"] * D
        if d_inner % headdim != 0:
            continue
        try:
            m2 = Mamba2(d_model=D, d_state=d_state, d_conv=PROD["d_conv"],
                        expand=PROD["expand"], headdim=headdim, ngroups=1,
                        chunk_size=256, use_mem_eff_path=True).to(dev)
        except Exception as e:  # noqa: BLE001
            print(f"  Mamba-2 (d_state={d_state}, headdim={headdim}) build failed: {e}")
            continue
        run(f"Mamba-2 SSD (d_state={d_state}, headdim={headdim})", m2,
            extra={"params": sum(p.numel() for p in m2.parameters())})
        del m2
        reset_memory()

    print_header("SUMMARY")
    print(fmt_results_table(results))
    ok = [r for r in results if not r.oom and not r.error]
    if len(ok) >= 2:
        base = ok[0]
        print(f"\nrelative to {base.name} ({base.step_ms:.1f} ms/step):")
        for r in ok[1:]:
            print(f"  {r.name:<44} {base.step_ms / r.step_ms:5.2f}x  "
                  f"({r.step_ms:.1f} ms, {r.peak_alloc_gb:.2f} GB)")
    save_results(results, args.out, meta={"probe": "mamba1_vs_mamba2"})


if __name__ == "__main__":
    main()
