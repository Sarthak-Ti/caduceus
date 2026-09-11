"""Why is HydraBlock 3x slower than Caduceus' BiMambaWrapper at L=524288?

The per-block sweep (tests/bench_striped.py --part blocks) found Hydra ~3.1x slower and
~2x the memory at matched parameter count, and *superlinear* in L. That is surprising:
Mamba-2's SSD is supposed to beat Mamba-1's selective scan, and an isolated
unidirectional Mamba2 module does. This isolates which of the differences is responsible:

  1. d_state       -- striped uses 64, Caduceus uses 16 (4x the SSM state)
  2. chunk_size    -- SSD's intra-chunk work is quadratic in chunk_size
  3. the fused path -- Hydra falls back to `mamba_chunk_scan_combined` + PyTorch gating
                       when use_mem_eff_path=False; confirm which path is actually taken
  4. Hydra's bidirectional wrapper itself -- vs 2x a plain unidirectional Mamba2

All on one GPU so the numbers are comparable.

    pixi run python -m tests.bench_hydra_isolate --seqlen 262144
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (env_info, fmt_results_table, measure, print_header,
                               reset_memory, save_results)

AUTOCAST = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
D_MODEL, EXPAND, HEADDIM = 256, 2, 64


def main():
    from src.models.nn.hydra import Hydra
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mamba_simple import Mamba
    from tests.bench_models import build_bimamba_layer

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="tests/results/hydra_isolate.json")
    args = ap.parse_args()

    B, L = args.batch_size, args.seqlen
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    print_header(f"HYDRA ISOLATION  B={B}  L={L}  d_model={D_MODEL}  expand={EXPAND}")

    x = torch.randn(B, L, D_MODEL, device="cuda", requires_grad=True)
    results = []

    def run(name, mod, note=""):
        def fwd():
            with AUTOCAST():
                return mod(x)
        r = measure(name, fwd, seqlen=L, batch_size=B, variant="isolate",
                    warmup=args.warmup, iters=args.iters,
                    extra={"params": sum(p.numel() for p in mod.parameters())})
        results.append(r)
        if r.oom:
            print(f"  {name:<46} OOM")
        elif r.error:
            print(f"  {name:<46} ERROR {r.error[:60]}")
        else:
            print(f"  {name:<46} fwd {r.fwd_ms:8.2f}  bwd {r.bwd_ms:8.2f}  "
                  f"step {r.step_ms:8.2f} ms  {r.peak_alloc_gb:6.2f} GB  {note}")
        reset_memory()
        return r

    # --- reference: the model in production today ---
    bi = build_bimamba_layer(d_model=D_MODEL)
    run("caduceus BiMamba (d_state=16) [reference]", bi)
    del bi; reset_memory()

    # --- 4. Hydra's bidirectional wrapper vs a plain unidirectional Mamba2 ---
    # d_conv=4, not striped's 5: the fused `causal_conv1d` kernel only supports width
    # 2-4. Hydra can use 5 precisely because it does NOT use that kernel -- it runs a
    # plain nn.Conv1d with two transposes instead, which is part of what we are pricing.
    for ds in (16, 64):
        for mem_eff in (True, False):
            m2 = Mamba2(d_model=D_MODEL, d_state=ds, d_conv=4, expand=EXPAND,
                        headdim=HEADDIM, ngroups=1, chunk_size=256,
                        use_mem_eff_path=mem_eff).cuda()
            run(f"Mamba2 unidirectional (d_state={ds}, mem_eff={mem_eff})", m2,
                note="x2 for a fair bidirectional comparison")
            del m2; reset_memory()

    # Hydra at d_conv=4 too, so the conv width is not a confound
    for ds in (16, 64):
        h4 = Hydra(d_model=D_MODEL, d_state=ds, d_conv=4, expand=EXPAND,
                   headdim=HEADDIM, ngroups=1, chunk_size=256,
                   use_mem_eff_path=False).cuda()
        run(f"Hydra bidirectional (d_state={ds}, d_conv=4, mem_eff=False)", h4)
        del h4; reset_memory()

    # --- 1 & 3. Hydra across d_state, both paths ---
    for mem_eff in (True, False):
        for ds in (16, 64):
            h = Hydra(d_model=D_MODEL, d_state=ds, d_conv=5, expand=EXPAND,
                      headdim=HEADDIM, ngroups=1, chunk_size=256,
                      use_mem_eff_path=mem_eff).cuda()
            assert h.use_mem_eff_path == mem_eff, "use_mem_eff_path did not propagate"
            run(f"Hydra bidirectional (d_state={ds}, mem_eff={mem_eff})", h)
            del h; reset_memory()

    # --- 2. chunk_size, at the production d_state ---
    for cs in (64, 128, 256, 512):
        h = Hydra(d_model=D_MODEL, d_state=64, d_conv=5, expand=EXPAND,
                  headdim=HEADDIM, ngroups=1, chunk_size=cs,
                  use_mem_eff_path=True).cuda()
        run(f"Hydra (d_state=64, chunk_size={cs})", h)
        del h; reset_memory()

    # --- headdim, which sets nheads = d_inner / headdim ---
    for hd in (32, 64, 128):
        if (EXPAND * D_MODEL) % hd:
            continue
        h = Hydra(d_model=D_MODEL, d_state=64, d_conv=5, expand=EXPAND,
                  headdim=hd, ngroups=1, chunk_size=256, use_mem_eff_path=True).cuda()
        run(f"Hydra (d_state=64, headdim={hd})", h)
        del h; reset_memory()

    print_header("SUMMARY")
    print(fmt_results_table(results))
    ok = [r for r in results if not r.oom and not r.error]
    if ok:
        base = ok[0]
        print(f"\nrelative to {base.name} ({base.step_ms:.1f} ms):")
        for r in ok[1:]:
            print(f"  {r.name:<46} {r.step_ms / base.step_ms:5.2f}x slower  "
                  f"({r.step_ms:.1f} ms, {r.peak_alloc_gb:.2f} GB)")
    save_results(results, args.out, meta={"probe": "hydra_isolate"})


if __name__ == "__main__":
    main()
