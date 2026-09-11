"""Benchmark driver for the pure (non-striped) Mamba/Caduceus architecture.

Modes
-----
micro     : per-op attribution at production shapes (which kernel eats the time)
backbone  : CaduceusMixerModel only, sequence-length sweep
full      : encoder -> backbone -> decoder -> joint_loss, sequence-length sweep
scaling   : batch-size / GPU-occupancy probe at a fixed shorter length
all       : micro + backbone + full

Usage
-----
    pixi run python -m tests.bench_mamba --mode all --out tests/results/baseline.json
"""

import argparse
import functools
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.bench_utils import (
    PROD, SWEEP_SEQLENS, TimingResult, check_cuda_extensions, env_info,
    fmt_results_table, measure, print_header, reset_memory, save_results,
)

AUTOCAST = functools.partial(torch.autocast, "cuda", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# micro: per-op attribution
# ---------------------------------------------------------------------------

def probe_dtypes(seqlen=8192, batch_size=1):
    """Print the dtype/shape of every tensor flowing through one BiMamba layer under
    bf16 autocast. Settles what precision the production model *actually* runs the
    scan kernels in, which drives every downstream memory estimate."""
    from tests.bench_models import build_backbone, make_caduceus_config
    from caduceus.modeling_caduceus import BiMambaWrapper

    print_header("DTYPE PROBE (bf16 autocast, as Lightning's precision=bf16 does)")
    cfg = make_caduceus_config(n_layer=2)
    model = build_backbone(cfg)
    seen = {}

    def hook(mod, args, out):
        if "bimamba_in" not in seen:
            seen["bimamba_in"] = (tuple(args[0].shape), args[0].dtype)
            seen["bimamba_out"] = (tuple(out.shape), out.dtype)

    def mamba_hook(mod, args, out):
        # must return None -- a forward hook that returns a value replaces the
        # module's output, which silently breaks the bidirectional add downstream
        seen.setdefault("mamba_out", (tuple(out.shape), out.dtype))

    for m in model.modules():
        if isinstance(m, BiMambaWrapper):
            m.register_forward_hook(hook)
            m.mamba_fwd.register_forward_hook(mamba_hook)
            break

    x = torch.randn(batch_size, seqlen, cfg.d_model, device="cuda")
    with AUTOCAST():
        out, _ = model(None, inputs_embeds=x)
    for k, (shape, dt) in seen.items():
        print(f"  {k:<16} shape={shape}  dtype={dt}")
    print(f"  {'backbone_out':<16} shape={tuple(out.shape)}  dtype={out.dtype}")
    p = next(model.parameters())
    print(f"  {'param dtype':<16} {p.dtype}   (bf16-mixed keeps fp32 master weights)")
    del model, x, out
    reset_memory()


def bench_micro(seqlen, batch_size=1, iters=5, warmup=2):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.modules.mamba_simple import Mamba

    from tests.bench_models import (
        build_bimamba_layer, make_caduceus_config, JointStep, make_joint_batch,
    )
    from src.tasks.metrics import joint_loss

    dev = "cuda"
    B, L = batch_size, seqlen
    D = PROD["d_model"]
    d_inner = PROD["expand"] * D
    d_state = PROD["d_state"]
    dt_rank = (D + 15) // 16   # math.ceil(d_model / 16) == 16 for d_model=256
    results = []

    def run(name, fn, backward=True, extra=None):
        r = measure(name, fn, seqlen=L, batch_size=B, variant="baseline",
                    backward=backward, warmup=warmup, iters=iters, extra=extra)
        results.append(r)
        if r.oom:
            print(f"  {name:<32} OOM")
        elif r.error:
            print(f"  {name:<32} ERROR: {r.error[:90]}")
        else:
            print(f"  {name:<32} fwd {r.fwd_ms:9.3f} ms  bwd {r.bwd_ms:9.3f} ms  "
                  f"peak {r.peak_alloc_gb:6.2f} GB")
        reset_memory()
        return r

    print_header(f"MICRO-BENCHMARK  B={B}  L={L}  d_model={D}  d_inner={d_inner}  d_state={d_state}")

    # --- tensor movement -----------------------------------------------------
    # Under bf16 autocast the activations entering BiMambaWrapper are bf16, so the
    # flips are timed in bf16 to match the real traffic.
    h = torch.randn(B, L, D, device=dev, requires_grad=True)
    hb = torch.randn(B, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    run("flip_hidden_bf16_(B,L,256)", lambda: hb.flip(dims=(1,)) * 1.0)

    xzb = torch.randn(B, 2 * d_inner, L, device=dev, dtype=torch.bfloat16, requires_grad=True)
    run("flip_xz_bf16_(B,1024,L)", lambda: xzb.flip(dims=(-1,)) * 1.0)

    run("contiguous_copy_bf16_(B,L,256)", lambda: hb.clone())

    # --- projections ---------------------------------------------------------
    in_proj = torch.nn.Linear(D, 2 * d_inner, bias=False).to(dev)

    def _in_proj():
        with AUTOCAST():
            return rearrange(in_proj.weight @ rearrange(h, "b l d -> d (b l)"),
                             "d (b l) -> b d l", l=L)
    run("in_proj_256->1024", _in_proj)

    out_proj = torch.nn.Linear(d_inner, D, bias=False).to(dev)
    y = torch.randn(B, L, d_inner, device=dev, requires_grad=True)

    def _out_proj():
        with AUTOCAST():
            return out_proj(y)
    run("out_proj_512->256", _out_proj)

    x_proj = torch.nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False).to(dev)
    dt_proj = torch.nn.Linear(dt_rank, d_inner, bias=True).to(dev)
    conv_out = torch.randn(B, d_inner, L, device=dev, requires_grad=True)

    def _xproj_dtproj():
        with AUTOCAST():
            x_dbl = x_proj(rearrange(conv_out, "b d l -> (b l) d"))
            dt = dt_proj.weight @ x_dbl[:, :dt_rank].t()
            return rearrange(dt, "d (b l) -> b d l", l=L)
    run("x_proj+dt_proj", _xproj_dtproj)

    # --- depthwise causal conv ----------------------------------------------
    conv_w = torch.randn(d_inner, PROD["d_conv"], device=dev, requires_grad=True)
    conv_b = torch.randn(d_inner, device=dev, requires_grad=True)
    xin = torch.randn(B, d_inner, L, device=dev, requires_grad=True)

    def _conv():
        with AUTOCAST():
            return causal_conv1d_fn(x=xin.to(torch.bfloat16), weight=conv_w.to(torch.bfloat16),
                                    bias=conv_b.to(torch.bfloat16), activation="silu")
    run("causal_conv1d_fn", _conv)

    # --- selective scan ------------------------------------------------------
    A = -torch.exp(torch.randn(d_inner, d_state, device=dev)).requires_grad_(True)
    Dp = torch.ones(d_inner, device=dev, requires_grad=True)
    delta = torch.randn(B, d_inner, L, device=dev, requires_grad=True)
    Bm = torch.randn(B, 1, d_state, L, device=dev, requires_grad=True)
    Cm = torch.randn(B, 1, d_state, L, device=dev, requires_grad=True)
    z = torch.randn(B, d_inner, L, device=dev, requires_grad=True)
    dt_bias = torch.randn(d_inner, device=dev, requires_grad=True)

    def _scan():
        with AUTOCAST():
            return selective_scan_fn(xin.to(torch.bfloat16), delta.to(torch.bfloat16), A,
                                     Bm.to(torch.bfloat16), Cm.to(torch.bfloat16), Dp,
                                     z=z.to(torch.bfloat16), delta_bias=dt_bias, delta_softplus=True)
    run("selective_scan_fn", _scan)

    # --- fully fused mamba inner (what the model actually calls) -------------
    mamba = Mamba(d_model=D, d_state=d_state, d_conv=PROD["d_conv"],
                  expand=PROD["expand"], use_fast_path=True).to(dev)

    def _inner():
        with AUTOCAST():
            xzl = rearrange(mamba.in_proj.weight @ rearrange(h, "b l d -> d (b l)"),
                            "d (b l) -> b d l", l=L)
            Aq = -torch.exp(mamba.A_log.float())
            return mamba_inner_fn(xzl, mamba.conv1d.weight, mamba.conv1d.bias,
                                  mamba.x_proj.weight, mamba.dt_proj.weight,
                                  mamba.out_proj.weight, mamba.out_proj.bias, Aq,
                                  None, None, mamba.D.float(),
                                  delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True)
    run("mamba_inner_fn(fused)", _inner)

    def _mamba():
        with AUTOCAST():
            return mamba(h)
    run("Mamba_module(1 direction)", _mamba)
    del mamba
    reset_memory()

    # --- bidirectional wrapper ----------------------------------------------
    bi = build_bimamba_layer(device=dev)

    def _bi():
        with AUTOCAST():
            return bi(h)
    run("BiMambaWrapper(1 layer)", _bi)
    del bi
    reset_memory()

    # --- norms ---------------------------------------------------------------
    norm = RMSNorm(D, eps=1e-5).to(dev)
    resid = torch.randn(B, L, D, device=dev, requires_grad=True)

    def _rmsnorm():
        with AUTOCAST():
            return rms_norm_fn(h, norm.weight, None, residual=resid, prenorm=True,
                               residual_in_fp32=False, eps=1e-5)[0]
    run("rms_norm_fn(fused add+norm)", _rmsnorm)

    # --- encoder / decoder / loss -------------------------------------------
    from src.tasks.encoders import JointCNN
    from src.tasks.decoders import JointMaskingDecoder
    enc = JointCNN(d_model=D, d_input1=PROD["d_input1"], d_input2=PROD["d_input2"],
                   joint=True, kernel_size=PROD["encoder_kernel_size"], norm=True,
                   acc_type="continuous").to(dev)
    x1 = torch.randn(B, PROD["d_input1"], L, device=dev, requires_grad=True)
    x2 = torch.randn(B, PROD["d_input2"], L, device=dev, requires_grad=True)

    def _enc():
        with AUTOCAST():
            return enc(x1, x2)[0]
    run("JointCNN_encoder", _enc)

    dec = JointMaskingDecoder(d_model=D, d_output1=PROD["d_output1"],
                              d_output2=PROD["d_output2"]).to(dev)

    def _dec():
        with AUTOCAST():
            a, b = dec(h)
            return a.float().sum() + b.float().sum()
    run("JointMaskingDecoder", _dec)

    y_seq = torch.zeros(B, L, PROD["d_output1"] + 1, device=dev)
    y_seq[..., 0] = 1.0
    y_acc = torch.stack([torch.rand(B, L, device=dev) * 3,
                         (torch.rand(B, L, device=dev) < 0.05).float()], dim=-1)
    seq_logits = torch.randn(B, L, PROD["d_output1"], device=dev, requires_grad=True)
    acc_pred = torch.randn(B, L, PROD["d_output2"], device=dev, requires_grad=True)

    def _loss():
        with AUTOCAST():
            return joint_loss((seq_logits, acc_pred), (y_seq, y_acc),
                              task1="mlm", task2="reg", count_weight=16)
    run("joint_loss", _loss)

    return results


def _autocast_call(mod, *a, **kw):
    with AUTOCAST():
        return mod(*a, **kw)


# ---------------------------------------------------------------------------
# backbone sweep
# ---------------------------------------------------------------------------
def bench_backbone(seqlens, batch_size=1, iters=5, warmup=2, variant="baseline",
                   n_layer=None, apply_variant=None):
    from tests.bench_models import build_backbone, make_caduceus_config, count_params

    results = []
    cfg = make_caduceus_config(n_layer=n_layer)
    model = build_backbone(cfg)
    if apply_variant is not None:
        model = apply_variant(model)
    print_header(f"BACKBONE SWEEP  variant={variant}  n_layer={cfg.n_layer}  "
                 f"d_model={cfg.d_model}  params={count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for L in seqlens:
        x = torch.randn(batch_size, L, cfg.d_model, device="cuda", requires_grad=True)

        def fwd():
            with AUTOCAST():
                out, _ = model(None, inputs_embeds=x)
            return out

        r = measure(f"caduceus_backbone", fwd, seqlen=L, batch_size=batch_size,
                    variant=variant, warmup=warmup, iters=iters,
                    zero_grad=lambda: opt.zero_grad(set_to_none=True))
        results.append(r)
        print(fmt_results_table([r]).split("\n")[-1])
        del x
        reset_memory()
    del model, opt
    reset_memory()
    return results


# ---------------------------------------------------------------------------
# full training step
# ---------------------------------------------------------------------------
def bench_full(seqlens, batch_size=1, iters=5, warmup=2, variant="baseline",
               n_layer=None, apply_variant=None, with_optimizer=True):
    from tests.bench_models import JointStep, make_joint_batch, make_caduceus_config, count_params

    results = []
    cfg = make_caduceus_config(n_layer=n_layer)
    model = JointStep(cfg)
    if apply_variant is not None:
        model.backbone = apply_variant(model.backbone)
    print_header(f"FULL STEP SWEEP  variant={variant}  params={count_params(model):,}  "
                 f"optimizer={'AdamW' if with_optimizer else 'none'}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for L in seqlens:
        try:
            x1, x2, y_seq, y_acc = make_joint_batch(L, batch_size)
        except RuntimeError as e:
            print(f"  L={L} batch alloc failed: {e}")
            continue

        def fwd():
            with AUTOCAST():
                return model(x1, x2, y_seq, y_acc)

        r = measure("joint_train_step", fwd, seqlen=L, batch_size=batch_size,
                    variant=variant, warmup=warmup, iters=iters,
                    zero_grad=lambda: opt.zero_grad(set_to_none=True))
        results.append(r)
        print(fmt_results_table([r]).split("\n")[-1])
        del x1, x2, y_seq, y_acc
        reset_memory()
    del model, opt
    reset_memory()
    return results



# ---------------------------------------------------------------------------
# inference
#
# Inference stores no activations, so the memory wall that forces batch_size=1 in
# training does not apply. The question this answers is how much of the latency is
# irreducible per-token scan work and how much is GPU underfill that a larger batch
# would amortize away.
# ---------------------------------------------------------------------------
def bench_inference(seqlens, batch_sizes=(1,), iters=5, warmup=2, variant="baseline",
                    n_layer=None, apply_variant=None):
    from tests.bench_models import build_backbone, make_caduceus_config

    results = []
    cfg = make_caduceus_config(n_layer=n_layer)
    model = build_backbone(cfg)
    if apply_variant is not None:
        model = apply_variant(model)
    model.eval()
    print_header(f"INFERENCE SWEEP  variant={variant}  (eval + no_grad, forward only)")
    for L in seqlens:
        for B in batch_sizes:
            x = torch.randn(B, L, cfg.d_model, device="cuda")

            def fwd():
                with torch.no_grad(), AUTOCAST():
                    out, _ = model(None, inputs_embeds=x)
                return out

            r = measure("caduceus_inference", fwd, seqlen=L, batch_size=B,
                        variant=variant, backward=False, warmup=warmup, iters=iters)
            results.append(r)
            print(fmt_results_table([r]).split("\n")[-1])
            del x
            reset_memory()
    model.train()
    del model
    reset_memory()
    return results


# ---------------------------------------------------------------------------
# occupancy / batch-size scaling probe
#
# At B=1 the selective-scan kernel launches grid=(batch, d_inner)=512 blocks. If the
# GPU is underfilled, step time should be sublinear in batch size -- that is the
# signature we are looking for here.
# ---------------------------------------------------------------------------
def bench_scaling(seqlen=65536, batch_sizes=(1, 2, 4, 8), iters=5, warmup=2, n_layer=None):
    from tests.bench_models import build_backbone, make_caduceus_config

    results = []
    cfg = make_caduceus_config(n_layer=n_layer)
    model = build_backbone(cfg)
    print_header(f"BATCH SCALING PROBE  L={seqlen}  n_layer={cfg.n_layer}")
    for B in batch_sizes:
        x = torch.randn(B, seqlen, cfg.d_model, device="cuda", requires_grad=True)

        def fwd():
            with AUTOCAST():
                out, _ = model(None, inputs_embeds=x)
            return out

        r = measure("backbone_batch_scaling", fwd, seqlen=seqlen, batch_size=B,
                    variant="baseline", warmup=warmup, iters=iters)
        results.append(r)
        print(fmt_results_table([r]).split("\n")[-1])
        del x
        reset_memory()
    del model
    reset_memory()
    return results


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["micro", "backbone", "full", "infer", "scaling", "all"])
    ap.add_argument("--seqlens", type=int, nargs="*", default=None)
    ap.add_argument("--micro-seqlen", type=int, default=131072,
                    help="length for the per-op attribution (default 131072 to leave "
                         "headroom for the many concurrently-live micro tensors)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--infer-batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8],
                    help="batch sizes for --mode infer")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--out", default="tests/results/bench.json")
    ap.add_argument("--variant", default="baseline")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmarks require a GPU"
    torch.manual_seed(0)
    print_header("ENVIRONMENT")
    for k, v in env_info().items():
        print(f"  {k:<16} {v}")
    ext = check_cuda_extensions()
    for k, v in ext.items():
        print(f"  {k:<16} {'OK' if v else 'MISSING  <-- falls back to slow path!'}")

    seqlens = args.seqlens or SWEEP_SEQLENS
    from tests.bench_variants import get_variant
    apply_variant = get_variant(args.variant)

    results = []
    if args.mode in ("micro", "all"):
        probe_dtypes()
        results += bench_micro(args.micro_seqlen, args.batch_size, args.iters, args.warmup)
    if args.mode in ("backbone", "all"):
        results += bench_backbone(seqlens, args.batch_size, args.iters, args.warmup,
                                  args.variant, args.n_layer, apply_variant)
    if args.mode in ("full", "all"):
        results += bench_full(seqlens, args.batch_size, args.iters, args.warmup,
                              args.variant, args.n_layer, apply_variant)
    if args.mode in ("infer", "all"):
        # in the combined run keep inference to the production length only, so the
        # sweep stays inside a 2h job
        infer_lens = seqlens if args.mode == "infer" else seqlens[-1:]
        results += bench_inference(infer_lens, tuple(args.infer_batch_sizes), args.iters,
                                   args.warmup, args.variant, args.n_layer, apply_variant)
    if args.mode in ("scaling", "all"):
        results += bench_scaling(65536, (1, 2, 4, 8), args.iters, args.warmup, args.n_layer)

    print_header("SUMMARY")
    print(fmt_results_table(results))
    save_results(results, args.out, meta={"mode": args.mode, "variant": args.variant,
                                          "prod_config": PROD})


if __name__ == "__main__":
    main()
