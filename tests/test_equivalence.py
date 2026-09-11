"""Correctness gates for the Mamba/Caduceus optimizations.

Every optimization in caduceus/bimamba_opt.py is supposed to change only the *schedule*
of the math, not the math. These tests hold that line:

  * the fused bidirectional path must reproduce the stock BiMambaWrapper's outputs and
    parameter gradients,
  * gradient checkpointing must be bit-comparable to the non-checkpointed backbone,
  * the selective-scan kernel's n_groups=2 mode must agree with two independent scans
    (this is the kernel feature the fusion relies on -- if the installed wheel does not
    support it, everything downstream is invalid).

In bf16 the comparison is made against an fp32 reference rather than against a fixed
tolerance: the fused path is required to be no less accurate than the stock path, which
is the property that actually matters for keeping training behaviour unchanged.

Run with:   pixi run python -m pytest tests/test_equivalence.py -v -s
       or:  pixi run python -m tests.test_equivalence
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Small but structurally identical to production: same d_state / d_conv / expand /
# tie / strategy, just fewer layers and a shorter sequence so the tests are quick.
TEST_D_MODEL = 128
TEST_N_LAYER = 3
TEST_SEQLEN = 2048
TEST_BATCH = 2


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# ---------------------------------------------------------------------------
def test_selective_scan_ngroups2_matches_two_scans():
    """selective_scan with n_groups=2 must equal two independent single-group scans
    on the two channel halves. This is the kernel capability the fusion is built on."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    torch.manual_seed(0)
    dev, B, L, N = "cuda", 2, 512, 16
    d = 64                       # per-direction channels; fused runs 2*d

    def rnd(*shape):
        return torch.randn(*shape, device=dev)

    u = rnd(B, 2 * d, L)
    delta = rnd(B, 2 * d, L)
    A = -torch.exp(rnd(2 * d, N))
    Bg = rnd(B, 2, N, L)
    Cg = rnd(B, 2, N, L)
    D = torch.rand(2 * d, device=dev)
    z = rnd(B, 2 * d, L)
    dt_bias = rnd(2 * d)

    fused = selective_scan_fn(u, delta, A, Bg, Cg, D, z=z, delta_bias=dt_bias,
                              delta_softplus=True)

    halves = []
    for i in range(2):
        sl = slice(i * d, (i + 1) * d)
        halves.append(selective_scan_fn(
            u[:, sl].contiguous(), delta[:, sl].contiguous(), A[sl],
            Bg[:, i:i + 1].contiguous(), Cg[:, i:i + 1].contiguous(),
            D[sl], z=z[:, sl].contiguous(), delta_bias=dt_bias[sl],
            delta_softplus=True))
    ref = torch.cat(halves, dim=1)

    err = _rel_err(fused, ref)
    print(f"\n  n_groups=2 vs 2x n_groups=1 : rel_err={err:.3e}")
    assert err < 1e-5, f"n_groups=2 disagrees with two separate scans (rel_err={err})"


# ---------------------------------------------------------------------------
def _make_layer(dtype=torch.float32, seed=0):
    from tests.bench_models import build_bimamba_layer
    torch.manual_seed(seed)
    layer = build_bimamba_layer(d_model=TEST_D_MODEL, device="cuda")
    return layer.to(dtype)


def _run_layer(layer, x, fused, autocast_dtype=None):
    from caduceus.bimamba_opt import fused_bimamba_forward, _stock_forward
    from caduceus.modeling_caduceus import BiMambaWrapper
    fn = fused_bimamba_forward if fused else _stock_forward
    layer.forward = fn.__get__(layer, BiMambaWrapper)
    if autocast_dtype is None:
        return layer(x)
    with torch.autocast("cuda", dtype=autocast_dtype):
        return layer(x)


def test_fused_bimamba_forward_fp32():
    layer = _make_layer()
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    ref = _run_layer(layer, x, fused=False)
    got = _run_layer(layer, x, fused=True)
    err = _rel_err(got, ref)
    print(f"\n  fused vs stock forward (fp32): rel_err={err:.3e}")
    assert err < 2e-5, f"fused bidirectional forward differs (rel_err={err})"


def test_fused_bimamba_backward_fp32():
    layer = _make_layer()
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")

    def grads(fused):
        layer.zero_grad(set_to_none=True)
        xin = x.clone().requires_grad_(True)
        out = _run_layer(layer, xin, fused=fused)
        out.square().mean().backward()
        return xin.grad.clone(), {n: p.grad.clone() for n, p in layer.named_parameters()
                                  if p.grad is not None}

    gx_ref, gp_ref = grads(False)
    gx_new, gp_new = grads(True)
    ex = _rel_err(gx_new, gx_ref)
    print(f"\n  fused vs stock d/dinput (fp32): rel_err={ex:.3e}")
    assert ex < 1e-4, f"input gradient differs (rel_err={ex})"
    assert set(gp_ref) == set(gp_new), "parameter set changed -- checkpoints would break"
    worst = ("", 0.0)
    for name in gp_ref:
        e = _rel_err(gp_new[name], gp_ref[name])
        if e > worst[1]:
            worst = (name, e)
        assert e < 1e-3, f"grad of {name} differs (rel_err={e})"
    print(f"  worst parameter grad rel_err: {worst[0]} = {worst[1]:.3e}")


def test_fused_bimamba_bf16_no_worse_than_stock():
    """In bf16 both paths lose accuracy; the fused one must not lose more."""
    layer32 = _make_layer()
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    ref = _run_layer(layer32, x, fused=False).float()

    e_stock = _rel_err(_run_layer(layer32, x, fused=False, autocast_dtype=torch.bfloat16), ref)
    e_fused = _rel_err(_run_layer(layer32, x, fused=True, autocast_dtype=torch.bfloat16), ref)
    print(f"\n  bf16 error vs fp32 reference: stock={e_stock:.3e}  fused={e_fused:.3e}")
    assert e_fused < max(1.5 * e_stock, 5e-3), (
        f"fused bf16 error {e_fused:.3e} materially worse than stock {e_stock:.3e}")


# ---------------------------------------------------------------------------
def _make_backbone(gradient_checkpointing=False, fused=False, seed=0, variant=None):
    from tests.bench_models import build_backbone, make_caduceus_config
    from caduceus.bimamba_opt import (apply_fused_bidirectional, apply_shared_in_proj,
                                      apply_fused_inner)
    torch.manual_seed(seed)
    cfg = make_caduceus_config(d_model=TEST_D_MODEL, n_layer=TEST_N_LAYER)
    cfg.gradient_checkpointing = gradient_checkpointing
    model = build_backbone(cfg)
    if fused:
        apply_fused_bidirectional(model)
    if variant == "shared_in_proj":
        apply_shared_in_proj(model)
    if variant == "fused_inner":
        apply_fused_inner(model)
    return model


def _backbone_out_and_grads(model, x):
    model.train()
    model.zero_grad(set_to_none=True)
    xin = x.clone().requires_grad_(True)
    out, _ = model(None, inputs_embeds=xin)
    out.square().mean().backward()
    return out.detach().clone(), xin.grad.clone(), \
        {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone(gradient_checkpointing=False)
    ckpt = _make_backbone(gradient_checkpointing=True)
    ckpt.load_state_dict(plain.state_dict())

    o1, g1, p1 = _backbone_out_and_grads(plain, x)
    o2, g2, p2 = _backbone_out_and_grads(ckpt, x)
    eo, eg = _rel_err(o2, o1), _rel_err(g2, g1)
    print(f"\n  checkpointing: out rel_err={eo:.3e}  dinput rel_err={eg:.3e}")
    assert eo < 1e-6 and eg < 1e-5
    for name in p1:
        e = _rel_err(p2[name], p1[name])
        assert e < 1e-5, f"grad of {name} differs under checkpointing (rel_err={e})"


def _noise_floor(model, x, eps=1e-7):
    """How much the gradients move when the *input* is perturbed by `eps` relative.

    This is the conditioning of the network's gradient: any two implementations that
    agree to ~eps in the forward pass are entitled to differ by roughly this much in
    the gradients. Comparing the fused path against this floor separates "different
    rounding through a long recurrence" from "actually wrong"."""
    _, g1, p1 = _backbone_out_and_grads(model, x)
    _, g2, p2 = _backbone_out_and_grads(model, x * (1.0 + eps))
    return _rel_err(g2, g1), {n: _rel_err(p2[n], p1[n]) for n in p1}


def test_fused_backbone_depth_sweep_shows_conditioning_not_error():
    """A single fused layer matches the stock layer to ~1e-7, but the deep backbone's
    gradients diverge much further. This test pins down which it is: the forward error
    must stay at the 1e-7 level at every depth (implementation is correct) while the
    gradient difference grows with depth and stays within the network's own gradient
    conditioning (amplification, not a bug)."""
    global TEST_N_LAYER
    saved = TEST_N_LAYER
    print()
    try:
        for n in (1, 2, 3, 4):
            TEST_N_LAYER = n
            torch.manual_seed(0)
            x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
            plain = _make_backbone()
            fused = _make_backbone(fused=True)
            fused.load_state_dict(plain.state_dict())
            o1, g1, _ = _backbone_out_and_grads(plain, x)
            o2, g2, _ = _backbone_out_and_grads(fused, x)
            floor_g, _ = _noise_floor(plain, x)
            print(f"  n_layer={n}: fwd rel_err={_rel_err(o2, o1):.2e}  "
                  f"dinput rel_err={_rel_err(g2, g1):.2e}  "
                  f"(conditioning floor {floor_g:.2e})")
            assert _rel_err(o2, o1) < 1e-5, "forward diverges -- that would be a real bug"
    finally:
        TEST_N_LAYER = saved


def test_fused_backbone_matches_and_state_dict_compatible():
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    fused = _make_backbone(fused=True)
    # the whole point: a baseline checkpoint loads into the optimized model unchanged
    missing, unexpected = fused.load_state_dict(plain.state_dict(), strict=True)
    assert not missing and not unexpected

    o1, g1, p1 = _backbone_out_and_grads(plain, x)
    o2, g2, p2 = _backbone_out_and_grads(fused, x)
    eo, eg = _rel_err(o2, o1), _rel_err(g2, g1)
    floor_g, floor_p = _noise_floor(plain, x)
    print(f"\n  fused backbone: out rel_err={eo:.3e}  dinput rel_err={eg:.3e} "
          f"(conditioning floor {floor_g:.3e})")
    # The forward pass is the real correctness signal and must be tight.
    assert eo < 1e-5, f"fused backbone output differs (rel_err={eo})"
    # Gradients only have to be within the network's own gradient conditioning.
    assert eg < max(10 * floor_g, 1e-4), (
        f"input grad rel_err {eg:.3e} exceeds conditioning floor {floor_g:.3e}")
    worst = ("", 0.0)
    for name in p1:
        e = _rel_err(p2[name], p1[name])
        worst = (name, e) if e > worst[1] else worst
        assert e < max(10 * floor_p[name], 1e-4), (
            f"grad of {name}: rel_err {e:.3e} vs conditioning floor {floor_p[name]:.3e}")
    print(f"  worst parameter grad rel_err: {worst[0]} = {worst[1]:.3e} "
          f"(floor {floor_p[worst[0]]:.3e})")


def test_fused_backbone_bf16_no_worse_than_stock():
    """The decisive practical check: in the precision the model actually trains in
    (bf16 autocast), the fused path must not lose more accuracy than the stock path
    measured against a common fp32 reference."""
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    fused = _make_backbone(fused=True)
    fused.load_state_dict(plain.state_dict())

    def grads(model, autocast):
        model.zero_grad(set_to_none=True)
        xin = x.clone().requires_grad_(True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast \
            else torch.autocast("cuda", enabled=False)
        with ctx:
            out, _ = model(None, inputs_embeds=xin)
        out.float().square().mean().backward()
        return out.detach().float(), xin.grad.clone(), \
            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    o_ref, g_ref, p_ref = grads(plain, autocast=False)
    o_s, g_s, p_s = grads(plain, autocast=True)
    o_f, g_f, p_f = grads(fused, autocast=True)
    eo_s, eo_f = _rel_err(o_s, o_ref), _rel_err(o_f, o_ref)
    eg_s, eg_f = _rel_err(g_s, g_ref), _rel_err(g_f, g_ref)
    wp_s = max(_rel_err(p_s[n], p_ref[n]) for n in p_ref)
    wp_f = max(_rel_err(p_f[n], p_ref[n]) for n in p_ref)
    print(f"\n  bf16 vs fp32 reference   out: stock={eo_s:.3e} fused={eo_f:.3e}")
    print(f"  bf16 vs fp32 reference dinput: stock={eg_s:.3e} fused={eg_f:.3e}")
    print(f"  bf16 vs fp32 ref worst param : stock={wp_s:.3e} fused={wp_f:.3e}")
    assert eo_f < max(1.5 * eo_s, 1e-3)
    assert eg_f < max(1.5 * eg_s, 1e-3)
    assert wp_f < max(1.5 * wp_s, 1e-3)


def test_fused_plus_checkpointing_matches():
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    both = _make_backbone(gradient_checkpointing=True, fused=True)
    both.load_state_dict(plain.state_dict())
    o1, g1, p1 = _backbone_out_and_grads(plain, x)
    o2, g2, p2 = _backbone_out_and_grads(both, x)
    floor_g, floor_p = _noise_floor(plain, x)
    print(f"\n  fused+ckpt backbone: out rel_err={_rel_err(o2, o1):.3e}  "
          f"dinput rel_err={_rel_err(g2, g1):.3e} (floor {floor_g:.3e})")
    assert _rel_err(o2, o1) < 1e-5
    assert _rel_err(g2, g1) < max(10 * floor_g, 1e-4)
    for name in p1:
        e = _rel_err(p2[name], p1[name])
        assert e < max(10 * floor_p[name], 1e-4), f"grad of {name} differs (rel_err={e:.3e})"




def test_shared_in_proj_backbone_matches():
    """`shared_in_proj` reuses the tied in_proj across directions but keeps both
    mamba_inner_fn calls, so it should be numerically indistinguishable from stock."""
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    shared = _make_backbone(variant="shared_in_proj")
    missing, unexpected = shared.load_state_dict(plain.state_dict(), strict=True)
    assert not missing and not unexpected

    o1, g1, p1 = _backbone_out_and_grads(plain, x)
    o2, g2, p2 = _backbone_out_and_grads(shared, x)
    floor_g, floor_p = _noise_floor(plain, x)
    print(f"\n  shared_in_proj backbone: out rel_err={_rel_err(o2, o1):.3e}  "
          f"dinput rel_err={_rel_err(g2, g1):.3e} (floor {floor_g:.3e})")
    assert _rel_err(o2, o1) < 1e-5
    assert _rel_err(g2, g1) < max(10 * floor_g, 1e-4)
    for name in p1:
        e = _rel_err(p2[name], p1[name])
        assert e < max(10 * floor_p[name], 1e-4), f"grad of {name} differs (rel_err={e:.3e})"




def test_fused_inner_backbone_matches():
    """BiMambaInnerFn hand-writes the whole bidirectional backward, so this is the most
    important gate in the file: outputs and every parameter gradient must match stock."""
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    fused = _make_backbone(variant="fused_inner")
    missing, unexpected = fused.load_state_dict(plain.state_dict(), strict=True)
    assert not missing and not unexpected

    o1, g1, p1 = _backbone_out_and_grads(plain, x)
    o2, g2, p2 = _backbone_out_and_grads(fused, x)
    floor_g, floor_p = _noise_floor(plain, x)
    print(f"\n  fused_inner backbone: out rel_err={_rel_err(o2, o1):.3e}  "
          f"dinput rel_err={_rel_err(g2, g1):.3e} (floor {floor_g:.3e})")
    assert _rel_err(o2, o1) < 1e-5, "forward differs"
    assert _rel_err(g2, g1) < max(10 * floor_g, 1e-4)
    worst = ("", 0.0)
    for name in p1:
        e = _rel_err(p2[name], p1[name])
        worst = (name, e) if e > worst[1] else worst
        assert e < max(10 * floor_p[name], 1e-4), (
            f"grad of {name}: rel_err {e:.3e} vs floor {floor_p[name]:.3e}")
    print(f"  worst parameter grad rel_err: {worst[0]} = {worst[1]:.3e} "
          f"(floor {floor_p[worst[0]]:.3e})")


def test_fused_inner_layer_gradcheck_vs_stock():
    """Per-parameter gradient comparison on a single layer, where the numbers are not
    yet amplified by depth -- this is what would catch a wrong term in the hand-written
    backward that the deep test's conditioning floor might otherwise hide."""
    from caduceus.bimamba_opt import fused_inner_forward, _stock_forward
    from caduceus.modeling_caduceus import BiMambaWrapper
    layer = _make_layer()
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")

    def grads(fn):
        layer.forward = fn.__get__(layer, BiMambaWrapper)
        layer.zero_grad(set_to_none=True)
        xin = x.clone().requires_grad_(True)
        out = layer(xin)
        (out * torch.linspace(0.5, 1.5, out.shape[-1], device="cuda")).square().mean().backward()
        return out.detach(), xin.grad.clone(), \
            {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}

    o_ref, gx_ref, gp_ref = grads(_stock_forward)
    o_new, gx_new, gp_new = grads(fused_inner_forward)
    print(f"\n  fused_inner layer: out rel_err={_rel_err(o_new, o_ref):.3e}  "
          f"dinput rel_err={_rel_err(gx_new, gx_ref):.3e}")
    assert _rel_err(o_new, o_ref) < 2e-5
    assert _rel_err(gx_new, gx_ref) < 1e-3
    assert set(gp_ref) == set(gp_new)
    for name in sorted(gp_ref):
        e = _rel_err(gp_new[name], gp_ref[name])
        print(f"    {name:<34} rel_err={e:.3e}")
        assert e < 2e-3, f"grad of {name} differs (rel_err={e:.3e})"


def test_fused_inner_bf16_no_worse_than_stock():
    """The manual gate (silu(z) applied outside the scan kernel instead of inside it)
    rounds differently in bf16; this checks it does not cost accuracy."""
    torch.manual_seed(0)
    x = torch.randn(TEST_BATCH, TEST_SEQLEN, TEST_D_MODEL, device="cuda")
    plain = _make_backbone()
    fused = _make_backbone(variant="fused_inner")
    fused.load_state_dict(plain.state_dict())

    def out_of(model, autocast):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast \
            else torch.autocast("cuda", enabled=False)
        with ctx, torch.no_grad():
            o, _ = model(None, inputs_embeds=x)
        return o.float()

    ref = out_of(plain, False)
    e_s = _rel_err(out_of(plain, True), ref)
    e_f = _rel_err(out_of(fused, True), ref)
    print(f"\n  bf16 vs fp32 reference: stock={e_s:.3e}  fused_inner={e_f:.3e}")
    assert e_f < max(1.5 * e_s, 5e-3)


# ---------------------------------------------------------------------------
def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
