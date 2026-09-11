"""Is flipping `use_mem_eff_path` safe, i.e. purely an implementation change?

`Hydra.forward` branches on `use_mem_eff_path`: True routes through the custom
`HydraSplitConv1dScanCombinedFn`, False runs the same math as ordinary PyTorch ops.
Benchmarks say False is both faster and lighter, but that is only actionable if the two
branches are numerically the same model -- they share every parameter, so if they agree
then flipping the flag cannot change training behaviour or invalidate a checkpoint.

    pixi run python -m pytest tests/test_striped_equivalence.py -v -s
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

D_MODEL, SEQLEN, BATCH = 256, 4096, 2


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _make_hydra(d_state=64, d_conv=5, seed=0):
    from src.models.nn.hydra import Hydra
    torch.manual_seed(seed)
    return Hydra(d_model=D_MODEL, d_state=d_state, d_conv=d_conv, expand=2,
                 headdim=64, ngroups=1, chunk_size=256, use_mem_eff_path=False).cuda()


def _out_and_grads(mod, x, mem_eff):
    mod.use_mem_eff_path = mem_eff
    mod.zero_grad(set_to_none=True)
    xin = x.clone().requires_grad_(True)
    out = mod(xin)
    (out * torch.linspace(0.5, 1.5, out.shape[-1], device="cuda")).square().mean().backward()
    return (out.detach().clone(), xin.grad.clone(),
            {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None})


# The fused path asserts `d_conv % 2 == 1` (hydra.py:306), so only odd widths exist to
# compare. Striped uses d_conv=5. Note this means `use_mem_eff_path=True` is simply
# unavailable at even conv widths -- it raises rather than silently differing.
@pytest.mark.parametrize("d_conv", [3, 5, 7])
def test_mem_eff_path_is_numerically_equivalent(d_conv):
    """The two branches must agree, otherwise `use_mem_eff_path` is a modelling choice
    rather than an implementation one."""
    mod = _make_hydra(d_conv=d_conv)
    x = torch.randn(BATCH, SEQLEN, D_MODEL, device="cuda")

    o_f, gx_f, gp_f = _out_and_grads(mod, x, mem_eff=False)
    o_t, gx_t, gp_t = _out_and_grads(mod, x, mem_eff=True)

    eo, egx = _rel_err(o_t, o_f), _rel_err(gx_t, gx_f)
    print(f"\n  d_conv={d_conv}  forward rel_err={eo:.3e}   d/dinput rel_err={egx:.3e}")
    assert set(gp_f) == set(gp_t), "the two paths touch different parameters"
    worst = ("", 0.0)
    for name in sorted(gp_f):
        e = _rel_err(gp_t[name], gp_f[name])
        worst = (name, e) if e > worst[1] else worst
        print(f"    {name:<24} rel_err={e:.3e}")
    print(f"  worst parameter grad: {worst[0]} = {worst[1]:.3e}")

    assert eo < 1e-5, f"forward differs between paths (rel_err={eo:.3e})"
    assert egx < 1e-3, f"input grad differs between paths (rel_err={egx:.3e})"
    assert worst[1] < 5e-3, f"grad of {worst[0]} differs (rel_err={worst[1]:.3e})"


def test_mem_eff_path_equivalent_in_bf16():
    """Same check under the bf16 autocast the model actually trains in."""
    mod = _make_hydra()
    x = torch.randn(BATCH, SEQLEN, D_MODEL, device="cuda")

    def out(mem_eff):
        mod.use_mem_eff_path = mem_eff
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            return mod(x).float()

    with torch.no_grad():
        ref = (lambda: (setattr(mod, "use_mem_eff_path", False), mod(x).float())[1])()
    e_f, e_t = _rel_err(out(False), ref), _rel_err(out(True), ref)
    print(f"\n  bf16 vs fp32 reference: mem_eff=False {e_f:.3e}  mem_eff=True {e_t:.3e}")
    assert e_t < max(1.5 * e_f, 5e-3)


# ---------------------------------------------------------------------------
# RMSBatchNorm1d rewrite (src/models/nn/sampling.py)
#
# The forward was changed from five full-size passes with two full-size temporaries to
# torch.var's fused reduction plus a single addcmul. Measured 1.103x on the whole pooled
# striped backbone at L=524288 and -30% memory. These tests hold it to the original
# formulation: forward, the running_var EMA, eval mode, gradients, and bf16.
# ---------------------------------------------------------------------------
def _original_rmsbatchnorm_forward(mod, x):
    """The pre-optimization implementation, kept here as the reference."""
    if mod.training:
        mean = x.mean(dim=[0, 1])
        var = (x - mean).pow(2).mean(dim=[0, 1])
        with torch.no_grad():
            mod.running_var.mul_(1 - mod.momentum).add_(var * mod.momentum)
    else:
        var = mod.running_var
    x = x / (var + mod.eps).sqrt()
    return x * mod.weight + mod.bias


def _fresh_norm(c=256, seed=0):
    from src.models.nn.sampling import RMSBatchNorm1d
    torch.manual_seed(seed)
    m = RMSBatchNorm1d(c).cuda()
    with torch.no_grad():
        m.weight.copy_(torch.randn(c, device="cuda") * 0.2 + 1.0)
        m.bias.copy_(torch.randn(c, device="cuda") * 0.1)
        m.running_var.copy_(torch.rand(c, device="cuda") + 0.5)
    return m


@pytest.mark.parametrize("training", [True, False])
def test_rmsbatchnorm_matches_original(training):
    c = 256
    x = torch.randn(2, 8192, c, device="cuda")

    ref_mod, new_mod = _fresh_norm(c), _fresh_norm(c)
    ref_mod.train(training)
    new_mod.train(training)

    ref = _original_rmsbatchnorm_forward(ref_mod, x)
    got = new_mod(x)
    e = _rel_err(got, ref)
    ev = _rel_err(new_mod.running_var, ref_mod.running_var)
    print(f"\n  training={training}: output rel_err={e:.3e}  running_var rel_err={ev:.3e}")
    assert e < 1e-5, f"output differs (rel_err={e:.3e})"
    assert ev < 1e-6, f"running_var EMA differs (rel_err={ev:.3e})"


def test_rmsbatchnorm_gradients_match():
    c = 256
    x = torch.randn(2, 8192, c, device="cuda")

    def run(use_new):
        mod = _fresh_norm(c)
        mod.train(True)
        xin = x.clone().requires_grad_(True)
        out = mod(xin) if use_new else _original_rmsbatchnorm_forward(mod, xin)
        (out * torch.linspace(0.5, 1.5, c, device="cuda")).square().mean().backward()
        return xin.grad.clone(), mod.weight.grad.clone(), mod.bias.grad.clone()

    gx0, gw0, gb0 = run(False)
    gx1, gw1, gb1 = run(True)
    for name, a, b in (("d/dinput", gx1, gx0), ("d/dweight", gw1, gw0), ("d/dbias", gb1, gb0)):
        e = _rel_err(a, b)
        print(f"  {name:<10} rel_err={e:.3e}")
        assert e < 1e-4, f"{name} differs (rel_err={e:.3e})"


def test_rmsbatchnorm_bf16_matches_and_keeps_dtype():
    """The rewrite changes the order of dtype promotion, so check bf16 explicitly."""
    c = 256
    x = torch.randn(2, 8192, c, device="cuda")
    ref_mod, new_mod = _fresh_norm(c), _fresh_norm(c)
    ref_mod.train(True); new_mod.train(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _original_rmsbatchnorm_forward(ref_mod, x)
        got = new_mod(x)
    e = _rel_err(got, ref)
    print(f"\n  bf16 autocast: rel_err={e:.3e}  ref dtype={ref.dtype}  new dtype={got.dtype}")
    assert e < 5e-3, f"bf16 output differs (rel_err={e:.3e})"
    assert got.dtype == ref.dtype, f"dtype changed: {ref.dtype} -> {got.dtype}"
