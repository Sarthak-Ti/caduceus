"""Alternative execution paths for Caduceus' bidirectional Mamba stack.

MEASURED OUTCOME -- read this before enabling anything here. Verdicts are for
**A100 80GB PCIe** (the GPU the production job runs on), L=524288, B=1, bf16 autocast:

    variant           training   inference B=1   peak mem   verdict
    ----------------  ---------  --------------  ---------  ----------------------------
    fused_bidir       -28 %      --              1.46x      REJECTED
    fused_inner       -10 %      +4.3 %          0.82x      inference only on A100
    shared_in_proj    -0.7 %     --              0.95x      neutral
    ckpt              -29 %      n/a             0.24x      USE WHEN MEMORY-BOUND

`fused_inner`'s verdict is hardware-dependent: it is **6-8 % FASTER on H200** (HBM3e,
~4.8 TB/s) and 10 % slower on A100 (~1935 GB/s). Its forward is faster on both; only its
backward regresses, because applying the gate outside the scan kernel adds memory traffic
that HBM3e absorbs and A100/L40S do not. Worth re-enabling if this model moves to
HBM3e-class hardware.

For A100 training, only `ckpt` (gradient checkpointing, implemented in
modeling_caduceus.py rather than here) is recommended, and it is a memory optimization,
not a speed one.

The fused variants are kept because they are correct (tests/test_equivalence.py verifies
every parameter gradient) and because they document what does *not* work: `mamba_inner_fn`
keeps the whole conv -> x_proj -> scan -> gate -> out_proj chain inside fused CUDA kernels
with almost nothing materialized, and on a ~1935 GB/s part every Python-level
restructuring measured here pays more in extra tensor materialization than it saves --
even when it provably removes GEMM work and halves kernel launches.
See claude_summaries/mamba_optimization_report.md.


Everything here is opt-in and **state-dict compatible**: the parameter tree
(`mamba_fwd.*` / `mamba_rev.*`) is untouched, so existing checkpoints under outputs/
load into an optimized model and vice-versa. Only the *schedule* of the math changes.

Why this exists
---------------
`BiMambaWrapper.forward` runs the two directions as two fully independent `Mamba`
calls. At the production shape (B=1, L=524288, d_model=256, d_inner=512) that means:

  * two `in_proj` GEMMs even though `bidirectional_weight_tie=True` makes the second
    one a reversal of the first (in_proj is position-wise, so
    `in_proj(flip(h)) == flip(in_proj(h))`),
  * two `out_proj` GEMMs whose results are then summed -- but out_proj is also
    position-wise and tied, so `out_proj(y_f) + flip(out_proj(y_r))` collapses to a
    single `out_proj(y_f + flip(y_r))`,
  * two `causal_conv1d` launches and, more importantly, two `selective_scan` launches
    each with grid = (batch, d_inner) = (1, 512) blocks. On an A100 (108 SMs) a single
    512-block launch does not fill the machine, and the two launches are serialized by
    the default stream.

`fused_bimamba_forward` runs both directions in one set of launches by concatenating
them along the channel axis and using the selective-scan kernel's `n_groups` support
(B/C of shape (batch, n_groups, d_state, L), channel d belongs to group
d // (dim // n_groups)) to keep each direction's input-dependent B/C separate.
Group 0 is the forward direction, group 1 the reverse.

Preconditions (asserted, with a silent fall back to the stock path otherwise):
  * bidirectional_weight_tie=True     -- required to share in_proj/out_proj
  * bidirectional_strategy="add"      -- required to collapse the two out_proj calls
  * inference_params is None          -- no recurrent-state / decoding support

Memory note
-----------
The fused path does not use `mamba_inner_fn`, so it loses that function's
`checkpoint_lvl=1` trick of recomputing conv1d_out and delta in the backward pass.
On its own it therefore stores *more* activation memory than the baseline. It is
meant to be paired with `apply_gradient_checkpointing`, which makes every layer's
internals transient and reduces total activation memory far below the baseline.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

from caduceus.bimamba_inner import bi_mamba_inner_fn
from caduceus.modeling_caduceus import BiMambaWrapper


def _block_diag2(w1, w2):
    """[[w1, 0], [0, w2]] -- lets two per-direction projections run as one GEMM whose
    output is already the concatenated (2 * d_inner) buffer the fused scan wants, so we
    avoid materializing and then `torch.cat`-ing a (B, 2*d_inner, L) tensor."""
    o1, i1 = w1.shape
    o2, i2 = w2.shape
    out = w1.new_zeros(o1 + o2, i1 + i2)
    out[:o1, :i1] = w1
    out[o1:, i1:] = w2
    return out


def fused_bimamba_forward(self, hidden_states, inference_params=None):
    """Drop-in replacement for `BiMambaWrapper.forward`.

    MEASURED 28 % SLOWER on A100 and 1.46x the memory. Correct, but a rejected
    optimization -- chaining ordinary autograd ops forfeits mamba_inner_fn's
    checkpoint_lvl=1 recompute of conv1d_out and delta. Use `fused_inner_forward`
    instead if you want this fusion; it is the same idea done properly.
    See the module docstring."""
    if inference_params is not None or not self.bidirectional:
        return _stock_forward(self, hidden_states, inference_params)

    f, r = self.mamba_fwd, self.mamba_rev
    batch, seqlen, _ = hidden_states.shape
    d_inner, d_state, dt_rank = f.d_inner, f.d_state, f.dt_rank

    # --- 1. in_proj once. Tied + position-wise => xz for the reverse direction is
    #        exactly this xz reversed along L.
    xz = rearrange(
        f.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=seqlen,
    )
    if f.in_proj.bias is not None:
        xz = xz + rearrange(f.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    x, z = xz.chunk(2, dim=1)                                   # (B, d_inner, L) each
    x2 = torch.cat([x, x.flip(-1)], dim=1)                      # (B, 2*d_inner, L)
    z2 = torch.cat([z, z.flip(-1)], dim=1)

    # --- 2. depthwise causal conv: one launch over both directions
    conv_w = torch.cat([f.conv1d.weight, r.conv1d.weight], dim=0)
    conv_b = torch.cat([f.conv1d.bias, r.conv1d.bias], dim=0)
    # The conv kernel is templated separately on input_t and weight_t, so the fp32
    # weights are passed through unchanged exactly as mamba_inner_fn does -- casting
    # them to bf16 here would make this path *less* accurate than the baseline.
    conv_out = causal_conv1d_fn(
        x=x2,
        weight=rearrange(conv_w, "d 1 w -> d w"),
        bias=conv_b,
        activation="silu",
    )                                                            # (B, 2*d_inner, L)

    # --- 3. x_proj per direction, kept in (B, d, L) layout.
    #        mamba_inner_fn does `rearrange(conv1d_out, 'b d l -> (b l) d')` before its
    #        F.linear, which forces a full transpose copy of a d_inner x L activation.
    #        Multiplying on the left keeps the layout the scan kernel already wants.
    xw = f.x_proj.weight.to(dtype=conv_out.dtype)                # (dt_rank+2N, d_inner)
    rw = r.x_proj.weight.to(dtype=conv_out.dtype)
    dbl_f = torch.matmul(xw, conv_out[:, :d_inner])              # (B, dt_rank+2N, L)
    dbl_r = torch.matmul(rw, conv_out[:, d_inner:])

    dt_in = torch.cat([dbl_f[:, :dt_rank], dbl_r[:, :dt_rank]], dim=1)   # (B, 2*dt_rank, L)
    # block-diagonal dt_proj: one GEMM that writes the concatenated delta directly
    w_dt = _block_diag2(f.dt_proj.weight, r.dt_proj.weight).to(dtype=dt_in.dtype)
    delta = torch.matmul(w_dt, dt_in)                            # (B, 2*d_inner, L)

    B_ssm = torch.stack([dbl_f[:, dt_rank:dt_rank + d_state],
                         dbl_r[:, dt_rank:dt_rank + d_state]], dim=1).contiguous()
    C_ssm = torch.stack([dbl_f[:, -d_state:],
                         dbl_r[:, -d_state:]], dim=1).contiguous()   # (B, 2, d_state, L)

    # --- 4. ONE selective scan over 2*d_inner channels, n_groups=2
    A = -torch.exp(torch.cat([f.A_log, r.A_log], dim=0).float())     # (2*d_inner, d_state)
    D = torch.cat([f.D, r.D], dim=0).float()
    dt_bias = torch.cat([f.dt_proj.bias, r.dt_proj.bias], dim=0).float()
    y = selective_scan_fn(conv_out, delta, A, B_ssm, C_ssm, D, z=z2,
                          delta_bias=dt_bias, delta_softplus=True)    # (B, 2*d_inner, L)

    # --- 5. out_proj once. Position-wise and tied, so
    #        out_proj(y_f) + flip(out_proj(y_r)) == out_proj(y_f + flip(y_r)).
    y_f, y_r = y.split(d_inner, dim=1)
    y_sum = y_f + y_r.flip(-1)
    bias = f.out_proj.bias
    if bias is not None:
        # the stock path adds the (tied) bias once per direction
        bias = bias * 2
    return F.linear(rearrange(y_sum, "b d l -> b l d"), f.out_proj.weight, bias)


def _stock_forward(self, hidden_states, inference_params=None):
    out = self.mamba_fwd(hidden_states, inference_params=inference_params)
    if self.bidirectional:
        out_rev = self.mamba_rev(
            hidden_states.flip(dims=(1,)), inference_params=inference_params
        ).flip(dims=(1,))
        if self.bidirectional_strategy == "add":
            out = out + out_rev
        else:
            out = out * out_rev
    return out


def can_fuse(mod: BiMambaWrapper) -> bool:
    """The fusion is only exact when in_proj/out_proj are shared and the two
    directions are combined additively."""
    if not mod.bidirectional or mod.mamba_rev is None:
        return False
    if mod.bidirectional_strategy != "add":
        return False
    tied = (mod.mamba_fwd.in_proj.weight is mod.mamba_rev.in_proj.weight
            and mod.mamba_fwd.out_proj.weight is mod.mamba_rev.out_proj.weight)
    return tied


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------
def apply_fused_bidirectional(model, strict=True):
    """Swap every eligible BiMambaWrapper onto the fused forward. Returns the model."""
    n_patched = n_skipped = 0
    for mod in model.modules():
        if not isinstance(mod, BiMambaWrapper):
            continue
        if not can_fuse(mod):
            n_skipped += 1
            if strict:
                raise ValueError(
                    "fused bidirectional path requires bidirectional_weight_tie=True and "
                    "bidirectional_strategy='add'"
                )
            continue
        mod.forward = fused_bimamba_forward.__get__(mod, BiMambaWrapper)
        n_patched += 1
    print(f"[bimamba_opt] fused bidirectional: patched {n_patched} layers "
          f"({n_skipped} ineligible)")
    return model


def apply_gradient_checkpointing(model):
    """Turn on per-Block activation checkpointing in a CaduceusMixerModel.

    Trades ~1 extra forward pass for an order-of-magnitude drop in stored activations,
    which at L=524288 is what caps the batch size at 1 (and therefore what leaves the
    selective-scan kernel's grid too small to fill the GPU)."""
    model.gradient_checkpointing = True
    print("[bimamba_opt] gradient checkpointing: enabled")
    return model



def _mamba_inner_from_xz(m, xz):
    """Run one Mamba's fused inner path on an already-projected xz."""
    A = -torch.exp(m.A_log.float())
    return mamba_inner_fn(
        xz, m.conv1d.weight, m.conv1d.bias, m.x_proj.weight, m.dt_proj.weight,
        m.out_proj.weight, m.out_proj.bias, A, None, None, m.D.float(),
        delta_bias=m.dt_proj.bias.float(), delta_softplus=True,
    )


def shared_in_proj_forward(self, hidden_states, inference_params=None):
    """Conservative sibling of `fused_bimamba_forward`.

    Keeps both `mamba_inner_fn` calls -- and therefore that function's
    `checkpoint_lvl=1` recompute of conv1d_out and delta, which the fully fused path
    gives up -- but still computes the tied `in_proj` only once. Because `in_proj` is
    position-wise and shared, `in_proj(flip(h)) == flip(in_proj(h))`, so the reverse
    direction can reuse the forward projection reversed along L.

    Trade: one 256-channel flip of `hidden_states` is replaced by one 1024-channel flip
    of `xz`, in exchange for dropping a 524288x256x1024 GEMM and its backward.
    """
    if inference_params is not None or not self.bidirectional:
        return _stock_forward(self, hidden_states, inference_params)

    f, r = self.mamba_fwd, self.mamba_rev
    seqlen = hidden_states.shape[1]
    xz = rearrange(
        f.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=seqlen,
    )
    if f.in_proj.bias is not None:
        xz = xz + rearrange(f.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    out = _mamba_inner_from_xz(f, xz)
    out_rev = _mamba_inner_from_xz(r, xz.flip(-1)).flip(dims=(1,))
    if self.bidirectional_strategy == "add":
        return out + out_rev
    return out * out_rev


def apply_shared_in_proj(model, strict=True):
    n_patched = n_skipped = 0
    for mod in model.modules():
        if not isinstance(mod, BiMambaWrapper):
            continue
        # only in_proj has to be tied here; the out_proj collapse is not used
        if not (mod.bidirectional and mod.mamba_rev is not None
                and mod.mamba_fwd.in_proj.weight is mod.mamba_rev.in_proj.weight):
            n_skipped += 1
            if strict:
                raise ValueError("shared in_proj requires bidirectional_weight_tie=True")
            continue
        mod.forward = shared_in_proj_forward.__get__(mod, BiMambaWrapper)
        n_patched += 1
    print(f"[bimamba_opt] shared in_proj: patched {n_patched} layers ({n_skipped} ineligible)")
    return model


def apply_shared_in_proj_ckpt(model):
    apply_shared_in_proj(model)
    return apply_gradient_checkpointing(model)



def fused_inner_forward(self, hidden_states, inference_params=None):
    """The fully fused bidirectional path, backed by `BiMambaInnerFn`.

    Unlike `fused_bimamba_forward` (which chains ordinary autograd ops and measured
    *slower* than stock because it forfeits mamba_inner_fn's checkpoint_lvl=1 recompute),
    this routes everything from the depthwise conv through out_proj into one custom
    autograd Function that recomputes conv1d_out and delta in the backward pass.
    """
    if inference_params is not None or not self.bidirectional:
        return _stock_forward(self, hidden_states, inference_params)

    f, r = self.mamba_fwd, self.mamba_rev
    seqlen = hidden_states.shape[1]
    xz = rearrange(
        f.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=seqlen,
    )
    if f.in_proj.bias is not None:
        xz = xz + rearrange(f.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    bias = f.out_proj.bias
    if bias is not None:
        bias = bias * 2   # the stock path adds the tied bias once per direction
    return bi_mamba_inner_fn(xz, f, r, f.out_proj.weight, bias)


def apply_fused_inner(model, strict=True):
    n_patched = n_skipped = 0
    for mod in model.modules():
        if not isinstance(mod, BiMambaWrapper):
            continue
        if not can_fuse(mod):
            n_skipped += 1
            if strict:
                raise ValueError(
                    "fused inner path requires bidirectional_weight_tie=True and "
                    "bidirectional_strategy='add'")
            continue
        mod.forward = fused_inner_forward.__get__(mod, BiMambaWrapper)
        n_patched += 1
    print(f"[bimamba_opt] fused inner (BiMambaInnerFn): patched {n_patched} layers "
          f"({n_skipped} ineligible)")
    return model


def apply_fused_inner_ckpt(model):
    apply_fused_inner(model)
    return apply_gradient_checkpointing(model)


def apply_fused_bidirectional_ckpt(model):
    apply_fused_bidirectional(model)
    return apply_gradient_checkpointing(model)


# ---------------------------------------------------------------------------
# benchmark variant registration (imported by tests/bench_variants.py)
# ---------------------------------------------------------------------------
try:
    from tests.bench_variants import register

    register("fused_bidir")(apply_fused_bidirectional)
    register("ckpt")(apply_gradient_checkpointing)
    register("fused_bidir_ckpt")(apply_fused_bidirectional_ckpt)
    register("shared_in_proj")(apply_shared_in_proj)
    register("shared_in_proj_ckpt")(apply_shared_in_proj_ckpt)
    register("fused_inner")(apply_fused_inner)
    register("fused_inner_ckpt")(apply_fused_inner_ckpt)
except ImportError:
    pass
