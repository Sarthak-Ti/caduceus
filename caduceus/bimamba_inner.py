"""`BiMambaInnerFn` -- both directions of a Caduceus BiMamba layer in one fused pass.

MEASURED OUTCOME (A100 80GB PCIe, L=524288, B=1, bf16 autocast):

    training  : 2353 vs 2133 ms baseline   -> 10 % SLOWER   (fwd 583 vs 622: faster;
                                                             bwd 1770 vs 1510: slower)
    inference : 586 vs 613 ms at B=1       -> 4.3 % FASTER  (loses at B>1)
    memory    : 55.98 vs 68.23 GB          -> 18 % LESS

On **H200** (HBM3e) the same code is **6-8 % faster for training**. The fusion is real --
the forward is faster on both GPUs -- but applying the gate outside the scan kernel adds
memory traffic in the backward that HBM3e absorbs and A100/L40S do not. Correctness is
verified in tests/test_equivalence.py: all 16 parameter gradients match the stock path to
~1e-7 and bf16 accuracy is unchanged.

Use it for low-latency single-sequence inference on A100, or for training on HBM3e-class
hardware. Do not use it for training on A100.

Why it loses, despite fusing exactly what the isolated probe said was worth ~11 %:
taking `z` out of the scan kernel (identity 3 below) moves the gate into ~8 separate
unfused PyTorch elementwise kernels. At 512 channels and ~0.27 GB per tensor that is
~10 GB/layer of extra memory traffic that the CUDA kernel previously did for free in
registers. The regression is almost entirely in the backward (698 -> 1037 ms) while the
forward barely moves (305 -> 342 ms). Putting `z` back in the kernel recovers roughly
half of that but then costs a 1024-channel `cat` of z, which analysis puts back at
roughly break-even -- so the whole direction was abandoned rather than iterated further.


This is the bidirectional analogue of `mamba_ssm.ops.selective_scan_interface.MambaInnerFn`,
and it exists because a first attempt that simply chained ordinary autograd ops measured
*slower* than the stock two-call path: writing the fusion in Python forfeits
`MambaInnerFn`'s `checkpoint_lvl=1` trick of recomputing `conv1d_out` and `delta` in the
backward pass, and the extra saved activations cost more than the fusion saves.

What the fusion buys (measured in isolation at B=1, L=262144, d_inner=512, d_state=16):

    selective_scan   2 x (512ch, 1 group)  36.44 ms  ->  1 x (1024ch, 2 groups)  30.50 ms
    causal_conv1d    2 x 512ch              3.31 ms  ->  1 x 1024ch               2.92 ms
    in_proj          2 x                    5.10 ms  ->  1 x                      2.70 ms
    out_proj         2 x                    2.24 ms  ->  1 x                      1.22 ms
    extra copy       flip h  1.16 ms        ->  cat[x, flip(x)]  3.98 ms

Three exact identities make it possible, all consequences of `bidirectional_weight_tie`
plus the fact that `in_proj` / `out_proj` act position-wise:

    1.  in_proj(flip(h)) == flip(in_proj(h))
            -> one in_proj GEMM; the reverse direction reuses xz reversed along L.
    2.  out_proj(y_f) + flip(out_proj(y_r)) == out_proj(y_f + flip(y_r))
            -> one out_proj GEMM on the combined result.
    3.  flip(silu(z_rev)) == silu(z), so the gate factors out of the bidirectional sum:
            silu(z)*scan_f + flip(silu(z_rev)*scan_r) == silu(z) * (scan_f + flip(scan_r))
            -> `z` is never flipped or concatenated, and is dropped from the scan
               kernel's inputs entirely (the gate is applied once, afterwards).

The two directions are then run as a single `selective_scan` over `2*d_inner` channels
using the kernel's `n_groups=2` mode (channel d belongs to group `d // d_inner`, so
group 0 carries the forward direction's input-dependent B/C and group 1 the reverse).
`tests/test_equivalence.py` verifies that `n_groups=2` is bit-exact against two
independent single-group scans.

Memory: the stock path stores `xz` twice (once per direction, ~2 x 2*d_inner*L) plus the
flipped hidden states and the reverse output. This stores `xz` once, so it is *lower*
than the baseline as well as faster.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd

import causal_conv1d_cuda
import selective_scan_cuda


def _cat_conv(w_f, b_f, w_r, b_r):
    """Depthwise conv weights concatenate along channels, so the two directions'
    convolutions become one launch over 2*d_inner channels."""
    w = torch.cat([w_f, w_r], dim=0)
    w = rearrange(w, "d 1 w -> d w")
    b = torch.cat([b_f, b_r], dim=0) if b_f is not None else None
    return w.contiguous(), (b.contiguous() if b is not None else None)


class BiMambaInnerFn(torch.autograd.Function):
    """Fused bidirectional Mamba inner path with checkpoint_lvl=1 style recompute.

    `xz` is the shared (B, 2*d_inner, L) projection of the layer input. Everything from
    the depthwise conv through `out_proj` happens here; the caller only supplies the
    already-projected `xz` and the two directions' parameters.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, xz,
                conv1d_weight_f, conv1d_bias_f, conv1d_weight_r, conv1d_bias_r,
                x_proj_weight_f, x_proj_weight_r,
                delta_proj_weight_f, delta_proj_weight_r,
                out_proj_weight, out_proj_bias,
                A_f, A_r, D_f, D_r, delta_bias_f, delta_bias_r,
                checkpoint_lvl=1):
        assert checkpoint_lvl in (0, 1)
        L = xz.shape[-1]
        d_inner = xz.shape[1] // 2
        dt_rank = delta_proj_weight_f.shape[1]
        d_state = A_f.shape[-1]

        if torch.is_autocast_enabled():
            dt = torch.get_autocast_gpu_dtype()
            x_proj_weight_f = x_proj_weight_f.to(dtype=dt)
            x_proj_weight_r = x_proj_weight_r.to(dtype=dt)
            delta_proj_weight_f = delta_proj_weight_f.to(dtype=dt)
            delta_proj_weight_r = delta_proj_weight_r.to(dtype=dt)
            out_proj_weight = out_proj_weight.to(dtype=dt)
            out_proj_bias = out_proj_bias.to(dtype=dt) if out_proj_bias is not None else None
        if xz.stride(-1) != 1:
            xz = xz.contiguous()

        x, z = xz.chunk(2, dim=1)
        conv_w, conv_b = _cat_conv(conv1d_weight_f, conv1d_bias_f,
                                   conv1d_weight_r, conv1d_bias_r)
        # identity 1: the reverse direction's conv input is the forward one reversed
        x2 = torch.cat([x, x.flip(-1)], dim=1).contiguous()
        conv_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x2, conv_w, conv_b, None, None, None, True)

        # x_proj left-multiplied so the (B, d, L) layout the scan wants is preserved --
        # mamba_inner_fn instead does `rearrange(conv1d_out, 'b d l -> (b l) d')`, a full
        # transpose copy of a d_inner x L activation per direction.
        dbl_f = torch.matmul(x_proj_weight_f, conv_out[:, :d_inner])
        dbl_r = torch.matmul(x_proj_weight_r, conv_out[:, d_inner:])
        delta = torch.cat([
            torch.matmul(delta_proj_weight_f, dbl_f[:, :dt_rank]),
            torch.matmul(delta_proj_weight_r, dbl_r[:, :dt_rank]),
        ], dim=1).contiguous()
        B_ssm = torch.stack([dbl_f[:, dt_rank:dt_rank + d_state],
                             dbl_r[:, dt_rank:dt_rank + d_state]], dim=1).contiguous()
        C_ssm = torch.stack([dbl_f[:, -d_state:],
                             dbl_r[:, -d_state:]], dim=1).contiguous()

        A = torch.cat([A_f, A_r], dim=0).contiguous()
        D = torch.cat([D_f, D_r], dim=0).contiguous()
        delta_bias = torch.cat([delta_bias_f, delta_bias_r], dim=0).contiguous()

        # identity 3: z is not passed to the kernel; the gate is applied once below
        out, scan_intermediates = selective_scan_cuda.fwd(
            conv_out, delta, A, B_ssm, C_ssm, D, None, delta_bias, True)

        y = out[:, :d_inner] + out[:, d_inner:].flip(-1)
        gated = y * F.silu(z)
        # identity 2: a single out_proj on the combined result
        result = F.linear(rearrange(gated, "b d l -> b l d"), out_proj_weight, out_proj_bias)

        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.d_inner, ctx.dt_rank, ctx.d_state = d_inner, dt_rank, d_state
        if checkpoint_lvl >= 1:
            conv_out, delta = None, None   # recomputed in backward
        ctx.save_for_backward(
            xz, conv1d_weight_f, conv1d_bias_f, conv1d_weight_r, conv1d_bias_r,
            x_proj_weight_f, x_proj_weight_r, delta_proj_weight_f, delta_proj_weight_r,
            out_proj_weight, dbl_f, dbl_r, conv_out, delta,
            A_f, A_r, B_ssm, C_ssm, D_f, D_r, delta_bias_f, delta_bias_r,
            scan_intermediates, out)
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, dresult):
        (xz, conv1d_weight_f, conv1d_bias_f, conv1d_weight_r, conv1d_bias_r,
         x_proj_weight_f, x_proj_weight_r, delta_proj_weight_f, delta_proj_weight_r,
         out_proj_weight, dbl_f, dbl_r, conv_out, delta,
         A_f, A_r, B_ssm, C_ssm, D_f, D_r, delta_bias_f, delta_bias_r,
         scan_intermediates, out) = ctx.saved_tensors

        L = xz.shape[-1]
        d_inner, dt_rank, d_state = ctx.d_inner, ctx.dt_rank, ctx.d_state
        x, z = xz.chunk(2, dim=1)
        if dresult.stride(-1) != 1:
            dresult = dresult.contiguous()

        conv_w, conv_b = _cat_conv(conv1d_weight_f, conv1d_bias_f,
                                   conv1d_weight_r, conv1d_bias_r)
        x2 = torch.cat([x, x.flip(-1)], dim=1).contiguous()
        if ctx.checkpoint_lvl == 1:
            conv_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x2, conv_w, conv_b, None, None, None, True)
            delta = torch.cat([
                torch.matmul(delta_proj_weight_f, dbl_f[:, :dt_rank]),
                torch.matmul(delta_proj_weight_r, dbl_r[:, :dt_rank]),
            ], dim=1).contiguous()

        # --- out_proj ---
        y = out[:, :d_inner] + out[:, d_inner:].flip(-1)
        sz = F.silu(z)
        gated = y * sz
        dres2 = rearrange(dresult, "b l e -> e (b l)")
        dgated = rearrange(out_proj_weight.t() @ dres2, "d (b l) -> b d l", l=L)
        dout_proj_weight = torch.einsum("eB,dB->ed", dres2,
                                        rearrange(gated, "b d l -> d (b l)"))
        dout_proj_bias = dresult.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None

        # --- gate: gated = y * silu(z) ---
        dy = dgated * sz
        sig = torch.sigmoid(z)
        dz = dgated * y * (sig * (1 + z * (1 - sig)))

        # --- y = out_f + flip(out_r) ---
        dout_scan = torch.cat([dy, dy.flip(-1)], dim=1).contiguous()

        A = torch.cat([A_f, A_r], dim=0).contiguous()
        D = torch.cat([D_f, D_r], dim=0).contiguous()
        delta_bias = torch.cat([delta_bias_f, delta_bias_r], dim=0).contiguous()
        # z was not given to the forward kernel, so `out` is not passed here either
        dconv_out, ddelta, dA, dB, dC, dD, ddelta_bias = selective_scan_cuda.bwd(
            conv_out, delta, A, B_ssm, C_ssm, D, None, delta_bias, dout_scan,
            scan_intermediates, None, None, True, False)

        dA_f, dA_r = dA[:d_inner], dA[d_inner:]
        dD_f, dD_r = dD[:d_inner], dD[d_inner:]
        ddb_f, ddb_r = ddelta_bias[:d_inner], ddelta_bias[d_inner:]

        # --- dt_proj (per direction) ---
        ddelta_f, ddelta_r = ddelta[:, :d_inner], ddelta[:, d_inner:]
        dt_in_f, dt_in_r = dbl_f[:, :dt_rank], dbl_r[:, :dt_rank]
        ddt_w_f = torch.einsum("bdl,brl->dr", ddelta_f, dt_in_f)
        ddt_w_r = torch.einsum("bdl,brl->dr", ddelta_r, dt_in_r)
        ddt_in_f = torch.matmul(delta_proj_weight_f.t(), ddelta_f)
        ddt_in_r = torch.matmul(delta_proj_weight_r.t(), ddelta_r)

        # --- x_proj (per direction): assemble d(x_dbl) from dt / B / C grads ---
        def _unwind(ddt_in, dB_g, dC_g, x_proj_w, conv_slice, dconv_slice):
            dxdbl = torch.cat([ddt_in, dB_g, dC_g], dim=1).contiguous()
            dx_proj_w = torch.einsum("bol,bdl->od", dxdbl, conv_slice)
            dconv_slice += torch.matmul(x_proj_w.t(), dxdbl)
            return dx_proj_w

        dx_proj_w_f = _unwind(ddt_in_f, dB[:, 0], dC[:, 0], x_proj_weight_f,
                              conv_out[:, :d_inner], dconv_out[:, :d_inner])
        dx_proj_w_r = _unwind(ddt_in_r, dB[:, 1], dC[:, 1], x_proj_weight_r,
                              conv_out[:, d_inner:], dconv_out[:, d_inner:])

        # --- depthwise conv ---
        dx2, dconv_w, dconv_b, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x2, conv_w, conv_b, dconv_out, None, None, None, None, False, True)
        dconv_w = rearrange(dconv_w, "d w -> d 1 w")
        dconv_w_f, dconv_w_r = dconv_w[:d_inner], dconv_w[d_inner:]
        if dconv_b is not None:
            dconv_b_f, dconv_b_r = dconv_b[:d_inner], dconv_b[d_inner:]
        else:
            dconv_b_f = dconv_b_r = None

        # --- x2 = cat([x, flip(x)]) ---
        dx = dx2[:, :d_inner] + dx2[:, d_inner:].flip(-1)
        dxz = torch.cat([dx, dz], dim=1)

        return (dxz,
                dconv_w_f, dconv_b_f, dconv_w_r, dconv_b_r,
                dx_proj_w_f, dx_proj_w_r,
                ddt_w_f, ddt_w_r,
                dout_proj_weight, dout_proj_bias,
                dA_f, dA_r, dD_f, dD_r, ddb_f, ddb_r,
                None)


def bi_mamba_inner_fn(xz, mamba_fwd, mamba_rev, out_proj_weight, out_proj_bias,
                      checkpoint_lvl=1):
    """Convenience wrapper pulling the two directions' parameters off their Mamba modules."""
    return BiMambaInnerFn.apply(
        xz,
        mamba_fwd.conv1d.weight, mamba_fwd.conv1d.bias,
        mamba_rev.conv1d.weight, mamba_rev.conv1d.bias,
        mamba_fwd.x_proj.weight, mamba_rev.x_proj.weight,
        mamba_fwd.dt_proj.weight, mamba_rev.dt_proj.weight,
        out_proj_weight, out_proj_bias,
        -torch.exp(mamba_fwd.A_log.float()), -torch.exp(mamba_rev.A_log.float()),
        mamba_fwd.D.float(), mamba_rev.D.float(),
        mamba_fwd.dt_proj.bias.float(), mamba_rev.dt_proj.bias.float(),
        checkpoint_lvl,
    )
