"""Shared utilities for the Mamba/Caduceus optimization benchmarks.

Everything here is deliberately dependency-light so the harness can run inside the
existing pixi env (torch 2.1.2 / mamba-ssm 2.1.0 / triton 2.1.0) with no new packages.

Timing uses CUDA events around a fully-synchronized region; memory is reported as
`torch.cuda.max_memory_allocated` (allocator high-water mark, excludes caching slack)
and `torch.cuda.max_memory_reserved` (what nvidia-smi would roughly show).
"""

import argparse
import contextlib
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Production configuration
#
# Mirrors slurm_scripts/train_joint_gm12878_mlmweight.sh with dataset.mlm=0, i.e.
#   model=caduceus d_model=256 n_layer=16 bidirectional=true
#   bidirectional_strategy=add bidirectional_weight_tie=true rcps=false
#   skip_embedding=true vocab_size=1 pad_vocab_size_multiple=1
#   encoder.joint=true encoder.norm=true (-> InstanceNorm1d)
#   dataset.length=524288 batch_size=1 trainer.precision=bf16
# ---------------------------------------------------------------------------
PROD = dict(
    d_model=256,
    n_layer=16,
    seqlen=524288,
    batch_size=1,
    d_state=16,
    d_conv=4,
    expand=2,
    # JointCNN encoder inputs: d_input1=6 (sequence one-hot + mask channels),
    # d_input2 = train.num_tracks + 1 = 2 (accessibility value + mask channel)
    d_input1=6,
    d_input2=2,
    # JointMaskingDecoder heads
    d_output1=5,
    d_output2=1,
    encoder_kernel_size=15,
)

SWEEP_SEQLENS = [16384, 32768, 65536, 131072, 262144, 524288]


def prod_ssm_cfg():
    """ssm_cfg exactly as in configs/model/caduceus.yaml."""
    return dict(
        d_state=PROD["d_state"],
        d_conv=PROD["d_conv"],
        expand=PROD["expand"],
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
    )


# ---------------------------------------------------------------------------
# Environment / device info
# ---------------------------------------------------------------------------
def env_info() -> Dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        import mamba_ssm
        info["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
    except ImportError:
        info["mamba_ssm"] = None
    try:
        import causal_conv1d
        info["causal_conv1d"] = getattr(causal_conv1d, "__version__", "unknown")
    except ImportError:
        info["causal_conv1d"] = None
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_mem_gb=round(props.total_memory / 1024**3, 1),
            sm_count=props.multi_processor_count,
            capability=f"{props.major}.{props.minor}",
        )
    return info


def check_cuda_extensions() -> Dict[str, bool]:
    """Confirm the compiled kernels are importable -- a missing one silently
    drops mamba onto a much slower pure-PyTorch reference path."""
    out = {}
    for name in ("selective_scan_cuda", "causal_conv1d_cuda"):
        try:
            __import__(name)
            out[name] = True
        except ImportError:
            out[name] = False
    try:
        from mamba_ssm.ops.triton.layer_norm import layer_norm_fn  # noqa: F401
        out["triton_layer_norm"] = True
    except ImportError:
        out["triton_layer_norm"] = False
    return out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass
class TimingResult:
    name: str
    seqlen: int
    batch_size: int
    variant: str = "baseline"
    fwd_ms: float = float("nan")
    bwd_ms: float = float("nan")
    step_ms: float = float("nan")
    fwd_std_ms: float = 0.0
    bwd_std_ms: float = 0.0
    peak_alloc_gb: float = float("nan")
    peak_reserved_gb: float = float("nan")
    tokens_per_s_fwd: float = float("nan")
    tokens_per_s_step: float = float("nan")
    n_iters: int = 0
    alloc_retries: int = 0
    oom: bool = False
    error: str = ""
    extra: Dict = field(default_factory=dict)


def _mean_std(xs: List[float]):
    n = len(xs)
    if n == 0:
        return float("nan"), 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def time_fwd_bwd(
    fwd: Callable[[], torch.Tensor],
    backward: bool = True,
    warmup: int = 2,
    iters: int = 5,
    zero_grad: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Time forward and (optionally) backward separately with CUDA events.

    `fwd` must return a scalar-reducible tensor. Backward is timed on `out.sum()`
    (or the tensor itself if already scalar) so the measured backward covers the
    whole graph built by `fwd`.
    """
    fwd_times, bwd_times = [], []

    def _one(record: bool):
        if zero_grad is not None:
            zero_grad()
        if record:
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            e0.record()
        out = fwd()
        if record:
            e1.record()
        if backward:
            loss = out if out.ndim == 0 else out.float().sum()
            loss.backward()
        if record:
            e2.record()
            torch.cuda.synchronize()
            fwd_times.append(e0.elapsed_time(e1))
            bwd_times.append(e1.elapsed_time(e2))
        del out
        if backward:
            # free the graph before the next iteration so peak memory is per-iter
            gc.collect()

    for _ in range(warmup):
        _one(record=False)
    cuda_sync()
    # Reset the *statistics* only -- deliberately NOT empty_cache(). Emptying the
    # allocator cache here would force the first timed iteration to re-cudaMalloc the
    # entire working set from the driver, which is a synchronous multi-hundred-ms cost
    # at this model's ~73 GB working set and silently penalises whichever variant uses
    # the most memory. Warmup above leaves the cache holding exactly what we need.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    for _ in range(iters):
        _one(record=True)
    cuda_sync()

    f_m, f_s = _mean_std(fwd_times)
    b_m, b_s = _mean_std(bwd_times)
    return dict(fwd_ms=f_m, fwd_std_ms=f_s, bwd_ms=b_m, bwd_std_ms=b_s)


def measure(
    name: str,
    fwd: Callable[[], torch.Tensor],
    seqlen: int,
    batch_size: int,
    variant: str = "baseline",
    backward: bool = True,
    warmup: int = 2,
    iters: int = 5,
    zero_grad: Optional[Callable[[], None]] = None,
    extra: Optional[Dict] = None,
    catch_errors: bool = True,
) -> TimingResult:
    res = TimingResult(name=name, seqlen=seqlen, batch_size=batch_size, variant=variant,
                       n_iters=iters, extra=extra or {})
    reset_memory()
    try:
        t = time_fwd_bwd(fwd, backward=backward, warmup=warmup, iters=iters, zero_grad=zero_grad)
    except torch.cuda.OutOfMemoryError as e:
        res.oom = True
        res.error = str(e).split("\n")[0]
        reset_memory()
        return res
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            res.oom = True
            res.error = str(e).split("\n")[0]
            reset_memory()
            return res
        if not catch_errors:
            raise
        # Record and continue: one broken micro-benchmark should not abort the sweep.
        res.error = str(e).split("\n")[0]
        reset_memory()
        return res
    except Exception as e:  # noqa: BLE001
        if not catch_errors:
            raise
        res.error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        reset_memory()
        return res
    res.fwd_ms = t["fwd_ms"]
    res.fwd_std_ms = t["fwd_std_ms"]
    res.bwd_ms = t["bwd_ms"] if backward else 0.0
    res.bwd_std_ms = t["bwd_std_ms"] if backward else 0.0
    res.step_ms = res.fwd_ms + res.bwd_ms
    res.peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
    res.peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    # cudaMalloc retries mean the allocator had to flush its cache mid-step: a strong
    # signal that the run is memory-bound and paying synchronization to stay alive.
    res.alloc_retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    tokens = seqlen * batch_size
    res.tokens_per_s_fwd = tokens / (res.fwd_ms / 1000.0) if res.fwd_ms > 0 else float("nan")
    res.tokens_per_s_step = tokens / (res.step_ms / 1000.0) if res.step_ms > 0 else float("nan")
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_results_table(results: List[TimingResult]) -> str:
    hdr = f"{'name':<34} {'variant':<16} {'B':>2} {'L':>8} {'fwd ms':>10} {'bwd ms':>10} {'step ms':>10} {'tok/s step':>12} {'peak GB':>9}"
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        if r.oom:
            lines.append(f"{r.name:<34} {r.variant:<16} {r.batch_size:>2} {r.seqlen:>8} {'OOM':>10} {'':>10} {'':>10} {'':>12} {'':>9}")
            continue
        if r.error:
            lines.append(f"{r.name:<34} {r.variant:<16} {r.batch_size:>2} {r.seqlen:>8}  ERROR {r.error[:40]}")
            continue
        lines.append(
            f"{r.name:<34} {r.variant:<16} {r.batch_size:>2} {r.seqlen:>8} "
            f"{r.fwd_ms:>10.2f} {r.bwd_ms:>10.2f} {r.step_ms:>10.2f} "
            f"{r.tokens_per_s_step:>12,.0f} {r.peak_alloc_gb:>9.2f}"
        )
    return "\n".join(lines)


def save_results(results: List[TimingResult], path: str, meta: Optional[Dict] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "meta": {**(meta or {}), "env": env_info(), "extensions": check_cuda_extensions(),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote {path}")


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
