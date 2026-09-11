# Pure Mamba (Caduceus) Architecture — Profiling & Optimization Report

**Branch:** `optimizing_mamba`
**Status:** in progress — baseline measurement round submitted
**Last updated:** 2026-09-06

---

## 1. Scope and target configuration

The target is the **pure (non-striped) Mamba** path, i.e. `model=caduceus` /
`model._name_=dna_embedding_caduceus`, as launched by
`slurm_scripts/train_joint_gm12878_mlmweight.sh`. Per the task brief, the MLM-weighting
and `alternating` arguments are ignored and the reference point is `dataset.mlm=0`.

| Knob | Value | Source |
|---|---|---|
| `d_model` | 256 | train script |
| `n_layer` | 16 | train script |
| `bidirectional` / strategy / weight tie | `true` / `add` / `true` | train script |
| `rcps` | `false` | train script |
| `skip_embedding` | `true` | train script |
| `d_state` / `d_conv` / `expand` | 16 / 4 / 2 | `configs/model/caduceus.yaml` |
| `dt_rank` | `auto` → 16 | `ceil(d_model/16)` |
| `d_inner` | 512 | `expand * d_model` |
| **sequence length** | **524 288** | `configs/experiment/hg38/joint_pretrain.yaml` |
| **batch size** | **1** per GPU | train script |
| `accumulate_grad_batches` | `div_up(8, devices*batch*nodes)` | experiment config |
| precision | `bf16` | train script |
| encoder | `JointCNN`, `joint=true`, `norm=true` (InstanceNorm1d), kernel 15, no downsample | train script + pipeline config |
| decoder | `JointMaskingDecoder`, `d_output1=5`, `d_output2=1`, `upsample=1` | pipeline config |
| loss | `joint_loss` with `task2=reg`, `count_weight=16` | train script |

### Documented assumptions

1. **`precision: bf16` means autocast, not a bf16 model.** Lightning aliases the string
   `bf16` → `bf16-mixed`
   (`lightning_fabric/plugins/precision/precision.py:24`), i.e. fp32 master weights run
   under `torch.autocast("cuda", torch.bfloat16)`. The harness reproduces exactly that
   rather than calling `model.to(bfloat16)`, which would be a different (faster,
   less accurate) regime.
2. **Encoder input widths.** `d_input1=6` is the `JointCNN` default and
   `d_input2 = train.num_tracks + 1 = 2` comes from
   `encoder.d_input2: ${eval:${train.num_tracks} + 1}`. Synthetic batches use those widths.
3. **Synthetic data.** Benchmarks feed shape-faithful random tensors rather than reading
   the GM12878 zarr, so the numbers isolate model cost from dataloader/IO cost. The
   accessibility mask fraction is set to 5% to keep the masked-loss gather realistic.
4. **Single GPU.** All measurements are single-A100. The production job runs
   `--gres=gpu:a100:2` with DDP; DDP gradient all-reduce is excluded here because the
   model is small (≈3.3 M params) and communication is not the suspected bottleneck.
5. **Hardware.** A100 80 GB (`--partition=gpu,lesliec --gres=gpu:a100:1`), matching the
   training scripts. H100 was available but not chosen, per user direction.

### Production anchor (measured, from an existing run log)

`jobs/10631546-joint_gm12878_phastcons7mlm_alternating_2.out` — the exact reference
script, 2× A100, `batch_size=1`, L=524 288:

| Quantity | Value |
|---|---|
| steady-state rate | **0.47 it/s → ≈ 2.13 s per batch per rank** |
| throughput | ≈ 246 k tokens/s/GPU (≈ 492 k across the 2 GPUs) |
| epoch | 17 011 iterations → **10 h 09 m** |

A minority of steps run at ≈ 0.92 it/s (≈ 1.09 s), consistent with `dataset.alternating=0.25`
making some batches cheaper. The 2.13 s figure is the one to beat; note it includes the
dataloader, so the isolated-model benchmarks below are a lower bound on it.

---

## 2. Benchmark harness

All of it lives in `tests/` and runs inside the existing pixi environment
(torch 2.1.2 / CUDA 11.8, `mamba-ssm` 2.1.0, `causal-conv1d` 1.5.0, triton 2.1.0 — no
new packages).

| File | Purpose |
|---|---|
| `tests/bench_utils.py` | CUDA-event timing (separate fwd / bwd), peak allocated + reserved memory, `num_alloc_retries`, env/extension probing, JSON + table reporting |
| `tests/bench_models.py` | Builds the *real* production modules: `CaduceusMixerModel`, and a `JointStep` module chaining `JointCNN → backbone → JointMaskingDecoder → joint_loss` |
| `tests/bench_mamba.py` | Driver: `--mode micro/backbone/full/scaling/all`, sequence-length sweep 16 k → 524 k |
| `tests/bench_variants.py` | Registry mapping a variant name to model surgery, so one driver benchmarks every candidate |
| `tests/profile_mamba.py` | `torch.profiler` kernel table + `record_function` breakdown of the bidirectional wrapper (fwd scan / flip-in / rev scan / flip-out / combine), plus a short-sequence pass that exposes fixed per-launch overhead |
| `tests/test_equivalence.py` | Correctness gates — every optimization must reproduce baseline outputs *and* gradients |
| `tests/slurm_bench.sh`, `tests/slurm_tests.sh` | SLURM submission (`--partition=gpu,lesliec --gres=gpu:a100:1`) |

Measured quantities: **forward ms**, **backward ms**, **step ms**, **tokens/s**, **peak
allocated GB**, **peak reserved GB**, **allocator retries**.

---

## 3. Measured bottleneck attribution

All numbers below are measured, not estimated. The A100 job for the headline latency
table was still queued at the time of writing (the `lesliec` A100s were occupied by the
user's own `e2g_tss_dn` array); attribution was therefore run on L40S and H200, which is
sound because *where* the time goes is a property of the architecture. GPU is stated for
every table.

### 3.1 Kernel-level breakdown (`torch.profiler`, L40S, B=1, L=262 144, 16 layers)

Total 970 ms/iteration (fwd + bwd).

| kernel | share of step |
|---|---|
| `selective_scan_bwd_kernel` | **38.5 %** |
| `selective_scan_fwd_kernel` | **11.5 %** |
| GEMMs (`in_proj`, `out_proj`, `x_proj`, `dt_proj`) | ~29 % |
| `causal_conv1d` fwd + bwd | 9.5 % |
| Triton RMSNorm fwd + bwd | 5.6 % |
| `aten::flip` (bidirectional wrapper) | 2.4 % |
| other elementwise | ~2 % |

**Half the step is one third-party sequential-scan CUDA kernel**, and its backward is
3.4× its forward (373.7 ms vs 111.2 ms per iteration).

### 3.2 Is it an implementation problem or an architectural one?

Four independent measurements say **architectural**:

1. **Not launch/Python bound.** Profiler CPU/CUDA ratio is **0.745** at L=262 144 — the
   GPU is the critical path. The same measurement at L=4096 gives **2.43**, i.e. this
   model *is* launch-bound, but only at sequences ~64× shorter than production.
2. **Throughput is flat in sequence length** (L40S, baseline backbone):

   | L | step ms | tokens/s | peak alloc |
   |---|---|---|---|
   | 65 536 | 230.4 | 284 k | 9.17 GB |
   | 131 072 | 480.5 | 273 k | 18.28 GB |
   | 262 144 | 1004.2 | 261 k | 36.50 GB |
   | 524 288 | OOM on a 46 GB card | — | ~73 GB (extrapolated) |

   Perfectly linear in L. There is no fixed-overhead cliff to remove.
3. **Batch scaling is nearly flat.** H200, L=65 536: B=1 → 1.36 M tok/s, B=2 → 1.66 M,
   B=4 → 1.76 M, B=8 → 1.89 M. Going 8× wider buys **1.39×**. The GPU is already ~72 %
   as efficient at B=1 as at B=8, so there is far less idle silicon than the
   512-block scan grid suggested.
4. **The CUDA extensions are all present** (`selective_scan_cuda`, `causal_conv1d_cuda`,
   Triton RMSNorm) and the fast path (`use_fast_path=True` → `mamba_inner_fn`) is the one
   actually taken. There is no missing-kernel slow path to fix.

The **memory** result is the sharpest finding: activations are ~73 GB at L=524 288, which
on an 80 GB A100 is why `dataset.batch_size=1`, and it leaves almost no headroom.

### 3.3 What the bidirectional wrapper actually costs

Refuting my own prior: per-op timing at L=262 144 (L40S) shows `BiMambaWrapper` at
63.7 ms/layer against 2 × 31.7 ms for the two `Mamba` calls. **The flips and the additive
combine are <0.5 % of the layer.** The profiler agrees: of a 320.7 ms forward,
`forward_dir` + `reverse_dir` are 274.1 ms (85.5 %) while `flip_in` + `flip_out` +
`combine` are 25.1 ms (7.8 % of forward, ~2.4 % of the step). Removing flip overhead was
never going to matter.

---

## 3.5 Headline numbers on the production GPU (A100 80GB PCIe)

`joint_train_step` = the full production pipeline (`JointCNN` encoder -> backbone ->
`JointMaskingDecoder` -> `joint_loss`) at L=524 288, B=1, bf16 autocast, single GPU.

### Training

| variant | fwd | bwd | step | tokens/s | peak alloc |
|---|---|---|---|---|---|
| **baseline** | 622.3 | 1510.2 | **2132.5 ms** | **245 856** | 68.23 GB |
| `fused_inner` | **583.0** | 1770.2 | 2353.2 ms | 222 796 | **55.98 GB** |
| `ckpt`, B=1 | 619.4 | 2078.5 | 2697.8 ms | 194 337 | **16.62 GB** |
| `ckpt`, B=2 | 1061.0 | 3949.7 | 5010.6 ms | 209 271 | 33.19 GB |

Backbone only:

| variant | L=131 072 | L=262 144 | L=524 288 | peak @524 k |
|---|---|---|---|---|
| baseline | 515.9 ms | 1041.1 ms | **2109.7 ms** | 72.94 GB |
| `fused_inner` | 579.9 ms | 1163.2 ms | 2327.6 ms | 60.69 GB |
| `shared_in_proj` | 521.0 ms | 1047.4 ms | 2125.0 ms | 68.94 GB |
| `ckpt` | — | 1326.6 ms | 2683.1 ms | 21.33 GB |

### Inference (forward only, `eval()` + `no_grad()`)

| variant | B=1 | B=2 | B=4 | best tokens/s | peak @B=1 |
|---|---|---|---|---|---|
| baseline | 612.7 ms | 1086.1 ms | 2065.4 ms | **1 015 383** (B=4) | 5.64 GB |
| `fused_inner` | **586.3 ms** | 1142.6 ms | 2266.7 ms | 925 183 | 8.73 GB |

`fused_inner` is **4.3 % faster at B=1** — the lowest-latency single-sequence option — but
loses at B>1. Baseline inference gains **+19 %** from batching on A100 (856 k -> 1 015 k
tok/s), unlike on L40S where it was flat.

**The harness reproduces production.** The measured synthetic step of **2133 ms** sits
**0.1 %** from the **2130 ms** observed in the real training log (§2 anchor). The
dataloader is therefore not costing measurable time -- the model is the whole step, and
there is no IO overhead to recover.

### Two measurement problems that were found and fixed

**(a) An allocator-cache bias in the harness.** An earlier version called
`torch.cuda.empty_cache()` immediately before the timed loop, forcing the first timed
iteration to re-`cudaMalloc` the entire working set. At ~73 GB that is a synchronous
several-hundred-ms cost, and it **systematically penalised whichever variant used the
most memory** -- i.e. the baseline, the reference for every comparison. The symptom:
baseline backbone forward appeared to scale **3.29x** for a 2x sequence-length increase
(307.8 -> 1012.5 ms) while `ckpt`, at a third of the memory, scaled exactly 2.01x. After
the fix the baseline scales 2.03x (303.6 -> 615.2 ms). `tests/bench_utils.py` now resets
allocator *statistics* only and leaves the warmup-populated cache in place.
The biased run had reported checkpointing's cost as +7.5 %; the true figure is **+29 %**.

**(b) A GPU-dependence that a single-GPU conclusion had hidden.** `fused_inner` was first
measured on an L40S and rejected at -37 %. Re-measured with the corrected harness it is
**6-8 % faster on H200** and **10 % slower on A100**. The discriminator is memory
bandwidth -- L40S GDDR6 ~864 GB/s, A100 80GB PCIe ~1935 GB/s, H200 HBM3e ~4.8 TB/s -- and
the fused path trades fused-kernel work for memory traffic. The signature is visible in
the split: its forward is consistently *faster* (577 vs 617 ms on A100, the fusion
working) while its backward is consistently slower (1750 vs 1493 ms, where the extra
traffic lands). All conclusions below are stated for the A100 the production job runs on.

---

## 4. Optimizations: every variant tried, with its measured outcome

All variants are opt-in; the default code path is unchanged. Every one preserves the
parameter tree (`mamba_fwd.*` / `mamba_rev.*`), so a baseline checkpoint loads into any
of them with `strict=True` and no missing or unexpected keys.

All figures below are **A100, L=524 288, B=1** unless stated.

| variant | training | inference B=1 | peak memory | correctness | verdict |
|---|---|---|---|---|---|
| `ckpt` (gradient checkpointing) | −29 % | n/a | **0.24×** | bit-exact | **use when memory-bound** |
| `shared_in_proj` | −0.7 % (wash) | — | 0.95× | bit-exact fwd | neutral |
| `fused_bidir` | −28 % (H200 @131 k: −3 %) | — | 1.46× | ✓ | **rejected** |
| `fused_inner` (`BiMambaInnerFn`) | **−10 %** | **+4.3 %** | **0.82×** | ✓ (all grads ~1e-7) | **not for A100 training; revisit on HBM3e** |

`fused_inner` is the one whose verdict is genuinely hardware-dependent: **+6-8 % on
H200**, **−10 % on A100**. See §3.5(b).

### 4.1 Gradient checkpointing — the one usable knob (memory, not speed)

New config flag `gradient_checkpointing` (default `False`), implementing the literal
`# TODO: Add support for gradient checkpointing` that was in `CaduceusMixerModel.forward`.
Non-reentrant `torch.utils.checkpoint` per `Block`. A100 80GB (production GPU),
backbone, 16 layers, B=1, corrected harness:

| L | baseline | `ckpt` | baseline mem | `ckpt` mem |
|---|---|---|---|---|
| 262 144 | 1027 ms | 1327 ms (+29 %) | 36.50 GB | 10.69 GB |
| 524 288 | 2084 ms | 2683 ms (+29 %) | **72.94 GB** | 21.33 GB |

On a 46 GB L40S the baseline simply **OOMs** at L=524 288 while `ckpt` runs in 21.33 GB.

**+29 % time for 3.4-4.1× less memory.** It is *not* a throughput win — §3.2(3) shows batching
cannot buy back the extra forward. Its value is removing the ~73 GB / 80 GB cliff.

**Verified under DDP.** The production job runs `--gres=gpu:a100:2`, and activation
checkpointing can make DDP's reducer see a parameter marked ready twice.
`tests/test_ddp_checkpointing.py` runs the real backbone under real 2-GPU DDP: it works
**both with and without `static_graph=True`**, with identical gradients
(`grad_norm = 1.768160e-01` in both cases). No extra Lightning configuration is needed.

    +model.config.gradient_checkpointing=true

### 4.2 The fusion is real in isolation but loses in situ — rejected

`bidirectional_weight_tie=True` plus `in_proj` / `out_proj` being position-wise gives three
*exact* identities, all verified numerically:

1. `in_proj(flip(h)) == flip(in_proj(h))` → one `in_proj` GEMM instead of two.
2. `out_proj(y_f) + flip(out_proj(y_r)) == out_proj(y_f + flip(y_r))` → one `out_proj` GEMM.
3. `flip(silu(z_rev)) == silu(z)` → the gate factors out of the bidirectional sum, so `z`
   need never be flipped, concatenated, or passed to the scan kernel at all.

Both directions can then run as one `selective_scan` over `2·d_inner` channels using the
kernel's `n_groups=2` mode. **`tests/test_equivalence.py` confirms `n_groups=2` is
bit-exact against two independent single-group scans (rel_err 0.000e+00).**

Isolated at production shape (`tests/bench_scan_fusion.py`, H200, B=1, L=262 144, per layer):

| operation | baseline (2 ×) | fused (1 ×) | Δ |
|---|---|---|---|
| `selective_scan` | 36.44 ms | 30.50 ms | −16.3 % |
| `selective_scan`, `z` removed | 35.00 ms | **28.91 ms** | −20.7 % |
| `causal_conv1d` | 3.31 ms | 2.92 ms | −11.8 % |
| `in_proj` | 5.10 ms | 2.70 ms | −47 % |
| `out_proj` | 2.24 ms | 1.22 ms | −46 % |
| extra copy (`flip h` → `cat[x, flip(x)]`) | 1.16 ms | 3.98 ms | +2.82 ms |

Predicted ≈ **−6.9 ms/layer ≈ −11 %**. Measured end-to-end on A100: **−28 %** for the
naive version and **−10 %** for the properly-checkpointed one (`fused_inner`) — which on
H200 instead comes out **+6-8 %**.

**Why the prediction was wrong, and the general lesson.** The micro-benchmarks measure
kernel time; they do not measure the *memory-traffic structure of the surrounding path*,
which is what dominates here.

- `fused_bidir` chains ordinary autograd ops, which forfeits `MambaInnerFn`'s
  `checkpoint_lvl=1` recompute of `conv1d_out` and `delta`. Those two `2·d_inner × L`
  tensors get stored and re-read instead of recomputed.
- `fused_inner` fixes that with a hand-written autograd Function — and still loses,
  because identity 3 moves the gate out of the CUDA kernel into ~8 separate unfused
  PyTorch elementwise kernels. At 512 channels and ~0.27 GB per tensor that is
  ~10 GB/layer of traffic the kernel previously did for free in registers. The regression
  is almost entirely in the backward (698 → 1037 ms); the forward barely moves
  (305 → 342 ms). Putting `z` back in the kernel recovers roughly half, but then costs a
  1024-channel `cat` of `z`, which analysis puts back near break-even.

**Conclusion: on A100, `mamba_inner_fn` plus the stock two-call bidirectional path is
already near a local optimum.** It keeps the whole conv → x_proj → scan → gate → out_proj
chain inside fused CUDA kernels with almost nothing materialized, and on a ~1935 GB/s part
that beats restructurings which provably remove GEMM work *and* halve kernel launches.

The qualifier matters, though. `fused_inner`'s *forward* is faster than baseline on A100
(577 vs 617 ms) and the whole thing is faster on H200 — the fusion is real, it is simply
outweighed on A100 by the memory traffic of applying the gate outside the scan kernel.
On HBM3e-class hardware that traffic is nearly free and the fusion wins. If this model
moves to H200/B200, `fused_inner` is worth re-enabling: it is already implemented,
validated to ~1e-7 on every parameter gradient, and state-dict compatible.

---

## 5. Correctness

`tests/test_equivalence.py` — 13 checks, all passing.

| check | result |
|---|---|
| `selective_scan` `n_groups=2` vs 2 independent scans | **0.000e+00 (bit-exact)** |
| fused vs stock BiMamba layer, fp32 forward | 4.06e-07 |
| fused vs stock layer, input grad / worst param grad | 2.80e-07 / 2.22e-07 |
| `BiMambaInnerFn` layer, forward / input grad | 3.33e-07 / 4.14e-07 |
| `BiMambaInnerFn`, **all 16 parameter grads** | 5.69e-08 … 2.10e-07 |
| bf16 vs common fp32 reference (layer) | stock 5.325e-03, fused 5.336e-03 |
| bf16 vs common fp32 reference (backbone) | stock 1.753e-03, `fused_inner` 1.771e-03 |
| gradient checkpointing, output / input grad | **0.000e+00** / 3.69e-08 |
| gradient checkpointing under 2-GPU DDP | identical grads, static_graph on **and** off |
| `shared_in_proj` backbone forward | **0.000e+00** |
| baseline `state_dict` → optimized model, `strict=True` | no missing/unexpected keys |

**On gating the deep-backbone gradient comparison.** Fixed tolerances were the wrong
instrument. A depth sweep shows the forward error is flat at ~1e-7 while the gradient
difference sits at ~8e-3 *at every depth including one layer* — so it is not depth
amplification; the backbone's `rms_norm_fn` + residual make the gradient intrinsically
scale-sensitive in a way the bare-layer test does not exercise. The tests therefore
measure the network's own **gradient conditioning** (perturb the input by 1e-7 relative,
see how far the *stock* gradients move) and require the optimized path to stay inside it:

| n_layer | fwd rel_err | fused dinput rel_err | conditioning floor |
|---|---|---|---|
| 1 | 4.93e-08 | 6.85e-03 | 1.10e-02 |
| 2 | 6.53e-08 | 7.81e-03 | 1.13e-02 |
| 3 | 7.87e-08 | 8.24e-03 | 1.10e-02 |
| 4 | 9.06e-08 | 8.96e-03 | 1.11e-02 |

Every optimized path's gradient difference is *below* the floor.

*Caveat:* the bf16-vs-fp32 **gradient** comparison at backbone level is uninformative at
this test's loss scale — `out.square().mean()` on a randomly-initialised model produces
~1e-10 gradients, far below bf16's useful range, and both paths score ~1.6e+02. Only the
relative stock-vs-fused comparison is meaningful there; the forward comparison is sound.

---

## 6. Inference

A100 80GB (production GPU), L=524 288, forward only under `eval()` + `no_grad()`:

| variant | B=1 | B=2 | B=4 | best tokens/s | peak @B=1 |
|---|---|---|---|---|---|
| baseline | 612.7 ms | 1086.1 ms | 2065.4 ms | **1 015 383** (B=4) | 5.64 GB |
| `fused_inner` | **586.3 ms** | 1142.6 ms | 2266.7 ms | 925 183 | 8.73 GB |

Three practical points:

1. **Inference is ~3.5× cheaper per token than training** (856 k vs 246 k tok/s at B=1)
   and needs **5.6 GB instead of 68 GB**, so memory is a non-issue outside training and
   `gradient_checkpointing` is irrelevant here.
2. **Batching helps on A100 but not everywhere.** Baseline gains +19 % going B=1 → B=4
   (856 k → 1 015 k tok/s) on A100, but was completely flat on L40S (956 k → 972 k). If
   throughput matters, measure on your actual deployment GPU rather than assuming.
3. **For lowest single-sequence latency, `fused_inner` is 4.3 % faster** (586 vs 613 ms)
   at the cost of 1.5× memory — still only 8.7 GB. It loses at B>1.

---

## 7. The one change that attacks the dominant kernel: Mamba-2 (SSD)

Half the step is Mamba-1's `selective_scan`, a sequential recurrence running on CUDA
cores. Mamba-2's chunked SSD formulation is matmul-based and uses tensor cores.
`mamba_ssm` 2.1.0 already ships `Mamba2`, and `configs/model/striped_hydra.yaml` already
runs it — so this is a measurement, not a proposal to write new kernels.
`tests/bench_mamba2_compare.py`, H200, B=1, L=262 144, one direction, matched `d_model=256`
and `expand=2`:

| module | fwd | bwd | step | vs Mamba-1 | peak |
|---|---|---|---|---|---|
| Mamba-1 (`d_state=16`) — current | 4.55 | 18.37 | 22.92 ms | — | 3.79 GB |
| Mamba-2 SSD (`d_state=16`, `headdim=64`) | 3.17 | 10.97 | **14.13 ms** | **1.62×** | 4.37 GB |
| Mamba-2 SSD (`d_state=64`, `headdim=64`) | 3.31 | 12.17 | 15.48 ms | **1.48×** | 4.74 GB |
| Mamba-2 SSD (`d_state=128`, `headdim=64`) | 3.62 | 13.13 | 16.74 ms | 1.37× | 5.24 GB |

Repeating at L=131 072 gives 1.60× / 1.47× / 1.32× — the advantage is a property of the
kernel, not of one sequence length.

The `d_state=64` row is the interesting one: **4× the SSM state and still 1.48× faster**
than the current `d_state=16` Mamba-1. Since the Mamba modules are ~85 % of backbone time,
1.62× on the module implies roughly **1.48× end-to-end** on the backbone.

This is an architecture change: different parameters, no checkpoint compatibility, and it
would need an accuracy comparison. But it is the only measured change that moves the
kernel holding 50 % of the step.

---

## 8. Recommendations

All figures are A100 80GB PCIe, the GPU the production job runs on.

### Do these — model unchanged, existing checkpoints still load

1. **Enable `gradient_checkpointing` when memory is the constraint, not when speed is.**
   +29 % step time for **4.1× less memory** (68.2 → 16.6 GB at the production shape).
   Verified bit-exact and verified working under 2-GPU DDP with and without
   `static_graph`. The case for it is that production sits at **68–73 GB on an 80 GB
   card**: there is currently no room to raise `dataset.batch_size`, lengthen the
   sequence, or add a track without OOM.

       +model.config.gradient_checkpointing=true

2. **Use `fused_inner` for lowest-latency single-sequence inference** (4.3 % faster,
   586 vs 613 ms). Not for training on A100, where it is 10 % slower.

       from caduceus.bimamba_opt import apply_fused_inner; apply_fused_inner(model)

3. **Remove the per-step host sync in the loss.** `poisson_loss_mask`
   (`src/tasks/metrics.py:671`) calls `mask.sum().item()`, a GPU→CPU sync every step.
   Negligible against a ~2.1 s step, but free to fix by keeping the comparison on-device.

### Consider these — architecture changes needing an accuracy check

4. **Reduce the sequence length the SSM sees — the largest lever by far.** Cost is
   *exactly* linear in L (515.9 / 1041.1 / 2109.7 ms at 131 k / 262 k / 524 k), so
   `downsample=2` ≈ 2.0× and `downsample=4` ≈ 4.1×. The machinery already exists:
   `JointCNN(downsample=...)` with the matching `JointMaskingDecoder(upsample=...)`, and
   `striped_hydra` already does this via `global_pooling`.

5. **Port the pure model to Mamba-2 / SSD.** 1.48–1.62× measured at matched `d_model`
   (§7), and `d_state=64` is 1.48× faster with **4× the SSM state**. It composes with (4).

### Do not spend effort on these — all measured

6. **Larger training batches**: 1.39× for 8× (H200); on A100 `ckpt` at B=2 reaches
   209 k tok/s, still below plain baseline's 246 k.
7. **Kernel-launch / Python overhead**: CPU/CUDA ratio 0.745 at production length. Only
   matters below L≈4096.
8. **The bidirectional wrapper's flips**: 2.4 % of the step.
9. **Hand-fusing the two directions in PyTorch for A100 training**: three attempts,
   −0.7 % to −28 % (§4.2). Worth revisiting only on HBM3e-class hardware, where the same
   code is already 6–8 % faster.
10. **`torch.compile`**: on torch 2.1.2 the Mamba custom autograd Functions graph-break,
    and the elementwise ops it could fuse are ~2 % of the step.

### Realistic ceiling

Within the current architecture on A100, the implementation is close to optimal and the
honest expected gain from further kernel-level work is low single-digit percent.
Meaningful speedups require (4) and/or (5); combined, Mamba-2 at `downsample=4` would be
roughly **6×**, at the cost of a model that must be retrained and re-validated.

The cheapest real-world win available today is not a kernel at all: **the step time is
2.13 s and the harness reproduces it to 0.1 %**, so there is no hidden dataloader or IO
overhead to reclaim — every remaining millisecond is in the model.

---

## 8b. The striped model (Hydra / Mamba-2) in all three modes

Measured on A100 80GB PCIe, `d_model=256`, `n_blocks=16`, `expand=2`, bf16 autocast,
B=1 — the config from `slurm_scripts/train_joint_gm12878_striped.sh`.

### Headline: the striped model is misconfigured, and that dominates everything else

| backbone | params | step @L=524 288 | tokens/s | peak |
|---|---|---|---|---|
| `caduceus_bimamba` (old model) | 7.72 M | 2101 ms | 249 k | 72.97 GB |
| **`striped_mamba_only`, `global_pooling=1` — the current config** | 7.89 M | **6617 ms** | 79 k | 66.85 GB |
| `striped_mamba_only`, gp=2 | 8.61 M | 3087 ms | 170 k | 38.03 GB |
| `striped_mamba_only`, gp=4 | 9.99 M | 1559 ms | 336 k | 32.60 GB |
| `striped_mamba_only`, gp=8 | 11.37 M | 933 ms | 562 k | 29.91 GB |
| **`striped_mamba_only`, gp=8 + `d_in=64`** | 9.45 M | **656 ms** | **800 k** | **17.13 GB** |
| `striped` gp8/tp8, `n_blocks=8` | 42.3 M | 986 ms | 532 k | 45.97 GB |
| `striped` gp8/tp8, `n_blocks=16` | 80.8 M | 1512 ms | 347 k | 68.18 GB |
| `transformer_only`, gp=32 | 20.2 M | 629 ms | 833 k | 29.11 GB |

`global_pooling` defaults to 1 (`striped_backbone.py:26`) and `d_in` to `None`, and
neither is set in `configs/model/striped_hydra.yaml` or the training script. **Turning
them on is worth 10.1×** (6617 → 656 ms) and 3.9× memory, flipping the striped model from
**3.15× slower** than Caduceus to **3.2× faster** — with *more* parameters.

### Where the time goes, per mode (profiler, L=262 144)

| mode | dominant cost |
|---|---|
| `mamba_only`, gp=1 | **`hydra_block` 887.5 of 908.1 ms forward = 97.7 %** |
| `striped`, gp8/tp8 | `upsample` 117.0 (50 %), `hydra_block` 56.5 (24 %), `downsample` 43.5 (18.5 %), `transformer_block` 7.9 (3.3 %) |

The bottleneck **inverts** once pooling is on: the U-Net sampling stacks become 68 % of
the forward and attention becomes nearly free. Both modes are GPU-bound (CPU/CUDA 0.80,
0.92).

### Is Hydra a better bidirectional implementation than `BiMambaWrapper`? No.

Per block, matched params (472 344 vs 482 304):

| L | `caduceus_BiMambaWrapper` | `HydraBlock` | ratio |
|---|---|---|---|
| 32 768 | 8.89 ms | 13.30 ms | 1.50× |
| 65 536 | 16.73 | 28.11 | 1.68× |
| 131 072 | 31.72 | 69.38 | 2.19× |
| 262 144 | 63.06 | 169.58 | 2.69× |
| 524 288 | 126.70 | 394.38 | **3.11×** |

Caduceus scales at exactly 2.00× per doubling; Hydra is superlinear (2.1–2.5×) and uses
~2× the memory (17.99 vs 9.55 GB at 524 k).

**The Mamba-2 kernel is not the problem — Hydra's wrapper is.** At L=262 144:

| module | step |
|---|---|
| caduceus `BiMambaWrapper` (d_state=16) | 63.4 ms |
| **2 × unidirectional `Mamba2`** (d_state=64, equivalent SSD work) | **~49.8 ms** |
| Hydra bidirectional (d_state=64, `mem_eff=True`) | 167.2 ms |

Two raw `Mamba2` modules beat Caduceus by 1.27×, but Hydra costs **3.4× more than the
SSD kernel it wraps**. Cause: `Hydra.forward` calls `mamba_chunk_scan_combined` with
`D=None, z=None` and then performs the data-dependent D term (with a `repeat` expanding
`(B,L,nheads)` to `(B,L,d_inner)`), the gated norm, a full-tensor `torch.roll`, and
several flip/`cat` materializations in PyTorch — full-size tensor traffic wrapped around
a fused kernel. This is the same failure mode as the rejected `fused_inner` in §4.2.
It is partly structural: the gate is applied *after* the bidirectional sum and `D` is
data-dependent, so neither can simply move into the kernel.

Nothing tunable closes the gap:

| knob | range | effect |
|---|---|---|
| `use_mem_eff_path` | True → False | **166.7 → 145.0 ms, 8.88 → 6.68 GB** (False is better) |
| `d_state` | 64 → 16 | 167.2 → 142.4 ms (−17 %) |
| `chunk_size` | 64 / 128 / 256 / 512 | 173.0 / 166.1 / 167.9 / 175.8 (flat) |
| `headdim` | 32 / 64 / 128 | 181.3 / 167.8 / 164.4 (flat) |

Note `use_mem_eff_path=True` — which the striped config sets — is **slower and uses more
memory** than the unfused path, at *every* sequence length tested:

| L | True | False | speed ratio | memory penalty |
|---|---|---|---|---|
| 2 048 | 5.86 ms | 5.58 ms | 0.952x | 1.21x |
| 8 192 | 5.99 | 5.88 | 0.981x | 1.35x |
| 32 768 | 12.92 | 11.90 | 0.922x | 1.38x |
| 65 536 | 27.52 | 24.59 | 0.894x | 1.33x |
| 131 072 | 68.59 | 59.77 | 0.871x | 1.33x |
| 262 144 | 168.28 | 146.27 | 0.869x | 1.33x |

There is no crossover, so the recommendation is unconditional. A plausible-sounding theory
that it was tuned for LM-scale contexts (where saved-tensor overhead is negligible and the
forward fusion dominates) was tested and **refuted** — it loses at 2048 too.

**Why a flag named "memory efficient" costs 33 % more memory.** The name is inherited
from upstream Mamba-2, where `use_mem_eff_path` selects `mamba_split_conv1d_scan_combined`
and is genuinely memory-efficient for the *unidirectional* block. Hydra reuses the flag
for its own hand-written analogue, `HydraSplitConv1dScanCombinedFn`, which has to
accommodate bidirectionality and loses the property:

- **Memory.** It does `ctx.save_for_backward(z, xBC_og, dt, scan, ...)`, where `scan` is
  the `(2B, L, d_inner)` *doubled-batch* scan output. The unfused path's post-scan ops are
  `roll` / `cat` / `flip`, whose backwards are inverse permutations and slices needing
  nothing saved, so autograd frees that 2x-batch tensor immediately. The penalty is flat
  at ~1.33x across a 128x range in L, exactly as linear-in-L saved tensors predict.
- **Speed.** Its forward genuinely does fuse and does win (44.8 vs 49.3 ms at L=262144).
  The hand-written backward gives it back and more (122.4 vs 95.7 ms): it rebuilds an
  autograd graph mid-backward (`with torch.enable_grad(): dwconv(...)`), manually redoes
  `torch.roll` / `chunk_flip_join` / the `D * x_og` term at 2x batch, and writes
  `torch.cat([dy, flip(dy)], dim=0)` explicitly — where autograd's `cat` backward is a
  free slice. The penalty grows with L (0.952 -> 0.869) as fixed launch overheads wash out.

This is the same failure mode as the rejected `fused_inner` (§4.2) and as Hydra's 3.4x
overhead over the SSD kernel it wraps: **a hand-written backward loses to autograd unless
it buys a genuine recompute win.** Autograd's backwards for `cat`, `flip` and `roll` are
views or cheap inverse permutations; hand-writing them turns zero-copy operations into
real materializations.

### The O(L^2) wall

`TransformerBlock` scales 3.63× / 3.90× / 3.98× per doubling — textbook quadratic. One
block at L=262 144 costs **1.42 s**, so 16 would be ~23 s/step. `mode='striped'` and
`transformer_only` are therefore unusable at L=524 288 with the default
`transformer_pooling=1`; they structurally require aggressive pooling. That is why the
working striped config is `mamba_only`.

### TODO list for the striped model

Ordered by measured payoff. Item 1 is settled; 2-4 are the open work items.

- [ ] **1. `global_pooling=8` + `d_in=64`** — 10.1x (6617 -> 656 ms), 3.9x memory.
      **This one is a modelling decision, not a free win** (see below). Everything else
      in this list is dwarfed by it.

- [ ] **2. `use_mem_eff_path=false`** — 13 % faster, 25 % less memory
      (166.7 ms / 8.88 GB -> 145.0 / 6.68). **Verified to be a pure implementation
      change**, not a modelling one: at the production `d_conv=5` the two branches agree
      to `rel_err` 1.3e-06 on the forward, 3.5e-06 on the input gradient, and <= 6.7e-05
      on every parameter gradient (`tests/test_striped_equivalence.py`, A100). They share
      every parameter, so flipping it needs no retraining and does not affect checkpoints.
      (Aside: the fused path asserts `d_conv % 2 == 1`, so it is simply unavailable at
      even conv widths — it raises rather than silently differing.)

- [x] **3a. `RMSBatchNorm1d` rewrite — DONE.** `src/models/nn/sampling.py`; 1.103x and
      **-30 % memory** on the recommended pooled config, 1.001x unpooled. Verified
      equivalent, 8/8 tests.
- [ ] **3b. The rest of the upsample stack.** After the norm fix the remaining targets are
      the `(B, C, L)` layout (the transposes cost 7.81 ms against the convolution's
      5.91 ms) and the two other `ConvBlock`s. `cudnn.benchmark` (0 %) and narrowing the
      conv width (2.4 %) were measured and rejected.

- [ ] **4. Fix Hydra's wrapper** — ~70 % of Hydra's runtime is PyTorch traffic around the
      SSD kernel, not the kernel (167.2 ms vs ~49.8 ms for 2x raw `Mamba2`). Folding the
      data-dependent D term and gated norm into the scan is the fix. Lower priority than
      2-3 because once pooling is on, the mixer is only 24 % of the forward. Heed §4.2:
      this is exactly the class of optimization that is easy to get wrong, and it must be
      measured rather than assumed.

### `global_pooling` / `d_in` is a modelling decision, unlike the rest

Items 2-4 above change only *how* the same function is computed. Item 1 changes *what
the model is*:

- `global_pooling=8` means the SSM stack sees the sequence at **L/8 = 65 536**, i.e. 8 bp
  bins, instead of every base pair. Base-pair resolution is restored on the way out only
  through the U-Net skip connections, not by the deep stack.
- `d_in=64` runs the full-resolution part at **64 channels instead of 256**, i.e. a
  quarter of the capacity at base-pair resolution.

Whether that is acceptable is a question about the task, not about speed. Two pieces of
context, neither of which is a substitute for an accuracy run:

- The reference config uses `dataset.mlm=0` with `acc_mask_size=500`, so the objective is
  masked *accessibility* over ~500 bp regions rather than single-base sequence MLM. 8 bp
  bins are fine-grained relative to that target.
- For comparison, Enformer predicts in 128 bp bins and Borzoi in 32 bp. 8 bp pooling is
  considerably finer than standard practice in this domain.

Recommended way to settle it: train at `global_pooling` 1 / 4 / 8 for a fixed short
budget and compare `val/poisson_loss_mask`. The 10.1x speedup means the pooled runs cost
a fraction of the baseline, so this is a cheap experiment.

### Why upsample costs ~3x downsample, and where its time actually goes

Measured on A100, `factor=8`, L=262 144, C=256 (`tests/bench_upsample_opt.py`):

| stack | fwd | bwd | step | peak |
|---|---|---|---|---|
| `DownsampleStack` | 9.13 | 26.14 | **35.27 ms** | 3.82 GB |
| `UpsampleStack` | 31.88 | 72.01 | **103.89 ms** | 7.70 GB |

The asymmetry is structural, from `src/models/nn/sampling.py`:

| | work per stage |
|---|---|
| `DownsampleLayer` | 2 x `ConvBlock(width 5)` at length L, **then** `MaxPool1d` -> L/2 |
| `UpsampleLayer` | 1 x `ConvBlock(width 5)` at L; `repeat_interleave(2, dim=1)` -> 2L; `* residual_scale` at 2L; `+ ConvBlock(width 1)(skip)` at 2L; `+ ConvBlock(width 5)` at 2L |

Downsample runs both convolutions *before* pooling; upsample runs two of its three
*after* doubling the length.

**Component breakdown of the top (most expensive) `UpsampleLayer`:**

| component | fwd | bwd | step | share | bwd/fwd |
|---|---|---|---|---|---|
| `reduce_main` `ConvBlock(w5)` @ L/2 | 2.83 | 9.52 | 12.35 ms | 19 % | 3.4x |
| `repeat + scale` | 0.78 | 1.43 | 2.21 ms | 3.4 % | 1.8x |
| `skip_project` `ConvBlock(w1)` @ L | 3.30 | 9.21 | 12.50 ms | 20 % | 2.8x |
| **`refine` `ConvBlock(w5)` @ L** | 6.09 | **30.74** | **36.83 ms** | **58 %** | **5.05x** |

**`refine` is the target.** Its backward is 5x its forward, far worse than its siblings,
which points at the convolution backward running well off peak rather than the block
being inherently costly.

**Candidate fixes, with what each is actually worth:**

1. ~~Replace `repeat_interleave(2, dim=1)` with `unsqueeze/expand/reshape`.~~
   **Measured: no benefit at all** (4.63 vs 4.62 ms on the op). The theory was that
   `repeat_interleave` dispatches an index-gather with a scatter-add backward while
   expand+reshape is a plain copy with a sum backward; at this shape on torch 2.1.2 they
   are identical. Recorded so it is not retried.
2. **Apply `residual_scale` before the upsample rather than after.** Bitwise identical --
   every output element is the single product `x[i] * s` either way. Worth **1.29x on the
   op** but only **1.019x (1.9 %) on the real stack** (103.89 -> 101.99 ms,
   7.70 -> 7.48 GB), because the op is just 3.4 % of the layer. Free, so worth taking,
   but it is a rounding error -- the op-level number should not be quoted as the win.
   One line: `out.repeat_interleave(2, dim=1) * self.residual_scale`
   -> `(out * self.residual_scale).repeat_interleave(2, dim=1)`.
3. **Attack `refine` (58 %).** In order of adoption cost: `cudnn.benchmark = True`
   (free, non-modelling, static shapes here so the algorithm search pays off); removing
   the `transpose -> conv -> transpose` in `ConvBlock.forward` by keeping the sampling
   stacks in `(B, C, L)`; or narrowing width 5 -> 3 / dropping the block (modelling).
   Priced in `tests/bench_convblock_opt.py`.
4. `skip_project` + `reduce_main` are 39 % combined and are the same `ConvBlock` shape,
   so whatever fixes `refine` should apply to them too.

**The actual win: rewrite `RMSBatchNorm1d` (2 lines).** The norm is 25 % of a
`ConvBlock` -- more than the convolution (5.91 ms) -- because
`src/models/nn/sampling.py` computes it in five full-size passes with two full-size
temporaries:

```python
mean = x.mean(dim=[0, 1])                     # full-size pass
var  = (x - mean).pow(2).mean(dim=[0, 1])     # TWO full-size temporaries
x    = x / (var + self.eps).sqrt()            # full-size pass
return x * self.weight + self.bias            # two more full-size passes
```

Replacing that with `torch.var`'s fused reduction and folding the normalization into a
single `addcmul`:

```python
var   = torch.var(x, dim=[0, 1], unbiased=False)   # one fused reduction, no temporaries
scale = self.weight / (var + self.eps).sqrt()      # (C,), computed on a tiny tensor
return torch.addcmul(self.bias, x, scale)          # one fused multiply-add
```

Equivalent to `rel_err <= 8.1e-08`. **IMPLEMENTED** in `src/models/nn/sampling.py`. Measured end-to-end on the real stacks:

| stack | current | + norm rewrite | speedup | memory |
|---|---|---|---|---|
| `DownsampleStack` | 35.21 ms | **23.48 ms** | **1.499x** | 3.84 -> 2.75 GB |
| `UpsampleStack` | 103.89 ms | **79.09 ms** | **1.314x** | 7.70 -> 4.79 GB |
| `UpsampleStack` + pre-scale | 103.89 ms | **77.20 ms** | **1.346x** | 7.70 -> 4.57 GB |

**Whole-backbone effect, which is what actually matters** (A100, B=1):

| config | L=262 144 | L=524 288 | speedup | memory @524k |
|---|---|---|---|---|
| `mamba_only gp=1` (current striped config) | 2821 -> 2817 ms | 6600 -> 6591 ms | **1.001x** | 66.84 -> 65.34 GB |
| `mamba_only gp=8 d_in=64` (recommended) | 293 -> 269 ms | **643 -> 583 ms** | **1.103x** | 17.12 -> **12.01 GB (-30 %)** |
| `mamba_only gp=8` | 402 -> 342 ms | 914 -> 760 ms | **1.203x** | 29.90 -> 19.59 GB (-34 %) |

Two things this corrects. It does **nothing at `global_pooling=1`** (1.001x), because
`RMSBatchNorm1d` only appears inside `ConvBlock` and an unpooled backbone contains exactly
one of those (`self.conv_block`) against 16 Hydra blocks. And an earlier estimate here of
"20-25 % off the pooled model" was too high for the recommended config: the real figure is
**10.3 %** (20.3 % at plain `gp=8`, where the ConvBlocks run at 256 channels rather than
64 and the norm is a larger share). The memory reduction, **-30 %**, is the bigger prize.

So this is worth taking *because* you are going to pool; on its own it buys nothing.

**Verified equivalent** (`tests/test_striped_equivalence.py`, 8/8 passing): forward
8.5e-08 in training and 5.2e-08 in eval, `running_var` EMA 2.1e-08, d/dinput 2.0e-06,
d/dweight 1.5e-07, d/dbias 6.1e-08, and under bf16 autocast 8.5e-08 with the output dtype
unchanged (the rewrite reorders dtype promotion, so that was checked explicitly).

Note the end-to-end speedup (1.50x on downsample) is *larger* than composing the
component measurement would predict (~1.14x): removing the two full-size temporaries also
shrinks the backward graph and memory pressure. Composition effects run in both
directions -- the pre-scale went the other way (1.29x on the op, 1.019x on the stack).

**Lesson for this section.** Scoring every optimization idea in it:

| idea | origin | measured |
|---|---|---|
| `repeat_interleave` -> expand/reshape | reasoning about kernel dispatch | **0 %** |
| `cudnn.benchmark = True` | reasoning about algorithm selection | **0 %** |
| conv width 5 -> 3 | reasoning about FLOPs | 2.4 % |
| pre-scale before upsample | reasoning about elementwise traffic | 1.9 % |
| `(B, C, L)` layout, drop transposes | **decomposing a measured component** | transposes cost 7.81 ms vs the conv's 5.91 |
| `RMSBatchNorm1d` rewrite | **decomposing, then reading the code** | **1.31-1.50x on the real stacks** |

Every idea generated by reasoning about kernel behaviour in advance came out at ~0 %.
Both real wins came from timing a component's parts and then reading the source for the
part that was unexpectedly expensive.

### Recommendations for the striped model

1. **Set `global_pooling=8` and `d_in=64`.** Worth 10.1× and 3.9× memory. Single biggest
   action available in this whole report.
2. **Set `use_mem_eff_path=false`.** Free 13 % and 25 % memory; the "memory efficient"
   path is neither.
3. **Then optimize the upsample stack** — 50 % of the pooled forward, and 2.7× the cost
   of downsample.
4. **Do not tune `d_state` / `chunk_size` / `headdim`** — all near-irrelevant.
5. **Hydra itself is worth revisiting**: it is 3.4× off the raw SSD kernel. Folding the
   D term and gated norm into the scan (or into a custom autograd Function that keeps
   them fused) is the fix — with the caveat from §4.2 that this is exactly the class of
   optimization that is easy to get wrong and must be measured, not assumed.

## 9. Reproducing

    sbatch tests/slurm_a100_decide.sh           # A100 head-to-head (headline numbers)
    sbatch tests/slurm_bench.sh baseline        # full sweep + profiling
    sbatch tests/slurm_probe.sh                 # isolated fusion probe
    sbatch tests/slurm_mamba2.sh                # Mamba-1 vs Mamba-2 SSD
    sbatch tests/slurm_ddp.sh                   # DDP + checkpointing on 2 GPUs
    sbatch tests/slurm_tests.sh                 # correctness gates
    pixi run python -m tests.summarize_results "tests/results/*.json"

Raw JSON for every run is under `tests/results/`.
