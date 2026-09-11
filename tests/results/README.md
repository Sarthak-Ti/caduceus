# Benchmark results

Raw JSON from every run behind `claude_summaries/mamba_optimization_report.md`.
Filenames are `<run>_<gpu>_<what>.json`; each file records the GPU, package versions and
CUDA-extension availability in its `meta` block, so results from different hardware are
never silently mixed.

| prefix | what it is |
|---|---|
| `a100d_*` | **A100 80GB PCIe — the headline numbers.** Final head-to-head with the corrected harness. |
| `a100_*` | earlier A100 run; baseline forward is inflated by the allocator-cache bias (see below) |
| `recheck_*` | H200 variant head-to-head, corrected harness |
| `probe_*_scanfusion` | isolated 2x512ch vs 1x1024ch scan/conv/GEMM probe |
| `m2_*` | Mamba-1 vs Mamba-2 SSD at matched d_model |
| `attrib_*`, `fi_*` | L40S attribution and first `fused_inner` run, **both pre-fix** |

## Caveat on the pre-fix runs

`a100_*`, `attrib_*` and `fi_*` were taken before `tests/bench_utils.py` stopped calling
`empty_cache()` immediately before the timed loop. That forced the first timed iteration
to re-`cudaMalloc` the whole working set, penalising whichever variant used the most
memory — i.e. the baseline. Their *relative* comparisons between similar-memory variants
hold; comparisons against a much lower-memory variant (`ckpt`, `fused_inner`) are biased
in that variant's favour. The report quotes only corrected numbers.

Regenerate with the `tests/slurm_*.sh` scripts; collate with

    pixi run python -m tests.summarize_results "tests/results/*.json"
