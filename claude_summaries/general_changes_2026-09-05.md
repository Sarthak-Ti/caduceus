# General code assessment — `evals/` and `src/`
_2026-09-05 — read-only review, nothing implemented_

Two audits, same method: measure duplication, find hot-path waste, verify every claim before
writing it down. Findings that did not survive verification were dropped rather than hedged.

**Read "Where this landed" first** (near the end) — it records which of these findings survived
review with the author and which were dismissed. Two of the biggest-sounding items in Part 2
(§2.1, §2.2) were tested and found not to be bottlenecks; §2.0 records an earlier claim that was
simply wrong. Items that have since been fixed have had their sections removed. The sections are kept intact rather than deleted so the measurements and the
corrected reasoning stay on record.

---

# Part 1 — `evals/`

## 1.1 The inlined `Evals` copies run fp32; the shared one runs bf16

`evals_utils_joint.Evals.__call__` wraps its forward in
`torch.autocast(device_type=..., dtype=torch.bfloat16)`. The ten `e2g_*.py` scripts that inline
their own `Evals` do not — verified by reading `e2g_tss.py`'s `__call__` directly, which has
`no_grad` but no `autocast` (`grep -c autocast` is 0 for all ten).

So today:

| script | source of `Evals` | precision |
|---|---|---|
| `e2g_striped.py`, `e2g_striped_dn.py` | imports `evals_utils_joint` | **bf16** |
| `e2g_enformer{,_dn}.py`, `e2g_borzoi{,_dn}.py`, `e2g_tss{,_profile,_profile_dn}.py`, `e2g_3prime.py` | inlined copy | **fp32** |

Two consequences:

1. **Speed.** On 524kb sequences I would expect roughly 2× wall-clock. *I inferred this from the
   precision change and did not benchmark it* — worth a timing run before relying on the number.
2. **Comparability.** Two script families compute the same quantity at different precision. If you
   ever put a striped result next to an enformer/borzoi result, that difference is baked in.

Fix: either add the `autocast` line to the inlined copies (one line each, eight files) or delete the
copies in favour of the shared class (§1.4), which gets it for free.

### 1.1a Measured: is bf16 safe for E2G? (jobs 11680394 / 11683144 / 11683218)

Tested rather than assumed, because E2G scores a *difference* between two nearly identical
predictions — the case most sensitive to a reduced mantissa. `evals/test_bf16_e2g.py` runs the same
elements twice through the same `Evals` object, once plain and once inside
`torch.autocast(bfloat16)`, and compares `delta = after - before`.

**Control first (fp32 vs fp32, 25 elements): bit-identical.** `max|fp32 - fp32| = 0`, Pearson and
Spearman both `1.000000`. So the model is fully deterministic at 524 kb — there is no kernel jitter,
and every bf16 difference below is genuinely precision. (Useful independently: repeated E2G runs
reproduce exactly.)

**On the 30 strongest true positives — the detections E2G is actually scored on:**

```
std(delta_fp32)                : 0.4563    <- the signal
max |delta_fp32 - delta_bf16|  : 0.0315    <- the noise
max noise / signal spread      : 6.9%
pearson / spearman             : 0.99954 / 0.98697
sign agreement                 : 100.00%
```

bf16 is safe here. **Verdict: adopt it** — faster, matches `trainer.precision=bf16` in training, and
ends the split where `e2g_striped*` runs bf16 while the other eight run fp32.

**A first pass over the first 50 in-context elements looked alarming** (80% noise/signal, 38% sign
agreement) and was wrong — a sampling artifact. The benchmark is 465 positives vs 9,841 negatives,
so an unselected sample is almost all nulls whose true effect is ~0; their sign is noise by
construction. Stratifying that same run by model response makes it obvious:

| subset | med \|delta\| | med noise | noise/sig | sign agree |
|---|---|---|---|---|
| top 5 | 0.0474 | 0.0083 | 17.5% | **100%** |
| top 10 | 0.0392 | 0.0052 | 13.2% | 90% |
| top 25 | 0.0200 | 0.0100 | 49.8% | 60% |
| all 50 | 0.0064 | 0.0054 | 84.4% | 38% |

**The one real caveat — do not mix precisions across a comparison.** Of the 419 in-context
positives, **28% have |delta| below the median bf16 noise (0.0054) and 78% below the max (0.0315)**.
The top-30 detections clear the noise by 3.6x, but they are 30 of 419. AUC/AUPRC is set by
positive-vs-negative ordering, and most positives sit in the zone bf16 perturbs — so a bf16 rerun
can move the metric slightly even though every strong call is unchanged. Existing fp32 outputs
(`k562_tss_sc_rna_poisson_ep35.npy` and siblings) should be **regenerated** after switching, not
compared against. Reading a precision shift as a model difference is exactly the trap the current
fp32/bf16 split already sets up between the two script families.

## 1.2 TSS-centered runs re-read the same window ~5×

> **Dismissed on review.** Author tested; with `load_in=True` this is CPU tensor construction, not
> I/O, and it is small against the GPU forward. "Dataset reads" was the wrong framing.

The CRISPR benchmark is **10,412 rows over 2,146 unique genes — 4.9 elements per gene**
(counted from `EPCrisprBenchmark_ensemble_data_GRCh38.tsv`).

For any `--center_tss` run, and for all of `e2g_tss*.py`, the sequence window is a function of the
gene alone. So `evals.dataset[idx]` returns byte-identical tensors ~4.9 times per gene, and re-reads
them from disk every time. Rows are gene-adjacent in the file, so an LRU cache of size 1 keyed on
`idx` removes ~80% of dataset reads.

No benefit in element-midpoint mode, where every row is a genuinely distinct window.

## 1.4 ~3,000 duplicated lines, and exactly where they are

Two overlapping clusters. Line counts are `wc -l`; identity is measured with `diff` ignoring
nothing.

### Cluster A — `class Evals` inlined 18 times (3,987 lines across the 14 script copies)

Only 4 of the 18 definitions are utils modules. The rest are scripts that pasted a copy. Overlap of
each inlined class body against `evals_utils_joint.Evals` (281 lines):

| file | class lines | shared with `evals_utils_joint` |
|---|---|---|
| `eqtl_onemodel.py` | 123 | 97 (79%) |
| `eval_joint_expression_old.py` | 123 | 97 (79%) |
| `e2g_enformer.py` | 115 | 86 (75%) |
| `e2g_enformer_dn.py` | 115 | 86 (75%) |
| `e2g_tss.py` | 120 | 85 (71%) |
| `e2g_borzoi.py` | 107 | 72 (67%) |
| `e2g_borzoi_dn.py` | 107 | 72 (67%) |
| `e2g_tss_profile.py` | 131 | 85 (65%) |
| `e2g_3prime.py` | 130 | 68 (52%) |

Also inlining a copy: `eqtl_onemodel_ctt_legacy.py`, `eval_joint_expression_oldctt.py`,
`dsqtl_onemodel_ctt_legacy.py`, `get_embedding.py`, `eval_fullenformer.py`.

The shared ~70 lines are pure boilerplate: read `.hydra/config.yaml` beside the ckpt, `torch.load`,
`consume_prefix_in_state_dict_if_present`, drop `torchmetrics.*` keys, split the state dict into
encoder/decoder/backbone by key prefix, `.to(device).eval()`.

The only real difference is how the decoder is built — and `evals_utils_joint` already solves that
generically with `decoders_module._instantiate(...)`. The copies hardcode a class
(`TSSDecoder` at `e2g_tss.py:107`, `EnformerDecoder` at `evals_utils_enformer.py:118`), which is
exactly what later forced `e2g_tss_profile.py:113` to reinvent registry lookup because the
hardcoded class TypeErrors on the other decoder's checkpoints. These copies predate the registry
approach; they are not load-bearing.

### Cluster B — the E2G driver loop copied 10 times (2,821 lines across `e2g_*.py`)

Pairwise identical lines:

| pair | identical | of |
|---|---|---|
| `e2g_enformer_dn.py` vs `e2g_borzoi_dn.py` | **248** | 324 / 338 |
| `e2g_tss.py` vs `e2g_tss_profile.py` | **246** | 259 / 281 |
| `e2g_borzoi.py` vs `e2g_borzoi_dn.py` | 228 | 266 / 338 |
| `e2g_enformer.py` vs `e2g_enformer_dn.py` | 223 | 243 / 324 |
| `e2g_striped.py` vs `e2g_striped_dn.py` | 122 | 167 / 259 |

`e2g_enformer_dn.py` and `e2g_borzoi_dn.py` are, discounting whitespace and comment rewording, the
same file with a different strand count.

All ten also re-open the same three files with the same ~12 lines of mapping code:

```
CollapsedGeneBounds.hg38.TSS500bp.bed              → all 10
EPCrisprBenchmark_ensemble_data_GRCh38.tsv         → all 10
k562_bulk_rna_info.json                            → 7
```

and the four `_dn` scripts additionally share, near-verbatim: the perturbation block (acc
scale-down + dinuc shuffle + `args.seed + i` keying + `except ValueError: pass` for low-complexity
elements), the chunked forward pass, the averaging/`shuffle_std` logic, the five-file save block,
and ~20 identical argparse declarations.

**Suggested shape**, if you decide it's worth it: fold Cluster A into `evals_utils_joint.Evals`
(add a `dataset_class=` arg and a hook for the `(profile, counts)` tuple collapse that only
`e2g_tss_profile.py:152-159` needs), then add `evals/utils/e2g_common.py` with
`load_crispr_benchmark()`, `add_e2g_args(parser)`, `perturb(...)` and `save_e2g_outputs(...)`.
Each script keeps only its window/coordinate mapping and output reshape. Roughly 2,900 → 900 lines.

**Whether to bother** depends on whether you're still adding model variants to this family. If E2G
work is winding down, the duplication is inert and §1.1/§1.2 are the only items worth doing.

---

# Part 2 — `src/` (24,376 lines)

## 2.0 Correction: the npz format is fine

An earlier draft of this document claimed the `.npz` data format cost ~1 s per sample because
`self.data[chrom][start:end]` decompresses a whole chromosome. **That was wrong on the part that
mattered.** Verified with `zipfile`: every member of `K562_DNase.npz` has `compress_type=0`
(STORED, uncompressed) — `chr8` is 1.16 GB raw and 1.16 GB stored. The ~983 ms I measured was a
1.16 GB sequential read at ~1.2 GB/s, not decompression. And with `load_in=True` — the normal mode
here — the array is resident in RAM and the slice is free.

One thing does survive from that section, and it matters *because* the data is held in memory:

**The accessibility npz is stored as `float64`.** `d['chr8'].dtype` is `float64`, so chr8 alone is
1.16 GB and the file is 24 GB. Accessibility counts do not need 11 significant digits; the existing
zarr stores in the same tree already use `float16`. At `float16` the same data is ~6 GB. That is a
**4× cut in resident RAM**, which directly sets how many cell types you can hold at once. `float32`
if you want headroom, still 2×.

## 2.1 45% of the DataLoader payload is redundant — but NOT the bottleneck

> **Dismissed on review.** The measurement below is correct, but author testing confirmed the
> transfer hides behind the forward pass, and the `mlm=0` behaviour is intended design (see
> "Where this landed"). Kept for the numbers.

**This is the finding worth acting on.** Measured by building `GeneralDataset` from the most recent
run config (`outputs/2026-08-13/16-16-44-251024`) and inspecting one sample:

```
outputs1[0] seq          (6, 524288)  float32   12.58 MB
outputs1[1] acc          (2, 524288)  float32    4.19 MB
outputs2[0] seq_unmask   (524288, 6)  float32   12.58 MB
outputs2[1] acc_unmask   (524288, 2)  float32    4.19 MB
outputs2[2] tracks       (524288, 2)  float32    4.19 MB
                                     ─────────────────────
             TOTAL per sample through DataLoader IPC  37.7 MB
```

That config runs `batch_size: 1`, `num_workers: 7`, `length: 524288`, `mlm: 0`, `acc_mlm: 0`.

**The `_unmask` tensors carry zero information in this configuration.** Verified directly:

```
seq mask channel (row 5) nonzero count : 0
seq[:5] == seq_unmask[:, :5].T         : True
acc mask rows nonzero                  : 0
acc[0]  == acc_unmask[:, 0]            : True
```

They are bit-identical copies of the inputs, and 16.8 MB of every 37.7 MB sample is spent shipping
them from worker to trainer through shared memory.

**Cause** — `general_dataset.py:373` and the acc equivalent:

```python
if self.mlm is not None:      # mlm = 0 is not None -> True
```

`0` is not `None`, so the entire masking path executes at rate 0: it allocates the 6th mask channel,
runs `mask_seq`, and returns a full unmasked copy. The guard almost certainly wants `if self.mlm:`
(or an explicit `is not None and > 0`). One-character class of fix, ~45% of the payload.

## 2.2 One-hot is built in the worker at float32 — 24× larger than it needs to be

> **Dismissed on review** for the same reason as §2.1: real waste, not the bottleneck.

`general_dataset.py:368`:

```python
x_onehot = torch.nn.functional.one_hot(x-7, num_classes=5).float()
```

At length 524288 that is **12.58 MB per sample**. The same information as an `int8` index vector is
**0.52 MB — 24× smaller**. The one-hot is then shipped through DataLoader IPC at the larger size.

The fix is feasible: `JointCNN` (`src/tasks/encoders.py:116`) consumes the sequence as a
`Conv1d` over `d_input1=6` channels, so the worker can emit indices and the encoder can do
`F.one_hot(...).float()` on GPU immediately before the conv. Same arithmetic, moved off the CPU and
off the IPC path.

**Combined with §2.1**, a sample would move roughly 4-8 MB instead of 37.7 MB — call it **5-9×**
less dataloader traffic. Whether that shows up as wall-clock depends on whether you are dataloader-
bound; at `batch_size: 1` with 7 workers feeding 524 kb sequences, it is worth measuring before and
after rather than assuming either way.

## 2.4 Dataset classes duplicated (2,136 lines across 7 files)

Same pattern as `evals/`, similar magnitude:

| pair | identical | of |
|---|---|---|
| `DNase_dataset.py` vs `DNase_sc_dataset.py` | **215** | 234 / 237 (92%) |
| `DNase_dataset.py` vs `DNase_ctst_dataset.py` | **213** | 234 / 247 (91%) |
| `GPNMSA_dataset.py` vs `GPNMSA_dataset_noparallel.py` | **310** | 411 / 354 |
| `profile_atac_long.py` vs `profile_atac_long_old.py` | 222 | 267 / 386 |
| `general_dataset.py` vs `tss_dataset.py` | 171 | 628 / 632 |

The three `DNase_*` variants are >90% identical — they differ by a cell-type-token branch and a
single-cell branch, both of which are constructor flags, not separate classes. Three files,
~718 lines → one class of maybe 260.

`general_dataset` vs `tss_dataset` share only 171/630 and have genuinely diverged; leave them.
The `_old` / `_noparallel` files look superseded — archive rather than merge.

## 2.5 What's in good shape

- **`tasks/decoders.py` (1,067 lines, 14 classes) is well factored.** All 14 inherit a common
  `Decoder` base; a scan of every `forward()` pair found exactly one near-duplicate
  (`EnformerDecoder` ~ `GraphRegDecoder`, 84%) — normal for two related heads.
- The `decoders`/`encoders` **registry + `_instantiate` pattern is the right abstraction**. The
  `evals/` problem is scripts bypassing it, not the pattern.
- `src/` has a coherent structure (`models/{nn,sequence,baseline}`, `tasks`,
  `dataloaders/{datasets,utils}`) that `evals/` lacks entirely.

---

# Where this landed (after review 2026-09-05)

Discussed each item with the author; several did not survive. Recording the outcome so the
dismissed ones don't get re-raised later.

## Dismissed

- **`mlm=0` vs `mlm=None` is intended API design.** `0` means "mask channel present, no masking";
  `None` means "no mask channel at all". `mask_seq` returns `(length, N+1)`, so the two produce
  *different shapes*, not identical output — and `JointCNN` has `d_input1=6`, so the branch is
  load-bearing. My proposed `if self.mlm:` would have dropped a channel and broken the model. The
  redundant `seq_unmask` at rate 0 is a consequence of a deliberate choice, and per the next item it
  costs nothing.
- **DataLoader payload size is not the bottleneck.** 37.7 MB/sample is real, but with 7 workers
  prefetching at `batch_size: 1` the transfer hides behind the forward pass. Author confirmed by
  testing that this is not where time goes. Applies to both the `_unmask` copies and the float32
  one-hot.
- **Re-reading windows in the E2G loop is not the bottleneck.** Author tested. With `load_in=True`
  there is no disk I/O; the repeat cost is CPU tensor construction, which is small against a 524 kb
  GPU forward. My "−80% dataset reads" framing was wrong — it was never I/O.
- **The three `DNase_*` dataset classes** are superseded code; not worth consolidating.

## Resolved

- **bf16 eval is safe for E2G — measured, see §1.1a.** Training is `trainer.precision=bf16`
  (`slurm_scripts/finetune_joint_k562_tss_sc_decima.sh`, and all 8 recent configs), so bf16 eval
  reproduces the training forward and fp32 is the outlier. My a-priori reasoning here was that
  before/after rounding would be common-mode and cancel; the measurement shows it only partly does
  — it cancels well enough for strong detections (100% sign agreement) but not for the ~78% of
  positives that sit below the noise floor. Numbers, caveat and the sampling artifact that made an
  early pass look bad are all in §1.1a.
- **float64 accessibility storage**: real (24 GB resident vs ~6 GB at float16) but deferred — the
  zarr stores are the current constraint, and the npz can be reconverted later.

## Done

- **`general_dataset.py:494` — numpy/torch `.flip` mismatch: FIXED.** `open_data()` returns numpy
  (or zarr), never a tensor, so `.flip(dims=[0])` raised
  `'numpy.ndarray' object has no attribute 'flip'`. The conversion is placed *above* the `if flip:`
  branch, not inside it:
  ```python
  additional_data = torch.FloatTensor(np.ascontiguousarray(additional_data))
  if flip:
      additional_data = additional_data.flip(dims=[0])
  ```
  Converting only inside the branch (the first version of this fix) would have left a batch mixing
  tensors from flipped samples with ndarrays from unflipped ones — `default_collate` rejects that
  with `TypeError: expected Tensor as element 1`, so it would have broken any `rc_aug=True` run at
  `batch_size > 1` while looking fine at `batch_size: 1`.
  Verified: 16 samples with `rc_aug=True` and `additional_data=GM12878CAGE.zarr` yield a single
  `(Tensor, torch.float32)` regardless of how the coin lands, and a `batch_size=4` loader collates
  to `(4, 6144, 2)`.
- **`correlate()` slowness: COMMENTED, not changed.** `evals_utils_enformer.py` now carries an
  in-place note giving the vectorized replacement for the `j` loop (`pearsonr2` directly for
  Pearson, `pearsonr2(rankdata(..., axis=0).T, ...)` for Spearman) and flagging that it is
  untested — the nan → 0.0 handling in particular has to be redone in the vectorized form.

## Recommended — measured, ready to do

- **Add `autocast(bfloat16)` to the 8 inlined `Evals` copies.** One line each. Verified safe on the
  detections that matter (§1.1a): 100% sign agreement, Pearson 0.9995, noise 6.9% of signal spread,
  and the fp32-vs-fp32 control is bit-identical so the model is deterministic. Gains speed and, more
  importantly, ends the fp32/bf16 split between `e2g_striped*` and everything else.
  **Regenerate the existing fp32 E2G outputs afterwards rather than comparing against them** — see
  the caveat in §1.1a.

## TODO — deferred, legitimate

- **Consolidate the duplicated `Evals` classes and add `evals/utils/e2g_common.py`** (§1.4). The
  concern is real — ~3,000 duplicated lines across 14 scripts with `Evals` pasted in and 10
  `e2g_*.py` sharing a driver loop — but revalidating every affected script is expensive right now.
  Revisit if the E2G family gets extended again. The two clusters and their exact files are
  catalogued in §1.4 above. Doing the autocast change above at the same time would avoid touching
  these eight scripts twice.

## Test tooling left behind

`evals/test_bf16_e2g.py` (+ `.sh`, `_control.sh`, `_strongpos.sh`) — reruns any of the three
comparisons. `--control` measures nondeterminism, `--rows_npy` targets specific `gs_df` rows.
Reads only; no existing code was modified to run it.

---
