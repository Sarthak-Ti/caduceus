# CD4 and NK Cis Mamba CAGE fine-tuning

This workflow extracts the **observed CAGE labels** from the existing Enformer
label store and fine-tunes the September 2026 CD4 and NK accessibility models.
These are expression training data, separate from the TenK10K caQTL benchmark.

## Locations

- Source labels: `/data1/lesliec/sarthak/data/enformer/data/labels.zarr`
- Extracted labels: `/data1/lesliec/sarthak/data/enformer/data/cd4_nk_CAGE/`
- Scripts: `/data1/lesliec/sarthak/caduceus/slurm_scripts/`
- Run configuration, logs and outputs:
  `/data1/lesliec/sarthak/caduceus/outputs/cage_finetuning/20261003_cd4_nk/`

The exact source paths and checkpoint SHA256 hashes are in the run's
`experiment.json`; Slurm commands and job IDs are in `submission.json`.
`provenance.json` records the repository commit and hashes of the training code.

## Extraction

| Cell | Source track index | Source identifier | Output file |
|---|---:|---|---|
| CD4 | 4759 | CNhs11955 | `CD4_CAGE.zarr` |
| NK | 4764 | CNhs11957 | `NK_CAGE.zarr` |

`extract_cd4_nk_cage.py` checks the track identifiers/descriptions against
`targets_enformer.txt`. One CPU job reads both tracks together in blocks aligned
to the source's 32-example chunks, using four workers. The source is read-only.

Each output contains one float32 track at output index **0**, with 896 bins of
128 bp per example. No clipping, scaling, averaging or precision conversion is
applied. The original sequence order and splits are retained: 34,021 train,
2,213 validation, and 1,937 test examples. The BED's `valid` split corresponds to
the label store's `val` array. A copy and SHA256 of the existing
`sequences_enformer.bed` are included.

Output chunks are `(32, 896, 1)`, uncompressed, rather than the source's
`(32, 896, 5313)`. Both extracted stores total about 274 MB of array values.

Every written block is reread and checked against a SHA256 digest of the selected
source values. Dimensions, finite/nonnegative values, and nonempty signal in each
split are checked. Partial results remain in `cd4_nk_CAGE.partial`; the whole
directory is renamed to `cd4_nk_CAGE` only after all checks pass. `SUCCESS` and
`extraction_manifest.json` record completion. Existing outputs are never
overwritten. After a failed extraction, inspect the partial directory before
choosing a fresh output location or explicitly removing it for a rerun.

## Fine-tuning

The starting recipe is `slurm_scripts/finetune_joint_k562.sh` with
`experiment=hg38/joint_finetune`. There is one separate training job per cell.

- CD4 initialization:
  `outputs/2026-09-23/11-42-40-305251/checkpoints/11-val_loss=0.09242.ckpt`.
- NK initialization:
  `outputs/2026-09-24/09-55-35-988419/checkpoints/11-val_loss=0.24462.ckpt`.
- Inputs remain the corresponding `CD4T_DNase.npz` and `NK_DNase.npz` under
  `data/DK_zarr/zarr_arrays/cell_type_arrays/`, plus the existing hg38 sequence.
- Load the pretrained backbone and `jointcnn` encoder (256 dimensions, 16 layers,
  bidirectional weight sharing). Both remain trainable. Initialize a new
  one-track Enformer-style CAGE decoder; discard the accessibility decoder.
- Input length 524,288 bp; predict the central 114,688 bp as 896 x 128-bp bins.
  Sequence input has six channels, accessibility has two, including the existing
  mask channels. Sequence/accessibility mask rates are both zero; sequence
  shifting and reverse-complement augmentation remain off, matching the old
  CAGE fine-tuning recipe.
- Poisson loss, AdamW, learning rate `1e-4`, weight decay `0.1`, seed `2222`.
  Batch size 1 per GPU, two A100 GPUs, gradient accumulation 4 (effective batch 8).
  Validation batch size 2 per GPU, full training and validation splits.
- Keep the standard 1,000-epoch cap and the previous 7-day Slurm allocation.
  Save every completed epoch and `last.ckpt`; select the lowest validation loss
  for later evaluation. A time-limited run may need resuming if more training is
  wanted. Fresh launches refuse existing checkpoint directories; explicit
  `--resume` continues the saved training state in the same output directory.
- Test labels are extracted for future evaluation; they are not used during
  training or model selection. W&B logging is offline under each run directory.

Each GPU job depends on successful extraction (`afterok`). It first executes
one training batch/optimizer update and one validation batch with the actual
two-GPU configuration. Only a successful smoke test creates `SMOKE_PASSED`.
The full run then starts again from the original accessibility checkpoint;
the smoke-test update is not reused.

`train.pretrained_model_path` supplies starting weights. The existing `train.py`
calls `fsspec_exists(train.ckpt)` even in its log statement, so a null `train.ckpt`
is unsafe in this version. The launcher uses `checkpoints/last.ckpt` in a new,
empty per-cell run directory and refuses a fresh launch when a checkpoint
directory already exists. Thus initial fine-tuning starts a new optimizer and
CAGE head. The explicit resume mode instead restores the existing CAGE run.

## Reproducing

Use the existing Python environment:
`/data1/lesliec/sarthak/caduceus/.pixi/envs/default/bin/python`.

1. Copy `experiment.json` to a new run directory, choose unused data/run output
   locations, and keep the source checkpoint hashes consistent.
2. Submit `extract_cd4_nk_cage.sh /absolute/path/to/experiment.json`, with an explicit
   `--output` log path and `--chdir` run directory.
3. Submit `finetune_joint_cd4_nk_cage.sh /absolute/path/to/experiment.json CD4` and the same
   command with `NK`, each with `--dependency=afterok:<extraction_job_id>` and
   `--kill-on-invalid-dep=yes`.

Configuration checks can be run with
`finetune_joint_cd4_nk_cage.py /absolute/path/to/experiment.json CD4 config` (or `NK`).
They compose and print the configuration without starting training. The
extraction fixture check is `test_extract_cd4_nk_cage.py`.

## Startup repair, October 3, 2026

The initial GPU jobs loaded their checkpoints but stalled in the DataLoader
before the first training batch: the run-directory TMPDIR exceeded the AF_UNIX
socket pathname limit. The launcher now uses node-local `/tmp/cage_<jobid>`,
checks socket creation before starting, and limits the smoke stage to 20 minutes
so a worker failure cannot hold GPUs indefinitely. Core training settings are
unchanged. The stalled jobs were cancelled and replaced; the completed,
verified CAGE extraction is reused.

The full-run sanity-validation count is now zero, matching the reference run
and the user's preference. The existing `on_validation_epoch_end` hook expects
`epoch_start_time` to have been set by `on_train_epoch_start`, so pre-training
validation failed before training. Regular epoch validation and the separate
train-then-validate GPU smoke test remain enabled.

## Resuming after the storage interruption, October 5, 2026

Jobs 16656968 (CD4) and 16656871 (NK) failed after the shared filesystem returned
`OSError: [Errno 28] No space left on device`. Extraction is complete and does not
need to run again. Both `train/checkpoints/last.ckpt` files load successfully and
contain finite model weights, the trained CAGE decoder, optimizer, scheduler,
and training-loop state. Their saved epoch is 1 (zero-based), global step 8,506:
two epochs completed. The unsaved portion of epoch 2 must be repeated.

Resume either job using the same Slurm launcher with a third argument:

```bash
sbatch --output=/absolute/run/logs/resume_CD4_%j.out \
  /data1/lesliec/sarthak/caduceus/slurm_scripts/finetune_joint_cd4_nk_cage.sh \
  /absolute/run/experiment.json CD4 --resume
```

Use `NK` and an NK log name for the other cell. The actual run directory is
listed under Locations above. No extraction dependency is needed when its
validated outputs already exist. The Python launcher's `config --resume` phase
checks the checkpoint and prints the resume configuration without training.

Resume sets `train.pretrained_model_path=null`,
`train.pretrained_model_state_hook._name_=null`, and `train.ckpt` to the absolute
path of the existing CAGE `last.ckpt`. Disabling the pretraining hook is necessary
to restore the trained CAGE decoder as well as the backbone and encoder.
Lightning's `Trainer.fit(ckpt_path=...)` restores optimizer, scheduler and loop
state too. Model, data, learning rate, batch sizes, epoch cap, validation and
checkpoint settings are unchanged; the epoch cap is the total across attempts.

Resume skips the initial smoke run and keeps sanity-validation batches at zero.
It checks checkpoint contents and cell-specific data paths, and requires the
existing `SMOKE_PASSED` marker. On rank zero it archives the previous Hydra config,
exact starting checkpoint, checkpoint SHA256 and overrides in
`<cell>/resume_<SlurmJobID>/` before training. Checkpoints and subsequent outputs
continue in `<cell>/train/`; prior epoch checkpoints are retained.

Submitted resume jobs: **17232989 (CD4)** and **17232990 (NK)**. Current states
are recorded with a timestamp in the run directory's `runtime_status.json`;
the commands and checkpoint hashes are in `submission.json` and
`resume_20261005_validation.json`.


## Held-out CAGE evaluation, October 12, 2026

Both training resumes completed with exit code 0 after epoch 16 (17 total epochs).
The runtime guard stopped each run before another epoch would exceed the seven-day
budget. Test evaluation fixes the lowest-validation-Poisson-loss checkpoint before
reading any test scores: CD4 epoch 16 (-0.09986649), NK epoch 8 (-0.27512944).
The checkpoint paths, SHA256 hashes, adjacent model-config hashes, Slurm commands
and job IDs are recorded in `test_evaluation_20261012.json` in the run directory.

`slurm_scripts/evaluate_cd4_nk_cage.py` reuses the strict `Evals` loader and
`eval_joint_expression.evaluate_all`. `evaluate_cd4_nk_cage.sh` runs one A100 job
per cell. It evaluates all 1,937 original test rows with the same genome and
cell-matched measured DNase input, no masking/shifts/RC augmentation, and forward
orientation. Inputs are 524,288 bp; CAGE outputs are the central 114,688 bp,
896 bins of 128 bp, one unstranded channel. The existing bf16 inference path is
used and results are stored in float32. The decoder applies Softplus once;
`Evals.skip_softplus=True` prevents an extra Softplus in the evaluation helper.
Data are preloaded into host memory to avoid repeated whole-chromosome NPZ reads.
There is no pooling or target filtering. Saved targets must match every value in
the extracted Zarr test array exactly; BED order, cell type, and track index are
also checked.

Results are written under `<run>/<cell>/test_evaluation_20261012/`:

- `expression_outputs.npy` and `expression_targets.npy`: shape `(1937, 1, 896, 1)`,
  axes `(test_region, cell_type, bin, CAGE_track)`.
- `expression_pearson.npy` and `expression_spearman.npy`: shape `(1937, 1, 1)`,
  raw-signal correlation across the 896 bins **within each region**.
- `regions.tsv`: original test-row index, cell type, BED/input/output coordinates
  (0-based, half-open; input coordinates may include padding), predicted/observed
  region sums, within-region correlations and per-region Poisson loss.
- `metrics.json` and `metrics.tsv`: named aggregate scores. Within-region means
  give every defined region correlation equal weight. Across-region correlations
  compare the 1,937 **summed region signals**, using Pearson and Spearman on raw
  totals and `log1p(total)`. Separately labelled pooled-bin correlations flatten
  all regions/bins. Log1p profile Pearson and training-compatible Poisson loss
  are also reported; the Poisson loss omits the target-only factorial term and
  can be negative. Constant-profile correlations are undefined (NA/null), with
  counts reported; these regions remain in saved arrays and other metrics.
- `provenance.json`: selected checkpoint/config/code hashes, evaluation settings,
  coordinate/array definitions, target identity check and output checksums.
  `SUCCESS` is written only after all predictions, metrics and checks complete.

A synthetic check verifies that within-region shape and across-region amplitude
can have opposite correlations, and that Poisson loss matches the training loss.
A completed evaluation is reused only for the same checkpoint; incomplete output
directories are not silently overwritten.

To reproduce, create a manifest with fixed validation-selected checkpoint/config
hashes and unused output paths, then submit the existing flat Slurm script:

```bash
sbatch --output=/absolute/run/logs/test_CD4_%j.out \
  /data1/lesliec/sarthak/caduceus/slurm_scripts/evaluate_cd4_nk_cage.sh \
  /absolute/run/test_evaluation_20261012.json CD4
```

Repeat with NK. This evaluates CAGE given sequence plus observed accessibility;
it is separate from the TenK10K caQTL benchmark and from allele-effect scoring.
