#!/bin/bash
# =============================================================================
# BASE COMMAND for k562 TSS finetune jobs. DO NOT submit directly.
# Source this from a submit script after setting WANDB_NAME and OVERRIDE_ARGS.
#
#   WANDB_NAME     - wandb run name for this task (required)
#   OVERRIDE_ARGS  - hydra args to append; processed last so they win over
#                    anything in this file. Use hydra syntax as-is:
#                      override existing key:  optimizer.lr=5e-5
#                      add new key:           +dataset.use_alt_tss=false
#                    Multiple args space-separated: "optimizer.lr=5e-5 +foo=bar"
# =============================================================================

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "=== Task $SLURM_ARRAY_TASK_ID | Run: $WANDB_NAME ==="
echo "  OVERRIDE_ARGS: ${OVERRIDE_ARGS:-<none>}"
echo "============================================="
pixi run srun python -m train \
  wandb.group=tss_finetune wandb.name="$WANDB_NAME" \
  experiment=hg38/joint_finetune \
  dataset.batch_size=1 \
  trainer.precision=bf16 \
  dataset.num_workers=$WORKERS loader.num_workers=$WORKERS \
  model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
  \
  model=caduceus \
  model.config.d_model=256 model.config.n_layer=16 \
  model.config.bidirectional=true \
  model._name_=dna_embedding_caduceus \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=false \
  optimizer.lr="1e-4" \
  +train.remove_test_loader_in_eval=true \
  \
  dataset._name_=TSSLoader \
  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
  dataset.load_in=false \
  dataset.shift_sequences=2000 \
  +dataset.rc_strand=true \
  +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json \
  dataset.acc_type=continuous \
  \
  +model.config.skip_embedding=true \
  trainer.devices=$NUM_GPUS \
  \
  task._name_=joint_tss \
  task.loss._name_=mse_tss \
  task.metrics=[mse_tss] \
  \
  decoder._name_=tss \
  +decoder.d_output=1 \
  +decoder.hidden_dim=128 \
  +decoder.bp_predictor=true \
  trainer.accumulate_grad_batches=16 \
  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt" \
  ${OVERRIDE_ARGS}
