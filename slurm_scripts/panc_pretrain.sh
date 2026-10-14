#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --job-name=panc_pretrain
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# PANC accessibility pretraining, wide256/gp128 striped Hydra with the cosine recipe
# (outputs/cismamba_tuning_20260928/training/cosine/config.json), modern env + torch.compile.
# Usage: sbatch --job-name=panc_<arm> slurm_scripts/panc_pretrain.sh <arm> [extra hydra overrides]
# Arms: de_only | multi_base | multi_ctt | multi_chromvar | multi_sea
#       seqonly_sea | heldout_pp2_sea | pp2_only (second round, 2026-10-12)
#
# All arms share one budget: 41,699 train samples per epoch (borzoi fold3-4 train size),
# 5,213 updates per epoch at accumulation 8, cosine over 30 epochs = 156,390 updates.
# Multi-cell arms draw a fresh random 1/13 of the (sequence, cell type) pairs each epoch.
# Validation is DE only for every arm: val is unshuffled and cell-type-major, and store row 0 is DE,
# so the first 6,910 val batches are exactly DE's val windows. pp2_only validates on PP2 (its only cell type).
# seqonly_sea masks 100% of the accessibility input (acc_mlm=1); the SequenceOnlyProtocol callback
# (tests/cismamba_sequence_only_protocol.py) samples an independent 25% / 500-bp loss mask, so it needs
# slurm_scripts/panc_train_entry.py to register the callback before train.main().
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache
export TRITON_CACHE_DIR=$T/triton_cache/modern
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4

arm=${1:?Specify arm: de_only | multi_base | multi_ctt | multi_chromvar | multi_sea | seqonly_sea | heldout_pp2_sea | pp2_only}
shift

MULTIOME=/data1/lesliec/sarthak/data/DE_danwei/multiome
DATA=$MULTIOME/panc_atac.zarr/panc
IDX13=$MULTIOME/idx_lists/panc_train13.json # store rows, excludes early_Delta (7), late_Delta (11), liver (14)
IDX12=$MULTIOME/idx_lists/panc_train12_noPP2.json # IDX13 without PP2 (store row 3), held out for generalization
TRANS=$MULTIOME/motifs/final/celltype_inputs.npz
N_TRAIN=41699 # train windows in sequences_borzoi_fold3-4.bed
N_VAL_DE=6910 # val windows in sequences_borzoi_fold3-4.bed
N_STEPS=156390 # 30 * ceil(N_TRAIN / 8)

limits=(trainer.limit_train_batches=$N_TRAIN trainer.limit_val_batches=$N_VAL_DE)
multi=(+dataset.data_idxs=$IDX13 "${limits[@]}")
sea=(+dataset.trans_info=$TRANS +dataset.trans_info_key=sea_scaled +encoder.trans_emb=8)
entry=(-m train)
case "$arm" in
  de_only) arm_args=(+dataset.data_idxs=0) ;;
  multi_base) arm_args=("${multi[@]}") ;;
  multi_ctt) arm_args=("${multi[@]}" +encoder.ctt=true +dataset.return_celltype_idx_og=true) ;;
  multi_chromvar) arm_args=("${multi[@]}" +dataset.trans_info=$TRANS +dataset.trans_info_key=chromvar_dev +encoder.trans_emb=8) ;;
  multi_sea) arm_args=("${multi[@]}" "${sea[@]}") ;;
  seqonly_sea)
    arm_args=("${multi[@]}" "${sea[@]}" dataset.acc_mlm=1
      +callbacks.sequence_only_protocol.loss_mask_rate=0.25 +callbacks.sequence_only_protocol.span=500)
    entry=(slurm_scripts/panc_train_entry.py) ;;
  heldout_pp2_sea) arm_args=(+dataset.data_idxs=$IDX12 "${limits[@]}" "${sea[@]}") ;;
  pp2_only) arm_args=(+dataset.data_idxs=3) ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac

run_dir="${PANC_RUN_DIR:-$PWD/outputs/panc_pretrain_20261010/$arm}"
# Resume from whichever is newer, the epoch checkpoint or the 256-step recovery snapshot.
resume="$run_dir/checkpoints/last.ckpt"
if [[ -f "$run_dir/recovery/last.ckpt" && "$run_dir/recovery/last.ckpt" -nt "$resume" ]]; then
  resume="$run_dir/recovery/last.ckpt"
fi

exec pixi run --manifest-path $T/pixi.toml python -u "${entry[@]}" \
  experiment=hg38/joint_pretrain model=striped_hydra wandb.group=panc_pretrain wandb.name=panc_$arm \
  "hydra.run.dir=$run_dir" train.test=false \
  +train.remove_test_loader_in_eval=true "train.ckpt=$resume" train.compile=true \
  trainer.devices=1 trainer.precision=bf16 dataset.batch_size=1 \
  dataset.batch_size_eval=1 trainer.accumulate_grad_batches=8 \
  dataset.num_workers=7 loader.num_workers=7 \
  model.config.d_model=256 model.config.n_blocks=8 \
  model.config.mode=striped model.config.expand=2 model.config.use_mem_eff_path=false \
  model.config.d_state=64 model.config.headdim=64 \
  +model.config.global_pooling=128 +model.config.d_in=256 ++model.config.use_gating=true \
  +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
  optimizer.lr=1e-3 train.task2=reg train.custom_metric=poisson_loss_mask \
  ++scheduler._name_=cosine_warmup ++scheduler.num_warmup_steps=1000 \
  ++scheduler.num_training_steps=$N_STEPS ++train.interval=step \
  dataset.acc_type=continuous dataset.acc_mlm=0.25 dataset.mlm=0 \
  +dataset.mask_only=true dataset.load_in=false \
  dataset.data_path=$DATA \
  +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi_fold3-4.bed \
  +callbacks.model_checkpoint_every_n_steps.monitor=null \
  +callbacks.model_checkpoint_every_n_steps.save_top_k=1 \
  +callbacks.model_checkpoint_every_n_steps.save_last=true \
  +callbacks.model_checkpoint_every_n_steps.every_n_train_steps=256 \
  +callbacks.model_checkpoint_every_n_steps.every_n_epochs=0 \
  +callbacks.model_checkpoint_every_n_steps.dirpath=recovery/ \
  '+callbacks.model_checkpoint_every_n_steps.filename="step-{step}"' \
  +callbacks.model_checkpoint_every_n_steps.auto_insert_metric_name=false \
  trainer.max_epochs=30 ++trainer.enable_progress_bar=false \
  "${arm_args[@]}" "$@"
