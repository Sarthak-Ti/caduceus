#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eqtl_borzoi
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

# The GM12878 CAGE finetune on Borzoi data: 32 bp bins, 6144 bins over the central
# 196608 bp, d_output=2 (channel 0 = plus strand, 1 = minus). Same run as
# evaluate_joint_expression.sh, so the two epochs here are the same two arms the
# comparing_model5 notebook calls cad_ep4 / cad_ep10.
RUN=/data1/lesliec/sarthak/caduceus/outputs/2026-08-11/19-22-12-155698/checkpoints
EQTL_DIR=/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs

outputs=( \
  "$EQTL_DIR/borzoi32_ep4.npy" \
  "$EQTL_DIR/borzoi32_ep10.npy" \
)

ckpts=( \
  "$RUN/04-val_loss=0.02610.ckpt" \
  "$RUN/10-val_loss=0.02781.ckpt" \
)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#–– run the task ––
# No --ctt_val: this encoder is a plain jointcnn with no cell type token.
# No --softplus: matches the --skip_softplus used for these ckpts in evaluate_joint_expression.sh.
# --pool 1 keeps the native 32 bp bins; use --pool 4 to view them at the 128 bp of the
# Enformer-data eQTL runs.
pixi run python -u eqtl_onemodel_striped.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --pool 1
#data idxs is only if trained on multiple cell types
