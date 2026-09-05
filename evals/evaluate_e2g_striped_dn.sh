#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_striped_dn
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- striped model only, two jobs: dinucleotide shuffle alone, and shuffle + accessibility reduction --
#new output names so the accessibility-reduction results from evaluate_e2g_striped.sh are not overwritten
outputs=("k562_striped_tss_centered_dinuc" "k562_striped_tss_centered_dinuc_acc")
perturbations=("shuffle" "both_perturbations")

CKPT="/data1/lesliec/sarthak/caduceus/outputs/2026-05-13/12-20-24-265445/checkpoints/02-step=19250.ckpt"
N_SHUFFLES=11

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
PERTURBATION=${perturbations[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, perturbation=$PERTURBATION, n_shuffles=$N_SHUFFLES"
#-- run the task --
#--scale_factor and --dist_additional_mask are inert for perturbation=shuffle, only used by both_perturbations
pixi run python -u e2g_striped_dn.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --perturbation "$PERTURBATION" \
  --n_shuffles "$N_SHUFFLES" \
  --dist_additional_shuffle 100 \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --seed 0 \
  --batch_size 2 \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
  --ctt_val 1 \
  --pool 32 \
  --center_tss
