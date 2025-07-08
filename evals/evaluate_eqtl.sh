#!/bin/bash

#SBATCH --partition=lesliec,gpu,gpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-5

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("base" "nomlm_maskonly" "nopretrain" "immune" "nobcell" "nobcell_nomlm_maskonly")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/18-07-46-083163/checkpoints/03-val_loss=-0.48683.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-05-08/16-33-43-098345/checkpoints/03-val_loss=-0.45033.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-21/15-40-14-845019/checkpoints/26-val_loss=-0.40909.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-05-15/07-13-50-754738/checkpoints/09-val_loss=-0.19663.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-05-12/17-08-00-389408/checkpoints/07-val_loss=-0.18782.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-05-20/13-43-37-446862/checkpoints/02-val_loss=-0.14833.ckpt" \
)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#–– run the ask ––
pixi run python eqtl_onemodel.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --load_data
#data idxs is only if trained on multiple cell types