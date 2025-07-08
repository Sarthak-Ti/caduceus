#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=100G
#SBATCH --job-name=caQTL
#SBATCH --gres=gpu:a100:1
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-3

echo "Arguments passed are TOTAL=$TOTAL, ZARR_OUT=$ZARR_OUT, MASK_SIZE=$MASK_SIZE, CKPT=$CKPT"


# 1) Grab the Slurm array size & this task’s index
TASKS=${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}
ID=$SLURM_ARRAY_TASK_ID          # 0-based if you did --array=0-…

# 2) Total work and “ideal” chunk size
# TOTAL=95065
# basic ceil division
CHUNK=$(( (TOTAL + TASKS - 1) / TASKS )) #with 16 it's 24584
CHUNK=$(( ((CHUNK + 99) / 100) * 100 )) #and round up to a multiple of 100

# 4) Calculate the start and end indices for this task
START=$(( ID * CHUNK )) #indeed does overlap, so 0-24584, 24584-49168, etc which is good!
END=$(( START + CHUNK ))

# Now simply run the script, we have our start and end!

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

pixi run python caqtl_onemodel.py \
  --start $START \
  --end $END \
  --ckpt_path $CKPT \
  --o $ZARR_OUT \
  --mask_size $MASK_SIZE \
  --load_data