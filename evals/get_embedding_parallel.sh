#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=k562_embedding
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-15

TIME=16 #have to manually define this sadly

# 1) Grab the Slurm array size & this task’s index
TASKS=${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}
ID=$SLURM_ARRAY_TASK_ID          # 0-based if you did --array=0-…
# TIME=$(squeue -j $SLURM_JOB_ID -h --Format=TimeLimit)
# ID=15
# TASKS=16

# 2) Total work and “ideal” chunk size
TOTAL=393328
# basic ceil division
CHUNK=$(( (TOTAL + TASKS - 1) / TASKS )) #with 16 it's 24584
# 3) (Optional) round up CHUNK to a multiple of 4
CHUNK=$(( ((CHUNK + 3) / 4) * 4 ))

# 4) Calculate the start and end indices for this task
START=$(( ID * CHUNK )) #indeed does overlap, so 0-24584, 24584-49168, etc which is good!
END=$(( START + CHUNK ))

# 5) now get GPU memory
GPU_MEM_MIB=$(nvidia-smi \
  --query-gpu=memory.total \
  --format=csv,noheader,nounits \
  | head -n1)
# convert to GiB (round up)
GPU_MEM_GIB=$(( (GPU_MEM_MIB + 1023) / 1024 ))

# 6) if GPU mem is bigger than 48, batch size or 4, otherwise 2
if [ $GPU_MEM_GIB -gt 48 ]; then
  BATCH_SIZE=4
else
  BATCH_SIZE=2
fi

CKPT_PATH="/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/13-44-58-301569/checkpoints/last.ckpt"
ZARR_PATH="/data1/lesliec/sarthak/data/joint_playground/koo_benchmark/embeddings_lentiMPRA_K562.zarr"

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

pixi run python get_embedding.py --start $START --end $END --batch_size $BATCH_SIZE --ckpt_path $CKPT_PATH  --zarr_path $ZARR_PATH --total_time $TIME --load_in