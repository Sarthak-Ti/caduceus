#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=GR_pure_CNN_100k
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%x-%j.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=graphreg wandb.name=$SLURM_JOB_NAME experiment=hg38/GraphReg_conv dataset.batch_size=128 dataset.batch_size_eval=128 \
 trainer.precision=bf16 dataset.num_workers=$SLURM_CPUS_PER_TASK \
 dataset.cell_type=K562 +dataset.clean_data=true dataset.max_length=100000 +dataset.remove_repeats=true

#to test this script
# python -m train wandb=null experiment=hg38/GraphReg_conv dataset.batch_size=1 \
#   trainer.precision=bf16 dataset.num_workers=1 trainer.accumulate_grad_batches=8 \
#   dataset.cell_type=K562