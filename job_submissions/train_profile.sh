#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 40:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/

python -m train wandb.group=Profile_1000_800_b wandb.name=bias_try5 experiment=hg38/profile_nobias dataset.batch_size=2048 trainer.precision=bf16 train.count_weight=64.7 \
    task.bias_model=/data/leslie/sarthak/data/chrombpnet_test/chrombpnet_model_1000/models/bias_model_scaled.h5
    # task.bias_model="/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-06-21/04-17-47-147286/checkpoints/231-val_loss\=251.30515.ckpt"
#these modifications of dataset and trainer are so will run on a100 efficiently, but noneed to adjust the yaml when debugging on another gpu
#cbpnet bias nonlinear