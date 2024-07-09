#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 168:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/

python -m train wandb.group=Profile_long wandb.name=1m_try3_multinomial experiment=hg38/profile_long dataset.batch_size=1 trainer.precision=16 train.count_weight=64.7 \
    dataset.max_length=1000000 model.n_layer=8 train.pretrained_model_path=/data/leslie/sarthak/hyena/hyena-dna/hyenadna-large-1m-seqlen/weights.ckpt
    # task.bias_model="/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-06-21/04-17-47-147286/checkpoints/231-val_loss\=251.30515.ckpt"
#these modifications of dataset and trainer are so will run on a100 efficiently, but noneed to adjust the yaml when debugging on another gpu
#but bf16 seems to have issues with this long stuff for some reason?
#train 1mil