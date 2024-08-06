#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=30]"
#BSUB -sla llSC2
#BSUB -W 168:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/

#pretrained_model_path is to load in the weights but adjust if like you don't want the head
#To resume training, pass in the checkpoint path

python -m train wandb.group=enformer wandb.name=enformer_cnn_fulldata_1 experiment=hg38/enformer dataset.batch_size=2 \
 trainer.precision=bf16 dataset.num_workers=4 train.pretrained_model_path=null +decoder.convolutions=true \
 dataset.max_length=196608 dataset.rc_aug=false
#train.pretrained_model_path=null \
#train.ckpt=/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-07-08/17-59-32-315467/checkpoints/last.ckpt

#enformer linear