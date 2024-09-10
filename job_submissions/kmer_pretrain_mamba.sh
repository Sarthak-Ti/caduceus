#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=30]"
#BSUB -R "select[hname!='lc03']"
#BSUB -sla llSC2
#BSUB -W 168:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/mambadna

cd /data/leslie/sarthak/caduceus/
nvidia-smi

LR="8e-3"
BIDIRECTIONAL_STRATEGY="add"
BIDIRECTIONAL_WEIGHT_TIE="true"
RCPS="true"
RC_AUG="false"

python -m train wandb.group=pretrain wandb.name=6mer_mamba_1 experiment=hg38/kmer_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=4 dataset.rc_aug=${RC_AUG} +dataset.mlm=true \
 +dataset.mlm_probability=0.15 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 optimizer.lr="${LR}" dataset.max_length=196608 dataset.kmer_len=6 \
 model.config.vocab_size=15632 model.config.pad_vocab_size_multiple=1 trainer.accumulate_grad_batches=8 \
 +model.config.checkpoint_mixer=true +model.config.checkpoint_mlp=true
#vocab size is 5^6 for 15625 and 7 for the other tokens
#mamba pretrain kmer