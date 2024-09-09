#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=30]"
#BSUB -R "select[hname!='lc03' && hname!='lc05']"
#BSUB -W 168:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/mambadna

cd /data/leslie/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
python -m train wandb.group=enformer wandb.name=enformer_kmer8_mamba_2 experiment=hg38/enformer dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=4 dataset.rc_aug=false +dataset.mlm=false \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 +decoder.conjoin_train=true +decoder.conjoin_test=false optimizer.lr="1e-3" \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=196608 dataset.kmer_len=8 \
 model.config.vocab_size=390625 model.config.pad_vocab_size_multiple=1 train.pretrained_safetensors_model_path=null \
 trainer.accumulate_grad_batches=8 \
 train.ckpt=/data/leslie/sarthak/caduceus/outputs/2024-08-30/14-13-25-111389/checkpoints/last.ckpt \
#  dataset.return_cage=true
#  trainer.accumulate_grad_batches=8
#  +dataset.cell_type=121 \
#  train.pretrained_safetensors_model_path=null \ #default is that it is defined
#  train.ckpt=/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-07-08/17-59-32-315467/checkpoints/last.ckpt \

#mamba enformer kmer