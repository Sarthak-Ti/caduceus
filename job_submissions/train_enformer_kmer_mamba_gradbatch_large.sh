#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 16
#BSUB -gpu "num=4:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=30]"
#BSUB -R "select[hname!='lc03']"
#BSUB -sla llSC2
#BSUB -W 1:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/mambadna

cd /data/leslie/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
python -m train wandb=null experiment=hg38/enformer dataset.batch_size=1 \
 trainer.precision=bf16-mixed dataset.num_workers=2 dataset.rc_aug=false +dataset.mlm=false \
 model=caduceus model.config.d_model=256 model.config.n_layer=24 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 +decoder.conjoin_train=true +decoder.conjoin_test=false optimizer.lr="1e-3" \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=196608 dataset.kmer_len=6 \
 model.config.vocab_size=15625 model.config.pad_vocab_size_multiple=1 train.pretrained_safetensors_model_path=null \
 trainer.accumulate_grad_batches=8 trainer.strategy=fsdp trainer.devices=2 \
 +model.config.checkpoint_mixer=true +model.config.checkpoint_mlp=true \
 +trainer.gradient_clip_algorithm=value
#gradbatch large