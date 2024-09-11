#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# SLURM directives
#SBATCH --partition=gpu        # Equivalent to -q in LSF
#SBATCH --ntasks=4             # Equivalent to -n in LSF
#SBATCH --gres=gpu:1           # Request one GPU
#SBATCH --mem=30G              # Memory per node (equivalent to LSF's rusage[mem=30])
#SBATCH --time=168:00:00       # Equivalent to -W in LSF, 168 hours
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j.out   # Equivalent to -o in LSF
#SBATCH --error=/data1/lesliec/sarthak/caduceus/jobs/%j.err    # Equivalent to -e in LSF

# Activate your environment
# source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=enformer wandb.name=enformer_kmer_mamba_gradbatches_1_repeat experiment=hg38/enformer dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=4 dataset.rc_aug=false +dataset.mlm=false \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 +decoder.conjoin_train=true +decoder.conjoin_test=false optimizer.lr="1e-3" \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=196608 dataset.kmer_len=6 \
 model.config.vocab_size=15625 model.config.pad_vocab_size_multiple=1 train.pretrained_safetensors_model_path=null \
 trainer.accumulate_grad_batches=8