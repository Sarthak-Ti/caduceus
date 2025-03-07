#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=gpnmsa_basic_cnn_phylopphastcons_2
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=gpnmsa wandb.name=$SLURM_JOB_NAME experiment=hg38/gpnmsa dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$SLURM_CPUS_PER_TASK model.config.vocab_size=12 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=512 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 +decoder.conjoin_train=false +decoder.conjoin_test=false optimizer.lr="1e-3" \
 \
 +decoder.convolutions=false dataset.max_length=196608 trainer.accumulate_grad_batches=8 decoder.d_output=674 \
 dataset.rc_aug=true \
 +model.config.cnn_embedding=true +model.config.cnn_embedding_dim=7 \
 +dataset.phylop=true +dataset.phastcons=true +dataset.human_only=true +dataset.pad_one_hot=null \
#  +model.config.skip_embedding=true
 
