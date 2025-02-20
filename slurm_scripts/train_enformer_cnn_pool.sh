#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=Enformer_cnn_128bp_196k_512size_rcaug
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

POOL_VALUE=128

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=enformer wandb.name=$SLURM_JOB_NAME experiment=hg38/enformer dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$SLURM_CPUS_PER_TASK loader.num_workers=$SLURM_CPUS_PER_TASK dataset.rc_aug=false +dataset.mlm=false \
 model.config.vocab_size=12 model.config.pad_vocab_size_multiple=1 +model.config.cnn_embedding=true \
 \
 model=caduceus model.config.d_model=512 model.config.n_layer=16 model._name_=dna_embedding_caduceus \
 model.config.bidirectional=true model.config.bidirectional_weight_tie=true model.config.bidirectional_strategy=add model.config.rcps=false \
 +decoder.conjoin_train=false +decoder.conjoin_test=false optimizer.lr="1e-3" \
 \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=196608 \
 train.pretrained_safetensors_model_path=null trainer.accumulate_grad_batches=8 \
 \
 +dataset.one_hot=true +dataset.pool=$POOL_VALUE +decoder.bin_size=$POOL_VALUE +dataset.cell_type=DNase \
 +dataset.data_path=/data1/lesliec/sarthak/data/borzoi/outputs/hg38/labels.zarr \
 decoder.d_output=674
#  train.ckpt=/data1/lesliec/sarthak/caduceus/outputs/2025-01-30/13-25-56-660102/checkpoints/last.ckpt