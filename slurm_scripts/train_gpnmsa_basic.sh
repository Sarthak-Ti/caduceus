#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=gpnmsa_basic_ohe_2_continued
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 2))

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=gpnmsa wandb.name=$SLURM_JOB_NAME experiment=hg38/gpnmsa dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS model.config.vocab_size=12 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=512 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 +decoder.conjoin_train=false +decoder.conjoin_test=false optimizer.lr="1e-3" \
 \
 +decoder.convolutions=false dataset.max_length=196608 trainer.accumulate_grad_batches=8 decoder.d_output=674 \
 dataset.rc_aug=true \
 +model.config.skip_embedding=true \
 train.ckpt=/data1/lesliec/sarthak/caduceus/outputs/2025-02-24/17-52-51-819562/checkpoints/last.ckpt \
 +train.pretrained_model_state_hook._name_=load_full_model \
 loader.num_workers=$WORKERS \
 #+model.config.cnn_embedding=true 

#here is how you can run it in the terminal with only gpu0
# CUDA_VISIBLE_DEVICES=3 pixi run python -m train wandb=null experiment=hg38/gpnmsa dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 model.config.vocab_size=12 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=512 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false optimizer.lr="1e-3" \
#  \
#  +decoder.convolutions=false dataset.max_length=196608 trainer.accumulate_grad_batches=8 decoder.d_output=674 \
#  dataset.rc_aug=true \
#  +model.config.skip_embedding=true \
#  train.ckpt=/data1/lesliec/sarthak/caduceus/outputs/2025-02-24/17-52-51-819562/checkpoints/last.ckpt \
#  +train.pretrained_model_state_hook._name_=load_full_model \
#  loader.num_workers=0