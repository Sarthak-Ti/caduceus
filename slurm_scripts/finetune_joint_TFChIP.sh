#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_combined_gm12878_finetune_chip
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
 \
 dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 +dataset.additional_tracks=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/TFChIP_gm12878.zarr \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
 +decoder.d_model=256 +decoder.d_output=162 +decoder.bin_size=1 +decoder.yshape=524288 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/last.ckpt"
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-29-49-150674/checkpoints/last.ckpt"
#either finetune model or continue training, if continue training then load decoder!

# CUDA_VISIBLE_DEVICES=2 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
#  \
#  dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  +dataset.additional_tracks=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/TFChIP_gm12878.zarr \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
#  +decoder.d_model=256 +decoder.d_output=162 +decoder.bin_size=1 +decoder.yshape=524288 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/last.ckpt"