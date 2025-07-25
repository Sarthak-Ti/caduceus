#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_combined_gm12878_finetune_RNAseq_2
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
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true 
 \
 dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi_fold3-4.bed \
 +decoder.yshape=196608 +decoder.bin_size=32 +dataset.additional_data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/gm12878_RNA.json \
 +dataset.additional_data=/data1/lesliec/sarthak/data/borzoi/borzoi.zarr \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
 +decoder.d_model=256 +decoder.d_output=30 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-04-08/11-50-37-816676/checkpoints/12-val_loss\=0.27185.ckpt"


#now let's set it to gpu 3 and then run it

# CUDA_VISIBLE_DEVICES=2 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" \
#  \
#  dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi_fold3-4.bed \
#  +decoder.yshape=196608 +decoder.bin_size=32 +dataset.additional_data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/gm12878_RNA.json \
#  +dataset.additional_data=/data1/lesliec/sarthak/data/borzoi/borzoi.zarr \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
#  +decoder.d_model=256 +decoder.d_output=1 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-04-08/11-50-37-816676/checkpoints/12-val_loss\=0.27185.ckpt"