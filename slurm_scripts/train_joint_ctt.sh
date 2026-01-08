#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_combined_nomlm_maskonly_immune_ctt_new
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

#NOTE PLEASE NOTE!!! If you want to run again, make sure you use the normal class without ctt in the name and att ctt argument as a bool that is true. Keep dataset arguments

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr \
 dataset.load_in=false +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/all_matched_immune.json \
 +dataset.return_celltype_idx_og=true +encoder.ctt=true \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0 \

#now let's set it to gpu 3 and then run it
# CUDA_VISIBLE_DEVICES=2 python -m train wandb=null experiment=hg38/joint_pretrain dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr \
#  dataset.load_in=false +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/all_matched_immune.json \
#  encoder._name_=jointcnn_ctt +dataset.return_celltype_idx_og=true \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0 \