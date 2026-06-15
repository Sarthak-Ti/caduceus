#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_k562plus10_striped_maskonly
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS trainer.devices=$NUM_GPUS \
 \
 model=striped_hydra model._name_=dna_embedding_striped \
 model.config.d_model=512 model.config.mode=striped model.config.d_conv=7 model.config.expand=2 \
 model.config.ngroups=8 model.config.residual_in_fp32=true \
 +model.config.d_in=128 +model.config.global_pooling=128 +model.config.transformer_pooling=4 \
 +model.config.ssm_per_transformer=3 +model.config.expansion_factor=2 \
 +model.config.dropout=0.1 +model.config.sampling_checkpoint=true \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr dataset.acc_mlm=0.25 dataset.mlm=0 +dataset.mask_only=true dataset.length=2097152 +dataset.return_celltype_idx_og=true \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/k562plus10.json \
 \
 encoder.joint=true +encoder.d_model=128 +encoder.transpose=true +encoder.norm=null +encoder.activation=null +encoder.ctt=true +encoder.celltypes=10 \
 \
 +callbacks=model_every_n_steps callbacks.model_checkpoint_every_n_steps.every_n_train_steps=2000 trainer.accumulate_grad_batches=16 \
 train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2026-04-09/22-16-57-437275/checkpoints/00-val_loss\=0.19021.ckpt" +train.pretrained_model_state_hook.load_decoder=true \

#run locally

# python -m train wandb=null experiment=hg38/joint_pretrain dataset.batch_size=2 \
#  trainer.precision=bf16 dataset.num_workers=0 loader.num_workers=0 trainer.devices=1 \
#  \
#  model=striped_hydra model._name_=dna_embedding_striped \
#  model.config.d_model=512 model.config.mode=striped model.config.d_conv=7 model.config.expand=2 \
#  model.config.ngroups=8 model.config.residual_in_fp32=true \
#  +model.config.d_in=128 +model.config.global_pooling=128 +model.config.transformer_pooling=4 \
#  +model.config.ssm_per_transformer=3 +model.config.expansion_factor=2 \
#  +model.config.dropout=0.1 +model.config.sampling_checkpoint=true \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr dataset.acc_mlm=0.25 dataset.mlm=0 +dataset.mask_only=true dataset.length=2097152 +dataset.return_celltype_idx_og=true \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/k562plus10.json \
#  \
#  encoder.joint=true +encoder.d_model=128 +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
#  \
#  +callbacks=model_every_n_steps callbacks.model_checkpoint_every_n_steps.every_n_train_steps=2000 trainer.accumulate_grad_batches=16