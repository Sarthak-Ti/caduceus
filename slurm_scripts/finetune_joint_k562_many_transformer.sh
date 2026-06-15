#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=168:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=finetune_k562plus10_transformer
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS trainer.devices=$NUM_GPUS \
 \
 model=striped_hydra model._name_=dna_embedding_striped \
 model.config.d_model=512 model.config.mode=transformer_only model.config.n_blocks=16 \
 model.config.residual_in_fp32=true \
 +model.config.d_in=128 +model.config.global_pooling=512 \
 +model.config.expansion_factor=2 \
 +model.config.dropout=0.1 +model.config.attention_dropout=0.1 +model.config.sampling_checkpoint=true \
 task.loss._name_=poisson_loss_nan \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 dataset.acc_type=continuous +dataset.crop_additional=524288 dataset.shift_sequences=1000 \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr dataset.length=2097152 +dataset.return_celltype_idx_og=true \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/k562plus10.json \
 \
 encoder.joint=true +encoder.d_model=128 +encoder.transpose=true +encoder.norm=null +encoder.activation=null +encoder.ctt=true +encoder.celltypes=10 \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=false \
 +decoder.d_model=128 +decoder.d_output=2 +decoder.yshape=1048576 +decoder.bin_size=1 \
 +dataset.additional_tracks=/data1/lesliec/sarthak/data/alphagenome/k562plus10_CAGE.zarr \
 +dataset.additional_tracks_stranded=true train.seed=2222 \
 \
 +callbacks=model_every_n_steps callbacks.model_checkpoint_every_n_steps.every_n_train_steps=1000 trainer.accumulate_grad_batches=32 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2026-05-01/15-20-40-118107/checkpoints/last.ckpt"
#  train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2026-04-27/09-13-27-212978/checkpoints/00-step\=9500.ckpt" +train.pretrained_model_state_hook.load_decoder=true \
 
#  train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2026-04-09/22-16-57-437275/checkpoints/00-val_loss\=0.19021.ckpt" +train.pretrained_model_state_hook.load_decoder=true \

#run locally

# python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=0 loader.num_workers=0 trainer.devices=1 \
#  \
#  model=striped_hydra model._name_=dna_embedding_striped \
#  model.config.d_model=512 model.config.mode=transformer_only model.config.n_blocks=16 \
#  model.config.residual_in_fp32=true \
#  +model.config.d_in=128 +model.config.global_pooling=512 \
#  +model.config.expansion_factor=2 \
#  +model.config.dropout=0.1 +model.config.attention_dropout=0.1 +model.config.sampling_checkpoint=true \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  dataset.acc_type=continuous +dataset.crop_additional=524288 dataset.shift_sequences=1000 \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr dataset.length=2097152 +dataset.return_celltype_idx_og=true \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/k562plus10.json \
#  \
#  encoder.joint=true +encoder.d_model=128 +encoder.transpose=true +encoder.norm=null +encoder.activation=null +encoder.ctt=true +encoder.celltypes=10 \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=false \
#  +decoder.d_model=128 +decoder.d_output=2 +decoder.yshape=1048576 +decoder.bin_size=1 \
#  +dataset.additional_tracks=/data1/lesliec/sarthak/data/alphagenome/k562plus10_CAGE.zarr \
#  +dataset.additional_tracks_stranded=true \
#  \
#  +callbacks=model_every_n_steps callbacks.model_checkpoint_every_n_steps.every_n_train_steps=2000 trainer.accumulate_grad_batches=16 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2026-04-14/17-23-24-651827/checkpoints/01-step\=16000.ckpt"
#  train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2026-04-19/21-58-28-647055/checkpoints/00-step\=4000.ckpt" +train.pretrained_model_state_hook.load_decoder=true \
