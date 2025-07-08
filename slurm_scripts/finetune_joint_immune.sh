#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_combined_gm12878_finetune_nobcell_nomlm_maskonly
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
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
 +decoder.d_model=256 +decoder.d_output=1 +dataset.additional_data=/data1/lesliec/sarthak/data/enformer/data/labels.zarr \
 +dataset.additional_data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/nob_immune_CAGE.json \
 +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/nob_immune.json \
 train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2025-05-20/13-43-37-446862/checkpoints/last.ckpt" +train.pretrained_model_state_hook.load_decoder=true
#  train.pretrained_model_path=/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-39-56-643580/checkpoints/last.ckpt \


#now let's set it to gpu 3 and then run it

# CUDA_VISIBLE_DEVICES=1 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" \
#  \
#  dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_processed.zarr \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
#  +decoder.d_model=256 +decoder.d_output=1 +dataset.additional_data=/data1/lesliec/sarthak/data/enformer/data/labels.zarr \
#  +dataset.additional_data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/nob_immune_CAGE.json \
#  +dataset.data_idxs=/data1/lesliec/sarthak/data/DK_zarr/idx_lists/nob_immune.json \
#  train.pretrained_model_path=/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/last.ckpt