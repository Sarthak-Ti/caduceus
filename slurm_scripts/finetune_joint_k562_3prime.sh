#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_k562_3prime_finetune_fixed
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# REGION-based (not gene-based) profile finetune on the 3' RNA-seq track, i.e.
# finetune_joint_k562.sh with the enformer labels.zarr target swapped for the stranded
# RNA-seq coverage track.
#
# What changed vs finetune_joint_k562.sh:
#   - dataset.additional_data=labels.zarr + additional_data_idxs=5111  REMOVED
#     dataset.additional_tracks=K562_rnaseq_stranded.npz               ADDED
#     additional_data is the enformer-style path: it indexes a pre-binned (896, n) array
#     by sequence index. additional_tracks is the genome-style path -- it slices
#     [chrom][rows, start:end] at BASE-PAIR resolution and is never pooled by the dataset
#     (dataset.pool only pools the accessibility targets). So the target now arrives as
#     (length, n_tracks) at bp resolution instead of (896, 1) at 128bp resolution.
#   - decoder.bin_size=1 and decoder.yshape=524288, so the decoder predicts every base
#     pair of the whole window instead of 896 128bp bins of the central 114688bp. The
#     decoder's AvgPool1d(kernel_size=1) is an identity at bin_size=1.
#   - decoder.d_output=2 instead of 1. K562_rnaseq_stranded.npz is chromosome-keyed
#     (2, chrom_len) float16, row 0 = plus strand, row 1 = minus strand, coverage
#     variance-stabilized as sqrt(1+x). celltypes stays at its default of 1 and no
#     data_idxs is set, so the dataset takes row_slice=slice(None) and returns BOTH
#     strands -- hence 2 output channels. (dataset.additional_tracks_stranded is
#     deliberately not set: it only has an effect when celltypes > 1.)
#   - no crop_additional, so the target spans the full length=524288 window and lines up
#     with yshape=524288. Deliberately left unset rather than set to 0.
#   - train.pretrained_model_path (MLM-pretrained backbone, fresh decoder) instead of
#     train.ckpt + load_decoder=true. The decoder's output shape changed from
#     (896, 1) to (524288, 2), so the old decoder weights are meaningless here. This is
#     the same pretrained checkpoint the TSS sc RNA profile runs start from.
#
# Costs barely more than the 896-bin version: with convolutions=true the decoder's
# final_pointwise (256 -> 512 channels) already runs over the FULL window before the
# yshape crop, so widening the crop from 114688 to 524288 only widens the final
# Linear(512 -> 2), whose output is 2 channels. If it does OOM, +decoder.convolutions=false
# drops the 512-channel intermediate entirely and reads out straight off d_model=256.
#
# Loss is poisson_loss_nan (poisson_loss_nll_nan) rather than the config's default
# poisson_loss: identical Poisson NLL, but it returns a zero-gradient batch instead of
# crashing if the loss goes non-finite while the predictions are still finite. bp-resolution
# 3' coverage is extremely sparse/spiky, so that guard is worth having. Use
# task.loss._name_=poisson_loss for the unguarded version.
#
# NOTE: keep dataset.rc_aug=false (the config default). The dataset's rc path flips
# additional_tracks along the sequence axis only (general_dataset.py:516) and does NOT
# swap the plus/minus rows, so reverse-complement augmentation would mislabel the strands
# of this target.

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
 \
 dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 task.loss._name_=poisson_loss_nan \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
 +decoder.d_model=256 +decoder.d_output=2 +decoder.yshape=524288 +decoder.bin_size=1 \
 +dataset.additional_tracks=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded_fixed.npz \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"


#now let's set it to gpu 3 and then run it

# CUDA_VISIBLE_DEVICES=3 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
#  \
#  dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  task.loss._name_=poisson_loss_nan \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true \
#  +decoder.d_model=256 +decoder.d_output=2 +decoder.yshape=524288 +decoder.bin_size=1 \
#  +dataset.additional_tracks=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded.npz \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"
