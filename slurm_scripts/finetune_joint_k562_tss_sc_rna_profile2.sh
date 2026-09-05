#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_sc_rna_profile_tss_fixed
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# RNA-seq PROFILE finetune built on the count-only recipe from
# finetune_joint_k562_tss_sc.sh, with the per-bp profile head bolted on as a heavily
# downweighted auxiliary task.
#
# Count head:
#   - the GENE-BODY mask is appended as an INPUT CHANNEL (+dataset.append_gene_mask=true),
#     not used as a pooling region. Switched from append_tss_mask so that this run is a
#     one-factor change from the count-only decima run (2026-07-27/21-54-24-337889, "decima 2"
#     in code_test/evaluation_tss.ipynb), which is gene mask + upstream=163840 + the plain
#     `tss` count decoder. That pair isolates the profile head. It is also a one-factor change
#     from 2026-08-05/12-48-43-803485 ("rna profile"), which is this exact recipe with the TSS
#     mask, so that pair isolates the mask type.
#   - the gene-body mask is the more informative channel for a 3' target: it marks the
#     transcript extent, i.e. where the 3' end the profile head must place coverage on
#     actually is. With only a TSS mask, gene extent has to be inferred from sequence.
#   - the head mean-pools over the WHOLE window (+decoder.count_region=all)
#   - same cellranger TSS json as every other run in the sweep, so the count target is
#     identical and the count metrics are directly comparable
#
# Window: ASYMMETRIC, +dataset.upstream=163840, so the window is [TSS-163840, TSS+360448).
# The TSS is still inside the window, just at index 163840 instead of the center; the extra
# 98kb is spent downstream instead of upstream. That is deliberate for a 3' RNA-seq target --
# coverage piles up at the last exon / polyA site, so the downstream half is where the signal
# to be predicted actually lives, and 360448bp downstream keeps essentially every gene's 3' end
# in context (gene bodies: median 8.9kb, p95 76kb, only 0.45% exceed 262144bp). A TSS-centered
# window would spend roughly half of profile_region=all's multinomial normalization on upstream
# sequence carrying essentially no sense-strand RNA.
#
# Consequence for comparisons: this window differs from finetune_joint_k562_tss_sc.sh, which is
# TSS-centered, so that script is NOT the baseline for this run -- it differs in mask, window
# and decoder all at once. The baseline is the decima count-only run (gene mask, upstream=163840,
# `tss` decoder), which differs only in the decoder. See code_test/evaluation_tss.ipynb P1/P8 for
# what the existing runs do and do not disentangle.
#
# Profile head -- per-bp RNA-seq coverage over the entire window
# (+decoder.profile_region=all, +task.loss.region=all).
#
# Loss = count_weight*MSE(count) + profile_weight*multinomial_NLL(profile).
# The two terms sit on wildly different scales: the NLL sums over the whole window
# (~600k) while the count MSE is order ~6. profile_weight=1e-6 pulls the profile term
# down to ~0.6 so the count term dominates the gradient. Scaling the profile side DOWN
# (rather than raising count_weight to 1e6) leaves the total loss magnitude -- and
# therefore the effective learning rate -- unchanged.
#
# Logging note: tss_profile_count_mse and tss_profile_multinomial are both RAW/unweighted,
# so they show the true per-head losses. The tss_profile_loss METRIC is unweighted too
# (metrics are called without the task.loss kwargs), so it will NOT match the training
# loss -- read train/loss for what is actually being optimized.
#
# append_tss_mask and append_gene_mask are mutually exclusive (tss_dataset.py:208 asserts it --
# there is only one extra channel slot), so append_tss_mask must be removed rather than just
# adding append_gene_mask. Note that neither HEAD reads the mask: with profile_region=all and
# loss region=all both heads span the whole window, so append_gene_mask changes only what the
# ENCODER sees on channel 6. The dataset returns tss_mask and gene_mask in outputs2 either way
# (tss_dataset.py:612-617), so nothing downstream changes.
#
# RNA-seq expression track: K562_rnaseq_stranded.npz is chromosome-keyed (2, chrom_len)
# float16 with row 0 = plus strand, row 1 = minus strand, coverage variance-stabilized
# as sqrt(1+x). expression_stranded=true makes the dataset return only the sense-strand
# track per gene (+ gene -> plus, - gene -> minus, flipped into transcription orientation
# by rc_strand), so expression is a single channel matching decoder.n_tracks=1.

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L | wc -l)

pixi run srun python -m train wandb.group=tss_finetune wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
 \
 dataset._name_=TSSLoader \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
 dataset.load_in=false \
 dataset.shift_sequences=2000 \
 +dataset.rc_strand=true \
 +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_sc_rna_info_cellranger.json \
 dataset.acc_type=continuous \
 +dataset.append_tss_mask=true \
 +dataset.upstream=163840 \
 +dataset.expression_data_path=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded_fixed.npz \
 +dataset.expression_stranded=true \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 task._name_=joint_tss \
 task.loss._name_=tss_profile_loss +task.loss.count_weight=1.0 +task.loss.profile_weight=1e-6 +task.loss.region=all \
 task.metrics=[tss_profile_loss,tss_profile_count_mse,tss_profile_multinomial] \
 \
 decoder._name_=tss_profile +decoder.n_tracks=1 +decoder.profile_region=all +decoder.count_region=all +decoder.hidden_dim=128 trainer.accumulate_grad_batches=16 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"


# CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
#  \
#  dataset._name_=TSSLoader \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
#  dataset.load_in=false \
#  dataset.shift_sequences=2000 \
#  +dataset.rc_strand=true \
#  +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_sc_rna_info_cellranger.json \
#  dataset.acc_type=continuous \
#  +dataset.append_tss_mask=true \
#  +dataset.upstream=163840 \
#  +dataset.expression_data_path=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded.npz \
#  +dataset.expression_stranded=true \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  task._name_=joint_tss \
#  task.loss._name_=tss_profile_loss +task.loss.count_weight=1.0 +task.loss.profile_weight=1e-6 +task.loss.region=all \
#  task.metrics=[tss_profile_loss,tss_profile_count_mse,tss_profile_multinomial] \
#  \
#  decoder._name_=tss_profile +decoder.n_tracks=1 +decoder.profile_region=all +decoder.count_region=all +decoder.hidden_dim=128 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"
