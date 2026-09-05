#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_sc_rna_poisson
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# POISSON profile finetune. This is a ONE-FACTOR change off
# finetune_joint_k562_tss_sc_rna_profile2.sh: same dataset, same window, same gene mask,
# same count head, same count target. Only the per-position profile term changes, from a
# multinomial NLL to a Poisson NLL. That pair isolates the loss family.
#
# Why: for the e2g enhancer-gene linking task, Poisson counts on the 3' data
# (finetune_joint_k562_3prime.sh) beat the multinomial profile model. The likely mechanism
# is that the multinomial is scale-free by construction, so ALL magnitude information has
# to squeeze through the count head's single mean-pooled d_model=256 vector for the whole
# ~524kb window. Under Poisson every base pair carries magnitude, so perturbing an enhancer
# moves the local predicted rate directly -- dense, local gradient signal, which is exactly
# what an e2g perturbation readout measures.
#
# MATCHING THE 3' RUN'S POISSON EXACTLY. +decoder.profile_softplus=true makes the profile
# head emit a non-negative RATE (Softplus) instead of a raw logit, which is what the 3'
# model's EnformerDecoder does unconditionally (decoders.py, EnformerDecoder.forward). The
# loss then runs F.poisson_nll_loss(log_input=False, full=False, eps=1e-6) -- the same call
# poisson_loss_nll_nan makes there. Without this flag the loss falls back to log_input=True
# on raw logits, which is the canonical log link but a DIFFERENT parameterization
# (exp vs softplus), so the comparison to the 3' run would no longer be one-factor. exp()
# also has multiplicative gradients that overflow readily under bf16; softplus is bounded
# and is the parameterization that actually trained on this target.
# The loss defaults to profile_softplus=True, so the two flags agree without setting it on
# task.loss -- but they MUST be kept in sync if either is changed.
#
# WEIGHTS ARE NOT THE PROFILE2 WEIGHTS. F.poisson_nll_loss reduces with a MEAN over
# positions, so its value is order 1e0; cbpnet_multinomial_nll SUMS over the ~524k-position
# length axis, so its value is order 1e5-1e6. The profile_weight=1e-6 that profile2 needed
# would be ~1e6 too small here. poisson_weight=1.0 puts the profile term at the same order
# as the count MSE, so neither head is silenced.
#
# Measured on a real 524288bp window (chr21:31457280-31981568, plus strand, 98% at floor)
# against the _fixed target below: the Poisson term runs from -0.57 (oracle, lambda=y) to
# +0.54 (constant baseline, lambda=mean), against a count MSE of ~1.1 at init. Same order,
# so poisson_weight=1.0 is calibrated and needs no adjustment. For reference the same window
# on the OLD floor-1.0 target spanned only 0.38 to 0.98 -- the fix nearly doubles the
# Poisson term's dynamic range (0.60 -> 1.11), i.e. more learnable signal per position.
# Both terms are logged unweighted (tss_profile_poisson, tss_profile_count_mse), so check
# their ratio in the first few hundred steps before trusting this.
#
# multinomial_weight defaults to 0 and is left off: the Poisson likelihood already contains
# the shape information the multinomial isolates, so running both is redundant by
# construction. +task.loss.multinomial_weight=<w> turns it on for the ablation, and
# tss_profile_multinomial_softplus logs the term either way (the plain
# tss_profile_multinomial metric would be WRONG here -- it log_softmaxes the rate instead of
# log(rate), so use the _softplus alias whenever profile_softplus=true).
#
# TARGET IS THE _fixed TRACK. K562_rnaseq_stranded.npz was written by bigwig_to_npz.py as
# sqrt(1 + x) applied to the WHOLE chromosome, which drops the "-1" of the intended
# Decima-style transform -1 + (1+x)^p: zero-coverage bases map to sqrt(1) = 1.0 rather than
# 0, and on chr21 93% of positions sit at that floor. K562_rnaseq_stranded_fixed.npz
# (make_fixed_npz.py, alongside the original) is that file minus 1, i.e. -1 + (1+x)^(1/2).
# The rewrite is exactly lossless: the stored values are sqrt(1+x) for INTEGER x, so x=0 and
# x=1 stay cleanly separated, and s-1 is exactly representable in float16.
#
# Why it matters here, and why it matters ASYMMETRICALLY: under the old floor, 93% of
# positions handed the model a constant target of 1.0. Poisson was mildly hurt -- it had to
# spend capacity reproducing a constant background instead of predicting sparse counts.
# The MULTINOMIAL was hurt far more: its softmax mass over ~524k positions was dominated by
# uniform background, so its shape term was fitting near-noise. That means the e2g result
# motivating this whole run (Poisson on 3' beat the multinomial profile model) is confounded
# in the direction that flatters Poisson. Treat "Poisson > multinomial" as UNRESOLVED until
# finetune_joint_k562_tss_sc_rna_profile2.sh is re-run against this same _fixed track.
#
# Note the 3' run (finetune_joint_k562_3prime.sh) still points at the ORIGINAL npz, so it is
# no longer target-matched to this run. Repoint it at _fixed before comparing the two.
#
# Poisson NLL still assumes counts while the target is variance-stabilized, so this remains
# Poisson on transformed counts -- common in this model family, and strictly more sensible
# now that zero maps to zero, but not textbook. Also note F.poisson_nll_loss uses full=False,
# so the log(y!) normalizer is dropped and the profile term CAN GO NEGATIVE on high-coverage
# positions. That is expected, not a bug; train/loss may dip below zero.
#
# Everything below this point is unchanged from finetune_joint_k562_tss_sc_rna_profile2.sh:
# gene-body mask as INPUT CHANNEL (+dataset.append_gene_mask=true, not a pooling region),
# asymmetric window +dataset.upstream=163840 -> [TSS-163840, TSS+360448) so the downstream
# half where 3' coverage piles up stays in context, count head mean-pooling the whole window
# (+decoder.count_region=all), profile spanning the whole window
# (+decoder.profile_region=all), and the same cellranger TSS json so the count target is
# identical and count metrics stay comparable across the sweep. dataset.crop_output is left
# at its default of 0, so the profile spans the full window.
#
# expression_stranded=true returns only the sense-strand track per gene, flipped into
# transcription orientation by rc_strand, so expression is a single channel matching
# decoder.n_tracks=1.

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
 +dataset.append_gene_mask=true \
 +dataset.upstream=163840 \
 +dataset.expression_data_path=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded_fixed.npz \
 +dataset.expression_stranded=true \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 task._name_=joint_tss \
 task.loss._name_=tss_profile_poisson_loss +task.loss.count_weight=1.0 +task.loss.poisson_weight=1.0 +task.loss.region=all \
 task.metrics=[tss_profile_count_mse,tss_profile_poisson,tss_profile_multinomial_softplus] \
 \
 decoder._name_=tss_profile +decoder.n_tracks=1 +decoder.profile_region=all +decoder.count_region=all +decoder.hidden_dim=128 +decoder.profile_softplus=true trainer.accumulate_grad_batches=16 \
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
#  +dataset.append_gene_mask=true \
#  +dataset.upstream=163840 \
#  +dataset.expression_data_path=/data1/lesliec/sarthak/data/DE_danwei/K562_gex/K562_rnaseq_stranded_fixed.npz \
#  +dataset.expression_stranded=true \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  task._name_=joint_tss \
#  task.loss._name_=tss_profile_poisson_loss +task.loss.count_weight=1.0 +task.loss.poisson_weight=1.0 +task.loss.region=all \
#  task.metrics=[tss_profile_count_mse,tss_profile_poisson,tss_profile_multinomial_softplus] \
#  \
#  decoder._name_=tss_profile +decoder.n_tracks=1 +decoder.profile_region=all +decoder.count_region=all +decoder.hidden_dim=128 +decoder.profile_softplus=true \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"
