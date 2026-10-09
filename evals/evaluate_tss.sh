#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=tss_eval
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-2

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))

#two decoder flavours appear in this file. A TSSProfileDecoder returns TWO outputs (per-bp
#profile logits + scalar count) and eval_tss_profile.py scores the COUNT head only; a
#single-output TSSDecoder returns just that count and runs through eval_tss.py. Both heads
#predict the same log-scale quantity, so every round in this file is directly comparable -- and
#the profile head is never scored here, whatever it was trained with. Pick the script to match
#the checkpoint: the two commands at the bottom differ only in which one is uncommented.

#current round -- the K562 2x2 over (pretraining x accessibility-input), all three cells that
#were still training as of 2026-09-14. These are single-output TSSDecoder models (decoder._name_=tss,
#pool_region=all, d_output=1), so they run through eval_tss.py, NOT eval_tss_profile.py -- uncomment
#the right command at the bottom. All three share the Decima geometry of the rounds below (gene mask
#appended as an input channel, upstream=163840 so the TSS sits at 163840 in a 524288 bp window,
#rc_strand on, mse_tss count loss, k562_sc_rna_info_cellranger.json), so the numbers are directly
#comparable to k562_decima_2 further down -- that run IS the fourth cell of the 2x2 (same 2025-07-18
#pretrained backbone, acc_mlm=0, identical window/gene-mask/json/decoder settings), so no extra run
#is needed for it; it just sits at epoch 16 rather than whatever epoch is picked here.
#
#            | ATAC input        | ATAC masked
#   ---------+-------------------+-------------------
#   pretrain | k562_decima_2     | 0 (job 11838935)
#   scratch  | 2 (job 11839840)  | 1 (job 11839705)
#
#0 = noacc            : pretrained backbone (2025-07-18 last.ckpt), acc_mlm=1 + mask_only=1
#1 = noacc_scratch    : random init, acc_mlm=1 + mask_only=1
#2 = scratch          : random init, accessibility as a normal input (acc_mlm=0)
#
#acc_mlm=1 with mask_only=1 means the accessibility track is masked at rate 1.0 with no BERT-style
#10% random / 10% unchanged, i.e. every position is zeroed and the mask channel is set to 1. That is
#deterministic, so evaluating with masking still on is reproducible AND is the only setting that
#matches training for cells 0 and 1 -- do not override acc_mlm to 0 for them, the model has never
#seen a real accessibility value. e2g_tss.py:Evals (which eval_tss.py imports) now accepts
#acc_mlm in (0, 1) and prints a warning when it is 1; anything strictly between is still rejected as
#stochastic and therefore irreproducible.
#
#E2G for all three lives in evaluate_e2g_tss_dn.sh tasks 10-12, one sbatch. The perturbation
#differs per cell because an accessibility knockdown cannot test a model that never sees
#accessibility:
#  indices 0, 1 (ATAC masked) -> tasks 10, 11, dinucleotide shuffle
#  index 2 (real ATAC)        -> task 12, accessibility knockdown (--perturbation acc, which for a
#                                single-output TSSDecoder reduces exactly to e2g_tss.py)
#This file is unaffected by that split: the expression eval is valid for all three.
#
#EPOCHS: each run at its own LOWEST val/loss epoch -- 04 / 19 / 27 respectively. Unlike the
#profile rounds elsewhere in these files, val/loss here IS the quantity the eval scores: all three
#use task.loss._name_=mse_tss with task.metrics=[mse_tss], and the logs confirm val/mse_tss equals
#val/loss exactly, so "lowest val/loss" needs no proxy argument and is comparable ACROSS the three
#(the loss is identical in form). Off-by-one reminder: the log's "Epoch e" progress-bar line reports
#what ModelCheckpoint saved as file e-1, so these filenames sit one below the log epochs.
#
#CONFOUND, read the numbers with it in mind: these are NOT matched training lengths.
#  noacc (pretrained)  min 1.53684 @ ep 04, then every one of the next 22 epochs is worse
#                      (1.833 @ ep 26) with train MSE at 0.0896 -- it memorised the training set
#                      almost immediately. Its "best" is a 4-epoch model.
#  noacc_scratch       min 1.79403 @ ep 19 (ep 16 is 1.79463, effectively tied), noisy throughout.
#  scratch (real ATAC) min 0.68757 @ ep 27, still drifting down slowly when training stopped.
#So a 4-epoch model is being compared against a 19- and a 27-epoch one. The pretrained run
#converging by epoch 5 and then overfitting hard, while the scratch runs were still descending, is
#itself the pretraining result -- but do not read the raw gap as a fixed-budget comparison. Rerun
#at a matched epoch (26 is the last checkpoint all three share) if that comparison is needed.
#
#TRAINING ENDED EARLY, all three cancelled 2026-09-14 as good enough rather than run to completion:
#jobs 11838935 and 11839705 had in fact been dead since 2026-09-13 (OSError Errno 28, /tmp full on
#iscb027 and iscb029 during dataloader worker startup), last checkpoint 26; job 11839840 was alive
#in epoch 32 and was cancelled deliberately. Set TMPDIR onto /data1 in the training script if these
#are ever rerun.
outputs=("k562_tss_sc_decima_noacc_ep04" "k562_tss_sc_decima_noacc_scratch_ep19" "k562_tss_sc_decima_scratch_ep27")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/11-32-12-827324/checkpoints/04-val_loss=1.53684.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/11-37-47-202762/checkpoints/19-val_loss=1.79403.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/13-23-33-047440/checkpoints/27-val_loss=0.68757.ckpt" \
)

#previous round -- the three runs trained on the rescaled RNA-seq npz
#(K562_rnaseq_stranded_fixed.npz). The npz only feeds the PROFILE head, and this script forces
#expression_data_path=None, so it never opens it: these count numbers are directly comparable to
#the k562_tss_sc_rna_profile2_ep31 row below (trained on the un-rescaled npz) and to the
#epoch-16 single-output runs further down.
#0 = poisson    : gene mask channel, Poisson profile NLL (weight 1.0), decoder profile_softplus
#1 = decima_fixed: gene mask channel, multinomial profile NLL (weight 1e-6)
#2 = tss_fixed  : TSS mask channel, multinomial profile NLL (weight 1e-6)
#each run at its own LOWEST val/loss epoch (35 / 34 / 10), not a matched epoch. Two caveats:
#val/loss is only comparable WITHIN a run -- across these three the loss terms differ (poisson
#carries a mean-reduced Poisson NLL at weight 1.0, the other two a summed multinomial at 1e-6) --
#and it is not the count MSE this script actually scores. Val count MSE at these epochs is
#0.705 / 0.689 / 0.723, vs each run's best of 0.666 (ep 12) / 0.648 (ep 16) / 0.723 (ep 10) --
#so tss_fixed lands exactly on its best and the other two give up ~0.04 (~1.2 sd of the
#epoch-to-epoch spread, which is 0.03-0.04 over epochs >=10). Beware when reading these epoch
#numbers off the training log: the progress bar line labelled "Epoch e" reports the metrics
#ModelCheckpoint saved as epoch e-1, so log epochs run one ahead of the checkpoint filenames.
#(use #SBATCH --array=0-2 for these, and eval_tss_profile.py rather than eval_tss.py)
# outputs=("k562_tss_sc_rna_poisson_ep35" "k562_tss_sc_rna_profile_decima_fixed_ep34" "k562_tss_sc_rna_profile_tss_fixed_ep10")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
# )

#previous round -- the same profile decoder trained on the un-rescaled npz
#(use #SBATCH --array=0-0 for this one)
#RNA-seq PROFILE model (job 6052571, k562_tss_sc_rna_profile2): TSSProfileDecoder, so the
#decoder returns TWO outputs (per-bp profile logits + scalar count). eval_tss_profile.py scores
#the COUNT head only, which is the same log-scale quantity the single-output models predict, so
#these numbers are directly comparable to the epoch-16 runs commented out below.
#TSS mask appended as an input channel, asymmetric window with the TSS at 163840, count head
#pools over the whole window (count_region=all).
# outputs=("k562_tss_sc_rna_profile2_ep31")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-08-05/12-48-43-803485/checkpoints/31-val_loss=1.47817.ckpt" \
# )

#previous round -- single-output TSSDecoder models, run with eval_tss.py (see the commented
#command at the bottom). All three are single-cell (cellranger json) models retrained under the
#revised TSSDecoder (plain regression head, no softplus / no log2 -- the target is already
#log1p), so their predictions are directly on the natural-log target scale
#0 = decima style model (gene mask as an input channel, asymmetric window with the TSS at
#    163840, decoder pools over the whole window)
#1 = TSS masking model (window centered on the TSS, decoder pools over the TSS mask)
#2 = TSS mask as an input channel (window centered on the TSS, decoder pools over the whole window)
#epoch 16 for all three so the comparison is at matched training length
#(use #SBATCH --array=0-2 for these)
# outputs=("k562_decima_2" "k562_finetune_2" "k562_tssmask")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/22-39-32-007131/checkpoints/16-val_loss=0.84871.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
# )

#previous round (old decoder: softplus + log2 head). The 2026-03-11 simple-decoder checkpoints
#can no longer be loaded by the current TSSDecoder -- their key is output_transform.0.weight,
#while `simple=True` is now a bare nn.Linear (output_transform.weight):
#  k562_tss_simpledecoder     outputs/2026-03-11/15-12-02-454124/checkpoints/08-val_loss=2.25911.ckpt
#  k562_decima                outputs/2026-07-21/03-05-20-648050/checkpoints/22-val_loss=0.72138.ckpt
#  k562_tss_simpledecoder_sc  outputs/2026-03-11/15-11-46-799677/checkpoints/06-val_loss=2.01853.ckpt

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#-- run the task --
#no --load_data here: the dataset reopens the npz per item, so workers stream it lazily.
#if you switch to --load_data, drop --num_workers to 0 or every worker copies the array.
#single-output (TSSDecoder) models -- eval_tss.py builds a TSSDecoder directly and expects a
#single tensor out of the decoder. This is what the current round needs.
pixi run python -u eval_tss.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
  --split test \
  --batch_size 1 \
  --num_workers $WORKERS \

#two-output (TSSProfileDecoder) models -- same arguments, but eval_tss_profile.py, which also
#forces expression_data_path=None so the RNA-seq npz is never opened
# pixi run python -u eval_tss_profile.py \
#   -o "$OUTPUT" \
#   --ckpt_path "$CKPT" \
#   --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
#   --split test \
#   --batch_size 1 \
#   --num_workers $WORKERS \


#and here's how we could test in terminal
# profile model (two-output decoder), epoch 31 of job 6052571
# pixi run python -u eval_tss_profile.py \
#   --ckpt_path "/data1/lesliec/sarthak/caduceus/outputs/2026-08-05/12-48-43-803485/checkpoints/31-val_loss=1.47817.ckpt" \
#   -o smoke_k562_tss_sc_rna_profile2_ep31 \
#   --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
#   --split test --limit 8 --batch_size 4 --num_workers 1

#the older single-output checkpoints below run through eval_tss.py
# index 0 -- TSS-masked / simple decoder (bulk RNA json)
# pixi run python -u eval_tss.py \
#   --ckpt_path "/data1/lesliec/sarthak/caduceus/outputs/2026-03-11/15-12-02-454124/checkpoints/08-val_loss=2.25911.ckpt" \
#   -o smoke_k562_tss_simpledecoder \
#   --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
#   --split test --limit 8 --batch_size 4 --num_workers 1

# index 1 -- decima / gene-mask channel (sc cellranger json)
# pixi run python -u eval_tss.py \
#   --ckpt_path "/data1/lesliec/sarthak/caduceus/outputs/2026-07-21/03-05-20-648050/checkpoints/22-val_loss=0.72138.ckpt" \
#   -o smoke_k562_decima \
#   --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
#   --split test --limit 8 --batch_size 4 --num_workers 1