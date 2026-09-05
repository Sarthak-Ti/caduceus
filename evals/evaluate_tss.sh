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

#every array below is a TSSProfileDecoder unless noted: the decoder returns TWO outputs (per-bp
#profile logits + scalar count) and eval_tss_profile.py scores the COUNT head only. That is the
#same log-scale quantity a single-output TSSDecoder predicts, so all rounds in this file are
#directly comparable -- and the profile head is never scored here, whatever it was trained with.
#current round -- the three runs trained on the rescaled RNA-seq npz
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
outputs=("k562_tss_sc_rna_poisson_ep35" "k562_tss_sc_rna_profile_decima_fixed_ep34" "k562_tss_sc_rna_profile_tss_fixed_ep10")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
)

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
pixi run python -u eval_tss_profile.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --output_dir /data1/lesliec/sarthak/data/joint_playground/model_out \
  --split test \
  --batch_size 1 \
  --num_workers $WORKERS \

#old single-output (TSSDecoder) models -- same arguments, but eval_tss.py, which builds a
#TSSDecoder directly and expects a single tensor out of the decoder
# pixi run python -u eval_tss.py \
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