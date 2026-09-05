#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-2

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#same checkpoints as evaluate_tss.sh, so the E2G and test-set numbers line up per model.
#NOTE: the count head is a plain regression head, so these E2G predictions live on the
#natural-log target scale and CAN BE NEGATIVE -- score the perturbation as a log-space
#difference (col 1 - col 0), not a ratio. Holds for every round in this file.
#
#current round -- the three runs trained on the rescaled RNA-seq npz
#(K562_rnaseq_stranded_fixed.npz). All three are TSSProfileDecoder: the decoder returns
#(profile, counts) and e2g_tss_profile.py scores the COUNT head, unperturbed (col 0) and
#accessibility-knocked-down (col 1). It also overrides expression_data_path to None, so the npz
#that differs between these and the earlier profile2 round is never opened -- the count-head
#numbers stay comparable across every round in this file.
#0 = poisson     : gene mask channel, Decima window (TSS at 163840), Poisson profile NLL
#                  (weight 1.0), decoder profile_softplus, count_region=all
#1 = decima_fixed: gene mask channel, Decima window, multinomial profile NLL (weight 1e-6)
#2 = tss_fixed   : TSS mask channel, TSS-centered window, multinomial profile NLL (weight 1e-6)
#each run at its own lowest val/loss epoch (35 / 34 / 10) rather than a matched epoch -- see the
#header comment in evaluate_tss.sh for why val/loss is not comparable ACROSS these three, and for
#the log-vs-filename epoch off-by-one.
outputs=("k562_tss_sc_rna_poisson_ep35" "k562_tss_sc_rna_profile_decima_fixed_ep34" "k562_tss_sc_rna_profile_tss_fixed_ep10")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
)

#previous round -- the same profile decoder trained on the un-rescaled npz
#(job 6052571, k562_tss_sc_rna_profile2), epoch 31. TSS mask as an input channel, asymmetric
#window with the TSS at 163840, count_region=all.
#(use #SBATCH --array=0-0 for this one)
# outputs=("k562_tss_sc_rna_profile2_ep31")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-08-05/12-48-43-803485/checkpoints/31-val_loss=1.47817.ckpt" \
# )

#previous round -- single-output TSSDecoder models, run with e2g_tss.py (commented command below)
#0 = decima style model (gene mask as an input channel, asymmetric window with the TSS at
#    163840, decoder pools over the whole window)
#1 = TSS masking model (window centered on the TSS, decoder pools over the TSS mask)
#2 = TSS mask as an input channel (window centered on the TSS, decoder pools over the whole window)
#epoch 16 for all three so the comparison is at matched training length.
#(use #SBATCH --array=0-2 for these)
# outputs=("k562_decima_2" "k562_finetune_2" "k562_tssmask")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/22-39-32-007131/checkpoints/16-val_loss=0.84871.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
# )

#previous round (old decoder: softplus + log2 head):
#  k562_tss_simpledecoder     outputs/2026-03-11/15-12-02-454124/checkpoints/08-val_loss=2.25911.ckpt
#  k562_decima                outputs/2026-07-21/03-05-20-648050/checkpoints/22-val_loss=0.72138.ckpt
#  k562_tss_simpledecoder_sc  outputs/2026-03-11/15-11-46-799677/checkpoints/06-val_loss=2.01853.ckpt

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#-- run the task --
#--load_data still only loads the accessibility npz: e2g_tss_profile.py overrides the profile
#model's expression_data_path to None, so the 12G RNA-seq npz is never opened
pixi run python -u e2g_tss_profile.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --load_data \

#old single-output (TSSDecoder) models -- same arguments, but e2g_tss.py, which builds a
#TSSDecoder directly and expects a single tensor out of the decoder
# pixi run python -u e2g_tss.py \
#   -o "$OUTPUT" \
#   --ckpt_path "$CKPT" \
#   --scale_factor 100 \
#   --dist_additional_mask 100 \
#   --load_data \
