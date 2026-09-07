#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_tss_dn
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-9

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- dinucleotide-shuffle E2G for the TSS models, the sequence counterpart of evaluate_e2g_tss.sh --
#new output names (_dinuc / _dinuc_acc) so the accessibility-only results from evaluate_e2g_tss.sh
#are never overwritten. Only the count head is scored, unperturbed (col 0) vs perturbed (cols 1..).
#
#NOTE: the count head is a plain regression head, so these live on the natural-log target scale and
#CAN BE NEGATIVE -- score as a log-space difference, not a ratio. --save_all_shuffles keeps every
#individual shuffle (one scalar per row, ~1MB total) so the mean is not baked in as a geometric mean
#in expression space; outputs[:, 1:].mean(axis=1) reproduces the averaged 2-column layout exactly.
#
#--time is 168h, not the 12h of evaluate_e2g_tss.sh: the accessibility-only runs took ~9h20 at 2
#sequences per element, and 11 shuffles means 12 sequences per element.
#
#five checkpoints x two perturbation arms:
#0/5 = decima_fixed_ep34 : TSSProfileDecoder, gene mask channel, Decima window, multinomial profile NLL (1e-6)
#1/6 = decima_2          : TSSDecoder (count only), gene mask channel, Decima window -- the matched
#                          count-only control for decima_fixed_ep34: identical dataset, window, mask
#                          channel and MSE count target, the profile head is the only difference
#2/7 = poisson_ep35      : TSSProfileDecoder, gene mask channel, Decima window, Poisson profile NLL (1.0)
#3/8 = tss_fixed_ep10    : TSSProfileDecoder, TSS mask channel, Decima window, multinomial profile NLL (1e-6)
#4/9 = tssmask           : TSSDecoder (count only), TSS mask channel, TSS-CENTERED window (TSS at
#                          262144, not 163840). Second profile-vs-count-only contrast, for the TSS
#                          mask channel -- but the window differs from tss_fixed_ep10, so that pair
#                          is confounded by geometry; it also gives the TSS-centered vs Decima
#                          window contrast against decima_2. Its in-context set is 3957 elements
#                          rather than the 3914 of the other four, so restrict to the intersection
#                          (in_context AND) before pooling metrics across all five.
#tasks 0-4 shuffle the element only, tasks 5-9 shuffle AND knock down accessibility.
outputs=( \
  "k562_tss_sc_rna_profile_decima_fixed_ep34_dinuc" \
  "k562_decima_2_dinuc" \
  "k562_tss_sc_rna_poisson_ep35_dinuc" \
  "k562_tss_sc_rna_profile_tss_fixed_ep10_dinuc" \
  "k562_tssmask_dinuc" \
  "k562_tss_sc_rna_profile_decima_fixed_ep34_dinuc_acc" \
  "k562_decima_2_dinuc_acc" \
  "k562_tss_sc_rna_poisson_ep35_dinuc_acc" \
  "k562_tss_sc_rna_profile_tss_fixed_ep10_dinuc_acc" \
  "k562_tssmask_dinuc_acc" \
)

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
)

perturbations=("shuffle" "shuffle" "shuffle" "shuffle" "shuffle" \
               "both_perturbations" "both_perturbations" "both_perturbations" \
               "both_perturbations" "both_perturbations")

#matches evaluate_e2g_striped_dn.sh so the shuffle noise floor is comparable across model families
N_SHUFFLES=11

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
PERTURBATION=${perturbations[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, perturbation=$PERTURBATION, n_shuffles=$N_SHUFFLES"
#-- run the task --
#--scale_factor and --dist_additional_mask are inert for perturbation=shuffle, only used by
#both_perturbations. --dist_additional_shuffle matches --dist_additional_mask so the sequence and
#accessibility perturbations cover the identical interval in the joint arm.
#--load_data still only loads the accessibility npz: e2g_tss_profile_dn.py overrides the profile
#model's expression_data_path to None, so the 12G RNA-seq npz is never opened
pixi run python -u e2g_tss_profile_dn.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --perturbation "$PERTURBATION" \
  --n_shuffles "$N_SHUFFLES" \
  --save_all_shuffles \
  --dist_additional_shuffle 100 \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --seed 0 \
  --batch_size 2 \
  --load_data \

#the five checkpoints, for reference (see the table in the notes for why these five):
#  decima_fixed_ep34  outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt
#  decima_2           outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt
#  poisson_ep35       outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt
#  tss_fixed_ep10     outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt
#  tssmask            outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt
#
#not run here:
#  k562_finetune_2 (2026-07-27) -- TSS-centered window AND no appended mask channel AND
#    pool_region='tss', so it varies three things at once against everything else here
#  k562_tss_sc_rna_profile2_ep31 (2026-08-05) -- superseded round, trained on the un-rescaled
#    K562_rnaseq_stranded.npz
#  k562_decima (2026-07-21) -- loads strict-clean but predates the TSSDecoder head change
#    (trailing Softplus + log2 removed); confirm with eval_tss.py before trusting it
#  k562_tss_simpledecoder / _sc (2026-03-11) -- simple=True went from Sequential(Linear, Softplus)
#    to a bare Linear, so these no longer load strict at all
