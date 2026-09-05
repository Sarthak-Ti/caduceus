# Decima-style RNA model — dataset handoff

**Goal:** Adapt a sequence+ATAC → expression model to a Decima-style approach for a
*single cell type* (K562). Predicting one expression value per gene overfit badly
(enhancer–gene links collapsed) with the old TSS-centered / TSS-region-averaging setup.
Hypothesis: a wide window + gene mask + ATAC input can recover regulatory signal at one
cell type (ATAC substitutes for Decima's cross-cell-type contrast).

**Downstream work still needed:** encoder, decoder, task head, loss.

## Dataset: `src/dataloaders/datasets/tss_dataset.py` (`TSSDataset`)

Returns two tuples per gene (one gene = one sample):

- **`outputs1 = (seq, targets)`**
  - `seq`: `(N, L)`, N=5 one-hot (A,C,G,T,N), or 6 if `mlm` masking on. `L = length`.
  - `targets`: ATAC, `(n_atac_tracks, L_out)`, `L_out = (L − 2·crop_output) / pool`.
    May be `acc_mlm`-masked (pretraining).
- **`outputs2 = (seq_unmask, acc_umask, counts, tss_mask, gene_mask, strand, expression)`**
  - `counts`: scalar tensor = **natural log(1+CPM)**, single value per gene (finetuning target).
  - `tss_mask`: `(L_out,)` binary, TSS neighborhoods.
  - `gene_mask`: `(L_out,)` binary, **gene body of the gene of interest only** (Decima's 5th track).
  - `strand`: scalar `+1 / −1`.
  - `expression`: `(n_rna_tracks, L_out)` per-bp RNA-seq coverage, aligned to `targets`,
    OR `torch.empty(0)` if `expression_data_path` unset.
  - `seq_unmask` / `acc_umask`: empty unless MLM / acc-MLM masking on.

## Key config for Decima-style placement

`upstream=163840`, `length=524288`, `rc_strand=True`.
Places the TSS `upstream` bp from the left edge with the gene body extending right,
strand-normalized (minus-strand genes reverse-complemented → promoter always on the left).
`upstream=None` = old TSS-centered behavior.

## Alignment guarantees

`seq`, `targets`, `expression`, `tss_mask`, `gene_mask` all share the same window,
padding, RC flip, crop, and pool. `targets` and `expression` go through the *same*
`_load_aligned()` method → aligned bin-for-bin. All JSON coords are **0-based, end-exclusive**.
(`seq` itself is not cropped/pooled, so per-bp identity with `seq` holds only at
`pool=1, crop=0`; otherwise everything is binned in lockstep.)

Prerequisite: the expression zarr must have the same per-chromosome lengths as the
genome/ATAC (same invariant `self.data` already relies on).

## Data source: `/data1/lesliec/sarthak/data/DE_danwei/build_rna_info.py`

Builds `k562_sc_rna_info.json` (`--mode sc`, default; also `bulk` / `both`).
Per gene: `chrom, tss, alt_tss, strand, gene_start, gene_end, counts, split`.
`counts` = **ln(1+CPM)** (single pseudobulk = sum of all K562 cells, CPM-to-1M).
Splits from Borzoi folds.

## Downstream decisions to make (encoder / decoder / head / loss)

1. **Gene mask usage:** currently returned separately in `outputs2`, **not** concatenated
   onto `seq`. Decima concatenates it as an input channel *and* mean-pools embeddings over
   the sequence, using the mask to select the gene. The load-bearing use is at the
   **readout/head** (gene-specific masked pooling); adding it as an input channel is a
   secondary soft hint.
2. **ATAC input:** keep seq+ATAC as encoder inputs (the differentiator from Decima; the
   hypothesis for why it can work at one cell type).
3. **Head / target — two options:**
   - (a) predict scalar `counts` via masked-mean-pool of embeddings over `gene_mask` → 1 value/gene; or
   - (b) predict the per-bp `expression` track (Poisson, CAGE-style) and aggregate over the gene body.
4. **Loss:** for scalar `counts`, Decima-style = MSE + a cross-gene Pearson-correlation term
   (only one cell type, so correlation is across genes). For per-bp expression, Poisson.
   Overfitting is the central problem to watch.
