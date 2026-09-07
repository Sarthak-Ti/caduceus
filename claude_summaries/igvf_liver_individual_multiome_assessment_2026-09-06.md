# Individual-level liver multiome — data inventory and feasibility assessment
_2026-09-06 — read-only survey of `/data1/deyk/IGVF/igvf_lipids_atlas`, nothing implemented_

**Question asked:** if a model takes accessibility *and* sequence, can it predict variant effects
in individuals better than sequence alone, and can an individual's local chromatin stand in for
the trans effects that S2F models cannot see? What data exists to test this, and is it a good idea?

**Short answer:** the data for the *expression* half of the experiment is here and is unusually
clean — 39 donors of paired snRNA+snATAC liver multiome, already pseudobulked per donor. The
*variant* half is blocked: there are no donor genotypes anywhere on disk. The idea itself is
sound and matches where the field is moving, but the headline risk is leakage, not biology.

---

# Part 1 — What is actually on disk

## 1.1 Mohlke liver multiome (the primary resource)

`/data1/deyk/IGVF/igvf_lipids_atlas/data/mohlke_liver_multiome/`

GEO **GSE296875**. 10x Multiome — snRNA and snATAC measured in the *same nucleus* — on human
liver, 8 wells, donor-multiplexed and demultiplexed with both demuxlet and souporcell.

| | |
|---|---|
| Cells (post-QC, in metadata) | 68,398 |
| Confident singlets (demuxlet ∧ souporcell agree) | 66,657 |
| demuxlet singlet / souporcell doublet | 1,252 |
| demuxlet singlet / souporcell unassigned | 489 |
| **Donors** | **39** |
| Wells | 8 (1,954 – 12,582 cells each) |

Cell types (`celltype_dXRNA_res.0.25_harmony`):

| Cell type | Cells |
|---|---|
| Hepatocytes | 46,286 |
| LSEC | 7,379 |
| Mesenchymal | 4,781 |
| Kupffer | 4,335 |
| NK-T | 2,701 |
| Cholangiocytes | 1,886 |
| B cells | 1,030 |

The key point is that **the per-donor work is already done for hepatocytes**:

- `pseudobulk_hepatocyte_fragments_by_donor/` — all **39 donors**, each with
  `fragments.tsv.gz`, `.cpm.bedgraph`, `.cpm.filtered.bedgraph`, and a **`.cpm.bw` bigWig**.
- `pseudobulk_hep_rna_matrices_by_donor/` — the matching 39 `hep_raw_gex_matrix_donor_*.csv.gz`.

So donor-matched (ATAC track, expression vector) pairs exist today with no preprocessing.

Hepatocytes per donor are heavily skewed — 3,973 (donor623) down to 98 (donor374):

| Threshold | Donors passing |
|---|---|
| ≥ 800 hepatocytes | 26 |
| ≥ 500 hepatocytes | 29 |
| any | 39 |

Also present:
- `mohlke_liver_multiome_metadata.csv` — 55 per-cell columns: `demuxlet_sample_id`,
  `soc_sample_id`, `merged_status`, celltype calls, FRiP, TSS enrichment, nucleosome signal,
  ambient-RNA (dXbg) corrected counts, WNN weights.
- Per-celltype pseudobulk fragments for all 7 types (`pseudobulk_fragments_sorted/`), aggregated
  across donors. Note there are `_WRONG` sibling directories — use the plain / `_sorted` ones.
- Raw: 8 × `raw_feature_bc_matrix.h5`, 8 × `atac_fragments.tsv.gz` (+tbi), the 72 GB
  `GSE296875_RAW.tar`, and the 8.5 GB harmonized Seurat WNN object.
- `celltype_barcodes/` — barcode lists per cell type.
- `ldl_gene_peak_donor_var_fracs.csv` — **someone has already computed, per peak, the fraction
  of variance attributable to donor**. This is directly the quantity that motivates the project,
  and is worth reading before designing anything.
- `ldl_gene_sce2g_links.csv`, `..._overlap_finemap_gwas_chromhmm.csv`,
  `sce2g_donor_glmm_results/` (b_cells only so far).

## 1.2 scE2G run on the same data

`/data1/deyk/IGVF/igvf_lipids_atlas/scE2G/results/2025_1212_mohlke_liver_multiome/`

Complete per-celltype runs (hepatocytes finished): MACS2 peaks, Neighborhoods, Kendall
correlations, ARC, `Predictions/EnhancerPredictionsAllPutative.tsv.gz`, and
`multiome_powerlaw_v3/scE2G_predictions.tsv.gz`.

Hepatocyte run summary (from `scE2G_predictions_threshold0.177_stats.tsv`):

| | |
|---|---|
| Cells used | 20,000 (subsampled) |
| Fragments | 790,195,763 |
| UMIs | 213,116,420 |
| Enhancer elements | 31,214 |
| Genes with ≥1 enhancer | 9,617 |
| Enhancer–gene links | 48,457 |
| Mean enhancers per gene | 5.04 |
| Mean distance to TSS | 83.0 kb |
| Genes with active promoter | 14,363 |

Useful for two things: defining which windows count as "local chromatin" for a gene, and as the
ABC-style baseline that a joint model would have to beat.

## 1.3 Variant-effect readouts (liver context, no donor genotypes needed)

`/data1/deyk/IGVF/igvf_lipids_atlas/lipid_variant_screening/data/`

- `2025_12_04_igvf_lipidvariant_mpra.csv` — **40,252 variants**, UNC MPRA. Per-variant
  ref/alt input and output counts, log2FC, p/q, the 250 bp tested sequence itself, rsID,
  nearest gene, and ref/alt alleles. Directly usable as a variant-effect eval set.
- `2025_12_04_igvf_lipidvariant_baseediting.csv` — **2,492 variants**, base-editing screen with
  LDL-uptake `mu_Z_adj`, CRISPRi BEAN scores, HepG2 ChromBPNet ref/alt scores, CatBoost
  varACCESS scores, per-ancestry MAF/LD, and multiple fine-mapping PIPs (RSparsePro, SuSiE).
  Also carries a **`caQTL_Mohlke`** column — see §2.

## 1.4 Supporting / lower-priority

- `data/encode_FuncGen_allHumanLiver_data/` — 6,843 files, 2,036 experiments. Dominated by
  HepG2 (1,793 experiments). Primary liver tissue is only ~235 experiments (liver 115,
  right lobe 82, hepatocyte 33, left lobe 5), and assays are not matched per donor:
  DNase 21, ATAC 10, snATAC 19, snRNA 14, total RNA 17, polyA RNA 17, histone ChIP 77,
  TF ChIP 891 (nearly all HepG2), genotyping array 5.
- `UKBB_individual_level/` — **phenotypes only**: lipids, clinical biomarkers, statin use,
  Olink proteomics, diet, lifestyle, socio-economic. No genotypes.
- `derived_data/` + `ANNOTATIONS_hg38/` — ChromHMM "fluid vs stable" state calls, S-LDSC
  heritability results, statin-sensitive/stable variant lists, GTEx liver expression dynamics.
- `data/kushal/Liver_GxE_finemapping_results.txt` — GxE fine-mapping (e.g. HDL × alcohol
  frequency) with PIPs and per-variant coordinates.
- `data/karthik/` — fine-mapped GWAS variants per trait/tissue, ChromHMM state proportions,
  1000G common variant bed, pilot CRISPR screen Z-scores.

---

# Part 2 — The blocker: no genotypes

I searched `/data1/deyk` to depth 5 for `*.vcf*`, `*.bim`, `*.pgen`, `*genotyp*`, `*demuxlet*`,
`*souporcell*`. **Nothing for the Mohlke liver donors.** The only hits were unrelated
(pancreatic village demultiplexing, mouse eQTL VCFs, iPSC contraPC).

They must exist upstream: demuxlet requires a reference genotype panel, the metadata has a
populated `demuxlet_sample_id`, and the base-editing table has a `caQTL_Mohlke` column, which
means caQTLs were called on these donors. They are simply not in this directory.

Without per-donor genotypes you cannot construct personal sequences, so **the variant half of
the idea is not runnable today**. Two routes:

1. **Ask.** Karthik / Kushal / the Mohlke lab. Cheapest path, and it gets you genome-wide
   genotypes plus their existing caQTL calls as a validation target.
2. **Call genotypes from the ATAC yourself.** Each donor has 100–500 MB of gzipped hepatocyte
   fragments (790M fragments total across the cell type). That is ample depth to call common
   SNPs at accessible sites with cellSNP-lite / bcftools — which is essentially what souporcell
   already did to demultiplex. You only get genotypes in accessible and expressed regions, but
   that is precisely where local-sequence variant effects live, so the restriction costs less
   than it sounds. This is a real fallback, not a consolation prize.

---

# Part 3 — Is the idea good?

## 3.1 The premise is correct and well-documented

Two 2023 *Nature Genetics* papers established, independently, that sequence-only models fail at
exactly the task in question:

- **Sasse et al. 2023** — benchmarked on paired WGS + expression from **839 ROSMAP
  individuals**. Current models predict expression well *across genes* but poorly *across
  individuals*, and frequently get the **direction** of a variant's effect wrong. They trace
  this to insufficiently learned motif grammar.
  Nat Genet 55:2060–2064. [10.1038/s41588-023-01524-6](https://doi.org/10.1038/s41588-023-01524-6)

- **Huang et al. 2023** — four state-of-the-art models on paired personal genome + transcriptome
  data. Limited ability to explain inter-individual expression variation from cis-regulatory
  variants; models often fail on direction of effect.
  Nat Genet 55:2056–2059. [10.1038/s41588-023-01574-w](https://doi.org/10.1038/s41588-023-01574-w)

- **Karollus, Mauermeier & Gagneur 2023** — the complementary structural failure: Enformer
  captures promoter determinants but largely ignores distal enhancers, especially at medium-to-
  long range, and integrates long-range information far more weakly than its receptive field
  suggests.
  Genome Biol 24:56. [10.1186/s13059-023-02899-9](https://doi.org/10.1186/s13059-023-02899-9)

So the motivation is not speculative. Personal sequence explains little inter-individual
expression variance, and the models that do exist are weak on exactly the distal-and-personal
regime this project targets.

## 3.2 Prior art for the proposed fix

- **GenoME** — Wei J, Xue Y, Chai H, Gao YQ. *GenoME: a MoE-based generative model for
  individualized, multimodal prediction and perturbation of genomic profiles.* bioRxiv,
  2025-12-28. [10.64898/2025.12.28.696482](https://doi.org/10.64898/2025.12.28.696482)

  The closest published thing to the proposed architecture. Per the abstract: a Mixture-of-
  Experts generative model taking **DNA sequence plus cell-type-specific ATAC-seq** and
  predicting a unified profile spanning epigenomics, transcriptomics and chromatin architecture
  at bp-to-kb resolution. The claim most relevant here is that it "generalizes to predict the
  full regulatory landscape of **unseen or individualized cell types from a single ATAC-seq
  input**" — i.e. ATAC as the conditioning channel that carries cell/individual identity. It
  also ships an in-silico perturbation framework and reports beating Activity-by-Contact on
  enhancer–promoter connections.

  _Caveat on this citation:_ bioRxiv rate-limited (HTTP 429) on both PDF and full-text fetch, so
  **only the abstract was read**. Sarthak's reading of the paper is that it is largely
  ATAC-driven and does not genuinely fuse sequence with accessibility, which would conflict with
  the abstract's framing. Worth resolving from the full text before positioning against it —
  if the fusion is weak or absent, that is the gap this project occupies.

## 3.3 The two things to get right

**(a) Separate the two claims.** These are different experiments and should not be conflated:

1. *Local chromatin lets me predict expression in an individual that sequence alone cannot.*
2. *Local chromatin lets me predict variant effects in an individual.*

Claim 1 is clean and runnable now. Claim 2 is subtler, because measured ATAC at a locus already
**contains** the cis effect of variants at that locus — you are partly conditioning on the
answer. Any claim-2 result needs the variant's own neighbourhood ablated from the ATAC input, or
it proves nothing.

**(b) Leakage is the dominant threat, not data size.** Promoter and enhancer accessibility is
strongly correlated with expression. A model given both can score a high across-donor
correlation with sequence contributing exactly zero. That is still a *useful predictor*, but it
is not evidence the model understands regulatory sequence. Required controls, all cheap:

| Control | Tests |
|---|---|
| Sequence only (no ATAC) | the Sasse/Huang baseline; expected near-zero across donors |
| ATAC only (no sequence) | how much is pure chromatin readout |
| Full model | the actual claim — must beat *both* above |
| **Mismatched donor ATAC** | shuffle ATAC across donors; performance must collapse |
| ATAC masked in a ±N bp window around the variant/TSS | removes the circular path |

The mismatched-donor control is the single most informative one and costs nothing to run.

---

# Part 4 — The experiment I would run first

**Held-out-donor expression prediction in hepatocytes.** No genotypes required.

- **Input:** reference sequence around the TSS + *that donor's own* hepatocyte ATAC signal.
- **Target:** that donor's hepatocyte pseudobulk expression for that gene.
- **Split:** by donor (and check a gene-held-out variant too — the two splits answer different
  questions).
- **Metric:** per-gene Spearman *across donors*, which is the Sasse/Huang benchmark. Report the
  distribution over genes, not a pooled correlation — pooling is dominated by across-gene
  variance and will look great regardless.

This is the existing `slurm_scripts/finetune_joint_k562_tss_sc.sh` recipe with the accessibility
track varying **per donor** instead of per cell type. Architecturally nothing new is needed.

### Practical notes

- **Power.** 39 donors is small. SE of a per-gene Spearman is ≈ 1/√(n−3): **0.17 at n=39**,
  **0.21 at n=26**. Real effects are detectable; subtle ones are not. Plan to report the
  distribution shift across thousands of genes rather than per-gene significance.
- **Donor filtering.** Drop the donors under ~500 hepatocytes (374, 450, 724, 733, 783) — their
  pseudobulk is noise-dominated on both modalities. 26 donors at ≥800 cells is the safer set;
  29 at ≥500 if you need the power more than the cleanliness.
- **Zonation is the confounder to watch.** Hepatocytes are zonated and polyploid, and zonal
  composition varies by donor. That variation shows up in *both* ATAC and RNA, so "local
  chromatin predicts expression" will partly be "cell-state composition predicts expression."
  Either regress out composition (the metadata has the cluster assignments) or report it
  explicitly. Same for ambient RNA — the `dXbg_contamination` column exists for this.
- **Depth normalization.** Donors differ ~5× in fragment count. The bigWigs are CPM, which
  handles scale but not the mean-variance relationship; expect heteroscedasticity by donor depth.
- **Real advantage over doing this in GTEx:** RNA and ATAC come from the *same nuclei*. No
  cross-assay donor mismatch, no batch confound between the conditioning modality and the
  target. That is the reason to use this dataset specifically.

### Engineering

`src/dataloaders/datasets/tss_dataset.py` reads accessibility from an npz/zarr keyed by
chromosome (`open_data`), continuous, with the gene index coming from a TSS JSON carrying
`chrom / tss / counts / alt_tss / strand / gene_start / gene_end`. To adapt:

- 39 whole-genome donor tracks at bp resolution is too large to hold as npz. Extract TSS ± window
  slices from `donor*_pseudobulk.cpm.bw` into the same per-chromosome layout, or add a bigWig
  path to `TSSDataset` and slice lazily.
- The TSS JSON becomes per-donor (`counts` differ by donor) — or the loader gains a donor axis
  and the JSON carries a counts *vector*. The latter is less duplication.
- Sample identity is then (gene, donor) rather than (gene), so the epoch grows ~26–39×.
  `claude_summaries/npz_to_fold_zarr.py` is the closest existing conversion utility.

---

# Part 5 — Open questions / asks

1. **Get the donor genotype VCF** from the Mohlke lab via Karthik/Kushal. This unblocks the
   variant experiment and comes with their caQTL calls as an independent validation target.
   If unavailable, cost out cellSNP-lite on the per-donor fragment files.
2. **Read `ldl_gene_peak_donor_var_fracs.csv` first.** If donor explains almost no peak-level
   variance in this cohort, the premise is weaker here than the literature suggests and the
   experiment should be re-scoped before any training.
3. **Resolve the GenoME question** — does it actually fuse sequence and ATAC, or is it
   ATAC-dominated? Determines whether this is a replication or a genuine gap.
4. **Decide the scope of "local".** scE2G says mean enhancer–TSS distance is 83 kb and 5.04
   enhancers per gene. That sets the minimum useful context window, and it is larger than the
   current TSS-centered setup assumes.
5. The MPRA (40k variants) and base-editing (2.5k) sets are available as variant-effect evals
   that need no genotypes — worth using as a sanity check on the sequence arm regardless of
   whether the individual-level experiment goes ahead.

---

## References

Publication metadata for the three peer-reviewed references below was retrieved from **PubMed**.

- Sasse A, Ng B, Spiro AE, Tasaki S, Bennett DA, Gaiteri C, De Jager PL, Chikina M, Mostafavi S.
  Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence
  highlights shortcomings. *Nat Genet* 2023;55(12):2060–2064.
  [10.1038/s41588-023-01524-6](https://doi.org/10.1038/s41588-023-01524-6)
- Huang C, Shuai RW, Baokar P, Chung R, Rastogi R, Kathail P, Ioannidis NM. Personal
  transcriptome variation is poorly explained by current genomic deep learning models.
  *Nat Genet* 2023;55(12):2056–2059.
  [10.1038/s41588-023-01574-w](https://doi.org/10.1038/s41588-023-01574-w)
- Karollus A, Mauermeier T, Gagneur J. Current sequence-based models capture gene expression
  determinants in promoters but mostly ignore distal enhancers. *Genome Biol* 2023;24:56.
  [10.1186/s13059-023-02899-9](https://doi.org/10.1186/s13059-023-02899-9)
- Wei J, Xue Y, Chai H, Gao YQ. GenoME: a MoE-based generative model for individualized,
  multimodal prediction and perturbation of genomic profiles. *bioRxiv* 2025.12.28.696482.
  [10.64898/2025.12.28.696482](https://doi.org/10.64898/2025.12.28.696482)
  *(abstract only — full text was not retrievable, see §3.2)*
