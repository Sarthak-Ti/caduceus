# Handoff: K562 enhancer specificity for linking; GM12878 eQTL work paused

Updated: 2026-09-08. Repository: `/data1/lesliec/sarthak/caduceus`.

## Current scope — read first

- **eQTL work is paused at the user's request.** Results are inconclusive, not proof of no specificity effect. Preserve the latest continuous analysis and median-split notebook; do not resume without a new request.
- **Implemented: enhancer specificity alone for K562 enhancer–gene linking** in `code_test/EP_linking10.ipynb`, transferring the dsQTL DNase **peak-breadth** method. Do not substitute the eQTL mean-CAGE or ratio definitions.
- Keep eQTL and linking as separate tasks in this one handoff. The joint enhancer × gene analysis remains a possible later extension, not part of the initial linking implementation.
- Reuse existing genome-wide DNase bin calls. Switching to K562 does not require rerunning per-track peak calling or reading the source Zarr. Build separate derived enhancer annotations; preserve the dsQTL cache.
- If a source-cache rebuild or additional source extraction becomes necessary, explain the need to the user before running it.
- Preserve existing notebooks/results; put linking work in a successor to the current linking notebook (check the next available number). Use concise method bullets and cache-only reruns.
- The poster's nine eQTL arms apply only to eQTL. Current `EP_linking9.ipynb` supplies the linking models, scoring and context masks.

## Source of truth: poster eQTL code

`/data1/lesliec/sarthak/caduceus/code_test/IGVF_meeting_figures.ipynb`

Read code cells, especially `load_eqtl`, `abs_logfc`, `eqtl_table`, and `panel_D`. Cell indices at inspection: loader 14, evaluation 15, Panel D 21; locate by function name because cell indices can change.

Prediction/input directory:

`/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/`

Use exactly these nine arms unless the user changes the requested set:

| Model | Prediction filename (relative to directory above) | bp per prediction bin |
|---|---|---:|
| cis-Mamba (1 cell type) | `nomlm_maskonly.npy` | 128 |
| cis-Mamba, no pretraining | `nopretrain.npy` | 128 |
| cis-Mamba (8 cell types) | `immune.npy` | 128 |
| Enformer (pretrained) | `enformer_eqtl.npy` | 128 |
| Borzoi (pretrained) | `borzoi_eqtl.npy` | 32 |
| Borzoi LoRA (GM12878) | `borzoi_eqtl_lorav2.npy` | 32 |
| Borzoi full FT (GM12878) | `borzoi_eqtl_trunk.npy` | 32 |
| Borzoi from scratch (GM12878) | `borzoi_eqtl_scratch.npy` | 32 |
| AlphaGenome | `alphagenome_eqtl.zarr` | 1 |

Do not substitute the 12-arm dsQTL list or add ChromBPNet/Enformer-retrained arms just because they occur elsewhere.

Poster scoring / cohort:

- 1,331 original benchmark rows; retain their original row indices explicitly.
- Input arrays have shape `(N, profile_bins, 2)` with ref/alt in the last dimension.
- Crop the central **2,048 bp**: `nb = 2048 // bp`; `start = full.shape[1] // 2 - nb // 2`; `full[:, start:start+nb, :]`.
- Score = `abs(log2(mean(ref) / mean(alt)))` over that crop.
- Poster marks an arm unscored when **both** ref and alt sums are zero; intersect scorable masks across all nine arms for the primary comparison.
- Keep true zero effects. Do not filter on score != 0.
- Check nonfinite values / one-sided zero means explicitly. If a stricter validity rule changes the poster cohort, report the difference rather than silently claiming exact reproduction.
- Primary metric: `average_precision_score` (AP / AUPRC), not trapezoidal PR-curve area.
- Poster Panel D describes 1,023 common-support rows; recompute and validate this reference count rather than hardcoding it.
- Existing distance categories: `<1 kb`, `1–10 kb`, `10–262,144 bp`, using distance from variant to the target gene TSS. Keep distance as a diagnostic/confounder when stratifying specificity.
- Poster notes accessibility was held fixed between ref and alt; preserve the established prediction/scoring setup.

### Is `eQTL_benchmark6.ipynb` different?

`/data1/lesliec/sarthak/caduceus/code_test/eQTL_benchmark6.ipynb`

Inspected both notebooks' code on 2026-09-07:

- Same nine eQTL model labels, prediction filenames, bin sizes, benchmark input directory, central 2,048-bp crop, absolute-logFC scoring, and common-support construction.
- `eQTL_benchmark6` has fewer explicit assertions in its loader and focuses on model overlap/top-ranked variants; the poster presents distance-stratified AP.
- `eQTL_benchmark6` caches collapsed common-support matrices in:
  `/data1/lesliec/sarthak/caduceus/code_test/.upset_cache.npz`
- Its `eq_X`, `eq_y`, `eq_labels`, `eq_fam` do **not** preserve the original benchmark row indices or full common-support mask. `eq_labels` are model names.
- **Do not join that reduced matrix positionally to all 1,331 benchmark rows.** Reconstruct/validate the original mask and row mapping, or rebuild a dedicated score cache with explicit row IDs and source provenance.
- Retain `IGVF_meeting_figures.ipynb` as source of truth if these notebooks later diverge.

## Core benchmark / coordinate inputs

Under `/data1/lesliec/sarthak/data/joint_playground/eQTL/EPCOTv2_LCLs/`:

- `LCLs.txt`: read with the poster's whitespace-separated schema:
  `label, qtl_idx, gene_idx, chrom, gene_start, gene_end, strand, qtl_loc, ref, alt, sign_target`.
- `genes.pickle`: poster uses `sorted(pickle.load(...).keys())` to map `gene_idx` to gene ID. Preserve this exact ordering.
- `ensemblTSS.txt`: gene ID and TSS position.
- Poster computes distance as `abs(tss[genes[gene_idx]] - (qtl_loc - 1))`.
- Verify genome build and TSS coordinate convention against prediction-generation code before any overlaps. Do not reuse the dsQTL-specific hg19-to-hg38 BIM mapping by assumption.
- Treat benchmark rows as variant–gene pairs; repeated variants and repeated genes must retain their separate pair IDs.

## Expression source — RESOLVED: Borzoi, not Enformer

**Decision made and implemented.** The measured-CAGE readout uses
`/data1/lesliec/sarthak/data/borzoi/borzoi.zarr` with
`/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi.bed` and `targets_borzoi.txt`.

Why Borzoi over the Enformer store described below:

- Its CAGE panel is the **same 638 biosamples** as Enformer's, just split by strand (1,276 columns),
  so nothing is lost — and strand-split lets the sense strand be selected per gene, which matters
  because the antisense track at a promoter measures divergent transcription, not the gene.
- Chunks are `(1, 6144, 7611)` — **one span per chunk**, ~55 MB — so per-locus reads are cheap.
  Enformer's `(32, 896, 5313)` chunks pull 32 spans (~305 MB) for a single locus.
- 32 bp bins rather than 128 bp.
- The BED interval **is** the measured interval (196,608 bp = 6144 x 32), so unlike Enformer there is
  no input-vs-output crop to undo.

Row order: within a fold, that fold's order in `sequences_borzoi.bed`.
`sequences_borzoi_fold3-4.bed` independently records (fold, within-fold index) and is used to verify
the mapping rather than define it.

### Original Enformer notes, kept for reference only



These are historical alternative-source notes. The implemented expression cache uses Borzoi; do not start a second extraction from Enformer for this task.

```text
/data1/lesliec/sarthak/data/enformer/data/labels.zarr
/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed
/data1/lesliec/sarthak/data/enformer/data/targets.txt
```

Observed layout:

- BED columns: chromosome, input start, input end, split (`train`/other splits).
- Input spans are 131,072 bp.
- `labels.zarr/test` shape `(1937, 896, 5313)`; chunks `(32, 896, 5313)`; dtype float32.
- 896 output bins × 128 bp = 114,688 measured bp, centered within the input.
- Therefore measured start = input start + **8,192 bp**; measured end = input end − 8,192 bp.
- For a zero-based position `p` inside the measured interval: `bin = (p - measured_start) // 128`.
- Confirm the row order separately for each split by filtering the BED while preserving its original order. Verify corresponding array lengths and any ordering evidence in data-generation code.
- Target index **5110** is GM12878 CAGE in the checked `targets.txt`.
- Existing comparison code treats CAGE tracks as beginning at index 4675; verify using the target descriptions, not a blind index range.
- CAGE measures transcription initiation at promoters; do not label it whole-gene RNA abundance without justification.

## Reusable expression-extraction principles (reference; eQTL paused)

- Preserve original pair rows, gene IDs, labels, scores and explicit support masks; do not positionally join a reduced score cache to the full benchmark.
- Measured CAGE is a promoter-initiation proxy, not whole-gene RNA abundance. Processed signal does not justify a raw-count Poisson test automatically.
- Resolve build, strand, TSS and window definitions before a join. Annotated promoter unions can differ from a benchmark's selected TSS.
- Map windows to actual measured spans; group reads by source chunk. Preserve chosen span, original row ID, bin boundaries and actual bp summed.
- Full coverage is required for the current gene readout. Missing or partial coverage is not silence. Inspect gene IDs and coverage before claiming a new gene set is cached.
- Thresholds and reference populations must be independent of benchmark labels and model predictions. Repeated pairs must not reweight a feature-level reference distribution.
- Compare models on identical pairs, report prevalence and exclusions, and account for shared genes/elements/loci. AP minus prevalence is descriptive, not a class-imbalance correction.

## Reusable dsQTL implementation and cache

All code under `/data1/lesliec/sarthak/caduceus/code_test/`:

```text
dsQTL_benchmark8.ipynb           # complete methods, executed results, original + clean plots
dsQTL_benchmark8_pipeline.py     # pooling, histogram BH, binary calls, breadth, merged peaks
dsQTL_benchmark8_analysis.py     # mapping, scoring, stratified metrics, paired bootstrap
run_dsQTL_benchmark8.py          # notebook execution with checkpointed outputs/status
run_dsQTL_benchmark8.slurm       # CPU batch template
test_dsqtl_benchmark8.py         # numerical, interval, matrix and analysis checks
dsQTL_benchmark8_cache/README.md # cache schema and rerun instructions
```

Useful functions:

- Pipeline: `pool_chrom`, `bh_count_threshold`, `call_peaks`, `build_cache`.
- Analysis: `annotate_variants`, `stratified_metrics`, `variant_comparisons`.
- The dsQTL `map_variants` and model list are benchmark-specific; adapt rather than blindly reuse.
- Clean plot code is an additional cell in `dsQTL_benchmark8.ipynb`, not in the analysis helper export.

Cache root:

`/data1/lesliec/sarthak/caduceus/code_test/dsQTL_benchmark8_cache/`

- `tracks.tsv`: row metadata for all 674 DNase tracks.
- `gm12878_bins.tsv.gz`, `gm12878_merged.tsv.gz`: hg38 coordinates, breadth, specificity labels and peak IDs.
- `<chrom>/coverage.npy`: float32 track × all 128-bp bins.
- `<chrom>/tracks_x_peak_bins.npy`: uint8 track × union peak-bin calls; genomic bin indices in `peak_bin_indices.npy`.
- `<chrom>/calls.packed.npy`: all-bin calls packed along bins, little-endian bit order.
- `<chrom>/tracks_x_merged_fraction.npy`, `tracks_x_merged_presence.npy`: track × merged GM12878 peak; column index is chromosome-local `merged_index` in the merged table.
- `coverage_manifest.json`, `track_thresholds.tsv`, `peaks.done.json`: provenance and completed peak-call settings.
- `variant_scores_and_specificity.tsv.gz`, `stratified_metrics.tsv`, `paired_model_differences.tsv`: examples of benchmark exports.

Original DNase source and track metadata:

```text
/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_reprocessed.zarr
/data1/lesliec/sarthak/data/DK_zarr/targets_DNase_borzoi.txt
```

Completed dsQTL method, for context (not instructions to impose these thresholds on expression):

- Sum 128-bp bins on hg38 chr1–22/X/Y; floor only for the approximate Poisson count test.
- Per-track global lambda; histogram-based BH at nominal 0.01 over all genome bins.
- Source was normalized to 100M reads **and subsequently scaled/clipped/adjusted**. This is an exploratory processed-coverage approximation, not calibrated raw-read FDR or a MACS2 run.
- Source omitted 180/3,102 chunks; use declared zero-fill semantics and record omitted indices.
- GM12878 indices 12 and 69; >=30% support means either track is sufficient.
- High specificity: breadth <=108 tracks for bins; <=68 for merged peaks. These are genome-wide medians among GM12878 peaks, not exclusivity thresholds.
- Merged peaks join adjacent GM12878-positive bins; a track is present if >=50% of constituent bins are called.
- Genome-wide cache and notebook completed successfully in about 61 minutes in job 11796379; routine cached analysis took about a minute.
- Exact CPU float16-to-float32 conversion with PyTorch avoided a large NumPy conversion bottleneck. This detail matters only if reading comparable float16 source data; Enformer labels checked here are float32.

## Enhancer–gene linking: K562 benchmark and current models

Source of truth: `/data1/lesliec/sarthak/caduceus/code_test/EP_linking9.ipynb`.
Read its later multi-model section (`MODELS`, `load_slab`, `score_slab`, `summary2`), not just the early three-run Borzoi section. Cell numbers may change.

Inputs:

```text
/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv
/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed
/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json
/data1/lesliec/sarthak/data/joint_playground/e2g/
```

- CRISPR TSV: 10,412 rows, all `CellType=K562`; the mapped benchmark has 10,280 pairs and 463 `Regulated` positives.
- Reproduce the exact symbol → Ensembl filtering/order before attaching prediction arrays. Preserve original TSV row IDs as well as mapped-array row IDs.
- Keep the existing positive/negative pair universe and labels. Accessibility annotates tested elements; it does not establish whether a link is functional.
- Read candidate construction and experimental metadata before interpreting negatives or independence. A nonregulated tested pair is not automatically an inactive enhancer.

Current model configurations and filename bases (under `e2g/`; `.npy` unless stated):

| Configuration | Arms / filename bases | Own in-context pairs |
|---|---|---:|
| ours-borzoi | `acc`: `k562_stranded_borzoi_cage_tss_centered`; `shuffle`: same + `_dinuc`; `shuffle+acc`: same + `_dinuc_acc` | 3,950 |
| ours-enformer | `acc`: `k562_unstranded_enformer_tss_centered`; `shuffle`: same + `_dinuc`; `shuffle+acc`: same + `_dinuc_acc` | 3,950 |
| ours-striped | `acc`: `k562_striped_tss_centered`; `shuffle`: same + `_dinuc`; `shuffle+acc`: same + `_dinuc_acc` | 9,732 |
| Borzoi | `shuffle`: `k562_borzoi_cage_tss_centered_dinuc` | 3,950 |
| Enformer | `shuffle`: `k562_enformer_e2g_tss_centered_dinuc` | 1,763 |
| AlphaGenome | `shuffle`: `k562_alphagenome_e2g_tss_centered_dinuc.zarr` | 6,624 |

There are **six configurations and twelve model–perturbation combinations**. A notebook print string saying “15 runs” is stale; derive the count from `MODELS`.

### Scoring and comparison cohorts

- Preserve `score_slab`: gene-matching strand where available, signed log2 altered/original ratio with pseudocount `1e-2`, and the current central TSS window (`HALF_BP=576`, converted to each model's bins). Use absolute scores for AP. Preserve the existing bin rounding; do not claim identical actual bp widths across resolutions.
- Use **`average_precision_score`**, as in the later multi-model section. Earlier cells use trapezoidal PR area, retained there only for comparison. Do not mix those metrics or reuse hardcoded full-benchmark bars for new strata.
- Model-specific `in_context` masks differ. Arms within a model must share masks; all models do not share their own-context cohort.
- **All-model common context: 1,763 pairs.** Derive actual positive counts and annotation losses from this mask.
- **262-kb subset: 3,950 pairs / 425 positives / 1,184 target genes**, prevalence about 0.108. This applies to the eligible longer-context models, not pretrained Enformer. It is not the universal cohort.
- Compare models on the same supported rows within each stratum. Show all-model common context and, if useful, the explicitly labeled 262-kb cohort for models that support it. Preserve out-of-context status instead of interpreting its zero placeholder as a scored negative.
- Inspect `shuffled` sidecars and AlphaGenome `done` flags using the existing loader; verify perturbation completion and report failures. Keep original masks and any additional validity masks separately.

## Linking implementation specification — implemented in EP_linking10

### 1. Reuse the DNase cache and define K562 eligibility

- Read `dsQTL_benchmark8_cache/tracks.tsv` and cached provenance. Use the existing all-track bin calls; do **not** rerun `call_peaks` with changed GM/K562 settings against the dsQTL output directory.
- Primary K562 anchor: exact description **`DNASE:K562`**, indices **121, 122, 123, 625**. Verify indices against metadata at runtime.
- Indices **33, 34, 35** are vorinostat-treated K562, G2 phase and G1 phase. Do not treat all seven name matches as equivalent untreated replicates. Altered-condition anchor inclusion can be a separately named sensitivity analysis.
- Retain all **674 tracks** in the primary breadth panel to match dsQTL, including the altered-condition tracks as panel entries, not untreated-anchor replicates. This counts tracks, not unique cell types; unequal sampling/replication remains a limitation.
- Freeze settings before model/label inspection. Proposed dsQTL-consistent defaults: track present when ≥50% of enhancer bp overlaps called bins; K562 active when ≥30% of the four untreated tracks are present, i.e. **at least two of four**.

### 2. Annotate unique benchmark enhancer intervals

- Verify hg38, chromosome names, enhancer coordinate columns and zero-based half-open conventions against the prediction-generation inputs. Do not use dsQTL variant liftover or eQTL gene coordinates by assumption.
- Build a unique enhancer table from the mapped linking benchmark before per-model support filtering. Preserve enhancer IDs and the exact benchmark interval; do not replace it with an entire overlapping GM12878 peak or a nearest peak.
- For enhancer `[start,end)` and track t, compute `sum(overlap_bp(bin, enhancer) * peak_call[t,bin]) / (end-start)`. Weight partial edge bins by actual overlap. Require full representation in the cached bin domain; out-of-domain/terminal uncovered bp is missing, not zero.
- Read `<chrom>/calls.packed.npy` (little-endian; ignore padding past real chromosome bins) or `tracks_x_peak_bins.npy` with `peak_bin_indices.npy`. A bin absent from the union has no track calls, provided it is inside the cached domain.
- Distinguish valid stored zeros from absent genomic coverage. Retain the source's documented sparse zero-fill provenance; it is not new evidence of measured biological silence.
- Cache track × enhancer overlap fractions and binary presence. Breadth is the sum of track presences. Reuse optional continuous coverage from `<chrom>/coverage.npy` for accessibility-strength diagnostics.
- **Reference median:** unique fully covered, K562-active enhancers in the mapped benchmark, counted once per enhancer, before per-model context filtering. Breadth ≤ median = higher specificity; > median = lower specificity. Keep ties together. Label this **benchmark-relative**, not genome-wide or K562-exclusive.
- Compute a new enhancer median. The dsQTL 108-bin and 68-merged-peak cutoffs do not transfer to these intervals. Global per-track peak-call thresholds **do** transfer under the unchanged cached calling method.
- Keep inactive, missing and invalid-coordinate enhancers separate. Join annotations back onto every original mapped pair without relabeling links. The annotation step should need no source Zarr reads.

### 3. Evaluate, inspect confounding and retain uncertainty

- Reproduce current unstratified scores/AP before stratifying. Cache row IDs, pair labels, all model scores, perturbation arms and context/validity masks with provenance.
- Report original → supported → covered → K562-active counts, then each specificity group: pairs, positives/prevalence, unique enhancers, unique genes, and positive-bearing enhancers/genes.
- Inspect enhancer length, K562 signal strength, breadth and enhancer–TSS distance jointly. Check their association and group overlap. This diagnoses potential confounding; a correlation is not proof of its cause or a reason to stop automatically.
- Do not transfer the GM12878 promoter-CAGE correlation or expression quantiles to K562 enhancer accessibility. These are different measurements and features.
- Preserve AP as the main ranking metric, with AUROC and prevalence if useful. AP minus prevalence is optional descriptive output, not adjustment for different candidate difficulty.
- Inspect shared enhancer, target-gene and genomic-locus dependencies before choosing grouped resampling. Use paired draws across models; do not assume independent pairs or independent strata if genes/loci cross the strata.
- Compare model gaps directly with paired uncertainty. If claiming specificity-dependent model advantage, estimate the change in that gap between groups, not merely a difference in bar heights.
- Keep arm contrasts explicit: sequence-only models have `shuffle`; joint models also have `acc` and `shuffle+acc`. Do not attribute a perturbation-arm change solely to model architecture.
- Plot lower specificity left, higher specificity right, consistent colors/order, numeric scores and counts. Preserve diagnostic uncertainty plots; clean presentation plots can omit error bars/baselines. Label a truncated axis and never hide values below it.
- Save per-enhancer matrices, per-pair annotations/scores, counts, thresholds, metrics and uncertainty in a separate linking output directory. Make subsequent regrouping and plotting cache-only.
- Add checks for interval-edge weighting, repeated enhancer joins, missing status, row alignment, unchanged labels/scores, and identical comparison cohorts. Deliver in a successor linking notebook with brief documentation.

Suggested initial pair export:

`original_row_id, mapped_row_id, pair_id, enhancer_id, chrom, enhancer_start, enhancer_end, gene_id, label, enhancer_breadth, enhancer_specificity, enhancer_K562_status, enhancer_mapping_status, distance_to_TSS`, plus each model/arm score and its support/validity masks.

### Scope and power limits

- Begin with enhancer specificity alone. The available positives may make this more informative than the old extreme eQTL groups, but **do not assume 425 positives survive annotation or split evenly**.
- A median across unique enhancers does not balance pair counts, positive counts or independent loci. Do not promise ~210 positives per group, ~105 per joint cell, or adequate power before measuring them.
- Additional independent positive loci could improve precision; the required number depends on effect size, prevalence, clustering and design. No defensible “10×” or “13,000 pairs” requirement has been established here.
- Keep the gene axis deferred. Revisit a joint enhancer × gene analysis only if requested and justified by coverage, group counts and the scientific question—not solely because a first-stage p-value is favorable.

### Gene-expression cache for a possible later extension

- K562 has two CAGE biosamples in `eQTL_benchmark7_cache/biosamples.tsv`: `CAGE:chronic myelogenous leukemia cell line:K562` and the corresponding `K562 ENCODE, biol_` entry.
- Use `gene_cage_sense_biosamples.npy` or derive matching-strand values from `gene_cage_columns.npy`; `gene_cage_biosamples.npy` is the original **both-strand sum**, not the primary sense readout.
- The cache includes 20,229 annotated genes, with coverage statuses. It is not a guarantee every linking gene/TSS is present or fully covered. Audit the join first.
- Reconcile both K562 biosamples and normalize/define a K562-specific reference if this axis is later implemented. Do not copy GM12878 expression thresholds or silently pool the two biosamples.
- `raw_bins.npy` contains ±512-bp windows for eQTL benchmark TSSs only. It does not provide arbitrary linking-TSS windows; window changes also need a compatible background reference.
- Missing/new loci or changed promoter geometry may require source extraction. Explain this to the user before running it; do not launch a scan automatically.
- A later joint design must define gene groups explicitly. K562-low/others-high is an inverse group, not a synonym for broad K562-active expression. Do not revive the superseded GM12878 four-way instructions.

## Latest eQTL status — paused, inconclusive (2026-09-08)

The user paused this task after the continuous analysis was implemented. Do not resume it unless asked. The previous extreme-group construction remains secondary; it is no longer the main analysis.

### Current definitions and outputs

- Nine poster scores and original labels are unchanged; **1,023 common-support pairs / 218 positives**.
- Main eligible cohort: fully covered genes with GM12878 ≥20 processed promoter-CPM; **765 pairs / 147 positives / 409 genes**. The 20-CPM gate is an analysis choice, not a calibrated detection threshold.
- Main inference: separate per-model binomial GLMs for eQTL status with standardized score midrank, `-log2(others_mean+0.1)` specificity, log GM expression, log distance, and score interactions with each covariate. Uses all eligible pairs.
- Gene-cluster CR1 covariance and paired model contrasts; shared variants/LD across genes remain unmodeled. Coefficients describe conditional score-label association, **not adjusted AP, causal effects or held-out prediction performance**.
- Main display: `PRESENT = ['gm_active_others_lower', 'gm_active_others_higher']`, exported in `specificity_group`. Split others mean below versus at/above the median among covered GM-active protein-coding background genes: **77.8796 promoter-CPM**.
- Display strata: higher specificity **397 pairs / 83 positives / 208 genes**; lower specificity **368 / 64 / 201**. No eligible middle is excluded.
- Secondary `directional_group` retains extreme groups (178/33 versus 136/27 pairs/positives), inverse and other categories. `expression_group` retains the ratio split. Do not mix these columns or sum overlapping annotation categories.
- Focused comparisons: Borzoi full FT, LoRA and scratch each versus pretrained Borzoi. Full FT has a positive continuous interaction difference, nominal p≈0.0355, **Holm-adjusted p≈0.1066** across the three contrasts. No focused continuous comparison passes 0.05 after this correction.
- Findings are exploratory after prior threshold/result inspection. “Inconclusive” is appropriate; failure to reject is not proof of a null effect. Expression imbalance is a plausible confound, not an established explanation for all model trends.
- Source extraction arrays and original model-score cache were reused unchanged; only derived annotations, statistics and figures were regenerated.

Files (relative to repository root):

```text
code_test/eQTL_benchmark7.ipynb           # executed; 27 cells, 16 code cells
code_test/eqtl_benchmark7_pipeline.py    # extraction and completed-cache validation
code_test/eqtl_benchmark7_analysis.py    # scoring, median/extreme annotations, continuous inference, plots
code_test/test_eqtl_benchmark7.py        # 37 tests passed at latest implementation
code_test/run_eqtl_benchmark7.py         # cache | notebook | all
code_test/run_eqtl_benchmark7.slurm
code_test/eQTL_benchmark7_cache/README.md
```

Derived output files in `code_test/eQTL_benchmark7_cache/`:

- `classification_manifest.json`, `continuous_manifest.json`: current definitions, exact thresholds and provenance.
- `gene_expression_specificity.tsv.gz`, `pair_scores_and_specificity.tsv.gz`, `cohort_attrition.tsv`.
- `stratified_metrics.tsv`, `focused_AP_comparisons.tsv`: descriptive AP analyses with gene bootstrap.
- `continuous_coefficients.tsv`, `continuous_focused_comparisons.tsv`, `continuous_cohort.tsv`.
- `matched_extremes_caliper_0.5_seed_7.tsv` and matching gene-membership export: secondary fixed-caliper analysis. The ambiguous `matched_sensitivity.tsv` was archived; do not mix it with the current run.
- `figures/relative_specificity_*`, `figures/extreme_specificity_*`, `figures/continuous_specificity.*`.
- Latest previous-state archive: `legacy_extremes_before_continuous/`. Older archives are historical, not current definitions.

## Landmines — all four cost real time this session

1. **`read_chunk` silently returned zero-fill for a missing chunk.** A NaN promoted `span_row` to
   float, so the chunk path became `"1316.0.0.0"`, which does not exist; the sparse-store branch
   then fabricated zeros. The scan "completed" in 160 s having read nothing, and every value was 0.
   `read_chunk` now asserts integer indices and refuses to fabricate zeros unless
   `allow_missing=True`. `borzoi.zarr` has a chunk file for all 55,497 rows — verify this before
   ever enabling that flag.
2. **NaN promoting an index column to float** is the general form of the above. Any column used to
   build a path or index must be cast explicitly with a sentinel for missing rows.
3. **PLINK chromosome coding**: the eQTL benchmark encodes X as `23`, so `'chr' + str(chrom)` gives
   `chr23` and every X-linked gene fails a GENCODE join. Map 23→X, 24→Y.
4. **Merge windows at bin width, not at overlap.** Sums are over whole 32 bp bins, so two windows
   1–31 bp apart still share an edge bin and double-count it — 272 of 20,229 genes at a 0 bp merge
   threshold. Merging when the gap is under one bin makes a shared bin impossible at any grid offset.

Also worth knowing: **span starts are not aligned to a common 32 bp grid** (starts mod 32 are spread
over all residues), so overlapping spans carry different bin boundaries and duplicate reads of the
same base pair do not agree exactly. Record the chosen span and the bp actually summed.

Performance note: span assignment is a minimum-interval-stabbing problem — each span is the same
width, so the spans covering a window form a contiguous slice of sorted starts, and the
right-endpoint greedy is optimal. That cut 30,207 gene windows to **7,547 chunk reads** (~4x), and
the genome-wide scan runs in ~40 min on 2 cores.


## EP_linking10 implementation status (2026-09-08)

- Executed notebook: `code_test/EP_linking10.ipynb` (28 cells, 14 code cells).
- Helpers: `code_test/ep_linking10_pipeline.py`, `code_test/ep_linking10_analysis.py`.
- Runner: `code_test/run_ep_linking10.py` (`build | notebook | all`); optional `code_test/run_ep_linking10.slurm`.
- Tests: `code_test/test_ep_linking10.py` — 18 passed.
- Outputs: `code_test/EP_linking10_cache/`; README describes matrix ordering, support, provenance and reruns.
- Reproduces all twelve EP9 common-context AP values. No failed in-context shuffles in these inputs.
- Matrix: 674 tracks × 3,926 unique enhancer intervals. All intervals covered; 3,749 K562-active, median breadth 270. Higher specificity ≤270: 1,884 enhancers; lower >270: 1,865.
- Common cohort after accessibility filtering: 723 higher-specificity pairs / 217 positives; 1,002 lower-specificity pairs / 165 positives. All 382 original common-context positives retained.
- Longer-context cohort after accessibility filtering: 1,564 higher-specificity pairs / 242 positives; 2,280 lower-specificity pairs / 183 positives. All 425 original longer-context positives retained.
- Connected components of shared genes or overlapping enhancer intervals define resampling units. Same bootstrap weights are applied across strata and arms. Cross-stratum dependencies are retained; arbitrary distal LD and study-wide dependence remain unmodeled.
- Diagnostic/presentation figures are separate by cohort and perturbation arm; model and perturbation contrasts are exported with paired intervals.
- A conditional ranking section now follows the pooled analysis. It computes AUROC within each `dataset × gene × specificity` unit containing both labels, then macro-averages units equally. Common support has 23 higher-specificity and 32 lower-specificity informative units; longer-context support has 30 and 37.
- Shared component weights provide paired intervals across specificity and model arms. No individual model's higher-minus-lower macro-AUROC interval excludes zero. Exact same-gene/dataset units in both specificity groups number only 7 and 9; their observed point estimates favor lower specificity for every displayed shuffle model, but they are explicitly too small for inference.
- Conditional outputs: `within_gene_unit_AUROC.tsv`, `within_gene_macro_AUROC.tsv`, `within_gene_specificity_differences.tsv`, `within_gene_model_interactions.tsv`, `within_gene_shared_unit_sensitivity.tsv`, and `within_gene_manifest.json`. Focused interactions are exploratory and not multiplicity-adjusted.
- Source DNase arrays/thresholds and original eQTL/dsQTL notebooks were not changed. First build reads existing prediction arrays; completed reruns are EP10-cache-only plus small input hashes.
- Gene-expression stratification remains deferred; do not resume eQTL work or add a gene axis automatically.
