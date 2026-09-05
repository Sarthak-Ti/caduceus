**Pancreatic multiome expansion: data audit, modeling risks, and experiment design**

Prepared September 4, 2026. This assessment uses read-only inspection of the supplied data, current source code, selected saved Hydra configurations, and primary literature. No model was trained and no source code or input dataset was changed. Numerical checks below are exploratory data checks, not model-performance results. The precise enhancer screen, biological meaning of F1–F7, and availability of matched RNA BAMs remain unconfirmed; questions were sent during the review.

The approach is worth pursuing. My main reservation is not that 3′ RNA-seq is unsuitable for gene-expression modeling. It is that strong expression prediction, cell-type specificity, sample efficiency, and causal enhancer–gene linking are four different claims. Accessibility can improve the first three while allowing a model to bypass the sequence dependencies needed for some versions of the fourth. The experiment should measure each separately.

I would begin with a shared sequence-plus-accessibility model predicting gene-level pseudobulk expression, use a fixed annotation-derived gene mask, and treat RNA profiles as an additional experiment once their provenance is established. Before a large sweep, resolve the two missing accessibility tracks, normalize the accessibility input deliberately, establish genomic holdouts that account for full windows, and freeze the enhancer-screen evaluation protocol.

**What I verified in the supplied pancreatic data**

The RNA file is `/data1/lesliec/jiaxin/PANC/data/CellRanger/PANC_CellRanger_38k_rna_Oct_18.h5ad`. It contains 38,568 cells and 36,601 gene features. `X` is CSR sparse with 108,381,767 stored values. I scanned all stored values: they are nonnegative integers represented as float64, with nonzero values ranging from 1 to 2,947. Cell-wise sums agree with `obs.nCount_RNA`. `raw/X` has the same shape, number of stored values, and integer/nonnegative properties; I did not test elementwise equality between `X` and `raw/X`. There are no named layers.

All 16 requested labels occur in `obs.celltype_correct`. The feature index and `var.features` contain gene symbols in the examples inspected, and the full feature index has no duplicates. This is not an Ensembl-ID-indexed matrix ready to join directly to your existing K562 JSON. Establish the symbol-to-gene-ID mapping against the actual Cell Ranger reference and record ambiguities and exclusions; unique symbols in the matrix do not guarantee a unique annotation mapping.

The following ATAC values are bigWig header `sumData`, expressed in millions. They summarize stored signal and should not be relabeled as unique fragments or cells without checking the generating pipeline.

| Cell type | RNA cells | Total RNA UMIs, millions | Median UMIs/cell | ATAC signal, millions |
|---|---:|---:|---:|---:|
| DE | 1,955 | 14.45 | 6,092 | 97.81 |
| Mesenchyme | 1,136 | 4.55 | 3,239 | Missing at supplied pattern |
| Nonendocrine_cells | 6,388 | 40.24 | 4,602.5 | Missing at supplied pattern |
| Nonendocrine_prog | 268 | 2.71 | 8,297.5 | 16.90 |
| PP1 | 2,586 | 14.07 | 4,500 | 151.36 |
| PP2 | 962 | 5.86 | 4,900.5 | 52.82 |
| PP2_neurog3pos | 776 | 4.75 | 5,896 | 42.70 |
| early_Alpha | 1,140 | 7.63 | 6,508 | 39.91 |
| early_Beta | 2,597 | 21.19 | 7,791 | 92.57 |
| early_Delta | 510 | 2.72 | 5,193 | 20.08 |
| early_Enterochromaffin | 4,363 | 33.06 | 7,114 | 191.13 |
| late_Alpha | 4,702 | 26.42 | 5,287 | 162.76 |
| late_Beta | 6,320 | 49.69 | 7,130 | 256.28 |
| late_Delta | 244 | 1.38 | 5,376 | 8.78 |
| late_Enterochromaffin | 3,909 | 25.69 | 5,891 | 167.92 |
| late_prolif_Alpha | 712 | 5.74 | 7,387.5 | 29.92 |

Fourteen files resolve at `.../Oct_celltype_specific/{celltype}_50k_peaks/auxiliary/data_unstranded.bw`. The directory listing also lacks the corresponding `Mesenchyme_50k_peaks` and `Nonendocrine_cells_50k_peaks` directories. This establishes missing files at the proposed location, not that those data do not exist elsewhere. Do not silently omit these populations or substitute another population's track.

All 14 readable tracks have 24 chromosome entries, with chr1 length 248,956,422, consistent with hg38. I did not compare every chromosome against your tokenized reference. Their header minimum stored values are 1, their maximum values vary from 58 to 1,269, and total signal varies 29.2-fold. Sparse bigWigs distinguish an unstored base from a stored value: decide when an unstored base means zero observed cuts and when a region is missing or excluded. Genome gaps, missing chromosomes, and invalid regions should not become ordinary zero-expression training examples.

The DE ChromBPNet argument file identifies ATAC fragment input and an hg38 reference. The input filename explicitly mentions barcode correction. ChromBPNet documents `data_unstranded.bw` as observed training accessibility; it is not the bias-corrected model prediction merely because it sits under a ChromBPNet output directory. Confirm the local fork's shifts, deduplication, blacklist handling, and any scaling. [ChromBPNet output documentation](https://github.com/kundajelab/chrombpnet/wiki/Output-format).

**Sample composition is a major confound to resolve**

Metadata include `orig.ident` = F1–F7, lanes 1–4, and two Yes/No protocol fields. Cell types are strongly associated with `orig.ident`: 1,953/1,955 DE cells are F1, 2,507/2,586 PP1 cells are F2, 920/962 PP2 cells are F3, and 765/776 PP2_neurog3pos cells are F3. Early endocrine populations mainly occupy F4/F5; late populations mainly occupy F6/F7. Protocol proportions also vary substantially. For example, early Enterochromaffin contains 728 F4 and 3,573 F5 cells; late Enterochromaffin contains 1,223 F6 and 2,673 F7 cells.

These could be intentional stage/protocol differences, not a processing error. But you cannot infer donor replication from lanes or treat F1–F7 as exchangeable biological replicates without the sample sheet. If a state exists in only one biological sample, its state effect and that sample's technical effect cannot be disentangled by modeling alone. Batch correction cannot recover information absent from the experimental design, and indiscriminate correction could remove real differentiation biology.

Preserve cell type × biological sample pseudobulks wherever there are enough cells, and record protocol and stage. Aggregate technical lanes belonging to the same sample appropriately. Construct ATAC and RNA aggregates from the same biological populations, ideally intersected barcode sets, or explicitly describe differences. Establish a mapping between the h5ad barcodes and fragment/BAM barcodes; do not remove suffixes blindly. A profile from all cells in a stage is not a matched label for a subtype-specific ATAC input.

**The count targets look usable, but overall correlation conceals the hard part**

I performed one random split-half check per annotated population using seed 42. Within each half I summed raw counts and calculated natural-log `log1p(CPM)`. For a descriptive filter I retained the 14,058 genes with at least 10 CPM in any full-data population. This all-data filter is acceptable for this audit; it should not become a test-informed feature-selection rule in the benchmark.

I also double-centered each half's cell-type × gene matrix by subtracting its gene and cell-type means and adding the grand mean. The resulting correlations emphasize differential expression rather than shared expression magnitude.

| Population | Split-half expression Pearson r | Split-half double-centered r | Other 15 populations' mean vs this population, r |
|---|---:|---:|---:|
| DE | 0.995 | 0.987 | 0.752 |
| PP1 | 0.995 | 0.976 | 0.866 |
| PP2 | 0.987 | 0.944 | 0.868 |
| PP2_neurog3pos | 0.983 | 0.896 | 0.917 |
| Nonendocrine_prog | 0.969 | 0.913 | 0.810 |
| early_Beta | 0.996 | 0.969 | 0.923 |
| early_Delta | 0.972 | 0.795 | 0.934 |
| late_Beta | 0.998 | 0.983 | 0.926 |
| late_Delta | 0.946 | 0.687 | 0.926 |

Across all 16 populations, ordinary split-half correlations range from 0.946 to 0.998. This supports gene-count modeling. However, the late-Delta differential signal is much noisier, and merely predicting the average expression of the other populations already correlates 0.926 with late Delta. That comparator has access to the same genes' expression elsewhere: it is a diagnostic or a legitimate seen-gene transfer baseline, not a permissible strict unseen-gene predictor.

Random cell halves share biological samples and technical effects. These values do not measure biological replication, do not establish a strict model ceiling, and do not provide independent uncertainty estimates across donors. Repeat with sample holdouts where the design permits; use repeated matched-cell subsamples to assess robustness, especially in small populations.

Report per-cell-type expression correlation and error, but make gene-wise cell-type contrasts, early-versus-late contrasts, and double-centered correlation central to the specificity claim. A model can predict broadly expressed genes accurately while missing which population uses an enhancer. With only 16 correlated states, individual gene-wise correlations also need expression/variance filters and uncertainty estimates.

**Accessibility scale and shortcut learning**

The example's pretrained checkpoint is run 87, `outputs/2025-07-18/00-23-52-538795`. Its saved config uses K562 DNase, `mlm=0`, accessibility masking rate 0.25, and 500-bp accessibility masks. It is accessibility-reconstruction pretraining with sequence available; the config does not establish that predictions rely mostly on sequence. Predictability from neighboring accessibility is an empirical question.

There is also an assay/scale shift. A 400,000-base sample from four chr21 intervals in the K562 DNase NPZ had median 0.0225, 99th percentile 0.2085, maximum 3.752, and approximately 79% noninteger values. These are limited sampled values, not genome-wide quantiles. They nevertheless show that the old input is not simply the same raw integer-valued signal convention suggested by the pancreatic bigWig headers. The current TSS loader reads continuous values without an automatic DNase-to-ATAC calibration.

Use an explicit normalization convention tied to library size and the relevant genomic territory, followed by a documented zero-preserving transform if appropriate. Fit clipping or learned scale parameters using training data. Compare normalized and depth-matched versions. Avoid assuming quantile normalization is harmless: it can remove real differences in global accessibility. Without external standards, distinguish relative accessibility from absolute changes per cell.

Separate three controls: an ATAC swap between cell types at the same locus, an ATAC depth rescaling within a cell type, and an ablation of distal accessibility while preserving the promoter. The first tests biological context use, the second tests nuisance sensitivity, and the third tests whether distal context adds information beyond local activity. Use within-lineage swaps as well as very different states. An ATAC-only predictor and a promoter-only predictor are essential: an accessibility-conditioned model could achieve strong expression metrics mainly from promoter/gene-body accessibility.

Pretraining comparisons should include the same architecture without pretraining, a sequence-only counterpart trained for that input regime, and a sequence-plus-ATAC model with the same target/head. Simply zeroing ATAC in a trained multimodal model is a robustness test, not an adequately trained sequence-only baseline. Additional pancreatic ATAC pretraining is reasonable, but disclose whether test-state or test-locus ATAC was seen. That can constitute transductive use of unlabeled data rather than strict induction.

**3′ RNA-seq: use counts first, assess profiles on their own terms**

Gene UMI counts are a defensible primary target. Sum raw UMIs within each chosen pseudobulk, then normalize and log-transform once. A practical matched target is `ln(1 + 10^6 * gene_UMIs / pseudobulk_total_UMIs)`, with the denominator gene universe fixed and documented. This matches the convention in your inspected count-building code and Decima's documented preprocessing. Do not apply transcript-length normalization as if these were conventional full-length bulk RNA-seq fragment counts. [Decima fine-tuning tutorial](https://genentech.github.io/decima/tutorials/3-finetune.html).

Pooling raw counts then normalizing estimates a library-weighted RNA composition. Averaging per-cell normalized expression weights cells differently; averaging logged expression estimates something different again. Pick the quantity relevant to the claim and use it for all matched baselines. Inspect sensitivity to composition normalization and to excluding mitochondrial genes and very abundant transcripts. INS is about 2.26% of late-Beta UMIs and GCG about 3.23% of early-Alpha UMIs in this matrix; these are appreciable but do not establish overwhelming hormone domination. Ambient RNA, doublets, and immature/polyhormonal states can all affect marker patterns; marker presence alone cannot identify which explanation applies.

The h5ad cannot reconstruct real genomic RNA coverage. Profile training requires BAMs or existing appropriately generated stranded coverage. Fabricating profiles by distributing gene counts across annotated exons or near the 3′ end supplies an annotation-derived target, not an independent assay measurement.

For real profiles, establish library chemistry, cell versus nucleus preparation, transcript strand, Cell Ranger version, gene-assignment rules, and whether introns contributed to UMIs. Cell Ranger's intron behavior depends on version and pipeline; the supplied filename does not establish its settings. Intronic priming can also produce signal away from the terminal exon. [10x gene-expression algorithm](https://www.10xgenomics.com/support/cn/software/cell-ranger/latest/algorithms-overview/cr-gex-algorithm), [10x intronic-read technical note](https://cdn.10xgenomics.com/image/upload/v1660261285/support-documents/CG000554_Interpreting_SingleCellGEX_with_introns_RevA.pdf).

Counted UMI representatives, all aligned reads, and read-base coverage are different targets. Even one retained alignment per molecule contributes multiple covered bases, so integrated coverage is not a gene UMI count. Reads spanning splice junctions must not fill introns as contiguous coverage. Use barcode- and UMI-aware processing with the correct gene assignment, and validate aggregate counts against the corresponding matrix. Verify strand on several isolated genes on both genomic strands. Your K562 pipeline already has explicit strand swapping and UMI-representative filtering; pancreatic files require their own verification.

A 3′ profile can encode abundance, alternative polyadenylation, internal priming, transcript processing, RNA stability, and assay capture. A profile gain therefore need not be a gain in enhancer regulation. A useful diagnostic is to compare gains at distal candidate enhancers with gains at terminal exons/polyadenylation-related sequence. Scooby provides direct precedent for adapting Borzoi to single-cell RNA profiles and their 3′ bias; the assay is a modeling challenge, not a reason to reject profile supervision automatically. [Scooby paper](https://www.nature.com/articles/s41592-025-02854-5).

Use fixed annotation-derived gene masks across cell types for the main experiment. Choosing the active TSS, terminal exon, or polyadenylation site from each population's target RNA and then supplying it as input risks giving the model information you want it to predict. A union-of-annotation mask is legitimate prior information, but alternative TSSs and transcript ends should be analyzed for ambiguity and context truncation. Recompute length and truncation statistics for these genes; the K562 percentages in the script comments are not pancreatic measurements.

**Profile loss and scoring have several non-obvious traps**

Your `TSSProfileDecoder` emits profile logits and a separate pooled count prediction. Under the multinomial loss, `p_i = softmax(z)_i` describes relative position probabilities. It does not predict absolute abundance. Adding the same constant to every logit leaves the loss unchanged; summing logits or summing the full softmax therefore cannot provide an identified expression magnitude. A profile-only model trained with a Poisson/rate loss is a different case: it can carry magnitude.

For whole-window multinomial predictions, the fraction of mass inside a gene can be gene-specific, but it is compositional. A neighboring gene becoming more strongly predicted can reduce that fraction without any change in the focal gene's absolute transcription. Under gene-restricted normalization, summing the probabilities over the gene is identically one. Define the enhancer score and its biological interpretation explicitly before comparing profile-only and count-based models.

I inspected `evals/e2g_tss_profile.py`: its default enhancer calculation uses the **count head**, despite the profile-related filename, and decreases ATAC in the element plus a flank. It does not by default assess a profile-derived enhancer score. `evals/e2g_3prime.py` separately scores rate-style profiles and documents approximate inversion of the fixed square-root transform. Keep these evaluations distinct in figure labels and run manifests.

The multinomial's model-dependent term is `-sum_i y_i log(p_i)`, with logit gradient `N*p_i - y_i`, where `N=sum_i y_i`. Thus profile gradient strength depends on total target mass and shape, not simply window length or the numerical value of the logged loss. The factorial terms change the loss value but contribute no model gradient. Fractional, variance-stabilized targets make this a weighted shape objective, not a literal likelihood for original UMI counts. A Poisson loss on transformed coverage likewise should not be described as a calibrated raw-count likelihood.

Do not transplant `profile_weight=1e-6` without checking gradient norms and validation tradeoffs after changes in depth, normalization, bin size, and support. A normalized cross-entropy with explicit sample weights is one possible shape objective; compare its effect on weakly expressed genes. Empty profiles contribute no shape information. Estimate losses in adequate precision and handle empty masks/invalid positions explicitly. Also, scaling a loss under AdamW is not generally equivalent to multiplying the learning rate: optimizer moments, clipping, epsilon, and weight decay matter.

The old K562 profile NPZ used `sqrt(1+x)`, which maps zero coverage to one. The example's **fixed** file is generated by subtracting one, restoring a zero background. I verified this contrast in four sampled chr21 intervals totaling 400,000 bases; those particular intervals were entirely background. The current example uses the fixed file, so I am not reporting that old floor bug as still present in this run. Comparing old and fixed targets would confound a profile-method comparison, and the comments still describe the old transform in places.

Whole-window profile supervision also repeatedly presents overlapping genomic labels around neighboring genes. This changes the effective weighting of gene-dense regions and the amount of supervision relative to count-only training. A gain may reflect useful dense supervision, different sampling, or more observed labels, not only a better head. Match available RNA evidence for external baselines where possible and report what differs.

**Concrete implementation and split findings to account for**

The example script's comments describe a gene-body input mask, but its active command sets `append_tss_mask=true`. The saved config for run 223 (`outputs/2026-08-13/16-15-08-293121`) agrees with the command. Run 213 (`outputs/2026-07-27/21-54-24-337889`) sets `append_gene_mask=true`. Both use upstream 163,840 and whole-window count pooling. Therefore run 223 versus run 213 is not an isolated profile-head ablation: the input mask differs as well. Labels and comments alone are not sufficient experimental provenance.

The current ordinary `TSSDecoder` averages embeddings over the selected region. Its `bp_predictor` branch instead applies a nonnegative per-position prediction, sums, and applies `log2(1+sum)`. `TSSProfileDecoder` averages embeddings for its count head. These are materially different readouts. For a fixed full-window length, sum and mean pooling differ by a constant that a subsequent learned layer can often absorb; variable mask sizes and nonlinear per-position prediction make the distinction more consequential. Contextual embeddings already contain information from other positions, so even a sum of per-position readouts should not be interpreted as independent biological contributions of the underlying bases.

Your supplied JSON passes its `counts` field through the dataset without a fresh logarithm. The inspected builder uses natural-log CPM, whereas the additive decoder explicitly uses log2. The model may compensate during training, but inverse transforms and reported fold changes must follow the actual target and checkpoint, not a generic description of a count head.

The present `TSSDataset` has one item per gene; it is not automatically a gene × cell-type loader. Merely placing 16 tracks in a file does not create 16 paired training examples. In its default aligned loading branch it selects the first track; passing a list explicitly loads selected tracks together. The future design must pair the chosen population's ATAC with its corresponding expression target and preserve splits across all copies of a locus. Check tensor shapes and broadcasting explicitly when going from scalar to multi-track labels.

`TSSLoader.setup()` constructs only train and validation datasets, and `test_dataloader()` returns the validation dataset. Separate evaluation scripts can explicitly request the JSON test split, so this does not invalidate every prior evaluation. It does mean that an automatic trainer test result is not evidence of held-out test performance. The current setup also passes the same dataset kwargs to training and validation; with the inspected config, validation does not automatically receive `evaluating=true` and can retain sequence shifts. Use deterministic held-out evaluation when comparing checkpoints.

I calculated interval overlap using the actual K562 Cell Ranger JSON and the example's 524,288-bp, strand-aware asymmetric geometry, without augmentation. There are 14,791 train, 2,038 validation, and 1,844 test genes.

| Held-out split | Windows overlapping a training window | Held-out TSS inside a training window |
|---|---:|---:|
| Validation | 184 / 2,038 (9.0%) | 85 / 2,038 (4.2%) |
| Test | 146 / 1,844 (7.9%) | 70 / 1,844 (3.8%) |

This is geometry, not proof that every affected label leaked or that any reported metric is inflated by a particular amount. In scalar training, overlap exposes shared sequence/context; in whole-window RNA profile training, it can expose held-out genes' RNA labels. Strand and the location of the actual coverage determine the exact label overlap. The inspected split builder assigns splits from the TSS position, explaining why full windows can cross boundaries. Sequence jitter can expand the overlap further.

Use chromosome holdouts or buffered genomic blocks with checks on full input windows and supervised outputs, including augmentation. Keep all cell-type copies and alternative transcripts of a locus together. Distinguish fine-tuning holdout status from pretraining exposure. For a strict unseen-locus comparison, audit which loci and modalities each pretrained backbone saw and use compatible pretrained folds where feasible. It is legitimate to separately evaluate unseen perturbations at familiar loci, but label that generalization setting accurately.

The NPZ path also deserves a throughput check before scaling. `np.load(...)[chrom]` materializes a chromosome member before the slice, even for an uncompressed NPZ; it is not equivalent to a chunked random-access array. The loader's per-example access and multiple workers can amplify memory and I/O costs across 16 populations and RNA profiles. Estimate CPU throughput and RSS on a small real batch before allocating a large GPU sweep. Chunked storage is a future implementation option, not something changed here.

**Define the enhancer-screen question before relying on it as ground truth**

Please confirm the exact screen and processed result table. A plausible local/published match is Kaplan et al.'s pancreatic differentiation CRISPRi study, which screened effects on a PDX1 reporter and investigated a long-range ONECUT1 enhancer. If that is your screen, a PDX1-reducing hit can act through another regulator or alter differentiation; it is not automatically a direct cis enhancer–PDX1 link. Restrict direct-link claims to pairs with appropriate target-gene validation, or define the task as predicting effects on the differentiation/reporter phenotype. I have not established that this is your intended screen. [Kaplan et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11406439/).

Developmental accessibility can precede transcription, and an early enhancer perturbation can have a later phenotype. A static contemporaneous ATAC-to-RNA predictor does not by itself model that trajectory. Match perturbation stage, phenotype time, differentiation protocol, and assayed cell state. A single-state screen cannot validate specificity across all 16 states; measured negatives in additional states are much stronger than merely observing different model scores there.

For `f(sequence, accessibility, annotation)`, preserve and label three distinct perturbation scores:

| Intervention | What it measures operationally | Main limitation |
|---|---|---|
| Perturb sequence, hold ATAC fixed | Sequence dependence conditional on observed accessibility | Intact-enhancer evidence can remain in ATAC |
| Reduce ATAC, hold sequence fixed | Predicted response to imposed accessibility reduction | CRISPRi is not a calibrated uniform ATAC division |
| Perturb both | Joint disruption sensitivity | May be out of distribution and need not equal either causal intervention |

Your current E2G code implements the second, with a configurable scale factor and flank. This can be a useful operational benchmark. It should not be described as an experimentally calibrated fold change after CRISPRi. Compare a predeclared small range of attenuation strengths and spatial extents using development data, inspect monotonicity, and avoid selecting the winning setting on the final screen. If input accessibility is transformed, divide in raw signal space and reapply the transform when the intended operation is a fold reduction in observed cuts; dividing transformed values means something different. Keep annotation fixed under sequence shuffling/ATAC attenuation.

Also inspect whether a candidate overlaps the focal promoter, another promoter, or transcribed sequence. CRISPRi spread or direct promoter suppression can make a nominal distal-element hit easier for reasons unrelated to enhancer specificity. Accessibility correlations can arise from shared state or transcription-associated chromatin, so observational ATAC-to-RNA learning alone does not identify intervention effects.

Freeze the tested candidate universe, coordinate build, gene mapping, label criteria, and missing-prediction policy before scoring. Untested pairs are not negatives. Non-significant pairs need sufficient power to be credible negatives. Report positive prevalence, AUPRC with its exact implementation, precision at useful recall, and effect-size/sign agreement where the experiment supports it. Show distance strata, expression strata, and per-gene results; use paired resampling clustered by gene/locus or another defensible independent experimental unit. Do not count multiple guides for one enhancer as independent biological links. Recent scE2G benchmarking explicitly addresses well-powered nonhits and indirect effects, providing a useful protocol reference. [scE2G paper](https://www.nature.com/articles/s41588-026-02695-8).

The asymmetric input has about 164 kb upstream and 360 kb downstream in transcription orientation. This helps include transcript ends but excludes more far-upstream enhancers. Report a common eligible set across models and a separate full-screen coverage analysis. Excluding out-of-window positives silently makes models with different context lengths incomparable; assigning them ordinary zero scores also obscures the difference between unsupported context and predicted inactivity.

**Baselines that distinguish the scientific claims**

The proposed Borzoi, AlphaGenome, and Decima comparisons are useful, but insufficient on their own to attribute a gain to your training approach. Your model receives experimentally measured cell-specific accessibility at inference. A sequence-only model does not. Beating it can establish the practical value of adding accessibility, but cannot isolate an architecture or pretraining advantage.

| Comparator | Purpose and fair setup |
|---|---|
| Same model, random initialization, sequence + ATAC | Isolate pretraining benefit at matched labels, architecture, and training budget |
| Same model, sequence only | Measure added value of observed ATAC; train under this input regime |
| ATAC-only and promoter/local ATAC regression | Detect whether a much simpler conditional predictor explains the gain |
| Sequence + local ATAC versus sequence + full ATAC | Test whether distal accessibility improves differential expression and E2G |
| Fine-tuned Borzoi with a matched gene head | Compare transfer with the same gene targets, masks, splits, and RNA budget |
| Borzoi trained from scratch | Quantify architecture performance at the available scale; report convergence and compute |
| Fine-tuned pretrained Decima | Strong single-cell expression comparator; distinguish it from starting from Borzoi to create a new Decima-style model |
| AlphaGenome | Separate frozen/zero-shot track scoring from pancreatic fine-tuning; disclose track identity and input/output context |
| Distance, enhancer accessibility, and accessibility × distance | Essential cheap E2G controls, especially for an ATAC-attenuation score |
| ABC-style method and a multiome linker | Test whether links improve over directly relevant accessibility-based approaches |

Borzoi predicts RNA coverage and its published work reports benefits from including accessibility prediction tasks during training. Predicting ATAC as an output is still different from conditioning on measured ATAC at inference. [Borzoi paper](https://www.nature.com/articles/s41588-024-02053-6).

Decima is explicitly a gene-expression model trained on large single-cell collections and provides gene-mask-aware prediction and fine-tuning resources. Audit its pretraining dataset for possible overlap with this study and distinguish pretrained Decima adaptation from Borzoi initialization. [Decima paper](https://www.nature.com/articles/s41592-026-03102-0), [Decima repository](https://github.com/Genentech/decima).

AlphaGenome's official research repository now exposes model code and a fine-tuning module. Do not assume it is restricted to API inference or that pancreatic fine-tuning is unavailable. I verified the official module exists, not that a pancreatic adapter runs in this workspace. If you use frozen predictions, a bulk pancreas track is not automatically a PP2 or late-Beta prediction. Its published E2G evaluation is a useful reference, but its K562 result does not settle this developmental benchmark. [Official research repository](https://github.com/google-deepmind/alphagenome_research), [official fine-tuning module](https://github.com/google-deepmind/alphagenome_research/tree/main/src/alphagenome_research/finetuning), [AlphaGenome paper](https://www.nature.com/articles/s41586-025-10014-0).

Scooby is particularly relevant to the multiome/3′-profile story because it adapts Borzoi with a cell-specific decoder and parameter-efficient fine-tuning. Consider it if implementation cost permits; audit the provenance of its cell embeddings so held-out RNA is not used to construct test-time context under a no-RNA transfer claim. SCARlink is a direct ATAC-to-RNA multiome comparator, although its gene-specific regressions answer a different generalization question from a shared unseen-gene model. ABC and scE2G provide relevant E2G comparators; use a clearly labeled approximation if you lack the required contact or activity measurements. [SCARlink paper](https://www.nature.com/articles/s41588-024-01689-8), [ABC paper](https://www.nature.com/articles/s41588-019-0538-0).

Give transferred models reasonable head-only and partial/full fine-tuning options, use validation-only hyperparameter selection, and report both practical compute and matched-budget comparisons. Training a large Borzoi from scratch on a tiny dataset and observing failure is not, by itself, strong evidence for uniquely effective pretraining.

**What “small dataset” and “new cell type” should mean here**

Separate low cell number, low sequencing depth, few labeled genes, few labeled cell types, and low optimization cost. Pseudobulking 244 cells still provides thousands of gene labels; it is not the same setting as training on 244 independent labeled examples. Sixteen related populations also share sequences and developmental programs.

For cell-efficiency curves, subsample matched cells before constructing both modalities. Run both an RNA-reduction experiment with ATAC held fixed and an experiment where both modalities shrink; these answer different practical questions. Include an ATAC-depth sensitivity curve at fixed RNA. For label-efficiency curves, hold the underlying assay depth fixed and reduce the number of supervised loci or populations. Keep test data and model-selection budgets fixed across the curve, use repeated subsets/seeds, and show uncertainty. Account for pretraining separately from adaptation cost, and include data-processing/I/O cost when claiming practical efficiency.

Use distinct evaluation settings: unseen loci in trained populations; new populations at previously supervised loci; and both unseen loci and unseen populations. A held-out late-Beta state with early Beta in training measures a different degree of transfer from holding out an entire lineage. A shared ATAC-conditioned model can be evaluated on a new state's ATAC without learning a new arbitrary ID. A model with one independently learned output head or embedding per training state cannot do this without additional adaptation. State clearly whether new-state RNA is allowed for calibration or fine-tuning.

Avoid making a strong zero-shot claim from a model trained with target-state RNA-derived embeddings, active-TSS annotations, or peak-gene links. Standard RNA-informed cell annotation is useful, but it defines an evaluation on already annotated populations rather than an end-to-end system that needs no RNA at deployment.

**A manageable first experiment sequence**

1. Recover the two missing ATAC datasets or explicitly scope an initial 14-population study. Obtain the sample/protocol sheet and screen metadata. Build a manifest of barcode membership, reference annotation, track processing, normalization, and allowed pretraining exposure.
2. Create gene-count pseudobulks with biological sample provenance, fixed annotations, and buffered genomic holdouts. Verify scale, strand, IDs, missingness, and deterministic evaluation. Check that genes of the enhancer screen are retained and their candidate elements fit the input windows.
3. Establish one shared pretrained sequence-plus-ATAC count model, the same model from scratch, a trained sequence-only version, and simple ATAC/local predictors. Compare with fine-tuned Decima or Borzoi using the same count labels. Evaluate both expression magnitude and cell-type contrasts.
4. Test real versus swapped ATAC, promoter versus distal information, and depth sensitivity. These inexpensive diagnostics can reveal whether more elaborate profile work is likely to support the intended claim.
5. If matched BAMs exist, add a profile head while holding mask, window, target normalization, and count head fixed. Compare a predeclared binned rate-profile model separately from the multinomial auxiliary model. Binning at a modest resolution is a reasonable first experiment given 3′ coverage uncertainty; do not interpret individual covered bases as independent labels.
6. Select the limited architecture/loss/scoring choices using development data, then run the untouched enhancer-screen comparison with common candidate eligibility, simple E2G baselines, and clustered uncertainty. If the screen was already used to choose profile methods, designate it development data and retain another genuinely held-out evaluation where possible.
7. Run the cell/label-efficiency and lineage-transfer experiments on the surviving setup. Report specificity, E2G performance, uncertainty, and compute as separate outcomes.

**Evidence and remaining limits**

The audit used the workspace Python environment with h5py, NumPy, SciPy, and pyBigWig; anndata was not installed, so the h5ad was read directly. It scanned sparse RNA values, computed per-population summaries and one split-half comparison, inspected headers for all available ATAC tracks, sampled the K562 DNase and old/fixed RNA NPZs, and calculated interval overlaps from the actual K562 JSON. Current source inspection covered the TSS dataset/loader, decoders/losses, E2G evaluation paths, the example script, run labels, selected saved configs, and external K562 preprocessing scripts.

No checkpoint forward passes, training jobs, full BAM audits, pancreatic gene-coordinate construction, complete reference-genome comparison, or enhancer-screen label audit were performed. Existing code has uncommitted changes, so current source behavior is not proof of the exact implementation used for every historical checkpoint. Numerical audit intermediates were written under `/tmp`; the substantive results and methods are recorded here. No code was edited.

The most consequential unanswered details are the screen's direct versus reporter/phenotypic readout; the identity and replication structure of F1–F7; availability and barcode provenance of RNA BAMs and missing ATAC tracks; whether the intended claim includes zero-shot new populations; and whether the final E2G score will use the scalar count head, a rate profile, or a normalized shape statistic. These details change interpretation and experiment design, but they do not prevent beginning the gene-count data preparation once the sample mapping and splits are fixed.
