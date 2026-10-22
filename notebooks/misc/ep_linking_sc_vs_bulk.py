"""Distance-stratified E-P linking AUPRC: single-cell expression models vs bulk CAGE models.

Single-cell runs come from evaluation_tss3.ipynb section 8 (one scalar per gene, score =
perturbed - reference in ln space). Bulk runs come from EP_linking9.ipynb Part 2 (profile
heads, score = log2 ratio summed over +-576 bp around the TSS), plus the 1-cell-type
cis-Mamba acc run (k562_stranded_borzoi_cage_tss_centered.npy, "ours-borzoi" in EP_linking9).
Two Poisson-loss RNA-profile models from evaluation_tss2.ipynb are added as their own group:
  - rna_profile_poisson: sc count head + bp profile head (Poisson profile term), scored off the
    count head like the other sc models; both the acc and shuffle arms.
  - old_poisson_regions: bp-resolution stranded 3' RNA profile model (poisson_loss_nan only), from
    k562_3prime_fixed_e2g_ep7.csv, scored as log2fc_sqrt_tp_sense (evaluation_tss2's headline:
    sqrt space is exact for this csv, raw_* inverts the wrong transform; see that notebook, 8).

The two notebooks are not directly comparable as they stand, so everything is recomputed here
on one footing:
  - pairs: the two notebooks filter the CRISPR benchmark with different gene JSONs
    (10306 sc vs 10280 bulk), so pairs are joined on the chrom:start-end_gene key.
  - in-context: the sc window is strand-oriented and asymmetric (163840 bp upstream of the
    TSS) while the bulk windows are TSS-centred, so only pairs in context for EVERY model
    shown are scored. The pretrained Enformer (+-98 kb reach) is left out because it would
    empty the >100 kb bin.
  - distance: rE2G DistanceToTSS for both (what evaluation_tss3 uses).
  - metric: average_precision_score on |score| (EP_linking9 Part 2's convention; the sc
    notebook's auc(recall, precision) numbers will differ slightly).

Run: cd /data1/lesliec/sarthak/decima && pixi run python <this file>
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import average_precision_score

E2G_DIR = "/data1/lesliec/sarthak/data/joint_playground/e2g"
DECIMA_E2G_DIR = "/data1/lesliec/sarthak/decima/runs/e2g"
RNA_INFO_DIR = "/data1/lesliec/sarthak/data/DE_danwei"
TSS_BOUNDS_FILE = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
CRISPR_FILE = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
RE2G_FILE = (f"{E2G_DIR}/ENCODE-rE2G_extended_CrossValidated_K562_"
             "EPCrisprBenchmark_ensemble_data_GRCh38.predictions.tsv.gz")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_SHUFFLES = 11
HALF_BP = 576        # bulk TSS window, +-18 bins of 32 bp (EP_linking4/9)
FINETUNED = "k562_tss_finetunedcage_3prime"

# (label, stem, arm); stems and arms as in evaluation_tss3 section 8
SC_RUNS = [
    ("gene_mask [acc]",               "k562_decima_2",                          "acc"),
    ("gene_mask_scratch [acc]",       "k562_tss_sc_decima_scratch_ep27",        "acc"),
    ("3' RNA fine-tune [acc]",        FINETUNED,                                "acc"),
    ("gene_mask [shuffle]",           "k562_decima_2_dinuc",                    "shuffle"),
    ("gene_mask_scratch [shuffle]",   "k562_tss_sc_decima_scratch_ep27_dinuc",  "shuffle"),
    ("gene_mask_noacc [shuffle]",     "k562_tss_sc_decima_noacc_ep04_dinuc",    "shuffle"),
    ("gene_mask_noacc_scratch [shuffle]", "k562_tss_sc_decima_noacc_scratch_ep19_dinuc", "shuffle"),
    ("Decima ep0 [shuffle]",          "k562_decima_borzoi_ep0_dinuc",           "shuffle"),
]
# Poisson-loss profile models (evaluation_tss2)
POISSON_RUNS = [
    ("rna_profile_poisson [acc]",     "k562_tss_sc_rna_poisson_ep35",           "acc"),
    ("rna_profile_poisson [shuffle]", "k562_tss_sc_rna_poisson_ep35_dinuc",     "shuffle"),
]
THREE_PRIME_CSV = f"{E2G_DIR}/k562_3prime_fixed_e2g_ep7.csv"
THREE_PRIME_LABEL = "old_poisson_regions 3' profile [acc]"
# (label, stem, bin size); arms as in EP_linking9's stratified panel, plus the acc run asked for
BULK_OURS = [
    ("cis-Mamba 1 cell type [acc]",         "k562_stranded_borzoi_cage_tss_centered",          32),
    ("cis-Mamba 1 cell type [shuffle+acc]", "k562_stranded_borzoi_cage_tss_centered_dinuc_acc", 32),
    ("ours-enformer [shuffle+acc]",         "k562_unstranded_enformer_tss_centered_dinuc_acc",  128),
    # ours-striped in EP_linking9: 2026-05-13/12-20-24, trained with celltypes: 10 (k562plus10_CAGE)
    ("cis-Mamba 10 cell types [acc]",         "k562_striped_tss_centered",                      32),
    ("cis-Mamba 10 cell types [shuffle+acc]", "k562_striped_tss_centered_dinuc_acc",            32),
]
BULK_PUBLISHED = [
    ("Borzoi [shuffle]",      "k562_borzoi_cage_tss_centered_dinuc",            32),
    ("AlphaGenome [shuffle]", "k562_alphagenome_e2g_tss_centered_dinuc.zarr",   32),
]

GROUP_COLOR = {"single-cell": "#2a78d6", "Poisson RNA profile": "#6250d6", "bulk (ours)": "#eb6834",
               "bulk (published)": "#1baf7a", "baseline": "#8a8985"}

DIST_BINS = [("All", lambda d: np.ones(len(d), bool)),
             ("(0, 10] kb", lambda d: d <= 10_000),
             ("(10, 100] kb", lambda d: (d > 10_000) & (d <= 100_000)),
             ("> 100 kb", lambda d: d > 100_000)]


# --- benchmark pairs ---------------------------------------------------------------------------
tss_bounds = pd.read_csv(TSS_BOUNDS_FILE, sep="\t")
crispr = pd.read_csv(CRISPR_FILE, sep="\t")
crispr["master"] = (crispr["chrom"] + ":" + crispr["chromStart"].astype(str) + "-"
                    + crispr["chromEnd"].astype(str) + "_" + crispr["measuredGeneSymbol"])
assert crispr["master"].is_unique

re2g = pd.read_csv(RE2G_FILE, sep="\t")
re2g["master"] = (re2g["chr"] + ":" + re2g["start"].astype(str) + "-"
                  + re2g["end"].astype(str) + "_" + re2g["TargetGene"])
dist_by_master = re2g.set_index("master")["DistanceToTSS"]


def build_gs(json_name):
    '''The gene filter from evals/e2g_tss.py, for one rna_info json.'''
    with open(f"{RNA_INFO_DIR}/{json_name}") as f:
        tss_dict = json.load(f)
    name_to_ensg = tss_bounds.set_index("name")["Ensembl_ID"].to_dict()
    name_to_ensg = {n: e for n, e in name_to_ensg.items() if e in tss_dict}
    g = crispr.copy()
    g["ensg_id"] = g["measuredGeneSymbol"].map(name_to_ensg)
    return g.dropna(subset=["ensg_id", "Regulated"]).reset_index(drop=True), tss_dict


gs_sc, _ = build_gs("k562_sc_rna_info_cellranger.json")
gs_bulk, bulk_tss = build_gs("k562_bulk_rna_info.json")
bulk_strand = gs_bulk["ensg_id"].map(lambda e: bulk_tss[e]["strand"]).values

# everything below is indexed by the shared pair list, in sc order
common = gs_sc["master"][gs_sc["master"].isin(set(gs_bulk["master"]))].values
sc_idx = pd.Series(np.arange(len(gs_sc)), index=gs_sc["master"]).loc[common].values
bulk_idx = pd.Series(np.arange(len(gs_bulk)), index=gs_bulk["master"]).loc[common].values
y_true = gs_sc["Regulated"].astype(int).values[sc_idx]
assert (y_true == gs_bulk["Regulated"].astype(int).values[bulk_idx]).all()
dist = np.abs(pd.Series(common).map(dist_by_master).values.astype(float))
print(f"pairs: {len(gs_sc)} sc, {len(gs_bulk)} bulk, {len(common)} shared")


# --- scores ------------------------------------------------------------------------------------
scores, ctxs, groups = {}, {}, {}

for label, stem, arm in SC_RUNS + POISSON_RUNS:
    if stem == FINETUNED:
        tsv = pd.read_csv(Path(E2G_DIR) / FINETUNED / "scores.tsv", sep="\t",
                          float_precision="round_trip").set_index("pair_id").loc[common]
        s = (tsv["count_after"] - tsv["count_before"]).to_numpy()
        ic = tsv["element_in_context"].to_numpy(bool)
    else:
        prefix = Path(DECIMA_E2G_DIR if stem.startswith("k562_decima_borzoi") else E2G_DIR) / stem
        out = np.load(f"{prefix}.npy")[sc_idx]
        ic = np.load(f"{prefix}_in_context.npy")[sc_idx]
        if arm == "acc":
            assert out.shape[1] == 2
            s = out[:, 1] - out[:, 0]
        else:
            assert out.shape[1] == 1 + N_SHUFFLES
            s = out[:, 1:].mean(axis=1) - out[:, 0]
    scores[label], ctxs[label] = s, ic
    groups[label] = "Poisson RNA profile" if (label, stem, arm) in POISSON_RUNS else "single-cell"

# the 3' model's csv is on the bulk-json pair set, rows grouped by gene; NaN where the 3'-terminal
# window falls outside the input window
tp_df = pd.read_csv(THREE_PRIME_CSV).sort_values("row_idx").reset_index(drop=True)
tp_df["master"] = (tp_df["chrom"] + ":" + tp_df["chromStart"].astype(str) + "-"
                   + tp_df["chromEnd"].astype(str) + "_" + tp_df["measuredGeneSymbol"])
assert (tp_df["master"].values == gs_bulk["master"].values).all()
tp_s = tp_df["log2fc_sqrt_tp_sense"].values[bulk_idx]
scores[THREE_PRIME_LABEL] = tp_s
ctxs[THREE_PRIME_LABEL] = tp_df["in_context"].values[bulk_idx].astype(bool) & ~np.isnan(tp_s)
groups[THREE_PRIME_LABEL] = "Poisson RNA profile"


def bulk_score(stem, bin_size):
    '''score_slab from EP_linking9: log2 ratio over +-HALF_BP on the centre (TSS) bin.'''
    hb = int(round(HALF_BP / bin_size))
    if stem.endswith(".zarr"):
        z = zarr.open(f"{E2G_DIR}/{stem}", mode="r")
        assert z["done"][:].all()
        c = z["preds"].shape[-1] // 2
        sl = np.asarray(z["preds"][:, :, c - hb:c + hb + 1]).astype(np.float64)
        ic = z["in_context"][:]
    else:
        a = np.load(f"{E2G_DIR}/{stem}.npy", mmap_mode="r")
        c = a.shape[-1] // 2
        sl = np.asarray(a[:, :, c - hb:c + hb + 1]).astype(np.float64)
        ic = np.load(f"{E2G_DIR}/{stem}_in_context.npy")
    assert len(sl) == len(gs_bulk)
    # rows: 4 -> [before+, before-, after+, after-], 2 -> [before, after] (unstranded)
    if sl.shape[1] == 4:
        og = np.where(bulk_strand == "+", 0, 1)
        alt = og + 2
    else:
        og = np.zeros(len(sl), dtype=int)
        alt = og + 1
    i = np.arange(len(sl))
    s = np.log2((sl[i, alt].sum(1) + 1e-2) / (sl[i, og].sum(1) + 1e-2))
    return s[bulk_idx], ic[bulk_idx]


for group, runs in [("bulk (ours)", BULK_OURS), ("bulk (published)", BULK_PUBLISHED)]:
    for label, stem, bin_size in runs:
        scores[label], ctxs[label] = bulk_score(stem, bin_size)
        groups[label] = group
        print(f"loaded {label}")

# only pairs every model can perturb
ctx = np.logical_and.reduce(list(ctxs.values())) & ~np.isnan(dist)
for label in scores:
    assert np.isfinite(scores[label][ctx]).all(), label
scores["1/distance"] = 1.0 / (dist + 1)
groups["1/distance"] = "baseline"
print(f"in context for every model: {ctx.sum()} pairs, {y_true[ctx].sum()} regulated; "
      f"max distance {dist[ctx].max() / 1000:.0f} kb")


# --- AUPRC table -------------------------------------------------------------------------------
rows, bins_n = [], []
for bin_label, fn in DIST_BINS:
    m = ctx & fn(dist)
    bins_n.append((bin_label, int(m.sum()), int(y_true[m].sum()), y_true[m].mean()))
    for label, s in scores.items():
        rows.append(dict(model=label, group=groups[label], bin=bin_label,
                         auprc=average_precision_score(y_true[m], np.abs(s[m]))))
res = pd.DataFrame(rows)
table = res.pivot(index="model", columns="bin", values="auprc").loc[list(scores),
                                                                     [b for b, _ in DIST_BINS]]
models = list(scores)
pd.set_option("display.width", 200)
print()
print(table.round(3))
print()
for b, n, npos, base in bins_n:
    print(f"{b:14s} N={n:5d}  regulated={npos:4d}  random={base:.3f}")
table.round(4).to_csv(os.path.join(OUT_DIR, "ep_linking_sc_vs_bulk.tsv"), sep="\t")


# --- figure: one panel per distance bin, models as rows ----------------------------------------
yy = np.arange(len(models))[::-1]
fig, axes = plt.subplots(1, len(DIST_BINS), figsize=(17, 7.4), sharey=True)
for ax, (bin_label, n, npos, base) in zip(axes, bins_n):
    vals = table[bin_label].values
    for y, label, v in zip(yy, models, vals):
        ax.barh(y, v, height=0.72, color=GROUP_COLOR[groups[label]],
                hatch="//" if label == "1/distance" else None, edgecolor="white", linewidth=0.8,
                zorder=3)
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.axvline(base, color="#0b0b0b", ls="--", lw=1.1, zorder=4)
    ax.text(base, len(models) - 0.35, f" random {base:.3f}", fontsize=7.5, color="#52514e",
            va="bottom")
    ax.set_xlim(0, max(1.0, vals.max() + 0.12) if bin_label != "> 100 kb" else vals.max() * 1.3)
    ax.set_title(f"{bin_label}\nN={n}, {npos} regulated", fontsize=11)
    ax.set_xlabel("AUPRC (average precision)")
    ax.grid(axis="x", color="#e4e3df", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(models, fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
fig.legend(handles, list(GROUP_COLOR), loc="lower center", ncol=5, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"E-P linking (ENCODE CRISPRi, K562): single-cell vs bulk models on the {ctx.sum()} "
             f"pairs in context for every model (enhancer within {dist[ctx].max() / 1000:.0f} kb)",
             fontsize=12)
plt.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"ep_linking_sc_vs_bulk.{ext}"), dpi=200,
                bbox_inches="tight")
print("saved", os.path.join(OUT_DIR, "ep_linking_sc_vs_bulk.png"))
