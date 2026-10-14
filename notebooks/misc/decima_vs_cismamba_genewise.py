"""Gene-wise comparison of the Decima (Borzoi-init) baseline against cis-Mamba models
and a DNase-only baseline on the K562 test split (1,844 genes).

Each model predicts ONE scalar per gene (ln(1+CPM)), so a per-gene correlation does not
exist. Instead each panel plots the per-gene absolute error |pred - observed| for Decima
(x) against the comparison model (y). Points below the diagonal are genes where the
comparison model is closer to the observed value. The overall Pearson r across genes
(the number reported in claude_summaries/decima_handoff.md) is in each axis label.

DNase baseline (no saved predictions; rebuilt from the recipe in decima_handoff.md):
mean K562 DNase in [TSS-r, TSS+r) for r = 128, 500, 2000 bp, ln(1 + mean) per feature,
OLS with intercept fit on ALL training genes (the handoff's alternative fit, r = 0.7797).
The handoff's "best tested fit" (train genes with target > 1, r = 0.7806) has nearly the
same r but over-predicts low-expression genes, which inflates absolute error, so it is
the wrong variant for an error comparison.

Run: cd /data1/lesliec/sarthak/decima && pixi run python <this file>
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

MODEL_OUT = "/data1/lesliec/sarthak/data/joint_playground/model_out"
TSS_JSON = "/data1/lesliec/sarthak/data/DE_danwei/k562_sc_rna_info_cellranger.json"
DNASE = "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DECIMA = ("Decima (Borzoi init, ep0)", "k562_decima_borzoi_ep0.npz")
MAMBA = [
    ("cis-Mamba, CAGE-finetuned", "k562_tss_finetunedcage_3prime.npz"),
    ("cis-Mamba, gene_mask (decima_2)", "k562_decima_2.npz"),
    ("cis-Mamba, RNA profile Poisson", "k562_tss_sc_rna_poisson_ep35.npz"),
    ("cis-Mamba, tss_mask", "k562_tssmask.npz"),
]
DNASE_LABEL = "DNase-only baseline (OLS)"
RADII = (128, 500, 2000)


def load(fname):
    d = np.load(os.path.join(MODEL_OUT, fname), allow_pickle=True)
    return d["genes"], d["preds"].reshape(-1), d["targets"].reshape(-1)


def dnase_baseline(test_genes):
    info = json.load(open(TSS_JSON))
    dnase = np.load(DNASE)
    genes = [g for g, v in info.items() if v["split"] in ("train", "test")]
    feats = {}
    for chrom in sorted({info[g]["chrom"] for g in genes}):
        if chrom not in dnase.files:
            continue
        track = dnase[chrom][0]
        csum = np.concatenate([[0.0], np.cumsum(track)])
        n = len(track)
        for g in genes:
            if info[g]["chrom"] != chrom:
                continue
            tss = info[g]["tss"]
            row = []
            for r in RADII:
                s, e = max(tss - r, 0), min(tss + r, n)
                row.append(np.log1p((csum[e] - csum[s]) / (e - s)))
            feats[g] = row

    train = [g for g in genes if info[g]["split"] == "train" and g in feats]
    X = np.c_[np.ones(len(train)), [feats[g] for g in train]]
    y = np.array([info[g]["counts"] for g in train])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.c_[np.ones(len(test_genes)), [feats[g] for g in test_genes]]
    return Xt @ beta


def main():
    genes, dec_pred, target = load(DECIMA[1])
    comps = []
    for label, fname in MAMBA:
        g, p, t = load(fname)
        assert (g == genes).all() and np.allclose(t, target), fname
        comps.append((label, p))
    comps.append((DNASE_LABEL, dnase_baseline(list(genes))))

    # Row 0: all test genes. Row 1: drop the near-zero blob, keeping observed ln(1+CPM) >= 1
    # (same 1,112-gene subset as the handoff's "> 1" filter; no target equals exactly 1).
    rows = [
        ("All test genes", np.ones(len(target), bool)),
        ("Observed ln(1+CPM) ≥ 1", target >= 1),
    ]
    # A few Decima genes have errors up to ~15; cap the axes and pin those to the edge.
    lim = 6.0

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.4), constrained_layout=True)
    for row_axes, (row_name, m) in zip(axes, rows):
        t = target[m]
        dec_err = np.abs(dec_pred[m] - t)
        r_dec = pearsonr(dec_pred[m], t)[0]
        print(f"--- {row_name} (n = {m.sum()})")
        for ax, (label, pred) in zip(row_axes, comps):
            p = pred[m]
            err = np.abs(p - t)
            r = pearsonr(p, t)[0]
            better = np.mean(err < dec_err)
            ax.plot([0, lim], [0, lim], color="0.55", lw=1, ls="--", zorder=1)
            n_clip = int(np.sum((dec_err > lim) | (err > lim)))
            ax.scatter(np.minimum(dec_err, lim), np.minimum(err, lim), s=7, alpha=0.35, color="#2a6fdb", edgecolors="none", zorder=2)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{DECIMA[0]}\n|pred − obs|   (r = {r_dec:.3f})")
            ax.set_ylabel(f"{label}\n|pred − obs|   (r = {r:.3f})")
            ax.set_title(f"vs {label}\n{row_name}, n = {m.sum()}", fontsize=10)
            ax.text(
                0.97, 0.03,
                f"below diagonal (better than Decima):\n{better:.0%} of genes\n"
                f"median |err|: {np.median(err):.2f} vs {np.median(dec_err):.2f}"
                + (f"\n{n_clip} gene(s) > {lim:g} pinned to edge" if n_clip else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.2",
            )
            ax.grid(color="0.92", lw=0.6)
            print(f"{label:35s} r={r:.4f} rho={spearmanr(p, t)[0]:.4f} better_than_decima={better:.3f}")
        print(f"{DECIMA[0]:35s} r={r_dec:.4f} rho={spearmanr(dec_pred[m], t)[0]:.4f}")

    fig.suptitle(
        "K562 test genes: per-gene absolute error, Decima vs each model "
        "(r = Pearson across the row's genes, predicted vs observed ln(1+CPM))",
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"decima_vs_cismamba_genewise.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
