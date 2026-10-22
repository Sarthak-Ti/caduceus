"""Cell-type specificity benchmark for the PANC RNA fine-tuned models.

Plan: claude_summaries/panc_finetune_plan_20261020.md. Four steps, each a subcommand:

  features    Per (gene, cell type) ATAC features from the PANC ATAC store, for every gene in the
              json: mean signal in TSS +-128 / 500 / 2000 bp and over the gene body. Cached once;
              used by the baselines and the gene-set thresholds. CPU.
  thresholds  Distributions on TRAINING genes of the quantities the gene sets are defined by, to
              fix the thresholds before any model result is seen. CPU.
  predict     Run one checkpoint on (gene x cell types) of a split, saving the count head and
              the profile sum (sense rate over the gene body plus 1 kb past the 3' end) with the
              matching targets, plus the full predicted profiles (float16) unless --no_profiles,
              so other profile readouts need no GPU. --celltypes all13 (default) scores the 13 training cell types, so
              a DE-only checkpoint is run on every cell type's ATAC (the ATAC swap test); 'trained'
              uses the checkpoint's own cell types. --trans_row_override gives every sample one
              cell type's trans vector. GPU.
  score       Metrics for prediction files, the ATAC baselines (OLS fit on training genes) and the
              observed profile sum, on all / variable / hard genes (+ strict versions). Writes a
              summary (one row per model and readout), every metric, and a per-cell-type table.
              --celltypes restricts the scored cell types; with one (e.g. DE) the specificity
              metrics are skipped and only the across-gene ones remain. CPU.

Metrics (rows = genes, columns = cell types, values ln(1+CPM)); see metrics() for definitions:
  overall_r, per_celltype_r      ordinary accuracy, always reported
  per_gene_r_*, double_centered_r  pattern of changes between cell types (2+ cell types)
  change_slope, change_rmse      size of those changes: slope 1 = right size, < 1 = too small
  partial_r_given_promoter       what the model explains beyond the promoter-ATAC baseline
Gene sets: variable = expression SD across the 13 cell types >= VARIABLE_SD; hard = variable genes
whose promoter ATAC is open in every cell type and nearly constant across them.
Profile sums are compared as ln(1 + sum).

Run (from the caduceus repo, modern env):
  pixi run --manifest-path /data1/lesliec/sarthak/triton_upgrade_test/pixi.toml python -u \
      evals/panc_specificity.py features
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ROOT = Path('/data1/lesliec/sarthak/caduceus')
sys.path.append(str(ROOT))
MULTIOME = '/data1/lesliec/sarthak/data/DE_danwei/multiome'
ATAC = f'{MULTIOME}/panc_atac.zarr/panc'
JSON = f'{MULTIOME}/panc_sc_rna_info.json'
IDX13 = f'{MULTIOME}/idx_lists/panc_train13.json'
OUT = ROOT / 'outputs/panc_eval'
FEATURES = OUT / 'atac_features.npz'
PROMOTER_RADII = (128, 500, 2000)
FEATURE_NAMES = [f'prom{r}' for r in PROMOTER_RADII] + ['body']
MIN_CPM = 10
EXTEND_3PRIME_BINS = 8  # 1 kb past the 3' end at 128 bp, where 3' reads pile up
POOL = 128

# Gene-set thresholds, fixed from the training-gene distributions (`thresholds`) before scoring.
# Fixed 2026-10-22 from 9,397 expressed training genes, before any model was scored. Test genes kept:
# variable 804 of 1,263, hard 335 (median r(expression, promoter ATAC) across cell types 0.003);
# strict: variable 466, hard 132.
VARIABLE_SD = 0.3            # SD across the 13 cell types of ln(1+CPM) (q50 0.39 in training genes)
VARIABLE_SD_STRICT = 0.5     # sensitivity set
CONSTANT_PROMOTER_SD = 0.10  # SD across cell types of ln(1 + mean ATAC in TSS +-500) (q50 0.091)
OPEN_PROMOTER_MIN = 0.52     # min over cell types of ln(1 + mean ATAC in TSS +-500) (q25)


def load_genes():
    with open(JSON) as f:
        genes = json.load(f)
    with open(IDX13) as f:
        rows = json.load(f)
    names = list(zarr.open(ATAC, mode='r').attrs['celltypes'])
    return genes, rows, names


def expressed(genes, names13):
    """The training gene filter: CPM >= MIN_CPM in at least one of the 13 cell types."""
    min_log = np.log1p(MIN_CPM)
    return [g for g, v in genes.items() if max(v['counts'][n] for n in names13) >= min_log]


# ---------------------------------------------------------------------------------- features
def features(args):
    genes, rows, names = load_genes()
    ids = list(genes)
    store = zarr.open(ATAC, mode='r')
    feats = np.full((len(ids), len(rows), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    by_chrom = {}
    for i, g in enumerate(ids):
        by_chrom.setdefault(genes[g]['chrom'], []).append(i)
    for chrom, gene_idx in sorted(by_chrom.items()):
        for j, row in enumerate(rows):
            track = np.asarray(store[chrom][row])  # whole chromosome row, float16
            n = track.shape[0]
            for i in gene_idx:
                v = genes[ids[i]]
                windows = [(v['tss'] - r, v['tss'] + r) for r in PROMOTER_RADII] + [(v['gene_start'], v['gene_end'])]
                for k, (lo, hi) in enumerate(windows):
                    lo, hi = max(0, lo), min(n, hi)
                    feats[i, j, k] = track[lo:hi].astype(np.float32).mean()
        print(f'{chrom}: {len(gene_idx)} genes', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURES, genes=np.array(ids), rows=np.array(rows), celltypes=np.array([names[r] for r in rows]),
             feature_names=np.array(FEATURE_NAMES), features=feats)
    print(f'wrote {FEATURES}: {feats.shape}')


def load_features():
    with np.load(FEATURES) as f:
        return {k: f[k] for k in f.files}


def expression_matrix(genes, ids, names13):
    return np.array([[genes[g]['counts'][n] for n in names13] for g in ids])


# -------------------------------------------------------------------------------- thresholds
def thresholds(args):
    genes, rows, names = load_genes()
    names13 = [names[r] for r in rows]
    f = load_features()
    index = {g: i for i, g in enumerate(f['genes'])}
    train = [g for g in expressed(genes, names13) if genes[g]['split'] == 'train']
    y = expression_matrix(genes, train, names13)
    prom = np.log1p(f['features'][[index[g] for g in train], :, FEATURE_NAMES.index('prom500')])
    expr_sd = y.std(axis=1)
    prom_sd, prom_min = prom.std(axis=1), prom.min(axis=1)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    table = pd.DataFrame({
        'expression SD across cell types': np.quantile(expr_sd, qs),
        'promoter ATAC SD across cell types': np.quantile(prom_sd, qs),
        'promoter ATAC min over cell types': np.quantile(prom_min, qs),
    }, index=[f'q{int(q * 100)}' for q in qs]).T
    pd.set_option('display.width', 200)
    print(f'{len(train)} expressed training genes\n')
    print(table.round(3))
    print('\nfraction of training genes kept as the threshold varies:')
    for t in (0.3, 0.5, 0.75, 1.0):
        print(f'  variable (expression SD >= {t}): {(expr_sd >= t).mean():.2f}')
    for t in (0.1, 0.2, 0.3):
        print(f'  constant promoter (ATAC SD <= {t}): {(prom_sd <= t).mean():.2f}')
    # how much of the expression spread a promoter can explain, by promoter ATAC SD tercile
    r = np.array([np.corrcoef(y[i], prom[i])[0, 1] if prom[i].std() > 0 else np.nan for i in range(len(train))])
    terciles = np.quantile(prom_sd, [1 / 3, 2 / 3])
    for lab, m in [('low', prom_sd <= terciles[0]), ('mid', (prom_sd > terciles[0]) & (prom_sd <= terciles[1])),
                   ('high', prom_sd > terciles[1])]:
        print(f'  promoter ATAC SD tercile {lab}: median per-gene r(expression, promoter ATAC) = '
              f'{np.nanmedian(r[m]):.3f}, median expression SD = {np.median(expr_sd[m]):.3f}')


# ----------------------------------------------------------------------------------- predict
def predict(args):
    """Model loading follows evals/evals_utils_joint.Evals (the striped-checkpoint loader used by the
    dsQTL and CAGE evaluations): the run's .hydra/config.yaml, DNAEmbeddingModelStriped, and strict
    loads of encoder, backbone and decoder. Evals.__call__ passes neither trans_info nor the gene
    masks, so its loaded modules are called directly here, as JointTSS.forward does in training."""
    import torch
    from torch.utils.data import DataLoader, default_collate
    from src.dataloaders.datasets.tss_dataset import TSSDataset
    sys.path.insert(0, str(ROOT / 'evals'))
    from evals_utils_joint import Evals

    ckpt = best_checkpoint(Path(args.ckpt)) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f'checkpoint: {ckpt}', flush=True)
    ev = Evals(str(ckpt), dataset=object(), device='cuda')  # dataset built below from the same config
    cfg = ev.cfg
    if args.compile:
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True  # eager-matching bf16 rounding, as in dsqtl_compiled.py
        ev.encoder, ev.backbone, ev.decoder = (torch.compile(m) for m in (ev.encoder, ev.backbone, ev.decoder))

    # Scoring always covers the 13 cell types; a DE-only checkpoint thereby gets each cell type's ATAC
    data = {k: v for k, v in cfg['dataset'].items()
            if k not in ('_name_', 'dataset_name', 'batch_size', 'batch_size_eval', 'shuffle',
                         'pin_memory', 'num_workers', 'val_max_genes', 'val_gene_seed', 'fault_tolerant',
                         'ddp', 'drop_last')}
    trained_rows = _trained_rows(cfg)
    if args.celltypes == 'all13':
        with open(IDX13) as f:
            rows = json.load(f)
    elif args.celltypes == 'trained':
        rows = trained_rows
    else:
        store_names = list(zarr.open(ATAC, mode='r').attrs['celltypes'])
        rows = [store_names.index(n) for n in args.celltypes.split(',')]
    data.update(data_idxs=rows, evaluating=True, max_genes=args.max_genes, gene_seed=0, load_in=False)
    if cfg['encoder'].get('ctt'):
        # the token is a position in the model's cell type list, not in the list being scored
        data.update(return_celltype_idx_og=True, ctt_rows=data.get('ctt_rows') or trained_rows)
    if args.trans_row_override is not None:
        assert data.get('trans_info'), 'trans_row_override needs a model with trans_info'
        data['trans_row_override'] = args.trans_row_override
    ds = TSSDataset(split=args.split, **data)
    n_genes, n_ct = len(ds.genes), len(ds.celltype_rows)
    names13 = [ds.row_names[r] for r in ds.celltype_rows]
    strands = np.array([1 if ds.tss_dict[g]['strand'] == '+' else -1 for g in ds.genes])

    count_pred = np.full((n_genes, n_ct), np.nan, dtype=np.float32)
    profile_sum_pred = np.full((n_genes, n_ct), np.nan, dtype=np.float32)
    count_target = np.full((n_genes, n_ct), np.nan, dtype=np.float32)
    profile_sum_target = np.full((n_genes, n_ct), np.nan, dtype=np.float32)
    # full predicted sense profiles (float16, ~130 MB for test x 13), so other profile readouts can be
    # computed later without a GPU; observed profiles can be re-read from the RNA store
    profile_pred = None if args.no_profiles else np.zeros((n_genes, n_ct, 0), dtype=np.float16)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        collate_fn=default_collate)
    index = 0
    with torch.no_grad():
        for x, y in loader:
            x = [t.to(ev.device, non_blocking=True) for t in x]
            y = [t.to(ev.device, non_blocking=True) for t in y]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # encoder(*x): (seq, acc[, celltype token[, trans_info]]), exactly as in training
                h, intermediates = ev.encoder(*x)
                h, _ = ev.backbone(h)
                profile, counts = ev.decoder(h, intermediates=intermediates, mask=y[3], gene_mask=y[4])
            profile = profile.float()[:, 0]          # (B, 4096) sense rate
            expression = y[6].float()[:, 0]          # (B, 4096) observed sense coverage
            gene_mask = y[4].float()                 # (B, 4096)
            if profile_pred is not None and profile_pred.shape[2] == 0:
                profile_pred = np.zeros((n_genes, n_ct, profile.shape[1]), dtype=np.float16)
            for b in range(profile.shape[0]):
                ct, g = divmod(index, n_genes)
                if profile_pred is not None:
                    profile_pred[g, ct] = profile[b].cpu().numpy().astype(np.float16)
                region = readout_region(gene_mask[b], strands[g])
                count_pred[g, ct] = counts[b, 0].float().item()
                profile_sum_pred[g, ct] = profile[b, region].sum().item()
                profile_sum_target[g, ct] = expression[b, region].sum().item()
                count_target[g, ct] = y[2][b].item()
                index += 1
            if index % (50 * args.batch_size) < args.batch_size:
                print(f'{index}/{len(ds)}', flush=True)
    assert index == len(ds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, genes=np.array(ds.genes), celltypes=np.array(names13), count_pred=count_pred,
             profile_sum_pred=profile_sum_pred, count_target=count_target,
             profile_sum_target=profile_sum_target, ckpt=str(ckpt), split=args.split,
             trans_row_override=-1 if args.trans_row_override is None else args.trans_row_override,
             trained_celltypes=np.array([ds.row_names[r] for r in trained_rows]),
             strands=strands, **({} if profile_pred is None else {'profile_pred': profile_pred}))
    print(f'wrote {out}')


def best_checkpoint(run_dir):
    """Minimum-validation-loss checkpoint of a run directory (checkpoints/<epoch>-val_loss=<loss>.ckpt)."""
    import re
    found = []
    for path in (run_dir / 'checkpoints').glob('*-val_loss=*.ckpt'):
        m = re.fullmatch(r'(\d+)-val_loss=(.+)\.ckpt', path.name)
        if m:
            found.append((float(m.group(2)), int(m.group(1)), path))
    assert found, f'no validated checkpoints in {run_dir}/checkpoints'
    return min(found)[2]


def _trained_rows(cfg):
    from src.dataloaders.datasets.tss_dataset import get_data_idxs
    return [int(r) for r in get_data_idxs(cfg['dataset']['data_idxs'], None)]


def readout_region(gene_mask, strand):
    """Gene-body bins plus EXTEND_3PRIME_BINS past the 3' end, in the (unflipped) eval orientation:
    the 3' end is the right edge of the mask for + genes and the left edge for - genes."""
    import torch
    on = torch.nonzero(gene_mask).squeeze(1)
    lo, hi = int(on.min()), int(on.max()) + 1
    if strand == 1:
        hi = min(gene_mask.shape[0], hi + EXTEND_3PRIME_BINS)
    else:
        lo = max(0, lo - EXTEND_3PRIME_BINS)
    return slice(lo, hi)


# ------------------------------------------------------------------------------------- score
def corr(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1] if a.size > 1 and a.std() > 0 and b.std() > 0 else np.nan


def pearson_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def double_center(m):
    return m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()


def partial_corr(x, y, z):
    """Pearson of x and y after regressing z (with an intercept) out of both."""
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    if np.allclose(x, z):  # the promoter baseline itself: nothing is left once it is regressed out
        return np.nan
    design = np.column_stack([np.ones_like(z), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return corr(rx, ry)


def metrics(pred, target, promoter):
    """pred, target, promoter (the promoter-ATAC baseline's prediction): (genes, cell types).

    Always: overall_r over all values, per_celltype_r (across genes within a cell type, averaged), and
    partial_r_given_promoter. With 2+ cell types also the specificity metrics, on double-centered
    values (gene and cell-type means removed), so per-cell-type offsets such as ATAC depth drop out:
      per_gene_r_*       Pearson across cell types per gene (median, mean, fraction > 0)
      double_centered_r  pattern of changes between cell types
      change_slope       slope of predicted on observed changes: 1 = right size, < 1 = too small
      change_rmse        error on the changes (ln units for the count head)
      partial_r_given_promoter  changes explained beyond the promoter-ATAC baseline; with one cell
                         type, expression across genes explained beyond it
    For the profile-sum readout (ln(1 + summed rate)) change_slope and change_rmse are on that scale.
    """
    out = {'overall_r': corr(pred, target), 'per_celltype_r': np.nanmean(pearson_rows(pred.T, target.T))}
    if pred.shape[1] == 1:
        out['partial_r_given_promoter'] = partial_corr(pred, target, promoter)
        return out
    per_gene = pearson_rows(pred, target)
    dp, dt, db = double_center(pred), double_center(target), double_center(promoter)
    out.update({
        'per_gene_r_median': np.nanmedian(per_gene), 'per_gene_r_mean': np.nanmean(per_gene),
        'per_gene_r_frac_pos': np.nanmean(per_gene > 0),
        'double_centered_r': corr(dp, dt),
        'change_slope': (dp * dt).sum() / (dt ** 2).sum(),
        'change_rmse': np.sqrt(np.mean((dp - dt) ** 2)),
        'partial_r_given_promoter': partial_corr(dp, dt, db),
    })
    return out


def per_celltype(pred, target, names):
    """Per cell type: r across genes on raw values, and on double-centered values (which genes are up
    in this cell type relative to their own average) when there are 2+ cell types."""
    rows = []
    dp, dt = (double_center(pred), double_center(target)) if pred.shape[1] > 1 else (None, None)
    for j, name in enumerate(names):
        rows.append({'celltype': name, 'r': corr(pred[:, j], target[:, j]),
                     'double_centered_r': corr(dp[:, j], dt[:, j]) if dp is not None else np.nan})
    return rows


def gene_sets(genes, ids, feats_for):
    """Defined on all 13 training cell types whatever subset is scored: they are properties of genes."""
    _, rows, names = load_genes()
    names13 = [names[r] for r in rows]
    sd = expression_matrix(genes, ids, names13).std(axis=1)
    prom = np.log1p(feats_for(ids)[:, :, FEATURE_NAMES.index('prom500')])
    constant_open = (prom.std(axis=1) <= CONSTANT_PROMOTER_SD) & (prom.min(axis=1) >= OPEN_PROMOTER_MIN)
    return {'all': np.ones(len(ids), bool),
            'variable': sd >= VARIABLE_SD, 'hard': (sd >= VARIABLE_SD) & constant_open,
            'variable_strict': sd >= VARIABLE_SD_STRICT, 'hard_strict': (sd >= VARIABLE_SD_STRICT) & constant_open}


def baselines(genes, names, feats_for, test_ids):
    """OLS on ln(1 + mean ATAC) features, fit on expressed training genes pooled over the scored cell
    types, with the features of each (gene, cell type) predicting that cell type's expression."""
    _, rows13, store_names = load_genes()
    train_ids = [g for g in expressed(genes, [store_names[r] for r in rows13]) if genes[g]['split'] == 'train']
    col = [list(load_features()['celltypes']).index(n) for n in names]
    out = {}
    for name, feats in [('baseline_promoter_atac', [0, 1, 2]), ('baseline_promoter_body_atac', [0, 1, 2, 3])]:
        def design(ids):
            x = np.log1p(feats_for(ids)[:, col][:, :, feats]).reshape(-1, len(feats))
            return np.hstack([np.ones((len(x), 1)), x])
        coef, *_ = np.linalg.lstsq(design(train_ids), expression_matrix(genes, train_ids, names).ravel(), rcond=None)
        out[name] = (design(test_ids) @ coef).reshape(len(test_ids), len(names))
    return out


SUMMARY_METRICS = ['overall_r', 'per_celltype_r', 'double_centered_r', 'per_gene_r_median', 'change_slope',
                   'partial_r_given_promoter']


def score(args):
    genes, _, _ = load_genes()
    f = load_features()
    index = {g: i for i, g in enumerate(f['genes'])}
    feats_for = lambda ids: f['features'][[index[g] for g in ids]]
    rows_out, ct_rows, done_baselines = [], [], set()

    def add(label, readout, pred, target, promoter, sets, names, file):
        for s, m in sets.items():
            if m.sum() < 3:
                continue
            base = {'file': file, 'model': label, 'readout': readout, 'gene_set': s, 'n_genes': int(m.sum()),
                    'n_celltypes': len(names)}
            rows_out.append({**base, **metrics(pred[m], target[m], promoter[m])})
            ct_rows.extend({**base, **r} for r in per_celltype(pred[m], target[m], names))

    for path in args.preds:
        p = np.load(path)
        ids, names = list(p['genes']), list(p['celltypes'])
        cols = list(range(len(names)))
        if args.celltypes:
            keep = args.celltypes.split(',')
            missing = [n for n in keep if n not in names]
            assert not missing, f'{path} lacks cell types {missing}'
            cols, names = [names.index(n) for n in keep], keep
        target = expression_matrix(genes, ids, names)
        assert np.allclose(target, p['count_target'][:, cols], atol=1e-5), f'{path}: targets disagree with the json'
        sets = gene_sets(genes, ids, feats_for)
        base_preds = baselines(genes, names, feats_for, ids)
        promoter = base_preds['baseline_promoter_atac']
        key = (tuple(ids), tuple(names))
        if key not in done_baselines:  # baselines and the profile-readout ceiling, once per gene/cell-type set
            done_baselines.add(key)
            for name, pred in base_preds.items():
                add(name, 'ols', pred, target, promoter, sets, names, '-')
            add('observed_profile_sum', 'profile_sum', np.log1p(p['profile_sum_target'][:, cols]), target,
                promoter, sets, names, '-')
        label = Path(path).stem
        for readout, pred in [('count', p['count_pred'][:, cols]), ('profile_sum', np.log1p(p['profile_sum_pred'][:, cols]))]:
            add(label, readout, pred, target, promoter, sets, names, str(path))

    table, ct_table = pd.DataFrame(rows_out), pd.DataFrame(ct_rows)
    summary_sets = [s for s in ('all', 'hard') if s in set(table['gene_set'])]
    cols = [c for c in SUMMARY_METRICS if c in table.columns]
    summary = (table[table['gene_set'].isin(summary_sets)]
               .pivot_table(index=['model', 'readout', 'n_celltypes'], columns='gene_set', values=cols, sort=False))
    summary.columns = [f'{s}:{m}' for m, s in summary.columns]
    summary = summary[[f'{s}:{m}' for s in summary_sets for m in cols if f'{s}:{m}' in summary.columns]]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 50)
    print(summary.round(3).to_string())
    if args.full:
        print(table.round(3).to_string(index=False))
    if args.out:
        prefix = Path(args.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(f'{prefix}_metrics.tsv', sep='\t', index=False)
        ct_table.to_csv(f'{prefix}_per_celltype.tsv', sep='\t', index=False)
        summary.to_csv(f'{prefix}_summary.tsv', sep='\t')
        print(f'wrote {prefix}_metrics.tsv, {prefix}_per_celltype.tsv, {prefix}_summary.tsv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('features')
    sub.add_parser('thresholds')
    p = sub.add_parser('predict')
    p.add_argument('--ckpt', required=True, help='checkpoint, or a run directory to use its min-val-loss checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--celltypes', default='all13',
                   help="'all13' (DE-only checkpoints then get each cell type's ATAC), 'trained' (the "
                        "checkpoint's own cell types), or comma-separated store names")
    p.add_argument('--trans_row_override', type=int, default=None)
    p.add_argument('--max_genes', type=int, default=None, help='seeded gene subset, for quick checks')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=7)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--no_profiles', action='store_true', help='do not save the full predicted profiles')
    s = sub.add_parser('score')
    s.add_argument('preds', nargs='+')
    s.add_argument('--celltypes', default=None,
                   help='comma-separated cell types to score (e.g. DE); one cell type skips the specificity metrics')
    s.add_argument('--out', default=None, help='output prefix for _metrics/_per_celltype/_summary .tsv')
    s.add_argument('--full', action='store_true', help='also print every metric for every gene set')
    args = parser.parse_args()
    {'features': features, 'thresholds': thresholds, 'predict': predict, 'score': score}[args.cmd](args)
