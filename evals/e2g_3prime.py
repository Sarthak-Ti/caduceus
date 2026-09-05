#E2G for the bp-resolution 3' RNA-seq profile model (decoder yshape=524288, bin_size=1, d_output=2).
#Same experiment as e2g_borzoi.py -- knock down accessibility around a CRISPR element and read the change
#in predicted expression -- but the profile is reduced to scalars ON THE FLY instead of being saved.
#Saving the profile is not an option here: 10280 elements x 4 x 524288 x float32 = 86 GB.
#
#Differences from e2g_borzoi.py, all forced by the bp-resolution decoder / genome-style target:
#  - dataset override additional_tracks=None. The training config sets dataset.additional_tracks, so
#    __getitem__ returns outputs2 = [seq_unmask, acc_umask, rna_track] (general_dataset.py:518) and
#    e2g_borzoi.py's `((s, a), (su, au)) = dataset[idx]` raises on the 3-tuple. Nulling it also skips a
#    pointless bp-resolution npz slice per iteration. (additional_data is already null in this config.)
#  - no N_BINS. The decoder returns (batch, 524288, 2) = (window, [plus, minus]), not (6144, n_strands).
#  - one forward pass and one data load PER GENE for the unmasked prediction, not per element. With
#    center_tss the window and the unmasked accessibility depend only on the gene, so the 10280 rows
#    collapse to 2084 genes: 2084 unmasked passes + 10280 masked instead of 20560. Rows are grouped by
#    ensg_id to make this work, so the output is NOT in input row order -- see the row_idx column.
#  - masked elements of one gene are batched --batch_size at a time.
#
#Aggregation windows (all on the sense strand, with the antisense strand kept as a control that should
#be much less responsive). 3' RNA-seq piles reads up at the last exon / polyA site, so a TSS window is
#nearly empty of signal -- but whether this model actually learned that bias or just smears is an open
#question, so all three windows are saved and the data can decide:
#  - body: the collapsed gene body from CollapsedGeneBounds.hg38.bed, clipped to the window.
#  - tp:   the 3'-terminal --three_prime_len bp of the gene (gene end on +, gene start on -).
#  - tss:  TSS +/- --tss_len bp, the control that should NOT be the best window for 3' data.
#Gene bodies fit: median 8.9kb, p95 76kb, only 0.45% exceed the 262144bp half-window (body_truncated).
#
#Two output spaces. THE TARGET TRANSFORM IS ASSUMED TO BE `-1 + sqrt(1 + x)` -- the *_fixed npz
#written by K562_gex/make_fixed_npz.py, which is what every checkpoint from 2026-08-13 on was
#trained against. Loading a checkpoint trained on the pre-fix npz (plain sqrt(1+x), floor 1.0) is
#a hard error, see FIXED_TARGET_MARKER below; there is no way to detect it from the predictions.
#  - raw_*:  clip((pred + 1)**2 - 1, 0, None) summed = sum(pred**2 + 2*pred), the inverse of
#    -1 + sqrt(1+x), i.e. approximate read counts. The decoder ends in a softplus (decoders.py:471)
#    so pred > 0 and the clip never fires; it stays as a guard. Squaring is slightly biased
#    (Jensen: E[sqrt(1+x)]**2 != E[x]+1).
#    HISTORY: this used to be clip(pred**2 - 1, 0, None), the inverse of the *unfixed* sqrt(1+x).
#    Applied to a fixed-target checkpoint that zeroes every bp with pred <= 1 (x <= 3) and
#    understates the rest by 2*pred+1, which is not recoverable from the saved sums. csvs written
#    before this change carry the old meaning under the same column names -- check the `transform`
#    column, which exists from here on, before comparing across runs.
#  - sqrt_*: predictions summed as-is, in the space the model was trained in. Under the fixed
#    transform the no-signal baseline is 0, so this no longer accumulates a per-bp baseline the way
#    it did pre-fix, and it is exact rather than approximate. Prefer it for rank-based scoring.
#Sums are saved rather than only the ratios, so any pseudocount can be tried without rerunning.
#log2fc is negative for a true enhancer (knockdown lowers expression), so score with -log2fc.
print('E2G for bp-resolution 3prime RNA profile model', flush=True)

import numpy as np
import pandas as pd
import json
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
import torch
from tqdm import tqdm
import argparse
from src.dataloaders.datasets.general_dataset import GeneralDataset
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import registry as decoder_registry
from src.tasks.encoders import JointCNN
from caduceus.configuration_caduceus import CaduceusConfig
import yaml
from omegaconf import OmegaConf
import os
import inspect

try:
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except ValueError as e:
    if "Resolver already registered" in str(e):
        print("Resolver already exists, skipping registration.")

INPUT_LEN = 524288
HALF      = INPUT_LEN // 2

#row 0 = plus strand, row 1 = minus strand of K562_rnaseq_stranded_fixed.npz (bigwig_to_npz.py:7-8,
#which also documents the deliberate reverse/forward bigwig swap -- the "reverse" file is the +
#strand). make_fixed_npz.py preserves the row order.
STRAND_ROW = {'+': 0, '-': 1}

#the target npz filename must contain this, i.e. be the -1 + sqrt(1+x) version that reduce_profile
#inverts. Checked against the checkpoint's own training config in Evals.__init__.
FIXED_TARGET_MARKER = '_fixed'
#stamped into every output csv so a file is self-describing about which inverse built its raw_* cols
TARGET_TRANSFORM = 'minus1_plus_sqrt1p'


class Evals():
    def __init__(self,
                 ckpt_path,
                 dataset=None,
                 split='test',
                 device=None,
                 load_data=False,
                 **dataset_overrides
                 ) -> None:

        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)

        state_dict = torch.load(ckpt_path, map_location='cpu')
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        if dataset is None:
            dataset_args = self.cfg['dataset']
            assert dataset_args['mlm'] == 0 and dataset_args['acc_mlm'] == 0, "MLM and acc_mlm should be 0 for the training"
            sig = inspect.signature(GeneralDataset.__init__)
            sig = {k: v for k, v in sig.parameters.items() if k != 'self'}
            to_remove = []
            for k, v in dataset_args.items():
                if k not in sig:
                    to_remove.append(k)
            for k in to_remove:
                del dataset_args[k]
            dataset_args['split'] = split
            dataset_args['evaluating'] = True
            dataset_args['load_in'] = load_data
            dataset_args['additional_data'] = None
            #read the target path BEFORE nulling it below -- dataset_args aliases self.cfg['dataset']
            self.target_track_path = dataset_args.get('additional_tracks')
            #the target track is never needed here -- we only read predictions. Leaving it set would
            #make __getitem__ return a 3-element outputs2 and slice the npz at bp resolution 2084 times.
            dataset_args['additional_tracks'] = None

            for k, v in dataset_overrides.items():
                if k in sig:
                    dataset_args[k] = v
                    print(f"Overriding {k} with {v}")
                else:
                    print(f"Warning: {k} not in dataset args, skipping")

            self.dataset_args = dataset_args
            self.dataset = GeneralDataset(**dataset_args)
        else:
            self.dataset = dataset

        assert self.dataset.length == INPUT_LEN, \
            f"dataset.length is {self.dataset.length}, this script assumes {INPUT_LEN}"

        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )
        model_state_dict = state_dict["state_dict"]
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)
        decoder_state_dict = {}
        for key in list(model_state_dict.keys()):
            if "decoder" in key:
                decoder_state_dict[key[10:]] = model_state_dict.pop(key)
        encoder_state_dict = {}
        for key in list(model_state_dict.keys()):
            if "encoder" in key:
                encoder_state_dict[key[10:]] = model_state_dict.pop(key)

        cfg['model']['config'].pop('_target_')
        caduceus_cfg = CaduceusConfig(**cfg['model']['config'])

        self.backbone = DNAEmbeddingModelCaduceus(config=caduceus_cfg)
        self.backbone.load_state_dict(model_state_dict, strict=True)

        #build the decoder from the config's _name_ rather than hardcoding, so this also works if the
        #decoder class changes. For this run _name_ is 'enformer' -> EnformerDecoder.
        decoder_name = self.cfg['decoder'].pop('_name_')
        self.cfg['decoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.decoder = decoder_registry[decoder_name](**self.cfg['decoder'])
        self.decoder.load_state_dict(decoder_state_dict, strict=True)

        del self.cfg['encoder']['_name_']
        self.cfg['encoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.encoder = JointCNN(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)

        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()

        self._check_target_transform()

    def _check_target_transform(self):
        """Fail loudly unless this checkpoint was trained on the -1 + sqrt(1+x) target.

        reduce_profile's raw_* inverse is specific to that transform, and a checkpoint trained on
        the pre-fix npz produces predictions that look completely normal while making raw_* wrong
        in a way no downstream check can catch. So it is verified here rather than assumed.
        """
        path = getattr(self, 'target_track_path', None)
        if path is None:
            print("WARNING: no dataset.additional_tracks in the training config, cannot verify the "
                  f"target transform. raw_* assumes '{TARGET_TRANSFORM}' (-1 + sqrt(1+x)).",
                  flush=True)
            return
        print(f"target track: {path}", flush=True)
        assert FIXED_TARGET_MARKER in os.path.basename(path), (
            f"this checkpoint trained on {os.path.basename(path)}, whose name lacks "
            f"'{FIXED_TARGET_MARKER}'. This script assumes the target is -1 + sqrt(1+x) "
            f"(K562_gex/make_fixed_npz.py). For a plain sqrt(1+x) checkpoint the raw_* inverse "
            f"below is wrong -- either rerun make_fixed_npz.py and refinetune, or restore "
            f"clip(pred**2 - 1, 0, None) in reduce_profile and drop this assert."
        )

    def predict(self, seq, acc):
        """seq: (B, 6, INPUT_LEN), acc: (B, 2, INPUT_LEN) -> (B, INPUT_LEN, 2) float32 numpy"""
        seq, acc = seq.to(self.device), acc.to(self.device)
        with torch.no_grad():
            x, _ = self.encoder(seq, acc)
            x, _ = self.backbone(x)
            x = self.decoder(x)
        return x.float().cpu().numpy()


def load_benchmark():
    """The gs_df construction from e2g_borzoi.py, plus collapsed gene bodies."""
    TSS_bounds_file = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
    tss = pd.read_csv(TSS_bounds_file, sep="\t")

    path = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    gs_df = pd.read_csv(path, sep="\t")

    json_path = '/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json'
    with open(json_path, 'r') as f:
        tss_dict = json.load(f)

    name_to_ensg = tss.set_index('name')['Ensembl_ID'].to_dict()
    name_to_ensg = {name: ensg for name, ensg in name_to_ensg.items() if ensg in tss_dict}
    gs_df['ensg_id'] = gs_df['measuredGeneSymbol'].map(name_to_ensg)
    gs_df = gs_df.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)

    #collapsed gene bodies, keyed the same way as the TSS500bp file. A few Ensembl_IDs appear on
    #multiple rows, so take the outermost span.
    bounds = pd.read_csv("/data1/deyk/extras/CollapsedGeneBounds.hg38.bed", sep="\t")
    bounds = bounds.groupby('Ensembl_ID').agg(
        gene_start=('start', 'min'), gene_end=('end', 'max'), gene_strand=('strand', 'first'))
    missing = ~gs_df['ensg_id'].isin(bounds.index)
    if missing.any():
        print(f"dropping {missing.sum()} rows whose gene has no entry in CollapsedGeneBounds.hg38.bed")
        gs_df = gs_df[~missing].reset_index(drop=True)
    gs_df = gs_df.join(bounds, on='ensg_id')

    #tss/chrom/strand/split come from the json the model's expression targets were defined against
    gs_df['tss']    = gs_df['ensg_id'].map(lambda e: tss_dict[e]['tss'])
    gs_df['strand'] = gs_df['ensg_id'].map(lambda e: tss_dict[e]['strand'])
    gs_df['gene_chrom'] = gs_df['ensg_id'].map(lambda e: tss_dict[e]['chrom'])
    gs_df['gene_split'] = gs_df['ensg_id'].map(lambda e: tss_dict[e]['split'])

    disagree = (gs_df['gene_strand'] != gs_df['strand']).sum()
    if disagree:
        print(f"NOTE: {disagree} rows where CollapsedGeneBounds strand != json strand; using json strand")
    outside = ((gs_df['tss'] < gs_df['gene_start']) | (gs_df['tss'] > gs_df['gene_end'])).sum()
    if outside:
        print(f"NOTE: {outside} rows where the json TSS falls outside the collapsed gene body")
    return gs_df, tss_dict


def windows_for(row, seq_start, three_prime_len, tss_len):
    """Genomic aggregation intervals -> window-local [lo, hi) slices, clipped to the window.

    seq_start is the genomic coordinate of local index 0. __getitem__ with start==stop==tss and
    length=INPUT_LEN gives start = tss - INPUT_LEN//2, and any out-of-chromosome part is zero-padded
    in place, so local index 0 is always tss - HALF even at chromosome ends.
    """
    tss, gs, ge, strand = row.tss, row.gene_start, row.gene_end, row.strand
    if strand == '+':
        tp = (max(gs, ge - three_prime_len), ge)   # 3' end is the gene end
    else:
        tp = (gs, min(ge, gs + three_prime_len))   # 3' end is the gene start
    out = {}
    for name, (a, b) in [('body', (gs, ge)), ('tp', tp), ('tss', (tss - tss_len, tss + tss_len))]:
        lo = max(0, a - seq_start)
        hi = min(INPUT_LEN, b - seq_start)
        out[name] = (lo, hi)
    return out


def reduce_profile(pred, wins, sense_row):
    """pred: (INPUT_LEN, 2) -> {space}_{win}_{sense|anti} sums.

    raw_*  inverts the target transform -1 + sqrt(1+x), i.e. x = (pred+1)**2 - 1, giving approximate
    counts. sqrt_* leaves the model's own space, whose no-signal baseline is already 0 under that
    transform. The clip never fires (softplus output) and is only a guard.
    """
    raw = np.clip((pred.astype(np.float64) + 1.0) ** 2 - 1.0, 0, None)
    anti_row = 1 - sense_row
    out = {}
    for win, (lo, hi) in wins.items():
        if hi <= lo:  # window entirely outside the sequence, shouldn't happen but stays defined
            for space in ('raw', 'sqrt'):
                out[f'{space}_{win}_sense'] = np.nan
                out[f'{space}_{win}_anti'] = np.nan
            continue
        out[f'raw_{win}_sense']  = float(raw[lo:hi, sense_row].sum())
        out[f'raw_{win}_anti']   = float(raw[lo:hi, anti_row].sum())
        out[f'sqrt_{win}_sense'] = float(pred[lo:hi, sense_row].sum())
        out[f'sqrt_{win}_anti']  = float(pred[lo:hi, anti_row].sum())
    return out


def main(args):
    gs_df, _ = load_benchmark()
    if args.limit_genes is not None:
        keep = gs_df.ensg_id.drop_duplicates().head(args.limit_genes)
        gs_df = gs_df[gs_df.ensg_id.isin(keep)].reset_index(drop=True)
        print(f"--limit_genes {args.limit_genes}: smoke test on {len(gs_df)} elements")
    print(f"{len(gs_df)} elements, {gs_df.ensg_id.nunique()} genes, "
          f"{int(gs_df.Regulated.sum())} positives")
    print(gs_df.gene_split.value_counts().to_string())

    evals_kwargs = dict(load_data=args.load_data)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path
    evals = Evals(args.ckpt_path, **evals_kwargs)

    records = []
    #group by gene: the window, the sequence and the unmasked accessibility are gene-specific, so
    #they are fetched and predicted once and reused across that gene's elements
    for ensg, group in tqdm(gs_df.groupby('ensg_id', sort=False), total=gs_df.ensg_id.nunique()):
        g0 = group.iloc[0]
        chrom, tss, strand = g0.gene_chrom, int(g0.tss), g0.strand
        seq_start = tss - HALF
        sense_row = STRAND_ROW[strand]

        idx = evals.dataset.expand_seqs(chrom, tss, tss)
        (seq, acc), _ = evals.dataset[idx]

        pred_before = evals.predict(seq.unsqueeze(0), acc.unsqueeze(0))[0]

        rows = list(group.itertuples())
        for chunk_start in range(0, len(rows), args.batch_size):
            chunk = rows[chunk_start:chunk_start + args.batch_size]
            accs = []
            in_ctx = []
            for row in chunk:
                start, end = int(row.chromStart), int(row.chromEnd)
                a = acc.clone()
                #the element has to sit inside the window for the knockdown to mean anything; if it
                #does not, the masked prediction is identical to the unmasked one and log2fc is 0
                ok = (start >= seq_start) and (end <= seq_start + INPUT_LEN)
                if ok:
                    lo = max(0, (start - seq_start) - args.dist_additional_mask)
                    hi = min(INPUT_LEN, (end - seq_start) + args.dist_additional_mask)
                    a[0, lo:hi] = a[0, lo:hi] / args.scale_factor
                accs.append(a)
                in_ctx.append(ok)

            batch_acc = torch.stack(accs)
            batch_seq = seq.unsqueeze(0).expand(len(chunk), -1, -1)
            pred_after = evals.predict(batch_seq, batch_acc)

            for j, row in enumerate(chunk):
                wins = windows_for(row, seq_start, args.three_prime_len, args.tss_len)
                rec = {
                    'row_idx': row.Index, 'chrom': row.chrom,
                    'chromStart': int(row.chromStart), 'chromEnd': int(row.chromEnd),
                    'measuredGeneSymbol': row.measuredGeneSymbol, 'ensg_id': ensg,
                    'Regulated': bool(row.Regulated), 'gene_split': row.gene_split,
                    'tss': tss, 'strand': strand,
                    'gene_start': int(row.gene_start), 'gene_end': int(row.gene_end),
                    'in_context': in_ctx[j],
                    #a gene longer than the half-window gets its body clipped, so its body sum covers
                    #only part of the gene. 0.45% of rows.
                    'body_truncated': bool(row.gene_start < seq_start or
                                           row.gene_end > seq_start + INPUT_LEN),
                    'body_bp': wins['body'][1] - wins['body'][0],
                }
                for k, v in reduce_profile(pred_before, wins, sense_row).items():
                    rec[f'before_{k}'] = v
                for k, v in reduce_profile(pred_after[j], wins, sense_row).items():
                    rec[f'after_{k}'] = v
                records.append(rec)

    df = pd.DataFrame(records)
    #stamp the inverse used for raw_*, so a csv can be compared across runs without guessing
    df['transform'] = TARGET_TRANSFORM

    #convenience log2fc columns; the sums are saved too so a different pseudocount needs no rerun.
    #negative = knockdown lowered predicted expression = enhancer-like, so score with -log2fc.
    for space in ('raw', 'sqrt'):
        for win in ('body', 'tp', 'tss'):
            for st in ('sense', 'anti'):
                b = df[f'before_{space}_{win}_{st}']
                a = df[f'after_{space}_{win}_{st}']
                df[f'log2fc_{space}_{win}_{st}'] = np.log2((a + args.pseudocount) /
                                                           (b + args.pseudocount))

    out_path = os.path.join(args.output_dir, args.output + '.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {len(df)} rows to {out_path}')
    print(f"in_context: {df.in_context.sum()}/{len(df)}   body_truncated: {df.body_truncated.sum()}")

    #quick sanity readout: mean log2fc on positives vs negatives per window. Positives should be more
    #negative than negatives on the sense strand, and the antisense control should separate less.
    print('\nmean log2fc by Regulated (want positives more negative):')
    cols = [c for c in df.columns if c.startswith('log2fc_')]
    print(df.groupby('Regulated')[cols].mean().T.to_string())
    print('\nE2G 3prime run complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2G for the bp-resolution 3' RNA-seq profile model")
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='k562_3prime_e2g',
                        help='Output filename (no extension)')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/lesliec/sarthak/data/joint_playground/e2g/',
                        help='Directory to save the output csv')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by in the masked region')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp to mask on each side of the element')
    parser.add_argument('--three_prime_len', type=int, default=10000,
                        help="Length of the 3'-terminal aggregation window")
    parser.add_argument('--tss_len', type=int, default=2000,
                        help='Half-width of the TSS aggregation window')
    parser.add_argument('--pseudocount', type=float, default=1.0,
                        help='Pseudocount added to both sums before the log2 ratio')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Masked elements per forward pass. Raise if it fits; the decoder holds a '
                             '(B, 524288, 512) fp32 intermediate, ~1.1GB per element.')
    parser.add_argument('--limit_genes', type=int, default=None,
                        help='Only run the first N genes. For a cheap smoke test of the forward pass.')
    parser.add_argument('--load_data', action='store_true',
                        help='Load genome and accessibility into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to accessibility data (overrides dataset config)')
    args = parser.parse_args()

    print(args)
    main(args)
