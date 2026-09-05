#e2g model with borzoi output (stranded CAGE/RNA)
#variant of e2g_borzoi.py that can also dinucleotide shuffle the element instead of, or as well as,
#scaling down its accessibility. See --perturbation.
print('E2G borzoi evaluation (dinucleotide shuffle)', flush=True)

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
from src.tasks.decoders import EnformerDecoder
from src.tasks.encoders import JointCNN
from evals.utils.dinuc_shuffle import dinucleotide_shuffle
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

SEQ_LEN  = 196608
BIN_SIZE = 32
N_BINS   = SEQ_LEN // BIN_SIZE  # 6144


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

        del self.cfg['decoder']['_name_']
        self.cfg['decoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.decoder = EnformerDecoder(**self.cfg['decoder'])
        self.decoder.load_state_dict(decoder_state_dict, strict=True)

        del self.cfg['encoder']['_name_']
        self.cfg['encoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.encoder = JointCNN(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)

        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()

    def __call__(self, idx=None, data=None):
        if data is None:
            (seq, acc), _ = self.dataset[idx]
            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
        else:
            (x, y), _ = data
            if x.dim() == 2:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            x1, _ = self.encoder(x, y)
            x1, _ = self.backbone(x1)
            x1 = self.decoder(x1)

        return x1  # shape: [batch, N_BINS, n_strands]


def main(args):
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

    n_strands = 4 if args.assay_type == 'both' else 2

    #row 0 of each batch is the unperturbed reference, rows 1..n_alt are the perturbed versions
    n_alt = args.n_shuffles if args.perturbation in ('shuffle', 'both_perturbations') else 1

    #by default collapse the shuffles to their mean, so the output keeps the original row layout.
    #--save_all_shuffles keeps every individual shuffle instead
    average_shuffles = args.perturbation in ('shuffle', 'both_perturbations') and not args.save_all_shuffles
    n_blocks = 2 if average_shuffles else 1 + n_alt

    # Output layout: n_blocks blocks of n_strands rows, block 0 is the unperturbed 'before',
    # blocks 1.. are the perturbed 'after' (a single mean block unless --save_all_shuffles).
    #   CAGE or RNA  → 4 rows per element: [before+, before-, after+, after-]
    #   both         → 8 rows per element: [before_CAGE+, before_CAGE-, before_RNA+, before_RNA-,
    #                                        after_CAGE+,  after_CAGE-,  after_RNA+,  after_RNA-]
    n_rows = n_blocks * n_strands
    outputs = np.zeros((len(gs_df), n_rows, N_BINS), dtype=np.float32)
    in_context = np.ones(len(gs_df), dtype=bool)
    shuffled = np.zeros(len(gs_df), dtype=bool) #whether the element actually got dinucleotide shuffled
    #spread across the shuffles, otherwise averaging throws that information away
    shuffle_std = np.zeros((len(gs_df), n_strands, N_BINS), dtype=np.float32) if average_shuffles else None

    evals_kwargs = dict(load_data=args.load_data, additional_data=None)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path

    evals = Evals(args.ckpt_path, **evals_kwargs)

    for i, row in tqdm(gs_df.iterrows(), total=len(gs_df)):
        chrom  = row['chrom']
        start  = row['chromStart']
        end    = row['chromEnd']
        ensgid = row['ensg_id']
        temp_tss = tss_dict[ensgid]['tss']

        if args.center_tss:
            # Center the window at the TSS; check if enhancer falls within the window
            in_context[i] = 1 if (start >= temp_tss - HALF and end <= temp_tss + HALF) else 0
            idx = evals.dataset.expand_seqs(chrom, temp_tss, temp_tss)
        else:
            idx = evals.dataset.expand_seqs(chrom, start, end)

        ((s, a), (su, au)) = evals.dataset[idx]

        s = s.unsqueeze(0).repeat(1 + n_alt, 1, 1)  # (1+n_alt) x 6 x INPUT_LEN
        a = a.unsqueeze(0).repeat(1 + n_alt, 1, 1)  # (1+n_alt) x 2 x INPUT_LEN

        if in_context[i]:
            # Locate the element within the window first, then derive each perturbation region from it
            if args.center_tss:
                seq_start = temp_tss - HALF
                e_start = start - seq_start
                e_end   = end   - seq_start
            else:
                elen    = end - start
                e_start = HALF - elen // 2
                e_end   = e_start + elen

            # Mask accessibility around element
            if args.perturbation in ('acc', 'both_perturbations'):
                startmask = max(0, e_start - args.dist_additional_mask)
                endmask   = min(INPUT_LEN, e_end + args.dist_additional_mask)
                a[1:, 0, startmask:endmask] = a[1:, 0, startmask:endmask] / args.scale_factor

            # Dinucleotide shuffle the element itself, leaving the accessibility track intact
            if args.perturbation in ('shuffle', 'both_perturbations'):
                sh_start = max(0, e_start - args.dist_additional_shuffle)
                sh_end   = min(INPUT_LEN, e_end + args.dist_additional_shuffle)
                if sh_end - sh_start >= args.min_shuffle_len:
                    try:
                        # shuffle from row 0, which is never perturbed. seed is keyed to the row index so
                        # the shuffles are reproducible regardless of batching or where you restart
                        shuf = dinucleotide_shuffle(s[0, :, sh_start:sh_end], n_shuffles=n_alt,
                                                    random_state=args.seed + i)
                        for k in range(n_alt):
                            s[1 + k, :, sh_start:sh_end] = shuf[k]
                        shuffled[i] = True
                    except ValueError:
                        # every shuffle came back identical (low complexity element), leave it unshuffled
                        pass

        # chunk the forward pass, a full (1+n_alt) batch will not fit for large --n_shuffles
        outs = []
        for j in range(0, 1 + n_alt, args.batch_size):
            o = evals(data=((s[j:j+args.batch_size], a[j:j+args.batch_size]), (None, None)))
            outs.append(o.cpu().numpy())  # float32 from model
        preds = np.concatenate(outs, axis=0)  # (1+n_alt) x N_BINS x n_strands
        assert preds.shape[-1] >= n_strands, f'model gave {preds.shape[-1]} strand channels, need {n_strands}'

        # (1+n_alt) x n_strands x N_BINS, so block b strand c lands on row b*n_strands + c
        blocks = preds[:, :, :n_strands].transpose(0, 2, 1)

        if average_shuffles:
            outputs[i] = np.concatenate([blocks[0], blocks[1:].mean(axis=0)], axis=0)
            shuffle_std[i] = blocks[1:].std(axis=0)
        else:
            outputs[i] = blocks.reshape(n_rows, N_BINS)

    out_path = os.path.join(args.output_dir, args.output + '.npy')
    np.save(out_path, outputs)
    print(f'saved E2G borzoi results to {out_path}')

    #saved unconditionally now, with shuffling you need to know which rows were actually perturbed
    ic_path = os.path.join(args.output_dir, args.output + '_in_context.npy')
    np.save(ic_path, in_context)
    print(f'saved in_context flags to {ic_path}')

    if args.perturbation in ('shuffle', 'both_perturbations'):
        sh_path = os.path.join(args.output_dir, args.output + '_shuffled.npy')
        np.save(sh_path, shuffled)
        print(f'saved shuffled flags to {sh_path} ({shuffled.sum()}/{len(shuffled)} elements shuffled)')

    if average_shuffles:
        std_path = os.path.join(args.output_dir, args.output + '_shuffle_std.npy')
        np.save(std_path, shuffle_std)
        print(f'saved per-shuffle std to {std_path}')

    #sidecar so the npy is self describing
    meta_path = os.path.join(args.output_dir, args.output + '_args.json')
    with open(meta_path, 'w') as f:
        json.dump({**vars(args), 'n_alt': n_alt, 'n_strands': n_strands, 'n_blocks': n_blocks,
                   'averaged': average_shuffles, 'shape': list(outputs.shape)}, f, indent=2)
    print(f'saved run args to {meta_path}')
    print('E2G borzoi run complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run E2G borzoi evaluation (stranded)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='k562_stranded_borzoi',
                        help='Output filename (no extension)')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/lesliec/sarthak/data/joint_playground/e2g/',
                        help='Directory to save output .npy file')
    parser.add_argument('--assay_type', type=str, default='CAGE',
                        choices=['CAGE', 'RNA', 'both'],
                        help='Assay type: CAGE or RNA (2 strand channels) or both (4 strand channels)')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by in the masked region')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp to mask on each side of the element')
    parser.add_argument('--center_tss', action='store_true',
                        help='Center sequence window at TSS instead of enhancer. '
                             'Enhancers outside the window are predicted without modulation (in_context=0).')
    parser.add_argument('--perturbation', type=str, default='acc',
                        choices=['acc', 'shuffle', 'both_perturbations'],
                        help='acc: scale down accessibility over the element (original behaviour). '
                             'shuffle: dinucleotide shuffle the element, accessibility left intact. '
                             'both_perturbations: apply both to every alternate block. '
                             "(named both_perturbations so it does not clash with --assay_type both)")
    parser.add_argument('--n_shuffles', type=int, default=5,
                        help='Number of dinucleotide shuffles per element (ignored for --perturbation acc)')
    parser.add_argument('--save_all_shuffles', action='store_true',
                        help='Save every individual shuffle instead of their mean. Default is to average '
                             'the shuffles, keeping the original 4 (or 8) row layout.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed; element i uses seed + i, so shuffles are reproducible per row')
    parser.add_argument('--dist_additional_shuffle', type=int, default=0,
                        help='Extra bp to shuffle on each side of the element. Kept separate from '
                             '--dist_additional_mask so the sequence and accessibility regions can differ.')
    parser.add_argument('--min_shuffle_len', type=int, default=10,
                        help='Skip shuffling elements shorter than this many bp')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of sequences per forward pass')
    parser.add_argument('--load_data', action='store_true',
                        help='Load data into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to accessibility data (overrides dataset config)')
    args = parser.parse_args()

    print(args)
    print(f'Running E2G borzoi ({args.assay_type}, {args.perturbation}) and saving results to {args.output}')
    print(f'Loading checkpoint from {args.ckpt_path}')

    main(args)
