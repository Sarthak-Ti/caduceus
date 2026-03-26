#e2g model with enformer output (unstranded cage)
print('E2G for centering at TSS model', flush=True)

import numpy as np
import pandas as pd
import json
import sys
sys.path.append('/data1/lesliec/sarthak/caduceus/')
import torch
from tqdm import tqdm
import argparse
from src.dataloaders.datasets.tss_dataset import TSSDataset as GeneralDataset
from src.models.sequence.dna_embedding import DNAEmbeddingModelCaduceus
from src.tasks.decoders import TSSDecoder
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

class Evals():
    def __init__(self,
                 ckpt_path,
                 dataset=None,
                 split = None,
                 device = None,
                 load_data=False,
                 **dataset_overrides #Don't pass None into overrides unless you intentionally want it to be None! Pass in items only that you need
                 ) -> None:

        #now load the cfg from the checkpoint path
        model_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '.hydra', 'config.yaml')
        cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)

        state_dict = torch.load(ckpt_path, map_location='cpu')
        if device is not None:
            #if we are given a device, we will use that device
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        #now set up dataset
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
            dataset_args['evaluating'] = True #this tells it to not do things like random shifting and rc aug, still does random masking tho, can get og sequence easily
            dataset_args['load_in'] = load_data
            # dataset_args['additional_data'] = None #override so we can load the dataset!

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
        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)
        # the state_dict keys slightly mismatch from Lightning..., so we fix it here
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
        # self.decoder = EnformerDecoder(**self.cfg['decoder'])
        self.decoder = TSSDecoder(**self.cfg['decoder']) #could do with instantiating, but that is rather complex
        self.decoder.load_state_dict(decoder_state_dict, strict=True)

        del self.cfg['encoder']['_name_']
        self.cfg['encoder']['d_model'] = self.cfg['model']['config']['d_model']
        self.encoder = JointCNN(**self.cfg['encoder'])
        self.encoder.load_state_dict(encoder_state_dict, strict=True)

        self.encoder.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.decoder.to(self.device).eval()

    def __call__(self, idx=None, data=None, softplus=False, og=False, ctt_val=None):
        #now evaluate the model on one example
        if data is None:
            (seq,acc),(su,au,counts,tss_mask,strand) = self.dataset[idx]

            x = seq.unsqueeze(0)
            y = acc.unsqueeze(0)
        else:
            (x,y),(su,au,counts,tss_mask,strand) = data

            if x.dim() == 2:
                x = x.unsqueeze(0) #add batch dim
                y = y.unsqueeze(0) #add batch dim

        x,y = x.to(self.device), y.to(self.device)
        tss_mask = tss_mask.to(self.device) if tss_mask is not None else None

        with torch.no_grad():
            x1,_ = self.encoder(x,y)
            x1,_ = self.backbone(x1)
            x1 = self.decoder(x1, mask=tss_mask)

        return x1


def main(args):
    TSS_bounds_file = "/data1/deyk/extras/CollapsedGeneBounds.hg38.TSS500bp.bed"
    tss = pd.read_csv(TSS_bounds_file, sep="\t")

    path = "/data1/deyk/ENCODE/CRISPR/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    gs_df = pd.read_csv(path, sep="\t")

    json_path = '/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json'
    with open(json_path, 'r') as f:
        tss_dict = json.load(f)

    name_to_ensg = tss.set_index('name')['Ensembl_ID'].to_dict()
    name_to_ensg = {name: ensg for name, ensg in name_to_ensg.items() if ensg in tss_dict} #remove extra values
    gs_df['ensg_id'] = gs_df['measuredGeneSymbol'].map(name_to_ensg)
    gs_df = gs_df.dropna(subset=['ensg_id', 'Regulated']).reset_index(drop=True)

    SEQ_LEN = 524288
    HALF = SEQ_LEN // 2

    outputs = np.zeros((len(gs_df), 2)) #num_samples x 2 outputs x 1 value
    in_context = np.ones(len(gs_df), dtype=bool)

    evals_kwargs = dict(load_data=args.load_data)
    if args.data_path is not None:
        evals_kwargs['data_path'] = args.data_path

    evals = Evals(args.ckpt_path, **evals_kwargs)
    # look_tss_len = evals.cfg['dataset']['tss_distance']

    for i, row in tqdm(gs_df.iterrows(), total=len(gs_df)):
        chrom = row['chrom']
        start = row['chromStart']
        end = row['chromEnd']
        ensgid = row['ensg_id']
        temp_tss = tss_dict[ensgid]['tss']

        # if args.center_tss:
        #     # Center the window at the TSS; check if enhancer falls within the window
        in_context[i] = 1 if (start >= temp_tss - HALF and end <= temp_tss + HALF) else 0
        # idx = evals.dataset.expand_seqs(chrom, temp_tss, temp_tss)
        # else:
        #     idx = evals.dataset.expand_seqs(chrom, start, end)
        #get idx based on the protein
        idx = evals.dataset.genes.index(ensgid)
        

        ((s,a),(su,au,counts,tss_mask,strand)) = evals.dataset[idx]
        s = s.unsqueeze(0).repeat(2, 1, 1) #now is 2 x 6 x 524288
        a = a.unsqueeze(0).repeat(2, 1, 1) #now is 2 x 2 x 524288
        tss_mask = tss_mask.unsqueeze(0).repeat(2, 1) if tss_mask is not None else None #now is 2 x 524288

        if in_context[i]:
            #alter the accessibility in the region around the element
            seq_start = temp_tss - HALF
            startmask = (start - seq_start) - args.dist_additional_mask
            endmask   = (end   - seq_start) + args.dist_additional_mask
            startmask = max(0, startmask)
            endmask   = min(SEQ_LEN, endmask)
            
            a[1,0,startmask:endmask] = a[1,0,startmask:endmask] / args.scale_factor

        out = evals(data=((s,a), (None, None, None, tss_mask, None))).cpu().numpy().squeeze()
        outputs[i] = out

    out_path = os.path.join(args.output_dir, args.output + '.npy')
    np.save(out_path, outputs)
    print(f'saved E2G results to {out_path}')
    if args.center_tss:
        ic_path = os.path.join(args.output_dir, args.output + '_in_context.npy')
        np.save(ic_path, in_context)
        print(f'saved in_context flags to {ic_path}')
    print('E2G run complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run E2G enformer evaluation')
    parser.add_argument('--ckpt_path', type=str,
                        default='/data1/lesliec/sarthak/caduceus/outputs/2026-03-17/14-58-01-055972/checkpoints/13-val_loss=2.18242.ckpt',
                        help='Path to the model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='k562_tss_500bp_mask',
                        help='Output filename (no extension)')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/lesliec/sarthak/data/joint_playground/e2g/',
                        help='Directory to save output .npy file')
    parser.add_argument('--scale_factor', type=float, default=100,
                        help='Factor to divide accessibility by in the masked region')
    parser.add_argument('--dist_additional_mask', type=int, default=100,
                        help='Extra bp to mask on each side of the element')
    parser.add_argument('--load_data', action='store_true',
                        help='Load data into memory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to accessibility data (overrides dataset config)')
    args = parser.parse_args()

    print(args)
    print(f'Running E2G on model and saving results to {args.output}')
    print(f'Loading checkpoint from {args.ckpt_path}')

    main(args)
