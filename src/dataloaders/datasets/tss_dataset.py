"""
TSS-centered dataset that loads gene-level data from a JSON dictionary.

Sequences are centered around the transcription start site (TSS) of each gene.
Unlike GeneralDataset, each gene is one sample (no celltypes multiplier).
Additional data (enformer-style) has been removed and replaced with TSS-derived outputs.

The TSS JSON file should map gene names to a sub-dict with:
    'chrom'   : str         - chromosome name
    'tss'     : int         - main TSS genomic coordinate (sequence is centered here)
    'counts'  : float       - expression count for the gene
    'alt_tss' : list of int - alternative TSS genomic coordinates

Returns three output tuples per sample:
    outputs1: (seq, targets)
    outputs2: (seq_unmask, acc_umask)
    outputs3: (counts_tensor, tss_mask)
"""

import numpy as np
import zarr
import os
import torch
import json
import sys
from random import random
sys.path.append('/data1/lesliec/sarthak/caduceus/')
from src.dataloaders.utils.mask_seq import mask_seq


def open_data(data_path, load_in=False):
    if data_path is None:
        return None
    if data_path.endswith('.zarr'):
        data = zarr.open(data_path, mode='r')
        if load_in:
            data = {key: np.array(data[key]) for key in data}
    else:
        if load_in:
            with np.load(data_path) as data:
                data = {key: np.array(data[key]) for key in data}
        else:
            data = np.load(data_path)
    return data


def get_data_idxs(data_path, data):
    if data_path is None:
        return None

    if data_path == 'all':
        data_idxs = np.array(range(data['chr22'].shape[0]))

    elif isinstance(data_path, int):
        data_idxs = np.array([data_path])

    elif isinstance(data_path, list):
        data_idxs = np.array(data_path)

    elif isinstance(data_path, str) and data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data_idxs = json.load(f)
        data_idxs = np.array(data_idxs, dtype=int)
    else:
        raise ValueError(f"data_idxs must be a list or a json file, got {data_path}")
    return data_idxs


def coin_flip():
    return random() > 0.5


class TSSDataset():
    def __init__(
        self,
        split: str,
        data_path: str,
        tss_json_file: str,           # path to JSON: {gene: {chrom, tss, counts, alt_tss}}
        length: int = None,
        tss_distance: int = 64,       # half-width (bp) of the TSS mask region around each TSS
        use_alt_tss: bool = True,     # whether to include alt_tss positions in the TSS mask
        data_idxs: str = None,        # JSON or list to select specific tracks from data_path
        genome_seq_file: str = '/data1/lesliec/sarthak/data/chrombpnet_test/hg38_tokenized.npz',
        shift_sequences: int = 0,
        load_in: bool = False,
        one_hot: bool = True,
        pool: int = 1,
        pool_type: str = 'mean',
        return_target: bool = True,
        rc_aug: bool = False,
        rc_strand: bool = False,  # if True, disables rc_aug and instead forces RC for minus-strand genes
        crop_output: int = 0,
        mlm: int = None,
        acc_mlm: int = None,
        acc_type: str = 'continuous',
        acc_mask_size: int = 500,
        pair_mask: bool = False,
        replace_with_N: bool = False,
        acc_threshold: float = 1,
        weight_peaks: bool = False,
        evaluating: bool = False,
        mask_only: bool = False,
        mask_tie: float = 1.0,
        independent_tracks: bool = False,
        alternating: int = 0,
        weights_seq: str = None,
        binary_score_threshold: float = None,
        max_neg_to_pos_ratio: float = 0.1,
        max_scale: float = 3,
        log_weights: bool = False,
        neg_maskrate: float = None,
        minimum_neg_masks: float = 0,
        weight_floor: float = 0.1,
    ):
        """
        TSS-centered dataset. Sequences are centered around the TSS of each gene.
        Gene-level metadata (counts, TSS positions) comes from tss_json_file.
        Each gene is one sample; there is no celltypes multiplier.

        Args:
            split (str): dataset split (train/val/test) — kept for compatibility
            data_path (str): path to chromatin data (npz or zarr), chromosome-keyed,
                             shape per chrom: (n_tracks, chrom_len)
            tss_json_file (str): path to JSON mapping gene names to {chrom, tss, counts, alt_tss}
            length (int): sequence length; sequence is centered on TSS
            tss_distance (int): half-width (bp) of the TSS mask region around each TSS
            use_alt_tss (bool): if True, also mark alt_tss positions in the TSS mask
            data_idxs (str): JSON file path or list to select specific tracks from data_path
            ... (remaining args same as GeneralDataset)
        """
        self.evaluating = evaluating
        self.rc_strand = rc_strand
        if rc_strand:
            rc_aug = False  # rc_strand takes over orientation; random rc_aug would conflict
        if self.evaluating:
            rc_aug = False
            shift_sequences = 0

        self.split = split
        self.genome_seq_file = genome_seq_file
        self.data_path = data_path
        self.pool = pool
        self.pool_type = pool_type
        self.length = length
        self.rc_aug = rc_aug
        self.shift_sequences = shift_sequences
        self.return_target = return_target
        self.one_hot = one_hot
        self.crop_output = crop_output
        self.mlm = mlm
        self.acc_mlm = acc_mlm
        self.acc_mask_size = acc_mask_size
        self.pair_mask = pair_mask
        self.replace_with_N = replace_with_N
        self.load_in = load_in
        self.acc_type = acc_type
        self.acc_threshold = acc_threshold
        self.weight_peaks = weight_peaks
        self.mask_only = mask_only
        self.mask_tie = mask_tie
        self.independent_tracks = independent_tracks
        self.alternating = alternating
        if self.alternating:
            self.mlm_backup = self.mlm
            self.acc_mlm_backup = self.acc_mlm

        self.tss_distance = tss_distance
        self.use_alt_tss = use_alt_tss

        self.weights_seq_path = weights_seq

        # Verify pool and tss_distance are compatible for clean pooling boundaries
        if pool > 1:
            print(f"using max_pool with pool size {pool} and tss_distance {tss_distance}")
            if pool % tss_distance != 0:
                print(
                    f"WARNING: pool ({pool}) is not divisible by tss_distance ({tss_distance}). "
                    "This may lead to pooling bins that partially overlap TSS mask regions, which could affect model performance. "
                    "Consider adjusting pool size or tss_distance for cleaner pooling boundaries."
                    "will max pool so if any overlap, that bin will be fully used in the mask!"
                )
            # assert pool % tss_distance == 0, (
            #     f"pool ({pool}) must be divisible by tss_distance ({tss_distance}) "
            #     "to ensure clean pooling boundaries over TSS mask regions"
            # )

        self.weight_options = {
            'max_scale': max_scale,
            'binary_score_threshold': binary_score_threshold,
            'max_neg_to_pos_ratio': max_neg_to_pos_ratio,
            'neg_maskrate': neg_maskrate,
            'log_weights': log_weights,
            'weight_floor': weight_floor,
            'minimum_neg_masks': minimum_neg_masks,
        }

        if mask_only:
            if mask_only == 1:
                self.mask_only_seq = True
                self.mask_only_acc = True
            if mask_only == 0.5:
                self.mask_only_seq = False
                self.mask_only_acc = True
        else:
            self.mask_only_seq = False
            self.mask_only_acc = False

        # Load genome sequence, primary chromatin data, and optional sequence weights
        self.genome = open_data(genome_seq_file, load_in)
        self.data = open_data(data_path, load_in)
        self.weights_seq = open_data(weights_seq, load_in)

        # Optional track selection from data_path
        self.data_idxs = get_data_idxs(data_idxs, self.data)

        # Load TSS dictionary and filter to the requested split.
        # 'val' and 'valid' are treated as equivalent to match either naming convention.
        with open(tss_json_file, 'r') as f:
            self.tss_dict = json.load(f)
        split_aliases = {'val', 'valid'} if split in ('val', 'valid') else {split} #basically val and valid are used interchangabelyy, this solves this problem
        self.tss_dict = {k: v for k, v in self.tss_dict.items() if v.get('split') in split_aliases}
        self.genes = list(self.tss_dict.keys())
        print(f"TSSDataset: {len(self.genes)} genes in split '{split}'")

        # RC augmentation complement lookup (A=7, C=8, G=9, T=10, N=11)
        max_key = 11
        self.complement_array = np.zeros(max_key + 1, dtype=int)
        complement_map = {"7": 10, "8": 9, "9": 8, "10": 7, "11": 11}
        for k, v in complement_map.items():
            self.complement_array[int(k)] = v

    def __len__(self):
        return len(self.genes)

    def _build_tss_mask(self, tss, alt_tss, shift=0):
        """Build a binary mask tensor of length self.length with 1s within
        tss_distance of the main TSS and (optionally) each alt TSS.

        All input coordinates are genomic. The main TSS maps to sequence position
        (self.length // 2 - shift); shift accounts for any sequence shift applied
        before this call so the mask stays aligned with the actual sequence content.
        Any alt TSS positions outside the sequence window are silently ignored.

        Args:
            tss (int): main TSS genomic coordinate
            alt_tss (list of int): alternative TSS genomic coordinates
            shift (int): the shift applied to the sequence window (default 0)
        Returns:
            mask (torch.FloatTensor): shape (self.length,)
        """
        mask = torch.zeros(self.length)
        center = self.length // 2 - shift  # TSS moves left when sequence shifts right

        # Main TSS is always at center
        lo = max(0, center - self.tss_distance)
        hi = min(self.length, center + self.tss_distance)
        mask[lo:hi] = 1.0

        if self.use_alt_tss:
            for alt in alt_tss:
                # Convert genomic coordinate to sequence coordinate
                alt_pos = center + (alt - tss)
                lo = max(0, alt_pos - self.tss_distance)
                hi = min(self.length, alt_pos + self.tss_distance)
                if lo < hi:  # skip if entirely outside sequence bounds
                    mask[lo:hi] = 1.0

        return mask

    def __getitem__(self, index):
        """Get the item at the index.

        Args:
            index (int): gene index into self.genes
        Returns:
            outputs1 (tuple): (seq, targets). seq is shape NxL. N is 5 if one_hot, 6 if masking. targets is shape MxL. M is number of targets
            outputs2 (tuple): (seq_unmask, acc_umask). same as seq and targets but unmasked with an indication of mask. also transposed to LxN and LxM.
            outputs3 (tuple): (counts_tensor, tss_mask). single value tensor of gene expression counts, and binary mask of TSS positions in the sequence (shape (self.length,))
        """
        if not self.load_in:
            self.genome = open_data(self.genome_seq_file, load_in=False)
            self.data = open_data(self.data_path, load_in=False)
            self.weights_seq = open_data(self.weights_seq_path, load_in=False)

        acc_mlm_rate = self.acc_mlm
        mlm_rate = self.mlm

        if self.alternating:
            outcome = random()
            if outcome <= self.alternating:
                mlm_rate = self.mlm_backup
                acc_mlm_rate = 0
            else:
                mlm_rate = 0
                acc_mlm_rate = self.acc_mlm_backup

        seq_unmask = torch.empty(0)
        acc_umask = torch.empty(0)

        # Look up gene metadata from TSS dictionary
        gene = self.genes[index]
        gene_info = self.tss_dict[gene]
        chrom = gene_info['chrom']
        tss = gene_info['tss']
        counts = gene_info['counts']
        alt_tss = gene_info['alt_tss']
        strand = gene_info['strand']
        
        if strand == '-':
            strand=-1
        else:
            strand=1

        # Center sequence around TSS.
        # BUG FIX (vs GeneralDataset): original used symmetric diff//2 expansion which
        # produces sequences 1bp shorter than requested when (length - interval_size) is
        # odd. Using end = start + length guarantees the exact requested length.
        start = tss - self.length // 2
        end = start + self.length

        shift = 0
        if self.shift_sequences > 0:
            shift = np.random.randint(-self.shift_sequences, self.shift_sequences + 1)
            start = start + shift
            end = end + shift

        # Pad with N (token 11) if sequence extends beyond chromosome bounds
        leftpad = np.zeros(0)
        rightpad = np.zeros(0)
        if start < 0:
            leftpad = np.ones(-start) * 11
            start = 0
        chromlen = self.genome[chrom].shape[0]
        if end > chromlen:
            rightpad = np.ones(end - chromlen) * 11
            end = chromlen
        seq = np.concatenate([
            leftpad.astype(np.int8),
            self.genome[chrom][start:end],
            rightpad.astype(np.int8)
        ])

        if self.rc_aug and coin_flip():
            seq = self.complement_array[seq[::-1]]
            flip = True
        elif self.rc_strand and strand == -1:
            seq = self.complement_array[seq[::-1]]
            flip = True
        else:
            flip = False
        # NOTE: not reliable under num_workers > 1, use for single-worker debugging only
        self.last_flip = flip

        seq = torch.LongTensor(seq)

        if self.one_hot:
            x = seq
            x_onehot = torch.nn.functional.one_hot(x - 7, num_classes=5).float()
            seq = x_onehot

        if self.mlm is not None:
            if not self.one_hot:
                raise ValueError("MLM only works with one hot encoding for now")

            if self.weights_seq is not None:
                weights = np.concatenate((leftpad * 0, self.weights_seq[chrom][start:end], rightpad * 0))
                weights = torch.FloatTensor(weights)
            else:
                weights = None

            seq, seq_unmask = mask_seq(
                seq, mask_pct=mlm_rate, replace_with_N=self.replace_with_N,
                mask_only=self.mask_only_seq, weights=weights, **self.weight_options
            )

        seq = seq.transpose(1, 0)

        if not self.return_target:
            return seq, seq_unmask

        # Load primary chromatin data, padding to match sequence length
        if self.data_idxs is not None:
            # Load all tracks then select; np.array() materialises zarr lazily
            track_data = np.array(self.data[chrom][:, start:end])[self.data_idxs]  # (n_tracks, seq_len)
            n_tracks = len(self.data_idxs)
            lpad = np.zeros((n_tracks, len(leftpad)))
            rpad = np.zeros((n_tracks, len(rightpad)))
            data = np.concatenate([lpad, track_data, rpad], axis=1)
        else:
            data = np.concatenate([
                leftpad[None] * 0,
                self.data[chrom][0:1, start:end],
                rightpad[None] * 0
            ], axis=1)

        data = data.transpose(1, 0)  # (seq_len, n_tracks)

        targets = torch.FloatTensor(data)
        if flip:
            targets = targets.flip(dims=[0])

        if self.crop_output > 0:
            targets = targets[self.crop_output:-self.crop_output]

        if self.pool > 1:
            if targets.shape[0] % self.pool != 0:
                raise ValueError('Pool size must divide sequence length')
            targets = targets.view(targets.size(0) // self.pool, self.pool, targets.size(1))
            if self.pool_type != 'mean':
                raise NotImplementedError('Only mean pooling implemented')
            targets = targets.mean(dim=1)

        if self.acc_mlm is not None:
            assert not self.pair_mask, "Pair masking not implemented yet"
            if self.weight_peaks:
                weights = targets
            else:
                weights = None

            if targets.shape[1] > 1:
                assert self.acc_type == 'continuous', \
                    "Only continuous acc type implemented for multiple target tracks"
                targets, acc_umask = mask_seq(
                    targets, mask_pct=acc_mlm_rate, span=self.acc_mask_size,
                    stype=self.acc_type, weights=weights, mask_only=self.mask_only_acc,
                    mask_tie=self.mask_tie, independent_tracks=self.independent_tracks
                )
            else:
                targets = targets.squeeze(1)
                if self.acc_type == 'category':
                    targets = (targets > self.acc_threshold).long()
                    targets = torch.nn.functional.one_hot(targets, num_classes=2).float()
                targets, acc_umask = mask_seq(
                    targets, mask_pct=acc_mlm_rate, span=self.acc_mask_size,
                    stype=self.acc_type, weights=weights, mask_only=self.mask_only_acc,
                    mask_tie=1
                )

        targets = targets.transpose(1, 0)

        # Build TSS mask in sequence coordinates, then crop/pool to match targets
        tss_mask = self._build_tss_mask(tss, alt_tss, shift=shift)

        if flip:
            tss_mask = tss_mask.flip(dims=[0])

        if self.crop_output > 0:
            tss_mask = tss_mask[self.crop_output:-self.crop_output]

        if self.pool > 1:
            # Max-pool: a bin is active if any position in it overlaps a TSS region
            tss_mask = tss_mask.view(-1, self.pool).max(dim=1).values

        outputs1 = [seq, targets]
        outputs2 = [seq_unmask, acc_umask, torch.tensor(counts, dtype=torch.float32), tss_mask, torch.tensor(strand)]

        return tuple(outputs1), tuple(outputs2)