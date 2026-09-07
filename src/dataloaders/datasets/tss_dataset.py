"""
TSS-centered dataset that loads gene-level data from a JSON dictionary.

Sequences are centered around the transcription start site (TSS) of each gene.
Unlike GeneralDataset, each gene is one sample (no celltypes multiplier).
Additional data (enformer-style) has been removed and replaced with TSS-derived outputs.

The TSS JSON file should map gene names to a sub-dict with:
    'chrom'      : str         - chromosome name
    'tss'        : int         - main TSS genomic coordinate (0-based)
    'counts'     : float       - expression count for the gene
    'alt_tss'    : list of int - alternative TSS genomic coordinates (0-based)
    'strand'     : str         - '+' or '-'
    'gene_start' : int         - gene-body start (0-based, inclusive)
    'gene_end'   : int         - gene-body end (0-based, exclusive)

Returns two output tuples per sample:
    outputs1: (seq, targets)
    outputs2: (seq_unmask, acc_umask, counts, tss_mask, gene_mask, strand, expression)
"""

import json
import os
import sys
from random import random

import numpy as np
import torch
import zarr

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
        upstream: int = None,         # if set, place TSS this many bp from the left edge (Decima-style) instead of centering
        append_gene_mask: bool = False,  # if True, append the gene-body mask as an extra sequence channel (Decima-style), in place of MLM masking
        append_tss_mask: bool = False,   # same idea, but appends the TSS-neighborhood mask instead of the gene body; mutually exclusive with append_gene_mask
        data_idxs: str = None,        # JSON or list to select specific tracks from data_path
        expression_data_path: str = None,   # optional RNA-seq zarr/npz, same format as data_path; aligned per-bp like self.data
        expression_data_idxs: str = None,   # optional track selection for the expression data
        expression_stranded: bool = False,  # if True, pick a single strand-matched expression track per gene
        expression_plus_track: int = 0,     # track index used for + strand genes when expression_stranded
        expression_minus_track: int = 1,    # track index used for - strand genes when expression_stranded
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
            upstream (int): if None, the sequence is centered on the TSS (TSS at
                length//2). If set, the TSS is placed `upstream` bp from the left edge
                of the final sequence and the gene body extends downstream (Decima-style
                off-center placement). Strand-aware: for minus-strand genes the window is
                oriented so that, after RC, the TSS still sits `upstream` bp from the left.
            append_gene_mask (bool): if True, provide the gene-of-interest body mask as the
                extra (6th) channel on the one-hot sequence (Decima-style), in the slot the
                MLM mask indicator would occupy. If mlm is set (e.g. rate 0), this
                OVERWRITES the mask channel mask_seq produced rather than adding another; if
                mlm is None, the gene mask is appended. Either way the input is
                5 one-hot + gene mask = 6 channels. Requires one_hot=True.
            append_tss_mask (bool): identical to append_gene_mask, except the appended
                channel is the TSS-neighborhood mask (the same tensor returned as
                outputs2[3]) rather than the gene body. Mutually exclusive with
                append_gene_mask -- only one channel slot exists, so asking for both is an
                error rather than a silent overwrite. Both masks are still returned in
                outputs2 regardless of which (if either) is appended.
            data_idxs (str): JSON file path or list to select specific tracks from data_path
            expression_data_path (str): optional path to an additional per-bp data source
                (zarr/npz, same chromosome-keyed format as data_path) — e.g. RNA-seq
                coverage tracks. When set, it is loaded and aligned to the sequence and to
                self.data at base-pair resolution (identical padding, RC flip, crop, and
                pooling) and returned as `expression` in outputs2. It is not masked.
            expression_data_idxs (str): optional JSON/list track selection for the
                expression data (same semantics as data_idxs).
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
        self.upstream = upstream
        self.append_gene_mask = append_gene_mask
        self.append_tss_mask = append_tss_mask
        assert not (append_gene_mask and append_tss_mask), (
            "append_gene_mask and append_tss_mask are mutually exclusive: there is only one "
            "extra sequence channel, so appending both would silently drop one"
        )
        if upstream is not None and length is not None:
            assert 0 <= upstream < length, (
                f"upstream ({upstream}) must be in [0, length={length}) so the TSS "
                "falls inside the sequence window"
            )

        self.weights_seq_path = weights_seq
        self.expression_data_path = expression_data_path

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
        # Optional additional per-bp tracks (e.g. RNA-seq), same format as self.data
        self.expression_data = open_data(expression_data_path, load_in)

        # Optional track selection from data_path / expression_data_path
        self.data_idxs = get_data_idxs(data_idxs, self.data)
        self.expression_data_idxs = get_data_idxs(expression_data_idxs, self.expression_data)

        # Strand-matched expression: when set, the expression track is chosen per gene by
        # the gene's strand (+ genes -> plus_track, - genes -> minus_track) so only the
        # sense-strand RNA-seq coverage is returned (single channel). Overrides
        # expression_data_idxs for the expression source. Expects the expression npz/zarr
        # to store plus at index `expression_plus_track` and minus at `expression_minus_track`
        # (e.g. row 0 = plus, row 1 = minus from bigwig_to_npz.py). The existing RC flip for
        # minus-strand genes then reverses the minus track into transcription orientation.
        self.expression_stranded = expression_stranded
        self.expression_plus_track = expression_plus_track
        self.expression_minus_track = expression_minus_track

        # Load TSS dictionary and filter to the requested split.
        # 'val' and 'valid' are treated as equivalent to match either naming convention.
        with open(tss_json_file, 'r') as f:
            self.tss_dict = json.load(f)
        split_aliases = {'val', 'valid'} if split in ('val', 'valid') else {split} #basically val and valid are used interchangabelyy, this solves this problem
        if split is not None:
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

    def _build_tss_mask(self, tss, alt_tss, window_start):
        """Build a binary mask tensor of length self.length with 1s within
        tss_distance of the main TSS and (optionally) each alt TSS, in pre-flip
        (genomic-orientation) sequence coordinates.

        A genomic coordinate g maps to pre-flip sequence index (g - window_start),
        where window_start is the genomic coordinate at pre-flip sequence index 0.
        This holds across left/right padding because leftpad exactly fills the
        clipped bases. Positions outside the sequence window are silently ignored.

        Args:
            tss (int): main TSS genomic coordinate (0-based)
            alt_tss (list of int): alternative TSS genomic coordinates (0-based)
            window_start (int): genomic coordinate mapped to sequence index 0
        Returns:
            mask (torch.FloatTensor): shape (self.length,)
        """
        mask = torch.zeros(self.length)
        positions = [tss] + (list(alt_tss) if self.use_alt_tss else [])
        for pos in positions:
            seq_pos = pos - window_start
            lo = max(0, seq_pos - self.tss_distance)
            hi = min(self.length, seq_pos + self.tss_distance)
            if lo < hi:  # skip if entirely outside sequence bounds
                mask[lo:hi] = 1.0
        return mask

    def _build_gene_mask(self, gene_start, gene_end, window_start):
        """Build a binary mask tensor of length self.length that is 1 across the gene
        body of the gene of interest ONLY (not any other gene in the window), in
        pre-flip (genomic-orientation) sequence coordinates.

        gene_start/gene_end are 0-based half-open genomic bounds [gene_start, gene_end).
        A genomic coordinate g maps to pre-flip sequence index (g - window_start).
        Returns an all-zero mask if the bounds are missing.

        Args:
            gene_start (int): gene-body start genomic coordinate (0-based, inclusive)
            gene_end (int): gene-body end genomic coordinate (0-based, exclusive)
            window_start (int): genomic coordinate mapped to sequence index 0
        Returns:
            mask (torch.FloatTensor): shape (self.length,)
        """
        mask = torch.zeros(self.length)
        if gene_start is None or gene_end is None:
            return mask
        lo = max(0, gene_start - window_start)
        hi = min(self.length, gene_end - window_start)
        if lo < hi:  # skip if the gene body is entirely outside the sequence window
            mask[lo:hi] = 1.0
        return mask

    def _load_aligned(self, data_source, data_idxs, chrom, start, end, leftpad, rightpad, flip):
        """Load a base-pair-resolution data source (zarr/npz, same chromosome-keyed
        format as self.data) for the current window and align it exactly to the sequence
        and to self.data: pad to full length, transpose to (seq_len, n_tracks), reverse if
        the window is RC-flipped, then crop and pool identically to the primary targets.

        Returns a tensor of shape (seq_len_after_crop_pool, n_tracks). Both self.data
        (ATAC targets) and the optional expression data go through this same code path so
        they stay aligned at base-pair resolution.
        """
        if data_idxs is not None:
            # Load all tracks then select; np.array() materialises zarr lazily
            track_data = np.array(data_source[chrom][:, start:end])[data_idxs]  # (n_tracks, seq_len)
            n_tracks = len(data_idxs)
            lpad = np.zeros((n_tracks, len(leftpad)))
            rpad = np.zeros((n_tracks, len(rightpad)))
            data = np.concatenate([lpad, track_data, rpad], axis=1)
        else:
            data = np.concatenate([
                leftpad[None] * 0,
                data_source[chrom][0:1, start:end],
                rightpad[None] * 0
            ], axis=1)

        data = data.transpose(1, 0)  # (seq_len, n_tracks)

        out = torch.FloatTensor(data)
        if flip:
            out = out.flip(dims=[0])

        if self.crop_output > 0:
            out = out[self.crop_output:-self.crop_output]

        if self.pool > 1:
            if out.shape[0] % self.pool != 0:
                raise ValueError('Pool size must divide sequence length')
            out = out.view(out.size(0) // self.pool, self.pool, out.size(1))
            if self.pool_type != 'mean':
                raise NotImplementedError('Only mean pooling implemented')
            out = out.mean(dim=1)

        return out

    def __getitem__(self, index):
        """Get the item at the index.

        Args:
            index (int): gene index into self.genes
        Returns:
            outputs1 (tuple): (seq, targets). seq is shape NxL. N is 5 if one_hot, or 6 if masking (5 one-hot + MLM indicator) or append_gene_mask / append_tss_mask (5 one-hot + that mask). targets is shape MxL. M is number of targets
            outputs2 (tuple): (seq_unmask, acc_umask, counts, tss_mask, gene_mask, strand, expression).
                counts is a single-value tensor of gene expression. tss_mask marks
                TSS-neighborhood positions; gene_mask marks the gene body of the gene of
                interest only. Both masks have shape (self.length,) (or the cropped/pooled
                length). strand is a scalar tensor (+1 / -1). expression is the optional
                additional per-bp data (shape (n_tracks, seq_len), aligned to targets) or
                an empty tensor when expression_data_path is not set.
        """
        if not self.load_in:
            self.genome = open_data(self.genome_seq_file, load_in=False)
            self.data = open_data(self.data_path, load_in=False)
            self.weights_seq = open_data(self.weights_seq_path, load_in=False)
            self.expression_data = open_data(self.expression_data_path, load_in=False)

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
        gene_start = gene_info.get('gene_start')
        gene_end = gene_info.get('gene_end')

        if strand == '-':
            strand=-1
        else:
            strand=1

        # Decide orientation up front: the flip must be known before placing the window
        # so the TSS lands at the requested offset in the FINAL (post-flip) sequence.
        if self.rc_aug and coin_flip() or self.rc_strand and strand == -1:
            flip = True
        else:
            flip = False
        # NOTE: not reliable under num_workers > 1, use for single-worker debugging only
        self.last_flip = flip

        # Where the TSS sits in the final (post-flip) sequence:
        #   upstream is None -> centered (TSS at length//2), the original behavior
        #   upstream set     -> Decima-style: TSS placed `upstream` bp from the left edge,
        #                       gene body extending downstream (to the right after any RC)
        tss_final_pos = self.upstream if self.upstream is not None else self.length // 2
        # Masks/targets are built in pre-flip coords then flipped, so map the target
        # position back through the flip.
        tss_pre_pos = tss_final_pos if not flip else (self.length - 1 - tss_final_pos)

        shift = 0
        if self.shift_sequences > 0:
            shift = np.random.randint(-self.shift_sequences, self.shift_sequences + 1)

        # window_start is the genomic coordinate mapped to pre-flip sequence index 0; any
        # genomic coordinate g maps to pre-flip index (g - window_start). Using
        # end = start + length (not symmetric diff//2 expansion) guarantees the exact
        # requested length even when (length - interval) is odd.
        window_start = tss - tss_pre_pos + shift
        start = window_start
        end = start + self.length

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

        if flip:
            seq = self.complement_array[seq[::-1]]

        seq = torch.LongTensor(seq)

        # POSSIBLE SPEEDUP (not done): the LongTensor + one_hot below allocates an int64
        # (L,) and an int64 (L,5) that are thrown away immediately (~25 MB at L=524288).
        # Scattering 1.0 into a preallocated float32 (L,5) is bit-identical and ~2x faster
        # (~8.6 -> ~5.5 ms). Left alone because training is GPU-bound and 8 prefetching
        # workers hide this entirely; revisit only if the loader starts starving the GPU.
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

        # TSS mask built here (rather than further down with the gene mask) so it is
        # available to append as a sequence channel below. Pre-flip -> flip puts it in the
        # same orientation as seq; it is cropped/pooled later alongside gene_mask.
        tss_mask = self._build_tss_mask(tss, alt_tss, window_start)
        if flip:
            tss_mask = tss_mask.flip(dims=[0])

        # Optionally append the gene-of-interest body mask OR the TSS-neighborhood mask as
        # an extra sequence channel (Decima-style), in place of masking the sequence. Built
        # at full length in the same (post-flip) orientation as seq so it aligns
        # base-for-base; the gene mask is reused later (cropped/pooled) for outputs2.
        gene_mask_full = None
        if self.append_gene_mask or self.append_tss_mask:
            if not self.one_hot:
                raise ValueError("append_gene_mask/append_tss_mask require one_hot=True")
            if self.append_gene_mask:
                gene_mask_full = self._build_gene_mask(gene_start, gene_end, window_start)
                if flip:
                    gene_mask_full = gene_mask_full.flip(dims=[0])
                append_mask = gene_mask_full
            else:
                # already in final orientation from the build above
                append_mask = tss_mask
            if self.mlm is not None:
                # mask_seq already appended a mask channel (all-zero when mlm rate is 0);
                # OVERWRITE that last channel with the chosen mask rather than adding another,
                # so the input stays 5 one-hot + mask = 6 channels.
                seq[:, -1] = append_mask
            else:
                # no mask channel present; append the chosen mask as the extra channel
                seq = torch.cat([seq, append_mask.unsqueeze(1)], dim=1)  # (L, C+1)

        seq = seq.transpose(1, 0)

        if not self.return_target:
            return seq, seq_unmask

        # Load primary chromatin data (ATAC), padded/aligned to the sequence window
        targets = self._load_aligned(self.data, self.data_idxs, chrom, start, end, leftpad, rightpad, flip)

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

        # Optional additional per-bp tracks (e.g. RNA-seq), same zarr format as self.data,
        # loaded through the identical code path so they align to the sequence and targets
        # at base-pair resolution (same padding, RC flip, crop, and pooling). Not masked.
        if self.expression_data is not None:
            # Strand-matched selection: pick only the sense-strand track for this gene so a
            # + gene sees the plus track and a - gene sees the minus track (single channel).
            # Otherwise fall back to the fixed expression_data_idxs for every gene.
            if self.expression_stranded:
                expr_idxs = np.array(
                    [self.expression_plus_track if strand == 1 else self.expression_minus_track]
                )
            else:
                expr_idxs = self.expression_data_idxs
            expression = self._load_aligned(
                self.expression_data, expr_idxs,
                chrom, start, end, leftpad, rightpad, flip
            )
            expression = expression.transpose(1, 0)  # (n_tracks, seq_len)
        else:
            expression = torch.empty(0)

        # tss_mask was built above (before the seq channel append); it still needs the same
        # crop/pool treatment as the gene mask to stay aligned with targets.
        # Gene mask: reuse the full-length tensor already built for the seq channel when
        # append_gene_mask is on (it's already in final orientation); otherwise build it
        # here. Both are then cropped/pooled the same way as tss_mask.
        if gene_mask_full is not None:
            gene_mask = gene_mask_full
        else:
            gene_mask = self._build_gene_mask(gene_start, gene_end, window_start)
            if flip:
                gene_mask = gene_mask.flip(dims=[0])

        if self.crop_output > 0:
            tss_mask = tss_mask[self.crop_output:-self.crop_output]
            gene_mask = gene_mask[self.crop_output:-self.crop_output]

        if self.pool > 1:
            # Max-pool: a bin is active if any position in it overlaps the region
            tss_mask = tss_mask.view(-1, self.pool).max(dim=1).values
            gene_mask = gene_mask.view(-1, self.pool).max(dim=1).values

        outputs1 = [seq, targets]
        outputs2 = [seq_unmask, acc_umask, torch.tensor(counts, dtype=torch.float32), tss_mask, gene_mask, torch.tensor(strand), expression]

        return tuple(outputs1), tuple(outputs2)
    
    