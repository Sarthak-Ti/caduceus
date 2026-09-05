"""Convert the split-keyed CAGE npz into a fold-keyed zarr.

The general_dataset fold path does:
    split  = fold[0]   # 'fold0' ... 'fold7'  (col 5 of the bed)
    tindex = fold[1]   # 0-based index within that fold (col 6)
    additional_data = self.additional_data[split][tindex]

so the store must be keyed by fold name, with row `j` of fold F being the bed row
where col5 == F and col6 == j.

The npz is keyed by split ('train'/'val'/'test') and its rows follow bed row order
within each split. Because the bed is grouped by fold (fold0 block, fold1 block, ...)
and col6 ascends 0..n-1 inside each block, each fold occupies a contiguous slice of
its split's array. Both properties are asserted below rather than assumed.

Chunked (1, 6144, 2) so one sample is one chunk (~48 KB), sharded (64, ...) to keep
the file count down (~870 files instead of ~55k).
"""

import numpy as np
import pandas as pd
import zarr

BED = '/data1/lesliec/sarthak/data/DK_zarr/sequences_borzoi_fold3-4.bed'
SRC = '/data1/lesliec/sarthak/data/borzoi/GM12878CAGE.npz'
DST = '/data1/lesliec/sarthak/data/borzoi/GM12878CAGE.zarr'

bed = pd.read_csv(BED, sep='\t', header=None,
                  names=['chrom', 'start', 'end', 'split', 'fold', 'fidx'])
src = np.load(SRC)
dst = zarr.open(DST, mode='w')

# provenance for whoever finds this store later
dst.attrs['source_npz'] = SRC
dst.attrs['source_bed'] = BED
dst.attrs['layout'] = 'group[fold] -> (n_seqs_in_fold, 6144, 2), 32bp bins over the 196608bp bed interval'

written = {}
for split in src.files:
    sub = bed[bed['split'] == split].reset_index(drop=True)
    arr = src[split]
    assert len(sub) == arr.shape[0], \
        f"{split}: bed has {len(sub)} rows but npz has {arr.shape[0]}"

    start = 0
    for fold in sub['fold'].unique():  # order of first appearance == bed order
        m = (sub['fold'] == fold).values
        n = int(m.sum())
        # the fold's rows must be one contiguous run, in per-fold index order
        assert m[start:start + n].all() and m.sum() == m[start:start + n].sum(), \
            f"{fold} is not contiguous within split {split}"
        assert (sub.loc[m, 'fidx'].values == np.arange(n)).all(), \
            f"{fold} per-fold indices are not 0..{n - 1} in order"

        a = dst.create_array(fold, shape=(n, 6144, 2), dtype='float32',
                             chunks=(1, 6144, 2), shards=(64, 6144, 2))
        a[:] = arr[start:start + n]
        a.attrs['split'] = split
        written[fold] = (split, n)
        print(f"{fold}: {n:6d} rows  <- {split}[{start}:{start + n}]", flush=True)
        start += n

    assert start == arr.shape[0], f"{split}: only consumed {start}/{arr.shape[0]} rows"
    del arr

print()
print("verifying against the npz by re-reading random rows through the dataset's access path")
rng = np.random.default_rng(0)
z = zarr.open(DST, mode='r')  # same call open_data() makes
for fold, (split, n) in sorted(written.items()):
    sub = bed[bed['split'] == split].reset_index(drop=True)
    offset = int(np.flatnonzero((sub['fold'] == fold).values)[0])
    arr = src[split]
    for j in rng.choice(n, size=5, replace=False):
        got = z[fold][np.int64(j)]                 # exactly what __getitem__ does
        assert np.array_equal(got, arr[offset + j]), f"mismatch {fold}[{j}]"
    del arr
    print(f"{fold}: 5/5 rows match", flush=True)

print()
print({k: (z[k].shape, str(z[k].dtype)) for k in sorted(z.array_keys())})
