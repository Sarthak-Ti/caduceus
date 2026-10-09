from pathlib import Path
import tempfile
import sys
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_cd4_nk_cage import extract

with tempfile.TemporaryDirectory(prefix='cage_extraction_check_', dir='/tmp') as tmp:
    root = Path(tmp)
    source = zarr.open_group(str(root / 'source.zarr'), mode='w-', zarr_format=2)
    for split, n in [('train', 5), ('val', 3), ('test', 1)]:
        values = np.arange(n * 896 * 5, dtype=np.float32).reshape(n, 896, 5) / 128
        arr = source.create_array(split, shape=values.shape, chunks=(2, 896, 5), dtype='float32', compressors=None)
        arr[:] = values
    (root / 'targets.tsv').write_text('index\tidentifier\tdescription\n1\tcage_cd4\tCAGE:CD4\n3\tcage_nk\tCAGE:NK\n')
    (root / 'sequences.bed').write_text(''.join(f'chr1\t{i}\t{i+131072}\t{split}\n' for split, n in [('train', 5), ('valid', 3), ('test', 1)] for i in range(n)))
    cfg = {
        'source_zarr': str(root / 'source.zarr'), 'source_targets': str(root / 'targets.tsv'),
        'sequences_bed': str(root / 'sequences.bed'), 'data_dir': str(root / 'data'),
        'cells': {'CD4': {'track_index': 1, 'track_identifier': 'cage_cd4', 'description': 'CAGE:CD4'},
                  'NK': {'track_index': 3, 'track_identifier': 'cage_nk', 'description': 'CAGE:NK'}},
    }
    result = extract(cfg, workers=2)
    for cell, index in [('CD4', 1), ('NK', 3)]:
        dest = zarr.open_group(str(root / 'data' / f'{cell}_CAGE.zarr'), mode='r')
        for split in ['train', 'val', 'test']:
            np.testing.assert_array_equal(dest[split][:], source[split][:, :, index:index+1])
    assert result['status'] == 'complete' and (root / 'data/SUCCESS').exists()
    assert not (root / 'data.partial').exists()
    try:
        extract(cfg)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing output must not be overwritten')
    cfg['data_dir'] = str(root / 'invalid')
    cfg['cells']['CD4']['description'] = 'WRONG'
    try:
        extract(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong track metadata must be rejected')
    assert not (root / 'invalid').exists()
    print('PASS: exact target selection, split/row order, short final chunks, parallel writes, overwrite and metadata guards')
