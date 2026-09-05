"""Read-only validation that model_outputs_fold3.zarr matches fresh model0_best.h5 predictions.

Never opens any zarr in write mode. Checks the natsorted-file -> global-index mapping at
BOTH ends of fold3 (first file and last file), which is what would break if ordering were wrong.
"""
import glob, json, os
import numpy as np, zarr, tensorflow as tf
from natsort import natsorted
from tensorflow.keras.utils import to_categorical
from baskerville import seqnn

FOLD, NBINS, NTRK, NSEQ = 3, 6144, 7611, 6888
TRACKS = [870, 871]
PRED_Z = '/data1/lesliec/sarthak/data/borzoi/model_outputs_fold3.zarr'
# GM12878CAGE.zarr is a zarr v3 store and the borzoi env ships zarr v2, so compare against
# borzoi.zarr (v2), the full target store GM12878CAGE.zarr was derived from.
OBS_Z  = '/data1/lesliec/sarthak/data/borzoi/borzoi.zarr'

feature_desc = {"sequence": tf.io.FixedLenFeature([], tf.string),
                "target":   tf.io.FixedLenFeature([], tf.string)}
def _parse_fn(proto):
    ex = tf.io.parse_single_example(proto, feature_desc)
    return tf.io.decode_raw(ex["sequence"], tf.uint8), tf.io.decode_raw(ex["target"], tf.float16)

files = natsorted(glob.glob('/data1/lesliec/sarthak/data/borzoi/tfr_records/fold%d-*.tfr' % FOLD))
print(f"{len(files)} tfr files; first={os.path.basename(files[0])} last={os.path.basename(files[-1])}", flush=True)

z   = zarr.open(PRED_Z, mode='r')[f'fold{FOLD}']   # READ ONLY
obs = zarr.open(OBS_Z,  mode='r')[f'fold{FOLD}']   # READ ONLY
print(f"pred zarr {z.shape} {z.dtype} | obs zarr {obs.shape} {obs.dtype}", flush=True)

with open('/data1/lesliec/sarthak/borzoi/params.json') as f:
    params_model = json.load(f)
model = seqnn.SeqNN(params_model['model'])
model.restore('/data1/lesliec/sarthak/borzoi/model0_best.h5')
print("model restored (no rc/shift ensembling, matching evaluate_borzoi.py)", flush=True)

def records(path):
    return list(tf.data.TFRecordDataset(path, compression_type="ZLIB").map(_parse_fn))

def check(path, base, take, label):
    recs = records(path)
    print(f"\n===== {label}: {os.path.basename(path)} has {len(recs)} records -> global idx {base}..{base+len(recs)-1} =====", flush=True)
    for j in take:
        if j >= len(recs): continue
        seq, tgt = recs[j]
        gi = base + j
        t = np.asarray(tgt).reshape(NBINS, NTRK)
        o = np.asarray(obs[gi])
        print(f"  idx {gi}: tfrecord target vs borzoi.zarr, ALL {NTRK} tracks -> exact match: {np.array_equal(t, o)}", flush=True)
        print(f"           tracks {TRACKS} only -> exact match: {np.array_equal(t[:, TRACKS], o[:, TRACKS])}", flush=True)
        x = np.expand_dims(to_categorical(np.asarray(seq), num_classes=5)[:, :-1], 0)
        p = model.predict(x, verbose=0)[0].astype(np.float16)
        s = np.asarray(z[gi])
        r = np.corrcoef(p.ravel().astype(np.float32), s.ravel().astype(np.float32))[0, 1]
        d = np.abs(p.astype(np.float32) - s.astype(np.float32))
        print(f"           pred {p.shape} vs stored {s.shape}", flush=True)
        print(f"           all-7611-track r = {r:.6f}  max|diff| = {d.max():.4g}  mean|diff| = {d.mean():.3g}  stored mean = {s.astype(np.float32).mean():.4f}", flush=True)
        for c, nm in zip(TRACKS, ['870(+)', '871(-)']):
            rr = np.corrcoef(p[:, c].astype(np.float32), s[:, c].astype(np.float32))[0, 1]
            print(f"           track {nm}: r = {rr:.6f}  max|diff| = {np.abs(p[:,c].astype(np.float32)-s[:,c].astype(np.float32)).max():.4g}", flush=True)

check(files[0],  0, [0, 1], 'HEAD')
nlast = len(records(files[-1]))
check(files[-1], NSEQ - nlast, [nlast - 1, 0], 'TAIL')
print("\ndone", flush=True)
