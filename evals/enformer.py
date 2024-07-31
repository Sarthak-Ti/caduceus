import tensorflow as tf
import tensorflow_hub as hub

enformer_model = hub.load("https://kaggle.com/models/deepmind/enformer/frameworks/TensorFlow2/variations/enformer/versions/1").model

SEQ_LENGTH = 393_216
import h5py
import numpy as np
split = 'test'
data_path=f'/data/leslie/sarthak/data/enformer/data/{split}_seq.npz'
seq_data = np.load(data_path)
seq = np.array(seq_data['sequence_array'])
del seq_data
labels = h5py.File(data_path.replace('_seq.npz', '_label.h5'),'r')['labels']
#load labels into memory
labels = np.array(labels)

def ohe(data):
    out = np.zeros((data.shape[0],4))
    for i,nuc in enumerate(data):
        if nuc == 7:
            out[i] = np.array([1,0,0,0])
        elif nuc == 8:
            out[i] = np.array([0,1,0,0])
        elif nuc == 9:
            out[i] = np.array([0,0,1,0])
        elif nuc == 10:
            out[i] = np.array([0,0,0,1])
        else:
            out[i] = np.array([0,0,0,0])
    padding_size = (SEQ_LENGTH - out.shape[0]) // 2
    padded_array = np.pad(out, ((padding_size, padding_size), (0, 0)), mode='constant', constant_values=0)
    return padded_array[np.newaxis, ...]

def pearsonr2(x, y):
    # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
    # because they want to compute everything pairwise between rows of a
    # stacked x and y.
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = np.sum((x - xm) * (y - ym), axis=-1)/(x.shape[-1]-1)
    sx = np.std(x, ddof=1, axis=-1)
    sy = np.std(y, ddof=1, axis=-1)
    rho = cov/(sx * sy)
    return rho

from tqdm import tqdm

#first let' sone hot encode all the data
#first check if it already exists
import os
if not os.path.exists('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_seq_data.npy'):
    seq_data = np.zeros((seq.shape[0], SEQ_LENGTH, 4), dtype=np.int8)
    for i in tqdm(range(seq.shape[0]), total=seq.shape[0]):
        seq_data[i] = ohe(seq[i])
    np.save('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_seq_data.npy', seq_data)
else:
    seq_data = np.load('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_seq_data.npy')
#save out seq data


#let's batch it together
# batch_size = 2
# seq_data = np.array_split(seq_data, seq.shape[0]//batch_size)
# labels = np.array_split(labels, seq.shape[0]//batch_size)
# #now evaluate
# corrs = np.zeros((seq.shape[0], 4675))
# for i in tqdm(range(len(seq_data)), total=len(seq_data)):
#     x = seq_data[i]
#     y = labels[i]
#     y_pred = enformer_model.predict_on_batch(x)
#     corrs[i*batch_size:(i+1)*batch_size] = np.array([pearsonr2(y[j], y_pred[j]) for j in range(y.shape[0])])

#let's do a nonbatched version
corrs = np.zeros((seq.shape[0], labels.shape[2]))
for i in tqdm(range(seq.shape[0]), total=seq.shape[0]):
    x = seq_data[i].astype(np.float32)
    y = labels[i]
    y_pred = enformer_model.predict_on_batch(x[np.newaxis, ...])['human'][0]
    corrs[i] = pearsonr2(y.T, y_pred.numpy().T)

np.save('/data/leslie/sarthak/hyena/hyena-dna/evals/enformer_corrs.npy', corrs)