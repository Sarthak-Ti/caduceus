"""Standalone dinucleotide shuffling for one-hot encoded sequences.

Adapted from tangermeme (`tangermeme/ersatz.py`, Jacob Schreiber), which in turn
adapted it from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

This is a trimmed copy that keeps only the single-sequence shuffle, so it has no
tangermeme dependency (only numba/numpy/torch). It operates directly on a
(n_channels, length) one-hot slice rather than on a full batch, which avoids
cloning the whole 524kb sequence once per shuffle.

Works with the channel layout used by GeneralDataset, i.e. (6, length) where
channels 0-4 are A,C,G,T,N and channel 5 is the mask channel. Channels that
never occur in the slice simply get a zero-count transition list and are never
visited, so the extra channels are harmless. N (channel 4) is treated as its own
symbol, so its count and its transitions are preserved like any other base.
"""

import warnings

import numba
import numpy
import torch


params = 'void(int64, int64, int32[:], int32[:, :], int32[:], '
params += 'int32[:, :], float32[:, :, :], int32)'
@numba.jit(params, nopython=True, cache=True)
def _fast_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts,
    counters, shuffled_sequences, random_state):
    """An internal function for fast dinucleotide shuffling using numba."""

    numpy.random.seed(random_state)

    for i in range(n_shuffles):
        for char in range(n_chars):
            n = next_idxs_counts[char]

            next_idxs_ = numpy.arange(n)
            next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index
            next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

        idx = 0
        shuffled_sequences[i, idxs[idx], 0] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = next_idxs[char, count]

            counters[i, char] += 1
            shuffled_sequences[i, idxs[idx], j] = 1


def dinucleotide_shuffle(X, n_shuffles=1, random_state=None, verbose=False):
    """Dinucleotide shuffle a single one-hot encoded sequence.

    This function takes in a one-hot encoded sequence (not a string) and
    returns a set of one-hot encoded sequences that are dinucleotide
    shuffled. The approach constructs a transition matrix between
    nucleotides, keeps the first and last nucleotide constant, and then
    randomly at uniform selects transitions until all nucleotides have
    been observed. This is a Eulerian path. Because each nucleotide has
    the same number of transitions into it as out of it (except for the
    first and last nucleotides) the greedy algorithm does not need to
    check at each step to make sure there is still a path.

    Pass in only the region you want shuffled, e.g. `seq[:, start:end]`, and
    write the result back into that slice yourself.

    Parameters
    ----------
    X: torch.Tensor, shape=(n_channels, length)
        A single one-hot encoded sequence to be shuffled. May live on any
        device; the shuffle itself runs on CPU.

    n_shuffles: int, optional
        The number of shuffled sequences to produce. Default is 1.

    random_state: int or None, optional
        The random seed to use when generating shuffles. If None, draw a new
        seed at random. Default is None.

    verbose: bool, optional
        Whether to warn when at least one position is identical across all
        shuffles. Default is False.

    Returns
    -------
    shuffled_sequences: torch.Tensor, shape=(n_shuffles, n_channels, length)
        The shuffled sequences, as float32 on the same device as `X`.
    """

    if X.dim() != 2:
        raise ValueError(f"X must be 2D of shape (n_channels, length), got {tuple(X.shape)}")

    if random_state is None:
        random_state = numpy.random.randint(0, 9999999)

    n_chars, seq_len = X.shape
    idxs = X.argmax(axis=0).cpu().numpy().astype(numpy.int32)

    next_idxs = numpy.zeros((n_chars, seq_len), dtype=numpy.int32)
    next_idxs_counts = numpy.zeros(n_chars, dtype=numpy.int32)

    for char in range(n_chars):
        next_idxs_ = numpy.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_sequences = numpy.zeros((n_shuffles, *X.shape), dtype=numpy.float32)
    counters = numpy.zeros((n_shuffles, n_chars), dtype=numpy.int32)

    _fast_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts,
        counters, shuffled_sequences, random_state)

    shuffled_sequences = torch.from_numpy(shuffled_sequences).to(X.device)

    conserved = shuffled_sequences[:, :, 1:-1].sum(dim=0)
    if conserved.max() == n_shuffles:
        if verbose:
            warnings.warn(
                "At least one position in dinucleotide shuffle is identical "
                "across all positions.", UserWarning, stacklevel=2)
    if conserved.max(dim=0).values.min() == n_shuffles and n_shuffles > 1:
        raise ValueError("All dinucleotide shuffles yield identical " +
            "sequences, potentially due to a lack of diversity in sequence.")

    return shuffled_sequences


if __name__ == "__main__":
    #example usage / self test on a GeneralDataset-style (6, length) one-hot
    def _dinuc_counts(idxs, n_chars):
        counts = numpy.zeros((n_chars, n_chars), dtype=int)
        for a, b in zip(idxs[:-1], idxs[1:]):
            counts[a, b] += 1
        return counts

    torch.manual_seed(0)
    length = 2000
    tokens = torch.randint(0, 4, (length,))
    seq = torch.nn.functional.one_hot(tokens, num_classes=6).float().T  # 6 x length
    seq[:, 500:510] = 0
    seq[4, 500:510] = 1  # a stretch of Ns, to check they are handled

    start, end = 200, 1200
    shuffled = dinucleotide_shuffle(seq[:, start:end], n_shuffles=5, random_state=0)
    print('input', tuple(seq.shape), '-> shuffled', tuple(shuffled.shape))

    orig_idxs = seq[:, start:end].argmax(0).numpy()
    for i in range(shuffled.shape[0]):
        shuf_idxs = shuffled[i].argmax(0).numpy()
        assert torch.allclose(shuffled[i].sum(0), torch.ones(end - start)), 'not one-hot'
        assert torch.equal(shuffled[i].sum(1), seq[:, start:end].sum(1)), 'composition changed'
        assert numpy.array_equal(_dinuc_counts(orig_idxs, 6), _dinuc_counts(shuf_idxs, 6)), 'dinuc counts changed'
        assert orig_idxs[0] == shuf_idxs[0] and orig_idxs[-1] == shuf_idxs[-1], 'ends not conserved'
        assert not numpy.array_equal(orig_idxs, shuf_idxs), 'shuffle returned the input'

    #and check reproducibility
    again = dinucleotide_shuffle(seq[:, start:end], n_shuffles=5, random_state=0)
    assert torch.equal(shuffled, again), 'not reproducible with a fixed seed'
    print('all checks passed')
