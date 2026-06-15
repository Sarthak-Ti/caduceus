import gzip
import numpy as np


def fasta_to_genome(fasta_path):
    """Read a FASTA file and return a dict of {chrom: int8 array}.
    Encoding matches the NPZ genome format: A=7, C=8, G=9, T=10, N/other=11.
    Handles both plain and gzip-compressed FASTA files.
    """
    lut = np.full(256, 11, dtype=np.int8)  # default all to N
    for char, val in [('A', 7), ('C', 8), ('G', 9), ('T', 10),
                      ('a', 7), ('c', 8), ('g', 9), ('t', 10)]:
        lut[ord(char)] = val

    opener = gzip.open if fasta_path.endswith('.gz') else open
    genome = {}
    current_chrom = None
    chunks = []

    with opener(fasta_path, 'rt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if current_chrom is not None:
                    seq_bytes = np.frombuffer(b''.join(chunks), dtype=np.uint8)
                    genome[current_chrom] = lut[seq_bytes]
                current_chrom = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.encode())

    if current_chrom is not None:
        seq_bytes = np.frombuffer(b''.join(chunks), dtype=np.uint8)
        genome[current_chrom] = lut[seq_bytes]

    return genome
