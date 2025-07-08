#super simple script to create a zarr array given argparse argument of location

import zarr
import argparse
from numcodecs import Blosc

def main(args):
    # Open the zarr file in write mode
    if args.compression:
        compression = Blosc(cname='zlib', clevel=9, shuffle=Blosc.BITSHUFFLE)
        # root.create_array('data', shape=(args.samples, args.chrom_length), chunks=(args.chunk_size,), dtype='float16', compressor=compression)
    else:
        compression = None
    
    arr = zarr.create(
        store = args.zarr_file,
        shape=(args.samples, args.size, 2), #samples by 
        chunks=(args.chunk_size, args.size, 2),
        dtype='float16',
        compressor=compression,
        overwrite=True,             # same as mode='w'
        zarr_format=2
    )
    # Create a zarr array with the specified shape and chunk size

    print(f"Zarr file created at {args.zarr_file} with chunk size {args.chunk_size} and shape {arr.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a Zarr file with specified parameters.')
    parser.add_argument('--zarr_file', type=str, required=True, help='Path to the Zarr file to create.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for the Zarr array.')
    parser.add_argument('--samples', type=int, required=True, help='Number of samples.')
    parser.add_argument('--size', type=int, default=500, help='Size of the mask.')
    parser.add_argument('--compression', action='store_true', help='Enable compression for the Zarr array.')
    
    args = parser.parse_args()
    main(args)