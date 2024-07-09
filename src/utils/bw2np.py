import numpy as np
import pyBigWig
import argparse

def extract_chromosome_data(bw, chrom):
    # Get the length of the chromosome
    chrom_length = bw.chroms()[chrom]
    
    # Retrieve values for the entire chromosome
    values = np.nan_to_num(bw.values(chrom, 0, chrom_length))
    #check to ensure it's all integers
    if not np.all(np.equal(np.mod(values, 1), 0)):
        print('error')
    if not np.all(values >= 0):
        print('error')
    return values.astype(np.uint16) #no counts should be above 

def save_bigwig_to_npz(bigwigfile, output_file):
    # Open the BigWig file
    bw = pyBigWig.open(bigwigfile)
    
    # Dictionary to hold the data for each chromosome
    chrom_data = {}
    
    # Extract data for each chromosome
    for chrom in bw.chroms().keys():
        print(f"Processing {chrom}")
        chrom_data[chrom] = extract_chromosome_data(bw, chrom)
    
    # Save the data to a .npz file
    np.savez(output_file, **chrom_data)
    
    # Close the BigWig file
    bw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a BigWig file and save the data to a .npz file.')
    parser.add_argument('bigwigfile', type=str, help='Path to the BigWig file')
    parser.add_argument('output_file', type=str, help='Path to the output .npz file')
    
    args = parser.parse_args()
    
    save_bigwig_to_npz(args.bigwigfile, args.output_file)