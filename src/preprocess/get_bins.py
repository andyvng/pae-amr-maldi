import os
import math
import pandas as pd
import numpy as np
import argparse
import pickle

def determine_bin_width(row, max_no_peaks, min_no_peaks, range_width):
    # Dynamic binning
    return 1 + math.floor((max_no_peaks - row['avg_counts'])/(max_no_peaks - min_no_peaks) * (range_width - 1))

def main():
    parser = argparse.ArgumentParser(description="Get bins")
    parser.add_argument("peaks_fp", type=str, help="Peak count file path")
    parser.add_argument("binning_type", type=str, help="Binning type")
    parser.add_argument("output_fp", type=str, help="Output file path")
    parser.add_argument("--bin_size", type=int, default=3, help="Bin size")
    parser.add_argument("--range", type=int, default=100, help="Range for dynamic binning")
    parser.add_argument("--region", type=int, default=500, help="Region for dynamic binning")

    args = parser.parse_args()

    # Generate bins file
    bins = None
    if args.binning_type.capitalize() == "Dynamic":
        # Load peak count file
        peak_counts = pd.read_csv(args.peaks_fp).iloc[:, 1:]
        count_df = pd.DataFrame(peak_counts.mean(axis=0).values, columns=['avg_counts'])
        max_no_peaks = np.max(count_df['avg_counts'])
        min_no_peaks = np.min(count_df['avg_counts'])

        count_df['width'] = count_df.apply(lambda row: determine_bin_width(row, max_no_peaks, min_no_peaks, args.range),
                                           axis=1)
        
        bins = []
        for index, bin_size in enumerate(count_df['width'].values):
            lower_bound = 2000 + index*args.region
            upper_bound = 2000 + (index+1)*args.region
            # print(f"{lower_bound} - {upper_bound} - {bin_size}")
            bins.extend(list(range(lower_bound, upper_bound - bin_size, bin_size)))
        
        bins = np.array(bins)
    else:
        bins = np.arange(2000, 20000, args.bin_size)

    # Save bins
    with open(args.output_fp, "wb") as f:
        pickle.dump(bins, f)

if __name__ == "__main__":
    main()