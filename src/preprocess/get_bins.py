import os
import argparse
import pandas as pd
import numpy as np
import math
import pickle


def determine_bin_width(row, max, min, range_width):
    return math.floor(1 + np.abs(max - row['avg_counts'])/(max - min) * range_width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_file", type=str)
    parser.add_argument("--outfile", type=str)

    args = parser.parse_args()

    peak_counts = pd.read_csv(args.count_file).iloc[:, 1:]  # remove ID column

    max_int = np.round(max(peak_counts.mean(axis=0)))
    min_int = np.round(min(peak_counts.mean(axis=0)))

    count_df = pd.DataFrame(peak_counts.mean(
        axis=0).values, columns=['avg_counts'])

    count_df['bin_width'] = count_df.apply(
        lambda row: determine_bin_width(row, max_int, min_int, 100), axis=1)

    bins = []

    for index, bin_size in enumerate(count_df['bin_width'].values):
        lower_bound = 2000 + index*500
        upper_bound = 2000 + (index+1)*500
        bins.extend(list(range(lower_bound, upper_bound, bin_size)))

    bins.append(20000)  # Appending the rightmost edge

    with open(args.outfile, "wb") as f:
        pickle.dump(bins, f)


if __name__ == "__main__":
    main()
