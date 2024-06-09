import os
import pandas as pd
import numpy as np
import argparse
import scipy.signal
import scipy.stats
import glob
from tqdm import tqdm


def get_baseline(intensities, *, niter=100):
    # https://doi.org/10.1016/0168-583X(88)90063-8
    # https://doi.org/10.1016/j.nima.2008.11.132
    y = intensities.copy()
    n = intensities.size
    z = np.zeros(n)
    # Inner loop vectorised by rolling arrays to align pairs
    for p in reversed(range(1, niter+1)):
        a1 = y[p:n-p]
        a2 = (np.roll(y, p)[p:n-p] + np.roll(y, -p)[p:n-p]) / 2
        y[p:n-p] = np.minimum(a1, a2)
    return y


def calibrate_intensities(intensities, masses):
    scale = sum((intensities[:-1] + intensities[1:]) / 2 * np.diff(masses))
    if scale == 0:
        raise ValueError("Scale cannot be equal to zero!")
    return intensities / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        type=str,
                        help="Directory of raw spectra file")
    parser.add_argument("output_dir",
                        type=str,
                        help="Directory to save preprocessed file")
    parser.add_argument("--binned_output_dir",
                        type=str,
                        help="Directory to save binning preprocessed file",
                        default="")
    parser.add_argument("--delimiter",
                        type=str,
                        help="Delimiter used to seperate mass and intensity values",
                        default=",")
    parser.add_argument("--skiprows",
                        type=int,
                        help="Skip metadata rows",
                        default=0)

    args = parser.parse_args()

    spectra_files = glob.glob(f"{args.input_dir}/*.txt", recursive=True)

    for spectra_file in tqdm(spectra_files, total=len(spectra_files)):

        if os.path.exists(os.path.join(args.output_dir, spectra_file.split('/')[-1])):
            continue

        tmp_df = pd.read_csv(spectra_file,
                             header=None,
                             names=['mass', 'intensity'],
                             delimiter=args.delimiter,
                             skiprows=args.skiprows)

        masses = tmp_df['mass'].to_numpy()
        intensities = tmp_df['intensity'].to_numpy()

        try:

            # Apply square root transformation and smooth intensities
            intensities = np.sqrt(intensities)
            intensities = scipy.signal.savgol_filter(intensities, 21, 3)

            # Remove baseline and calibrate intensities
            baseline = get_baseline(intensities)
            intensities = intensities - baseline
            intensities = calibrate_intensities(intensities, masses)

            # Remove nans and replace negative intensities with zero
            nan_indices = np.argwhere(np.isnan(intensities))
            if nan_indices.size == intensities.size:
                intensities = np.zeros(intensities.size)
            elif nan_indices.size > 0:
                intensities = np.delete(intensities, nan_indices)
                masses = np.delete(masses, nan_indices)
            intensities[intensities < 0] = 0

        except:
            print(spectra_file)
            continue

        tmp_df = pd.DataFrame({"mass": masses,
                               "intensity": intensities})
        # Trim to 2000-20000 Da
        tmp_df = tmp_df.copy()[(tmp_df['mass'] >= 2000)
                               & (tmp_df['mass'] <= 20000)]

        tmp_df.to_csv(os.path.join(args.output_dir, spectra_file.split('/')[-1]),
                      index=False,
                      header=False)

        if args.binned_output_dir:
            try:

                bins = [2000 + i for i in range(18001)]
                binned_intensities = scipy.stats.binned_statistic(
                    tmp_df['mass'], tmp_df['intensity'], 'max', bins=bins).statistic

                np.nan_to_num(binned_intensities, copy=False, nan=0)

                binned_df = pd.DataFrame(
                    {'mass': bins[:-1], 'intensity': binned_intensities})

                binned_df.to_csv(os.path.join(args.binned_output_dir, spectra_file.split('/')[-1]),
                                 index=False,
                                 header=False)
            except:
                print(spectra_file)
                continue


if __name__ == "__main__":
    main()
