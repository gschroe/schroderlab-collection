#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read in a particle STAR file and corresponding class averages (.mrcs),
compute class polarity, flip classes for consistent orientation,
and write out a new .mrcs file.

Usage:
  python polarity.py -i particles.star -c class_averages.mrcs -o output_flipped.mrcs
"""

# schroderlab-collection: Tool collection for the processing of Cryo-EM Datasets
# Copyright (C) 2025 Gunnar Schroeder

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import starfile
import mrcfile
from collections import Counter
import collections

def read_particle_star(filepath: Path) -> (pd.DataFrame, collections.OrderedDict):
    """
    Reads a .star file and returns the "particles" DataFrame and the full OrderedDict.
    """
    if filepath.suffix != ".star":
        raise ValueError("File is not in .star format")
    all_data = starfile.read(filepath)
    if "particles" not in all_data:
        raise ValueError(f"{filepath} does not contain a 'particles' table")
    particle_df = all_data["particles"].copy()
    return particle_df, all_data


def hash_particle_df(particle_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Adds filamentHash and particleHash columns to a copy of the input DataFrame.
    """
    df = particle_df.copy()
    # Drop any old hash columns if present
    for col in ["Hash ID", "Hash", "filamentHash", "particleHash"]:
        if col in df.columns:
            if verbose:
                print(f"Dropping old '{col}' column")
            df = df.drop(columns=[col])
    # We require rlnHelicalTubeID and rlnMicrographName to build a filamentHash
    if not {"rlnHelicalTubeID", "rlnMicrographName", "rlnHelicalTrackLengthAngst"}.issubset(df.columns):
        raise KeyError("Missing required columns to hash particles (rlnHelicalTubeID, rlnMicrographName, rlnHelicalTrackLengthAngst)")
    df["filamentHash"] = df["rlnHelicalTubeID"].astype(str) + df["rlnMicrographName"]
    df["particleHash"] = (
        df["rlnHelicalTubeID"].astype(str)
        + df["rlnMicrographName"]
        + df["rlnHelicalTrackLengthAngst"].astype(str)
    )
    total_filaments = df["filamentHash"].nunique()
    total_particles = df["particleHash"].nunique()
    if total_particles != len(df.index):
        raise AssertionError("particle_df contains duplicate particles after hashing")
    if verbose:
        print(f"Total filaments: {total_filaments}, total particles: {total_particles}")
    return df


def add_classID_to_particle_df(hashed_particle_df: pd.DataFrame, classIDdict: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Adds a 'classID' column to a copy of the hashed_particle_df using a mapping from rlnClassNumber to an integer ID.
    """
    df = hashed_particle_df.copy()
    if "classID" in df.columns:
        if verbose:
            print("Dropping old 'classID' column")
        df = df.drop(columns=["classID"])
    if "rlnClassNumber" not in df.columns:
        raise KeyError("hashed_particle_df does not contain 'rlnClassNumber'")
    df["classID"] = df["rlnClassNumber"].map(classIDdict)
    total = df["classID"].nunique()
    if verbose:
        print(f"Assigned {total} unique classIDs")
    return df


def add_filamentID_to_particle_df(hashed_particle_df: pd.DataFrame, filamentIDdict: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Adds a 'filamentID' column to a copy of the hashed_particle_df using a mapping from filamentHash to an integer ID.
    """
    df = hashed_particle_df.copy()
    if "filamentID" in df.columns:
        if verbose:
            print("Dropping old 'filamentID' column")
        df = df.drop(columns=["filamentID"])
    if "filamentHash" not in df.columns:
        raise KeyError("hashed_particle_df does not contain 'filamentHash'")
    df["filamentID"] = df["filamentHash"].map(filamentIDdict)
    total = df["filamentID"].nunique()
    if verbose:
        print(f"Assigned {total} unique filamentIDs")
    return df


def compute_particle_counts_df(
    hashed_particle_df: pd.DataFrame,
    classIDdict: dict = None,
    filamentIDdict: dict = None,
    verbose: bool = False
) -> (pd.DataFrame, pd.DataFrame, Counter, dict, Counter, dict):
    """
    Computes a class × filament particle count matrix and returns:
      - counts_df: DataFrame of shape (n_classes, n_filaments)
      - IDadded_particle_df: DataFrame with 'classID' and 'filamentID' columns added
      - classcount: Counter of number of particles per rlnClassNumber (1‐based)
      - classIDdict: mapping from rlnClassNumber to classID (0..Nc-1)
      - filamentcount: Counter of number of particles per filamentHash
      - filamentIDdict: mapping from filamentHash to filamentID (int)
    """
    if "rlnClassNumber" not in hashed_particle_df.columns or "filamentHash" not in hashed_particle_df.columns:
        raise KeyError("hashed_particle_df must contain 'rlnClassNumber' and 'filamentHash'")
    # Count unique classes and filaments (rlnClassNumber is 1‐based in STAR)
    classcount = Counter(hashed_particle_df["rlnClassNumber"].to_list())
    if classIDdict is None:
        # classIDdict maps 1‐based rlnClassNumber → zero‐based index into counts matrix
        classIDdict = {val: idx for idx, val in enumerate(sorted(classcount.keys()))}
    filamentcount = Counter(hashed_particle_df["filamentHash"].to_list())
    if filamentIDdict is None:
        filamentIDdict = {val: idx for idx, val in enumerate(filamentcount.keys())}
    if verbose:
        print(f"Total classes = {len(classcount)}, total filaments = {len(filamentcount)}, total particles = {len(hashed_particle_df)}")
        print("Particles per class:", classcount)
    # Add integer IDs for class (classID) and filament (filamentID)
    df_with_classID = add_classID_to_particle_df(hashed_particle_df, classIDdict, verbose=False)
    df_with_IDs = add_filamentID_to_particle_df(df_with_classID, filamentIDdict, verbose=False)
    # Build counts matrix
    n_classes = len(classIDdict)
    n_filaments = len(filamentIDdict)
    count_matrix = np.zeros((n_classes, n_filaments))
    for _, row in df_with_IDs.iterrows():
        cID = int(row["classID"])
        fID = int(row["filamentID"])
        count_matrix[cID, fID] += 1
    counts_df = pd.DataFrame(
        count_matrix,
        index=[str(x) for x in sorted(classIDdict.keys())],  # 1‐based labels
        columns=[str(i) for i in range(n_filaments)],
    )
    counts_df.index.name = "rlnClassNumber"
    counts_df.columns.name = "filamentID"
    return counts_df, df_with_IDs, classcount, classIDdict, filamentcount, filamentIDdict


def compute_polarity_matrix_unique_threshold(unique_class_numbers, IDadded_particle_df, min_shared=5):
    """
    For each pair of classes (i,j) in unique_class_numbers (1‐based labels),
    compute a polarity score P_ij, but only if there are at least `min_shared` filaments
    carrying both classes.

    Polarity is defined as the sign of cos(psi_i - psi_j), averaged over all common filaments.
    If fewer than min_shared filaments are common, set P_ij = 0.

    Args:
      unique_class_numbers : list[int]
          The unique class numbers (1‐based ints from STAR).
      IDadded_particle_df : pandas DataFrame
          Must contain columns:
            - 'rlnClassNumber'  (int, 1‐based)
            - 'rlnAnglePsi'     (float, in degrees)
            - 'rlnMaxValueProbDistribution' (float weight)
            - 'filamentID'      (int)

      min_shared : int
          Minimum number of filaments that must carry both classes before trusting the average.
          If fewer filaments appear in common, P_ij is set to 0.

    Returns:
      P : np.ndarray, shape (N, N)
        Polarity matrix with entries in [–1,+1], or 0 if not enough shared filaments.
        N = len(unique_class_numbers).
        The ordering of rows/columns in P matches the ordering in unique_class_numbers.
    """
    import numpy as np

    N = len(unique_class_numbers)
    P = np.zeros((N, N), dtype=float)

    # Build an index lookup: 1‐based classID → row/column index in P
    index_of = {cl: idx for idx, cl in enumerate(unique_class_numbers)}

    for i, cl1 in enumerate(unique_class_numbers):
        # cl1 is 1‐based label; select its rows:
        df1 = IDadded_particle_df[IDadded_particle_df.rlnClassNumber.astype(int) == cl1]
        grouped1 = df1.groupby('filamentID')

        for j, cl2 in enumerate(unique_class_numbers):
            if j <= i:
                continue

            # Select df for cl2:
            df2 = IDadded_particle_df[IDadded_particle_df.rlnClassNumber.astype(int) == cl2]
            grouped2 = df2.groupby('filamentID')

            # Find filaments where both cl1 and cl2 appear:
            common_filaments = np.intersect1d(df1.filamentID.unique(),
                                              df2.filamentID.unique())

            # ------- FIX: when printing “shared filaments,” use cl1 and cl2, not i/j -------
            # (i,j are indices of unique_class_numbers; cl1/cl2 are the actual 1‐based labels)
            fil_i = set(IDadded_particle_df.loc[
                IDadded_particle_df.rlnClassNumber == cl1, "filamentID"
            ])
            fil_j = set(IDadded_particle_df.loc[
                IDadded_particle_df.rlnClassNumber == cl2, "filamentID"
            ])
            n_shared = len(fil_i.intersection(fil_j))
            print(f"Shared filaments between class {cl1} and class {cl2}: {n_shared}")
            # -------------------------------------------------------------------------------

            if len(common_filaments) < min_shared:
                P[i, j] = 0.0
                continue

            s_list = []
            for fid in common_filaments:
                psi1 = np.average(
                    grouped1.get_group(fid)['rlnAnglePsi'].astype(float),
                    weights=grouped1.get_group(fid)['rlnMaxValueProbDistribution'].astype(float)
                )
                psi2 = np.average(
                    grouped2.get_group(fid)['rlnAnglePsi'].astype(float),
                    weights=grouped2.get_group(fid)['rlnMaxValueProbDistribution'].astype(float)
                )

                psi1_rad = np.deg2rad(psi1)
                psi2_rad = np.deg2rad(psi2)

                s = np.sign(np.cos(psi1_rad - psi2_rad))
                s_list.append(s)

            P[i, j] = float(np.mean(s_list))

    P = P + P.T
    return P


def _local_flip_improve(P: np.ndarray, x_init: np.ndarray) -> np.ndarray:
    """
    Starting from x_init (±1 vector), do greedy single-bit flips until convergence.
    Return the locally optimal ±1 vector.
    """
    x = x_init.copy()
    N = len(x)
    Px = P.dot(x)
    score = x * Px

    improved = True
    while improved:
        improved = False
        for i in np.random.permutation(N):
            gain = -2 * score[i]
            if gain > 1e-12:
                x[i] = -x[i]
                Px -= 2 * (-x[i]) * P[:, i]
                score = x * Px
                improved = True
    return x


def optimize_polarity_advanced(
    P: np.ndarray,
    n_restarts: int = 50,
    use_spectral: bool = True
) -> tuple:
    """
    More intensive optimizer for the “polarity” Ising objective:
       F(x) = sum_{i<j} x[i] * x[j] * P[i,j],   x[i] in {+1, -1}.

    Args:
      P           : (N×N) symmetric numpy array of polarity scores.
      n_restarts  : how many random ±1 seeds to try (default 50).
      use_spectral: if True, include the “sign of top-eigenvector” as one initialization.

    Returns:
      (x_best, F_best):
        x_best  : length-N array of ±1 that (approximately) maximizes F(x).
        F_best  : the objective value F(x_best) = ∑_{i<j} x_i x_j P_{ij}.
    """
    N = P.shape[0]
    if P.shape != (N, N) or not np.allclose(P, P.T, atol=1e-8):
        raise ValueError("P must be a symmetric N×N array.")

    def objective(x_vec: np.ndarray) -> float:
        # xᵀ P x = 2 * ∑_{i<j} x_i x_j P[i,j], because P[i,i]=0
        return 0.5 * x_vec.dot(P.dot(x_vec))

    x_best = None
    F_best = -np.inf

    # 1) spectral initialization
    if use_spectral:
        w, v = np.linalg.eigh(P)
        v_top = v[:, np.argmax(w)]
        x_spec = np.sign(v_top)
        x_spec[x_spec == 0] = 1
        x_loc = _local_flip_improve(P, x_spec)
        F_loc = objective(x_loc)
        x_best = x_loc
        F_best = F_loc

    # 2) random restarts
    for _ in range(n_restarts):
        x0 = np.random.choice([1, -1], size=N)
        x_loc = _local_flip_improve(P, x0)
        F_loc = objective(x_loc)
        if F_loc > F_best:
            x_best, F_best = x_loc, F_loc
        print("F_loc ", F_loc)
    print("Ideal maximum if all P[i,j]=+1:", N * (N - 1) / 2)

    return x_best, F_best


def main():
    parser = argparse.ArgumentParser(
        description="Flip 2D class-average images based on polarity so all point consistently."
    )
    parser.add_argument(
        "-i", "--input-star", type=str, required=True,
        help="Path to the particle STAR file (data.star) with 1-based rlnClassNumber, rlnAnglePsi, etc."
    )
    parser.add_argument(
        "-c", "--class-mrcs", type=str, required=True,
        help="Path to the input class averages file (.mrcs)"
    )
    parser.add_argument(
        "-o", "--output-mrcs", type=str, required=True,
        help="Path for the output flipped class averages file (.mrcs)"
    )
    args = parser.parse_args()

    particle_star = Path(args.input_star)
    class_mrcs = Path(args.class_mrcs)
    output_mrcs = Path(args.output_mrcs)

    # 1) Read particle STAR
    particle_df, all_particle_data = read_particle_star(particle_star)

    # 2) Hash particles
    hashed_df = hash_particle_df(particle_df, verbose=True)

    # 3) Compute class/filament counts and get IDadded_particle_df
    _, IDadded_particle_df, _, classIDdict, _, filamentIDdict = compute_particle_counts_df(
        hashed_df, verbose=True
    )

    # 4) Determine which 1-based class IDs actually appear in the STAR
    present_classes = sorted(set(hashed_df["rlnClassNumber"].astype(int)))
    print("Present 1-based classes in STAR:", present_classes)

    # 5) Build a small polarity matrix only for those present 1-based classes
    print("Computing polarity matrix for present classes...")
    P_small = compute_polarity_matrix_unique_threshold(
        present_classes,
        IDadded_particle_df,
        min_shared=3
    )

    # 6) Optimize flips for present classes
    print("Optimizing polarity for present classes...")
    flip_present, F_best = optimize_polarity_advanced(
        P_small,
        n_restarts=100,
        use_spectral=True
    )
    print("Flip solution for present classes:")
    for cls, flip in zip(present_classes, flip_present):
        print(f"  Class {cls}: {'no flip' if flip > 0 else 'flip'}")
    print(f"Objective value F_best = {F_best:.6f}")

    # 7) Read class-average images from .mrcs
    with mrcfile.open(class_mrcs, permissive=True) as mrcs:
        class_stack = mrcs.data.astype(np.float32)  # shape = (N, H, W)
        voxel_size = mrcs.voxel_size  # preserve voxel size metadata

    n_slices = class_stack.shape[0]
    print(f"Number of slices in MRCs: {n_slices} (class IDs 1..{n_slices})")

    # 8) Build a “full” flip array of length n_slices, default +1 (no flip)
    flip_full = np.ones(n_slices, dtype=int)

    # 9) Map each present 1-based class ID to its 0-based slice index = (classID−1)
    for cls, flip in zip(present_classes, flip_present):
        idx0 = cls - 1                                # ← FIX: convert 1‐based to 0‐based
        if 0 <= idx0 < n_slices:
            flip_full[idx0] = flip

    # 10) Apply flips to each class-average image
    flipped_stack = np.empty_like(class_stack)
    for idx in range(n_slices):
        img = class_stack[idx]
        if flip_full[idx] < 0:
            flipped_stack[idx] = np.fliplr(img)
        else:
            flipped_stack[idx] = img.copy()

    # 11) Write out new .mrcs with flipped images
    print(f"Writing flipped class averages to {output_mrcs} ...")
    with mrcfile.new(output_mrcs, overwrite=True) as mrc:
        mrc.set_data(flipped_stack.astype(np.float32))
        try:
            mrc.voxel_size = voxel_size
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()

