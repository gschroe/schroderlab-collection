#!/usr/bin/env python3
"""
coord_dist_2dclavgs.py

Calculate optimal 2D class ordering based on coordinate distances of particle centers of mass.

This script analyzes RELION particle star files from 2D classification to determine
the optimal ordering of 2D classes based on the coordinate distances between their
centers of mass along filaments. The output helps with stitching 2D class averages in the correct order
for initial model generation.

Author: Simon Sommerhage
Date: 2025-12-19
"""

# schroderlab-collection: Tool collection for the processing of Cryo-EM Datasets
# Copyright (C) 2025 Simon Sommerhage

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
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib.star_parser import parse_star_file


def get_stk_index(rlnReferenceImage: str):
    """Extract stack index from RELION reference image string."""
    return int(rlnReferenceImage.split("@")[0]) - 1


def get_stk_index_from_class_id(class_id: int, clavgs_metadata: pd.DataFrame):
    """Get stack index for a given class ID from metadata."""
    class_row = clavgs_metadata[clavgs_metadata["_rlnClassNumber"] == class_id]
    assert class_row.shape[0] == 1, f"Expected 1 row for class {class_id}, got {class_row.shape[0]}"
    rln_ref_image = class_row["_rlnReferenceImage"].values[0]
    return get_stk_index(rln_ref_image)


def hash_filament(_rlnMicrographName, _rlnHelicalTubeID):
    """Create unique hash for filament identification."""
    mic_stem = Path(_rlnMicrographName).stem
    return f"{mic_stem}-{_rlnHelicalTubeID}"


def hash_filament_from_particle_row(particle_row):
    """Create filament hash from particle row."""
    return hash_filament(particle_row["_rlnMicrographName"], particle_row["_rlnHelicalTubeID"])


def calculate_weighted_COM(x, y, prob):
    """Calculate weighted center of mass using probability weights."""
    com_x = np.average(x, weights=prob)
    com_y = np.average(y, weights=prob)
    return com_x, com_y


def coordinate_distance_per_filament(class_id_A: int, class_id_B: int, particles_df,
                                     ANGPIX_ORI_MICS: float, class_ids: np.ndarray,
                                     refined_coords: bool = False) -> tuple:
    """
    Calculate center-of-mass distances between two classes across all filaments.

    Parameters
    ----------
    class_id_A : int
        First class ID
    class_id_B : int
        Second class ID
    particles_df : pd.DataFrame
        DataFrame containing particle information
    ANGPIX_ORI_MICS : float
        Pixel size of original micrographs in Angstroms
    class_ids : np.ndarray
        Array of all valid class IDs
    refined_coords : bool, optional
        Whether to use refined coordinates (default: False)

    Returns
    -------
    com_dists_per_fib : np.ndarray
        Array of COM distances for each filament
    mean_max_probs_per_fib : np.ndarray
        Array of mean maximum probabilities for each filament
    """
    assert class_id_A in class_ids, f"Class {class_id_A} not found in class_ids"
    assert class_id_B in class_ids, f"Class {class_id_B} not found in class_ids"

    # Collect COM distances for each filament
    com_dists_per_fib = []
    # Collect mean max prob. of all particles from Class A and Class B per fibril
    mean_max_probs_per_fib = []

    for fid, per_fid_df in particles_df.groupby("Filament_UID"):
        classA_particles = per_fid_df[per_fid_df["_rlnClassNumber"] == class_id_A]
        if classA_particles.shape[0] == 0:
            continue
        classB_particles = per_fid_df[per_fid_df["_rlnClassNumber"] == class_id_B]
        if classB_particles.shape[0] == 0:
            continue

        # Coordinates
        x_coords_A = classA_particles["_rlnCoordinateX"] * ANGPIX_ORI_MICS
        y_coords_A = classA_particles["_rlnCoordinateY"] * ANGPIX_ORI_MICS

        x_coords_B = classB_particles["_rlnCoordinateX"] * ANGPIX_ORI_MICS
        y_coords_B = classB_particles["_rlnCoordinateY"] * ANGPIX_ORI_MICS

        if refined_coords:
            x_coords_A += classA_particles["_rlnOriginXAngst"]
            y_coords_A += classA_particles["_rlnOriginYAngst"]

            x_coords_B += classB_particles["_rlnOriginXAngst"]
            y_coords_B += classB_particles["_rlnOriginYAngst"]

        # Maximum 2D-Class assignment probability of each particle
        max_prob_A = classA_particles["_rlnMaxValueProbDistribution"]
        max_prob_B = classB_particles["_rlnMaxValueProbDistribution"]

        # Calculate COM distance
        comA_x, comA_y = calculate_weighted_COM(
            x=x_coords_A,    # unit: A
            y=y_coords_A,    # unit: A
            prob=max_prob_A
        )

        comB_x, comB_y = calculate_weighted_COM(
            x=x_coords_B,    # unit: A
            y=y_coords_B,    # unit: A
            prob=max_prob_B
        )

        com_dx = comB_x - comA_x
        com_dy = comB_y - comA_y

        com_dist = np.sqrt(com_dx**2 + com_dy**2)  # unit: A

        com_dists_per_fib.append(com_dist)

        # Calculate mean max prob
        mean_max_prob = np.append(max_prob_A, max_prob_B).mean()
        mean_max_probs_per_fib.append(mean_max_prob)

    com_dists_per_fib = np.array(com_dists_per_fib)
    mean_max_probs_per_fib = np.array(mean_max_probs_per_fib)

    return com_dists_per_fib, mean_max_probs_per_fib


def dict_to_distance_matrix(distance_dict):
    """
    Convert a dictionary of pairwise distances to a distance matrix.

    Parameters
    ----------
    distance_dict : dict
        Dictionary with keys as tuples (class_id1, class_id2) and values as distances

    Returns
    -------
    matrix : numpy.ndarray
        Square distance matrix
    class_ids : list
        Ordered list of class IDs corresponding to matrix indices
    """
    # Extract all unique class IDs
    all_classes = set()
    for (class1, class2) in distance_dict.keys():
        all_classes.add(class1)
        all_classes.add(class2)

    # Sort class IDs for consistent ordering
    class_ids = sorted(all_classes)
    n_classes = len(class_ids)

    # Create class ID to matrix index mapping
    id_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}

    # Initialize distance matrix
    matrix = np.zeros((n_classes, n_classes))

    # Fill the matrix
    for (class1, class2), distance in distance_dict.items():
        idx1 = id_to_idx[class1]
        idx2 = id_to_idx[class2]
        matrix[idx1, idx2] = distance
        matrix[idx2, idx1] = distance  # Ensure symmetry

    return matrix, class_ids


def optimal_1d_ordering(distance_matrix, class_ids):
    """
    Find optimal 1D ordering of classes based on distance matrix using hierarchical clustering.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square symmetric distance matrix
    class_ids : list
        List of class IDs corresponding to matrix rows/columns

    Returns
    -------
    sorted_class_ids : list
        Class IDs in optimal order
    sorting_indices : numpy.ndarray
        Indices of the optimal ordering
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    # Convert square distance matrix to condensed form for scipy
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering with optimal leaf ordering
    linkage_matrix = hierarchy.linkage(condensed_dist, method='average', optimal_ordering=True)

    # Extract the optimal order of classes from linkage matrix
    sorting_indices = hierarchy.leaves_list(linkage_matrix)
    sorted_class_ids = [class_ids[i] for i in sorting_indices]

    return sorted_class_ids, sorting_indices


def load_and_prepare_particles(particle_star: Path):
    """
    Load particle star file and add unique IDs.

    Parameters
    ----------
    particle_star : Path
        Path to RELION particle star file

    Returns
    -------
    particles_df : pd.DataFrame
        DataFrame with particles and unique IDs
    ANGPIX_ORI_MICS : float
        Original micrograph pixel size
    ANGPIX_PARTICLES : float
        Particle/2D class average pixel size
    """
    # Parse particle star file
    data = parse_star_file(particle_star)

    optics_df = data["optics"]
    assert optics_df.shape[0] == 1, "Expected exactly one optics group"

    ANGPIX_ORI_MICS = optics_df["_rlnMicrographOriginalPixelSize"].values[0]
    ANGPIX_PARTICLES = optics_df["_rlnImagePixelSize"].values[0]

    particles_df = data["particles"]

    # Add unique particle ID
    num_particles = particles_df.shape[0]
    particles_df["particle_UID"] = np.arange(num_particles)

    # Add unique filament ID
    particles_df["filament_hash"] = particles_df.apply(
        hash_filament_from_particle_row,
        axis=1
    )
    particles_df['Filament_UID'] = particles_df.groupby('filament_hash', sort=False).ngroup()

    # Count number of particles per filament
    particles_df["num_particles_per_filament"] = particles_df.groupby(
        "Filament_UID"
    )["_rlnClassNumber"].transform("count")

    return particles_df, ANGPIX_ORI_MICS, ANGPIX_PARTICLES


def calculate_all_pairwise_distances(particles_df, class_ids, ANGPIX_ORI_MICS):
    """
    Calculate all pairwise COM distances between classes.

    Parameters
    ----------
    particles_df : pd.DataFrame
        DataFrame with particle information
    class_ids : np.ndarray
        Array of class IDs
    ANGPIX_ORI_MICS : float
        Original micrograph pixel size

    Returns
    -------
    coord_dists_dict : dict
        Dictionary of coordinate distances for each class pair
    mean_max_probs_dict : dict
        Dictionary of mean probabilities for each class pair
    """
    coord_dists_dict = {}
    mean_max_probs_dict = {}

    for i in tqdm(range(len(class_ids)), desc="Calculating pairwise 2D class distances", position=0):
        for j in tqdm(range(i+1, len(class_ids)), leave=False, position=1):
            classA = class_ids[i]
            classB = class_ids[j]

            coord_dists, mean_max_probs = coordinate_distance_per_filament(
                classA, classB, particles_df, ANGPIX_ORI_MICS, class_ids
            )

            coord_dists_dict[(classA, classB)] = coord_dists
            mean_max_probs_dict[(classA, classB)] = mean_max_probs

    return coord_dists_dict, mean_max_probs_dict


def reduce_distances_to_metric(coord_dists_dict, metric='median'):
    """
    Reduce distance distributions to single metric values.

    Parameters
    ----------
    coord_dists_dict : dict
        Dictionary of distance arrays for each class pair
    metric : str, optional
        'median' or 'mean' (default: 'median')

    Returns
    -------
    class_dists_dict : dict
        Dictionary of single distance values for each class pair
    """
    class_dists_dict = {}
    for class_pair, val in coord_dists_dict.items():
        if metric == 'median':
            class_dists_dict[class_pair] = np.median(val)
        elif metric == 'mean':
            class_dists_dict[class_pair] = np.mean(val)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return class_dists_dict


def write_class_order_output(sorted_class_ids, dist_matrix_df, output_file: Path, float_decimals=4):
    """
    Write ordered class distances to output file in STAR-like format.

    Parameters
    ----------
    sorted_class_ids : list
        Ordered list of class IDs
    dist_matrix_df : pd.DataFrame
        Distance matrix as DataFrame
    output_file : Path
        Output file path
    float_decimals : int, optional
        Number of decimal places (default: 4)
    """
    num_classes = len(sorted_class_ids)

    # Create class distance order table
    class_dist_order = {
        "Class": [],
        "Next_Class": [],
        "Distance_Ang": []
    }

    for i in range(num_classes - 1):
        class_id = sorted_class_ids[i]
        next_class_id = sorted_class_ids[i+1]

        dist = dist_matrix_df.loc[class_id, next_class_id]

        class_dist_order["Class"].append(class_id)
        class_dist_order["Next_Class"].append(next_class_id)
        class_dist_order["Distance_Ang"].append(dist)

    class_dist_order_df = pd.DataFrame(class_dist_order)

    # Write in STAR-like format
    header = """
data_

loop_
_Class #1
_NextClass #2
_DistanceAng #3
"""

    out_star = header
    for idx, row in class_dist_order_df.iterrows():
        star_row = f"{int(row['Class'])}\t{int(row['Next_Class'])}\t{np.round(row['Distance_Ang'], float_decimals)}\n"
        out_star += star_row

    with open(output_file, mode="w") as f:
        f.write(out_star)

    return class_dist_order_df


# Plotting functions (optional)
def plot_class_distances(class_distances, classA, classB, clavgs_stk, clavgs_metadata,
                         ANGPIX_PARTICLES, output_dir):
    """Plot histogram of distances and the two class averages."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Histogram
    axes[0].hist(class_distances, bins="auto")
    axes[0].set_xlabel("d / Å")
    axes[0].set_ylabel("counts")
    axes[0].set_title(f"{classA} <-> {classB} Per-filament COM distances")

    classA_stk_idx = get_stk_index_from_class_id(classA, clavgs_metadata)
    classB_stk_idx = get_stk_index_from_class_id(classB, clavgs_metadata)

    height, width = clavgs_stk.shape[1:]
    extent = [0, width * ANGPIX_PARTICLES, 0, height * ANGPIX_PARTICLES]

    # Class A
    axes[1].imshow(clavgs_stk[classA_stk_idx], cmap="gray", extent=extent)
    axes[1].set_xlabel('X (Å)')
    axes[1].set_ylabel('Y (Å)')
    axes[1].set_title(f"Class {classA}")

    # Class B
    axes[2].imshow(clavgs_stk[classB_stk_idx], cmap="gray", extent=extent)
    axes[2].set_xlabel('X (Å)')
    axes[2].set_ylabel('Y (Å)')
    axes[2].set_title(f"Class {classB}")

    plt.tight_layout()
    output_file = output_dir / f"dist_class_{classA}_vs_{classB}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distance_matrix(dist_matrix, class_ids, output_dir, title="Distance Matrix"):
    """Plot distance matrix as heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_ids)))
    ax.set_yticks(np.arange(len(class_ids)))
    ax.set_xticklabels([f"{id}" for id in class_ids])
    ax.set_yticklabels([f"{id}" for id in class_ids])

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Distance (Å)", rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Class ID")
    ax.set_title(title)

    plt.tight_layout()
    output_file = output_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ordered_classes(sorted_class_ids, clavgs_stk, clavgs_metadata, output_dir):
    """Plot all classes in optimal order."""
    import matplotlib.pyplot as plt

    num_classes = len(sorted_class_ids)
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 2.5, 3))

    if num_classes == 1:
        axs = [axs]

    for i, class_id in enumerate(sorted_class_ids):
        stk_idx = get_stk_index_from_class_id(class_id, clavgs_metadata)
        clavg = clavgs_stk[stk_idx]
        axs[i].imshow(clavg, cmap="gray")
        axs[i].set_title(f"Class {class_id}")
        axs[i].axis("off")

    plt.tight_layout()
    output_file = output_dir / "ordered_classes.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mean_prob_vs_distance(coord_dists_dict, mean_max_probs_dict, output_dir, max_plots=10):
    """Plot mean probability vs distance joint plots for class pairs."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_plots = 0
    for (classA, classB), dists in coord_dists_dict.items():
        if n_plots >= max_plots:
            break

        mean_probs = mean_max_probs_dict[(classA, classB)]

        g = sns.jointplot(
            data={
                "mean max. prob. per fibril": mean_probs,
                "COM distance per fibril": dists,
            },
            x="mean max. prob. per fibril",
            y="COM distance per fibril",
        )
        g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
        g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

        output_file = output_dir / f"prob_vs_dist_class_{classA}_vs_{classB}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        n_plots += 1


def generate_plots(particles_df, class_ids, coord_dists_dict, mean_max_probs_dict,
                   sorted_class_ids, dist_matrix, clavgs_stk, clavgs_metadata,
                   ANGPIX_PARTICLES, output_dir):
    """
    Generate all plots if class averages are provided.

    Parameters
    ----------
    particles_df : pd.DataFrame
        Particle data
    class_ids : list
        List of class IDs
    coord_dists_dict : dict
        Dictionary of distance distributions
    mean_max_probs_dict : dict
        Dictionary of mean probabilities
    sorted_class_ids : list
        Ordered class IDs
    dist_matrix : np.ndarray
        Distance matrix
    clavgs_stk : np.ndarray
        Stack of class averages
    clavgs_metadata : pd.DataFrame
        Class average metadata
    ANGPIX_PARTICLES : float
        Pixel size
    output_dir : Path
        Output directory
    """
    print("\nGenerating plots...")

    # Plot distance matrix (unsorted)
    plot_distance_matrix(dist_matrix, class_ids, output_dir,
                        title="Pairwise 2D Class Distance Matrix")

    # Plot distance matrix (sorted)
    sorted_dist_matrix = np.zeros((len(class_ids), len(class_ids)))
    id_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}

    for i in range(len(sorted_class_ids)):
        for j in range(i+1, len(sorted_class_ids)):
            class1 = sorted_class_ids[i]
            class2 = sorted_class_ids[j]

            idx1 = id_to_idx[class1]
            idx2 = id_to_idx[class2]
            dist = dist_matrix[idx1, idx2]

            sorted_dist_matrix[i, j] = dist
            sorted_dist_matrix[j, i] = dist

    plot_distance_matrix(sorted_dist_matrix, sorted_class_ids, output_dir,
                        title="Sorted 2D Class Distance Matrix")

    # Plot ordered classes
    plot_ordered_classes(sorted_class_ids, clavgs_stk, clavgs_metadata, output_dir)

    # Plot some pairwise distance distributions
    max_dist_plots = 10
    n_plots = 0
    for (classA, classB), dists in coord_dists_dict.items():
        if n_plots >= max_dist_plots:
            break
        plot_class_distances(dists, classA, classB, clavgs_stk, clavgs_metadata,
                           ANGPIX_PARTICLES, output_dir)
        n_plots += 1

    # Plot probability vs distance
    plot_mean_prob_vs_distance(coord_dists_dict, mean_max_probs_dict, output_dir, max_plots=10)

    print(f"Plots saved to {output_dir}")


def main():
    """Main function to run the coordinate distance analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate optimal 2D class ordering based on coordinate distances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s -i particles.star -o class_order.dat
  %(prog)s -i particles.star -o class_order.dat -c class_averages.mrcs -cm class_averages.star -p
        """
    )

    # Required arguments
    parser.add_argument(
        '-i', '--input_particles',
        type=Path,
        required=True,
        help='Input RELION star file with particles from Select job after 2D classification'
    )

    parser.add_argument(
        '-o', '--output_class_order',
        type=Path,
        required=True,
        help='Output file describing optimal order of 2D classes and distances'
    )

    # Optional arguments for plotting
    parser.add_argument(
        '-c', '--class_averages',
        type=Path,
        help='MRC/MRCS file with 2D class averages (required for plotting)'
    )

    parser.add_argument(
        '-cm', '--classes_metadata',
        type=Path,
        help='RELION star file with class average metadata (required for plotting)'
    )

    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Generate plots (requires -c and -cm arguments)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        choices=['median', 'mean'],
        default='median',
        help='Metric to reduce distance distributions (default: median)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_particles.exists():
        print(f"Error: Input file {args.input_particles} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.plot:
        if args.class_averages is None or args.classes_metadata is None:
            print("Error: --plot requires both -c/--class_averages and -cm/--classes_metadata",
                  file=sys.stderr)
            sys.exit(1)

        if not args.class_averages.exists():
            print(f"Error: Class averages file {args.class_averages} does not exist",
                  file=sys.stderr)
            sys.exit(1)

        if not args.classes_metadata.exists():
            print(f"Error: Class metadata file {args.classes_metadata} does not exist",
                  file=sys.stderr)
            sys.exit(1)

    # Create output directory
    output_dir = args.output_class_order.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading particles from {args.input_particles}...")
    particles_df, ANGPIX_ORI_MICS, ANGPIX_PARTICLES = load_and_prepare_particles(args.input_particles)

    num_particles = particles_df.shape[0]
    num_filaments = particles_df["Filament_UID"].max() + 1

    print(f"Loaded {num_particles} particles from {num_filaments} filaments")
    print(f"Pixel sizes: Original micrographs = {ANGPIX_ORI_MICS} Å, Particles = {ANGPIX_PARTICLES} Å")

    # Get unique class IDs
    class_ids = particles_df["_rlnClassNumber"].unique()
    class_ids = np.sort(class_ids)
    num_classes = len(class_ids)

    print(f"Found {num_classes} unique classes: {class_ids}")

    # Calculate all pairwise distances
    print("\nCalculating pairwise coordinate distances...")
    coord_dists_dict, mean_max_probs_dict = calculate_all_pairwise_distances(
        particles_df, class_ids, ANGPIX_ORI_MICS
    )

    # Reduce to single metric
    print(f"\nReducing distance distributions using {args.metric}...")
    class_dists_dict = reduce_distances_to_metric(coord_dists_dict, metric=args.metric)

    # Convert to distance matrix
    dist_matrix, mat_class_ids = dict_to_distance_matrix(class_dists_dict)
    dist_matrix_df = pd.DataFrame(dist_matrix, index=class_ids, columns=class_ids)

    # Find optimal ordering
    print("\nFinding optimal 1D ordering using hierarchical clustering...")
    sorted_class_ids, ordering_idxs = optimal_1d_ordering(dist_matrix, class_ids)

    print(f"Optimal class order: {sorted_class_ids}")

    # Write output
    print(f"\nWriting class order to {args.output_class_order}...")
    class_dist_order_df = write_class_order_output(sorted_class_ids, dist_matrix_df,
                                                    args.output_class_order)

    print("\nClass order with neighbor distances:")
    print(class_dist_order_df.to_string(index=False))

    # Generate plots if requested
    if args.plot:
        import mrcfile

        print(f"\nLoading class averages from {args.class_averages}...")
        with mrcfile.open(args.class_averages) as mrc:
            clavgs_stk = mrc.data

        print(f"Loading class metadata from {args.classes_metadata}...")
        clavgs_data = parse_star_file(args.classes_metadata)
        # Handle both Select job format (data_) and other formats
        if "" in clavgs_data:
            clavgs_metadata = clavgs_data[""]
        elif "model_classes" in clavgs_data:
            clavgs_metadata = clavgs_data["model_classes"]
        else:
            # Take the first available data block
            clavgs_metadata = list(clavgs_data.values())[0]

        clavgs_metadata["StackIndex"] = clavgs_metadata["_rlnReferenceImage"].apply(get_stk_index)

        generate_plots(particles_df, class_ids, coord_dists_dict, mean_max_probs_dict,
                      sorted_class_ids, dist_matrix, clavgs_stk, clavgs_metadata,
                      ANGPIX_PARTICLES, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
