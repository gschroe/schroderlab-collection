"""
Estimate crossover distances from particle coordinates in a STAR file.

This script analyzes cryo-EM particle coordinates from helical filaments to estimate
the crossover distance - the distance along the fibril axis where the helix makes
a complete turn. It calculates pairwise distances between particles of the same class
within each filament and uses peak detection on the distance histogram to identify
the characteristic crossover distance.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
import starfile

# Constants
CROSS_BETA_SPACING_ANGSTROM = 4.75  # Cross-beta sheet spacing in Angstroms
DEFAULT_HISTOGRAM_BINS = 190
PLOT_STYLE_ALPHA = 0.7
PLOT_STYLE_LINEWIDTH = 1
PLOT_STYLE_MARKERSIZE = 8


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate crossover distances from particle coordinates in a STAR file. '
                    'Example usage: python ./sl_crossover_from_coords.py example_2dclass_data/particles.star 0.82'
    )
    parser.add_argument(
        'particle_star',
        type=str,
        help='Path to input particle STAR file'
    )
    parser.add_argument(
        'angpix_mic',
        type=float,
        help="Pixel size in Å of the micrographs from which the particles were extracted"
    )
    parser.add_argument(
        '--min-particles',
        type=int,
        default=20,
        help='Minimum number of particles per fibril (default: 20). '
             'All fibrils with fewer particles are disregarded'
    )
    parser.add_argument(
        '--prominence',
        type=float,
        default=0.05,
        help='Peak prominence threshold as fraction of maximum count (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=DEFAULT_HISTOGRAM_BINS,
        help=f'Number of histogram bins (default: {DEFAULT_HISTOGRAM_BINS})'
    )
    parser.add_argument(
        '--cross-beta-spacing',
        type=float,
        default=CROSS_BETA_SPACING_ANGSTROM,
        help=f'Cross-beta spacing in Å for twist calculation (default: {CROSS_BETA_SPACING_ANGSTROM})'
    )
    return parser.parse_args()


def validate_star_file(particle_star_path: Path) -> None:
    """
    Validate that the STAR file exists and is readable.

    Args:
        particle_star_path: Path to the STAR file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not particle_star_path.exists():
        raise FileNotFoundError(f"STAR file not found: {particle_star_path}")
    if not particle_star_path.is_file():
        raise ValueError(f"Path is not a file: {particle_star_path}")


def load_particle_data(particle_star_path: Path, angpix_mic: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load particle data from a STAR file and extract metadata.

    Args:
        particle_star_path: Path to the particle STAR file
        angpix_mic: Micrograph pixel size in Angstroms

    Returns:
        Tuple containing:
            - particles_df: DataFrame with particle data
            - metadata: Dictionary with optics information

    Raises:
        KeyError: If required columns are missing from the STAR file
        ValueError: If the STAR file format is invalid
    """
    try:
        particle_data = starfile.read(particle_star_path)
    except Exception as e:
        raise ValueError(f"Failed to read STAR file: {e}")

    # Validate required keys
    if "particles" not in particle_data:
        raise KeyError("STAR file missing 'particles' section")
    if "optics" not in particle_data:
        raise KeyError("STAR file missing 'optics' section")

    particles_df = particle_data["particles"]
    optics_df = particle_data["optics"]

    # Validate required columns
    required_particle_columns = [
        "rlnMicrographName", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst",
        "rlnClassNumber", "rlnCoordinateX", "rlnCoordinateY"
    ]
    missing_columns = [col for col in required_particle_columns if col not in particles_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in particles: {missing_columns}")

    required_optics_columns = ["rlnImagePixelSize", "rlnImageSize"]
    missing_optics = [col for col in required_optics_columns if col not in optics_df.columns]
    if missing_optics:
        raise KeyError(f"Missing required columns in optics: {missing_optics}")

    # Extract metadata
    angpix_box = optics_df.loc[0, "rlnImagePixelSize"]
    box_size = optics_df.loc[0, "rlnImageSize"]
    box_size_angstrom = box_size * angpix_box

    metadata = {
        "angpix_mic": angpix_mic,
        "angpix_box": angpix_box,
        "box_size": box_size,
        "box_size_angstrom": box_size_angstrom
    }

    print(f"Micrograph pixel size: {angpix_mic} Å")
    print(f"Particle pixel size: {angpix_box} Å")
    print(f"Box size: {box_size} px = {box_size_angstrom:.1f} Å")

    return particles_df, metadata


def create_filament_identifiers(particles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique identifiers for filaments and particles.

    Args:
        particles_df: DataFrame with particle data

    Returns:
        DataFrame with added 'filamentHash' and 'particleHash' columns
    """
    df = particles_df.copy()

    # Create filament hash from micrograph name and tube ID
    df["filamentHash"] = (
        df["rlnMicrographName"].astype(str) + "_" +
        df["rlnHelicalTubeID"].astype(str)
    )

    # Create unique particle hash
    df["particleHash"] = (
        df["rlnMicrographName"].astype(str) + "_" +
        df["rlnHelicalTubeID"].astype(str) + "_" +
        df["rlnHelicalTrackLengthAngst"].astype(str)
    )

    # Validate uniqueness
    num_particles = df.shape[0]
    num_unique_hashes = df["particleHash"].nunique()
    if num_particles != num_unique_hashes:
        print(f"Warning: Found duplicate particle hashes ({num_particles} particles, "
              f"{num_unique_hashes} unique hashes)")

    return df


def filter_by_filament_length(particles_df: pd.DataFrame,
                               min_particles_threshold: int) -> pd.DataFrame:
    """
    Filter particles to only include those from filaments with sufficient particles.

    Args:
        particles_df: DataFrame with particle data
        min_particles_threshold: Minimum number of particles required per filament

    Returns:
        Filtered DataFrame containing only particles from long filaments
    """
    # Count particles per filament
    particles_df["num_particles_per_filament"] = particles_df.groupby(
        "filamentHash"
    )["rlnClassNumber"].transform("count")

    # Filter
    filtered_df = particles_df[
        particles_df["num_particles_per_filament"] > min_particles_threshold
    ].copy()

    num_original_filaments = particles_df["filamentHash"].nunique()
    num_filtered_filaments = filtered_df["filamentHash"].nunique()
    num_original_particles = particles_df.shape[0]
    num_filtered_particles = filtered_df.shape[0]

    print(f"\nFilament filtering:")
    print(f"  Original: {num_original_filaments} filaments, {num_original_particles} particles")
    print(f"  After filtering (>{min_particles_threshold} particles): "
          f"{num_filtered_filaments} filaments, {num_filtered_particles} particles")

    if filtered_df.empty:
        raise ValueError(
            f"No filaments remain after filtering with threshold {min_particles_threshold}. "
            f"Try lowering --min-particles."
        )

    return filtered_df


def calculate_pairwise_distances(particles_df: pd.DataFrame,
                                 angpix_mic: float) -> np.ndarray:
    """
    Calculate pairwise distances between particles of the same class within each filament.

    For each filament, this function computes the Euclidean distance between all pairs
    of particles that belong to the same 2D class. This reveals the periodic spacing
    of crossovers along the fibril axis.

    Args:
        particles_df: DataFrame with particle data (must have filamentHash column)
        angpix_mic: Micrograph pixel size in Angstroms for unit conversion

    Returns:
        Array of pairwise distances in Angstroms
    """
    class_numbers = particles_df["rlnClassNumber"].unique()
    print(f"\nCalculating pairwise distances for {len(class_numbers)} classes...")

    all_distances = []

    for filament_hash, filament_particles in particles_df.groupby("filamentHash"):
        for class_num in class_numbers:
            # Get particles of this class in this filament
            class_particles = filament_particles[
                filament_particles["rlnClassNumber"] == class_num
            ]

            if class_particles.shape[0] < 2:
                continue

            # Extract coordinates
            x_coords = class_particles["rlnCoordinateX"].values
            y_coords = class_particles["rlnCoordinateY"].values
            points = np.column_stack((x_coords, y_coords))

            # Calculate pairwise distances
            distances = pdist(points, metric='euclidean')

            # Validate distance count
            n_points = len(x_coords)
            n_distances = len(distances)
            expected_distances = n_points * (n_points - 1) // 2
            if n_distances != expected_distances:
                raise ValueError(
                    f"Unexpected number of distances: got {n_distances}, "
                    f"expected {expected_distances} for {n_points} points"
                )

            all_distances.extend(distances)

    # Convert to Angstroms
    distances_array = np.array(all_distances) * angpix_mic
    print(f"Calculated {len(distances_array)} pairwise distances")

    return distances_array


def detect_peaks(counts: np.ndarray,
                bin_centers: np.ndarray,
                prominence_factor: float) -> Tuple[np.ndarray, Dict]:
    """
    Detect peaks in the distance histogram.

    Args:
        counts: Histogram bin counts
        bin_centers: Center position of each histogram bin
        prominence_factor: Peak prominence threshold as fraction of max count

    Returns:
        Tuple of (peak_indices, peak_properties)

    Raises:
        ValueError: If no peaks are detected
    """
    prominence_threshold = counts.max() * prominence_factor
    peaks, properties = find_peaks(
        counts,
        height=None,
        prominence=prominence_threshold
    )

    if len(peaks) == 0:
        raise ValueError(
            f"No peaks detected with prominence factor {prominence_factor}. "
            f"Try lowering the --prominence value."
        )

    print(f"\nDetected {len(peaks)} peaks at distances (Å):")
    for peak_idx in peaks:
        print(f"  {bin_centers[peak_idx]:.1f} Å (count: {counts[peak_idx]:.0f})")

    return peaks, properties


def plot_histogram_with_peaks(distances_angstrom: np.ndarray,
                              prominence_factor: float,
                              min_particles_threshold: int,
                              num_bins: int,
                              save_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create and save a histogram of pairwise distances with detected peaks marked.

    Args:
        distances_angstrom: Array of distances in Angstroms
        prominence_factor: Peak prominence threshold for peak detection
        min_particles_threshold: Minimum particles threshold (for plot title)
        num_bins: Number of histogram bins
        save_path: Path where to save the plot

    Returns:
        Tuple of (counts, bin_centers, peak_indices)
    """
    plt.figure(num="Intra-class pairwise distances", figsize=(10, 6))

    # Create histogram
    counts, bin_edges, _ = plt.hist(
        distances_angstrom,
        bins=num_bins,
        edgecolor='black',
        linewidth=0.5
    )

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Detect peaks
    peaks, _ = detect_peaks(counts, bin_centers, prominence_factor)

    # Plot peaks
    for n, peak_idx in enumerate(peaks, start=1):
        plt.axvline(
            bin_centers[peak_idx],
            color='red',
            linestyle='--',
            alpha=PLOT_STYLE_ALPHA,
            linewidth=PLOT_STYLE_LINEWIDTH
        )
        plt.plot(
            bin_centers[peak_idx],
            counts[peak_idx],
            'ro',
            markersize=PLOT_STYLE_MARKERSIZE
        )
        plt.text(
            bin_centers[peak_idx],
            counts[peak_idx],
            f'n={n}: {bin_centers[peak_idx]:.1f} Å',
            rotation=90,
            verticalalignment='bottom',
            horizontalalignment='right'
        )

    plt.xlabel("Pairwise intra-class distances [Å]")
    plt.ylabel(f"Counts (total = {len(distances_angstrom)})")
    plt.title(f"Crossover estimation (min. particles per fibril = {min_particles_threshold})")
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

    return counts, bin_centers, peaks


def calculate_helical_twist(crossover_distance: float,
                           cross_beta_spacing: float) -> float:
    """
    Calculate helical twist angle from crossover distance.

    The twist angle represents the rotation per cross-beta spacing along the fibril axis.

    Args:
        crossover_distance: Distance for one complete helical turn (Å)
        cross_beta_spacing: Rise per cross-beta sheet (Å)

    Returns:
        Helical twist angle in degrees
    """
    twist_degrees = (cross_beta_spacing / crossover_distance) * 180
    return twist_degrees


def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    particle_star_path = Path(args.particle_star)
    angpix_mic = args.angpix_mic
    min_particles_threshold = args.min_particles
    prominence_factor = args.prominence
    num_bins = args.bins
    cross_beta_spacing = args.cross_beta_spacing

    try:
        # Validate input
        validate_star_file(particle_star_path)

        # Load data
        print(f"Loading particle data from: {particle_star_path}")
        particles_df, metadata = load_particle_data(particle_star_path, angpix_mic)

        # Create identifiers
        particles_df = create_filament_identifiers(particles_df)

        # Filter by filament length
        filtered_particles_df = filter_by_filament_length(
            particles_df,
            min_particles_threshold
        )

        # Calculate pairwise distances
        distances_angstrom = calculate_pairwise_distances(
            filtered_particles_df,
            angpix_mic
        )

        # Create output path
        save_path = particle_star_path.parent / f"Intra_class_distance_histogram_ths{min_particles_threshold}.svg"

        # Plot histogram and detect peaks
        counts, bin_centers, peaks = plot_histogram_with_peaks(
            distances_angstrom,
            prominence_factor,
            min_particles_threshold,
            num_bins,
            save_path
        )

        # Extract crossover estimate from first peak
        crossover_estimate = bin_centers[peaks[0]]
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Estimated crossover distance: {crossover_estimate:.2f} Å")

        # Calculate helical twist
        twist = calculate_helical_twist(crossover_estimate, cross_beta_spacing)
        print(f"For a rise of {cross_beta_spacing} Å (cross-beta spacing):")
        print(f"  Helical twist: ±{twist:.2f}°")
        print(f"{'='*60}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
