# %%
# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import starfile #TODO implement own star Parser
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
import argparse

# %%
# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate crossover distances from particle coordinates in a STAR file'
    )
    parser.add_argument(
        'particle_star',
        type=str,
        help='Path to input particle STAR file'
    )
    parser.add_argument(
        'angpix_mic',
        type=float,
        help="Pixel size in Å of the micrographs from which the particles where extracted"
    )
    parser.add_argument(
        '--min-particles',
        type=int,
        default=20,
        help='Minimum number of particles per fibril (default: 20). All fibrils wiht less particles are disregarded'
    )
    # TODO add interactive plot with thrshold slider
    parser.add_argument(
        '--prominence',
        type=float,
        default=0.05,
        help='Peak prominence threshold as fraction of maximum count (default: 0.05 = 5%%)'
    )
    return parser.parse_args()

# %%
# Inputs:

args = parse_args()
particle_star = Path(args.particle_star)
len_ths = args.min_particles
prominence_factor = args.prominence

parti_dict = starfile.read(particle_star)

# get original micrograph pixel size in angstrom:
ANGPIX_MIC = args.angpix_mic
print(f"Micrograph pixel size: {ANGPIX_MIC} Å")

# Get particle pixel size:
ANGPIX_BOX = parti_dict["optics"].loc[0, "rlnImagePixelSize"]
print(f"Particle pixel size: {ANGPIX_BOX} Å")

# Get particle box size:
BOX_SIZE = parti_dict["optics"].loc[0, "rlnImageSize"]
BOX_SIZE_ANG = BOX_SIZE * ANGPIX_BOX
print(f"Box size: {BOX_SIZE} px = {BOX_SIZE_ANG} Å")

# get particle metadata as pandas dataframe:
parti_df = parti_dict["particles"]

# %% 
## Hardcoded Inputs for interactive debugging:
#particle_star = Path("/home/simon/judac_scratch_mount/sommerhage2/Aros_sarkosyl_wash_slot3_Krios_K3_20250612/relion_proc/Select/job089/particles.star") 
#ANGPIX_MIC = 0.82
#print(f"Micrograph pixel size: {ANGPIX_MIC} Å")
#parti_dict = starfile.read(particle_star)
#
## Get particle pixel size:
#ANGPIX_BOX = parti_dict["optics"].loc[0, "rlnImagePixelSize"]
#print(f"Particle pixel size: {ANGPIX_BOX} Å")
#
## Get particle box size:
#BOX_SIZE = parti_dict["optics"].loc[0, "rlnImageSize"]
#BOX_SIZE_ANG = BOX_SIZE * ANGPIX_BOX
#print(f"Box size: {BOX_SIZE} px = {BOX_SIZE_ANG} Å")
#
## get particle metadata as pandas dataframe:
#parti_df = parti_dict["particles"]
#
#len_ths = 20
#prominence_factor = 0.05

# %%
# Hash filaments and particles

parti_df["filamentHash"] = parti_df["rlnMicrographName"].astype(str) + parti_df["rlnHelicalTubeID"].astype(str)
parti_df["particleHash"] = parti_df["rlnMicrographName"].astype(str) + parti_df["rlnHelicalTubeID"].astype(str) + parti_df["rlnHelicalTrackLengthAngst"].astype(str)

num_filaments = parti_df["filamentHash"].nunique()
num_particles = parti_df.shape[0]
assert num_particles == parti_df["particleHash"].nunique()


# %%
# Calculate number of particles per filament
parti_df["num_particles_per_filament"] = parti_df.groupby("filamentHash")["rlnClassNumber"].transform(lambda x: len(x))

# Filter by number of particles per filament
parti_df_long_filaments = parti_df[parti_df["num_particles_per_filament"] > len_ths]

# %%
# Iterate over filaments and classes 
# For each filament: Calculate pairwise distances of all particles from the same class for all (provided) classes.

class_numbers = parti_df_long_filaments["rlnClassNumber"].unique() # either all class numbers or provided list of classes (#TODO)

pairwise_intra_class_distances = []
for filament_hash, filament_particles in parti_df_long_filaments.groupby("filamentHash"):
    for class_num in class_numbers:
        class_filament_particles = filament_particles[filament_particles["rlnClassNumber"] == class_num]
        if class_filament_particles.shape[0] == 0:
            continue
        
        x_coords = class_filament_particles["rlnCoordinateX"]
        y_coords = class_filament_particles["rlnCoordinateY"]
        
        points = np.column_stack((x_coords, y_coords))
        
        # Calculate pairwise distances
        distances = pdist(points, metric='euclidean')

        # sanity check:
        n_points = len(x_coords)
        n_distances = len(distances)
        assert n_distances == n_points * (n_points - 1) / 2

        pairwise_intra_class_distances += list(distances)
    
pairwise_intra_class_distances = np.array(pairwise_intra_class_distances)
pairwise_intra_class_distances_ang = pairwise_intra_class_distances * ANGPIX_MIC

num_distances = len(pairwise_intra_class_distances_ang)


plt.figure(num="Intra class pairwise distances")
#plt.hist(pairwise_intra_class_distances_ang, bins="auto")
counts, bin_edges, patches = plt.hist(pairwise_intra_class_distances_ang, bins=190)
plt.xlabel("pairwise intra-class distances [Å]")
plt.ylabel(f"Counts (total = {num_distances})")

# Peak detection
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
peaks, properties = find_peaks(counts, height=None, prominence=counts.max()*prominence_factor)

# Plot peaks
for n, peak_idx in enumerate(peaks, start=1):   # this starts at the first detected peak. Just the naming variable n stats at 1 to indicate that its not the paek at pairwise distance == 0
    plt.axvline(bin_centers[peak_idx], color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.plot(bin_centers[peak_idx], counts[peak_idx], 'ro', markersize=8)
    plt.text(bin_centers[peak_idx], counts[peak_idx], f'n={n}: {bin_centers[peak_idx]:.1f} Å',
             rotation=90, verticalalignment='bottom', horizontalalignment='right')

print(f"\nDetected {len(peaks)} peaks at distances (Å):")
for peak_idx in peaks:
    print(f"  {bin_centers[peak_idx]:.1f} Å (count: {counts[peak_idx]:.0f})")

print(f"Minimum particles per fibrils threshold: {len_ths}")

plt.title(f"Crossover estimation (min. particles per fibril = {len_ths})")
plt.savefig(f"Inter_class_distance_histogram_ths{len_ths}.svg")
plt.show()

print(f"Particles from: {particle_star}")
# %%
# Get crossover from first non-zero distance peak:

crossover_estimate_ang = bin_centers[peaks[0]]
print(f"Estimated crossover distance: {crossover_estimate_ang} Å")

cross_beta_dist_ang = 4.75
twist_deg = cross_beta_dist_ang / crossover_estimate_ang * 180
print(f"For a rise of {cross_beta_dist_ang} Å (cross-beta spacing) this corresponds to a helical twist of ± {twist_deg} °")

