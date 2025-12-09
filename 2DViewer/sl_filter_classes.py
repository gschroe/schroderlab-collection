#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script filters a particle STAR file and a corresponding class averages MRCs file
to include only the classes specified in a set file (one class ID per line).

Usage:
  python sl_filter_classes.py -i particles.star -c class_averages.mrcs -s class_set.txt -o output_folder
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import starfile
import mrcfile

def read_set_file(set_filepath: Path) -> set:
    """Reads a set file (one class ID per line) and returns a set of integers."""
    with open(set_filepath, "r") as f:
        lines = f.readlines()
    # Strip whitespace and ignore blank lines
    ids = {int(line.strip()) for line in lines if line.strip()}
    #ids = {int(line.strip()) + 1 for line in lines if line.strip()}

    #ids = {int(line.strip()) - 1 for line in lines if line.strip()}


    return ids

def filter_particle_star(particle_star_path: Path, class_set: set) -> (pd.DataFrame, dict):
    """Reads a particle STAR file and returns a filtered particle DataFrame and all_data dict."""
    all_data = starfile.read(particle_star_path)
    particle_df = all_data["particles"].copy()
    # Ensure there is no extra whitespace in column names:
    particle_df.columns = particle_df.columns.str.strip()
    # Convert rlnClassNumber to integer for filtering.
    particle_df["rlnClassNumber"] = particle_df["rlnClassNumber"].astype(int)
    filtered_df = particle_df[particle_df["rlnClassNumber"].isin(class_set)].copy()
        
    # Now, reassign the class numbers to be sequential.
    unique_filtered = sorted(filtered_df["rlnClassNumber"].unique())
    # Create a mapping: original class -> new sequential number starting at 1.
    mapping = {old: new for new, old in enumerate(unique_filtered, start=1)}
    print("Mapping of original to new class numbers:", mapping)
        
    # Update the rlnClassNumber column.
    filtered_df["rlnClassNumber"] = filtered_df["rlnClassNumber"].map(mapping)
        
    return filtered_df, all_data

def write_filtered_star(filtered_df: pd.DataFrame, all_data: dict, output_path: Path):
    """Writes out a new STAR file with filtered particles."""
    all_data["particles"] = filtered_df
    out_file = output_path / "filtered_particles.star"
    starfile.write(all_data, str(out_file), overwrite=True)
    print(f"Filtered STAR file written to {out_file}")

# def filter_class_averages_mrcs(class_mrcs_path: Path, filtered_class_ids: set, particle_df: pd.DataFrame, output_path: Path):
def filter_class_averages_mrcs(class_mrcs_path: Path, filtered_class_ids: set, output_path: Path):    
    """
    Reads the class averages MRCs file (a stack of images).
    It is assumed that the ordering of images in the stack corresponds to the sorted order
    of the unique class numbers from the particle STAR file.
    
    The function selects only those images corresponding to class IDs in filtered_class_ids
    and writes out a new MRCs file.
    """
    # Get the full list of class averages from the MRCs file.
    class_avgs, voxel_size = read_mrcs_file(class_mrcs_path)
    # Determine the unique class numbers present in the particle STAR file.
    # We assume that these unique class numbers are in sorted order and correspond one-to-one
    # with the images in the MRCs file.
    # unique_classes = sorted(set(particle_df["rlnClassNumber"].astype(int)))
    # print("Unique class numbers (from particle STAR):", unique_classes)
    
    # Build a mapping: index -> class ID, where index corresponds to the order in the MRCs stack.
    # This assumes that the first image corresponds to the smallest class number, and so on.
    # mapping = {unique_classes[i]: i for i in range(len(unique_classes))}        
    # Determine which indices to keep
    # indices_to_keep = [mapping[cls] for cls in unique_classes if cls in filtered_class_ids]
    
    indices_to_keep = filtered_class_ids
    print("Indices to keep in MRCs:", indices_to_keep)
    
    # Select only these images.
    filtered_avgs = [class_avgs[i-1] for i in indices_to_keep]
    filtered_avgs = np.array(filtered_avgs)
    
    # Write out a new MRCs file.
    out_file = output_path / "filtered_class_averages.mrcs"
    with mrcfile.new(str(out_file), overwrite=True) as mrc:
        mrc.set_data(filtered_avgs.astype(np.float32))
    print(f"Filtered class averages written to {out_file}")

def read_mrcs_file(filepath: Path) -> (list, np.recarray):
    """Reads an MRCs file and returns a list of images and the voxel size."""
    data = []
    with mrcfile.open(filepath, permissive=True) as mrcs:
        for frame in mrcs.data:
            data.append(frame)
        voxel_size = mrcs.voxel_size
    return data, voxel_size

def main():
    parser = argparse.ArgumentParser(
        description="Filter particle STAR and class averages MRCs files by a set of class IDs."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the particle STAR file (data.star)")
    parser.add_argument("-c", "--class-averages", type=str, required=True,
                        help="Path to the class averages file (.mrcs)")
    parser.add_argument("-s", "--set", type=str, required=True,
                        help="Path to the set file (one class ID per line)")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output folder")
    args = parser.parse_args()
    
    particle_star_path = Path(args.input)
    class_mrcs_path = Path(args.class_averages)
    set_filepath = Path(args.set)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("Particle STAR file:", particle_star_path)
    print("Class averages file:", class_mrcs_path)
    print("Set file:", set_filepath)
    print("Output folder:", output_folder)
    
    # Read the set file
    class_set = read_set_file(set_filepath)
    print("Selected class IDs:", class_set)

    particle_df = starfile.read(particle_star_path) 
    unique_classes = sorted(set(particle_df['particles']["rlnClassNumber"].astype(int)))
    print("Unique classes in STAR file:", unique_classes)
    
    # Read and filter the particle STAR file.
    filtered_particle_df, all_particle_data = filter_particle_star(particle_star_path, class_set)
    print(f"Filtered particle STAR file contains {len(filtered_particle_df)} particles.")
    write_filtered_star(filtered_particle_df, all_particle_data, output_folder)
    
    # Filter the class averages MRCs file.
    # Here we use the filtered_particle_df to determine the unique classes (assuming ordering in MRCs corresponds to sorted unique classes).
    filter_class_averages_mrcs(class_mrcs_path, class_set, output_folder)

if __name__ == "__main__":
    main()

