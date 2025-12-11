#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Filter Relion starfile based on particle proximity with different helical tube IDs')
    parser.add_argument('starfile', type=str, help='Input Relion starfile')
    parser.add_argument('-d', '--distance', type=float, help='Distance threshold for filtering particles (default: boxsize)', default=None)
    parser.add_argument('-o', '--output', type=str, help='Output starfile (default: [input]_filtered.star)', default=None)
    return parser.parse_args()

def read_starfile(starfile):
    """Read Relion 3.0+ starfile format"""
    with open(starfile, 'r') as f:
        content = f.readlines()
    
    data_blocks = {}
    current_block = None
    in_loop = False
    headers = []
    data = []
    header_indices = {}
    
    i = 0
    while i < len(content):
        line = content[i].strip()
        i += 1
        
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('data_'):
            if current_block and headers:
                data_blocks[current_block] = {'headers': headers, 'df': pd.DataFrame(data, columns=[h.split()[0] for h in headers])}
            
            current_block = line[5:]
            in_loop = False
            headers = []
            data = []
            header_indices = {}
        
        elif line == 'loop_':
            in_loop = True
            
        elif in_loop and line.startswith('_rln'):
            # Extract the header name and its index
            parts = line.split()
            header_name = parts[0]
            header_index = int(parts[1][1:]) - 1  # Convert from 1-based to 0-based indexing
            headers.append(line)
            header_indices[header_name] = header_index
            
        elif in_loop and current_block and not line.startswith('_'):
            # Split the line by whitespace
            fields = line.split()
            # Handle fields with spaces (enclosed in quotes)
            processed_fields = []
            j = 0
            while j < len(fields):
                field = fields[j]
                if field.startswith('"') and not field.endswith('"'):
                    # Field starts with a quote but doesn't end with one, so it contains spaces
                    combined_field = field
                    j += 1
                    while j < len(fields) and not fields[j].endswith('"'):
                        combined_field += ' ' + fields[j]
                        j += 1
                    if j < len(fields):
                        combined_field += ' ' + fields[j]
                    processed_fields.append(combined_field.strip('"'))
                else:
                    processed_fields.append(field)
                j += 1
            
            # Ensure we have the right number of fields
            if len(processed_fields) >= len(headers):
                data.append(processed_fields[:len(headers)])
    
    # Add the last data block if it exists
    if current_block and headers:
        data_blocks[current_block] = {'headers': headers, 'df': pd.DataFrame(data, columns=[h.split()[0] for h in headers])}
    
    return data_blocks

def write_starfile(data_blocks, output_file):
    """Write data blocks to a starfile"""
    with open(output_file, 'w') as f:
        f.write("# version 30001\n")
        
        for block_name, block_data in data_blocks.items():
            f.write(f"\ndata_{block_name}\n")
            f.write("loop_\n")
            
            # Write headers with their indices
            for i, header in enumerate(block_data['headers']):
                f.write(f"{header}\n")
            
            # Write data rows
            for _, row in block_data['df'].iterrows():
                f.write(' '.join(map(str, row.values)) + '\n')

def get_boxsize(data_blocks):
    """Extract box size from starfile if available"""
    if 'optics' in data_blocks:
        try:
            return float(data_blocks['optics']['df']['_rlnImageSize'].iloc[0])
        except:
            pass
    
    # Default if box size can't be determined
    return 256.0

def calculate_distances(coords):
    """Calculate pairwise distances between all particles"""
    return np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))

def filter_particles(data_blocks, distance_threshold):
    """Filter particles based on proximity to particles with different helical tube IDs"""
    particles_df = data_blocks['particles']['df']
    
    # Check for helical tube ID column
    fibril_col = '_rlnHelicalTubeID'
    if fibril_col not in particles_df.columns:
        raise ValueError("Helical tube ID column (_rlnHelicalTubeID) not found in starfile")
    
    # Coordinate columns
    x_col = '_rlnCoordinateX'
    y_col = '_rlnCoordinateY'
    
    if x_col not in particles_df.columns or y_col not in particles_df.columns:
        raise ValueError("Coordinate columns not found in starfile")
    
    # Micrograph name column
    mic_col = '_rlnMicrographName'
    if mic_col not in particles_df.columns:
        raise ValueError("Micrograph column not found in starfile")
    
    # Group particles by micrograph
    micrographs = particles_df[mic_col].unique()
    particles_to_remove = set()
    total_particles = len(particles_df)
    
    print(f"Processing {len(micrographs)} micrographs...")
    
    for mic in micrographs:
        mic_particles = particles_df[particles_df[mic_col] == mic]
        mic_indices = mic_particles.index
        
        # Skip if only one particle in this micrograph
        if len(mic_particles) <= 1:
            continue
        
        # Get coordinates and fibril IDs
        coords = mic_particles[[x_col, y_col]].astype(float).values
        fibril_ids = mic_particles[fibril_col].astype(str).values
        
        # Calculate pairwise distances
        distances = calculate_distances(coords)
        
        # Find particles to remove
        for i in range(len(mic_particles)):
            local_idx = mic_indices[i]
            for j in range(len(mic_particles)):
                if i == j:
                    continue
                    
                # Check if particles have different fibril IDs and are close
                if fibril_ids[i] != fibril_ids[j] and distances[i, j] < distance_threshold / 2:
                    particles_to_remove.add(local_idx)
                    break  # If one close particle with different ID found, move to next particle
    
    # Remove flagged particles
    filtered_indices = [idx for idx in particles_df.index if idx not in particles_to_remove]
    filtered_particles_df = particles_df.loc[filtered_indices]
    
    # Update the data blocks
    result_blocks = {}
    for block_name, block_data in data_blocks.items():
        if block_name == 'particles':
            result_blocks[block_name] = {
                'headers': block_data['headers'],
                'df': filtered_particles_df
            }
        else:
            result_blocks[block_name] = block_data
    
    particles_removed = total_particles - len(filtered_particles_df)
    return result_blocks, particles_removed, len(filtered_particles_df)

def main():
    args = parse_args()
    
    print(f"Reading starfile: {args.starfile}")
    data_blocks = read_starfile(args.starfile)
    
    # Determine distance threshold
    distance_threshold = args.distance
    if distance_threshold is None:
        distance_threshold = get_boxsize(data_blocks)
        print(f"Using box size as distance threshold: {distance_threshold}")
    else:
        print(f"Using specified distance threshold: {distance_threshold}")
    
    # Filter particles
    print(f"Filtering particles...")
    filtered_data, particles_removed, particles_kept = filter_particles(data_blocks, distance_threshold)
    
    # Write output
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.starfile)
        output_file = f"{base}_filtered{ext}"
    
    print(f"Writing filtered starfile to: {output_file}")
    write_starfile(filtered_data, output_file)
    
    # Print statistics
    print(f"\nFiltering complete:")
    print(f"  - Particles removed: {particles_removed}")
    print(f"  - Particles kept: {particles_kept}")
    print(f"  - Output written to: {output_file}")

if __name__ == "__main__":
    main()
