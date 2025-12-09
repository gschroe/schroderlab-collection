#Consistent Polarity for 2D Helical Class Averages



`sl_polarity.py` reads a **particle STAR file** and the corresponding **2D class averages** (`.mrcs`), estimates the **relative polarity** of 2D classes using filament information and Euler angles, and flips class-average images left–right so that all classes point in a **consistent direction**.

This is useful for helical samples (e.g. amyloid fibrils) where some 2D classes appear “left” and others “right”, but you want a uniform polarity before further analysis (e.g. stitching) or visualization.

---

## Features

- Uses filament connectivity and `rlnAnglePsi` to compute a **class–class polarity matrix**.
    
- Requires a minimum number of shared filaments between classes (default `min_shared=3`) to trust polarity.
    
- Optimizes a global ±1 flipping pattern over all classes (Ising-like objective).
    
- Outputs a new `.mrcs` file with **selected classes flipped horizontally** (`np.fliplr`), preserving voxel size.
    

---

## Requirements

The particle STAR file must contain at least:

- `rlnHelicalTubeID`
    
- `rlnMicrographName`
    
- `rlnHelicalTrackLengthAngst`
    
- `rlnClassNumber` (1-based)
    
- `rlnAnglePsi` (degrees)
    
- `rlnMaxValueProbDistribution` (used as weight)
    
- columns needed by `starfile` to parse the `particles` table
    

The class averages `.mrcs` file is assumed to be ordered such that **class ID `k` corresponds to slice `k`** (1-based) in the stack.

---

## Usage

`python sl_polarity.py \   -i particles.star \   -c class_averages.mrcs \   -o flipped_class_averages.mrcs`

Arguments:

- `-i, --input-star`  
    Particle STAR file (e.g. `run_data.star`) with helical and angular metadata.
    
- `-c, --class-mrcs`  
    Original class averages stack (`.mrcs`).
    
- `-o, --output-mrcs`  
    Output `.mrcs` file with polarity-corrected class averages.
    
