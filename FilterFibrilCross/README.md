# Relion‑Starfile Particle Filter  

A small command‑line utility to **filter particles in a Relion 3.0+ `.star` file** based on their proximity to particles that belong to **different helical tube IDs**.  
It is useful for cleaning up helical reconstructions where overlapping tubes can corrupt downstream processing.

---

## Table of Contents  

- [Features](#features)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [Command‑Line Interface](#command-line-interface)  
- [How It Works](#how-it-works)  
- [Dependencies](#dependencies)  

---

## Features  

- **Automatic distance threshold**: if you don’t provide one, the script falls back to the box size stored in the optics block (or 256 px by default).  
- **Micrograph‑aware filtering** – particles are only compared within the same micrograph, preserving cross‑micrograph independence.  
- **Helical‑tube aware** – removes a particle when it is *too close* (less than `threshold/2`) to another particle that belongs to a **different** `_rlnHelicalTubeID`.  
- **Preserves all other STAR blocks** (optics, class, etc.) while only touching the `particles` block.  
- **Pure Python** – no compiled extensions, easy to install on any system with Python 3.7+.  

---

## Installation  

### 1. Clone the repository  

```bash
git clone https://github.com/your-username/relion-starfile-filter.git
cd relion-starfile-filter
```
### 2. Install required Python packages and create conda env

```bash
conda create -n relion-filter python=3.11 numpy pandas -c conda-forge -y
```
To activate conda:

```bash
conda activate relion-filter
```

### 3. Make the script executable (Unix‑like)

```bash
chmod +x filter_starfile.py
```

You can now run it directly with `./filter_starfile.py` or via `python filter_starfile.py`. (If the conda env is activated)

---

## Quick Start  

```bash
# Basic usage – let the script infer the distance threshold from the box size
python filter_starfile.py my_particles.star

# Specify a custom distance threshold (in pixels)
python filter_starfile.py my_particles.star -d 120

# Write output to a custom filename
python filter_starfile.py my_particles.star -o cleaned.star
```

The script will generate a new STAR file named `<input>_filtered.star` (or the name you supplied) and print a short summary:

```
Reading starfile: my_particles.star
Using box size as distance threshold: 256.0
Processing 42 micrographs...
Filtering particles...
Writing filtered starfile to: my_particles_filtered.star

Filtering complete:
  - Particles removed: 1234
  - Particles kept: 9876
  - Output written to: my_particles_filtered.star
```

---

## Command‑Line Interface  

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `starfile` | – | `str` | – | **Required** – Path to the input Relion `.star` file. |
| `-d, --distance` | `-d` | `float` | `None` | Distance threshold (in pixel units). If omitted, the script uses the box size from the optics block. |
| `-o, --output` | `-o` | `str` | `None` | Output file name. If omitted, `<input>_filtered.star` is created in the same folder. |

Run `python filter_starfile.py -h` for the built‑in help.

---

## How It Works  

1. **Parse the STAR file** – The script reads all data blocks (`optics`, `particles`, …) into pandas DataFrames while preserving the original header order.  
2. **Determine a distance threshold** –  
   - If you supplied `-d`, that value is used.  
   - Otherwise the script extracts `_rlnImageSize` from the `optics` block (or falls back to `256`).  
3. **Group particles by micrograph** (`_rlnMicrographName`).  
4. **For each micrograph**:  
   - Compute the full pair‑wise Euclidean distance matrix of X/Y coordinates.  
   - For each particle, check whether any *different* helical tube (`_rlnHelicalTubeID`) lies closer than `threshold/2`.  
   - Flag such particles for removal.  
5. **Write a new STAR file** – All blocks are written back in the original order; only the `particles` block is filtered.  

The algorithm is **O(N²)** per micrograph, which is acceptable for typical cryo‑EM datasets (hundreds to a few thousand particles per micrograph).

---

## Dependencies  

| Package | Minimum version |
|---------|----------------|
| Python | 3.7+ |
| numpy | 1.18+ |
| pandas | 1.0+ |

All dependencies are pure‑Python and install via `pip`.
