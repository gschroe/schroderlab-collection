# 📂 MRC‑Stack Cross‑Correlation Analyzer  

A command‑line utility to:

1. **Load** one or more MRC files (3‑D image stacks).  
2. **Crop** each stack in Y and project the central *X‑percent* of pixels → a 1‑D projection per slice.  
3. **Cross‑correlate** the projections against a set of reference stacks (Pearson‑ or raw‑correlation).  
4. **Score** each projection against **all** reference stacks and report a fitness value + a percentage‑membership for every stack.  

The tool is useful for cryo‑EM / tomography workflows where you want to know which of several reference volumes a given experimental stack resembles the most.

---  

## Table of Contents  

- [Features](#features)  
- [Installation (Conda)](#installation-conda)  
- [Quick Start](#quick-start)  
- [Command‑Line Interface](#command-line-interface)  
- [How the Pipeline Works](#how-the-pipeline-works)  
- [Example Workflow](#example-workflow)  
- [Output Explained](#output-explained)

---  

## Features  

- **Pure‑Python** (no compiled extensions) – runs on any platform with a recent Python interpreter.  
- **Conda‑managed environment** – one command creates an isolated environment with exact package versions.  
- Handles an **arbitrary number of MRC files** (minimum two).  
- Flexible **Y‑cropping** (`n_pixels_crop`) and **X‑projection** (`x_percent`).  
- **Normalized** (Pearson) or **raw** cross‑correlation.  
- Returns **fitness scores** and **percentage‑membership** for each reference stack, plus a pretty‑printed summary.  

---  

## Installation (Conda)

### 1️⃣ Clone the repository  

```bash
git clone https://github.com/your‑username/mrc‑stack‑correlator.git
cd mrc‑stack‑correlator
```

### 2️⃣ Create the Conda environment  

All required packages are listed in `environment.yml`.  
Run the single command below:

```bash
conda env create -f environment.yml
```

> **What this does**  
> - Creates an environment called **CTU* (you can rename it in the YAML file).  
> - Installs Python 3.11, `numpy`, `scipy`, `mrcfile`, and `argparse` (built‑in).  

### 3️⃣ Activate the environment  

```bash
conda activate CTU
```

You are now ready to run the script.

---  

## Quick Start  

Assume you have three MRC stacks:

- `exp_stack1.mrc` – the *projection* stack you want to classify.  
- `ref_A.mrc` and `_B.mrc` – reference stacks.

```bash
python mrc_analyzer.py exp_stack1.mrc ref_A.mrc ref_B.mrc \
    -c 10            # crop 10 pixels from top & bottom (Y)  
    -x 60            # use the central 60 % of the X‑axis for projection  
    -n 5             # take the top‑5 correlations per reference for averaging  
    -norm            # use Pearson‑normalised correlation (default)
```

The script writes a new file `exp_stack1_filtered.mrc` (the projected stack) and prints a concise analysis to the console.

---  

## Command‑Line Interface  

```text
usage: mrc_analyzer.py [-h] [-c CROP] [-x X_PERCENT] [-n TOP_N] [--norm | --no-norm]
                       mrc_files [mrc_files ...]

Positional arguments:
  mrc_files             One or more MRC files. The **first** file is treated as the
                        projection stack; the remaining files are reference stacks.

Optional arguments:
  -h, --help            show this help message and exit
  -c, --crop CROP       Number of pixels to remove from the top and bottom of the
                        Y‑dimension (default: 0)
  -x, --x-percent X_PERCENT
                        Central X‑percentage used for projection (default: 50.0)
  -n, --top-n TOP_N     Number of highest correlations per reference that are
                        averaged to obtain a stack‑score (default: 5)
  --norm / --no-norm    Apply Pearson normalisation (default: --norm)
```

> **Note** – The script expects **at least two** MRC files. If you only provide one, it will raise an error.

---  

## How the Pipeline Works  

1. **`load_mrc_files`** – Reads each supplied MRC file into a `numpy.ndarray` (3‑D stack).  
2. **`process_image_stack`** –  
   - Crops `n_pixels_crop` rows from the top and bottom (Y).  
   - Takes the central `x_percent` of columns, sums them, and stores the result as a 1‑D projection per slice (`(z, height, 1)`).  
3. **`cross_correlation_analysis`** – For every projection slice, computes the maximal correlation against every slice of every reference stack (`calculate_max_correlation`).  
4. **`compare_reduced_to_all_stacks`** –  
   - Collects all correlation values per reference stack.  
   - Averages the *top‑n* values for each stack → a **stack‑average**.  
   - Normalises these averages to percentages (sum = 100 %).  
   - The highest average becomes the **fitness score** for that projection.  
5. **`print_stack_analysis`** – Nicely formats the percentages, dominant stacks, and fitness statistics.  

---  

## Example Workflow  

```bash
# -------------------------------------------------
# 1️⃣  Create the environment (once)
conda env create -f environment.yml
conda activate mrc-analysis
# -------------------------------------------------
# 2️⃣  Run the analysis on a set of 4 stacks
python mrc_analyzer.py exp.mrc ref1.mrc ref2.mrc ref3.mrc \
    -c 12 -x 55 -n 7 --norm
# -------------------------------------------------
# 3️⃣  Inspect the console output
# (sample excerpt)
#
# Loaded: exp.mrc, Shape: (120, 512, 512)
# Loaded: ref1.mrc, Shape: (80, 512, 512)
# …
# Stack processed: (120, 512, 512) -> (120, 488, 1)
#
# Berechne Kreuzkorrelation: 120 Projektionen x 3 Referenzen
# Verarbeitet: 10/120 Projektionen
# …
#
# === STACK ZUGEHÖRIGKEITS-ANALYSE ===
# Projektions-Stack: exp.mrc
# Vergleichs-Stacks: Stack_0(ref1.mrc), Stack_1(ref2.mrc), Stack_2(ref3.mrc)
#
# Durchschnittliche Zugehörigkeit über alle Projektionen:
#   Stack_0(ref1.mrc): 42.3%
#   Stack_1(ref2.mrc): 35.7%
#   Stack_2(ref3.mrc): 22.0%
#
# Bilder mit stärkster Zugehörigkeit (>70%):
#   Bild 3: 78.5% → Stack_0(ref1.mrc) (Fitness: 0.912)
#   Bild 57: 71.2% → Stack_1(ref2.mrc) (Fitness: 0.887)
#   …
# -------------------------------------------------
```

The script also writes the **projected stack** (`exp_filtered.mrc`) that you can visualise in any MRC viewer (e.g., `e2display.py`, `IMOD`, `ChimeraX`).

---  

## Output Explained  

| Variable | Shape | Meaning |
|----------|-------|---------|
| `projection_stack` | `(z, h, 1)` | Cropped & X‑projected version of the first MRC file. |
| `correlation_matrix` | `(n_projections, n_references)` | Max correlation of each projection slice against each reference slice. |
| `fitness_values` | `(n_projections,)` | Highest average correlation per projection (used for ranking). |
| `stack_percentages` | `(n_projections, n_stacks)` | Normalised percentages that sum to 100 % for each projection. |

The console summary (`print_stack_analysis`) gives you:

- **Overall stack composition** (mean percentages).  
- **Dominant stack per projection** and how many projections exceed a 70 % threshold.  
- **Fitness range** across all projections.
