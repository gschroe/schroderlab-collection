##2D Class Selection and Filtering Workflow


This repository contains two tools:

1. **`sl_display.py`** — an interactive viewer for inspecting and selecting 2D class averages (`.mrcs`).
    
2. **`sl_filter_classes.py`** — a script to filter a particle `.star` file and the corresponding `.mrcs` file based on selected class IDs.
    

The tools allow rapid identification of good 2D classes and creation of a cleaned dataset for downstream refinement.

---

## 1. Inspecting and Selecting 2D Classes (`dxdisplay.py`)

### Launch the viewer

`python sl_display.py class_averages.mrcs [--cols 8]`

### Key features

- Click on images to toggle selection (highlighted by a red border).
    
- Each image shows a **green ID** (1-based index).
    
- **Brightness/Contrast**: right-click → _Brightness / Contrast_ (or press `B`).
    
- **Sort by entropy**: right-click → _Sort by Entropy_ to sort by "quality" (very rough but fast).
    
- **Zoom**: `Ctrl` + mouse wheel or pinch gesture.
    

### Saving and loading selections

- Save selected IDs: right-click → **Save Selection as List** → `good_classes.txt`
    
    - File format: one class ID per line (1-based).
        
- Load an existing list: right-click → **Load Selection**.
    
- Optional: save selected classes as a new `.mrcs` stack via **Save Selection**.
    

---

## 2. Filtering STAR and MRCS Files (`sl_filter_classes.py`)

This script creates a filtered particle dataset and a matching class‐average stack.

### Usage

`python sl_filter_classes.py \   -i particles.star \   -c class_averages.mrcs \   -s good_classes.txt \   -o filtered_output`

### What the script does

- Reads class IDs from `good_classes.txt`.
    
- Filters `particles.star` to keep only particles from the selected classes.
    
- Renumbers remaining `rlnClassNumber` values sequentially (1, 2, 3, …).
    
- Creates:
    
    - `filtered_output/filtered_particles.star`
        
    - `filtered_output/filtered_class_averages.mrcs` (only selected class images)
        

### Assumptions

- Class ID _k_ corresponds to the _k-th_ image in `class_averages.mrcs`.
    
- The ID list is 1-based (consistent with RELION and the viewer).
    

---

## 3. Typical Workflow

1. Run **`sl_display.py`** to inspect and select good classes.
    
2. Save selection to `good_classes.txt`.
    
3. Run **`sl_filter_classes.py`** to produce filtered `.star` and `.mrcs`.
    
4. Use `filtered_particles.star` for downstream refinement in RELION or cryoSPARC.
