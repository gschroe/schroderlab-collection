# Tools Collection

A collection of programs and scripts used for cryo-EM analysis of amyloid fibrils.
These tools support micrograph analysis, 2D/3D processing, polymorph clustering, model–map comparison, and initial volume generation.    

## Content
- [Sort Micrographs by Cross-β Signal](#sort-micrographs-by-cross-β-signal)
- [Crossover Length Determination](#crossover-length-determination)
- [Orient 2D Class Averages into the Same Polarity](#orient-2d-class-averages-into-the-same-polarity)
- [Initial Volume Generation](#initial-volume-generation)
  - Initial Volume from Distance-Matrix Ordering
  - Initial Volume by Stitching 2D Class Averages
- [Remove Fibril Crossings](#remove-fibril-crossings)
- [Compare Untwisted with Twisted 2D Class Averages](#compare-untwisted-with-twisted-2d-class-averages)
- [Micrograph Polymorph Assignment Viewer](#micrograph-polymorph-assignment-viewer)
- [Polymorph Separation Tools](#polymorph-separation-tools)
- [Density Sharpening](#density-sharpening)
- [Model-Map FSC](#model-map-fsc)

---

## Napari-based Tool for Manual Fibril Annotation
This tool is hosted in a separate repository:

https://github.com/sim-som/cryoem_fibril_annotator

### Features
- **Interactive Display**: Real-time visualization of cryo-EM micrographs with napari
- **Power Spectrum Support**: Synchronized display of micrographs and their corresponding power spectra
- **Real-time Filtering**: Butterworth lowpass filtering with Angstrom-based resolution control
- **Flexible Fibril Annotation**: Line and polyline tracing tools for straight **and** curved fibrils
- **Multi-layer Support**: Create separate annotation layers for different fibril types (Aβ42, Tau, α-synuclein, etc.)
- **Memory Efficient**: Handles large datasets (>GB) using Dask arrays and lazy loading
- **Annotation Persistence**: Save and load annotations with full metadata preservation
---

## **Sort Micrographs by Cross-β Signal**

`sort_micrographs_crossbeta/`  
Amyloid fibrils show a characteristic **4.7 Å cross-β reflection** in the Fourier spectrum.  
This tool computes the signal strength in each micrograph and sorts micrographs according to the visibility of fibrils.  
This helps to identify micrographs with high fibril content before particle extraction.

---

## **Crossover Length Determination**

`crossover_from_particle_locations/`  
Computes the **crossover distance** of amyloid fibrils from 2D class averages.  
The tool identifies the periodic modulation of fibrils and extracts the crossover spacing used to estimate the helical twist.

---

## **Orient 2D Class Averages into the Same Polarity**

`polarity/`  
Aligns all selected 2D class averages so that they point in the **same axial direction**.  
This step is useful before initial model generation as it reduced the search space for stitching the class averages.

---

## Initial Volume Generation

### **Initial Volume by Stitching 2D Class Averages**

`initial_volume_stitching/`  
Generates an initial 3D model by **placing 2D class averages sequentially along the fibril axis** and reconstructing a volume from these aligned images.  Useful when no reference is available.

### **Initial Volume from Distance-Matrix Ordering** 

`initial_volume_distance_matrix/`  
Computes a pairwise distance matrix between 2D class averages, orders them along the fibril trajectory, and constructs an initial volume using this ordering.  This is an alternative to stitching images by optimizing image overlap.

---

## **Remove Fibril Crossings** 

`remove_crossings/`  
Removes image segments that contain **crossings or overlaps** between neighboring fibrils.  
This is especially important when extracting long fibrils, where crossing regions can mislead helical reconstruction.

---

## Micrograph Polymorph Assignment Viewer

`class_assignment_gui/`  
A GUI to inspect where segments from **specific polymorphs** appear on individual micrographs.  Supports visual checks after clustering (ASHP/CHEP) to ensure correct polymorph assignment.

---

### **Compare Untwisted with Twisted 2D Class Averages** 

`compare_twisted_untwisted/`  
Compares the appearance of **untwisted fibril classes** against **twisted fibril classes** to identify whether a straight fibril variant corresponds to a known twisted polymorph.  

---

## Polymorph Separation Tools

These tools are hosted in separate repositories:

- **CHEP** – Clustering based on 2D class assignments  
    https://github.com/gschroe/chep
    
- **ASHP** – Clustering based on comparing 2D class averages, includes CHEP  
    https://github.com/JanusLammert/ASHP 

---
## Density Sharpening
 
**VISDEM** – Density sharpening with optional helical symmetry  
    https://github.com/gschroe/visdem


---
## Model-Map FSC

`model_map_fsc/`

- [`sl_pdb2mrc.c`](https://github.com/gschroe/schroderlab-collection/tree/main/model_map_fsc/pdb2mrc): Simulate cryo-EM map from pdb coordinates.
- [`sl_fsc.c`](https://github.com/gschroe/schroderlab-collection/tree/main/model_map_fsc/fsc): Calculate the FSC of two 3D maps in mrc format.

(plot fsc curves via [`plot_fsc_data.py`](https://github.com/gschroe/schroderlab-collection/blob/main/model_map_fsc/plot_fsc_data.py))
    Plots FSC curves produced by the above tools.
    

These programs allow quantitative model–map agreement evaluation, independent of any cryo-EM software package.

---
# License

schroderlab-collection: Toolcollection for the processing of Cryo-EM Datasets
Copyright (C) 2025  Gunnar Schröder
Copyright (C) 2025  Simon Sommerhage
Copyright (C) 2025  Janus Lammert

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.



