# Cross-Beta Micrograph Sorting Tool

A command-line tool for identifying and ranking cryo-EM micrographs by cross-beta diffraction signal strength. This tool analyzes CTFFIND4 output from RELION workflows to detect the characteristic 4.7 Å cross-beta reflection found in amyloid fibrils.

## Overview

The cross-beta structure is characteristic of amyloid fibrils and produces a strong diffraction peak at approximately 1/4.7 Å⁻¹ in the power spectrum. This tool:

1. Reads CTFFIND4 output from RELION
2. Analyzes the radial power spectrum in the cross-beta frequency region
3. Calculates a cross-beta score combining peak height and area under curve
4. Corrects for CTF effects at the cross-beta frequency
5. Outputs a sorted STAR file for downstream processing

## Requirements

```bash
pip install numpy pandas starfile scikit-learn tqdm mrcfile matplotlib matplotlib-scalebar
```

Note: `matplotlib` and `matplotlib-scalebar` are only required if you want to generate HTML reports with `--report`.

## Usage

### Basic Usage

```bash
./sort_mics_by_cross_beta.py -i CtfFind/job005/micrographs_ctf.star -o micrographs_sorted.star
```

### Full Example

```bash
./sort_mics_by_cross_beta.py \
  -i /path/to/relion/CtfFind/job005/micrographs_ctf.star \
  -o /path/to/output/micrographs_crossbeta_sorted.star \
  --ctf-max-res 5.0 \
  --verbose
```

### Process Only First 100 Micrographs (for testing and optimizing parameters)

```bash
./sort_mics_by_cross_beta.py \
  -i micrographs_ctf.star \
  -o sorted.star \
  -n 100 \
  --verbose
```

### Generate HTML Report with Plots

```bash
./sort_mics_by_cross_beta.py \
  -i micrographs_ctf.star \
  -o sorted.star \
  --report cross_beta_report.html \
  --n-report 5 \
  --verbose
```

This will create a scrollable HTML report showing the top 5 and bottom 5 micrographs with:
- Micrograph images (with sigma contrast)
- 2D power spectra
- Radial power spectra with cross-beta region highlighted

## Command Line Arguments

### Required Arguments

- `-i, --input`: Input STAR file from CTFFIND4 (e.g., `micrographs_ctf.star`)
- `-o, --output`: Output STAR file with micrographs sorted by cross-beta score

### Optional Arguments

#### Directory Settings
- `-r, --relion-dir`: RELION working directory (default: auto-detected from input path)

#### Cross-Beta Detection Parameters
- `-k, --cross-beta-k`: Cross-beta spatial frequency in 1/Å (default: 0.2128 = 1/4.7)
- `--rel-width`: Relative width of cross-beta frequency window (default: 0.05 = 5%)
- `--weight`: Weight factor for max vs AUC in scoring (default: 0.7)
  - Score = `weight × max_peak + (1-weight) × area_under_curve`

#### CTF Filtering Parameters
- `--ctf-max-res`: Maximum CTF resolution threshold in Å (default: 6.0)
  - Micrographs with worse CTF resolution are excluded
- `--ctf-stability`: Minimum CTF² value for stable correction (default: 0.1)
  - Micrographs with CTF² below this threshold are marked as unreliable (NaN score)

#### Processing Options
- `-n, --n-mics`: Number of micrographs to process (default: all micrographs)
  - Useful for testing on a subset before processing entire dataset
- `--verbose`: Print detailed progress information

#### HTML Report Options
- `--report`: Path to output HTML report file (optional)
  - If specified, generates an interactive HTML report with plots
- `--n-report`: Number of top/bottom micrographs to show in report (default: 3)
  - Only used when `--report` is specified

## Algorithm Details

### Cross-Beta Score Calculation

The cross-beta score combines two metrics:

1. **Peak Height**: Maximum value in the cross-beta frequency window
2. **Area Under Curve**: Integrated signal in the cross-beta region

```
score = weight × max_peak + (1 - weight) × AUC
```

Default weight of 0.7 emphasizes peak height while still considering integrated signal.

### CTF Correction

The measured power spectrum is affected by the CTF:

```
P(k) = |F(k)|² × CTF(k)² × E(k)² + N(k)
```

Where:
- F(k): True structure factor
- CTF(k): Contrast transfer function
- E(k): Envelope function
- N(k): Noise

The tool calculates the radially (and azimuthally) averaged CTF² at the cross-beta frequency and corrects the score:

```
corrected_score = raw_score / CTF²
```

Micrographs where CTF² is too low (below `--ctf-stability`) are marked as unreliable (NaN).

### Frequency Window

The cross-beta signal is searched in a frequency window around the central frequency:

```
k_min = (1 - rel_width/2) × k_crossbeta
k_max = (1 + rel_width/2) × k_crossbeta
```

Default: 0.2021 - 0.2234 Å⁻¹ (4.48 - 4.95 Å)

## Output

### STAR File Output

The output STAR file contains all original columns plus:

- `CrossBetaScore`: CTF-corrected cross-beta signal score
  - Higher scores indicate stronger cross-beta signal
  - NaN indicates unreliable measurements (CTF² too low)

Micrographs are sorted in descending order by `CrossBetaScore` (NaN values last).

### HTML Report Output (Optional)

When `--report` is specified, an HTML report is generated with:

**For each shown micrograph:**
1. **Micrograph image**: Displayed with 3σ contrast adjustment and scale bar
2. **2D Power Spectrum**: The CTFFIND4 power spectrum image
3. **Radial Power Spectrum**: 1D plot showing the cross-beta frequency region highlighted in green
4. **Metadata**: Cross-beta score, CTF resolution, and defocus values

**Report structure:**
- Summary section with analysis timestamp and total count
- Top N micrographs (highest cross-beta scores)
- Bottom N micrographs (lowest cross-beta scores)

The report is self-contained (all images embedded as base64) and can be opened in any web browser.

### Summary Statistics

The tool prints summary statistics including:
- Number of micrographs processed
- Number with valid scores
- Number marked unreliable
- Mean, median, std dev, min, max of scores

## Example Output

```
Summary:
  Total micrographs processed: 96
  Valid cross-beta scores: 85
  Unreliable (CTF² too low): 11

Cross-beta score statistics:
  Mean: 0.125678
  Median: 0.089234
  Std dev: 0.156789
  Min: 0.012345
  Max: 12.527604

Output written to: micrographs_crossbeta_sorted.star
Done!
```

## Advanced Examples

### Custom Cross-Beta Frequency

For a different reflection (e.g., 10 Å):

```bash
./sort_mics_by_cross_beta.py -i input.star -o output.star -k 0.1
```

### Stricter CTF Requirements

```bash
./sort_mics_by_cross_beta.py \
  -i input.star \
  -o output.star \
  --ctf-max-res 4.0 \
  --ctf-stability 0.2
```

### Wider Frequency Window

To search a broader region (10% width):

```bash
./sort_mics_by_cross_beta.py -i input.star -o output.star --rel-width 0.10
```

### Emphasize Area Under Curve

To weight AUC more heavily than peak height:

```bash
./sort_mics_by_cross_beta.py -i input.star -o output.star --weight 0.3
```

### Complete Analysis with Report

Process all micrographs and generate a comprehensive report:

```bash
./sort_mics_by_cross_beta.py \
  -i CtfFind/job005/micrographs_ctf.star \
  -o micrographs_sorted.star \
  --report analysis_report.html \
  --n-report 10 \
  --ctf-max-res 5.0 \
  --verbose
```

## Troubleshooting

### "Input file not found"
Ensure the path to the STAR file is correct and the file exists.

### "Radial PS file not found"
The tool requires CTFFIND4 output files (`*_avrot.txt`). Ensure CTFFIND4 was run successfully and the files are in the expected location relative to the RELION working directory.

### "No micrographs passed CTF resolution filter"
All micrographs were filtered out due to poor CTF fit. Try increasing `--ctf-max-res` or check your data quality.

### RELION Directory Auto-Detection Fails
Use `-r, --relion-dir` to explicitly specify the RELION working directory.

## File Requirements

The tool expects the following RELION/CTFFIND4 file structure:

```
relion_working_directory/
├── CtfFind/
│   └── job005/
│       ├── micrographs_ctf.star          # Input STAR file
│       └── movies/
│           ├── *_avrot.txt               # CTFFIND4 radial PS files
│           └── *.ctf                     # CTFFIND4 2D PS images
└── MotionCorr/
    └── job004/
        └── movies/
            └── *.mrc                      # Motion-corrected micrographs
```
If motioncorrected micrographs were imported to RELION the tool should work too, but only if the mcirgoraphs (or links to them) reside in a folder inside the relion directory:
```
relion_working_directory/
└── Import/
    └── job004/
        └── micrographs.star              # Star file listing imported micrographs
└── Folder_containing_micrographs/
```

<!---
## Citation

If you use this tool in your research, please cite the original cross-beta detection methodology and the tools it depends on:

- RELION: Scheres, S.H.W. (2012). J. Struct. Biol.
- CTFFIND4: Rohou, A. & Grigorieff, N. (2015). J. Struct. Biol.
- starfile: Burnley, T. et al. (2017). Acta Cryst. D

## License

This tool is provided as-is for research purposes.
--->