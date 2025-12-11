#!/usr/bin/env python3
"""
Cross-beta signal detection and ranking for cryo-EM micrographs.

This script analyzes CTFFIND4 output from RELION to identify micrographs with
strong cross-beta diffraction signals (typically at 1/4.7 Å⁻¹). It calculates
a cross-beta score for each micrograph, corrects for CTF effects, and outputs
a sorted STAR file.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import starfile
from sklearn.metrics import auc
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sort cryo-EM micrographs by cross-beta signal strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (processes all micrographs)
  %(prog)s -i CtfFind/job005/micrographs_ctf.star -o output_sorted.star

  # Process only first 100 micrographs for testing
  %(prog)s -i micrographs_ctf.star -o sorted.star -n 100

  # Generate HTML report with top/bottom micrographs
  %(prog)s -i micrographs_ctf.star -o sorted.star --report report.html --n-report 5

  # With custom cross-beta frequency and CTF threshold
  %(prog)s -i micrographs_ctf.star -o sorted.star -k 0.21 --ctf-max-res 5.0
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input STAR file from CTFFIND4 (e.g., micrographs_ctf.star)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output STAR file with micrographs sorted by cross-beta score'
    )

    parser.add_argument(
        '-r', '--relion-dir',
        type=str,
        default=None,
        help='RELION working directory (default: parent of input file)'
    )

    parser.add_argument(
        '-k', '--cross-beta-k',
        type=float,
        default=1/4.7,
        help='Cross-beta spatial frequency in 1/Å (default: 1/4.7 = 0.2128)'
    )

    parser.add_argument(
        '--rel-width',
        type=float,
        default=0.05,
        help='Relative width of cross-beta frequency window (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '--ctf-max-res',
        type=float,
        default=6.0,
        help='Maximum CTF resolution threshold in Å (default: 6.0)'
    )

    parser.add_argument(
        '--ctf-stability',
        type=float,
        default=0.1,
        help='Minimum CTF² value for stable correction (default: 0.1)'
    )

    parser.add_argument(
        '--weight',
        type=float,
        default=0.7,
        help='Weight factor for max vs AUC in scoring (default: 0.7)'
    )

    parser.add_argument(
        '-n', '--n-mics',
        type=int,
        default=None,
        help='Number of micrographs to process (default: all micrographs)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Generate HTML report with micrograph plots (provide output path)'
    )

    parser.add_argument(
        '--n-report',
        type=int,
        default=3,
        help='Number of top/bottom micrographs to show in report (default: 3)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    return parser.parse_args()


def read_mrc_file(mrc_fpath):
    """Read MRC file and return as numpy array."""
    import mrcfile
    with mrcfile.open(mrc_fpath) as f:
        arr = f.data
    return arr.astype("float32")


def parse_ctffind4_radial_ps(ctf_image_path, relion_wdir):
    """
    Parse CTFFIND4 radial power spectrum data.

    Parameters
    ----------
    ctf_image_path : str or Path
        Path to CTF image from STAR file
    relion_wdir : Path
        RELION working directory

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: SpatialFrequency, RotAvg_noAsti, RotAvg,
        CTF_fit, CC, 2Sigma_noise
    """
    ctf_image = Path(ctf_image_path)
    rot_ps_file = f"{ctf_image.stem}_avrot.txt"
    rot_ps_fpath = relion_wdir / ctf_image.parent / Path(rot_ps_file)

    if not rot_ps_fpath.exists():
        raise FileNotFoundError(f"Radial PS file not found: {rot_ps_fpath}")

    rows = []
    with open(rot_ps_fpath, mode="r") as f:
        for row in f:
            if row[0] == "#":
                continue
            row = row.split()
            row = np.array(row).astype(float)
            rows.append(row)

    ps_data_dict = {
        "SpatialFrequency": rows[0],
        "RotAvg_noAsti": rows[1],
        "RotAvg": rows[2],
        "CTF_fit": rows[3],
        "CC": rows[4],
        "2Sigma_noise": rows[5]
    }

    return pd.DataFrame(ps_data_dict)


def area_under_curve(x, y, xmin, xmax):
    """
    Calculate area under curve in a specific interval.

    Parameters
    ----------
    x : np.ndarray
        X values (spatial frequency)
    y : np.ndarray
        Y values (power spectrum)
    xmin : float
        Lower bound of interval
    xmax : float
        Upper bound of interval

    Returns
    -------
    float
        Area under curve in the specified interval
    """
    dx = x[1] - x[0]

    min_idx = np.where(np.isclose(x, xmin, atol=dx/2))
    assert len(min_idx) == 1
    min_idx = min_idx[0][0]

    max_idx = np.where(np.isclose(x, xmax, atol=dx/2))
    assert len(max_idx) == 1
    max_idx = max_idx[0][0]

    x_interval = x[min_idx:max_idx + 1]
    y_interval = y[min_idx:max_idx + 1]

    return auc(x_interval, y_interval)


def measure_cross_beta_signal(k, raps, k_min, k_max, weight=0.7):
    """
    Measure cross-beta signal strength.

    Combines maximum value and area under curve in the cross-beta region.

    Parameters
    ----------
    k : np.ndarray
        Spatial frequency array
    raps : np.ndarray
        Rotationally averaged power spectrum
    k_min : float
        Lower bound of cross-beta frequency window
    k_max : float
        Upper bound of cross-beta frequency window
    weight : float
        Weight factor: score = weight * max + (1-weight) * auc

    Returns
    -------
    float
        Cross-beta signal score
    """
    #TODO Implement a more sophisticated approach using peak finding (e.g. discard peak if it's at the edge of the cross-beta intervall)

    auc_val = area_under_curve(k, raps, k_min, k_max)

    min_idx = np.argmin(np.abs(k - k_min))
    max_idx = np.argmin(np.abs(k - k_max))
    max_val = np.max(raps[min_idx:max_idx])

    return weight * max_val + (1 - weight) * auc_val


def calculate_ctf2_at_frequency(
    k, defocus1, defocus2, astig_angle,
    voltage, cs, amp_contrast
):
    """
    Calculate azimuthally averaged CTF² at a given spatial frequency.

    Parameters
    ----------
    k : float
        Spatial frequency in 1/Å
    defocus1 : array-like
        Defocus U in Å (underfocus positive)
    defocus2 : array-like
        Defocus V in Å
    astig_angle : array-like
        Astigmatism angle in degrees
    voltage : float
        Acceleration voltage in kV
    cs : float
        Spherical aberration in mm
    amp_contrast : float
        Amplitude contrast (typically 0.07-0.10)

    Returns
    -------
    np.ndarray or float
        CTF² values between 0 and 1
    """
    # Ensure arrays
    defocus1 = np.atleast_1d(np.asarray(defocus1))
    defocus2 = np.atleast_1d(np.asarray(defocus2))
    astig_angle = np.atleast_1d(np.asarray(astig_angle))

    # Convert units
    cs_angstrom = cs * 1e7  # mm to Å
    astig_angle_rad = np.radians(astig_angle)

    # Electron wavelength in Å
    voltage_ev = voltage * 1e3
    wavelength = 12.2643 / np.sqrt(voltage_ev + 0.97845e-6 * voltage_ev**2)

    # Azimuthal angles: shape (1, n_angles) for broadcasting
    n_angles = 360
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)[np.newaxis, :]

    # Defocus parameters: shape (n_micrographs, 1) for broadcasting
    defocus_mean = (0.5 * (defocus1 + defocus2))[:, np.newaxis]
    defocus_diff = (0.5 * (defocus1 - defocus2))[:, np.newaxis]
    astig_angle_rad = astig_angle_rad[:, np.newaxis]

    # Defocus as function of azimuthal angle: shape (n_micrographs, n_angles)
    defocus = defocus_mean + defocus_diff * np.cos(2 * (theta - astig_angle_rad))

    # Phase shift χ(k)
    k2 = k ** 2
    chi = np.pi * wavelength * k2 * (defocus - 0.5 * cs_angstrom * wavelength**2 * k2)

    # CTF with amplitude contrast
    w1 = np.sqrt(1 - amp_contrast**2)
    ctf = -w1 * np.sin(chi) - amp_contrast * np.cos(chi)

    # Azimuthally averaged CTF²: average over theta
    ctf2_avg = np.mean(ctf ** 2, axis=1)

    # Return scalar if input was scalar
    if ctf2_avg.size == 1:
        return ctf2_avg[0]
    return ctf2_avg


def generate_html_report(
    micctf_df_sorted, relion_wdir, k_min, k_max,
    n_top=3, n_bottom=3, output_path="report.html", angpix=0.82
):
    """
    Generate an HTML report with micrograph plots.

    Parameters
    ----------
    micctf_df_sorted : pd.DataFrame
        Sorted dataframe with cross-beta scores
    relion_wdir : Path
        RELION working directory
    k : np.ndarray
        Spatial frequency array
    k_min : float
        Lower bound of cross-beta window
    k_max : float
        Upper bound of cross-beta window
    n_top : int
        Number of top micrographs to show
    n_bottom : int
        Number of bottom micrographs to show
    output_path : str
        Path to output HTML file
    angpix : float
        Pixel size in angstrom
    """
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib_scalebar.scalebar import ScaleBar

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cross-Beta Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 40px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
            }
            .micrograph-section {
                background-color: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .micrograph-header {
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 10px;
                color: #2196F3;
            }
            .metadata {
                background-color: #f9f9f9;
                padding: 10px;
                border-left: 4px solid #4CAF50;
                margin: 10px 0;
                font-family: monospace;
                font-size: 0.9em;
            }
            .plot-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }
            .plot {
                flex: 1;
                min-width: 400px;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .summary {
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Cross-Beta Micrograph Analysis Report</h1>
        <div class="summary">
            <p><strong>Generated:</strong> """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p><strong>Total micrographs analyzed:</strong> """ + str(len(micctf_df_sorted)) + """</p>
        </div>
    """

    def plot_to_base64(fig):
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    def create_micrograph_plots(mic_row):
        """Create plots for a single micrograph."""
        plots = {}

        # Read micrograph
        mic_fpath = relion_wdir / Path(mic_row["rlnMicrographName"])
        if not mic_fpath.exists():
            return None

        mic_arr = read_mrc_file(mic_fpath)
        mu = mic_arr.mean()
        sigma = mic_arr.std()

        # Micrograph plot with sigma contrast
        fig, ax = plt.subplots(figsize=(10, 10))
        vmin = mu - 3 * sigma / 2
        vmax = mu + 3 * sigma / 2
        ax.imshow(mic_arr, cmap="gray", vmin=vmin, vmax=vmax)
        scalebar = ScaleBar(angpix * 0.1, "nm", location="lower right", color="white")
        ax.add_artist(scalebar)
        ax.set_title(f"Micrograph (3σ contrast)")
        ax.axis('off')
        plots['micrograph'] = plot_to_base64(fig)

        # Power spectrum plot
        ctfimage = Path(mic_row["rlnCtfImage"])
        ctf_image_name = ctfimage.name.replace(".ctf:mrc", ".ctf")
        ctfimage = ctfimage.parent / Path(ctf_image_name)
        ps_fpath = relion_wdir / Path(ctfimage)

        if ps_fpath.exists():
            ps_arr = read_mrc_file(ps_fpath)
            if ps_arr.ndim == 3:
                ps_arr = ps_arr[0, :, :]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(ps_arr, cmap="gray")
            ax.set_title("2D Power Spectrum")
            ax.axis('off')
            plots['power_spectrum'] = plot_to_base64(fig)

        # Radial power spectrum plot
        try:
            ps_df = parse_ctffind4_radial_ps(mic_row["rlnCtfImage"], relion_wdir)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ps_df["SpatialFrequency"], ps_df["RotAvg"],
                   label="Rotational Average", linewidth=1.5)
            ax.axvspan(k_min, k_max, alpha=0.25, color='green',
                      label=f'Cross-beta region ({1/k_min:.2f}-{1/k_max:.2f} Å)')
            ax.set_xlabel("Spatial Frequency (1/Å)", fontsize=12)
            ax.set_ylabel("Power", fontsize=12)
            ax.set_title("Radial Power Spectrum", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plots['radial_ps'] = plot_to_base64(fig)
        except:
            pass

        return plots

    # Process top micrographs
    html_content += "<h2>Top " + str(n_top) + " Micrographs (Highest Cross-Beta Score)</h2>\n"

    valid_sorted = micctf_df_sorted[~micctf_df_sorted["CrossBetaScore"].isna()]

    for i in range(min(n_top, len(valid_sorted))):
        idx = valid_sorted.index[i]
        mic_row = valid_sorted.loc[idx]

        html_content += '<div class="micrograph-section">\n'
        html_content += f'<div class="micrograph-header">Rank #{i+1}: {Path(mic_row["rlnMicrographName"]).name}</div>\n'
        html_content += '<div class="metadata">\n'
        html_content += f'Cross-Beta Score: {mic_row["CrossBetaScore"]:.6f}<br>\n'
        html_content += f'CTF Max Resolution: {mic_row["rlnCtfMaxResolution"]:.2f} Å<br>\n'
        html_content += f'Defocus U: {mic_row["rlnDefocusU"]:.1f} Å<br>\n'
        html_content += f'Defocus V: {mic_row["rlnDefocusV"]:.1f} Å<br>\n'
        html_content += '</div>\n'

        plots = create_micrograph_plots(mic_row)
        if plots:
            html_content += '<div class="plot-container">\n'
            if 'micrograph' in plots:
                html_content += f'<div class="plot"><img src="data:image/png;base64,{plots["micrograph"]}" /></div>\n'
            if 'power_spectrum' in plots:
                html_content += f'<div class="plot"><img src="data:image/png;base64,{plots["power_spectrum"]}" /></div>\n'
            html_content += '</div>\n'
            if 'radial_ps' in plots:
                html_content += f'<div><img src="data:image/png;base64,{plots["radial_ps"]}" /></div>\n'

        html_content += '</div>\n'

    # Process bottom micrographs
    html_content += "<h2>Bottom " + str(n_bottom) + " Micrographs (Lowest Cross-Beta Score)</h2>\n"

    for i in range(min(n_bottom, len(valid_sorted))):
        idx = valid_sorted.index[len(valid_sorted) - 1 - i]
        mic_row = valid_sorted.loc[idx]

        html_content += '<div class="micrograph-section">\n'
        html_content += f'<div class="micrograph-header">Rank #{len(valid_sorted) - i}: {Path(mic_row["rlnMicrographName"]).name}</div>\n'
        html_content += '<div class="metadata">\n'
        html_content += f'Cross-Beta Score: {mic_row["CrossBetaScore"]:.6f}<br>\n'
        html_content += f'CTF Max Resolution: {mic_row["rlnCtfMaxResolution"]:.2f} Å<br>\n'
        html_content += f'Defocus U: {mic_row["rlnDefocusU"]:.1f} Å<br>\n'
        html_content += f'Defocus V: {mic_row["rlnDefocusV"]:.1f} Å<br>\n'
        html_content += '</div>\n'

        plots = create_micrograph_plots(mic_row)
        if plots:
            html_content += '<div class="plot-container">\n'
            if 'micrograph' in plots:
                html_content += f'<div class="plot"><img src="data:image/png;base64,{plots["micrograph"]}" /></div>\n'
            if 'power_spectrum' in plots:
                html_content += f'<div class="plot"><img src="data:image/png;base64,{plots["power_spectrum"]}" /></div>\n'
            html_content += '</div>\n'
            if 'radial_ps' in plots:
                html_content += f'<div><img src="data:image/png;base64,{plots["radial_ps"]}" /></div>\n'

        html_content += '</div>\n'

    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    """Main function."""
    args = parse_arguments()

    # Convert paths
    input_star = Path(args.input)
    output_star = Path(args.output)

    if not input_star.exists():
        print(f"Error: Input file not found: {input_star}", file=sys.stderr)
        sys.exit(1)

    # Determine RELION working directory
    if args.relion_dir:
        relion_wdir = Path(args.relion_dir)
    else:
        # Assume input is in CtfFind/jobXXX/ subdirectory
        relion_wdir = input_star.parent.parent.parent

    if args.verbose:
        print(f"RELION working directory: {relion_wdir}")
        print(f"Input STAR file: {input_star}")
        print(f"Output STAR file: {output_star}")

    # Calculate cross-beta frequency window
    k_cross_beta = args.cross_beta_k
    k_min = (1 - args.rel_width/2) * k_cross_beta
    k_max = (1 + args.rel_width/2) * k_cross_beta

    if args.verbose:
        print(f"\nCross-beta detection parameters:")
        print(f"  Central frequency: {k_cross_beta:.4f} 1/Å ({1/k_cross_beta:.2f} Å)")
        print(f"  Frequency window: {k_min:.4f} - {k_max:.4f} 1/Å")
        print(f"  Relative width: {args.rel_width*100:.1f}%")
        print(f"  CTF max resolution threshold: {args.ctf_max_res:.1f} Å")
        print(f"  CTF stability threshold: {args.ctf_stability:.2f}")

    # Read input STAR file
    if args.verbose:
        print(f"\nReading STAR file...")

    mic_data = starfile.read(input_star)
    micctf_df = mic_data["micrographs"]
    mic_optics = mic_data["optics"]

    # Limit number of micrographs if specified
    if args.n_mics is not None:
        if args.verbose:
            print(f"Limiting to first {args.n_mics} micrographs")
        micctf_df = micctf_df.iloc[:args.n_mics].copy()
    else:
        if args.verbose:
            print(f"Processing all {len(micctf_df)} micrographs")
        micctf_df = micctf_df.copy()

    # Filter by CTF max resolution
    n_before = len(micctf_df)
    micctf_df = micctf_df[micctf_df["rlnCtfMaxResolution"] <= args.ctf_max_res]
    micctf_df = micctf_df.reset_index(drop=True)
    n_after = len(micctf_df)

    if args.verbose:
        print(f"Filtered {n_before - n_after} micrographs by CTF resolution (kept {n_after})")

    if len(micctf_df) == 0:
        print("Error: No micrographs passed CTF resolution filter", file=sys.stderr)
        sys.exit(1)

    # Extract optics parameters
    angpix = mic_optics["rlnMicrographPixelSize"].iloc[0]
    cs = mic_optics["rlnSphericalAberration"].iloc[0]
    voltage = mic_optics["rlnVoltage"].iloc[0]
    amp_contrast = mic_optics["rlnAmplitudeContrast"].iloc[0]

    if args.verbose:
        print(f"\nOptics parameters:")
        print(f"  Pixel size: {angpix} Å/pix")
        print(f"  Voltage: {voltage} kV")
        print(f"  Cs: {cs} mm")
        print(f"  Amplitude contrast: {amp_contrast}")

    # Parse radial power spectra for all micrographs
    if args.verbose:
        print(f"\nParsing CTFFIND4 radial power spectra...")

    k_arrays = []
    raps_arrays = []

    for idx in tqdm(range(len(micctf_df)), disable=not args.verbose):
        try:
            ps_df = parse_ctffind4_radial_ps(
                micctf_df.iloc[idx]["rlnCtfImage"],
                relion_wdir
            )
            k_arrays.append(ps_df["SpatialFrequency"].values)
            raps_arrays.append(ps_df["RotAvg"].values)
        except Exception as e:
            print(f"Error parsing micrograph {idx}: {e}", file=sys.stderr)
            sys.exit(1)

    k_arrays = np.array(k_arrays)

    # Verify spatial frequency arrays are consistent
    if not np.all(k_arrays == k_arrays[0]):
        print("Warning: Spatial frequency arrays differ between micrographs", file=sys.stderr)

    k = k_arrays[0]
    raps = np.array(raps_arrays)

    # Calculate cross-beta scores
    if args.verbose:
        print(f"\nCalculating cross-beta scores...")

    def cross_beta_score_wrapper(rot_ps_input):
        return measure_cross_beta_signal(
            k, rot_ps_input, k_min, k_max, weight=args.weight
        )

    cross_beta_scores = np.apply_along_axis(
        cross_beta_score_wrapper, axis=1, arr=raps
    )

    # Calculate CTF² values at cross-beta frequency
    if args.verbose:
        print(f"Calculating CTF² values at cross-beta frequency...")

    ctf2_values = calculate_ctf2_at_frequency(
        k=k_cross_beta,
        defocus1=micctf_df["rlnDefocusU"].values,
        defocus2=micctf_df["rlnDefocusV"].values,
        astig_angle=micctf_df["rlnDefocusAngle"].values,
        voltage=voltage,
        cs=cs,
        amp_contrast=amp_contrast
    )

    # Correct cross-beta scores by CTF²
    if args.verbose:
        print(f"Correcting cross-beta scores for CTF effects...")

    corrected_scores = np.zeros_like(cross_beta_scores)
    n_unreliable = 0

    for i in range(len(cross_beta_scores)):
        if ctf2_values[i] > args.ctf_stability:
            corrected_scores[i] = cross_beta_scores[i] / ctf2_values[i]
        else:
            corrected_scores[i] = np.nan
            n_unreliable += 1

    if args.verbose and n_unreliable > 0:
        print(f"Marked {n_unreliable} micrographs as unreliable (CTF² < {args.ctf_stability})")

    # Add cross-beta scores to dataframe
    micctf_df["CrossBetaScore"] = corrected_scores

    # Sort by cross-beta score (descending, NaN last)
    mic_df_sorted = micctf_df.sort_values(
        by=["CrossBetaScore"], ascending=False, na_position='last'
    )

    # Prepare output data
    mic_data_out = mic_data.copy()
    mic_data_out["micrographs"] = mic_df_sorted

    # Write output STAR file
    if args.verbose:
        print(f"\nWriting sorted STAR file to: {output_star}")

    output_star.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(mic_data_out, output_star)

    # Print summary statistics
    valid_scores = corrected_scores[~np.isnan(corrected_scores)]

    print(f"\nSummary:")
    print(f"  Total micrographs processed: {len(micctf_df)}")
    print(f"  Valid cross-beta scores: {len(valid_scores)}")
    print(f"  Unreliable (CTF² too low): {n_unreliable}")

    if len(valid_scores) > 0:
        print(f"\nCross-beta score statistics:")
        print(f"  Mean: {np.mean(valid_scores):.6f}")
        print(f"  Median: {np.median(valid_scores):.6f}")
        print(f"  Std dev: {np.std(valid_scores):.6f}")
        print(f"  Min: {np.min(valid_scores):.6f}")
        print(f"  Max: {np.max(valid_scores):.6f}")

    print(f"\nOutput written to: {output_star}")

    # Generate HTML report if requested
    if args.report:
        if args.verbose:
            print(f"\nGenerating HTML report (takes some time depending on the microgrph location)...")

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        generate_html_report(
            micctf_df_sorted=mic_df_sorted,
            relion_wdir=relion_wdir,
            k_min=k_min,
            k_max=k_max,
            n_top=args.n_report,
            n_bottom=args.n_report,
            output_path=str(report_path),
            angpix=angpix
        )

        print(f"HTML report written to: {report_path}")

    print("Done!")


if __name__ == "__main__":
    main()
