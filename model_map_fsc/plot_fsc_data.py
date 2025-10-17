#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def parse_fsc_file(filename):
    """Parse FSC data from file with format: x_value\ty_value"""
    x_values = []
    y_values = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by tab
                data_parts = line.split('\t')
                if len(data_parts) == 2:
                    try:
                        x_val = float(data_parts[0])
                        y_val = float(data_parts[1])
                        x_values.append(x_val)
                        y_values.append(y_val)
                    except ValueError:
                        continue

    return np.array(x_values), np.array(y_values)

def find_resolution_at_threshold(x_values, y_values, threshold):
    """Find resolution (1/Å) where FSC drops below threshold"""
    for i in range(len(y_values)):
        if y_values[i] < threshold:
            if i > 0:
                # Linear interpolation between points
                x1, y1 = x_values[i-1], y_values[i-1]
                x2, y2 = x_values[i], y_values[i]
                x_interp = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                return x_interp
            else:
                return x_values[i]
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Plot Fourier Shell Correlation (FSC) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s fsc.txt
  %(prog)s fsc.txt -o output.png
  %(prog)s fsc.txt --thresholds 0.5 0.143 0.3
  %(prog)s fsc.txt --title "My FSC Analysis" --dpi 600
        '''
    )

    parser.add_argument('input',
                        help='Input FSC file (tab-separated: spatial_frequency\\tFSC_value)')
    parser.add_argument('-o', '--output',
                        default='fsc_plot.png',
                        help='Output plot filename (default: fsc_plot.png)')
    parser.add_argument('-t', '--thresholds',
                        nargs='+',
                        type=float,
                        default=[0.5, 0.143],
                        help='FSC threshold values to mark (default: 0.5 0.143)')
    parser.add_argument('--title',
                        default='Fourier Shell Correlation',
                        help='Plot title (default: "Fourier Shell Correlation")')
    parser.add_argument('--dpi',
                        type=int,
                        default=300,
                        help='Output resolution in DPI (default: 300)')
    parser.add_argument('--figsize',
                        nargs=2,
                        type=float,
                        default=[12, 8],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 12 8)')
    parser.add_argument('--color',
                        default='blue',
                        help='Line color (default: blue)')
    parser.add_argument('--no-grid',
                        action='store_true',
                        help='Disable grid')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Suppress output messages')

    args = parser.parse_args()

    # Parse FSC file
    try:
        x, y = parse_fsc_file(args.input)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found!", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Parsed {len(x)} FSC data points from '{args.input}'")

    if len(x) == 0:
        print("Error: No data found in file!", file=sys.stderr)
        sys.exit(1)

    # Calculate resolution at thresholds
    for threshold in sorted(args.thresholds, reverse=True):
        res = find_resolution_at_threshold(x, y, threshold)
        if res and not args.quiet:
            print(f"FSC drops below {threshold} at {res:.4f} (1/Å), resolution: {1/res:.2f} Å")
        elif not args.quiet:
            print(f"FSC never drops below {threshold}")

    # Create the plot
    plt.figure(figsize=tuple(args.figsize))
    plt.plot(x, y, '-', color=args.color, linewidth=1.5, label='FSC')


    # Add horizontal reference lines and vertical lines at resolution thresholds
    colors = ['gray', 'darkgray', 'silver', 'lightgray']
    linestyles = ['--', ':', '-.', '--']
    text_positions = [0.4, 0.05, 0.35, 0.0]

    for i, threshold in enumerate(sorted(args.thresholds, reverse=True)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        # Horizontal threshold line
        plt.axhline(y=threshold, color=color, linestyle=linestyle,
                   alpha=0.7, label=f'FSC = {threshold}')

        # Vertical line at resolution
        res = find_resolution_at_threshold(x, y, threshold)
        if res:
            plt.axvline(x=res, color=args.color, linestyle=linestyle, alpha=0.5)
            text_pos = text_positions[i % len(text_positions)]
            plt.text(res, text_pos, f'{1/res:.1f}Å', rotation=90,
                    ha='right', va='bottom', color=args.color)

    plt.xlabel('Spatial Frequency (1/Å)')
    plt.ylabel('Fourier Shell Correlation')
    plt.title(args.title)
    plt.legend()

    if not args.no_grid:
        plt.grid(True, alpha=0.3)

    plt.xlim(0, max(x))
    plt.ylim(min(y), 1)

    # Save the plot
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')

    if not args.quiet:
        print(f"Plot saved as '{args.output}'")

if __name__ == '__main__':
    main()