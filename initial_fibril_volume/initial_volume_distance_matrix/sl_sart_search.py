#!/usr/bin/env python3
import numpy as np
from skimage import transform
from skimage.transform import iradon_sart
from skimage import io
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import mrcfile
from tqdm import tqdm



def calculate_entropy(image, n_bins=256):
    """
    Calculates the Shannon entropy of an image.

    Parameters:
      image : 2D numpy array (float or uint8)
      n_bins : int, number of bins for histogram (default: 256 for 8-bit images)

    Returns:
      entropy : float
    """
    # Convert to uint8 if float
    if np.issubdtype(image.dtype, np.floating):
        image_uint8 = (255 * (image - np.min(image)) / (np.ptp(image) + 1e-8)).astype(np.uint8)
    else:
        image_uint8 = image

    hist, _ = np.histogram(image_uint8, bins=n_bins, range=(0, 255), density=True)
    hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def save_mrc_cryoem_standard(volume, filename, apix=None):
    """
    Save a 3D numpy array as a standard cryo-EM MRC file.
    Optionally set pixel size (apix, Å).
    """
    assert volume.ndim == 3
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32))
        mrc.header.cella = (
            volume.shape[2] * apix if apix else volume.shape[2],
            volume.shape[1] * apix if apix else volume.shape[1],
            volume.shape[0] * apix if apix else volume.shape[0]
        )
        mrc.header.cellb = (90.0, 90.0, 90.0)
        mrc.header.mx = volume.shape[2]
        mrc.header.my = volume.shape[1]
        mrc.header.mz = volume.shape[0]
        if apix is not None:
            mrc.voxel_size = apix
        mrc.header.map = b"MAP "
        # mrc.header.machst = mrcfile.constants.MRC_MACHINE_STAMP
        # mrc.header.machst = mrcfile.default_machinestamp
        mrc.update_header_stats()
        mrc.update_header_from_data()
    print(f"Saved {filename} with shape {volume.shape} (Z, Y, X), apix={apix}")




def apply_cylindrical_mask(volume, radius):
    """
    Sets voxels outside a cylinder of given radius to zero, centered in x/y.
    Cylinder axis is along z.
    """
    zdim, ydim, xdim = volume.shape
    center_x = xdim // 2
    center_y = ydim // 2
    Y, X = np.ogrid[:ydim, :xdim]
    r2 = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask2d = r2 <= radius ** 2
    mask3d = np.broadcast_to(mask2d, (zdim, ydim, xdim))
    return np.where(mask3d, volume, 0)



def helical_3d_map_from_section_cubic(cross_section, degrees=180, output_file="helical_density.mrc", radius=None, apix=1.0):
    """
    Create a 3D cubic helical density map from a 2D cross-section.
    Each z-slice rotates by twist_per_slice so the full box covers the specified degrees.
    The output shape is (N, N, N), with N = cross_section.shape[0].

    Parameters:
      cross_section : 2D numpy array, shape (N, N)
      degrees : int, 180 or 360 (the angular span covered)
      output_file : str or Path
    """
    N = cross_section.shape[0]
    twist_per_slice = degrees / N
    volume = np.zeros((N, N, N), dtype=np.float32)
    for i in range(N):
        angle = i * twist_per_slice
        rotated = rotate(cross_section, angle=angle, reshape=False, order=1, mode='nearest')
        volume[i, :, :] = rotated
        
    if radius is not None:
        volume = apply_cylindrical_mask(volume, radius)
        print(f"Applied cylindrical mask with radius {radius} pixels.")        
        
                 
    save_mrc_cryoem_standard(volume, output_file, apix=apix)
    print(f"3D cubic helical density saved as {output_file}")


def prealign_fibril(image, plot=False):
    """
    Rotate and shift a 2D class average so the main axis is horizontal (x)
    and center of mass is at center in y-direction.
    Returns the aligned image.
    """
    from scipy.ndimage import rotate, shift
    # 1. Threshold and find coordinates of nonzero pixels
    coords = np.column_stack(np.nonzero(image > (0.2 * np.max(image))))
    if coords.shape[0] < 10:
        return image  # Not enough signal, skip

    # 2. PCA: get main axis direction
    coords_mean = coords.mean(axis=0)
    coords_centered = coords - coords_mean
    U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
    main_axis = Vt[0]  # direction of max variance

    # 3. Angle to x-axis
    angle_rad = np.arctan2(main_axis[0], main_axis[1])  # [y, x] because of numpy ordering
    angle_deg = np.rad2deg(angle_rad)
    # Rotate so main axis is horizontal (x)
    img_rot = rotate(image, angle=angle_deg, reshape=False, order=1, mode='nearest')

    # 4. Center mass in y
    img_rot_thr = np.where(img_rot > 0.2 * np.max(img_rot), img_rot, 0)
    y_mass = np.sum(img_rot_thr, axis=1)
    y_current = np.sum(np.arange(len(y_mass)) * y_mass) / (np.sum(y_mass) + 1e-8)
    y_center = (image.shape[0] - 1) / 2
    img_aligned = shift(img_rot, shift=(y_center - y_current, 0), order=1, mode='nearest')

    # Optionally plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img_aligned, cmap='gray')
        plt.title('Prealigned fibril')
        plt.show()
    return img_aligned



def main():
    parser = argparse.ArgumentParser(description="SART cross-section reconstruction from a filament image")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="Output directory (default: current dir)")
    parser.add_argument("--crop", type=int, nargs=4, metavar=('YSTART', 'YEND', 'XSTART', 'XEND'),
                        default=None, help="Cropping coordinates: y_start y_end x_start x_end (default: full image)")
    parser.add_argument("--angles", type=float, nargs=3, metavar=('START', 'END', 'STEP'),
                        default=[0.0, 0.0, 1.0], help="Rotation angle range in degrees: start end step (default: 0 0 1)")
    parser.add_argument("--shifts", type=float, nargs=3, metavar=('START', 'END', 'STEP'),
                        default=[0.0, 0.0, 1.0], help="Translation shift range (pixels): start end step (default: 0 0 1)")
    parser.add_argument("--proj360", action="store_true",
                        help="Use 360 degree projection (default: 180)")
    parser.add_argument("--show", action="store_true",
                        help="Display the reconstruction image(s)")
    parser.add_argument("--box-range", type=int, nargs=3, metavar=("START", "END", "STEP"),
               default=None, help="Iterate over x-crop widths: start end step (in pixels, e.g. 800 1200 100)")
    parser.add_argument("--radius", type=int, default=None,
        help="Radius (in pixels) for cylindrical mask to apply to 3D density map")
    parser.add_argument("--apix", type=float, default=1.0, help="Pixel size in Angstrom")
    parser.add_argument("--prealign", action="store_true", help="Pre-align each class average")
    parser.add_argument("--resize", type=float, default=1.0, help="Factor to resize image grid (default: 1.0)")




    args = parser.parse_args()

    best_entropy = None
    best_recon = None
    best_fname = None


    # Prepare paths
    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load image
    a = io.imread(str(input_file))
    if a.ndim == 3:  # If RGB or RGBA
        a = np.mean(a[..., :3], axis=2)  # Convert to grayscale

    # Crop
    if args.crop:
        y0, y1, x0, x1 = args.crop
    else:
        y0, y1, x0, x1 = 0, a.shape[0], 0, a.shape[1]
    a_cropped = a[y0:y1, x0:x1]
    print(f"Input image: {a.shape}, Cropped: {a_cropped.shape}")


    # --- RESIZE ---
    if args.resize != 1.0:
        new_shape = (int(a_cropped.shape[0] * args.resize),
                     int(a_cropped.shape[1] * args.resize))
        a_cropped = transform.resize(a_cropped, new_shape, order=3, preserve_range=True, anti_aliasing=True)
        a_cropped = a_cropped.astype(np.float32)
        print(f"Resized image to {a_cropped.shape}")


    if args.prealign:
        a_cropped = prealign_fibril(a_cropped)


    box_length_x_values = [a_cropped.shape[1]]  # default: use full width
    if args.box_range is not None:
        # start, end, step = args.box_range
        start, end, step = [int(x * args.resize) for x in args.box_range]

        if (step < 1):
            step = 1
        box_length_x_values = list(range(start, end + 1, step))

    # Angle and shift ranges
    ang_start, ang_end, ang_step = args.angles
    if (ang_step < 1):
        ang_step = 1
    shift_start, shift_end, shift_step = args.shifts
    if (shift_step < 1):
        shift_step = 1
    angle_range = np.arange(ang_start, ang_end + ang_step, ang_step)
    shift_range = np.arange(shift_start, shift_end + shift_step, shift_step)

    # Precompute total number of iterations
    total_pairs = len(box_length_x_values) * len(angle_range) * len(shift_range)
    pbar        = tqdm(total=total_pairs, desc='Grid search')


    # Write out entropies to a file
    entropy_log_file = output_dir / "entropies.txt"
    with open(entropy_log_file, "w") as elog:
        elog.write("box_length_x\tangle\tshift\tentropy\tfilename\n")


        for box_length_x in box_length_x_values:
            
            width = a_cropped.shape[1]
            center_x = width // 2
            half_box = box_length_x // 2
            # For even box_length_x, this will be symmetric; for odd, right side will be one pixel wider
            start_x = center_x - half_box
            end_x = start_x + box_length_x
            # Ensure bounds
            if start_x < 0:
                start_x = 0
                end_x = box_length_x
            if end_x > width:
                end_x = width
                start_x = width - box_length_x
            a_cropped_x = a_cropped[:, start_x:end_x]
            # print(f"box_length_x={box_length_x}, cropping to x={start_x}:{end_x}, shape={a_cropped_x.shape}")
            
            
            # # Symmetric crop in x for current box_length_x
            # if box_length_x < a_cropped.shape[1]:
            #     width = a_cropped.shape[1]
            #     start_x = (width - box_length_x) // 2
            #     end_x = start_x + box_length_x
            #     a_cropped_x = a_cropped[:, start_x:end_x]
            #     print(f"Symmetric crop along x: x={start_x}:{end_x} (width={box_length_x})")
            # else:
            #     a_cropped_x = a_cropped
            #     print(f"No x-cropping (using width={a_cropped.shape[1]})")

    
            # SART
            for angle in angle_range:
                for shift in shift_range:
                    # print(f"Processing angle={angle}, shift={shift}")
                    pbar.update(1)

        
                    tform = transform.EuclideanTransform(rotation=np.deg2rad(angle), translation=[0, shift])
                    c = transform.warp(a_cropped_x, tform)
        
                    theta_max = 360 if args.proj360 else 180
                    # theta = np.linspace(0., theta_max, max(c.shape), endpoint=False)
                    theta = np.linspace(0., theta_max, c.shape[1], endpoint=False)                    
                    try:
                        reconstruction_sart = iradon_sart(c, theta=theta)
                    except Exception as e:
                        print(f"  Skipped (iradon_sart error): {e}")
                        continue
        
                    # Normalize for output
                    recon_norm = (reconstruction_sart - np.min(reconstruction_sart)) / (np.max(reconstruction_sart) - np.min(reconstruction_sart) + 1e-8)
        
                    #entropy = shannon_entropy(recon_norm*100)
                    entropy = calculate_entropy(recon_norm)
    
                    # print(f"  Entropy: {entropy:.6f}")
        
                    # Track the lowest entropy
                    if (best_entropy is None) or (entropy < best_entropy):
                        best_entropy = entropy
                        best_angle = angle
                        best_shift = shift
                        best_box_length_x = box_length_x
                        best_recon = recon_norm.copy()
                        best_fname = f"{output_dir.name}-best-rec.png"
    
        
                    # Save
                    #fname = f"recon_{input_file.stem}_shift_{shift:.2f}_angle_{angle:.2f}.png"
                    fname = f"recon_{input_file.stem}_width_{box_length_x}_shift_{shift:.2f}_angle_{angle:.2f}.png"
    
                    # Write results
                    elog.write(f"{box_length_x}\t{angle}\t{shift}\t{entropy:.8f}\t{fname}\n")
                    elog.flush()  # Ensure write even if interrupted
    
                    io.imsave(str(output_dir / fname), (recon_norm * 255).astype(np.uint8))
                    # print(f"  Saved: {fname}")
    
        
                    # Optionally display
                    if args.show:
                        plt.figure()
                        plt.imshow(recon_norm, cmap='gray')
                        plt.title(f"Reconstruction angle={angle}, shift={shift}")
                        plt.axis('off')
                        plt.show()

    pbar.close()

    # After all reconstructions, save the best one
    if best_recon is not None:
        out_best = output_dir / best_fname
        io.imsave(str(out_best), (best_recon * 255).astype(np.uint8))
        print(f"\nLowest entropy: {best_entropy:.6f} at angle: {best_angle:.1f}, shift: {best_shift}, box: {best_box_length_x}")
        print(f"Best reconstruction saved as: {out_best}")
    else:
        print("No valid reconstruction was found.")

    # Write 3D volume
    out_mrc = output_dir / "helical_density_best.mrc"
    helical_3d_map_from_section_cubic(best_recon, degrees=theta_max, output_file=out_mrc, radius=args.radius, apix=args.apix)
    #helical_3d_map_from_section(best_recon, n_slices, degrees, output_file="helical_density.mrc")



if __name__ == "__main__":
    main()

