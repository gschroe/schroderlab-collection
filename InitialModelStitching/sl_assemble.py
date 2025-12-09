#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:01:20 2025

@author: gunnar
"""

import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from itertools import combinations
import mrcfile
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim


def lowpass(img, sigma=2):
    """Apply Gaussian low-pass filter."""
    return gaussian_filter(img, sigma=sigma)


def symmetric_y_crop_pair(arr1, arr2, target_height):
    """
    Crops or pads arr1 and arr2 along y so both have shape (target_height, arrX.shape[1]),
    centered vertically, with identical top/bottom coordinates.
    """
    assert arr1.shape[1] == arr2.shape[1], "Width must match"
    h1, w = arr1.shape
    h2, _ = arr2.shape

    # Use average center for safety
    c1 = h1 // 2
    c2 = h2 // 2
    center = (c1 + c2) // 2

    # Compute cropping window
    half = target_height // 2
    start = center - half
    end = start + target_height

    def pad_crop(a):
        pad_top = max(0, -start)
        pad_bot = max(0, end - a.shape[0])
        a_padded = np.pad(a, ((pad_top, pad_bot), (0, 0)), mode='constant')
        start2 = start + pad_top
        end2 = start2 + target_height
        return a_padded[start2:end2, :]

    arr1c = pad_crop(arr1)
    arr2c = pad_crop(arr2)
    return arr1c, arr2c



def symmetric_crop(img, target_shape):
    """
    Crops (or pads) img to target_shape (height, width), keeping the center fixed.
    If target_shape is larger than img, pads with zeros.
    """
    import numpy as np
    h, w = img.shape
    target_h, target_w = target_shape

    # --- Y dimension (vertical) ---
    if target_h < h:
        # Crop equally top and bottom
        crop_top = (h - target_h) // 2
        crop_bot = h - target_h - crop_top
        img = img[crop_top:h - crop_bot, :]
    elif target_h > h:
        # Pad equally top and bottom
        pad_top = (target_h - h) // 2
        pad_bot = target_h - h - pad_top
        img = np.pad(img, ((pad_top, pad_bot), (0, 0)), mode='constant')
    # else: already the right size

    # --- X dimension (horizontal) ---
    if target_w < w:
        crop_left = (w - target_w) // 2
        crop_right = w - target_w - crop_left
        img = img[:, crop_left:w - crop_right]
    elif target_w > w:
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant')
    # else: already right size

    return img


def crop_center(img, crop_size=60):
    y, x = img.shape
    startx = x // 2 - crop_size // 2
    starty = y // 2 - crop_size // 2
    return img[starty:starty+crop_size, startx:startx+crop_size]

def normalized_cross_correlation(a, b):
    arr1 = a - np.mean(a)
    arr2 = b - np.mean(b)
    denom = (np.std(arr1) + 1e-8) * (np.std(arr2) + 1e-8)
    if denom == 0:
        return -1
    return np.mean(arr1 * arr2) / denom

def rmsd(a, b):
    """Compute the root mean squared deviation between two arrays."""
    diff = a - b
    return np.sqrt(np.mean(diff ** 2))

def mae(a, b):
    return np.mean(np.abs(a - b))

def ssim_score(a, b):
    # ssim returns a score in [0,1], higher is better
    return ssim(a, b, data_range=a.max()-a.min())

def mutual_information(hgram):
    """Compute mutual information from a 2D histogram."""
    # Convert bins counts to probability
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]
    # Only non-zero
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mi_score(a, b, bins=64):
    """Mutual information between two images, using bins."""
    # Flatten and scale images to 0-1
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    hgram, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    return mutual_information(hgram)

def soft_edge_mask_precise(img, feather_width=5.0):
    """
    Mask is 1 everywhere inside the object except near the internal edge,
    where it falls smoothly to 0 (over feather_width pixels). Mask is 0 outside.
    """
    hard_mask = (img != 0)
    dist_inside = distance_transform_edt(hard_mask)
    mask = np.zeros_like(img, dtype=np.float32)
    inside = hard_mask & (dist_inside > 0)
    # Smooth transition in the feather region
    mask[inside] = np.clip(dist_inside[inside] / feather_width, 0, 1)
    # Fully 1 deep inside
    mask[dist_inside >= feather_width] = 1.0
    # Exactly zero outside
    mask[~hard_mask] = 0.0
    return mask


# def mask_circle_old(img, radius_fraction=0.9):
#     """Sets all values outside a central circle to zero.
#     radius_fraction: 0.9 means 90% of half the minimum image dimension.
#     !! I also need to set all values that are exactly
#     """
#     h, w = img.shape
#     cy, cx = h // 2, w // 2
#     max_r = radius_fraction * 0.5 * min(h, w)
#     Y, X = np.ogrid[:h, :w]
#     dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
#     mask = dist <= max_r
#     masked_img = np.zeros_like(img)
#     masked_img[mask] = img[mask]
#     return masked_img



def mask_circle(img, radius_fraction=0.9, threshold=1e-12):
    """Mask outside central circle. Values inside < threshold are replaced with +/-2e-12."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    max_r = radius_fraction * 0.5 * min(h, w)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist <= max_r

    masked_img = np.zeros_like(img)
    # Only set inside the circle
    masked_img[mask] = img[mask]
    # Find low-value pixels inside mask
    lowval = (np.abs(masked_img) < threshold) & mask
    # Randomly assign -2e-12 or +2e-12
    random_signs = np.random.choice([-1, 1], size=lowval.sum())
    masked_img[lowval] = 2e-12 * random_signs

    return masked_img




# def mask_circle_smooth(shape, radius_fraction=0.9, feather_sigma=0):
#     """Return a (smoothed) circular mask of the same shape as the image."""
#     h, w = shape
#     cy, cx = h // 2, w // 2
#     max_r = radius_fraction * 0.5 * min(h, w)
#     Y, X = np.ogrid[:h, :w]
#     dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
#     mask = (dist <= max_r).astype(np.float32)
#     if feather_sigma > 0:
#         mask = gaussian_filter(mask, sigma=feather_sigma)
#         # Re-normalize to [0, 1]
#         mask = mask / mask.max()
#     return mask

    
def alpha_blend(img1, img2, mask1=None, mask2=None, radius_fraction=0.9, feather_sigma=2):
 
    mask1 = soft_edge_mask_precise(img1, feather_width=12)
    mask2 = soft_edge_mask_precise(img2, feather_width=12)
   
    alpha_sum = mask1 + mask2
    alpha_sum[alpha_sum == 0] = 1
    blended = (img1 * mask1 + img2 * mask2) / alpha_sum
    # blended[(mask1 + mask2) == 0] = 0
    return blended


def transform_image(img, angle, tx, ty, out_shape):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    canvas = np.zeros(out_shape, dtype=rotated.dtype)

    y0, x0 = int(ty), int(tx)
    h, w = rotated.shape
    y1, x1 = y0 + h, x0 + w

    # Determine the overlap region
    oy0 = max(0, y0)
    ox0 = max(0, x0)
    oy1 = min(out_shape[0], y1)
    ox1 = min(out_shape[1], x1)
    # Corresponding region in rotated image
    ry0 = oy0 - y0
    rx0 = ox0 - x0
    ry1 = ry0 + (oy1 - oy0)
    rx1 = rx0 + (ox1 - ox0)

    # If there is any overlap, paste
    if oy0 < oy1 and ox0 < ox1:
        canvas[oy0:oy1, ox0:ox1] = rotated[ry0:ry1, rx0:rx1]
        return canvas
    else:
        return None





def confidence_weighted_blend(img1, img2, mask1=None, mask2=None):
    if mask1 is None:
        mask1 = (img1 > 0).astype(np.float32)
    if mask2 is None:
        mask2 = (img2 > 0).astype(np.float32)
    c1 = np.clip(img1, 0, 1) * mask1
    c2 = np.clip(img2, 0, 1) * mask2
    # both_nonzero = (mask1 > 0) & (mask2 > 0)
    alpha = c1 * c2
    avg = 0.5 * (img1 + img2)
    mx = np.maximum(img1, img2)
    out = alpha * avg + (1 - alpha) * mx
    out[(mask1 + mask2) == 0] = 0
    return out



def align_lowpass_images(
        img1, img2, angles,
        x_translations=None, y_translations=None,
        crop_hw=(100, 60), lowpass_sigma=0,
        min_overlap_frac=0.10,            # ← at least 10 % of img2 must overlap
        metric="rmsd"
    ):
    """
    Return None if no shift/angle yields                                         
        • a geometric overlap ≥ crop_size, **and**                              
        • at least `min_overlap_frac` of non-zero pixels of img2 overlap img1.  
    """

    crop_h, crop_w = crop_hw
    
    # 1  low-pass if wanted
    img1_lp = lowpass(img1, sigma=lowpass_sigma) if lowpass_sigma > 0 else img1
    img2_lp = lowpass(img2, sigma=lowpass_sigma) if lowpass_sigma > 0 else img2

    # 2  large canvas
    H = max(img1.shape[0], img2.shape[0]) * 2
    W = max(img1.shape[1], img2.shape[1]) * 2

    out_shape = (H, W)
    off_y = H // 2 - img1.shape[0] // 2
    off_x = W // 2 - img1.shape[1] // 2

    # 3  safe dx / dy so geom. overlap ≥ crop_size
    # dx_max = img1.shape[1] - crop_size
    dx_max = img1.shape[1] - crop_w

    # dy_max = img1.shape[0] - crop_size
    auto_x = np.arange(-dx_max, dx_max + 1, 1)
    # auto_y = np.arange(-dy_max, dy_max + 1, 1)

    x_translations = auto_x if x_translations is None else np.intersect1d(x_translations, auto_x)
    ## I always define y_translations outside, this range is fixed.
    # y_translations = auto_y if y_translations is None else np.intersect1d(y_translations, auto_y)

    # 4  static copy of img1 on canvas
    img1_can = np.zeros(out_shape, dtype=img1.dtype)
    img1_can[off_y:off_y+img1.shape[0], off_x:off_x+img1.shape[1]] = img1
    img1_lp_can = np.zeros_like(img1_can)
    img1_lp_can[off_y:off_y+img1.shape[0], off_x:off_x+img1.shape[1]] = img1_lp
    # mask1 = img1_can > 0
    mask1 = img1_can != 0           # accept any non-zero value


    # 5  threshold for content overlap
    # min_overlap_px = int(min_overlap_frac * (img2 > 0).sum())
    # min_overlap_px = int(min_overlap_frac * (img2 != 0).sum())
    # min_overlap_px = max(int(min_overlap_frac * (img2 != 0).sum()),
                      # (crop_h * crop_w))

    
    # if min_overlap_px < (crop_h * crop_w):        # never less than 1/8 window
    #     min_overlap_px = (crop_h * crop_w)
    min_overlap_px = (crop_h * crop_w)


    best_score  = -np.inf
    best_params = (0., 0, 0)

    for ang in angles:
        for dy in y_translations:
            for dx in x_translations:
                ty, tx = off_y + dy, off_x + dx
                img2_lp_can  = transform_image(img2_lp,  ang, tx, ty, out_shape)
                img2_raw_can = transform_image(img2,     ang, tx, ty, out_shape)
                if img2_lp_can is None or img2_raw_can is None:
                    continue

                # mask2 = img2_raw_can > 0
                mask2 = img2_raw_can != 0          # same for img2
                
                overlap = mask1 & mask2
                ov_count = int(overlap.sum())
                if ov_count < min_overlap_px:
                    continue                                # too little real overlap

                coords = np.argwhere(overlap)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                
                
                # if (y1 - y0) < crop_size or (x1 - x0) < crop_size:
                #     continue
                # half = crop_size // 2
                # cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                # sy = min(max(cy - half, y0), y1 - crop_size)
                # sx = min(max(cx - half, x0), x1 - crop_size)

                # crop1 = img1_lp_can[sy:sy+crop_size, sx:sx+crop_size]
                # crop2 = img2_lp_can[ sy:sy+crop_size, sx:sx+crop_size]

                # print("(y1 - y0) < crop_h   (x1 - x0) < crop_w ", (y1 - y0),crop_h,(x1 - x0), crop_w)
                if (y1 - y0) < crop_h or (x1 - x0) < crop_w:
                    continue
                half_h = crop_h // 2
                half_w = crop_w // 2
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                sy = min(max(cy - half_h, y0), y1 - crop_h)
                sx = min(max(cx - half_w, x0), x1 - crop_w)
                crop1 = img1_lp_can[sy:sy+crop_h, sx:sx+crop_w]
                crop2 = img2_lp_can[sy:sy+crop_h, sx:sx+crop_w]





                # s = normalized_cross_correlation(crop1, crop2)
                # s = -rmsd(crop1, crop2)  # negative, because lower RMSD is better
                # s = -mae(crop1, crop2)  # negative, because lower RMSD is better
                # s = ssim_score(crop1, crop2)
                # s = mi_score(crop1, crop2)  # Higher is better
                # print("metric", metric)

                if   metric == "ncc":
                    s =  normalized_cross_correlation(crop1, crop2)
                elif metric == "rmsd":
                    s = -rmsd(crop1, crop2)
                elif metric == "mae":
                    s = -mae(crop1, crop2)
                elif metric == "ssim":
                    s =  ssim_score(crop1, crop2)
                elif metric == "mi":
                    s =  mi_score(crop1, crop2)
                else:
                    raise ValueError(f"Unknown metric: {metric}")


                
                if s > best_score:
                    best_score, best_params = s, (ang, dx, dy)

    if best_score == -np.inf:           # no legal placement
        return None

    ang, dx, dy = best_params
    img2_aligned = transform_image(img2, ang, off_x+dx, off_y+dy, out_shape)
    return img1_can, img2_aligned, best_params, best_score, None





def align_images_lowpass_with_flips(
        img1, img2, angles,
        x_translations, y_translations,
        crop_hw, lowpass_sigma=2, metric="rmsd", flips="all"):
    """
    Try none / horizontal / vertical flip of img2; pick the best.
    Returns None if all flips fail.
    """
    # flip_variants = [
    #     ("none",  img2),
    #     ("hflip", np.fliplr(img2)),
    #     ("vflip", np.flipud(img2)),
    # ]
    if flips == "none":
        flip_variants = [("none",  img2)]
    elif flips == "vflips":
        flip_variants = [("none",  img2),
                         ("vflip", np.flipud(img2))]
    elif flips == "hflips":
        flip_variants = [("none",  img2),
                         ("hflip", np.fliplr(img2))]
    else:                               # "all"
        flip_variants = [("none",  img2),
                         ("vflip", np.flipud(img2)),
                         ("hflip", np.fliplr(img2))]
    
    
    best_score = -np.inf
    best_all   = None

    for flip_name, img2_var in flip_variants:
        res = align_lowpass_images(
            img1, img2_var, angles,
            x_translations, y_translations,
            crop_hw=crop_hw, lowpass_sigma=lowpass_sigma, metric=metric)
        if res is None:
            continue
        img1_canvas, img2_aligned, params, score, corr_map = res
        if score > best_score:
            best_score = score
            best_all   = (img1_canvas, img2_aligned, params,
                          score, corr_map, flip_name)

    return best_all   # may be None




def find_symmetric_y_crop_indices(img, threshold=1e-12):
    if img.ndim == 2:
        mask = img > threshold
    else:
        mask = np.any(img > threshold, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        # Return full image if empty
        return 0, img.shape[0], 0, img.shape[1]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # center_y = (y0 + y1) // 2
    # crop_height = y1 - y0
    
    center_y = img.shape[0] // 2
    crop_height = np.min([(img.shape[0]-y1), y0])
    
    half_crop = crop_height // 2
    starty = max(center_y - half_crop, 0)
    endy = min(center_y + half_crop + (crop_height % 2), img.shape[0])

    return starty, endy, x0, x1

def crop_to_content_pair_symmetric_y(sum_arr, count_arr, threshold=1e-12):
    starty, endy, x0, x1 = find_symmetric_y_crop_indices(sum_arr, threshold)
    return sum_arr[starty:endy, x0:x1], count_arr[starty:endy, x0:x1]



def crop_to_content_symmetric_y(img, threshold=1e-12):
    if img.ndim == 2:
        mask = img > threshold
    else:
        mask = np.any(img > threshold, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # --- Symmetric crop in y ---
    center_y = (y0 + y1) // 2
    crop_height = y1 - y0
    half_crop = crop_height // 2
    # Find center line, expand equally up and down, stay within image bounds
    starty = max(center_y - half_crop, 0)
    endy = min(center_y + half_crop + (crop_height % 2), img.shape[0])

    # --- Tight crop in x ---
    return img[starty:endy, x0:x1]


def crop_to_content(img, threshold=1e-12):
    if img.ndim == 2:
        mask = img > threshold
    else:
        mask = np.any(img > threshold, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def crop_to_content_pair_old(sum_arr, count_arr, threshold=1e-12):
    """
    Crop both sum and count arrays to the minimal bounding box containing any non-zero in either.
    """
    import numpy as np
    # mask where either sum or count has content
    mask = (count_arr > threshold) | (sum_arr != 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return sum_arr, count_arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return sum_arr[y0:y1, x0:x1], count_arr[y0:y1, x0:x1]



def read_images_from_folder(folder,  mask_radius=1.0):
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    images = [cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fname in image_files]
    images = [mask_circle(img, radius_fraction=mask_radius) for img in images]
    return images, image_files

def read_images_from_mrcs(mrcs_file,  mask_radius=1.0, class_ids=None):
    with mrcfile.open(mrcs_file, permissive=True) as mrc:
        stack = mrc.data.astype(np.float32)
    if class_ids is not None:
        class_ids = [i - 1 for i in class_ids]
        stack = stack[class_ids]        

    # print("Shape:", stack.shape)
    # print("Global min/max/mean:", stack.min(), stack.max(), stack.mean())
    # print("First image min/max:", stack[0].min(), stack[0].max())        
    images = [stack[i] for i in range(stack.shape[0])]
    images = [mask_circle(img, radius_fraction=mask_radius) for img in images]
    
    if class_ids is not None:
        image_files = [f"{Path(mrcs_file).stem}_{i:03d}" for i in class_ids]
    else:
        image_files = [f"{Path(mrcs_file).stem}_{i:03d}" for i in range(len(images))]
    return images, image_files    
    
    
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
        plt.figure()
        plt.imshow(img_aligned, cmap='gray')
        plt.title('Prealigned fibril')
        plt.show()
    return img_aligned


def align_pair(imgA, imgB, angles,
               x_translations, y_translations,
               crop_hw, lowpass_sigma=1.0, metric="rmsd",  flips="all"):
    """
    Align imgB onto imgA.  Returns None if no alignment succeeded.
    """
    result = align_images_lowpass_with_flips(
        imgA, imgB,
        angles, x_translations, y_translations,
        crop_hw=crop_hw, lowpass_sigma=lowpass_sigma, metric=metric, flips=flips)

    if result is None:
        return None

    imgA_canvas, imgB_aligned, params, score, _, best_flip = result
    if score == -np.inf or imgA_canvas is None or imgB_aligned is None:
        return None

    return imgA_canvas, imgB_aligned, score, params, best_flip



# ---------------------------------------------------------------------------
# 1.  helper: symmetric pad/crop to a fixed height
# ---------------------------------------------------------------------------
def _fix_height(arr, target_h):
    """Return `arr` with the same number of rows as `target_h`, centred."""
    h, w = arr.shape
    if h == target_h:
        return arr
    if h < target_h:                              # pad
        top = (target_h - h) // 2
        bot = target_h - h - top
        return np.pad(arr, ((top, bot), (0, 0)), mode="constant")
    # (h > target_h) -> crop
    top = (h - target_h) // 2
    return arr[top:top + target_h, :]

# ---------------------------------------------------------------------------
# 2.  x-only crop that keeps the reference height
# ---------------------------------------------------------------------------
def crop_to_content_pair_fixed_y(sum_arr, count_arr, target_h, thr=1e-12):
    """
    Remove empty columns (x) but keep *exactly* `target_h` rows, centred.
    Identical cropping is applied to sum and count.
    """
    # Detect non-empty columns
    mask = (count_arr > thr) | (sum_arr != 0)
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:                       # nothing – return centred empty arrays
        return _fix_height(sum_arr, target_h), _fix_height(count_arr, target_h)
    x0, x1 = cols[0], cols[-1] + 1

    sum_c  = _fix_height(sum_arr[:,  x0:x1], target_h)
    cnt_c  = _fix_height(count_arr[:, x0:x1], target_h)
    return sum_c, cnt_c




def crop_to_content_pair(sum_arr, count_arr, threshold=1e-12):
    """
    Crop both sum and count arrays to the minimal bounding box containing any non-zero in either.
    """
    import numpy as np
    mask = (count_arr > threshold) | (sum_arr != 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return sum_arr, count_arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return sum_arr[y0:y1, x0:x1], count_arr[y0:y1, x0:x1]



def progressive_assembly_new(images, angles,
                             x_translations, y_translations,
                             crop_hw,  lowpass=1.0, metric="rmsd", flips="all"):
    """
    Hierarchical merge.  Stops gracefully if a round finds no alignable pair.
    """
    from itertools import combinations
    import numpy as np

    crop_h, crop_w = crop_hw

    # --- fixed x-step for sub-grids ---
    orig_x_step = (x_translations[1] - x_translations[0]) if len(x_translations) > 1 else 1

    # --- initial single-image components ---
    components = []
    for i, img in enumerate(images):
        M = soft_edge_mask_precise(img, feather_width=12)
        components.append({'sum': img * M,
                           'count': M,
                           'indices': [i]})

    total_pairs = len(components) * (len(components) - 1) * (len(components) + 1) // 6
    pbar = tqdm(total=total_pairs, desc='Aligning pairs')

    H0 = images[0].shape[0]        # input box height – stays constant


    history, step = [], 0
    while len(components) > 1:
        best_score   = -np.inf
        best_pair    = None
        best_params  = None
        best_sum     = None
        best_count   = None
        best_indices = None



        # ------------ test all component pairs in this round -------------
        for i, j in combinations(range(len(components)), 2):
            compA, compB = components[i], components[j]
            avgA = compA['sum'] / np.maximum(compA['count'], 1.0)
            avgB = compB['sum'] / np.maximum(compB['count'], 1.0)

            # use min (width) to keep x-range tight
            w_pair = min(avgA.shape[1], avgB.shape[1])
            dx_max_pair = w_pair - crop_w                # width matters for X search
            # dx_max_pair = w_pair - crop_size
            x_sub = np.arange(-dx_max_pair, dx_max_pair+1, orig_x_step)

            result = align_pair(avgA, avgB, angles, x_sub, y_translations,
                                crop_hw=crop_hw, lowpass_sigma=lowpass,
                                metric=metric, flips=flips)
            

            pbar.update(1)
            
            if result is None:
                continue
            canvas_A, aligned_B, score, params, _ = result
            if score == -np.inf:          # alignment existed but overlap filter rejected all shifts
                continue
            if score <= best_score:
                continue

            # ---------- adopt this pair as current best ----------
            best_score, best_pair, best_params = score, (i, j), params

            out_shape = canvas_A.shape
            h, w = out_shape
            hA, wA = compA['sum'].shape
            off_y, off_x = h//2 - hA//2, w//2 - wA//2

            sumA = np.zeros(out_shape, dtype=compA['sum'].dtype)
            cntA = np.zeros(out_shape, dtype=compA['count'].dtype)
            sumA[off_y:off_y+hA, off_x:off_x+wA] = compA['sum']
            cntA[off_y:off_y+hA, off_x:off_x+wA] = compA['count']

            ang, dx, dy = params
            sumB = transform_image(compB['sum'],   ang, off_x+dx, off_y+dy, out_shape)
            cntB = transform_image(compB['count'], ang, off_x+dx, off_y+dy, out_shape)

            best_sum   = sumA + sumB
            best_count = cntA + cntB
            

            
            best_indices = compA['indices'] + compB['indices']

        # -- if no alignable pair in this round, stop assembly gracefully --
        if best_pair is None:
            pbar.close()
            print("\nNo further alignable pairs – assembly stops.")
            break


        # --- merge the selected pair ---        
        # Here, crop to desired shape (keep y centered)
        # target_y = images[0].shape[0]  # or set as needed
        # merged_sum = symmetric_crop(best_sum, (target_y, best_sum.shape[1]))
        # merged_count = symmetric_crop(best_count, (target_y, best_count.shape[1]))
        
        # merged_sum = crop_to_content_symmetric_y(best_sum)
        # merged_count = crop_to_content_symmetric_y(best_count)  
        
        # merged_sum, merged_count = crop_to_content_pair_symmetric_y(best_sum, best_count)
        
        merged_sum, merged_count = crop_to_content_pair_fixed_y(
                               best_sum, best_count, target_h=H0)


        
        # merged_sum, merged_count = crop_to_content_pair(best_sum, best_count)
        i, j = best_pair
                
        for idx in sorted((i, j), reverse=True):
            del components[idx]
        components.append({'sum': merged_sum,
                           'count': merged_count,
                           'indices': best_indices})

        history.append({'step': step,
                        'merged_indices': best_indices,
                        'score': best_score,
                        'params': best_params})
        step += 1


    pbar.close()


    # ---------- final output (may be >1 component if stopped early) ----------
    if len(components) == 1:
        final_sum  = components[0]['sum']
        final_cnt  = components[0]['count']
        final_comp = np.zeros_like(final_sum)
        m = final_cnt > 0
        final_comp[m] = final_sum[m] / final_cnt[m]
        final_idx = components[0]['indices']

    else:  # pad to same height, then h-stack
        # 1) determine max height
        heights = [c['sum'].shape[0] for c in components]
        Hmax    = max(heights)

        padded = []
        final_idx = []
        for c in components:
            img = c['sum'] / np.maximum(c['count'], 1.0)
            h, w = img.shape
            top  = (Hmax - h) // 2
            bot  = Hmax - h - top
            img_pad = np.pad(img, ((top, bot), (0, 0)), mode='constant')
            padded.append(img_pad)
            final_idx += c['indices']

        final_comp = np.hstack(padded)

    return final_comp, final_idx, history









def main():
    parser = argparse.ArgumentParser(description="Progressive assembly of images from a folder or an mrcs stack.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str, help="Folder containing 2D class average images")
    group.add_argument("--mrcs", type=str, help="Input .mrcs stack")
    parser.add_argument("--output", type=str, default="composite_stitched.png", help="Output image filename")
    parser.add_argument("--mask-radius", type=float, default=0.9, help="Radius fraction for circular mask")
    parser.add_argument("--angles", type=float, nargs=3, metavar=('START', 'END', 'STEP'),
                        default=[-5.0, 5.0, 1.0], help="Rotation angle range in degrees: start end step (default: 0 0 1)")
    parser.add_argument("--xshifts", type=int, nargs=3, metavar=('START', 'END', 'STEP'),
                        default=None, help="Translation shift range (pixels): start end step (default: 0 0 1)")
    parser.add_argument("--yshifts", type=int, nargs=3, metavar=('START', 'END', 'STEP'),
                        default=None, help="Translation shift range (pixels): start end step (default: 0 0 1)")

    # parser.add_argument("--cc-mask-radius", type=int, default=60, 
    #                     help="Radius (pixels) of the central circular region used for cross-correlation")

    parser.add_argument(
        "--cc-window", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
        default=[60, 30],                       # e.g. 60 px tall, 30 px wide
        help="Size (pixels) of the rectangular region used for the cross-"
             "correlation window: HEIGHT  WIDTH.  The window is centred on "
             "the overlap region each time.  Default: 60×30."
)
    parser.add_argument("--lowpass", type=int, default=0, help="Lowpass filter for local alignment")
    parser.add_argument("--prealign", action="store_true", help="Pre-align each class average")
    parser.add_argument("--prealign-plot", action="store_true", help="Show pre-alignment for each class average")
    parser.add_argument("--class-ids", type=int, nargs="+", default=None,                        
        help="For MRCS input: indices (1-based) of class averages to use, e.g., --class-ids 2 5 7 8"
    )
    parser.add_argument("--metric", choices=["ncc", "rmsd", "mae", "ssim", "mi"], default="rmsd",
                        help=("Similarity measure to maximise:\n"
          "  ncc  – normalised cross-correlation (higher = better)\n"
          "  rmsd – root-mean-square deviation   (lower = better)\n"
          "  mae  – mean absolute error          (lower = better)\n"
          "  ssim – structural similarity index  (higher = better)\n"
          "  mi   – mutual information           (higher = better)"))
    parser.add_argument("--flips",
        choices=["none", "vflips", "hflips", "all"],
        default="all",
        help=("Which flipped copies of the second class to test:\n"
              "  none             – only the original orientation\n"
              "  vflips           – original + vertical-flip (up/down) (for asymmetric fibrils)\n"
              "  hflips           – original + horizontal-flip (up/down)\n"              
              "  all            – original + vertical- and horizontal-flip\n")
    )

    # parser.add_argument("--min-score", type=float, default=None, help="Minimum score to continue merging")
    args = parser.parse_args()

    # Read images
    if args.images:
        images, image_files = read_images_from_folder(args.images,mask_radius=args.mask_radius)
    elif args.mrcs:
        images, image_files = read_images_from_mrcs(args.mrcs, mask_radius=args.mask_radius,  class_ids=args.class_ids)
    else:
        print("Error: Must specify either --images or --mrcs")
        sys.exit(1)

    if args.prealign:
        print("Pre-aligning class averages (horizontal fibril)...")
        if args.prealign_plot:
            images = [prealign_fibril(img, plot=True) for img in images]
        else:
            images = [prealign_fibril(img, plot=False) for img in images]

    # for i, img in enumerate(images):
    #     plt.imshow(img, cmap='gray')
    #     plt.title(f"Input image {i}")
    #     plt.show()
 

    # Use image size for smart default shifts if user supplied no range
    box_height, box_width = images[0].shape
    # crop = args.cc_mask_radius
    crop_h, crop_w = args.cc_window        # ints




    # Set xshifts (x_translations)
    if args.xshifts is None:
        # xshift_max = int((2 / 3) * box_width)
        # xshift_max  = max(0, box_width - crop)
        xshift_max = max(0, box_width - crop_w)   # <- WAS “crop”

        xshift_step = max(1, int(box_width / 30))
        xshift_start, xshift_end = -xshift_max, xshift_max
        x_translations = np.arange(xshift_start, xshift_end + 1, xshift_step)
        # print(f"Auto xshifts: {xshift_start} to {xshift_end} (step {xshift_step})")
    else:
        xshift_start, xshift_end, xshift_step = args.xshifts
        if xshift_step < 1:
            xshift_step = 1
        x_translations = np.arange(xshift_start, xshift_end + xshift_step, xshift_step)
    
    # Set yshifts (y_translations)
    if args.yshifts is None:
        # yshift_max = int(0.1 * box_height)
        # yshift_step = max(1, int(box_height / 30))
        # yshift_start, yshift_end = -yshift_max, yshift_max
        # y_translations = np.arange(yshift_start, yshift_end + 1, yshift_step)
        yshift_start = -10
        yshift_end = 10 
        yshift_step = 2 
        y_translations = np.arange(yshift_start, yshift_end + 1, yshift_step)
        # print(f"Auto yshifts: {yshift_start} to {yshift_end} (step {yshift_step})")
    else:
        yshift_start, yshift_end, yshift_step = args.yshifts
        if yshift_step < 1:
            yshift_step = 1
        y_translations = np.arange(yshift_start, yshift_end + yshift_step, yshift_step)

    ang_start, ang_end, ang_step = args.angles
    if (ang_step < 1):
        ang_step = 1
    angles = np.arange(ang_start, ang_end + ang_step, ang_step)        

    # xshift_start, xshift_end, xshift_step = args.xshifts
    # if (xshift_step < 1):
    #     xshift_step = 1
    # yshift_start, yshift_end, yshift_step = args.yshifts
    # if (yshift_step < 1):
    #     yshift_step = 1
    # x_translations = np.arange(xshift_start, xshift_end + xshift_step, xshift_step)
    # y_translations = np.arange(yshift_start, yshift_end + yshift_step, yshift_step)
    
    
    if (args.lowpass == None):
        lowpass = 0.0
    else:
        lowpass = args.lowpass

    # composite, order, history = progressive_assembly(
    #     images, angles, x_translations, y_translations, crop_size=args.cc_mask_radius, 
    #     min_score=None, lowpass=lowpass)

    composite, order, history = progressive_assembly_new(
        images, angles, x_translations, y_translations, 
        crop_hw=(crop_h, crop_w),          # <-- one tuple now
        # crop_size=args.cc_mask_radius, 
         lowpass=lowpass, metric=args.metric, flips=args.flips)



    comp = composite  # your output 2D numpy array (float32)
    comp_min, comp_max = np.min(comp), np.max(comp)
    if comp_max > comp_min:  # avoid division by zero
        comp_norm = (comp - comp_min) / (comp_max - comp_min)
    else:
        comp_norm = comp - comp_min  # will be all zeros

    print("Composite shape (Y, X):", comp.shape)
    print("Y (height):", comp.shape[0], "pixels")
    print("X (width):", comp.shape[1], "pixels")


    # Output composite
    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext in [".png", ".tif", ".tiff"]:
        cv2.imwrite(args.output, (comp_norm * 255).astype(np.uint8))
    elif out_ext in [".mrc", ".mrcs"]:
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(composite.astype(np.float32))
    else:
        print(f"Unsupported output extension: {out_ext}, use .png, .tif, .mrc, or .mrcs")
        sys.exit(1)

    print("Order of merged images (indices):", order)
    print(f"Composite saved as {args.output}")

if __name__ == "__main__":
    main()
