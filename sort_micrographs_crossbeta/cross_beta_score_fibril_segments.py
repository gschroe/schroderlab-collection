#!/usr/bin/env python3
"""
Calculate per-fibril cross-beta scores from RELION particle data.

For each fibril (identified by micrograph + helical tube ID), the script:
  1. Computes the averaged power spectrum of all segment images
  2. Rotates the PS so the fibril axis is horizontal (using rlnAnglePsiPrior)
  3. Measures the signal in the section of the cross-beta ring around the psi-prior (~4.75 A) relative to mean background signal
  4. Writes a scored .star file with a per_fibril_cross_beta_score column

Optionally thresholds and/or maps scores to a second particle set.
"""

import argparse
import sys
from pathlib import Path

import mrcfile
import numpy as np
from numpy import fft
import pandas as pd
import starfile
from skimage.transform import rotate
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_stack_name(rlnImageName: str) -> str:
    return rlnImageName.split("@")[1]


def get_stk_index(rlnImageName: str) -> int:
    return int(rlnImageName.split("@")[0]) - 1


def read_particle_stk(part_stk_fpath: Path):
    with mrcfile.open(part_stk_fpath) as f:
        part_stk = f.data
        angpix = float(f.voxel_size.x)
    return part_stk, angpix


def load_filament_stack(relion_project_dir: Path, particle_stack_mrc: str,
                        start_idx: int, end_idx: int) -> np.ndarray:
    part_stk_mrc_fpath = relion_project_dir / particle_stack_mrc
    if not part_stk_mrc_fpath.exists():
        raise FileNotFoundError(f"Particle stack not found: {part_stk_mrc_fpath}")
    with mrcfile.open(part_stk_mrc_fpath) as f:
        stk = f.data[start_idx:end_idx + 1]
    return stk


def padded_powerspectrum(particle_img: np.ndarray, angpix: float):
    """Compute the power spectrum of a particle image with 2x zero-padding."""
    h, w = particle_img.shape
    pad_h, pad_w = h // 2, w // 2
    particle_padded = np.pad(particle_img,
                             ((pad_h, pad_h), (pad_w, pad_w)),
                             mode="constant", constant_values=0)
    N_pad = particle_padded.shape[0]

    ps = np.fft.fft2(particle_padded)
    ps = np.abs(ps) ** 2
    ps = np.fft.fftshift(ps)

    k = fft.fftfreq(N_pad, angpix)
    k = fft.fftshift(k)
    return ps, k


def generate_psi_mask(kx: np.ndarray, ky: np.ndarray,
                      psi_range: float = 40) -> np.ndarray:
    """Boolean mask selecting Fourier pixels along the fibril axis (horizontal)."""
    psi_arr = np.rad2deg(np.arctan2(ky, kx))

    psi_min = -psi_range / 2
    psi_max = +psi_range / 2
    pos_x_half_mask = (psi_arr >= psi_min) & (psi_arr <= psi_max)
    quad2_mask = psi_arr >= 180 - psi_range / 2
    quad3_mask = psi_arr <= -180 + psi_range / 2

    return pos_x_half_mask | quad2_mask | quad3_mask


def generate_power_spectrum_mask(k: np.ndarray) -> np.ndarray:
    """Create a mask that is True where the cross-beta signal is *excluded*."""
    kx, ky = np.meshgrid(k, k[::-1])
    k_norm = np.sqrt(kx ** 2 + ky ** 2)

    k_cb_center = 1 / 4.75
    rel_width = 0.1
    k_min = (1 - rel_width / 2) * k_cb_center
    k_max = (1 + rel_width / 2) * k_cb_center
    cb_mask = (k_norm >= k_min) & (k_norm <= k_max)

    psi_mask = generate_psi_mask(kx, ky)
    return ~(cb_mask & psi_mask)


def calculate_per_fibril_cross_beta_score(fib_ps: np.ndarray,
                                          psi_prior: float,
                                          k: np.ndarray,
                                          k_min: float = 1 / 25) -> float:
    """Cross-beta score = mean signal in cross-beta ring / mean background."""
    fib_ps_halign = rotate(fib_ps.copy(), -psi_prior,
                           mode="constant", cval=fib_ps.mean())

    kx, ky = np.meshgrid(k, k[::-1])
    k_norm = np.sqrt(kx ** 2 + ky ** 2)
    dc_mask = k_norm <= k_min

    cross_beta_mask = generate_power_spectrum_mask(k)

    # Background: everything except DC and the cross-beta region
    bkgrd_mask = dc_mask | ~cross_beta_mask
    mean_bgrd = np.ma.masked_array(fib_ps_halign, mask=bkgrd_mask).mean()

    # Signal in cross-beta ring
    fib_ps_masked = np.ma.masked_array(fib_ps_halign, mask=cross_beta_mask)
    cb_score = float(fib_ps_masked.mean() / mean_bgrd)
    return cb_score


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def add_fibril_columns(part_df: pd.DataFrame) -> pd.DataFrame:
    """Add fibril_hash, fibril_id, particle_stack_mrc, stk_index columns."""
    part_df = part_df.copy()
    part_df["fibril_hash"] = (part_df["rlnMicrographName"] + "_"
                              + part_df["rlnHelicalTubeID"].astype(str))
    part_df["fibril_id"] = pd.factorize(part_df["fibril_hash"])[0]
    part_df["particle_stack_mrc"] = part_df["rlnImageName"].apply(get_stack_name)
    part_df["stk_index"] = part_df["rlnImageName"].apply(get_stk_index)
    return part_df


def get_per_fibril_psi_priors(part_df: pd.DataFrame) -> np.ndarray:
    """Return one psi prior per fibril_id (ordered by fibril_id)."""
    psi_priors = []
    for _fid, fibril_df in part_df.groupby("fibril_id"):
        psi = fibril_df["rlnAnglePsiPrior"].unique()
        if len(psi) != 1:
            raise ValueError(f"Expected one unique psi prior per fibril, "
                             f"got {len(psi)} for fibril {_fid}")
        psi_priors.append(psi[0])
    return np.array(psi_priors)


def compute_fibril_mean_ps(part_df: pd.DataFrame,
                           fibril_df: pd.DataFrame,
                           relion_project_dir: Path,
                           angpix: float) -> np.ndarray:
    """Compute the averaged power spectrum for a single fibril."""
    start_idx = fibril_df["stk_index"].min()
    end_idx = fibril_df["stk_index"].max()
    part_stk_mrc = fibril_df["particle_stack_mrc"].iloc[0]

    fibril_stack = load_filament_stack(relion_project_dir, part_stk_mrc,
                                       start_idx, end_idx)
    ps_list = [padded_powerspectrum(p, angpix)[0] for p in fibril_stack]
    return np.mean(ps_list, axis=0)


def compute_scores_streaming(part_df: pd.DataFrame,
                             relion_project_dir: Path,
                             angpix: float,
                             k: np.ndarray) -> np.ndarray:
    """Compute cross-beta scores on-the-fly without storing all PS in memory."""
    num_fibrils = part_df["fibril_id"].nunique()
    psi_priors = get_per_fibril_psi_priors(part_df)
    scores = np.empty(num_fibrils)

    for fid, fibril_df in tqdm(part_df.groupby("fibril_id"),
                               total=num_fibrils,
                               desc="Computing PS & scoring fibrils"):
        mean_ps = compute_fibril_mean_ps(part_df, fibril_df,
                                         relion_project_dir, angpix)
        scores[fid] = calculate_per_fibril_cross_beta_score(
            mean_ps, psi_priors[fid], k)

    return scores


def compute_scores_cached(part_df: pd.DataFrame,
                          relion_project_dir: Path,
                          angpix: float,
                          k: np.ndarray,
                          cache_path: Path) -> np.ndarray:
    """Compute scores using a .npy cache for the per-fibril averaged PS.

    If the cache exists, power spectra are loaded memory-mapped.
    Otherwise they are computed, saved to disk, and then scored.
    """
    num_fibrils = part_df["fibril_id"].nunique()
    psi_priors = get_per_fibril_psi_priors(part_df)

    if cache_path.exists():
        print(f"Loading cached power spectra from {cache_path}")
        fibril_ps_arr = np.load(cache_path, mmap_mode="r")
    else:
        print(f"Computing averaged power spectra for {num_fibrils} fibrils ...")
        fibril_powerspectra = []
        for _fid, fibril_df in tqdm(part_df.groupby("fibril_id"),
                                    total=num_fibrils,
                                    desc="Computing per-fibril PS"):
            mean_ps = compute_fibril_mean_ps(part_df, fibril_df,
                                             relion_project_dir, angpix)
            fibril_powerspectra.append(mean_ps)

        fibril_ps_arr = np.array(fibril_powerspectra)
        np.save(cache_path, fibril_ps_arr)
        print(f"Saved power spectra cache to {cache_path}")
        del fibril_powerspectra

    scores = np.empty(num_fibrils)
    for i in tqdm(range(num_fibrils), desc="Scoring fibrils"):
        scores[i] = calculate_per_fibril_cross_beta_score(
            fibril_ps_arr[i], psi_priors[i], k)

    return scores


def write_scored_star(part_data_dict: dict, part_df: pd.DataFrame,
                      out_path: Path) -> None:
    out = part_data_dict.copy()
    out["particles"] = part_df
    starfile.write(out, out_path, overwrite=True)
    print(f"Wrote {len(part_df)} particles to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "relion_project_dir", type=Path,
        help="Path to the RELION project directory.",
    )
    parser.add_argument(
        "particles_star_fpath", type=Path,
        help="Path to the particles .star file.",
    )
    parser.add_argument(
        "--cross-beta-min-threshold", type=float, default=None,
        help="If set, write an additional .star file containing only "
             "particles whose per-fibril cross-beta score >= this value.",
    )
    parser.add_argument(
        "--second-particles-star", type=Path, default=None,
        help="Path to a second particles .star file. Scores are mapped via "
             "fibril_hash. If --cross-beta-min-threshold is also given, a "
             "thresholded version of this file is written as well.",
    )
    parser.add_argument(
        "--ps-cache", type=Path, default=None,
        help="Path for a power-spectra .npy cache file. When given, all "
             "per-fibril averaged PS are stored on disk and reloaded via "
             "memory-mapping on subsequent runs (useful when trying "
             "different thresholds). Without this flag, scores are computed "
             "on-the-fly and no PS are kept in memory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    relion_dir: Path = args.relion_project_dir
    star_path: Path = args.particles_star_fpath
    threshold: float | None = args.cross_beta_min_threshold
    second_star_path: Path | None = args.second_particles_star
    ps_cache: Path | None = args.ps_cache

    # ------------------------------------------------------------------
    # 1. Read primary particle data
    # ------------------------------------------------------------------
    print(f"Reading {star_path} ...")
    part_data_dict = starfile.read(star_path)
    part_df = part_data_dict["particles"]
    part_df = add_fibril_columns(part_df)

    num_fibrils = part_df["fibril_id"].nunique()
    print(f"  {len(part_df)} particles, {num_fibrils} fibrils")

    # ------------------------------------------------------------------
    # 2. Determine pixel size from first particle stack
    # ------------------------------------------------------------------
    first_stk_fname = get_stack_name(part_df["rlnImageName"].iloc[0])
    first_stk_fpath = relion_dir / first_stk_fname
    _, angpix = read_particle_stk(first_stk_fpath)
    print(f"  Pixel size: {angpix:.4f} A/px")

    # ------------------------------------------------------------------
    # 3. Spatial frequency array (from a single example particle)
    # ------------------------------------------------------------------
    example_stk = load_filament_stack(
        relion_dir,
        part_df["particle_stack_mrc"].iloc[0],
        part_df["stk_index"].iloc[0],
        part_df["stk_index"].iloc[0],
    )
    _, k = padded_powerspectrum(example_stk[0], angpix)

    # ------------------------------------------------------------------
    # 4. Compute cross-beta scores
    # ------------------------------------------------------------------
    if ps_cache is not None:
        cb_scores = compute_scores_cached(
            part_df, relion_dir, angpix, k, ps_cache)
    else:
        cb_scores = compute_scores_streaming(
            part_df, relion_dir, angpix, k)

    part_df["per_fibril_cross_beta_score"] = part_df["fibril_id"].map(
        lambda fid: cb_scores[fid])

    # ------------------------------------------------------------------
    # 5. Write scored primary star file
    # ------------------------------------------------------------------
    out_scored = star_path.parent / f"{star_path.stem}-cross_beta_scored.star"
    write_scored_star(part_data_dict, part_df, out_scored)

    # ------------------------------------------------------------------
    # 6. Optionally threshold primary particles
    # ------------------------------------------------------------------
    if threshold is not None:
        df_ths = part_df[part_df["per_fibril_cross_beta_score"] >= threshold]
        out_ths = star_path.parent / (
            f"{star_path.stem}-cross_beta_scored-ths{threshold}.star")
        write_scored_star(part_data_dict, df_ths, out_ths)

    # ------------------------------------------------------------------
    # 7. Optionally process second particle set
    # ------------------------------------------------------------------
    if second_star_path is not None:
        print(f"\nReading second particle set: {second_star_path} ...")
        part_data_dict_2 = starfile.read(second_star_path)
        part_df_2 = part_data_dict_2["particles"]
        part_df_2 = add_fibril_columns(part_df_2)

        # Map scores via fibril_hash
        fibril_score_lookup = (part_df
                               .groupby("fibril_hash")
                               ["per_fibril_cross_beta_score"]
                               .first())
        part_df_2["per_fibril_cross_beta_score"] = (
            part_df_2["fibril_hash"].map(fibril_score_lookup))

        n_missing = part_df_2["per_fibril_cross_beta_score"].isna().sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} particles in the second set "
                  f"could not be matched by fibril_hash.")

        out_scored_2 = second_star_path.parent / (
            f"{second_star_path.stem}-cross_beta_scored.star")
        write_scored_star(part_data_dict_2, part_df_2, out_scored_2)

        if threshold is not None:
            df2_ths = part_df_2[
                part_df_2["per_fibril_cross_beta_score"] >= threshold]
            out_ths_2 = second_star_path.parent / (
                f"{second_star_path.stem}"
                f"-cross_beta_scored-ths{threshold}.star")
            write_scored_star(part_data_dict_2, df2_ths, out_ths_2)

    print("\nDone.")


if __name__ == "__main__":
    main()
