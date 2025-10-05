import pickle
import numpy as np
import os
import fnmatch

def boxtransit_vectorized(times, period, dur, t0=0.0, alpha=1.0):
    """
    Vectorized box-transit function for multiple light curves.
    
    Args:
        times : array of shape (n_bins,)
        period : array of shape (n_lc,)
        dur : array of shape (n_lc,)
        t0 : array of shape (n_lc,) or scalar
        alpha : array of shape (n_lc,)
        
    Returns:
        array of shape (n_lc, n_bins)
    """
    n_lc = len(period)
    n_bins = len(times)
    
    times_2d = np.tile(times, (n_lc, 1))           # shape (n_lc, n_bins)
    period_2d = period[:, np.newaxis]              # shape (n_lc, 1)
    dur_2d = dur[:, np.newaxis]                    # shape (n_lc, 1)
    t0_2d = t0 if np.isscalar(t0) else t0[:, np.newaxis]
    alpha_2d = alpha[:, np.newaxis]
    
    inside = ((times_2d - t0_2d + dur_2d/2) % period_2d) <= dur_2d
    return -alpha_2d * inside.astype(float)


def generate_transit_training_data_vectorized(
    lc_filepaths,
    noise,
    npz_output="training_data_vectorized_all.npz",
    dt_min=30,
    max_days=90,
    period_range=(1, 45),
    duration_hours=(1, 16),
    depth_range=(0.0001, 0.001),
    seed=None
):
    """
    Generate one transit per light curve (vectorized) and save to a single .npz
    """
    rng = np.random.default_rng(seed)
    
    all_flux_normal = []
    all_flux_with_planet = []
    periods = []
    durations = []
    depths = []
    n_bins = None
    time_binned = None
    
    # -----------------------------
    # Process each light curve
    # -----------------------------
    for lc_filepath in lc_filepaths:
        with open(lc_filepath, "rb") as f:
            lc_norm_data = pickle.load(f)
        
        cadence = lc_norm_data["cadence"]
        flux = lc_norm_data["processed_flux"]
        
        # Convert cadence to days and baseline 0
        cadence_days = (cadence - cadence.min()) * (2 / 1440)
        mask = cadence_days <= max_days
        days_90 = cadence_days[mask]
        flux_90 = flux[mask] - 1.0
        
        # Bin light curve
        dt_days = dt_min / (24 * 60)
        n_bins_tmp = int(max_days / dt_days)
        
        bin_indices = np.floor(days_90 / dt_days).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins_tmp-1)
        
        flux_sums = np.zeros(n_bins_tmp)
        counts = np.zeros(n_bins_tmp)
        
        valid_mask = ~np.isnan(flux_90)
        np.add.at(flux_sums, bin_indices[valid_mask], flux_90[valid_mask])
        np.add.at(counts, bin_indices[valid_mask], 1)
        
        flux_binned = np.full(n_bins_tmp, np.nan)
        nonzero = counts > 0
        flux_binned[nonzero] = flux_sums[nonzero] / counts[nonzero]
        
        if n_bins is None:
            n_bins = n_bins_tmp
            time_binned = (np.arange(n_bins) + 0.5) * dt_days
        else:
            assert n_bins == n_bins_tmp, "All light curves must have the same binning!"
        
        all_flux_normal.append(flux_binned)
    
    all_flux_normal = np.array(all_flux_normal)  # shape (n_lc, n_bins)
    n_lc = len(lc_filepaths)
    
    # -----------------------------
    # Add noise to light curves
    # -----------------------------
    print("all flux", np.nanmax(all_flux_normal), np.nanmin(all_flux_normal), np.nanstd(all_flux_normal))
    scaled_noise = (noise / (noise.max() - noise.min())) - noise.mean()
    scaled_noise = scaled_noise * np.nanstd(all_flux_normal) + np.nanmean(all_flux_normal)
    print("scaled noise", scaled_noise.max(), scaled_noise.min(), scaled_noise.std())

    all_flux_normal = np.nan_to_num(all_flux_normal, nan=0.1*scaled_noise)
    print("all flux after filling nans", np.max(all_flux_normal), np.min(all_flux_normal), np.std(all_flux_normal))

    all_flux_normal += 0.002 * noise
    print("noise flux", noise.max(), noise.min(), noise.std())
    print("all flux after noise", np.nanmax(all_flux_normal), np.nanmin(all_flux_normal), np.nanstd(all_flux_normal))

    # -----------------------------
    # Generate random transit parameters for each light curve
    # -----------------------------
    periods = rng.uniform(period_range[0], period_range[1], size=n_lc)
    durations = rng.uniform(duration_hours[0]/24, duration_hours[1]/24, size=n_lc)
    depths = rng.uniform(depth_range[0], depth_range[1], size=n_lc)
    
    # Generate all transits (vectorized)
    transits = boxtransit_vectorized(time_binned, periods, durations, alpha=depths)
    
    # Apply transits
    all_flux_with_planet = all_flux_normal + transits
    
    # -----------------------------
    # Save everything to a single NPZ
    # -----------------------------
    np.savez_compressed(
        npz_output,
        time=time_binned,
        flux_normal=all_flux_normal,
        flux_with_planet=all_flux_with_planet,
        period=periods,
        duration=durations,
        depth=depths
    )
    
    print(f"Saved {n_lc} light curves with one transit each to {npz_output}")
    return npz_output


# -----------------------------
# List all light curves
# -----------------------------
def list_files(directory, filetype):
    return [os.path.join(directory, f) for f in os.listdir(directory) if fnmatch.fnmatch(f, filetype)]

lightcurve_files = list_files("TESS_data/processed_data", '*_normalized.p')
print(len(lightcurve_files))

for i in range(3):
    lightcurve_files += lightcurve_files
print(len(lightcurve_files))

# -----------------------------
# Generate noise for all light curves
# -----------------------------

import numpy as np
from noise_generator import powerlaw_psd, simulate_from_psd
import matplotlib.pyplot as plt
NUM = 1000
N = 4320
FS = 1/120
ALPHA = 2.0
VAR = 2.0
SEED = 0

rng = np.random.default_rng(SEED)
_, S = powerlaw_psd(N, FS, alpha=ALPHA, variance=VAR)
noise_data = np.vstack([
    simulate_from_psd(S, N, FS, rng=np.random.default_rng(rng.integers(0, 2**63 - 1)))
    for _ in range(len(lightcurve_files))
])

# -----------------------------
# Apply vectorized to all light curves
# -----------------------------

generate_transit_training_data_vectorized(
    lightcurve_files,
    noise=noise_data,
    npz_output="training_data_vectorized_all.npz",
    seed=42
)
