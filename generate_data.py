import pickle
import numpy as np

def boxtransit_vectorized(times, period, dur, t0=0.0, alpha=1.0):
    """
    Vectorized box-transit function.
    
    Args:
        times : array of shape (..., n_times)
        period : scalar or array broadcastable to times.shape
        dur : scalar or array broadcastable to times.shape
        t0 : scalar or array broadcastable to times.shape
        alpha : transit depth (default 1)
    
    Returns:
        array of same shape as times, with 0 outside transit and -alpha during transit
    """
    # inside transit: ((times - t0 + dur/2) % period) <= dur
    inside = ((times - t0 + dur/2) % period) <= dur
    return -alpha * inside.astype(float)

def generate_transit_training_data(
    lc_filepath,
    npz_output="training_data_vectorized.npz",
    dt_min=30,
    max_days=90,
    period_range=(1, 45),
    duration_hours=(1, 16),
    duration_step=0.5,
    n_depths=2,
    depth_range=(0.0001, 0.001),
    seed=42
):
    """
    Generate vectorized transit training data and save to .npz.

    Parameters:
    -----------
    lc_filepath : str
        Path to light curve .p file (pickle) containing 'cadence' and 'processed_flux'.
    npz_output : str
        Path for the output .npz file.
    dt_min : float
        Bin size in minutes.
    max_days : float
        Maximum number of days of lightcurve to use.
    period_range : tuple
        Min and max period in days (inclusive).
    duration_hours : tuple
        Min and max transit duration in hours.
    duration_step : float
        Duration step in hours.
    n_depths : int
        Number of random depth values per period-duration pair.
    depth_range : tuple
        Min and max transit depth (fraction of flux).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    str
        Filepath of saved .npz file.
    """
    # -----------------------------
    # Load light curve
    # -----------------------------
    with open(lc_filepath, "rb") as f:
        lc_norm_data = pickle.load(f)
    
    cadence = lc_norm_data["cadence"]
    flux = lc_norm_data["processed_flux"]
    
    # Convert cadence to days and baseline 0
    cadence_days = (cadence - cadence.min()) * (2 / 1440)
    mask = cadence_days <= max_days
    days_90 = cadence_days[mask]
    flux_90 = flux[mask] - 1.0  # baseline 0
    
    # -----------------------------
    # Bin light curve
    # -----------------------------
    dt_days = dt_min / (24 * 60)
    n_bins = int(max_days / dt_days)
    
    bin_indices = np.floor(days_90 / dt_days).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins-1)
    
    flux_sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    valid_mask = ~np.isnan(flux_90)
    valid_bins = bin_indices[valid_mask]
    valid_flux = flux_90[valid_mask]
    
    np.add.at(flux_sums, valid_bins, valid_flux)
    np.add.at(counts, valid_bins, 1)
    
    flux_binned = np.full(n_bins, np.nan)
    nonzero = counts > 0
    flux_binned[nonzero] = flux_sums[nonzero] / counts[nonzero]
    
    time_binned = (np.arange(n_bins) + 0.5) * dt_days  # center of each bin
    
    # -----------------------------
    # Define period, duration, depth parameters
    # -----------------------------
    periods_days = np.arange(period_range[0], period_range[1]+1)
    durations_days = np.arange(duration_hours[0]/24, duration_hours[1]/24 + 1e-6, duration_step/24)
    
    # Vectorized combinations
    P, D, R = np.meshgrid(periods_days, durations_days, np.arange(n_depths), indexing="ij")
    periods_vec = P.ravel()
    durations_vec = D.ravel()
    n_samples = len(periods_vec)
    
    # Random depths
    np.random.seed(seed)
    depths_vec = np.random.uniform(depth_range[0], depth_range[1], n_samples)
    
    # -----------------------------
    # Generate transits
    # -----------------------------
    scaled_box_matrix = boxtransit_vectorized(
        time_binned[np.newaxis, :],        # shape (1, n_bins)
        period=periods_vec[:, np.newaxis], # shape (n_samples, 1)
        dur=durations_vec[:, np.newaxis],  # shape (n_samples, 1)
        t0=0.0,
        alpha=depths_vec[:, np.newaxis]    # shape (n_samples, 1)
    )
    
    # -----------------------------
    # Prepare training arrays
    # -----------------------------
    flux_normal = np.tile(flux_binned, (n_samples, 1))
    flux_with_planet = flux_normal + scaled_box_matrix
    time_array = np.tile(time_binned, (n_samples, 1))
    
    # -----------------------------
    # Save to NPZ
    # -----------------------------
    np.savez_compressed(
        npz_output,
        time=time_array,
        flux_normal=flux_normal,
        flux_with_planet=flux_with_planet,
        period=periods_vec,
        duration=durations_vec,
        depth=depths_vec
    )
    
    print(f"Saved {n_samples} training samples with {n_bins} time points each to {npz_output}")
    return npz_output

npz_file = generate_transit_training_data("lc_norm.p") # path of normalized light curve pickle file