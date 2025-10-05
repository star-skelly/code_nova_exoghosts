"""
Aim: to process and download lightcurves from TESS for a list of TIC IDs.
- we have downloded 3 sectors of data for each target i.e. sectors 73, 74, 75, 76
- we would like to combine these lightcurves into a single lightcurve for each target
- normalize the lightcurves and do nonuniform fft
saved file format is a pickle of a tuple:
(lc_data, processed_lc_data, quality_data, time_data, cam_data, ccd_data, centroid_xy_data, pos_xy_corr)
"""

import numpy as np
import matplotlib.pyplot as plt
import finufft as nufft
import lightkurve as lk
import os
import glob
import pickle

######################
# import files from TESS/data
data_path = os.path.expanduser('~/TESS/data/')
os.chdir(data_path)
file_names=glob.glob(data_path+'*.p')
def summarize(data):
    # The script that created these .p files saved a tuple:
    # (lc_data, processed_lc_data, quality_data, time_data, cam_data, ccd_data, centroid_xy_data, pos_xy_corr)
    try:
        (lc_data, proc_data, quality_data, time_data,
         cam_data, ccd_data, centroid_xy_data, pos_xy_corr) = data
    except Exception as e:
        print("Unexpected pickle shape:", type(data), e)
        return None
    print("Sectors present:", sorted(lc_data.keys()))
    for s in sorted(lc_data.keys()):
        print(f" sector {s}: raw points={len(lc_data[s])}, processed points={len(proc_data[s])}, "
              f"time/cadence points={len(time_data[s])}, camera={cam_data[s]}, ccd={ccd_data[s]}")
    return {
        'lc_data': lc_data,
        'proc_data': proc_data,
        'quality_data': quality_data,
        'time': time_data,
        'cam': cam_data,
        'ccd': ccd_data,
        'centroid': centroid_xy_data,
        'pos_xy_corr': pos_xy_corr
    }
def plot_sector(data_dict, sector, ax_flux=None, ax_centroid=None):
    lc = data_dict['lc_data'][sector]
    proc = data_dict['proc_data'][sector]
    time = data_dict['time'][sector]   # currently cadence numbers (see note)
    cent = data_dict['centroid'][sector]  # [x_array, y_array]
    pos = data_dict['pos_xy_corr'][sector]   # [pos_corr_x, pos_corr_y]
    cam = data_dict['cam'][sector]
    ccd = data_dict['ccd'][sector]

    # Defensive: ensure arrays are numpy arrays
    lc = np.asarray(lc)
    proc = np.asarray(proc)
    time = np.asarray(time)
    xcent = np.asarray(cent[0])
    ycent = np.asarray(cent[1])
    posx = np.asarray(pos[0])
    posy = np.asarray(pos[1])

    # Flux plot
    if ax_flux is None:
        fig, (ax_flux, ax_centroid) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # ax_flux.plot(time, lc, '.', color='gray', alpha=0.6, markersize=3, label='SAP flux (raw)')
    ax_flux.plot(time, proc, '.', color='C1', alpha=0.8, markersize=3, label='PDCSAP flux (processed)')
    ax_flux.set_ylabel('Flux (counts)')
    ax_flux.set_title(f"Sector {sector}  camera={cam}  ccd={ccd}")
    ax_flux.legend(loc='best')
    ax_flux.grid(True)

    # # Centroid scatter and pos_corr
    # if ax_centroid is None:
    #     ax_centroid = plt.gca()
    # sc = ax_centroid.scatter(xcent, ycent, c=np.arange(len(xcent)), cmap='viridis', s=6)
    # ax_centroid.set_xlabel('Cadence number (or index)')
    # ax_centroid.set_ylabel('Centroid (pix)')
    # ax_centroid.set_title('Centroid XY (color=time index)')
    # plt.colorbar(sc, ax=ax_centroid, label='index (time order)')

    return ax_flux, ax_centroid
def combine_sectors(data_dict, sectors=None, sort_by_time=True):
    """
    Combine processed flux arrays across multiple sectors into single numpy arrays.

    Inputs:
      - data_dict: dict returned by `summarize()` in the earlier script (keys: 'proc','time','cam','ccd',...)
      - sectors: list-like of sector numbers to include. If None, will use sorted(data_dict['proc'].keys()).
      - sort_by_time: if True, final arrays are sorted by cadence/time values (np.argsort on combined 'time').
      
    Returns: tuple of numpy arrays:
      (proc_all, time_all, sector_label_all, cam_all, ccd_all)
    - proc_all: concatenated processed flux (float)
    - time_all: concatenated cadence numbers (or relative indices if map_to_relative=True)
    - sector_label_all: integer array with sector number for each sample
    - cam_all, ccd_all: integer arrays for camera/ccd per sample
    """
    if sectors is None:
        sectors = sorted(data_dict['proc_data'].keys())

    proc_list = []
    time_list = []
    sector_list = []
    cam_list = []
    ccd_list = []

    for s in sectors:
        if s not in data_dict['proc_data']:
            raise KeyError(f"Sector {s} not in data_dict['proc']. Available: {sorted(data_dict['proc'].keys())}")
        p = np.asarray(data_dict['proc_data'][s])
        # normalize each sector by its median
        p /= np.nanmedian(p)
        t = np.asarray(data_dict['time'][s])
        # Defensive sizes: flux/time must be same length
        if p.shape[0] != t.shape[0]:
            # try to handle if time array was saved as longer quality-masked array
            raise ValueError(f"Length mismatch for sector {s}: proc has {p.shape[0]}, time has {t.shape[0]}")

        proc_list.append(p)
        time_list.append(t)
        sector_list.append(np.full(p.shape, s, dtype=int))
        cam_list.append(np.full(p.shape, data_dict['cam'][s], dtype=int))
        ccd_list.append(np.full(p.shape, data_dict['ccd'][s], dtype=int))

    if len(proc_list) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

    proc_all = np.concatenate(proc_list)
    time_all = np.concatenate(time_list)
    sector_label_all = np.concatenate(sector_list)
    cam_all = np.concatenate(cam_list)
    ccd_all = np.concatenate(ccd_list)

    if sort_by_time:
        order = np.argsort(time_all)
        return proc_all[order], time_all[order], sector_label_all[order], cam_all[order], ccd_all[order]
    else:
        return proc_all, time_all, sector_label_all, cam_all, ccd_all

# output_fname = f"{file_name[-10:-2]}_normalized.p"
output_path = os.path.expanduser('~/TESS/processed_data/')
if not os.path.exists(output_path):
    os.makedirs(output_path)
# read in the list of TIC IDs
for i in range(len(file_names)):
# for i in [0]:
    file_name=file_names[i]
    output_fname=file_name.split('/')[-1] # get the file name without path
    print('Processing file: ', file_name)
    with open(file_name, 'rb') as f:
        data=pickle.load(f)
    
    summary_data = summarize(data)
    if summary_data is None:
        continue
    sectors = sorted(summary_data['lc_data'].keys())
    if len(sectors) == 0:
        print("No sectors found in this file.")
        continue
    # If many sectors, you can change behavior. We'll plot the first sector by default, and offer to plot all.
    # Plot each sector in a separate figure
    # for sector in sectors:
    #     fig = plt.figure(figsize=(11,6))
    #     ax1 = fig.add_subplot()
    #     plot_sector(summary_data, sector, ax_flux=ax1)
    #     plt.tight_layout()
    #     plt.show()
    
    # # stitch the lightcurves together
    # combine all sectors present
    lc_processed, cadence_all, sector_labels, cams, ccds = combine_sectors(summary_data, sort_by_time=True)
    print("Combined processed flux shape:", lc_processed.shape)
    print("Cadence range:", cadence_all.min(), cadence_all.max())
    #plot combined lightcurve
    plt.figure(figsize=(10,4))
    plt.plot(cadence_all, lc_processed, '.', markersize=2, alpha=0.6)
    plt.xlabel('cadence (or relative index)')
    plt.ylabel('PDCSAP flux (processed)')
    plt.title(f'Combined sectors processed flux for {output_fname[:-2]}')
    plt.grid(True)
    plt.savefig(output_path+output_fname[:-2]+'_combined_flux.png', dpi=300)
    plt.close()
    # # save the normalized lightcurve to a file
    # create a dictionary with normalized lightcurve and corresponding cadence
    lc_normalized = {
        'cadence': cadence_all,
        'processed_flux': lc_processed,
        'sector_labels': sector_labels,
        'cams': cams,
        'ccds': ccds
    }
    # save the dictionary as a pickle file
    
    output_file = os.path.join(output_path, output_fname[:-2]+'_normalized.p')
    with open(output_file, 'wb') as f:
        pickle.dump(lc_normalized, f)
    print(f'Saved normalized lightcurve to {output_file}')
    ###########################################
    # estimate the smoothed periodogram of the lightcurve |NUFFT(lc_normalized)|^2
    # do nonuniform fft on the normalized lightcurve
    cadence_all = np.asarray(cadence_all, dtype=np.float64)       # FINUFFT x: float64
    lc_complex = np.asarray(lc_processed, dtype=np.complex128)      # FINUFFT c: complex128 (double prec)
    # Set FFT size (higher = better frequency resolution)
    fft_size = 4096  # Try 2048, 4096, or 8192
    # Set precision (tolerance for FINUFFT)
    eps = 1e-6 #is good balance between speed and accuracy
    # Compute Type-1 NUFFT: non-uniform time domain → uniform frequency domain
    # finufft.nufft1d1(x, c, n, eps) where:
    #   x: non-uniform points (time_scaled)
    #   c: complex values at those points (flux_complex)
    #   n: number of Fourier modes (fft_size)
    #   eps: precision tolerance
    # output is complex array of length n (fft_size)
    fft_lcdata = nufft.nufft1d1(cadence_all,lc_complex, fft_size, eps=eps)
    # Calculate frequency resolution
    duration_days = np.nanmax(cadence_all) - np.nanmin(cadence_all)
    freq_resolution = 1.0 / duration_days
    nyquist_freq = len(cadence_all) / (2 * duration_days)
    ###########################################
    # plot the periodogram of the lightcurve
    plt.figure(figsize=(10,4))
    freqs = np.fft.rfftfreq(fft_size, d=duration_days/len(cadence_all))  # assuming unit spacing in cadence
    pxx_one_sided = (2.0 * (1 / duration_days) / fft_size) * (np.abs(fft_lcdata) ** 2)
    pxx_one_sided[0] = ((1 / duration_days) / fft_size) * (np.abs(fft_lcdata[0]) ** 2)
    if fft_size % 2 == 0: # even length
        pxx_one_sided[-1] = ((1 / duration_days) / fft_size) * (np.abs(fft_lcdata[-1]) ** 2) # Nyquist-like

    # Ensure freqs and pxx have the same length for plotting
    # plt.loglog(freqs[1:fft_size//2], pxx_one_sided[1:fft_size//2], color='C2')
    # # plt.semilogy(freqs[1:fft_size//2], pxx_one_sided[1:fft_size//2], color='C2')
    # plt.xlabel('Frequency (1/day)')
    # plt.ylabel('Power Spectral Density')
    # plt.title(f'Periodogram from NUFFT for {output_fname[:-2]}')
    # plt.grid(True)
    # plt.show()
    ######################
    #plot periodogram period vs power
    plt.figure(figsize=(10,4))
    periods = 1.0 / freqs[1:fft_size//2]
    plt.loglog(periods, pxx_one_sided[1:fft_size//2], color='C2')
    plt.xlabel('Period (days)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Periodogram (Period vs Power) from NUFFT for {output_fname[:-2]}')
    plt.savefig(output_path+output_fname[:-2]+'_periodogram.png', dpi=300)
    plt.grid(True)
    plt.close()
    ##############################################
    # save the fft_lcdata to a file
    output_fname_fft = f"{output_fname[:-2]}_nufft.p"
    output_file_fft = os.path.join(output_path, output_fname_fft)
    with open(output_file_fft, 'wb') as f:
        pickle.dump(pxx_one_sided, f)
    print(f'Saved NUFFT of normalized lightcurve to {output_file_fft}')

    


