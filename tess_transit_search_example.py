import numpy as np
import pickle
from nufft_detector import matched_filter_snr_nufft
from noise_generator import boxtransit, periodogram_one_sided, simulate_cyclostationary, powerlaw_psd, detrend, centered_phase_fold, smoothed_periodogram, smoothed_periodogram_with_gaps
import matplotlib.pyplot as plt
from util import Progress

tid = 232616284
# tid = 229644321
with open("TESS_data/processed_data/%s_normalized.p" % tid, "rb") as f:   # "rb" = read in binary mode
    lc_norm_data = pickle.load(f)

cadence = lc_norm_data["cadence"]
flux = lc_norm_data["processed_flux"]
lc = flux - 1
N = len(flux)
fs = 1

true_period = 30*9*24
true_dur = 30*6
alpha = 0.002
print ('true period:', true_period, 'true dur:', true_dur)
template_true = boxtransit(cadence, period=true_period, dur=true_dur, t0=0, alpha=alpha)

# PSDs on rFFT grid used for NUFFT (smoothed)
#freqs, pxx_est = periodogram_one_sided(lc, fs)
freqs, pxx_est = smoothed_periodogram_with_gaps(lc, fs, kernel='hann', width=31, min_len=256)
# One-sided white PSD level matching this normalization: 2 * var / fs
sigma2 = float(np.nanvar(lc))
pxx_white = np.full_like(pxx_est, 2.0 * sigma2 / fs)

# Visualize estimated vs white PSD
mask = freqs > 0.0
plt.figure(figsize=(6, 4))
plt.loglog(freqs[mask], pxx_est[mask], label="Estimated PSD")
plt.loglog(freqs[mask], pxx_white[mask], label="White PSD (2*var/fs)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Hz")
plt.title("PSD comparison")
plt.legend()
plt.tight_layout()
plt.show()

lc = lc + template_true
planet_mask = template_true != 0
plt.figure(figsize=(10, 3))
time_days = cadence / float(30 * 24)
plt.scatter(time_days, lc, color='blue', lw=0.8, s=1., label='light curve')
plt.scatter(time_days[planet_mask], lc[planet_mask], color='red', s=1, zorder=3, label='SimulatedPlanet (in-transit)')
plt.title(f"TID {tid} — Light curve with injected transit")
plt.xlabel('Time [days]')
plt.ylabel('Flux-median')
plt.legend()
plt.tight_layout()
plt.show()

# Template grid and matched-filter scores (replicated from nufft_transit_search)
period_grid = np.arange(4*24*30, 30*24*30, 30*4)
dur_grid = np.arange(4*30, 10*30, 30)
scores_white = np.zeros((len(period_grid), len(dur_grid)))
scores_est = np.zeros((len(period_grid), len(dur_grid)))
prog = Progress(2 * scores_white.size)

for i, P in enumerate(period_grid):
    for j, D in enumerate(dur_grid):
        template = boxtransit(cadence, period=P, dur=D, t0=0, alpha=1)
        scores_white[i, j] = matched_filter_snr_nufft(cadence, lc, cadence, template, freqs, pxx_white) / (2 * N**.5)
        prog.step()
        scores_est[i, j] = matched_filter_snr_nufft(cadence, lc, cadence, template, freqs, pxx_est) / (2 * N**.5)
        prog.step()

prog.close()

imax_w = np.unravel_index(np.argmax(scores_white), scores_white.shape)
imax_e = np.unravel_index(np.argmax(scores_est), scores_est.shape)
est_period_w, est_dur_w = period_grid[imax_w[0]], dur_grid[imax_w[1]]
est_period_e, est_dur_e = period_grid[imax_e[0]], dur_grid[imax_e[1]]

print("white est :(period,dur)=", (est_period_w, est_dur_w), " max=", scores_white[imax_w])
print("est-PSD est:(period,dur)=", (est_period_e, est_dur_e), " max=", scores_est[imax_e])

# Save dashboard data (heatmaps and light curve) for Plotly Dash
NBINS = 200
edges = np.linspace(-0.5, 0.5, NBINS + 1)
phase_bins = 0.5 * (edges[:-1] + edges[1:])
pf_y = np.full((len(period_grid), len(dur_grid), NBINS), np.nan, dtype=float)
for i, P in enumerate(period_grid):
    for j, D in enumerate(dur_grid):
        ph, m = centered_phase_fold(cadence, P, D, alpha)
        y_dt = detrend(cadence, lc, deg=2, mask=m)
        idxb = np.digitize(ph, edges) - 1
        for b in range(NBINS):
            sel = idxb == b
            if np.any(sel):
                pf_y[i, j, b] = np.nanmedian(y_dt[sel])

"""
np.savez_compressed(
    "tess_dashboard_data.npz",
    period_grid=period_grid,
    dur_grid=dur_grid,
    scores_white=scores_white,
    scores_est=scores_est,
    cadence=cadence,
    lc=lc,
    alpha=alpha,
    true_period=true_period,
    true_dur=true_dur,
    phase_bins=phase_bins,
    pf_y=pf_y,
)
"""

# Heatmaps (axes as period [days], duration [hours])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
period_scale_days = 30 * 24
dur_scale_hours = 30
extent_scaled = [period_grid[0] / period_scale_days,
                 period_grid[-1] / period_scale_days,
                 dur_grid[0] / dur_scale_hours,
                 dur_grid[-1] / dur_scale_hours]

im0 = axes[0].imshow(scores_white.T, origin='lower', cmap='Blues', aspect='auto', extent=extent_scaled)
axes[0].set_title('Baseline:White PSD')
axes[0].set_xlabel('Period [days]')
axes[0].set_ylabel('Duration [hours]')
axes[0].plot(true_period / period_scale_days, true_dur / dur_scale_hours, 'rx', ms=6)
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(scores_est.T, origin='lower', aspect='auto', cmap='Blues', extent=extent_scaled)
axes[1].set_title('Estimated PSD')
axes[1].set_xlabel('Period [days]')
axes[1].set_ylabel('Duration [hours]')
axes[1].plot(true_period / period_scale_days, true_dur / dur_scale_hours, 'rx', ms=6)
fig.colorbar(im1, ax=axes[1])
plt.show()

# Phase-folded views (centered on transit; low-order polynomial detrend)
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

P_w, D_w = est_period_w, est_dur_w
ph_w, m_w = centered_phase_fold(cadence, P_w, D_w, alpha)
y_w = detrend(cadence, lc, deg=2, mask=m_w)
ax2[0].scatter(ph_w[~m_w], y_w[~m_w], s=2, color='blue', alpha=0.4, label='out-of-transit')
ax2[0].scatter(ph_w[m_w], y_w[m_w], s=6, color='red', alpha=0.9, label='in-transit')
ax2[0].set_xlim(-0.5, 0.5)
ax2[0].set_title('Phase-folded (White PSD)')
ax2[0].set_xlabel('Phase (centered)')
ax2[0].set_ylabel('Flux (detrended)')
ax2[0].legend()

P_e, D_e = est_period_e, est_dur_e
ph_e, m_e = centered_phase_fold(cadence, P_e, D_e, alpha)
y_e = detrend(cadence, lc, deg=2, mask=m_e)
ax2[1].scatter(ph_e[~m_e], y_e[~m_e], s=2, color='blue', alpha=0.4, label='out-of-transit')
ax2[1].scatter(ph_e[m_e], y_e[m_e], s=6, color='red', alpha=0.9, label='in-transit')
ax2[1].set_xlim(-0.5, 0.5)
ax2[1].set_title('Phase-folded (Estimated PSD)')
ax2[1].set_xlabel('Phase (centered)')
ax2[1].set_ylabel('Flux (detrended)')
ax2[1].legend()

plt.show()