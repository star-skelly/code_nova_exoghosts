import numpy as np
from nufft_detector import matched_filter_snr_nufft
from noise_generator import boxtransit, periodogram_one_sided, simulate_cyclostationary, powerlaw_psd, _centered_phase_and_detrend
import matplotlib.pyplot as plt
from util import Progress

rng = np.random.default_rng(0)

# synthetic data with cyclostationary noise
N = 90*24*2
fs = 1.0
t = np.arange(N)
true_period = 320
true_dur = 20
alpha = 0.2
template_true = boxtransit(t, period=true_period, dur=true_dur, t0=0, alpha=alpha)
_, S_red = powerlaw_psd(N, fs, alpha=2.0, variance=1.0)
cyclo = simulate_cyclostationary(S_red, N, fs, period=256.0, modulation_index=0.6, rng=rng)
white_noise = np.random.normal(0, 0.1, N)
lc = template_true + white_noise

# highlight transit in time series like in noise_sim
planet_mask = template_true != 0
plt.figure(figsize=(10, 3))
plt.plot(np.arange(N), lc, color='blue', lw=0.8, label='light curve')
plt.scatter(np.arange(N)[planet_mask], lc[planet_mask], color='red', s=6, zorder=3, label='Planet (in-transit)')
plt.legend()
plt.tight_layout()
plt.show()

# PSDs on rFFT grid used for NUFFT
freqs, pxx_est = periodogram_one_sided(white_noise, fs)
# One-sided white PSD level matching this normalization: 2 * var / fs
sigma2 = float(np.var(lc))
pxx_white = np.full_like(pxx_est, 2.0 * sigma2 / fs)

# template grid
period_grid = np.arange(200, 401, 10)
dur_grid = np.arange(8, 41, 2)
scores_white = np.zeros((len(period_grid), len(dur_grid)))
scores_est = np.zeros((len(period_grid), len(dur_grid)))
prog = Progress(2 * scores_white.size)

for i, P in enumerate(period_grid):
    for j, D in enumerate(dur_grid):
        template = boxtransit(t, period=P, dur=D, t0=0, alpha=1)
        scores_white[i, j] = matched_filter_snr_nufft(t, white_noise, t, template, freqs, pxx_white) / (2* N**.5)
        prog.step()
        scores_est[i, j] = matched_filter_snr_nufft(t, white_noise, t, template, freqs, pxx_est) / (2* N**.5)
        prog.step()

prog.close()

imax_w = np.unravel_index(np.argmax(scores_white), scores_white.shape)
imax_e = np.unravel_index(np.argmax(scores_est), scores_est.shape)
est_period_w, est_dur_w = period_grid[imax_w[0]], dur_grid[imax_w[1]]
est_period_e, est_dur_e = period_grid[imax_e[0]], dur_grid[imax_e[1]]

print("true:(period,dur)=", (true_period, true_dur))
print("white est :(period,dur)=", (est_period_w, est_dur_w), " max=", scores_white[imax_w])
print("est-PSD est:(period,dur)=", (est_period_e, est_dur_e), " max=", scores_est[imax_e])

# heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
extent = [period_grid[0], period_grid[-1], dur_grid[0], dur_grid[-1]]
im0 = axes[0].imshow(scores_white.T, origin='lower', cmap='Blues', aspect='auto', extent=extent)
axes[0].set_title('White PSD')
axes[0].set_xlabel('Period')
axes[0].set_ylabel('Duration')
axes[0].plot(true_period, true_dur, 'rx', ms=6)
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(scores_est.T, origin='lower', aspect='auto',cmap='Blues', extent=extent)
axes[1].set_title('Estimated PSD')
axes[1].set_xlabel('Period')
axes[1].set_ylabel('Duration')
axes[1].plot(true_period, true_dur, 'rx', ms=6)
fig.colorbar(im1, ax=axes[1])
plt.show()

# Phase-folded views (centered on transit; low-order polynomial detrend)
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

P_w, D_w = est_period_w, est_dur_w
ph_w, y_w, m_w = _centered_phase_and_detrend(t, lc, P_w, D_w, alpha)
ax2[0].scatter(ph_w[~m_w], y_w[~m_w], s=2, color='blue', alpha=0.4, label='out-of-transit')
ax2[0].scatter(ph_w[m_w], y_w[m_w], s=6, color='red', alpha=0.9, label='in-transit')
ax2[0].set_xlim(-0.5, 0.5)
ax2[0].set_title('Phase-folded (White PSD)')
ax2[0].set_xlabel('Phase (centered)')
ax2[0].set_ylabel('Flux (detrended)')
ax2[0].legend()

P_e, D_e = est_period_e, est_dur_e
ph_e, y_e, m_e = _centered_phase_and_detrend(t, lc, P_e, D_e, alpha)
ax2[1].scatter(ph_e[~m_e], y_e[~m_e], s=2, color='blue', alpha=0.4, label='out-of-transit')
ax2[1].scatter(ph_e[m_e], y_e[m_e], s=6, color='red', alpha=0.9, label='in-transit')
ax2[1].set_xlim(-0.5, 0.5)
ax2[1].set_title('Phase-folded (Estimated PSD)')
ax2[1].set_xlabel('Phase (centered)')
ax2[1].set_ylabel('Flux (detrended)')
ax2[1].legend()

plt.show()

